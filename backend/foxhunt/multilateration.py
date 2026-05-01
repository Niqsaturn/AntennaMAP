"""Physics-based RF transmitter location using all available signal data.

Implements four independent location methods and fuses their results by
inverse-variance weighting:

  1. RSSI multilateration  — path-loss inversion circles from N observer positions
  2. TDOA hyperbolic fix   — GPS-clock time differences between fixed KiwiSDR nodes
  3. Bearing WLS           — SNR-weighted least-squares bearing line intersection
  4. RSSI gradient bearing — steepest-ascent direction from mobile readings

Physical models used
  - Free-space path loss:  PL = 32.44 + 20·log10(f_MHz) + 20·log10(d_km)
  - Band-default EIRP from spectrum allocation lookup
  - HF ionospheric skip constraint (ITU-R P.533)
  - VHF+ line-of-sight limit: sqrt(2·R·h_tx) + sqrt(2·R·h_rx) × 1.15 (k=4/3)
  - TDOA: Chan iterative WLS on linearised hyperbolic range-difference equations

All coordinates WGS-84.  Distances in metres unless stated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

C_MS = 2.997_924_58e8   # speed of light m/s
R_EARTH = 6_371_000.0   # mean Earth radius m


# ── Input observation types ──────────────────────────────────────────────────

@dataclass
class RSSIObs:
    """Single-point RSSI observation at a known position."""
    lat: float
    lon: float
    rssi_dbm: float
    freq_hz: float
    eirp_dbm: float | None = None   # None → band-default lookup
    weight: float = 1.0             # node quality multiplier


@dataclass
class TDOAObs:
    """TDOA pair from two GPS-locked receivers (e.g. KiwiSDR nodes)."""
    lat_ref: float
    lon_ref: float
    lat_remote: float
    lon_remote: float
    tdoa_s: float       # t_remote − t_ref (seconds); negative = signal reached remote first
    freq_hz: float
    weight: float = 1.0


@dataclass
class BearingObs:
    """Directional bearing reading from an observer position."""
    lat: float
    lon: float
    bearing_deg: float  # compass degrees, 0 = North
    snr_db: float
    freq_hz: float
    sigma_deg: float | None = None  # None → computed from SNR + frequency model


@dataclass
class LocationEstimate:
    """Output of a single location method."""
    method: str
    lat: float
    lon: float
    uncertainty_m: float    # 1-sigma
    confidence: float       # 0–1
    residual_rms: float = 0.0
    iterations: int = 0
    notes: str = ""
    band_constrained: bool = False


@dataclass
class FusedFix:
    """Final fused transmitter position."""
    lat: float
    lon: float
    uncertainty_m: float
    confidence: float
    methods_used: list[str] = field(default_factory=list)
    per_method: list[LocationEstimate] = field(default_factory=list)
    ellipse_major_m: float = 0.0
    ellipse_minor_m: float = 0.0
    ellipse_angle_deg: float = 0.0
    freq_hz: float = 0.0


# ── Coordinate helpers ───────────────────────────────────────────────────────

def _to_local_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """(lat, lon) → (east_m, north_m) relative to origin (lat0, lon0)."""
    north = (lat - lat0) * 111_320.0
    east  = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    return east, north


def _from_local_m(east_m: float, north_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat = lat0 + north_m / 111_320.0
    cos_lat = max(math.cos(math.radians(lat0)), 1e-9)
    lon = lon0 + east_m / (111_320.0 * cos_lat)
    return lat, lon


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R_EARTH * 2 * math.asin(math.sqrt(a))


def _dead_reckon(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    d_R = dist_m / R_EARTH
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    br = math.radians(bearing_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(d_R) + math.cos(lat1) * math.sin(d_R) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d_R) * math.cos(lat1),
        math.cos(d_R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# ── Band physics ─────────────────────────────────────────────────────────────

def _default_eirp_dbm(freq_hz: float) -> float:
    """Typical EIRP from band allocation, falling back to generic table."""
    try:
        from backend.datasources.spectrum_allocations import allocations_for_freq
        allocs = allocations_for_freq(freq_hz / 1e6)
        if allocs:
            return float(allocs[0].get("typical_eirp_dbm", 47.0))
    except Exception:
        pass
    f = freq_hz / 1e6
    if f < 0.3:   return 90.0
    if f < 3:     return 80.0
    if f < 30:    return 60.0
    if f < 300:   return 50.0
    if f < 1000:  return 46.0
    if f < 6000:  return 40.0
    return 36.0


def _fspl_db(freq_hz: float, dist_m: float) -> float:
    d_km = max(dist_m, 1.0) / 1000.0
    f_mhz = max(freq_hz, 1.0) / 1e6
    return 32.44 + 20 * math.log10(f_mhz) + 20 * math.log10(d_km)


def _rssi_to_dist_m(rssi_dbm: float, freq_hz: float, eirp_dbm: float) -> float:
    """Invert FSPL: RSSI = EIRP - PL → solve for distance."""
    pl_db = eirp_dbm - rssi_dbm
    if pl_db <= 0:
        return 10.0
    f_mhz = max(freq_hz, 1.0) / 1e6
    exp = (pl_db - 32.44 - 20 * math.log10(f_mhz)) / 20.0
    return max(10.0, (10 ** exp) * 1000.0)


# ── Method 1: RSSI multilateration ──────────────────────────────────────────

def _solve_rssi_multilat(obs: list[RSSIObs]) -> LocationEstimate | None:
    """Weighted least-squares circle intersection from N ≥ 2 RSSI readings.

    Linearises the set of circle equations by subtracting the reference
    (obs[0]) from each subsequent equation, yielding a linear system:
        2(xᵢ−x₀)·x + 2(yᵢ−y₀)·y = dᵢ²−d₀² − (xᵢ²+yᵢ²) + (x₀²+y₀²)

    Weight per row: w = node_weight / sqrt(dᵢ)  (closer = more reliable).
    """
    if len(obs) < 2:
        return None

    freq = obs[0].freq_hz
    lat0 = sum(o.lat for o in obs) / len(obs)
    lon0 = sum(o.lon for o in obs) / len(obs)

    xs, ys, ds = [], [], []
    for o in obs:
        e, n = _to_local_m(o.lat, o.lon, lat0, lon0)
        xs.append(e); ys.append(n)
        eirp = o.eirp_dbm if o.eirp_dbm is not None else _default_eirp_dbm(freq)
        ds.append(_rssi_to_dist_m(o.rssi_dbm, freq, eirp))

    A, b, w = [], [], []
    x0, y0, d0 = xs[0], ys[0], ds[0]
    for i in range(1, len(obs)):
        xi, yi, di = xs[i], ys[i], ds[i]
        A.append([2 * (xi - x0), 2 * (yi - y0)])
        b.append(d0 ** 2 - di ** 2 - x0 ** 2 - y0 ** 2 + xi ** 2 + yi ** 2)
        w.append(obs[i].weight / max(di, 1.0) ** 0.5)

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    W = np.diag(np.clip(w, 0.01, 100.0))
    try:
        lhs = A_np.T @ W @ A_np + np.eye(2) * 1.0
        xy = np.linalg.solve(lhs, A_np.T @ W @ b_np)
    except np.linalg.LinAlgError:
        return None

    lat, lon = _from_local_m(float(xy[0]), float(xy[1]), lat0, lon0)
    resids = [abs(math.hypot(xy[0] - xs[i], xy[1] - ys[i]) - ds[i]) for i in range(len(obs))]
    rms = math.sqrt(sum(r ** 2 for r in resids) / len(resids))
    min_d = min(ds)
    unc = math.sqrt(rms ** 2 + (0.3 * min_d) ** 2)
    confidence = round(
        0.5 * min(1.0, len(obs) / 4.0) + 0.5 * max(0.0, 1.0 - rms / max(min_d, 1.0)),
        3,
    )
    return LocationEstimate(
        method="rssi_multilat",
        lat=round(lat, 6), lon=round(lon, 6),
        uncertainty_m=round(unc, 1), confidence=confidence,
        residual_rms=round(rms, 1), iterations=1,
        notes=f"n={len(obs)}, min_d={min_d:.0f}m, rms={rms:.0f}m",
    )


# ── Method 2: TDOA hyperbolic intersection (Chan WLS) ───────────────────────

def _solve_tdoa(obs: list[TDOAObs]) -> LocationEstimate | None:
    """Locate TX from GPS-locked TDOA pairs via iterative linearised WLS.

    Range-difference equation for pair (ref, remote):
        ‖p − p_ref‖ − ‖p − p_remote‖ = c · TDOA

    Jacobian row: ∂/∂p = (p−p_ref)/r_ref − (p−p_remote)/r_remote
    Iterate Newton–Raphson steps until convergence < 0.1 m.
    """
    if len(obs) < 2:
        return None

    all_lats = [o.lat_ref for o in obs] + [o.lat_remote for o in obs]
    all_lons = [o.lon_ref for o in obs] + [o.lon_remote for o in obs]
    lat0 = sum(all_lats) / len(all_lats)
    lon0 = sum(all_lons) / len(all_lons)

    p = np.zeros(2)  # [east_m, north_m] from centroid
    last_residuals: list[float] = []

    for iteration in range(30):
        H, z, weights = [], [], []
        for o in obs:
            e_ref, n_ref = _to_local_m(o.lat_ref, o.lon_ref, lat0, lon0)
            e_rem, n_rem = _to_local_m(o.lat_remote, o.lon_remote, lat0, lon0)
            r_ref = max(math.hypot(p[0] - e_ref, p[1] - n_ref), 1.0)
            r_rem = max(math.hypot(p[0] - e_rem, p[1] - n_rem), 1.0)
            meas_dr = o.tdoa_s * C_MS
            pred_dr = r_ref - r_rem
            residual = meas_dr - pred_dr
            H.append([
                (p[0] - e_ref) / r_ref - (p[0] - e_rem) / r_rem,
                (p[1] - n_ref) / r_ref - (p[1] - n_rem) / r_rem,
            ])
            z.append(residual)
            weights.append(o.weight)
        last_residuals = z
        H_np = np.array(H, dtype=float)
        z_np = np.array(z, dtype=float)
        W = np.diag(np.clip(weights, 0.1, 10.0))
        try:
            dp = np.linalg.solve(
                H_np.T @ W @ H_np + np.eye(2) * 1e-4,
                H_np.T @ W @ z_np,
            )
        except np.linalg.LinAlgError:
            break
        p = p + dp
        if np.linalg.norm(dp) < 0.1:
            break

    lat, lon = _from_local_m(float(p[0]), float(p[1]), lat0, lon0)
    rms = float(np.sqrt(np.mean(np.array(last_residuals) ** 2))) if last_residuals else 9999.0

    # Uncertainty from covariance trace
    try:
        H_np = np.array(H, dtype=float)
        W = np.diag(np.clip(weights, 0.1, 10.0))
        cov = np.linalg.inv(H_np.T @ W @ H_np)
        unc = float(math.sqrt(max(np.trace(cov), 1.0)))
    except Exception:
        unc = 5000.0

    confidence = round(min(1.0, max(0.0, 1.0 - unc / 10_000.0) * min(1.0, len(obs) / 3.0)), 3)
    return LocationEstimate(
        method="tdoa_hyperbolic",
        lat=round(lat, 6), lon=round(lon, 6),
        uncertainty_m=round(max(unc, 10.0), 1), confidence=confidence,
        residual_rms=round(rms, 1), iterations=iteration + 1,
        notes=f"n_pairs={len(obs)}, rms_range={rms:.0f}m",
    )


# ── Method 3: Bearing WLS ────────────────────────────────────────────────────

def _solve_bearing_wls(obs: list[BearingObs]) -> LocationEstimate | None:
    """WLS bearing line intersection.

    Each bearing gives a constraint:  n_i · p = n_i · x_i
    where n_i = perpendicular to the bearing direction vector.
    Weight = 1/σ²; σ synthesised from SNR and frequency if not supplied.

    Synthesised bearing uncertainty model (from bearing_tracker.py):
        σ_base = 30 / (1 + SNR/10)   degrees
        σ_freq = σ_base × sqrt(300 / freq_MHz)
        floor = 2°
    """
    if len(obs) < 2:
        return None

    lat0 = sum(o.lat for o in obs) / len(obs)
    lon0 = sum(o.lon for o in obs) / len(obs)

    A, b, w = [], [], []
    for o in obs:
        theta = math.radians(o.bearing_deg)
        d = np.array([math.sin(theta), math.cos(theta)])
        n = np.array([d[1], -d[0]])
        e, north = _to_local_m(o.lat, o.lon, lat0, lon0)
        A.append(n.tolist())
        b.append(float(n @ np.array([e, north])))
        if o.sigma_deg is not None:
            sigma = max(o.sigma_deg, 1.0)
        else:
            sigma_base = max(2.0, 30.0 / (1.0 + max(o.snr_db, 0) / 10.0))
            freq_mhz = o.freq_hz / 1e6
            sigma = max(2.0, sigma_base * math.sqrt(max(300.0, freq_mhz) / freq_mhz) ** 0.5)
        w.append(1.0 / sigma ** 2)

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    W = np.diag(w)
    try:
        lhs = A_np.T @ W @ A_np + np.eye(2) * 1e-6
        xy = np.linalg.solve(lhs, A_np.T @ W @ b_np)
    except np.linalg.LinAlgError:
        return None

    # Covariance ellipse
    try:
        cov = np.linalg.inv(lhs)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-6, None)
        idx = np.argsort(evals)[::-1]
        major = float(math.sqrt(evals[idx[0]]) * 2.5)
        minor = float(math.sqrt(evals[idx[1]]) * 2.5)
        angle = float(math.degrees(math.atan2(evecs[1, idx[0]], evecs[0, idx[0]])) % 360)
    except Exception:
        major = minor = 1000.0; angle = 0.0

    lat, lon = _from_local_m(float(xy[0]), float(xy[1]), lat0, lon0)

    bearings = [o.bearing_deg for o in obs]
    spread = float(np.std(np.unwrap(np.deg2rad(bearings))))
    diversity = min(1.0, spread / (math.pi / 2))

    # RMS bearing line residual
    resids = []
    for o in obs:
        theta = math.radians(o.bearing_deg)
        d = np.array([math.sin(theta), math.cos(theta)])
        e, north = _to_local_m(o.lat, o.lon, lat0, lon0)
        t = float(np.dot(d, xy - np.array([e, north])))
        resids.append(float(np.linalg.norm(xy - (np.array([e, north]) + t * d))))
    rms = math.sqrt(sum(r ** 2 for r in resids) / len(resids))

    confidence = round(min(1.0, 0.6 * diversity + 0.4 * min(1.0, len(obs) / 4)), 3)
    return LocationEstimate(
        method="bearing_wls",
        lat=round(lat, 6), lon=round(lon, 6),
        uncertainty_m=round(max(major, 10.0), 1), confidence=confidence,
        residual_rms=round(rms, 1), iterations=1,
        notes=f"n={len(obs)}, diversity={diversity:.2f}, major={major:.0f}m",
    )


# ── Method 4: RSSI gradient bearing ─────────────────────────────────────────

def _solve_rssi_gradient(obs: list[RSSIObs]) -> LocationEstimate | None:
    """Direction of steepest RSSI ascent → approximate bearing to TX.

    Fits RSSI ≈ a·east + b·north + c by least squares; the gradient
    (a, b) points toward the transmitter.  Combine with RSSI-based distance
    to dead-reckon a position.  Works with omnidirectional mobile SDR.
    """
    if len(obs) < 3:
        return None

    lat0 = sum(o.lat for o in obs) / len(obs)
    lon0 = sum(o.lon for o in obs) / len(obs)
    rssi_mean = sum(o.rssi_dbm for o in obs) / len(obs)
    freq = obs[0].freq_hz
    eirp = next((o.eirp_dbm for o in obs if o.eirp_dbm), _default_eirp_dbm(freq))

    X, y = [], []
    for o in obs:
        e, n = _to_local_m(o.lat, o.lon, lat0, lon0)
        X.append([e, n, 1.0])
        y.append(o.rssi_dbm)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(np.array(X, dtype=float), np.array(y, dtype=float), rcond=None)
    except Exception:
        return None

    ge, gn = float(coeffs[0]), float(coeffs[1])
    grad_mag = math.hypot(ge, gn)
    if grad_mag < 1e-8:
        return None

    bearing = math.degrees(math.atan2(ge, gn)) % 360
    dist_m = _rssi_to_dist_m(rssi_mean, freq, eirp)
    lat_tx, lon_tx = _dead_reckon(lat0, lon0, bearing, dist_m)

    unc = dist_m * 0.6   # gradient method is coarse
    confidence = round(min(0.45, grad_mag / 15.0), 3)
    return LocationEstimate(
        method="rssi_gradient",
        lat=round(lat_tx, 6), lon=round(lon_tx, 6),
        uncertainty_m=round(unc, 1), confidence=confidence,
        residual_rms=0.0, iterations=1,
        notes=f"bearing={bearing:.1f}°, dist≈{dist_m:.0f}m, grad={grad_mag:.3f}dB/m",
    )


# ── Band constraints ─────────────────────────────────────────────────────────

def _apply_band_constraints(
    est: LocationEstimate,
    obs_positions: list[tuple[float, float]],
    freq_hz: float,
) -> LocationEstimate:
    """Flag estimates that violate physical propagation limits.

    VHF+ (>30 MHz):  TX must be within 2× LoS range (allows ducting/diffraction).
        LoS_m = (sqrt(2·R·h_tx) + sqrt(2·R·h_rx)) × 1.15   (k=4/3 effective Earth)
    HF (3–30 MHz):   TX may not be closer than the ionospheric skip distance.
    """
    f_mhz = freq_hz / 1e6
    for obs_lat, obs_lon in obs_positions:
        dist_m = _haversine_m(obs_lat, obs_lon, est.lat, est.lon)
        if f_mhz > 30:
            los_m = (math.sqrt(2 * R_EARTH * 50.0) + math.sqrt(2 * R_EARTH * 1.5)) * 1.15
            if dist_m > los_m * 2:
                est.band_constrained = True
        if 3.0 <= f_mhz <= 30.0:
            try:
                from backend.rf_propagation import ionospheric_skip_distance_km
                skip = ionospheric_skip_distance_km(f_mhz)
                skip_m = skip.get("skip_distance_km", 0.0) * 1000.0
                if skip.get("propagation_possible") and dist_m < skip_m * 0.5:
                    est.band_constrained = True
            except Exception:
                pass
    return est


# ── Fusion ───────────────────────────────────────────────────────────────────

def _fuse_estimates(
    estimates: list[LocationEstimate], lat0: float, lon0: float
) -> FusedFix:
    """Inverse-variance weighted mean of all method estimates.

    w_i = confidence_i / σ_i²
    lat_fused = Σ(w_i · lat_i) / Σ(w_i)
    σ_fused   = 1 / sqrt(Σ(w_i))
    """
    if not estimates:
        return FusedFix(lat=lat0, lon=lon0, uncertainty_m=99_999.0, confidence=0.0)

    total_w = lat_sum = lon_sum = 0.0
    for e in estimates:
        w = e.confidence / max(e.uncertainty_m, 1.0) ** 2
        lat_sum += e.lat * w
        lon_sum += e.lon * w
        total_w += w

    if total_w <= 0:
        e = estimates[0]
        return FusedFix(
            lat=e.lat, lon=e.lon, uncertainty_m=e.uncertainty_m,
            confidence=e.confidence, methods_used=[e.method], per_method=estimates,
        )

    fused_lat = lat_sum / total_w
    fused_lon = lon_sum / total_w
    fused_unc = math.sqrt(1.0 / total_w)
    fused_conf = min(1.0, sum(e.confidence for e in estimates) / len(estimates)
                    * (1.0 + 0.1 * (len(estimates) - 1)))

    # Use only high-confidence estimates for ellipse spread (filter out failures)
    good = [e for e in estimates if e.confidence >= 0.1]
    if not good:
        good = estimates
    lats = [e.lat for e in good]
    lons = [e.lon for e in good]
    if len(good) > 1:
        lat_std = math.sqrt(sum((l - fused_lat) ** 2 for l in lats) / len(lats)) * 111_320.0
        lon_std = (math.sqrt(sum((l - fused_lon) ** 2 for l in lons) / len(lons))
                   * 111_320.0 * math.cos(math.radians(fused_lat)))
        # Cap spread at 10× the best-method uncertainty to avoid degenerate ellipses
        best_unc = min(e.uncertainty_m for e in good)
        lat_std = min(lat_std, best_unc * 10)
        lon_std = min(lon_std, best_unc * 10)
        major = max(math.sqrt(max(lat_std, lon_std) ** 2 + fused_unc ** 2), 10.0)
        minor = max(math.sqrt(min(lat_std, lon_std) ** 2 + fused_unc ** 2), 5.0)
    else:
        major = max(good[0].uncertainty_m, 10.0)
        minor = major * 0.6

    return FusedFix(
        lat=round(fused_lat, 6), lon=round(fused_lon, 6),
        uncertainty_m=round(fused_unc, 1), confidence=round(fused_conf, 3),
        methods_used=[e.method for e in estimates],
        per_method=estimates,
        ellipse_major_m=round(float(major), 1),
        ellipse_minor_m=round(float(minor), 1),
        ellipse_angle_deg=0.0,
    )


# ── Observability check ───────────────────────────────────────────────────────

def _check_observability(
    rssi_obs: list[RSSIObs],
    tdoa_obs: list[TDOAObs],
    bearing_obs: list[BearingObs],
) -> tuple[bool, str]:
    """Return (observable, reason).

    A geometry is considered observable if ANY of:
    - 3+ bearing lines with angular spread ≥ 15°  (non-collinear)
    - 3+ RSSI nodes with spread ≥ 10 dB
    - 1+ TDOA pair with baseline ≥ 1 km
    """
    # TDOA: baseline check
    for o in tdoa_obs:
        baseline_m = _haversine_m(o.lat_ref, o.lon_ref, o.lat_remote, o.lon_remote)
        if baseline_m >= 1000.0:
            return True, "tdoa_baseline_ok"

    # RSSI spread
    if len(rssi_obs) >= 3:
        rssi_vals = [o.rssi_dbm for o in rssi_obs]
        if max(rssi_vals) - min(rssi_vals) >= 10.0:
            return True, "rssi_spread_ok"

    # Bearing angular diversity
    if len(bearing_obs) >= 3:
        angles_rad = [math.radians(o.bearing_deg) for o in bearing_obs]
        # Pairwise angular differences
        max_diff = 0.0
        for i in range(len(angles_rad)):
            for j in range(i + 1, len(angles_rad)):
                diff = abs(math.degrees(
                    math.atan2(
                        math.sin(angles_rad[i] - angles_rad[j]),
                        math.cos(angles_rad[i] - angles_rad[j]),
                    )
                ))
                max_diff = max(max_diff, diff)
        if max_diff >= 15.0:
            return True, "bearing_diversity_ok"

    # Build reason string
    reasons = []
    if bearing_obs:
        reasons.append(f"{len(bearing_obs)} bearing(s)")
    if rssi_obs:
        spread = max(o.rssi_dbm for o in rssi_obs) - min(o.rssi_dbm for o in rssi_obs) if rssi_obs else 0
        reasons.append(f"{len(rssi_obs)} RSSI nodes, spread={spread:.1f}dB")
    if tdoa_obs:
        reasons.append(f"{len(tdoa_obs)} TDOA pair(s)")
    return False, "insufficient geometry: " + (", ".join(reasons) or "no observations")


# ── Public API ───────────────────────────────────────────────────────────────

def locate_transmitter(
    rssi_obs: list[RSSIObs] | None = None,
    tdoa_obs: list[TDOAObs] | None = None,
    bearing_obs: list[BearingObs] | None = None,
    freq_hz: float = 100e6,
    apply_constraints: bool = True,
) -> FusedFix:
    """Locate a transmitter using all available RF observations.

    Method priority (highest confidence wins in fusion):
        TDOA > bearing_WLS > RSSI_multilat > RSSI_gradient

    All applicable methods run independently; results fused by
    inverse-variance weighting so that lower-uncertainty estimates
    dominate.

    Args:
        rssi_obs:     RSSI readings at known positions (fixed nodes or mobile).
        tdoa_obs:     GPS-locked TDOA pairs from KiwiSDR or similar.
        bearing_obs:  Directional bearing readings with SNR.
        freq_hz:      Signal carrier frequency (Hz) for band-constraint selection.
        apply_constraints: Apply HF skip / VHF LoS physics filters.

    Returns:
        FusedFix with position, 1-sigma uncertainty, per-method breakdown,
        and GeoJSON-ready ellipse parameters.
    """
    rssi_obs = rssi_obs or []
    tdoa_obs = tdoa_obs or []
    bearing_obs = bearing_obs or []

    all_positions: list[tuple[float, float]] = (
        [(o.lat, o.lon) for o in rssi_obs]
        + [(o.lat_ref, o.lon_ref) for o in tdoa_obs]
        + [(o.lat, o.lon) for o in bearing_obs]
    )
    if all_positions:
        lat0 = sum(p[0] for p in all_positions) / len(all_positions)
        lon0 = sum(p[1] for p in all_positions) / len(all_positions)
    else:
        lat0 = lon0 = 0.0

    estimates: list[LocationEstimate] = []

    if len(tdoa_obs) >= 2:
        est = _solve_tdoa(tdoa_obs)
        if est:
            estimates.append(est)

    if len(bearing_obs) >= 2:
        est = _solve_bearing_wls(bearing_obs)
        if est:
            estimates.append(est)

    if len(rssi_obs) >= 2:
        est = _solve_rssi_multilat(rssi_obs)
        if est:
            estimates.append(est)

    if len(rssi_obs) >= 3 and not any(e.method in ("rssi_multilat", "tdoa_hyperbolic") for e in estimates):
        est = _solve_rssi_gradient(rssi_obs)
        if est:
            estimates.append(est)

    if apply_constraints and estimates:
        estimates = [_apply_band_constraints(e, all_positions, freq_hz) for e in estimates]

    observable, obs_note = _check_observability(rssi_obs, tdoa_obs, bearing_obs)
    fix = _fuse_estimates(estimates, lat0, lon0)
    fix.freq_hz = freq_hz

    # Downgrade confidence if geometry is not sufficiently observable
    if not observable:
        fix.confidence = min(fix.confidence, 0.15)
        for e in fix.per_method:
            e.notes = (e.notes + "; " if e.notes else "") + obs_note

    return fix


def ellipse_polygon(
    lat: float, lon: float,
    major_m: float, minor_m: float,
    angle_deg: float = 0.0,
    steps: int = 64,
) -> list[list[float]]:
    """GeoJSON polygon coordinates for an uncertainty ellipse (closed ring).

    Returns [[lon, lat], ...] with steps+1 vertices (last == first).
    major_m / minor_m are semi-axes.  angle_deg rotates major axis CW from North.
    """
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    cos_lat = max(math.cos(math.radians(lat)), 1e-9)
    coords = []
    for i in range(steps + 1):
        theta = 2 * math.pi * i / steps
        x = major_m * math.cos(theta)
        y = minor_m * math.sin(theta)
        xe = x * cos_a - y * sin_a
        yn = x * sin_a + y * cos_a
        coords.append([
            round(lon + xe / (111_320.0 * cos_lat), 7),
            round(lat + yn / 111_320.0, 7),
        ])
    return coords

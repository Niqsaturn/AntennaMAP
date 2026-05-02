from backend.analysis.track_pipeline import derive_track_candidates


def _pkt(ts: str, freq: float, psd: list[float], snr: float = 20.0):
    return {
        "timestamp": ts,
        "device_id": "d1",
        "rf": {"frequency_hz": freq, "bandwidth_hz": 2_000_000, "rssi_dbm": -70, "snr_db": snr},
        "spectral": {"psd_bins_db": psd, "waterfall_avg_db": -90, "waterfall_peak_db": -45},
        "location": {"lat": 32.0, "lon": -117.0, "heading_deg": 0},
    }


def test_track_candidate_derives_required_metrics():
    psd = [-95, -94, -93, -50, -48, -51, -92, -93, -52, -49, -53, -93]
    rows = [
        _pkt("2026-05-02T00:00:00Z", 100_000_000.0, psd),
        _pkt("2026-05-02T00:00:02Z", 100_000_010.0, psd),
        _pkt("2026-05-02T00:00:04Z", 100_000_020.0, psd),
        _pkt("2026-05-02T00:00:06Z", 100_000_030.0, psd),
    ]
    tracks = derive_track_candidates(rows)
    assert len(tracks) == 1
    t = tracks[0]
    assert t.detected_carriers >= 1
    assert t.occupied_bandwidth_hz > 0
    assert t.drift_hz_per_s > 0
    assert t.periodicity_s == 2.0
    assert t.modulation_hints
    assert isinstance(t.interference_signature, str)
    assert 0.0 <= t.confidence <= 1.0
    assert len(t.temporal_history) == 4

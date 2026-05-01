"""Static ITU/FCC spectrum allocation reference.

Maps frequency ranges to service descriptions, typical emitter types, and
expected EIRP ranges.  Used to annotate signals detected in waterfall data
and to seed known-signal lists for the computed-spectrum adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BandEntry:
    freq_low_mhz: float
    freq_high_mhz: float
    service: str
    emitter_type: str          # e.g. "broadcast", "cellular", "satellite"
    typical_eirp_dbm: float    # representative EIRP
    itu_band: str = ""         # ITU band designation (ELF, VLF, LF, MF, HF, …)
    propagation_mode: str = "" # ground-wave, sky-wave, LOS, satellite, space
    notes: str = ""


# Full ITU Region 2 (Americas) band plan — 3 Hz (ELF) through 300 GHz (EHF)
# propagation_mode: ground-wave | sky-wave | LOS | satellite | space | diffraction
BAND_PLAN: tuple[BandEntry, ...] = (
    # ── ELF 3–30 Hz ───────────────────────────────────────────────────────────
    BandEntry(0.000003, 0.00003, "ELF Submarine Comms", "military", 100.0,
              "ELF", "ground-wave", "USN Clam Lake/Republic transmitters, 76/82 Hz"),
    # ── SLF 30–300 Hz ─────────────────────────────────────────────────────────
    BandEntry(0.00003, 0.0003, "SLF Submarine / Power Line", "military", 90.0,
              "SLF", "ground-wave", "Power-line harmonics, submarine ELF relay"),
    # ── ULF 0.3–3 kHz ─────────────────────────────────────────────────────────
    BandEntry(0.0003, 0.003, "ULF Mine / Earthquake Monitoring", "government", 50.0,
              "ULF", "ground-wave", "Mine rescue comms, seismological sensors"),
    # ── VLF 3–30 kHz ──────────────────────────────────────────────────────────
    BandEntry(0.003, 0.0176, "VLF Navigation (NAVTEX/OMEGA)", "navigation", 60.0,
              "VLF", "sky-wave", "OMEGA (10.2 kHz), NAVTEX, NWR, NDB"),
    BandEntry(0.0176, 0.03, "WWVB / VLF Standard Freq", "government", 55.0,
              "VLF", "sky-wave", "WWVB (60 kHz time/freq std), MSK submarine"),
    # ── LF 30–300 kHz ─────────────────────────────────────────────────────────
    BandEntry(0.03, 0.09, "LF Longwave BC / LORAN", "navigation", 60.0,
              "LF", "ground-wave", "LORAN-C (100 kHz), European longwave BC"),
    BandEntry(0.09, 0.11, "LF Navigation Beacons (NDB)", "navigation", 50.0,
              "LF", "ground-wave", "Non-directional beacons"),
    BandEntry(0.11, 0.13, "LF DGPS / Maritime Beacons", "navigation", 50.0,
              "LF", "ground-wave", "DGPS corrections, maritime MF beacons"),
    BandEntry(0.13, 0.16, "LF Amateur / DGPS", "amateur", 40.0,
              "LF", "ground-wave", "2200m amateur band (135.7–137.8 kHz)"),
    BandEntry(0.16, 0.30, "LF Aeronautical NDB", "aviation", 45.0,
              "LF", "ground-wave", "NDB range 190–285 kHz"),
    # ── MF 300 kHz–3 MHz ──────────────────────────────────────────────────────
    BandEntry(0.30, 0.535, "MF Aeronautical Beacon", "aviation", 50.0,
              "MF", "ground-wave", "NDB 300–535 kHz"),
    BandEntry(0.535, 1.705, "AM Broadcast", "broadcast", 80.0,
              "MF", "ground-wave", "530–1700 kHz AM band"),
    BandEntry(1.705, 1.8, "MF Maritime Mobile", "maritime", 55.0,
              "MF", "ground-wave", "Maritime distress 2182 kHz (MF), SSB"),
    BandEntry(1.8, 2.0, "MF Amateur 160m", "amateur", 50.0,
              "MF", "sky-wave", "160m amateur band"),
    BandEntry(2.0, 3.0, "HF Maritime SSB / Emergency", "maritime", 55.0,
              "HF", "sky-wave", "2182 kHz maritime distress, USCG"),
    # ── HF 3–30 MHz ───────────────────────────────────────────────────────────
    BandEntry(3.0, 4.0, "HF Amateur 80m / Fixed", "amateur", 60.0,
              "HF", "sky-wave", "3.5–4 MHz 80m amateur, SW broadcasting"),
    BandEntry(4.0, 6.0, "HF SW Broadcasting / Fixed", "broadcast", 70.0,
              "HF", "sky-wave", "Shortwave bands 49m/41m, STANAG HF"),
    BandEntry(6.0, 10.0, "HF / Shortwave", "broadcast", 70.0,
              "HF", "sky-wave", "USCG, amateur 40m/30m, SW broadcast 31m/25m"),
    BandEntry(10.0, 14.35, "HF Amateur / BC", "amateur", 60.0,
              "HF", "sky-wave", "20m/17m amateur, SW 25m/22m broadcast"),
    BandEntry(14.35, 30.0, "HF SW / Amateur / FMCW Radar", "broadcast", 65.0,
              "HF", "sky-wave", "15m/12m/10m amateur, SW, OTH radar (FMCW)"),
    # ── VHF 30–300 MHz ────────────────────────────────────────────────────────
    BandEntry(30.0, 54.0, "VHF Low / Amateur 6m", "land_mobile", 46.0,
              "VHF", "LOS", "6m amateur (50–54 MHz), land mobile, MILSATCOM feeder"),
    BandEntry(54.0, 88.0, "VHF TV / Aircraft", "broadcast", 55.0,
              "VHF", "LOS", "Channels 2–6, ACARS (131.8 MHz)"),
    BandEntry(88.0, 108.0, "FM Broadcast", "broadcast", 72.0, "VHF", "LOS", "88–108 MHz"),
    BandEntry(108.0, 137.0, "Aviation VHF", "aviation", 40.0, "VHF", "LOS", "ILS, VOR, comm"),
    BandEntry(137.0, 144.0, "Satellite Downlink (APT)", "satellite", 10.0, "VHF", "satellite", "NOAA, Meteor-M"),
    BandEntry(144.0, 148.0, "Amateur 2m", "amateur", 50.0, "VHF", "LOS", "2m band"),
    BandEntry(148.0, 174.0, "VHF Land Mobile", "land_mobile", 46.0, "VHF", "LOS", "P25, DMR"),
    BandEntry(174.0, 216.0, "VHF TV", "broadcast", 55.0, "VHF", "LOS", "Channels 7–13"),
    # ── UHF 300 MHz–3 GHz ─────────────────────────────────────────────────────
    BandEntry(225.0, 400.0, "Military UHF SATCOM", "military", 50.0, "UHF", "satellite", "DAMA/HAVE QUICK"),
    BandEntry(406.0, 406.1, "Distress Beacon", "emergency", 30.0, "UHF", "satellite", "ELT/EPIRB/PLB → COSPAS-SARSAT"),
    BandEntry(406.1, 420.0, "Federal Land Mobile", "government", 46.0, "UHF", "LOS"),
    BandEntry(420.0, 450.0, "Amateur 70cm", "amateur", 50.0, "UHF", "LOS", "70cm band"),
    BandEntry(450.0, 470.0, "UHF Land Mobile", "land_mobile", 46.0, "UHF", "LOS", "FRS, GMRS, P25"),
    BandEntry(470.0, 698.0, "UHF TV / WMTS", "broadcast", 55.0, "UHF", "LOS", "Channels 14–51"),
    BandEntry(698.0, 806.0, "700 MHz LTE", "cellular", 46.0, "UHF", "LOS", "Band 12/17/13"),
    BandEntry(806.0, 902.0, "800 MHz SMR/Cellular", "cellular", 46.0, "UHF", "LOS", "Band 5"),
    BandEntry(902.0, 928.0, "ISM 915", "ism", 30.0, "UHF", "LOS", "LoRa, 802.15.4g"),
    BandEntry(928.0, 960.0, "900 MHz Cellular", "cellular", 46.0, "UHF", "LOS", "GSM-850/Band 5"),
    BandEntry(960.0, 1164.0, "Aeronautical DME/TACAN/JTIDS", "aviation", 43.0, "UHF", "LOS", "DME, TACAN, Link-16"),
    BandEntry(1164.0, 1300.0, "GPS L5 / Galileo E5", "navigation", 10.0, "UHF", "satellite", "GNSS L5/E5"),
    BandEntry(1300.0, 1400.0, "Aeronautical Radar / Radionavigation", "radar", 50.0, "UHF", "LOS", "L-band ATC radar"),
    BandEntry(1400.0, 1427.0, "Radioastronomy / Passive", "government", 0.0, "UHF", "satellite", "Protected; no transmissions"),
    BandEntry(1525.0, 1660.5, "L-band Satellite Mobile", "satellite", 15.0, "UHF", "satellite", "Iridium, Inmarsat, GPS L1"),
    BandEntry(1710.0, 1780.0, "PCS / AWS", "cellular", 46.0, "UHF", "LOS", "Band 4/66"),
    BandEntry(1850.0, 1990.0, "PCS 1900", "cellular", 46.0, "UHF", "LOS", "Band 2/25"),
    BandEntry(2025.0, 2110.0, "Space Research / TDRSS", "satellite", 30.0, "UHF", "satellite", "S-band uplink"),
    BandEntry(2300.0, 2400.0, "AWS-3 / WCS", "cellular", 46.0, "UHF", "LOS", "Band 30"),
    BandEntry(2400.0, 2483.5, "ISM 2.4 GHz", "ism", 30.0, "UHF", "LOS", "WiFi 802.11b/g/n, Bluetooth, ZigBee"),
    BandEntry(2496.0, 2690.0, "BRS / EBS / 5G NR n41", "cellular", 46.0, "UHF", "LOS", "Band 41 / 5G NR n41"),
    BandEntry(2700.0, 3000.0, "Aeronautical / Weather Radar", "radar", 55.0, "UHF", "LOS", "S-band weather radar"),
    # ── SHF 3–30 GHz ──────────────────────────────────────────────────────────
    BandEntry(3400.0, 3980.0, "C-band / CBRS / 5G n77/n78", "cellular", 44.0, "SHF", "LOS", "Band 48"),
    BandEntry(3700.0, 4200.0, "C-band Satellite Downlink", "satellite", 15.0, "SHF", "satellite", "FSS downlink"),
    BandEntry(4400.0, 5000.0, "Military C-band / Fixed Satellite", "military", 46.0, "SHF", "satellite"),
    BandEntry(5150.0, 5850.0, "UNII / 5 GHz WiFi", "ism", 30.0, "SHF", "LOS", "802.11a/n/ac/ax"),
    BandEntry(5925.0, 6425.0, "C-band FSS Uplink", "satellite", 46.0, "SHF", "satellite", "C-band uplink earth→sat"),
    BandEntry(6425.0, 7125.0, "6 GHz WiFi (UNII-5/7) / C-band", "ism", 30.0, "SHF", "LOS", "802.11ax 6 GHz"),
    BandEntry(7125.0, 8500.0, "X-band Fixed Satellite / Military", "satellite", 40.0, "SHF", "satellite", "WGS, X-band satcom"),
    BandEntry(8500.0, 10500.0, "X-band Radar / Satellite", "radar", 60.0, "SHF", "LOS", "Weather radar, ship radar"),
    BandEntry(10700.0, 12750.0, "Ku-band FSS Downlink", "satellite", 15.0, "SHF", "satellite", "DTH, VSAT downlink"),
    BandEntry(12750.0, 14500.0, "Ku-band FSS Uplink", "satellite", 46.0, "SHF", "satellite", "VSAT uplink, DSNG"),
    BandEntry(14000.0, 14500.0, "Ku-band Mobile Satellite", "satellite", 40.0, "SHF", "satellite", "Starlink Ku downlink"),
    BandEntry(17300.0, 21200.0, "Ka-band FSS", "satellite", 40.0, "SHF", "satellite", "Ka-band sat downlink/uplink"),
    BandEntry(21200.0, 24000.0, "K-band Satellite / Radar", "satellite", 40.0, "SHF", "satellite", "WGS Ka uplink"),
    BandEntry(24000.0, 24250.0, "K-band Radar", "radar", 50.0, "SHF", "LOS", "Traffic/police radar"),
    BandEntry(24500.0, 27500.0, "Ka-band / 5G mmWave n258/n261", "cellular", 30.0, "SHF", "LOS", "5G mmWave"),
    # ── EHF 30–300 GHz ────────────────────────────────────────────────────────
    BandEntry(27500.0, 31000.0, "Ka-band Satellite (high)", "satellite", 35.0, "EHF", "satellite", "HTS gateways"),
    BandEntry(37000.0, 40000.0, "V-band Satellite / Backhaul", "satellite", 30.0, "EHF", "satellite", "SpaceX V-band"),
    BandEntry(57000.0, 71000.0, "V-band 60 GHz WiFi / WiGig", "ism", 20.0, "EHF", "LOS", "802.11ad/ay, O₂ absorption dip"),
    BandEntry(71000.0, 76000.0, "E-band Backhaul", "fixed", 30.0, "EHF", "LOS", "Point-to-point 71–76 GHz"),
    BandEntry(76000.0, 77000.0, "77 GHz Automotive Radar", "radar", 20.0, "EHF", "LOS", "Automotive ADAS FMCW radar"),
    BandEntry(77000.0, 81000.0, "E-band 77–81 GHz", "fixed", 30.0, "EHF", "LOS", "Backhaul + automotive radar"),
    BandEntry(81000.0, 86000.0, "E-band 81–86 GHz", "fixed", 30.0, "EHF", "LOS", "Point-to-point backhaul"),
    BandEntry(92000.0, 100000.0, "W-band Radar / Imaging", "radar", 25.0, "EHF", "LOS", "High-res radar, security screening"),
    BandEntry(100000.0, 300000.0, "Sub-THz / EHF Emerging", "research", 10.0, "EHF", "LOS", "THz imaging, passive remote sensing"),
    # ── THz / Optical (reference only — not radio) ────────────────────────────
    BandEntry(300000.0, 3000000.0, "THz / Far Infrared (reference)", "research", 0.0,
              "THF", "LOS", "THz spectroscopy, LIDAR — not RF"),
)


def _band_dict(b: BandEntry) -> dict[str, Any]:
    return {
        "freq_low_mhz": b.freq_low_mhz,
        "freq_high_mhz": b.freq_high_mhz,
        "service": b.service,
        "emitter_type": b.emitter_type,
        "typical_eirp_dbm": b.typical_eirp_dbm,
        "itu_band": b.itu_band,
        "propagation_mode": b.propagation_mode,
        "notes": b.notes,
    }


def allocations_for_freq(freq_mhz: float) -> list[dict[str, Any]]:
    """Return all band entries that contain freq_mhz."""
    return [_band_dict(b) for b in BAND_PLAN if b.freq_low_mhz <= freq_mhz <= b.freq_high_mhz]


def allocations_in_range(
    freq_low_mhz: float, freq_high_mhz: float
) -> list[dict[str, Any]]:
    """Return all band entries that overlap [freq_low_mhz, freq_high_mhz]."""
    return [
        _band_dict(b)
        for b in BAND_PLAN
        if b.freq_low_mhz <= freq_high_mhz and b.freq_high_mhz >= freq_low_mhz
    ]


def known_signals_for_band(center_freq_hz: float, bandwidth_hz: float) -> list[dict[str, Any]]:
    """Build a known_signals list suitable for computed_spectrum.computed_psd().

    Populates signal entries for each allocation band that overlaps the receiver
    window, using the band's typical_eirp_dbm as an approximate received power
    (assumes nearby transmitter; caller may scale by path loss).
    """
    f_low = (center_freq_hz - bandwidth_hz / 2) / 1e6
    f_high = (center_freq_hz + bandwidth_hz / 2) / 1e6
    signals = []
    for alloc in allocations_in_range(f_low, f_high):
        fc_mhz = (alloc["freq_low_mhz"] + alloc["freq_high_mhz"]) / 2.0
        bw_hz = (alloc["freq_high_mhz"] - alloc["freq_low_mhz"]) * 1e6
        signals.append({
            "freq_hz": fc_mhz * 1e6,
            "power_dbm": alloc["typical_eirp_dbm"] - 60.0,  # -60 dB path loss estimate
            "bandwidth_hz": min(bw_hz, bandwidth_hz / 4),
            "service": alloc["service"],
        })
    return signals


def full_spectrum_summary() -> list[dict[str, Any]]:
    """Return all bands grouped by ITU designation with full metadata.

    Covers 3 Hz (ELF) through 300 GHz (EHF) plus THz reference.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for b in BAND_PLAN:
        groups.setdefault(b.itu_band or "Other", []).append(_band_dict(b))

    itu_order = ["ELF", "SLF", "ULF", "VLF", "LF", "MF", "HF", "VHF",
                 "UHF", "SHF", "EHF", "THF", "Other"]
    result = []
    for band in itu_order:
        if band in groups:
            entries = groups[band]
            result.append({
                "itu_band": band,
                "freq_low_mhz": min(e["freq_low_mhz"] for e in entries),
                "freq_high_mhz": max(e["freq_high_mhz"] for e in entries),
                "entry_count": len(entries),
                "entries": entries,
            })
    return result

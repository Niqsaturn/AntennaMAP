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
    notes: str = ""


# US/ITU Region 2 primary allocations (abbreviated, common bands)
BAND_PLAN: tuple[BandEntry, ...] = (
    BandEntry(0.535, 1.705, "AM Broadcast", "broadcast", 80.0, "530–1700 kHz"),
    BandEntry(2.0, 30.0, "HF / Shortwave", "broadcast", 70.0, "USCG, amateur, SW"),
    BandEntry(54.0, 88.0, "VHF TV / Aircraft", "broadcast", 55.0, "Channels 2–6, ACARS"),
    BandEntry(88.0, 108.0, "FM Broadcast", "broadcast", 72.0, "88–108 MHz"),
    BandEntry(108.0, 137.0, "Aviation VHF", "aviation", 40.0, "ILS, VOR, comm"),
    BandEntry(137.0, 144.0, "Satellite downlink", "satellite", 10.0, "NOAA, Meteor-M"),
    BandEntry(144.0, 148.0, "Amateur 2m", "amateur", 50.0, "2m band"),
    BandEntry(148.0, 174.0, "VHF Land Mobile", "land_mobile", 46.0, "P25, DMR"),
    BandEntry(174.0, 216.0, "VHF TV", "broadcast", 55.0, "Channels 7–13"),
    BandEntry(225.0, 400.0, "Military UHF SATCOM", "military", 50.0, "DAMA"),
    BandEntry(406.0, 406.1, "Distress Beacon", "emergency", 30.0, "ELT/EPIRB/PLB"),
    BandEntry(406.1, 420.0, "Federal Land Mobile", "government", 46.0),
    BandEntry(420.0, 450.0, "Amateur 70cm", "amateur", 50.0, "70cm band"),
    BandEntry(450.0, 470.0, "UHF Land Mobile", "land_mobile", 46.0, "FRS, GMRS, P25"),
    BandEntry(470.0, 698.0, "UHF TV / WMTS", "broadcast", 55.0, "Channels 14–51"),
    BandEntry(698.0, 806.0, "700 MHz LTE", "cellular", 46.0, "Band 12/17/13"),
    BandEntry(806.0, 902.0, "800 MHz SMR/Cellular", "cellular", 46.0, "Band 5"),
    BandEntry(902.0, 928.0, "ISM 915", "ism", 30.0, "LoRa, 802.15.4g"),
    BandEntry(928.0, 960.0, "900 MHz Cellular", "cellular", 46.0, "GSM-850/Band 5"),
    BandEntry(1164.0, 1300.0, "GPS L5 / Galileo", "navigation", 10.0, "GNSS"),
    BandEntry(1525.0, 1660.5, "L-band Satellite", "satellite", 15.0, "Iridium, Inmarsat"),
    BandEntry(1710.0, 1780.0, "PCS / AWS", "cellular", 46.0, "Band 4/66"),
    BandEntry(1850.0, 1990.0, "PCS 1900", "cellular", 46.0, "Band 2/25"),
    BandEntry(2300.0, 2400.0, "AWS-3 / WCS", "cellular", 46.0, "Band 30"),
    BandEntry(2400.0, 2483.5, "ISM 2.4 GHz", "ism", 30.0, "WiFi, Bluetooth, ZigBee"),
    BandEntry(2496.0, 2690.0, "BRS / EBS", "cellular", 46.0, "Band 41 / 5G NR n41"),
    BandEntry(3400.0, 3980.0, "C-band / CBRS / 5G", "cellular", 44.0, "Band 48 / n77/n78"),
    BandEntry(5150.0, 5850.0, "UNII / 5 GHz WiFi", "ism", 30.0, "802.11a/n/ac/ax"),
    BandEntry(5925.0, 7125.0, "6 GHz WiFi / C-band", "ism", 30.0, "802.11ax 6 GHz"),
    BandEntry(10000.0, 10500.0, "X-band Radar / Satellite", "radar", 60.0, "Weather radar"),
    BandEntry(24000.0, 24250.0, "K-band Radar", "radar", 50.0, "Traffic/police radar"),
    BandEntry(76000.0, 77000.0, "77 GHz Automotive Radar", "radar", 20.0, "Automotive ADAS"),
)


def allocations_for_freq(freq_mhz: float) -> list[dict[str, Any]]:
    """Return all band entries that contain freq_mhz."""
    return [
        {
            "freq_low_mhz": b.freq_low_mhz,
            "freq_high_mhz": b.freq_high_mhz,
            "service": b.service,
            "emitter_type": b.emitter_type,
            "typical_eirp_dbm": b.typical_eirp_dbm,
            "notes": b.notes,
        }
        for b in BAND_PLAN
        if b.freq_low_mhz <= freq_mhz <= b.freq_high_mhz
    ]


def allocations_in_range(
    freq_low_mhz: float, freq_high_mhz: float
) -> list[dict[str, Any]]:
    """Return all band entries that overlap [freq_low_mhz, freq_high_mhz]."""
    return [
        {
            "freq_low_mhz": b.freq_low_mhz,
            "freq_high_mhz": b.freq_high_mhz,
            "service": b.service,
            "emitter_type": b.emitter_type,
            "typical_eirp_dbm": b.typical_eirp_dbm,
            "notes": b.notes,
        }
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

"""Parse race XML configs for course geometry (start line bearing)."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from sailgp_analysis.config import DATA_ROOT


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _signed_angle_diff(a: float, b: float) -> float:
    """Signed difference a - b in [-180, 180]."""
    return (a - b + 180) % 360 - 180


def parse_start_line(xml_path: Path) -> dict | None:
    """Extract SL1/SL2 coordinates and start-line bearing from a race XML."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    sl1 = sl2 = None
    for compound in root.findall(".//CompoundMark"):
        name = compound.get("Name", "")
        if name != "SL1":
            continue
        for mark in compound.findall("Mark"):
            mname = mark.get("Name", "")
            lat = float(mark.get("TargetLat", "nan"))
            lng = float(mark.get("TargetLng", "nan"))
            if mname == "SL1":
                sl1 = (lat, lng)
            elif mname == "SL2":
                sl2 = (lat, lng)

    if sl1 is None or sl2 is None:
        return None

    bearing = _bearing_deg(sl1[0], sl1[1], sl2[0], sl2[1])
    mid_lat = (sl1[0] + sl2[0]) / 2
    mid_lng = (sl1[1] + sl2[1]) / 2
    return {
        "sl1": sl1,
        "sl2": sl2,
        "mid": (mid_lat, mid_lng),
        "bearing_deg": bearing,
    }


def load_race_start_lines(data_root: Path = DATA_ROOT) -> dict[tuple[str, str], dict]:
    """Load start-line geometry keyed by (venue, race_label)."""
    out: dict[tuple[str, str], dict] = {}
    for venue in ("Bermuda", "Halifax"):
        xmls_dir = data_root / venue / "xmls"
        if not xmls_dir.exists():
            continue
        for race_dir in sorted(xmls_dir.iterdir()):
            if not race_dir.is_dir():
                continue
            race_label = race_dir.name
            xml_files = sorted(race_dir.glob("*.xml"))
            if not xml_files:
                continue
            # Use latest XML snapshot (most recent course config)
            parsed = parse_start_line(xml_files[-1])
            if parsed:
                out[(venue, race_label)] = parsed
    return out


def start_line_bias(
    heading_deg: float,
    lat: float,
    lon: float,
    start_line: dict,
) -> int | None:
    """
    Classify boat position relative to start line.
    Returns 1 = port end (SL1 side), 0 = starboard end (SL2 side), None if unknown.
    """
    bearing = start_line["bearing_deg"]
    sl1 = start_line["sl1"]
    sl2 = start_line["sl2"]

    # Cross-track: which side of the line is the boat on?
    # Vector from SL1 to boat vs line bearing
    boat_bearing_from_sl1 = _bearing_deg(sl1[0], sl1[1], lat, lon)
    cross = _signed_angle_diff(boat_bearing_from_sl1, bearing)

    # Also consider heading relative to line
    heading_rel = _signed_angle_diff(heading_deg, bearing)

    # Port end favored if boat is on port side of line AND heading toward port
    if cross < 0 and heading_rel < 0:
        return 1
    if cross >= 0 and heading_rel >= 0:
        return 0
    # Fallback: use cross-track side only
    return 1 if cross < 0 else 0

"""
Fetch ALPR camera locations from two crowdsourced databases and merge them.

Source 1 — OpenStreetMap (Overpass API)
    Queries the Overpass API for nodes tagged as surveillance/ALPR cameras
    within a bounding box around Philadelphia.
    Tags searched: surveillance:type=ALPR, camera:type=ALPR,
                   highway=speed_camera, operator=Flock Safety
    Output: data/philly_alpr_osm.csv

Source 2 — DeFlock / ALPRWatch
    Downloads a KMZ from alprwatch.org (community-maintained database of
    known ALPR camera locations, primarily Flock Safety cameras).
    Parses the KML inside the KMZ, extracts coordinates, filters to the
    Philadelphia bounding box.
    Output: data/philly_alpr_deflock.csv

Deduplication
    Merges both sources and deduplicates by rounding lat/lon to 4 decimal
    places (~11 m precision).
    Output: data/philly_alpr_combined.csv

Usage:
    python -m alpr_data.fetch_alpr_data
    python -m alpr_data.fetch_alpr_data --skip_osm
    python -m alpr_data.fetch_alpr_data --skip_deflock
"""

import io
import os
import zipfile
import logging
import xml.etree.ElementTree as ET

import fire
import requests
import pandas as pd

# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Philadelphia bounding box (south, west, north, east)
PHILLY_BBOX = {
    "south": 39.8670,
    "west": -75.2803,
    "north": 40.1379,
    "east": -74.9558,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ALPRWATCH_KMZ_URL = "https://alprwatch.org/pub/avoidance/alprwatch-avoidance-latest.kmz"

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1 — OpenStreetMap (Overpass API)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_overpass_query() -> str:
    """Build an Overpass QL query for ALPR-related nodes in Philadelphia."""
    bb = PHILLY_BBOX
    bbox = f"{bb['south']},{bb['west']},{bb['north']},{bb['east']}"
    return f"""
[out:json][timeout:120];
(
  node["surveillance:type"="ALPR"]({bbox});
  node["camera:type"="ALPR"]({bbox});
  node["highway"="speed_camera"]({bbox});
  node["man_made"="surveillance"]["operator"="Flock Safety"]({bbox});
  node["man_made"="surveillance"]["surveillance:type"="camera"]["camera:type"="ALPR"]({bbox});
  node["man_made"="surveillance"]["description"~"ALPR|LPR|license plate|Flock",i]({bbox});
);
out body;
"""


def fetch_osm_alpr() -> pd.DataFrame:
    """Query the Overpass API and return a DataFrame of ALPR camera locations."""
    log.info("Querying Overpass API for ALPR cameras in Philadelphia …")
    query = _build_overpass_query()
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for element in data.get("elements", []):
        if element.get("type") != "node":
            continue
        tags = element.get("tags", {})
        rows.append({
            "lat": element["lat"],
            "lon": element["lon"],
            "osm_id": element["id"],
            "source": "osm",
            "tags": str(tags),
        })

    df = pd.DataFrame(rows)
    log.info("OSM: found %d ALPR-related nodes", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2 — DeFlock / ALPRWatch
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_kml_coordinates(kml_bytes: bytes) -> list[dict]:
    """Extract (lat, lon, name) tuples from a KML document.

    The ALPRWatch avoidance KMZ uses LineString elements where the first
    coordinate is the actual camera location and the remaining coordinates
    form a visibility-zone polygon around it.  We extract the first
    coordinate of each placemark as the camera position.
    """
    tree = ET.parse(io.BytesIO(kml_bytes))
    root = tree.getroot()

    placemarks = root.findall(".//kml:Placemark", KML_NS)
    results = []
    for pm in placemarks:
        name_el = pm.find("kml:name", KML_NS)
        name = name_el.text.strip() if name_el is not None and name_el.text else ""

        coord_el = pm.find(".//kml:coordinates", KML_NS)
        if coord_el is None or not coord_el.text:
            continue

        coord_text = coord_el.text.strip()
        # KML coordinates are "lon,lat[,alt]" — may be whitespace-separated
        # for LineString/Polygon geometries.  The first coordinate pair is
        # the camera location.
        first_token = coord_text.split()[0]
        parts = first_token.split(",")
        if len(parts) < 2:
            continue

        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue

        results.append({"lat": lat, "lon": lon, "name": name})

    return results


def _in_philly(lat: float, lon: float) -> bool:
    """Check if a coordinate falls inside the Philadelphia bounding box."""
    bb = PHILLY_BBOX
    return (bb["south"] <= lat <= bb["north"] and
            bb["west"] <= lon <= bb["east"])


def fetch_deflock_alpr() -> pd.DataFrame:
    """Download the DeFlock KMZ, parse it, and filter to Philadelphia."""
    log.info("Downloading DeFlock KMZ from %s …", ALPRWATCH_KMZ_URL)
    resp = requests.get(ALPRWATCH_KMZ_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        kml_names = [n for n in zf.namelist() if n.endswith(".kml")]
        if not kml_names:
            raise ValueError("No .kml file found inside the KMZ archive")
        kml_bytes = zf.read(kml_names[0])

    log.info("Parsing KML (%d bytes) …", len(kml_bytes))
    all_points = _parse_kml_coordinates(kml_bytes)
    log.info("KMZ total placemarks: %d", len(all_points))

    philly_points = [p for p in all_points if _in_philly(p["lat"], p["lon"])]
    log.info("DeFlock: %d cameras inside Philadelphia bbox", len(philly_points))

    df = pd.DataFrame(philly_points)
    if not df.empty:
        df["source"] = "deflock"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Merge & Deduplicate
# ═══════════════════════════════════════════════════════════════════════════════

def merge_and_deduplicate(osm_df: pd.DataFrame,
                          deflock_df: pd.DataFrame,
                          precision: int = 4) -> pd.DataFrame:
    """
    Merge two ALPR DataFrames and deduplicate by rounding lat/lon.
    precision=4 means ~11 m resolution, so cameras within ~11 m
    of each other are treated as duplicates.
    """
    frames = []
    if osm_df is not None and not osm_df.empty:
        frames.append(osm_df[["lat", "lon", "source"]])
    if deflock_df is not None and not deflock_df.empty:
        cols = ["lat", "lon", "source"]
        if "name" in deflock_df.columns:
            cols.append("name")
        frames.append(deflock_df[cols])

    if not frames:
        log.warning("No data from either source")
        return pd.DataFrame(columns=["lat", "lon", "source"])

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)

    combined["lat_round"] = combined["lat"].round(precision)
    combined["lon_round"] = combined["lon"].round(precision)
    # Keep first occurrence (OSM first if both present)
    combined = combined.drop_duplicates(subset=["lat_round", "lon_round"], keep="first")
    combined = combined.drop(columns=["lat_round", "lon_round"])
    combined = combined.reset_index(drop=True)

    log.info("Deduplication: %d → %d unique locations (precision=%d)",
             before, len(combined), precision)
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(skip_osm: bool = False, skip_deflock: bool = False):
    """
    Fetch ALPR locations from OSM and DeFlock, merge, and save CSVs.

    Args:
        skip_osm:     Skip the Overpass API query
        skip_deflock: Skip the DeFlock/ALPRWatch download
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    osm_path = os.path.join(DATA_DIR, "philly_alpr_osm.csv")
    deflock_path = os.path.join(DATA_DIR, "philly_alpr_deflock.csv")
    combined_path = os.path.join(DATA_DIR, "philly_alpr_combined.csv")

    osm_df = pd.DataFrame()
    deflock_df = pd.DataFrame()

    # ── Source 1: OSM ────────────────────────────────────────────────────
    if not skip_osm:
        osm_df = fetch_osm_alpr()
        osm_df.to_csv(osm_path, index=False)
        log.info("Saved OSM data → %s (%d rows)", osm_path, len(osm_df))
    elif os.path.exists(osm_path):
        log.info("Loading cached OSM data from %s", osm_path)
        osm_df = pd.read_csv(osm_path)

    # ── Source 2: DeFlock ────────────────────────────────────────────────
    if not skip_deflock:
        deflock_df = fetch_deflock_alpr()
        deflock_df.to_csv(deflock_path, index=False)
        log.info("Saved DeFlock data → %s (%d rows)", deflock_path, len(deflock_df))
    elif os.path.exists(deflock_path):
        log.info("Loading cached DeFlock data from %s", deflock_path)
        deflock_df = pd.read_csv(deflock_path)

    # ── Merge ────────────────────────────────────────────────────────────
    combined = merge_and_deduplicate(osm_df, deflock_df)
    combined.to_csv(combined_path, index=False)
    log.info("Saved combined data → %s (%d rows)", combined_path, len(combined))

    print(f"\n{'='*60}")
    print(f"  ALPR Data Summary")
    print(f"{'='*60}")
    print(f"  OSM locations      : {len(osm_df)}")
    print(f"  DeFlock locations  : {len(deflock_df)}")
    print(f"  Combined (deduped) : {len(combined)}")
    print(f"{'='*60}")
    print(f"  Files:")
    print(f"    {osm_path}")
    print(f"    {deflock_path}")
    print(f"    {combined_path}")
    print(f"{'='*60}")

    return combined


if __name__ == "__main__":
    fire.Fire(main)

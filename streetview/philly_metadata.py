"""
Generate metadata for all of Philadelphia:
  1. Sample points along the road network
  2. Query Google Street View Metadata API to get panorama IDs
  3. Output a deploy-ready CSV

Usage:
    python -m streetview.philly_metadata --key YOUR_GOOGLE_API_KEY

    Optional flags:
        --n_points      Number of sample points (default: 10000)
        --min_spacing   Minimum meters between points (default: 50)
        --n_threads     Parallel metadata API requests (default: 10)
        --output_path   Where to save the CSV (default: data/input_metadata/philly_meta.csv)
        --seed          Random seed (default: 42)
"""

import os
import sys
import time
import json
import fire
import random
import logging
import numpy as np
import pandas as pd
import requests
import osmnx as ox
import multiprocessing as mp
from tqdm import tqdm
from geographiclib.geodesic import Geodesic

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PLACE = "Philadelphia, Pennsylvania, USA"
SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


def _get_heading(lat1, lon1, lat2, lon2):
    """Geodesic heading from point 1 to point 2."""
    return Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']


def sample_road_points(n_points=10000, min_spacing=None, seed=42):
    """
    Download Philadelphia's driveable road network and uniformly sample
    *n_points* along it, enforcing a minimum spacing of *min_spacing* meters.
    Returns a DataFrame with columns: lat, lon, road_heading.
    """
    log.info("Downloading road network for %s …", PLACE)
    G = ox.graph_from_place(PLACE, network_type="drive")
    edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True)
    log.info("Road network: %d edges, %.1f km total length",
             len(edges), edges["length"].sum() / 1e3)

    np.random.seed(seed)
    random.seed(seed)

    lengths = edges["length"].values
    total_length = lengths.sum()
    probs = lengths / total_length

    rows = []
    seen = []                       # (lat, lon) of accepted points
    # Over-sample by 3× to account for spacing rejections
    indices = np.random.choice(len(edges), size=n_points * 3, p=probs)

    pbar = tqdm(total=n_points, desc="Sampling road points")
    accepted = 0
    j = 0

    while accepted < n_points and j < len(indices):
        idx = indices[j]
        j += 1
        row = edges.iloc[idx]
        line = row["geometry"]
        offset = np.random.rand() * line.length
        point = line.interpolate(offset)
        lat, lon = point.y, point.x

        # Enforce minimum spacing
        too_close = False
        if min_spacing is not None:
            for _lat, _lon in seen:
                # Quick Euclidean pre-filter (≈ 0.00045° per 50 m)
                if abs(lat - _lat) < 0.001 and abs(lon - _lon) < 0.001:
                    from geopy.distance import distance as geo_dist
                    if geo_dist((lat, lon), (_lat, _lon)).m < min_spacing:
                        too_close = True
                        break
            if too_close:
                continue

        # Compute road heading at this point
        frac = offset / line.length
        start = line.interpolate(max(0, frac - 0.05), normalized=True)
        end = line.interpolate(min(1, frac + 0.05), normalized=True)
        road_heading = _get_heading(start.y, start.x, end.y, end.x)

        rows.append({"lat": lat, "lon": lon, "road_heading": road_heading})
        seen.append((lat, lon))
        accepted += 1
        pbar.update(1)

    pbar.close()
    log.info("Sampled %d points", len(rows))
    return pd.DataFrame(rows)


def _init_worker(api_key):
    global _api_key
    _api_key = api_key


def _query_metadata(row_tuple):
    """Query GSV metadata API for a single (lat, lon). Returns a dict."""
    global _api_key
    idx, row = row_tuple
    lat, lon = row["lat"], row["lon"]
    road_heading = row["road_heading"]

    params = {
        "location": f"{lat},{lon}",
        "key": _api_key,
        "source": "outdoor",
    }

    try:
        resp = requests.get(SV_METADATA_URL, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        return {
            "lat_anchor": lat, "lon_anchor": lon,
            "road_heading": road_heading,
            "status": f"ERROR: {e}",
        }

    if data.get("status") != "OK":
        return {
            "lat_anchor": lat, "lon_anchor": lon,
            "road_heading": road_heading,
            "status": data.get("status", "UNKNOWN"),
        }

    pano_lat = data["location"]["lat"]
    pano_lon = data["location"]["lng"]
    panoid = data["pano_id"]
    capture_date = data.get("date", "")          # "YYYY-MM" or ""

    year, month = None, None
    if capture_date:
        parts = capture_date.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else None

    return {
        "lat_anchor": lat,
        "lon_anchor": lon,
        "lat": pano_lat,
        "lon": pano_lon,
        "panoid": panoid,
        "road_heading": road_heading,
        "year": year,
        "month": month,
        "status": "OK",
    }


def fetch_panorama_ids(points_df, api_key, n_threads=10):
    """
    For every sampled point, query the GSV Metadata API to get the nearest
    panorama ID, its actual coordinates, and capture date.
    Returns a DataFrame.
    """
    log.info("Querying GSV Metadata API for %d points (%d threads) …",
             len(points_df), n_threads)

    with mp.Pool(n_threads,
                 initializer=_init_worker,
                 initargs=(api_key,)) as pool:
        results = list(tqdm(
            pool.imap(_query_metadata, points_df.iterrows()),
            total=len(points_df),
            desc="Fetching panorama IDs",
            smoothing=0.1,
        ))

    meta = pd.DataFrame(results)
    n_ok = (meta["status"] == "OK").sum()
    log.info("API results: %d OK, %d failed/no coverage",
             n_ok, len(meta) - n_ok)
    return meta


def build_deploy_csv(meta, image_dir="./data/rawdata/image"):
    """
    From the raw metadata, produce a CSV that matches the format expected by
    the detection pipeline (see README / detection/lightning/detection.py):
        save_path | panoid | heading | downloaded | annotations | ...

    The heading is rotated 90° from the road direction (perpendicular, looking
    at buildings), randomly choosing left or right.
    """
    df = meta.query("status == 'OK'").copy()
    df = df.drop_duplicates(subset=["panoid"])

    # Perpendicular heading: road_heading ± 90° (random left/right)
    n = len(df)
    flip = (np.random.rand(n) > 0.5).astype(int)
    df["heading"] = ((df["road_heading"] + 90 - 180 * flip) % 360).astype(int)

    # Build save_path that matches download.py naming convention
    df["save_path"] = df.apply(
        lambda r: os.path.join(image_dir, f"{r['panoid']}_{r['heading']}.jpg"),
        axis=1,
    )
    df["gsv_image_path"] = df["save_path"]
    df["downloaded"] = False          # images not downloaded yet
    df["annotations"] = "[]"          # empty annotations for deployment

    keep_cols = [
        "panoid", "heading", "lat", "lon",
        "lat_anchor", "lon_anchor", "road_heading",
        "year", "month",
        "save_path", "gsv_image_path",
        "downloaded", "annotations",
    ]
    return df[keep_cols].reset_index(drop=True)


def generate_philly_metadata(
    key: str,
    n_points: int = 10000,
    min_spacing: int = None,
    n_threads: int = 10,
    output_path: str = "data/input_metadata/philly_meta.csv",
    seed: int = 42,
):
    """
    End-to-end: sample road points → query GSV API → save deploy CSV.

    Args:
        key:          Google Maps API key (must have Street View Static API enabled)
        n_points:     Number of road-network sample points
        min_spacing:  Minimum distance (meters) between sample points
        n_threads:    Parallel threads for the metadata API
        output_path:  Where to write the final CSV
        seed:         Random seed for reproducibility
    """
    points = sample_road_points(n_points=n_points, min_spacing=min_spacing, seed=seed)
    meta = fetch_panorama_ids(points, api_key=key, n_threads=n_threads)
    deploy_df = build_deploy_csv(meta)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    deploy_df.to_csv(output_path, index=False)
    log.info("Saved %d rows to %s", len(deploy_df), output_path)
    log.info("Preview:\n%s", deploy_df.head())


if __name__ == "__main__":
    fire.Fire(generate_philly_metadata)

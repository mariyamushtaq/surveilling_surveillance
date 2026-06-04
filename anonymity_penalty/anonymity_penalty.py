"""
anonymity_penalty.py
────────────────────
Builds a surveillance-weighted road graph from the local Philadelphia edges
shapefile and computes the Anonymity Penalty across a random sample of
origin-destination pairs.

Edge surveillance cost:
    S(e) = sum of confidence scores of all detections within `radius` metres
           of the edge midpoint  (one row per bounding-box detection)

Composite edge weight:
    privacy_weight(e) = length_m(e) + lambda * S(e)

Anonymity Penalty per O-D pair:
    AP_abs = privacy_route_length - shortest_route_length   (metres)
    AP_rel = AP_abs / shortest_route_length                 (fraction)

O-D pairs are sampled randomly from graph nodes, filtered to a minimum
straight-line distance of `min_dist_m` metres (default 1000 m) so trivially
short pairs don't dilute the results.

Usage:
    python anonymity_penalty.py
    python anonymity_penalty.py --radius 50 --lam 500 --n_pairs 200

Outputs (written under anonymity_penalty/routing_output/ next to this script):
    anonymity_penalty_results.csv   — one row per valid O-D pair
    summary.txt                     — aggregate statistics
    route_map.png                   — illustrative pair (shortest vs privacy)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import itertools

# ── paths ─────────────────────────────────────────────────────────────────────
# Script lives in anonymity_penalty/; repository root is one level up.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Road graph: Philadelphia street edges (with fnode_/tnode_/oneway) from nested repo.
EDGES_SHP = os.path.join(
    PROJECT_ROOT,
    "philly_surveillance_feb", "Data", "road_network", "Philadelphia", "edges.shp",
)

# Detections: main pipeline CSV (override with --detections if needed).
DEFAULT_DET = os.path.join(PROJECT_ROOT, "data", "full_run", "positive_detections.csv")

# All script outputs stay inside anonymity_penalty/ (not under data/).
DEFAULT_OUTDIR = os.path.join(SCRIPT_DIR, "routing_output")

# Max distance (m, EPSG:3857) between two placements of the same node ID before warning.
NODE_COORD_TOL_M = 1.0

CRS_GEO    = "EPSG:4326"
CRS_METRIC = "EPSG:3857"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Build graph from shapefile
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(shp_path: str) -> tuple[nx.DiGraph, dict]:
    """
    Build a directed NetworkX graph from edges.shp.

    Node IDs:    fnode_ / tnode_ integers.
    Node coords: derived from edge endpoint geometry, projected to EPSG:3857.
                 First observed coordinate per node ID is kept; later edges that
                 disagree beyond NODE_COORD_TOL_M trigger a warning count.

    Directionality (oneway field):
        B   — both directions → two arcs
        FT  — digitised direction only → fnode_ → tnode_
        TF  — opposite digitised direction → tnode_ → fnode_
        NaN — treated as B

    Edge attributes:
        length_m           from shapefile
        surveillance_cost  initialised 0.0
        privacy_weight     initialised to length_m
    """
    print(f"[graph] Loading {shp_path}")
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise ValueError(
            f"Shapefile has no CRS (.prj missing or unreadable): {shp_path}"
        )
    if gdf.crs.to_epsg() != 3857:
        gdf = gdf.to_crs(CRS_METRIC)

    print(f"[graph] {len(gdf):,} edge rows")

    # ── derive node coordinates from edge endpoints ───────────────────────────
    node_coords: dict[int, tuple[float, float]] = {}
    coord_conflict_edges = 0

    tol_sq = NODE_COORD_TOL_M ** 2

    def register_endpoint(nid: int, xy: tuple[float, float]) -> None:
        nonlocal coord_conflict_edges
        if nid not in node_coords:
            node_coords[nid] = xy
            return
        ox, oy = node_coords[nid]
        dx, dy = xy[0] - ox, xy[1] - oy
        if dx * dx + dy * dy > tol_sq:
            coord_conflict_edges += 1

    for row in gdf.itertuples():
        coords = list(row.geometry.coords)
        fnode  = int(row.fnode_)
        tnode  = int(row.tnode_)
        x0, y0 = coords[0][0],  coords[0][1]
        xn, yn = coords[-1][0], coords[-1][1]
        register_endpoint(fnode, (x0, y0))
        register_endpoint(tnode, (xn, yn))

    print(f"[graph] {len(node_coords):,} unique nodes")
    if coord_conflict_edges:
        print(
            f"[graph] WARNING: {coord_conflict_edges} edge endpoint(s) disagree "
            f"with stored coords for same node ID (tol={NODE_COORD_TOL_M} m); "
            "first-seen coords kept — verify shapefile topology."
        )

    # ── build directed graph ──────────────────────────────────────────────────
    G = nx.DiGraph()

    for nid, (x, y) in node_coords.items():
        G.add_node(nid, x=x, y=y)

    unexpected = 0
    for row in gdf.itertuples():
        fnode  = int(row.fnode_)
        tnode  = int(row.tnode_)
        oneway = str(row.oneway).strip() if pd.notna(row.oneway) else "B"
        length = float(row.length_m) if pd.notna(row.length_m) else 1.0

        attrs = {
            "length_m":          length,
            "surveillance_cost": 0.0,
            "privacy_weight":    length,
        }

        if oneway == "FT":
            G.add_edge(fnode, tnode, **attrs)
        elif oneway == "TF":
            G.add_edge(tnode, fnode, **attrs)
        elif oneway in ("B", "nan", ""):
            G.add_edge(fnode, tnode, **attrs)
            G.add_edge(tnode, fnode, **attrs)
        else:
            # unexpected value — treat as bidirectional
            unexpected += 1
            G.add_edge(fnode, tnode, **attrs)
            G.add_edge(tnode, fnode, **attrs)

    print(f"[graph] {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges"
          + (f"  (unexpected oneway values: {unexpected})" if unexpected else ""))

    return G, node_coords


# ══════════════════════════════════════════════════════════════════════════════
# 2. Detections
# ══════════════════════════════════════════════════════════════════════════════

def load_detections(csv_path: str, threshold: float) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["lat", "lon", "score"])
    df = df[df["score"] >= threshold].reset_index(drop=True)
    print(f"[detections] {len(df):,} rows at score >= {threshold}")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=CRS_GEO,
    ).to_crs(CRS_METRIC)
    return gdf


def build_kdtree(det_gdf: gpd.GeoDataFrame) -> tuple[cKDTree, np.ndarray]:
    coords = np.column_stack([det_gdf.geometry.x, det_gdf.geometry.y])
    scores = det_gdf["score"].to_numpy()
    return cKDTree(coords), scores


# ══════════════════════════════════════════════════════════════════════════════
# 3. Edge weighting
# ══════════════════════════════════════════════════════════════════════════════

def add_surveillance_weights(
    G:         nx.DiGraph,
    node_pos:  dict,
    tree:      cKDTree,
    scores:    np.ndarray,
    radius:    float,
    lam:       float,
) -> nx.DiGraph:
    """
    For every edge (u → v):
        midpoint  = mean of node u and node v projected coordinates
        S(e)      = sum of scores of all detections within `radius` metres
        privacy_weight(e) = length_m(e) + lam * S(e)
    """
    print(f"[weights] radius={radius} m  λ={lam} …")
    n_nonzero = 0

    for u, v, data in G.edges(data=True):
        xu, yu = node_pos[u]
        xv, yv = node_pos[v]
        mx, my = (xu + xv) / 2, (yu + yv) / 2

        idxs = tree.query_ball_point([mx, my], r=radius)
        s    = float(scores[idxs].sum()) if idxs else 0.0

        data["surveillance_cost"] = s
        data["privacy_weight"]    = data["length_m"] + lam * s

        if s > 0:
            n_nonzero += 1

    total = G.number_of_edges()
    print(f"[weights] Edges with cost > 0: "
          f"{n_nonzero:,} / {total:,} ({100 * n_nonzero / total:.1f}%)")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 4. O-D sampling
# ══════════════════════════════════════════════════════════════════════════════

def sample_od_pairs(
    node_pos:   dict,
    n_pairs:    int,
    min_dist_m: float,
    seed:       int,
) -> list[tuple[int, int]]:
    """
    Sample random (origin, dest) node pairs, keeping only those whose
    straight-line distance >= min_dist_m metres. Draws with replacement
    until n_pairs valid pairs are found or attempts are exhausted.
    """
    rng          = random.Random(seed)
    nodes        = list(node_pos.keys())
    pairs        = []
    attempts     = 0
    max_attempts = n_pairs * 20

    while len(pairs) < n_pairs and attempts < max_attempts:
        o, d  = rng.choice(nodes), rng.choice(nodes)
        attempts += 1
        if o == d:
            continue
        ox, oy = node_pos[o]
        dx, dy = node_pos[d]
        if ((ox - dx) ** 2 + (oy - dy) ** 2) ** 0.5 >= min_dist_m:
            pairs.append((o, d))

    print(f"[OD] {len(pairs)} pairs sampled "
          f"(min dist {min_dist_m} m, {attempts} attempts)")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 5. Routing
# ══════════════════════════════════════════════════════════════════════════════

def route_pair(G: nx.DiGraph, origin: int, dest: int) -> dict | None:
    try:
        short_nodes = nx.shortest_path(G, origin, dest, weight="length_m")
        short_len   = nx.path_weight(G, short_nodes, weight="length_m")
        short_surv  = nx.path_weight(G, short_nodes, weight="surveillance_cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    try:
        priv_nodes = nx.shortest_path(G, origin, dest, weight="privacy_weight")
        priv_len   = nx.path_weight(G, priv_nodes, weight="length_m")
        priv_surv  = nx.path_weight(G, priv_nodes, weight="surveillance_cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    ap_abs = priv_len - short_len
    ap_rel = ap_abs / short_len if short_len > 0 else 0.0

    return {
        "origin_node":        origin,
        "dest_node":          dest,
        "shortest_length_m":  round(short_len, 2),
        "shortest_surv_cost": round(short_surv, 4),
        "privacy_length_m":   round(priv_len, 2),
        "privacy_surv_cost":  round(priv_surv, 4),
        "ap_absolute_m":      round(ap_abs, 2),
        "ap_relative":        round(ap_rel, 6),
        "short_route_nodes":  short_nodes,
        "priv_route_nodes":   priv_nodes,
    }


def compute_ap_distribution(
    G:     nx.DiGraph,
    pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    results = []
    for i, (o, d) in enumerate(pairs):
        r = route_pair(G, o, d)
        if r is not None:
            results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(pairs)}] valid so far: {len(results)}")
    print(f"[routing] {len(results)} valid pairs from {len(pairs)} sampled.")
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Outputs
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(
    df:         pd.DataFrame,
    radius:     float,
    lam:        float,
    min_dist_m: float,
    out_path:   str,
):
    ap_abs     = df["ap_absolute_m"]
    ap_pct     = df["ap_relative"] * 100
    surv_delta = df["shortest_surv_cost"] - df["privacy_surv_cost"]

    lines = [
        "Anonymity Penalty Summary",
        "─" * 52,
        f"radius={radius} m  |  λ={lam}  |  min OD dist={min_dist_m} m  |  n={len(df)}",
        "",
        "AP absolute (extra metres):",
        f"  mean    {ap_abs.mean():.1f} m",
        f"  median  {ap_abs.median():.1f} m",
        f"  p75     {ap_abs.quantile(0.75):.1f} m",
        f"  p90     {ap_abs.quantile(0.90):.1f} m",
        f"  max     {ap_abs.max():.1f} m",
        f"  AP = 0  {(ap_abs == 0).sum()} pairs ({100*(ap_abs==0).mean():.1f}%)",
        "",
        "AP relative (% detour over shortest):",
        f"  mean    {ap_pct.mean():.2f}%",
        f"  median  {ap_pct.median():.2f}%",
        f"  p90     {ap_pct.quantile(0.90):.2f}%",
        f"  max     {ap_pct.max():.2f}%",
        "",
        "Surveillance cost: shortest → privacy route:",
        f"  mean reduction    {surv_delta.mean():.4f}",
        f"  median reduction  {surv_delta.median():.4f}",
        f"  pairs where privacy route = shortest: {(ap_abs <= 0).sum()}",
    ]

    text = "\n".join(lines)
    print(f"\n{text}\n")
    Path(out_path).write_text(text)
    print(f"[summary] → {out_path}")


def plot_route_comparison(
    G:        nx.DiGraph,
    node_pos: dict,
    result:   dict,
    det_gdf:  gpd.GeoDataFrame,
    out_path: str,
):
    import pyproj
    transformer = pyproj.Transformer.from_crs(CRS_METRIC, CRS_GEO, always_xy=True)

    def to_lonlat(nodes):
        xs = [node_pos[n][0] for n in nodes]
        ys = [node_pos[n][1] for n in nodes]
        lons, lats = transformer.transform(xs, ys)
        return lons, lats

    fig, ax = plt.subplots(figsize=(10, 10))

    # detection scatter in lon/lat
    det_lons, det_lats = transformer.transform(
        det_gdf.geometry.x.to_numpy(),
        det_gdf.geometry.y.to_numpy(),
    )
    ax.scatter(det_lons, det_lats, s=10, c="orange", alpha=0.4,
               zorder=2, label="Detections")

    # routes
    s_lons, s_lats = to_lonlat(result["short_route_nodes"])
    p_lons, p_lats = to_lonlat(result["priv_route_nodes"])

    ax.plot(s_lons, s_lats, color="#1f77b4", lw=2.5, zorder=3,
            label=f"Shortest ({result['shortest_length_m']/1000:.2f} km, "
                  f"surv={result['shortest_surv_cost']:.2f})")
    ax.plot(p_lons, p_lats, color="#d62728", lw=2.5, zorder=3, ls="--",
            label=f"Privacy-optimal ({result['privacy_length_m']/1000:.2f} km, "
                  f"surv={result['privacy_surv_cost']:.2f})")

    # origin / destination
    o_lon, o_lat = transformer.transform(
        node_pos[result["origin_node"]][0],
        node_pos[result["origin_node"]][1],
    )
    d_lon, d_lat = transformer.transform(
        node_pos[result["dest_node"]][0],
        node_pos[result["dest_node"]][1],
    )
    ax.scatter([o_lon], [o_lat], s=150, c="green",  zorder=5, label="Origin")
    ax.scatter([d_lon], [d_lat], s=150, c="purple", zorder=5, label="Destination")

    ap_pct = result["ap_relative"] * 100
    ax.set_title(
        f"Anonymity Penalty: +{result['ap_absolute_m']:.0f} m ({ap_pct:.1f}% detour)\n"
        f"Surveillance cost: {result['shortest_surv_cost']:.2f} → "
        f"{result['privacy_surv_cost']:.2f}",
        fontsize=11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Multiple output plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_multiple_routes(
    G:        nx.DiGraph,
    node_pos: dict,
    results_df: pd.DataFrame,
    det_gdf:  gpd.GeoDataFrame,
    out_dir:  str,
    n:        int = 4,
    seed:     int = 0,
):
    """
    Plot n random O-D pairs (excluding the max-AP pair already saved),
    each as a separate PNG. Picks pairs with AP > 0 so there's something
    visible to show.
    """
    import pyproj
    transformer = pyproj.Transformer.from_crs(CRS_METRIC, CRS_GEO, always_xy=True)

    def to_lonlat(nodes):
        xs = [node_pos[nd][0] for nd in nodes]
        ys = [node_pos[nd][1] for nd in nodes]
        lons, lats = transformer.transform(xs, ys)
        return lons, lats

    # only pairs where the two routes actually differ
    candidates = results_df[results_df["ap_absolute_m"] > 0].copy()
    # exclude the max-AP row (already in route_map.png)
    candidates = candidates[
        candidates["ap_absolute_m"] < candidates["ap_absolute_m"].max()
    ]

    sample = candidates.sample(min(n, len(candidates)), random_state=seed)

    det_lons, det_lats = transformer.transform(
        det_gdf.geometry.x.to_numpy(),
        det_gdf.geometry.y.to_numpy(),
    )

    for i, (_, row) in enumerate(sample.iterrows()):
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(det_lons, det_lats, s=10, c="orange", alpha=0.4,
                   zorder=2, label="Detections")

        s_lons, s_lats = to_lonlat(row["short_route_nodes"])
        p_lons, p_lats = to_lonlat(row["priv_route_nodes"])

        ax.plot(s_lons, s_lats, color="#1f77b4", lw=2.5, zorder=3,
                label=f"Shortest ({row['shortest_length_m']/1000:.2f} km, "
                      f"surv={row['shortest_surv_cost']:.2f})")
        ax.plot(p_lons, p_lats, color="#d62728", lw=2.5, zorder=3, ls="--",
                label=f"Privacy-optimal ({row['privacy_length_m']/1000:.2f} km, "
                      f"surv={row['privacy_surv_cost']:.2f})")

        o_lon, o_lat = transformer.transform(
            node_pos[row["origin_node"]][0], node_pos[row["origin_node"]][1])
        d_lon, d_lat = transformer.transform(
            node_pos[row["dest_node"]][0],  node_pos[row["dest_node"]][1])

        ax.scatter([o_lon], [o_lat], s=150, c="green",  zorder=5, label="Origin")
        ax.scatter([d_lon], [d_lat], s=150, c="purple", zorder=5, label="Destination")

        ap_pct = row["ap_relative"] * 100
        ax.set_title(
            f"Anonymity Penalty: +{row['ap_absolute_m']:.0f} m ({ap_pct:.1f}% detour)\n"
            f"Surveillance cost: {row['shortest_surv_cost']:.2f} → "
            f"{row['privacy_surv_cost']:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"route_map_sample_{i+1}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] sample {i+1} → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════════════════════

def main(
    detections:  str   = DEFAULT_DET,
    edges_shp:   str   = EDGES_SHP,
    threshold:   float = 0.4,
    radius:      float = 50.0,
    lam:         float = 500.0,
    n_pairs:     int   = 100,
    min_dist_m:  float = 1000.0,
    seed:        int   = 42,
    out_dir:     str   = DEFAULT_OUTDIR,
):
    """
    Args:
        detections   Path to positive_detections.csv
        edges_shp    Path to edges shapefile
        threshold    Min confidence score to include (default 0.4)
        radius       Spatial join radius in metres (default 50)
        lam          Lambda penalty scalar (default 500)
        n_pairs      Number of random O-D pairs to sample (default 100)
        min_dist_m   Min straight-line OD distance in metres (default 1000)
        seed         Random seed (default 42)
        out_dir      Output directory (default anonymity_penalty/routing_output/)
    """
    os.makedirs(out_dir, exist_ok=True)

    G, node_pos   = load_graph(edges_shp)
    det_gdf       = load_detections(detections, threshold)
    tree, scores  = build_kdtree(det_gdf)
    G             = add_surveillance_weights(G, node_pos, tree, scores, radius, lam)
    pairs         = sample_od_pairs(node_pos, n_pairs, min_dist_m, seed)

    print("[routing] Computing Anonymity Penalty …")
    results_df = compute_ap_distribution(G, pairs)

    if len(results_df) == 0:
        print("[error] No valid routes found. Check graph connectivity.")
        return

    plot_multiple_routes(G, node_pos, results_df, det_gdf, out_dir, n=4, seed=0)
    save_cols = [c for c in results_df.columns if "nodes" not in c]
    out_csv   = os.path.join(out_dir, "anonymity_penalty_results.csv")
    results_df[save_cols].to_csv(out_csv, index=False)
    print(f"[out] → {out_csv}")

    write_summary(results_df, radius, lam, min_dist_m,
                  os.path.join(out_dir, "summary.txt"))

    median_idx   = (
        results_df["ap_absolute_m"] - results_df["ap_absolute_m"].median()
    ).abs().idxmin()
    illustrative = results_df.loc[results_df["ap_absolute_m"].idxmax()].to_dict()
    plot_route_comparison(
        G, node_pos, illustrative, det_gdf,
        os.path.join(out_dir, "route_map.png"),
    )


if __name__ == "__main__":
    fire.Fire(main)
"""
Merge Philly full-run positive detections with ALPR detections and plot together.

- Philly rows: data/full_run/positive_detections.csv (already ≥ run threshold, default 0.4).
- ALPR rows: joined to alpr_meta for lat/lon/heading; filtered by min_score (default 0.4)
  so scales match the surveillance full run.

Writes:
  data/full_run/combined_positive_detections.csv
  data/full_run/visualizations_combined/*.png (+ optional annotated/)

Usage (from repository root):
  python scripts/visualize_combined_detections.py
  python scripts/visualize_combined_detections.py --min_score 0.3 --no_annotated
  python scripts/visualize_combined_detections.py --alpr_detections_csv path/to/all_detections.csv
"""

from __future__ import annotations

import ast
import json
import os

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from visualize_full_run_detections import draw_detections

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_PHILLY_CSV = os.path.join(PROJECT_ROOT, "data", "full_run", "positive_detections.csv")
DEFAULT_ALPR_DET = os.path.join(
    PROJECT_ROOT, "alpr_data", "threshold_analysis", "all_detections.csv"
)
DEFAULT_ALPR_META = os.path.join(
    PROJECT_ROOT, "alpr_data", "threshold_analysis", "alpr_meta.csv"
)
DEFAULT_PHILLY_IMAGES = os.path.join(PROJECT_ROOT, "data", "full_run", "images")
DEFAULT_ALPR_IMAGES = os.path.join(
    PROJECT_ROOT, "alpr_data", "threshold_analysis", "images"
)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "full_run", "visualizations_combined")
DEFAULT_COMBINED_CSV = os.path.join(
    PROJECT_ROOT, "data", "full_run", "combined_positive_detections.csv"
)


def _load_bbox_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    df["bbox"] = df["bbox"].apply(ast.literal_eval)
    return df


def build_combined_frame(
    philly_csv: str = DEFAULT_PHILLY_CSV,
    alpr_detections_csv: str = DEFAULT_ALPR_DET,
    alpr_meta_csv: str = DEFAULT_ALPR_META,
    min_score: float = 0.4,
    save_csv: str | None = DEFAULT_COMBINED_CSV,
) -> pd.DataFrame:
    philly = _load_bbox_df(philly_csv)
    philly = philly.copy()
    philly["origin"] = "philly_surveillance"

    alpr = pd.read_csv(alpr_detections_csv)
    if alpr.empty:
        raise ValueError(f"Empty ALPR detections: {alpr_detections_csv}")
    alpr = alpr.loc[alpr["score"] >= min_score].copy()
    alpr["bbox"] = alpr["bbox"].apply(ast.literal_eval)

    meta = pd.read_csv(alpr_meta_csv)
    meta = meta.rename(columns={"image_id": "_img_key"})
    alpr["_img_key"] = alpr["image"].str.replace(".jpg", "", regex=False)
    keep_meta = ["_img_key", "panoid", "heading", "lat", "lon"]
    missing = [c for c in keep_meta if c not in meta.columns]
    if missing:
        raise ValueError(f"alpr_meta missing columns: {missing}")
    alpr = alpr.merge(meta[keep_meta], on="_img_key", how="left")
    alpr = alpr.drop(columns=["_img_key"])
    n_miss = alpr["lat"].isna().sum()
    if n_miss:
        print(f"  Warning: {n_miss} ALPR detections missing lat/lon after meta join")

    alpr["origin"] = "alpr"

    cols = [
        "image",
        "panoid",
        "heading",
        "lat",
        "lon",
        "class",
        "class_id",
        "score",
        "bbox",
        "origin",
    ]
    for c in cols:
        if c not in philly.columns and c != "origin":
            philly[c] = np.nan
        if c not in alpr.columns and c != "origin":
            alpr[c] = np.nan

    combined = pd.concat(
        [philly[cols], alpr[cols]], ignore_index=True, sort=False
    )
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        combined.to_csv(save_csv, index=False)
        print(f"Wrote {save_csv} ({len(combined)} rows)")
    return combined


def plot_score_by_origin(df, out_dir: str, threshold_line: float = 0.4):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for ax, origin, color, label in [
        (axes[0], "philly_surveillance", "#4C72B0", "Philly surveillance sample"),
        (axes[0], "alpr", "#DD8452", "ALPR sites"),
    ]:
        sub = df.loc[df["origin"] == origin, "score"]
        if len(sub) == 0:
            continue
        ax.hist(
            sub,
            bins=50,
            range=(0, 1),
            color=color,
            edgecolor="white",
            alpha=0.55,
            label=f"{label} (n={len(sub)})",
        )

    axes[0].axvline(
        x=threshold_line,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"ALPR filter / ref line ({threshold_line})",
    )
    axes[0].set_xlabel("Confidence score")
    axes[0].set_ylabel("Detections")
    axes[0].set_title(f"Score distribution by origin (combined n={len(df)})")
    axes[0].legend()

    scores = df["score"].to_numpy()
    c0 = df["class_id"] == 0
    c1 = df["class_id"] == 1
    if c0.any():
        axes[1].hist(
            scores[c0],
            bins=50,
            range=(0, 1),
            color="#4C72B0",
            edgecolor="white",
            alpha=0.6,
            label=f"Directed (n={int(c0.sum())})",
        )
    if c1.any():
        axes[1].hist(
            scores[c1],
            bins=50,
            range=(0, 1),
            color="#DD8452",
            edgecolor="white",
            alpha=0.6,
            label=f"Dome (n={int(c1.sum())})",
        )
    axes[1].axvline(
        x=threshold_line,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Ref ({threshold_line})",
    )
    axes[1].set_xlabel("Confidence score")
    axes[1].set_ylabel("Detections")
    axes[1].set_title("Combined sample — per class")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "confidence_by_origin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_class_by_origin(df, out_dir: str):
    ct = df.groupby(["class", "origin"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"][: ct.shape[1]])
    ax.set_ylabel("Detections")
    ax.set_title("Detections by class and origin")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "class_by_origin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_spatial_by_origin(df, out_dir: str):
    geo = df.dropna(subset=["lat", "lon"])
    if geo.empty:
        print("  Skip spatial plots: no lat/lon")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    styles = [
        ("philly_surveillance", "#1f77b4", "o", "Philly SV"),
        ("alpr", "#d62728", "^", "ALPR"),
    ]
    for origin, color, marker, lab in styles:
        sub = geo[geo["origin"] == origin]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=14,
            alpha=0.45,
            c=color,
            marker=marker,
            label=f"{lab} (n={len(sub)})",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Combined detections — origin × location")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, "spatial_by_origin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    for origin, cmap in [("philly_surveillance", "Blues"), ("alpr", "Oranges")]:
        sub = geo[geo["origin"] == origin]
        if len(sub) < 2:
            continue
        hb = ax2.hexbin(
            sub["lon"],
            sub["lat"],
            gridsize=35,
            cmap=cmap,
            mincnt=1,
            alpha=0.65,
            label=origin,
        )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Density overlay (hexbin; two cmaps)")
    ax2.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "spatial_hexbin_overlay.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Wrote {p2}")


def plot_heading_by_origin(df, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for origin, color in [("philly_surveillance", "#8172B3"), ("alpr", "#55A868")]:
        sub = df.loc[df["origin"] == origin, "heading"].dropna().astype(float)
        if sub.empty:
            continue
        ax.hist(
            sub,
            bins=36,
            range=(0, 360),
            color=color,
            edgecolor="white",
            alpha=0.5,
            label=f"{origin} (n={len(sub)})",
        )
    ax.set_xlabel("Heading (degrees)")
    ax.set_ylabel("Detections")
    ax.set_title("Heading distribution by origin")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "heading_by_origin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def write_annotated_split(
    df,
    philly_image_dir: str,
    alpr_image_dir: str,
    ann_dir: str,
    max_annotated: int | None = None,
):
    os.makedirs(ann_dir, exist_ok=True)
    grouped = df.groupby("image")
    names = sorted(grouped.groups.keys())
    if max_annotated is not None:
        names = names[: int(max_annotated)]

    wrote, missing = 0, 0
    for image_name in names:
        group = grouped.get_group(image_name)
        origin = group["origin"].iloc[0]
        base = philly_image_dir if origin == "philly_surveillance" else alpr_image_dir
        img_path = os.path.join(base, image_name)
        if not os.path.isfile(img_path):
            missing += 1
            continue
        records = group.to_dict("records")
        img = Image.open(img_path).convert("RGB")
        img = draw_detections(img, records)
        stem = os.path.splitext(image_name)[0]
        out_path = os.path.join(ann_dir, f"{stem}_det.jpg")
        img.save(out_path)
        wrote += 1

    print(f"  Annotated: {wrote} → {ann_dir}/")
    if missing:
        print(f"  Missing source images: {missing}")


def write_summary_json(df, out_dir: str, min_score: float):
    by_o = df.groupby("origin").agg(
        n_detections=("score", "count"),
        n_images=("image", "nunique"),
        mean_score=("score", "mean"),
    )
    summary = {
        "min_score_alpr": min_score,
        "philly_note": "Rows are already filtered at full-run pipeline threshold",
        "totals": by_o.to_dict("index"),
        "combined_n_detections": int(len(df)),
        "combined_n_images": int(df["image"].nunique()),
    }
    path = os.path.join(out_dir, "combined_score_stats.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {path}")


def main(
    philly_csv: str = DEFAULT_PHILLY_CSV,
    alpr_detections_csv: str = DEFAULT_ALPR_DET,
    alpr_meta_csv: str = DEFAULT_ALPR_META,
    philly_image_dir: str = DEFAULT_PHILLY_IMAGES,
    alpr_image_dir: str = DEFAULT_ALPR_IMAGES,
    output_dir: str = DEFAULT_OUT,
    combined_csv: str = DEFAULT_COMBINED_CSV,
    min_score: float = 0.4,
    threshold_line: float = 0.4,
    max_annotated: int | None = None,
    no_annotated: bool = False,
):
    """Build merged CSV and combined visualization bundle."""
    if not os.path.isfile(philly_csv):
        raise SystemExit(f"Missing {philly_csv}")
    if not os.path.isfile(alpr_detections_csv):
        raise SystemExit(f"Missing {alpr_detections_csv}")

    df = build_combined_frame(
        philly_csv=philly_csv,
        alpr_detections_csv=alpr_detections_csv,
        alpr_meta_csv=alpr_meta_csv,
        min_score=min_score,
        save_csv=combined_csv,
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Figures → {output_dir}/")
    plot_score_by_origin(df, output_dir, threshold_line=threshold_line)
    plot_class_by_origin(df, output_dir)
    plot_heading_by_origin(df, output_dir)
    plot_spatial_by_origin(df, output_dir)
    write_summary_json(df, output_dir, min_score)

    if not no_annotated:
        ann = os.path.join(output_dir, "annotated")
        write_annotated_split(
            df,
            philly_image_dir,
            alpr_image_dir,
            ann,
            max_annotated=max_annotated,
        )

    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)

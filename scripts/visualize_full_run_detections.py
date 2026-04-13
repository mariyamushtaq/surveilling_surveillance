"""
Figures and annotated images for Philadelphia full-run positive detections.

Reads data/full_run/positive_detections.csv and writes to data/full_run/visualizations/.

Usage (from repository root):
    python scripts/visualize_full_run_detections.py
    python scripts/visualize_full_run_detections.py --max_annotated=100
    python scripts/visualize_full_run_detections.py --no_annotated
"""

import os
import json
import ast

import fire
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

DEFAULT_CSV = "data/full_run/positive_detections.csv"
DEFAULT_IMAGE_DIR = "data/full_run/images"
DEFAULT_OUT = "data/full_run/visualizations"
CLASS_NAMES = ["Directed Camera", "Dome Camera"]
CLASS_COLORS = {0: (20, 200, 60), 1: (255, 165, 0)}


def _font():
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except OSError:
        return ImageFont.load_default()


def draw_detections(img, detections):
    draw = ImageDraw.Draw(img)
    font = _font()
    for det in detections:
        bbox = det["bbox"]
        score = det["score"]
        class_id = int(det["class_id"])
        color = CLASS_COLORS.get(class_id, (255, 0, 0))
        label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        x1, y1, x2, y2 = bbox
        pad = 30
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(img.width, x2 + pad), min(img.height, y2 + pad)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.rectangle([x1p, y1p, x2p, y2p], outline=color, width=1)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - th - 4)
        draw.rectangle([x1, text_y, x1 + tw + 4, text_y + th + 4], fill=color)
        draw.text((x1 + 2, text_y + 2), label, fill="black", font=font)
    return img


def _load_df(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"No rows in {csv_path}")
    df["bbox"] = df["bbox"].apply(ast.literal_eval)
    return df


def plot_score_bundle(df, out_dir, threshold_line=0.4):
    scores = df["score"].to_numpy()
    c0 = df["class_id"] == 0
    c1 = df["class_id"] == 1
    scores_c0 = scores[c0]
    scores_c1 = scores[c1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].hist(scores, bins=50, range=(0, 1), color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(x=threshold_line, color="red", linestyle="--", linewidth=1.5,
                    label=f"Run threshold ({threshold_line})")
    axes[0].set_xlabel("Confidence score")
    axes[0].set_ylabel("Number of detections")
    axes[0].set_title(f"Detection confidence distribution (n={len(scores)})")
    axes[0].legend()

    if len(scores_c0) > 0:
        axes[1].hist(scores_c0, bins=50, range=(0, 1), color="#4C72B0", edgecolor="white",
                     alpha=0.6, label=f"Directed (n={len(scores_c0)})")
    if len(scores_c1) > 0:
        axes[1].hist(scores_c1, bins=50, range=(0, 1), color="#DD8452", edgecolor="white",
                     alpha=0.6, label=f"Dome (n={len(scores_c1)})")
    axes[1].axvline(x=threshold_line, color="red", linestyle="--", linewidth=1.5,
                    label=f"Run threshold ({threshold_line})")
    axes[1].set_xlabel("Confidence score")
    axes[1].set_ylabel("Number of detections")
    axes[1].set_title("Per-class confidence distribution")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "confidence_score_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    tvals = np.linspace(0, 1, 200)
    retained = [(scores >= t).sum() for t in tvals]
    ax2.plot(tvals, retained, color="#4C72B0", linewidth=2)
    ax2.axvline(x=threshold_line, color="red", linestyle="--", linewidth=1.5,
                label=f"Run threshold ({threshold_line})")
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("Detections retained (≥ threshold)")
    ax2.set_title("Detections retained vs. threshold (this run’s score mass)")
    ax2.legend()
    p2 = os.path.join(out_dir, "detections_vs_threshold.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Wrote {p2}")

    stats = {
        "total_detections": int(len(scores)),
        "directed_camera_count": int(c0.sum()),
        "dome_camera_count": int(c1.sum()),
        "mean_score": round(float(scores.mean()), 4),
        "median_score": round(float(np.median(scores)), 4),
        "std_score": round(float(scores.std()), 4),
        "min_score": round(float(scores.min()), 4),
        "max_score": round(float(scores.max()), 4),
        "images_with_detection": int(df["image"].nunique()),
        "threshold_line_plotted": threshold_line,
    }
    sp = os.path.join(out_dir, "score_stats.json")
    with open(sp, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote {sp}")


def plot_class_counts(df, out_dir):
    counts = df.groupby("class").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"][: len(counts)])
    ax.set_ylabel("Detections")
    ax.set_xlabel("")
    ax.set_title("Detections by class")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_detections_per_image(df, out_dir):
    per = df.groupby("image").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(per, bins=range(1, int(per.max()) + 2), align="left", rwidth=0.85, color="#55A868")
    ax.set_xlabel("Detections per image")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Images with ≥1 detection: {len(per)}")
    plt.tight_layout()
    path = os.path.join(out_dir, "detections_per_image.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_heading_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["heading"].astype(float), bins=36, range=(0, 360), color="#8172B3", edgecolor="white")
    ax.set_xlabel("Heading (degrees)")
    ax.set_ylabel("Detections")
    ax.set_title("Detections by panorama heading")
    plt.tight_layout()
    path = os.path.join(out_dir, "heading_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_spatial(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 10))
    for cid, color, label in [(0, "#1f77b4", "Directed"), (1, "#ff7f0e", "Dome")]:
        sub = df[df["class_id"] == cid]
        if len(sub) == 0:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=8, alpha=0.5, c=color, label=f"{label} (n={len(sub)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Detection locations (Street View sample positions)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", markerscale=2)
    plt.tight_layout()
    path = os.path.join(out_dir, "spatial_detections_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    hb = ax2.hexbin(df["lon"], df["lat"], gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax2, label="Count")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Detection density (hexbin)")
    ax2.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "spatial_detections_hexbin.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Wrote {p2}")


def write_annotated(df, image_dir, ann_dir, max_annotated=None):
    os.makedirs(ann_dir, exist_ok=True)
    grouped = df.groupby("image")
    names = sorted(grouped.groups.keys())
    if max_annotated is not None:
        names = names[: int(max_annotated)]

    missing = 0
    wrote = 0
    for image_name in names:
        img_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(img_path):
            missing += 1
            continue
        group = grouped.get_group(image_name)
        records = group.to_dict("records")
        img = Image.open(img_path).convert("RGB")
        img = draw_detections(img, records)
        stem = os.path.splitext(image_name)[0]
        out_path = os.path.join(ann_dir, f"{stem}_det.jpg")
        img.save(out_path)
        wrote += 1

    print(f"  Annotated images: {wrote} saved under {ann_dir}/")
    if missing:
        print(f"  ({missing} source images missing under {image_dir})")


def main(csv_path=DEFAULT_CSV,
         image_dir=DEFAULT_IMAGE_DIR,
         output_dir=DEFAULT_OUT,
         threshold_line=0.4,
         max_annotated=None,
         no_annotated=False):
    """Generate matplotlib figures and optional annotated JPEGs for full-run detections."""
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing CSV: {csv_path}")

    df = _load_df(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded {len(df)} detections, {df['image'].nunique()} images from {csv_path}")
    print(f"Writing figures to {output_dir}/ …")

    plot_score_bundle(df, output_dir, threshold_line=threshold_line)
    plot_class_counts(df, output_dir)
    plot_detections_per_image(df, output_dir)
    plot_heading_distribution(df, output_dir)
    plot_spatial(df, output_dir)

    if not no_annotated:
        ann = os.path.join(output_dir, "annotated")
        write_annotated(df, image_dir, ann, max_annotated=max_annotated)

    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)

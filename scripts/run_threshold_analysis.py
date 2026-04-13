"""
Threshold analysis: sample 500 images from philly_meta.csv, download them,
run Faster R-CNN detection with threshold=0, and plot the confidence score
distribution to inform threshold selection.

Usage (from repository root):
    python scripts/run_threshold_analysis.py --key YOUR_API_KEY

Optional:
    --n_sample        Number of images to sample (default: 500)
    --ckpt_path       Path to model checkpoint (default: detection/model/best.ckpt)
    --device          cpu or cuda (default: cpu)
    --seed            Random seed (default: 42)
"""

import os
import sys
import json

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "detection"))
import glob
import hashlib
import hmac
import base64
import urllib.parse as urlparse

import fire
import requests
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util import constants as C

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata

DET_META = Metadata()
DET_META.thing_classes = ["Directed Camera", "Dome Camera"]
DET_META.thing_colors = [[20, 200, 60], [11, 119, 32]]

OUT_DIR = "data/threshold_analysis"
IMAGE_DIR = os.path.join(OUT_DIR, "images")
DETECT_DIR = os.path.join(OUT_DIR, "detections")
META_CSV = "data/input_metadata/philly_meta.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Sample rows from philly_meta.csv
# ═══════════════════════════════════════════════════════════════════════════════

def sample_metadata(n_sample=500, seed=42):
    df = pd.read_csv(META_CSV)
    sample = df.sample(n=min(n_sample, len(df)), random_state=seed)
    print(f"\nStep 1: Sampled {len(sample)} rows from {META_CSV} ({len(df)} total)")
    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Download images
# ═══════════════════════════════════════════════════════════════════════════════

def _build_url(panoid, heading, key, sec=None):
    url_str = (f"https://maps.googleapis.com/maps/api/streetview?"
               f"size={C.SV_SIZE}&pano={panoid}&fov={C.SV_FOV}&"
               f"heading={heading}&pitch={C.SV_PITCH}&key={key}")
    if sec:
        parsed = urlparse.urlparse(url_str)
        url_to_sign = parsed.path + "?" + parsed.query
        decoded_key = base64.urlsafe_b64decode(sec)
        signature = hmac.new(decoded_key, url_to_sign.encode(), hashlib.sha1)
        encoded_sig = base64.urlsafe_b64encode(signature.digest()).decode()
        url_str += "&signature=" + encoded_sig
    return url_str


def download_images(sample_df, api_key, sec=None):
    print(f"\nStep 2: Downloading {len(sample_df)} images …")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    ok, skipped, errors = 0, 0, []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="   Download"):
        panoid = row["panoid"]
        heading = int(row["heading"])
        filename = f"{panoid}_{heading}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            skipped += 1
            continue

        url = _build_url(panoid, heading, api_key, sec)
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                ok += 1
            else:
                errors.append((panoid, heading, resp.status_code, len(resp.content)))
        except Exception as e:
            errors.append((panoid, heading, "ERR", str(e)))

    print(f"   Downloaded: {ok}  |  Cached: {skipped}  |  Failed: {len(errors)}")
    if errors:
        for p, h, s, d in errors[:5]:
            print(f"      {p}_{h}: status={s}, detail={d}")
    return ok + skipped


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Run detection with threshold=0 (capture everything)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device="cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.RETINANET.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model.model."):
            new_state[k[len("model.model."):]] = v
        elif k.startswith("model."):
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def prepare_image(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    image_tensor = torch.as_tensor(
        img[:, :, ::-1].copy().astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor, "height": h, "width": w,
            "file_name": image_path}


def run_detection(ckpt_path, device="cpu"):
    """Run detection with conf_threshold=0 to capture the full score distribution."""
    print(f"\nStep 3: Running detection (threshold=0, capturing all scores) …")
    model = load_model(ckpt_path, device=device)

    os.makedirs(DETECT_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not images:
        print(f"   No .jpg files found in {IMAGE_DIR}")
        return []

    print(f"   Processing {len(images)} images …")
    all_detections = []
    iou_threshold = 0.5

    for img_path in tqdm(images, desc="   Detect"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        inp = prepare_image(img_path)

        with torch.no_grad():
            preds = model([inp])[0]

        instances = preds["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        classes = instances.pred_classes

        if len(scores) == 0:
            continue

        # NMS only (no confidence filter — we want all scores)
        keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes, scores, classes = boxes[keep_nms], scores[keep_nms], classes[keep_nms]

        for j in range(len(scores)):
            det = {
                "image": os.path.basename(img_path),
                "class": DET_META.thing_classes[int(classes[j])],
                "class_id": int(classes[j]),
                "score": round(float(scores[j]), 6),
                "bbox": [int(x) for x in boxes[j].tolist()],
            }
            all_detections.append(det)

        img_dets = [d for d in all_detections
                    if d["image"] == os.path.basename(img_path)]
        with open(os.path.join(DETECT_DIR, f"{basename}_det.json"), "w") as f:
            json.dump({"image": os.path.basename(img_path),
                       "detections": img_dets}, f, indent=2)

    # Save all detections to a single CSV
    det_csv = os.path.join(OUT_DIR, "all_detections.csv")
    if all_detections:
        pd.DataFrame(all_detections).to_csv(det_csv, index=False)
    print(f"   {len(all_detections)} total detections across {len(images)} images")
    return all_detections


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Plot and analyze
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(all_detections):
    print(f"\nStep 4: Analyzing score distribution …")
    if not all_detections:
        print("   No detections to analyze.")
        return

    scores = np.array([d["score"] for d in all_detections])
    class_ids = np.array([d["class_id"] for d in all_detections])
    scores_c0 = scores[class_ids == 0]
    scores_c1 = scores[class_ids == 1]

    # ── Print stats ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONFIDENCE SCORE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total detections  : {len(scores)}")
    print(f"    Directed Camera : {len(scores_c0)}")
    print(f"    Dome Camera     : {len(scores_c1)}")
    print(f"  Mean score        : {scores.mean():.4f}")
    print(f"  Median score      : {np.median(scores):.4f}")
    print(f"  Std dev           : {scores.std():.4f}")
    print(f"  Min / Max         : {scores.min():.4f} / {scores.max():.4f}")

    print(f"\n  Bin distribution:")
    print(f"  {'Bin':>12s}  {'Count':>6s}  {'Pct':>6s}  {'Directed':>8s}  {'Dome':>8s}")
    print(f"  {'-'*48}")
    bins_edges = np.arange(0, 1.1, 0.1)
    for i in range(len(bins_edges) - 1):
        lo, hi = bins_edges[i], bins_edges[i + 1]
        mask = (scores >= lo) & (scores < hi) if hi < 1.0 else (scores >= lo) & (scores <= hi)
        c0_count = int(((scores_c0 >= lo) & (scores_c0 < hi) if hi < 1.0
                        else (scores_c0 >= lo) & (scores_c0 <= hi)).sum())
        c1_count = int(((scores_c1 >= lo) & (scores_c1 < hi) if hi < 1.0
                        else (scores_c1 >= lo) & (scores_c1 <= hi)).sum())
        count = int(mask.sum())
        pct = count / len(scores) * 100
        print(f"  [{lo:.1f}, {hi:.1f})  {count:6d}  {pct:5.1f}%  {c0_count:8d}  {c1_count:8d}")

    # Detections at various thresholds
    print(f"\n  Detections retained at different thresholds:")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = int((scores >= t).sum())
        print(f"    threshold={t:.1f}  →  {n:5d} detections ({n/len(scores)*100:5.1f}%)")
    print(f"{'='*60}")

    # ── Plot 1: Combined histogram ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(scores, bins=50, range=(0, 1), color="#4C72B0", edgecolor="white",
                 alpha=0.85)
    axes[0].axvline(x=0.3, color="red", linestyle="--", linewidth=1.5,
                    label="Default threshold (0.3)")
    axes[0].set_xlabel("Confidence Score", fontsize=12)
    axes[0].set_ylabel("Number of Detections", fontsize=12)
    axes[0].set_title(f"Detection Confidence Score Distribution (n={len(scores)})",
                      fontsize=14)
    axes[0].legend(fontsize=11)

    # ── Plot 2: Per-class histogram ───────────────────────────────────────
    if len(scores_c0) > 0:
        axes[1].hist(scores_c0, bins=50, range=(0, 1), color="#4C72B0",
                     edgecolor="white", alpha=0.6, label=f"Directed Camera (n={len(scores_c0)})")
    if len(scores_c1) > 0:
        axes[1].hist(scores_c1, bins=50, range=(0, 1), color="#DD8452",
                     edgecolor="white", alpha=0.6, label=f"Dome Camera (n={len(scores_c1)})")
    axes[1].axvline(x=0.3, color="red", linestyle="--", linewidth=1.5,
                    label="Default threshold (0.3)")
    axes[1].set_xlabel("Confidence Score", fontsize=12)
    axes[1].set_ylabel("Number of Detections", fontsize=12)
    axes[1].set_title("Per-Class Confidence Score Distribution", fontsize=14)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "confidence_score_distribution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {plot_path}")

    # ── Plot 3: Cumulative — detections retained vs threshold ─────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0, 1, 200)
    retained = [int((scores >= t).sum()) for t in thresholds]
    ax2.plot(thresholds, retained, color="#4C72B0", linewidth=2)
    ax2.axvline(x=0.3, color="red", linestyle="--", linewidth=1.5,
                label="Default threshold (0.3)")
    ax2.set_xlabel("Confidence Threshold", fontsize=12)
    ax2.set_ylabel("Detections Retained", fontsize=12)
    ax2.set_title("Detections Retained vs. Confidence Threshold", fontsize=14)
    ax2.legend(fontsize=11)

    plot2_path = os.path.join(OUT_DIR, "detections_vs_threshold.png")
    fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved to {plot2_path}")

    # Save stats to JSON
    stats = {
        "total_detections": len(scores),
        "directed_camera_count": int(len(scores_c0)),
        "dome_camera_count": int(len(scores_c1)),
        "mean_score": round(float(scores.mean()), 4),
        "median_score": round(float(np.median(scores)), 4),
        "std_score": round(float(scores.std()), 4),
        "min_score": round(float(scores.min()), 4),
        "max_score": round(float(scores.max()), 4),
        "images_sampled": len(set(d["image"] for d in all_detections)),
    }
    stats_path = os.path.join(OUT_DIR, "score_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(key,
         sec=None,
         n_sample=500,
         ckpt_path="detection/model/best.ckpt",
         device="cpu",
         seed=42,
         skip_download=False):
    """
    Run threshold analysis on a sample of Philly Street View images.

    Args:
        key:            Google API key (required)
        sec:            HMAC signing secret (optional)
        n_sample:       Number of images to sample (default: 500)
        ckpt_path:      Path to model checkpoint
        device:         'cpu' or 'cuda'
        seed:           Random seed for reproducibility
        skip_download:  If True, skip sampling + download, run detection on
                        whatever images are already in the output folder
    """
    print("=" * 60)
    print("  THRESHOLD ANALYSIS")
    print(f"  Sample size : {n_sample}")
    print(f"  Output      : {OUT_DIR}/")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not skip_download:
        sample_df = sample_metadata(n_sample=n_sample, seed=seed)
        sample_csv = os.path.join(OUT_DIR, "sample_meta.csv")
        sample_df.to_csv(sample_csv, index=False)
        download_images(sample_df, api_key=key, sec=sec)
    else:
        print("\nSkipping download (--skip_download=True)")

    all_detections = run_detection(ckpt_path=ckpt_path, device=device)
    analyze(all_detections)


if __name__ == "__main__":
    fire.Fire(main)

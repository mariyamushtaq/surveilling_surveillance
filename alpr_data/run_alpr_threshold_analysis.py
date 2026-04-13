"""
ALPR threshold analysis: download ALL images for known ALPR locations,
run Faster R-CNN detection with threshold=0, and analyze the confidence
score distribution. This is the ALPR equivalent of run_threshold_analysis.py.

Since there are only ~103 ALPR locations (~400 images at 4 headings each),
we run on all of them rather than sampling.

Usage:
    python -m alpr_data.run_alpr_threshold_analysis --key YOUR_API_KEY

Optional:
    --ckpt_path       Path to model checkpoint (default: detection/model/best.ckpt)
    --device          cpu or cuda (default: cpu)
    --skip_download   Skip metadata + download, run detection on existing images
"""

import os
import sys
import json
import glob
import time
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from util import constants as C

sys.path.insert(0, os.path.join(PROJECT_ROOT, "detection"))
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata

DET_META = Metadata()
DET_META.thing_classes = ["Directed Camera", "Dome Camera"]
DET_META.thing_colors = [[20, 200, 60], [11, 119, 32]]

ALPR_DIR = os.path.dirname(__file__)
LOCATIONS_CSV = os.path.join(ALPR_DIR, "data", "philly_alpr_combined.csv")
OUT_DIR = os.path.join(ALPR_DIR, "threshold_analysis")
IMAGE_DIR = os.path.join(OUT_DIR, "images")
DETECT_DIR = os.path.join(OUT_DIR, "detections")
SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Fetch panorama IDs for all ALPR locations
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_panorama_ids(api_key):
    print(f"\nStep 1: Loading ALPR locations from {LOCATIONS_CSV} …")
    locations = pd.read_csv(LOCATIONS_CSV)
    print(f"   {len(locations)} ALPR locations")

    print(f"\nStep 2: Fetching panorama IDs …")
    rows = []
    seen_panoids = set()

    for _, loc in tqdm(locations.iterrows(), total=len(locations), desc="   Metadata"):
        url = (f"{SV_METADATA_URL}"
               f"?location={loc['lat']},{loc['lon']}&key={api_key}"
               f"&source=outdoor")
        try:
            resp = requests.get(url, timeout=10).json()
        except Exception as e:
            print(f"   Warning: API error for ({loc['lat']}, {loc['lon']}): {e}")
            continue

        if resp.get("status") != "OK":
            continue

        panoid = resp["pano_id"]
        if panoid in seen_panoids:
            continue
        seen_panoids.add(panoid)

        pano_lat = resp["location"]["lat"]
        pano_lon = resp["location"]["lng"]
        capture_date = resp.get("date", "")

        for heading in [0, 90, 180, 270]:
            rows.append({
                "panoid": panoid,
                "heading": heading,
                "lat": pano_lat,
                "lon": pano_lon,
                "lat_alpr": loc["lat"],
                "lon_alpr": loc["lon"],
                "source": loc.get("source", ""),
                "capture_date": capture_date,
                "image_id": f"{panoid}_{heading}",
            })
        time.sleep(0.01)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_csv = os.path.join(OUT_DIR, "alpr_meta.csv")
    df.to_csv(meta_csv, index=False)
    print(f"   {len(df)} rows ({len(seen_panoids)} unique panoramas) saved to {meta_csv}")
    return df


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


def download_images(meta_df, api_key, sec=None):
    print(f"\nStep 3: Downloading {len(meta_df)} images …")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    ok, skipped, errors = 0, 0, []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="   Download"):
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
# STEP 3 — Run detection with threshold=0
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
    print(f"\nStep 4: Running detection (threshold=0, capturing all scores) …")
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

        keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes, scores, classes = boxes[keep_nms], scores[keep_nms], classes[keep_nms]

        # Save annotated image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        filtered = Instances(img_np.shape[:2])
        filtered.pred_boxes = Boxes(boxes)
        filtered.scores = scores
        filtered.pred_classes = classes

        v = Visualizer(img_np, metadata=DET_META, instance_mode=1)
        out = v.draw_instance_predictions(filtered)
        Image.fromarray(out.get_image()).save(
            os.path.join(DETECT_DIR, f"{basename}_det.jpg"))

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

    det_csv = os.path.join(OUT_DIR, "all_detections.csv")
    if all_detections:
        pd.DataFrame(all_detections).to_csv(det_csv, index=False)
    print(f"   {len(all_detections)} total detections across {len(images)} images")
    return all_detections


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Analyze and plot
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(all_detections, meta_csv=None):
    print(f"\nStep 5: Analyzing score distribution …")
    if not all_detections:
        print("   No detections to analyze.")
        return

    scores = np.array([d["score"] for d in all_detections])
    class_ids = np.array([d["class_id"] for d in all_detections])
    scores_c0 = scores[class_ids == 0]
    scores_c1 = scores[class_ids == 1]

    n_images_with_det = len(set(d["image"] for d in all_detections))
    n_total_images = len(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

    print(f"\n{'='*60}")
    print(f"  ALPR CONFIDENCE SCORE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total images      : {n_total_images}")
    print(f"  Images w/ dets    : {n_images_with_det} ({n_images_with_det/n_total_images*100:.1f}%)")
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

    print(f"\n  Detections retained at different thresholds:")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = int((scores >= t).sum())
        print(f"    threshold={t:.1f}  ->  {n:5d} detections ({n/len(scores)*100:5.1f}%)")
    print(f"{'='*60}")

    # ── Plot 1: Combined + per-class histogram ────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(scores, bins=50, range=(0, 1), color="#4C72B0", edgecolor="white",
                 alpha=0.85)
    axes[0].axvline(x=0.3, color="red", linestyle="--", linewidth=1.5,
                    label="Default threshold (0.3)")
    axes[0].set_xlabel("Confidence Score", fontsize=12)
    axes[0].set_ylabel("Number of Detections", fontsize=12)
    axes[0].set_title(f"ALPR Detection Confidence Score Distribution (n={len(scores)})",
                      fontsize=14)
    axes[0].legend(fontsize=11)

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
    axes[1].set_title("ALPR Per-Class Confidence Score Distribution", fontsize=14)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "confidence_score_distribution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {plot_path}")

    # ── Plot 2: Detections retained vs threshold ──────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0, 1, 200)
    retained = [int((scores >= t).sum()) for t in thresholds]
    ax2.plot(thresholds, retained, color="#4C72B0", linewidth=2)
    ax2.axvline(x=0.3, color="red", linestyle="--", linewidth=1.5,
                label="Default threshold (0.3)")
    ax2.set_xlabel("Confidence Threshold", fontsize=12)
    ax2.set_ylabel("Detections Retained", fontsize=12)
    ax2.set_title("ALPR: Detections Retained vs. Confidence Threshold", fontsize=14)
    ax2.legend(fontsize=11)

    plot2_path = os.path.join(OUT_DIR, "detections_vs_threshold.png")
    fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved to {plot2_path}")

    # ── Per-ALPR-location summary ─────────────────────────────────────────
    if meta_csv and os.path.exists(meta_csv):
        meta_df = pd.read_csv(meta_csv)
        det_df = pd.DataFrame(all_detections)
        det_df["panoid"] = det_df["image"].str.rsplit("_", n=1).str[0]
        det_per_pano = det_df.groupby("panoid").agg(
            n_detections=("score", "count"),
            max_score=("score", "max"),
            classes=("class", lambda x: ", ".join(sorted(set(x)))),
        ).reset_index()

        pano_meta = meta_df.drop_duplicates(subset=["panoid"])[
            ["panoid", "lat", "lon", "lat_alpr", "lon_alpr", "source"]
        ]
        merged = pano_meta.merge(det_per_pano, on="panoid", how="left")
        merged["n_detections"] = merged["n_detections"].fillna(0).astype(int)

        loc_csv = os.path.join(OUT_DIR, "alpr_location_detections.csv")
        merged.to_csv(loc_csv, index=False)

        n_with_det = (merged["n_detections"] > 0).sum()
        n_total_loc = len(merged)
        print(f"\n  Per-location summary: {n_with_det}/{n_total_loc} ALPR locations "
              f"({n_with_det/n_total_loc*100:.1f}%) have at least one detection")
        print(f"  Saved to {loc_csv}")

    # ── Visual inspection spreadsheet ─────────────────────────────────────
    det_df_all = pd.DataFrame(all_detections).sort_values("score", ascending=False)

    def score_band(s):
        if s >= 0.8: return "0.8-1.0 (high)"
        elif s >= 0.6: return "0.6-0.8"
        elif s >= 0.4: return "0.4-0.6"
        elif s >= 0.2: return "0.2-0.4"
        else: return "0.0-0.2 (low)"

    det_df_all["score_band"] = det_df_all["score"].apply(score_band)
    det_df_all["det_image"] = det_df_all["image"].str.replace(".jpg", "_det.jpg")
    det_df_all["is_correct"] = ""
    det_df_all["notes"] = ""

    inspect_csv = os.path.join(OUT_DIR, "visual_inspection.csv")
    det_df_all[["image", "det_image", "class", "class_id", "score",
                "score_band", "bbox", "is_correct", "notes"]].to_csv(inspect_csv, index=False)
    print(f"  Visual inspection spreadsheet saved to {inspect_csv}")

    # Save stats
    stats = {
        "total_images": n_total_images,
        "images_with_detections": n_images_with_det,
        "total_detections": len(scores),
        "directed_camera_count": int(len(scores_c0)),
        "dome_camera_count": int(len(scores_c1)),
        "mean_score": round(float(scores.mean()), 4),
        "median_score": round(float(np.median(scores)), 4),
        "std_score": round(float(scores.std()), 4),
        "min_score": round(float(scores.min()), 4),
        "max_score": round(float(scores.max()), 4),
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
         ckpt_path="detection/model/best.ckpt",
         device="cpu",
         skip_download=False):
    """
    Run threshold analysis on ALL ALPR location images.

    Args:
        key:            Google API key (required)
        sec:            HMAC signing secret (optional)
        ckpt_path:      Path to model checkpoint
        device:         'cpu' or 'cuda'
        skip_download:  Skip metadata + download, run detection on existing images
    """
    print("=" * 60)
    print("  ALPR THRESHOLD ANALYSIS")
    print(f"  Output: {OUT_DIR}/")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    meta_csv = os.path.join(OUT_DIR, "alpr_meta.csv")

    if not skip_download:
        meta_df = fetch_panorama_ids(api_key=key)
        download_images(meta_df, api_key=key, sec=sec)
    else:
        print("\nSkipping metadata + download (--skip_download=True)")

    all_detections = run_detection(ckpt_path=ckpt_path, device=device)
    analyze(all_detections, meta_csv=meta_csv)


if __name__ == "__main__":
    fire.Fire(main)

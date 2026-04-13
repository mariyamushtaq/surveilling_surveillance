"""
End-to-end Center City Philadelphia test:
  1. Sample road midpoints inside a tight bounding box
  2. Query Google Street View Metadata API for panorama IDs
  3. Download images (4 headings per pano for full coverage)
  4. Run the FasterRCNN detection model on every downloaded image
  5. Save annotated images, per-image JSON, and a summary CSV

Usage (from repository root):
    python scripts/run_centercity_test.py --key YOUR_API_KEY

Optional:
    --ckpt_path       Path to model checkpoint  (default: detection/model/best.ckpt)
    --conf_threshold   Min detection confidence  (default: 0.3)
    --device           cpu or cuda               (default: cpu)
"""

import os
import sys
import json

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "detection"))
import glob
import time
import hashlib
import hmac
import base64
import urllib.parse as urlparse

import fire
import requests
import osmnx as ox
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
# Root package `util/` (Street View constants), not detection/util/.
from util import constants as C

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata

# ── Detection metadata ────────────────────────────────────────────────────────
META = Metadata()
META.thing_classes = ["Directed Camera", "Dome Camera"]
META.thing_colors = [[20, 200, 60], [11, 119, 32]]

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR      = "data/centercity"
IMAGE_DIR     = os.path.join(DATA_DIR, "images")
OUTPUT_DIR    = os.path.join(DATA_DIR, "detections")
META_CSV      = os.path.join(DATA_DIR, "centercity_meta.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Sample road midpoints in Center City
# ═══════════════════════════════════════════════════════════════════════════════

def sample_centercity_points():
    """Return a DataFrame of road-midpoint lat/lon pairs inside Center City."""
    print("\n📍 Step 1: Sampling road midpoints in Center City …")
    G = ox.graph_from_bbox(
        north=39.9580,
        south=39.9420,
        east=-75.1380,
        west=-75.1780,
        network_type="drive",
    )
    edges = ox.graph_to_gdfs(G, nodes=False)

    points = []
    for _, row in edges.iterrows():
        pt = row.geometry.interpolate(0.5, normalized=True)
        points.append({"lat": pt.y, "lon": pt.x})

    df = pd.DataFrame(points)
    print(f"   → {len(df)} road midpoints")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Query GSV Metadata API for panorama IDs
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_panorama_ids(points_df, api_key):
    """Query the Metadata API and return a CSV-ready DataFrame (4 headings/pano)."""
    print("\n🔎 Step 2: Fetching panorama IDs …")
    rows = []
    seen = set()

    for _, pt in tqdm(points_df.iterrows(), total=len(points_df), desc="   Metadata"):
        url = (f"https://maps.googleapis.com/maps/api/streetview/metadata"
               f"?location={pt['lat']},{pt['lon']}&key={api_key}")
        resp = requests.get(url, timeout=10).json()
        if resp.get("status") == "OK":
            panoid = resp["pano_id"]
            if panoid in seen:
                continue
            seen.add(panoid)
            for heading in [0, 90, 180, 270]:
                rows.append({
                    "annotations": "",
                    "split": "deploy",
                    "image_id": f"{panoid}_{heading}",
                    "panoid": panoid,
                    "heading": heading,
                    "gsv_image_path": f"./{IMAGE_DIR}/{panoid}_{heading}.jpg",
                    "downloaded": False,
                })
        time.sleep(0.01)  # gentle rate limit

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(META_CSV, index=False)
    print(f"   → {len(df)} rows ({len(seen)} unique panoramas) saved to {META_CSV}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Download Street View images
# ═══════════════════════════════════════════════════════════════════════════════

def _build_url(panoid, heading, key, sec=None):
    """Build a GSV Static API URL, optionally HMAC-signed."""
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


def download_images(meta_df, api_key, sec=None, n_images=None):
    """Download images listed in meta_df.  If n_images is set, only download that many."""
    if n_images is not None:
        meta_df = meta_df.head(n_images)
    print(f"\n📥 Step 3: Downloading {len(meta_df)} images …")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    ok, skipped, errors = 0, 0, []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="   Download"):
        panoid  = row["panoid"]
        heading = row["heading"]
        filename = f"{panoid}_{heading}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)

        # Skip if already downloaded
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

    print(f"   → Downloaded: {ok}  |  Skipped (cached): {skipped}  |  Failed: {len(errors)}")
    if errors:
        for panoid, heading, status, detail in errors[:10]:
            print(f"      {panoid}_{heading}: status={status}, detail={detail}")
    return ok + skipped


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run detection
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device="cpu"):
    """Load the FasterRCNN model from a PyTorch Lightning checkpoint."""
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
    """Load an image and return a detectron2-format input dict."""
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    image_tensor = torch.as_tensor(
        img[:, :, ::-1].copy().astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor, "height": h, "width": w,
            "file_name": image_path}


def run_detection(ckpt_path, conf_threshold=0.3, iou_threshold=0.5, device="cpu"):
    """Run detection on all downloaded Center City images."""
    print(f"\n🔍 Step 4: Running detection (threshold={conf_threshold}) …")
    model = load_model(ckpt_path, device=device)
    print(f"   ✅ Model loaded from {ckpt_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not images:
        print(f"   ❌ No .jpg files found in {IMAGE_DIR}")
        return

    print(f"   Processing {len(images)} images …")
    all_detections = []
    total_cameras = 0

    for img_path in tqdm(images, desc="   Detect"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        inp = prepare_image(img_path)

        with torch.no_grad():
            preds = model([inp])[0]

        instances = preds["instances"].to("cpu")
        boxes   = instances.pred_boxes.tensor
        scores  = instances.scores
        classes = instances.pred_classes

        # Confidence filter
        keep = scores > conf_threshold
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        if len(scores) == 0:
            continue

        # NMS
        keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes, scores, classes = boxes[keep_nms], scores[keep_nms], classes[keep_nms]

        total_cameras += len(scores)

        # Save annotated image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        filtered = Instances(img_np.shape[:2])
        filtered.pred_boxes = Boxes(boxes)
        filtered.scores = scores
        filtered.pred_classes = classes

        v = Visualizer(img_np, metadata=META, instance_mode=1)
        out = v.draw_instance_predictions(filtered)
        Image.fromarray(out.get_image()).save(
            os.path.join(OUTPUT_DIR, f"{basename}_det.jpg"))

        # Collect per-image detections
        for j in range(len(scores)):
            det = {
                "image": os.path.basename(img_path),
                "class": META.thing_classes[int(classes[j])],
                "class_id": int(classes[j]),
                "score": round(float(scores[j]), 4),
                "bbox": [int(x) for x in boxes[j].tolist()],
            }
            all_detections.append(det)

        # Per-image JSON
        img_dets = [d for d in all_detections if d["image"] == os.path.basename(img_path)]
        with open(os.path.join(OUTPUT_DIR, f"{basename}_det.json"), "w") as f:
            json.dump({"image": os.path.basename(img_path),
                       "detections": img_dets}, f, indent=2)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = os.path.join(DATA_DIR, "detection_summary.csv")
    if all_detections:
        summary_df = pd.DataFrame(all_detections)
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df = pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"✅ RESULTS — Center City Detection Test")
    print(f"{'='*60}")
    print(f"   Images processed : {len(images)}")
    print(f"   Total cameras    : {total_cameras}")
    if total_cameras > 0:
        n_directed = sum(1 for d in all_detections if d["class_id"] == 0)
        n_dome     = sum(1 for d in all_detections if d["class_id"] == 1)
        print(f"     Directed       : {n_directed}")
        print(f"     Dome           : {n_dome}")
        print(f"   Avg score        : {np.mean([d['score'] for d in all_detections]):.3f}")
    print(f"   Annotated images : {OUTPUT_DIR}/")
    print(f"   Summary CSV      : {summary_path}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — run all steps
# ═══════════════════════════════════════════════════════════════════════════════

def main(key,
         sec=None,
         ckpt_path="detection/model/best.ckpt",
         conf_threshold=0.3,
         device="cpu",
         n_images=20,
         skip_download=False):
    """
    Run the full Center City pipeline.

    Args:
        key:             Google API key (required)
        sec:             HMAC signing secret (optional)
        ckpt_path:       Path to model checkpoint
        conf_threshold:  Minimum detection confidence
        device:          'cpu' or 'cuda'
        n_images:        Number of images to download (default: 20)
        skip_download:   If True, skip steps 1-3 and only run detection
                         (useful for re-running detection with different thresholds)
    """
    if not skip_download:
        # Step 1 — sample road points
        points = sample_centercity_points()

        # Step 2 — get panorama IDs
        meta_df = fetch_panorama_ids(points, api_key=key)

        # Step 3 — download images (only n_images)
        download_images(meta_df, api_key=key, sec=sec, n_images=n_images)
    else:
        print("⏩ Skipping metadata + download (--skip_download=True)")

    # Step 4 — run detection
    run_detection(ckpt_path, conf_threshold=conf_threshold, device=device)


if __name__ == "__main__":
    fire.Fire(main)

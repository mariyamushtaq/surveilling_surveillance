"""
Full Philadelphia pipeline: download all images from philly_meta.csv,
run detection at conf_threshold=0.4, and save positive detections with lat/lon
for mapping.

Usage (from repository root):
    python scripts/run_philly_full_pipeline.py --key YOUR_API_KEY

Detection only (images already under data/full_run/images), resume after interrupt:
    python scripts/run_philly_full_pipeline.py --skip_download=True --resume=True

Optional:
    --ckpt_path       Model checkpoint (default: detection/model/best.ckpt)
    --device          cpu or cuda (default: cpu)
    --skip_download   Skip download, run detection on existing images
    --image_dir       Where images are stored (default: data/full_run/images)
    --output_dir      Where to save detection results (default: data/full_run)
    --resume          Skip images that already have a *_det.json; write empty JSON
                      for no-detection images so reruns are complete (default: False)
    --key             API key (required unless --skip_download)
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

from util import constants as C

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import Instances, Boxes
from detectron2.data.catalog import Metadata

DET_META = Metadata()
DET_META.thing_classes = ["Directed Camera", "Dome Camera"]
DET_META.thing_colors = [[20, 200, 60], [11, 119, 32]]

META_CSV = "data/input_metadata/philly_meta.csv"
FULL_RUN_DIR = "data/full_run"
IMAGE_DIR = os.path.join(FULL_RUN_DIR, "images")
OUT_DIR = FULL_RUN_DIR
DETECT_DIR = os.path.join(OUT_DIR, "detections")


# ═══════════════════════════════════════════════════════════════════════════════
# Download
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


def download_images(meta_df, api_key, sec=None, image_dir=None):
    image_dir = image_dir or IMAGE_DIR
    print(f"\nStep 1: Downloading {len(meta_df)} images to {image_dir}/ …")
    os.makedirs(image_dir, exist_ok=True)

    ok, skipped, errors = 0, 0, []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="   Download"):
        panoid = row["panoid"]
        heading = int(row["heading"])
        filename = f"{panoid}_{heading}.jpg"
        filepath = os.path.join(image_dir, filename)

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
# Detection
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


def _positive_detections_from_json_dir(meta_csv, detect_dir):
    """Rebuild positive_detections rows from all *_det.json under detect_dir."""
    meta_df = pd.read_csv(meta_csv)
    meta_df["basename"] = meta_df.apply(
        lambda r: f"{r['panoid']}_{int(r['heading'])}.jpg", axis=1)
    meta_lookup = meta_df.set_index("basename").to_dict("index")

    rows = []
    for json_path in sorted(glob.glob(os.path.join(detect_dir, "*_det.json"))):
        with open(json_path) as f:
            payload = json.load(f)
        basename = payload.get("image", "")
        row_meta = meta_lookup.get(basename, {})
        lat = row_meta.get("lat", np.nan)
        lon = row_meta.get("lon", np.nan)
        panoid = row_meta.get("panoid", "")
        heading = row_meta.get("heading", "")
        for d in payload.get("detections") or []:
            rows.append({
                "image": basename,
                "panoid": panoid,
                "heading": heading,
                "lat": lat,
                "lon": lon,
                "class": d["class"],
                "class_id": d["class_id"],
                "score": d["score"],
                "bbox": d["bbox"],
            })
    return pd.DataFrame(rows)


def run_detection(ckpt_path,
                  conf_threshold=0.4,
                  device="cpu",
                  image_dir=None,
                  output_dir=None,
                  meta_csv=None,
                  resume=False):
    """Run detection at conf_threshold, save JSONs and aggregate positive detections."""
    image_dir = image_dir or IMAGE_DIR
    output_dir = output_dir or OUT_DIR
    meta_csv = meta_csv or META_CSV
    detect_dir = os.path.join(output_dir, "detections")

    print(f"\nStep 2: Running detection (threshold={conf_threshold}) …")
    if resume:
        print("   Resume: skipping images that already have *_det.json")
    model = load_model(ckpt_path, device=device)

    os.makedirs(detect_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        print(f"   No .jpg files in {image_dir}")
        return [], pd.DataFrame()

    print(f"   Processing {len(images)} images …")
    all_detections = []
    iou_threshold = 0.5

    meta_df = pd.read_csv(meta_csv)
    meta_df["basename"] = meta_df.apply(
        lambda r: f"{r['panoid']}_{int(r['heading'])}.jpg", axis=1)
    meta_lookup = meta_df.set_index("basename").to_dict("index")

    for img_path in tqdm(images, desc="   Detect"):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        out_json = os.path.join(detect_dir, f"{stem}_det.json")
        if resume and os.path.isfile(out_json):
            continue

        inp = prepare_image(img_path)

        with torch.no_grad():
            preds = model([inp])[0]

        instances = preds["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        classes = instances.pred_classes

        keep = scores >= conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        if len(scores) == 0:
            if resume:
                with open(out_json, "w") as f:
                    json.dump({"image": basename, "detections": []}, f, indent=2)
            continue

        keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        classes = classes[keep_nms]

        row_meta = meta_lookup.get(basename, {})
        lat = row_meta.get("lat", np.nan)
        lon = row_meta.get("lon", np.nan)

        img_dets = []
        for j in range(len(scores)):
            det = {
                "image": basename,
                "panoid": row_meta.get("panoid", ""),
                "heading": row_meta.get("heading", ""),
                "lat": lat,
                "lon": lon,
                "class": DET_META.thing_classes[int(classes[j])],
                "class_id": int(classes[j]),
                "score": round(float(scores[j]), 4),
                "bbox": [int(x) for x in boxes[j].tolist()],
            }
            all_detections.append(det)
            img_dets.append({k: v for k, v in det.items() if k != "lat" and k != "lon"})

        with open(out_json, "w") as f:
            json.dump({"image": basename, "detections": img_dets}, f, indent=2)

    det_csv = os.path.join(output_dir, "positive_detections.csv")
    det_df = _positive_detections_from_json_dir(meta_csv, detect_dir)
    if len(det_df) > 0:
        det_df.to_csv(det_csv, index=False)
        print(f"\n   {len(det_df)} positive detections saved to {det_csv}")
    else:
        print(f"\n   No positive detections at threshold {conf_threshold}")

    return all_detections, det_df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(key=None,
         sec=None,
         ckpt_path="detection/model/best.ckpt",
         device="cpu",
         conf_threshold=0.4,
         skip_download=False,
         image_dir=None,
         output_dir=None,
         resume=False):
    """
    Full Philly pipeline: download → detect at 0.4 → save positive_detections.csv
    """
    if not skip_download and not key:
        print("Error: provide --key or use --skip_download=True")
        sys.exit(1)

    image_dir = image_dir or IMAGE_DIR
    output_dir = output_dir or OUT_DIR

    print("=" * 60)
    print("  PHILLY FULL PIPELINE")
    print(f"  Threshold: {conf_threshold}")
    print(f"  Images: {image_dir}/")
    print(f"  Output: {output_dir}/")
    print(f"  Resume: {resume}")
    print("=" * 60)

    meta_df = pd.read_csv(META_CSV)
    print(f"\n  Loaded {len(meta_df)} rows from {META_CSV}")

    if not skip_download:
        download_images(meta_df, api_key=key, sec=sec, image_dir=image_dir)
    else:
        print("\n  Skipping download (--skip_download=True)")

    run_detection(
        ckpt_path=ckpt_path,
        conf_threshold=conf_threshold,
        device=device,
        image_dir=image_dir,
        output_dir=output_dir,
        resume=resume,
    )


if __name__ == "__main__":
    fire.Fire(main)

"""
End-to-end ALPR detection pipeline:
  1. Load ALPR camera locations (from fetch_alpr_data.py output)
  2. Query Google Street View Metadata API for panorama IDs near each location
  3. Download Street View images (4 headings per panorama for full coverage)
  4. Run the FasterRCNN detection model on every downloaded image
  5. Save annotated images, per-image JSON, and a summary CSV

Usage:
    # Test run (10 images, output in alpr_data/test_run/)
    python -m alpr_data.run_alpr_pipeline test --key YOUR_API_KEY

    # Full run (all images, output in alpr_data/data/)
    python -m alpr_data.run_alpr_pipeline run --key YOUR_API_KEY

Optional flags (for both test/run):
    --ckpt_path        Path to model checkpoint  (default: detection/model/best.ckpt)
    --conf_threshold   Min detection confidence   (default: 0.3)
    --device           cpu or cuda                (default: cpu)
    --n_images         Max images to download     (default: 10 for test, None for run)
    --skip_download    Only run detection on existing images
    --locations_csv    Path to ALPR locations CSV
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

# ── Project imports ───────────────────────────────────────────────────────────
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

# ── Detection metadata ────────────────────────────────────────────────────────
DET_META = Metadata()
DET_META.thing_classes = ["Directed Camera", "Dome Camera"]
DET_META.thing_colors = [[20, 200, 60], [11, 119, 32]]

# ── Default paths ─────────────────────────────────────────────────────────────
ALPR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ALPR_DIR, "data")
TEST_DIR = os.path.join(ALPR_DIR, "test_run")
LOCATIONS_CSV = os.path.join(DATA_DIR, "philly_alpr_combined.csv")

SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load ALPR locations
# ═══════════════════════════════════════════════════════════════════════════════

def load_alpr_locations(locations_csv: str = LOCATIONS_CSV) -> pd.DataFrame:
    """Load ALPR camera locations from the combined CSV."""
    print(f"\nStep 1: Loading ALPR locations from {locations_csv} …")
    if not os.path.exists(locations_csv):
        raise FileNotFoundError(
            f"{locations_csv} not found. Run fetch_alpr_data.py first:\n"
            f"  python -m alpr_data.fetch_alpr_data"
        )
    df = pd.read_csv(locations_csv)
    print(f"   -> {len(df)} ALPR camera locations loaded")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Query GSV Metadata API for panorama IDs
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_panorama_ids(locations_df: pd.DataFrame,
                       api_key: str,
                       out_dir: str = DATA_DIR,
                       image_dir: str = None) -> pd.DataFrame:
    """
    For each ALPR location, query the GSV Metadata API to get the nearest
    panorama ID and generate rows for 4 headings (0, 90, 180, 270).
    """
    print(f"\nStep 2: Fetching panorama IDs for {len(locations_df)} locations …")
    if image_dir is None:
        image_dir = os.path.join(out_dir, "images")
    meta_csv = os.path.join(out_dir, "alpr_meta.csv")

    rows = []
    seen_panoids = set()

    for idx, loc in tqdm(locations_df.iterrows(),
                         total=len(locations_df),
                         desc="   Metadata"):
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
                "gsv_image_path": os.path.join(image_dir, f"{panoid}_{heading}.jpg"),
                "downloaded": False,
            })
        time.sleep(0.01)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(meta_csv, index=False)
    print(f"   -> {len(df)} rows ({len(seen_panoids)} unique panoramas) saved to {meta_csv}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Download Street View images
# ═══════════════════════════════════════════════════════════════════════════════

def _build_url(panoid: str, heading: int, key: str, sec: str = None) -> str:
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


def download_images(meta_df: pd.DataFrame,
                    api_key: str,
                    image_dir: str,
                    sec: str = None,
                    n_images: int = None) -> int:
    """Download Street View images for ALPR locations."""
    if n_images is not None:
        meta_df = meta_df.head(n_images)
    print(f"\nStep 3: Downloading {len(meta_df)} images …")
    os.makedirs(image_dir, exist_ok=True)

    ok, skipped, errors = 0, 0, []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="   Download"):
        panoid = row["panoid"]
        heading = row["heading"]
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

    print(f"   -> Downloaded: {ok}  |  Skipped (cached): {skipped}  |  Failed: {len(errors)}")
    if errors:
        for panoid, heading, status, detail in errors[:10]:
            print(f"      {panoid}_{heading}: status={status}, detail={detail}")
    return ok + skipped


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run detection
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: str = "cpu"):
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


def prepare_image(image_path: str) -> dict:
    """Load an image and return a detectron2-format input dict."""
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    image_tensor = torch.as_tensor(
        img[:, :, ::-1].copy().astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor, "height": h, "width": w,
            "file_name": image_path}


def run_detection(ckpt_path: str,
                  image_dir: str,
                  output_dir: str,
                  out_dir: str,
                  meta_csv: str,
                  conf_threshold: float = 0.3,
                  iou_threshold: float = 0.5,
                  device: str = "cpu"):
    """Run detection on all downloaded ALPR Street View images."""
    print(f"\nStep 4: Running detection (threshold={conf_threshold}) …")
    model = load_model(ckpt_path, device=device)
    print(f"   Model loaded from {ckpt_path}")

    os.makedirs(output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        print(f"   No .jpg files found in {image_dir}")
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
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        classes = instances.pred_classes

        keep = scores > conf_threshold
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        if len(scores) == 0:
            continue

        keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes, scores, classes = boxes[keep_nms], scores[keep_nms], classes[keep_nms]

        total_cameras += len(scores)

        img_np = np.array(Image.open(img_path).convert("RGB"))
        filtered = Instances(img_np.shape[:2])
        filtered.pred_boxes = Boxes(boxes)
        filtered.scores = scores
        filtered.pred_classes = classes

        v = Visualizer(img_np, metadata=DET_META, instance_mode=1)
        out = v.draw_instance_predictions(filtered)
        Image.fromarray(out.get_image()).save(
            os.path.join(output_dir, f"{basename}_det.jpg"))

        for j in range(len(scores)):
            det = {
                "image": os.path.basename(img_path),
                "class": DET_META.thing_classes[int(classes[j])],
                "class_id": int(classes[j]),
                "score": round(float(scores[j]), 4),
                "bbox": [int(x) for x in boxes[j].tolist()],
            }
            all_detections.append(det)

        img_dets = [d for d in all_detections
                    if d["image"] == os.path.basename(img_path)]
        with open(os.path.join(output_dir, f"{basename}_det.json"), "w") as f:
            json.dump({"image": os.path.basename(img_path),
                       "detections": img_dets}, f, indent=2)

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "detection_summary.csv")
    if all_detections:
        summary_df = pd.DataFrame(all_detections)
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df = pd.DataFrame()

    # ── Merge detections back to ALPR locations ───────────────────────────
    _generate_alpr_detection_summary(all_detections, meta_csv, out_dir)

    print(f"\n{'='*60}")
    print(f"  ALPR Detection Results")
    print(f"{'='*60}")
    print(f"   Images processed : {len(images)}")
    print(f"   Total cameras    : {total_cameras}")
    if total_cameras > 0:
        n_directed = sum(1 for d in all_detections if d["class_id"] == 0)
        n_dome = sum(1 for d in all_detections if d["class_id"] == 1)
        print(f"     Directed       : {n_directed}")
        print(f"     Dome           : {n_dome}")
        print(f"   Avg score        : {np.mean([d['score'] for d in all_detections]):.3f}")
    print(f"   Annotated images : {output_dir}/")
    print(f"   Summary CSV      : {summary_path}")
    print(f"{'='*60}")


def _generate_alpr_detection_summary(all_detections: list,
                                     meta_csv: str,
                                     out_dir: str):
    """
    Cross-reference detections with the ALPR metadata CSV to produce a
    per-location summary showing how many cameras were detected near
    each known ALPR location.
    """
    if not os.path.exists(meta_csv) or not all_detections:
        return

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

    out_path = os.path.join(out_dir, "alpr_location_detections.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n   Per-location summary saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline runner (shared by test & run)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(key: str,
                  out_dir: str,
                  sec: str = None,
                  ckpt_path: str = "detection/model/best.ckpt",
                  conf_threshold: float = 0.3,
                  device: str = "cpu",
                  n_images: int = None,
                  skip_download: bool = False,
                  locations_csv: str = LOCATIONS_CSV):
    """Core pipeline logic shared by test() and run()."""
    image_dir = os.path.join(out_dir, "images")
    output_dir = os.path.join(out_dir, "detections")
    meta_csv = os.path.join(out_dir, "alpr_meta.csv")

    if not skip_download:
        locations = load_alpr_locations(locations_csv)
        meta_df = fetch_panorama_ids(locations, api_key=key,
                                     out_dir=out_dir, image_dir=image_dir)
        download_images(meta_df, api_key=key, image_dir=image_dir,
                        sec=sec, n_images=n_images)
    else:
        print("Skipping metadata + download (--skip_download=True)")

    run_detection(ckpt_path, image_dir=image_dir, output_dir=output_dir,
                  out_dir=out_dir, meta_csv=meta_csv,
                  conf_threshold=conf_threshold, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI commands
# ═══════════════════════════════════════════════════════════════════════════════

def test(key: str,
         sec: str = None,
         ckpt_path: str = "detection/model/best.ckpt",
         conf_threshold: float = 0.3,
         device: str = "cpu",
         n_images: int = 10,
         skip_download: bool = False,
         locations_csv: str = LOCATIONS_CSV):
    """
    Test run: download a small batch and run detection.
    Output goes to alpr_data/test_run/.

    Args:
        key:             Google API key (required)
        sec:             HMAC signing secret (optional)
        ckpt_path:       Path to model checkpoint
        conf_threshold:  Minimum detection confidence
        device:          'cpu' or 'cuda'
        n_images:        Number of images to download (default: 10)
        skip_download:   If True, skip steps 1-3 and only run detection
        locations_csv:   Path to ALPR locations CSV
    """
    print("=" * 60)
    print("  ALPR Pipeline — TEST RUN")
    print(f"  Output directory: {TEST_DIR}")
    print(f"  Images to download: {n_images}")
    print("=" * 60)

    _run_pipeline(key=key, out_dir=TEST_DIR, sec=sec, ckpt_path=ckpt_path,
                  conf_threshold=conf_threshold, device=device,
                  n_images=n_images, skip_download=skip_download,
                  locations_csv=locations_csv)


def run(key: str,
        sec: str = None,
        ckpt_path: str = "detection/model/best.ckpt",
        conf_threshold: float = 0.3,
        device: str = "cpu",
        n_images: int = None,
        skip_download: bool = False,
        locations_csv: str = LOCATIONS_CSV):
    """
    Full run: download all images and run detection.
    Output goes to alpr_data/data/.

    Args:
        key:             Google API key (required)
        sec:             HMAC signing secret (optional)
        ckpt_path:       Path to model checkpoint
        conf_threshold:  Minimum detection confidence
        device:          'cpu' or 'cuda'
        n_images:        Max images to download (None = all)
        skip_download:   If True, skip steps 1-3 and only run detection
        locations_csv:   Path to ALPR locations CSV
    """
    print("=" * 60)
    print("  ALPR Pipeline — FULL RUN")
    print(f"  Output directory: {DATA_DIR}")
    print("=" * 60)

    _run_pipeline(key=key, out_dir=DATA_DIR, sec=sec, ckpt_path=ckpt_path,
                  conf_threshold=conf_threshold, device=device,
                  n_images=n_images, skip_download=skip_download,
                  locations_csv=locations_csv)


if __name__ == "__main__":
    fire.Fire({"test": test, "run": run})

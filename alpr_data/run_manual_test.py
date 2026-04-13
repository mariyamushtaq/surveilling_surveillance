"""
Run the detection model on images in alpr_data/data/manual_tests/.
Saves annotated images and per-image JSON to manual_tests/detections/.

Usage:
    python -m alpr_data.run_manual_test

Optional:
    --ckpt_path        Path to model checkpoint  (default: detection/model/best.ckpt)
    --conf_threshold   Min detection confidence   (default: 0.3)
    --device           cpu or cuda                (default: cpu)
"""

import os
import sys
import json
import glob

import fire
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
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

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "manual_tests")
OUTPUT_DIR = os.path.join(IMAGE_DIR, "detections")


def load_model(ckpt_path: str, device: str = "cpu"):
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
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    image_tensor = torch.as_tensor(
        img[:, :, ::-1].copy().astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor, "height": h, "width": w,
            "file_name": image_path}


def main(ckpt_path: str = "detection/model/best.ckpt",
         conf_threshold: float = 0.3,
         iou_threshold: float = 0.5,
         device: str = "cpu"):
    """Run detection on all images in manual_tests/."""
    print(f"\nRunning detection on {IMAGE_DIR}")
    model = load_model(ckpt_path, device=device)
    print(f"Model loaded from {ckpt_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    )
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Processing {len(images)} images …\n")
    all_detections = []
    total_cameras = 0

    for img_path in tqdm(images, desc="Detect"):
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

        if len(scores) > 0:
            keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
            boxes, scores, classes = boxes[keep_nms], scores[keep_nms], classes[keep_nms]

        total_cameras += len(scores)

        # Save annotated image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        filtered = Instances(img_np.shape[:2])
        filtered.pred_boxes = Boxes(boxes)
        filtered.scores = scores
        filtered.pred_classes = classes

        v = Visualizer(img_np, metadata=DET_META, instance_mode=1)
        out = v.draw_instance_predictions(filtered)
        out_path = os.path.join(OUTPUT_DIR, f"{basename}_det.jpg")
        Image.fromarray(out.get_image()).save(out_path)

        # Per-image JSON
        img_dets = []
        for j in range(len(scores)):
            det = {
                "image": os.path.basename(img_path),
                "class": DET_META.thing_classes[int(classes[j])],
                "class_id": int(classes[j]),
                "score": round(float(scores[j]), 4),
                "bbox": [int(x) for x in boxes[j].tolist()],
            }
            img_dets.append(det)
            all_detections.append(det)

        with open(os.path.join(OUTPUT_DIR, f"{basename}_det.json"), "w") as f:
            json.dump({"image": os.path.basename(img_path),
                       "detections": img_dets}, f, indent=2)

        n = len(img_dets)
        status = f"{n} detection{'s' if n != 1 else ''}" if n > 0 else "no detections"
        print(f"  {os.path.basename(img_path):40s} -> {status}")

    print(f"\n{'='*60}")
    print(f"  Manual Test Results")
    print(f"{'='*60}")
    print(f"  Images processed : {len(images)}")
    print(f"  Total cameras    : {total_cameras}")
    if total_cameras > 0:
        n_directed = sum(1 for d in all_detections if d["class_id"] == 0)
        n_dome = sum(1 for d in all_detections if d["class_id"] == 1)
        print(f"    Directed       : {n_directed}")
        print(f"    Dome           : {n_dome}")
        print(f"  Avg score        : {np.mean([d['score'] for d in all_detections]):.3f}")
    print(f"  Output           : {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    fire.Fire(main)

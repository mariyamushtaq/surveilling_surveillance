"""
Run the pretrained FasterRCNN camera detection model on test images.

Usage (from repository root):
    python scripts/run_detection_test.py --ckpt_path detection/model/model.ckpt

Optional:
    --image_dir     Directory with test images (default: data/rawdata/image_test)
    --output_dir    Where to save annotated images + JSON (default: data/detection_output)
    --conf_threshold  Minimum confidence score (default: 0.3)
    --device        cpu or cuda (default: cpu)
"""

import os
import sys
import json

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "detection"))
import glob
import fire
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper


# ── Metadata for visualization ───────────────────────────────────────────────
META = Metadata()
META.thing_classes = ["Directed Camera", "Dome Camera"]
META.thing_colors = [[20, 200, 60], [11, 119, 32]]


# ── Load model from Lightning checkpoint ─────────────────────────────────────

def load_model(ckpt_path, device="cpu"):
    """Load the FasterRCNN model from a PyTorch Lightning checkpoint."""

    # Build detectron2 FasterRCNN with 2 classes
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.RETINANET.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)

    # Extract detectron2 state dict from the Lightning checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Lightning wraps keys as "model.model.xxx" — strip the prefix
    new_state = {}
    for k, v in state_dict.items():
        # model.model.backbone.xxx → backbone.xxx
        if k.startswith("model.model."):
            new_state[k[len("model.model."):]] = v
        elif k.startswith("model."):
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    print(f"✅ Model loaded from {ckpt_path} (device={device})")
    return model


# ── Prepare a single image for detectron2 ────────────────────────────────────

def prepare_image(image_path):
    """Load an image and return a detectron2-format input dict."""
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    # Detectron2 expects BGR float32 tensor
    image_tensor = torch.as_tensor(img[:, :, ::-1].copy().astype("float32").transpose(2, 0, 1))
    return {"image": image_tensor, "height": h, "width": w,
            "file_name": image_path}


# ── Run detection ────────────────────────────────────────────────────────────

def run_detection(ckpt_path,
                  image_dir="data/rawdata/image_test",
                  output_dir="data/detection_output",
                  conf_threshold=0.3,
                  iou_threshold=0.5,
                  device="cpu"):
    """Run detection on all JPGs in image_dir, save annotated images + JSON."""

    model = load_model(ckpt_path, device=device)
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        print(f"❌ No .jpg files found in {image_dir}")
        return

    print(f"Running detection on {len(images)} images …")
    total_detections = 0

    for img_path in tqdm(images):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        inp = prepare_image(img_path)

        with torch.no_grad():
            preds = model([inp])[0]

        instances = preds["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        classes = instances.pred_classes

        # Filter by confidence
        keep = scores > conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        if len(scores) == 0:
            continue

        # NMS
        keep_nms = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        classes = classes[keep_nms]

        total_detections += len(scores)

        # Save annotated image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        filtered = Instances(img_np.shape[:2])
        filtered.pred_boxes = Boxes(boxes)
        filtered.scores = scores
        filtered.pred_classes = classes

        v = Visualizer(img_np, metadata=META, instance_mode=1)
        out = v.draw_instance_predictions(filtered)
        annotated = Image.fromarray(out.get_image())
        annotated.save(os.path.join(output_dir, f"{basename}_det.jpg"))

        # Save JSON results
        detections = []
        for j in range(len(scores)):
            det = {
                "class": META.thing_classes[int(classes[j])],
                "class_id": int(classes[j]),
                "score": round(float(scores[j]), 4),
                "bbox": [int(x) for x in boxes[j].tolist()],
            }
            detections.append(det)

        with open(os.path.join(output_dir, f"{basename}_det.json"), "w") as f:
            json.dump({"image": os.path.basename(img_path),
                       "detections": detections}, f, indent=2)

    print(f"\n✅ Done! {total_detections} cameras detected across {len(images)} images")
    print(f"   Annotated images + JSON saved to: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    fire.Fire(run_detection)

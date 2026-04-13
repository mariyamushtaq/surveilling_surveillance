"""
Generate annotated detection images from existing all_detections.csv.
Draws bounding boxes + labels on source images and saves to detections folder.

Usage (from repository root):
    python scripts/generate_detection_images.py
"""

import os
import ast

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGE_DIR = os.path.join(_REPO_ROOT, "data", "threshold_analysis", "images")
DETECT_DIR = os.path.join(_REPO_ROOT, "data", "threshold_analysis", "detections")
CSV_PATH = os.path.join(_REPO_ROOT, "data", "threshold_analysis", "all_detections.csv")

CLASS_COLORS = {
    0: (20, 200, 60),
    1: (255, 165, 0),
}
CLASS_NAMES = ["Directed Camera", "Dome Camera"]


def draw_detections(img, detections):
    draw = ImageDraw.Draw(img)
    for det in detections:
        bbox = det["bbox"]
        score = det["score"]
        class_id = det["class_id"]
        color = CLASS_COLORS.get(class_id, (255, 0, 0))
        label = f"{CLASS_NAMES[class_id]} {score:.2f}"

        x1, y1, x2, y2 = bbox
        pad = 30
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(img.width, x2 + pad), min(img.height, y2 + pad)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.rectangle([x1p, y1p, x2p, y2p], outline=color, width=1)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - th - 4)
        draw.rectangle([x1, text_y, x1 + tw + 4, text_y + th + 4], fill=color)
        draw.text((x1 + 2, text_y + 2), label, fill="black", font=font)

    return img


def main():
    df = pd.read_csv(CSV_PATH)
    df["bbox"] = df["bbox"].apply(ast.literal_eval)

    grouped = df.groupby("image")
    print(f"Generating annotated images for {len(grouped)} images …")

    for image_name, group in grouped:
        img_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(img_path):
            print(f"  Skipping {image_name} — source not found")
            continue

        img = Image.open(img_path).convert("RGB")
        detections = group.to_dict("records")
        img = draw_detections(img, detections)

        basename = os.path.splitext(image_name)[0]
        out_path = os.path.join(DETECT_DIR, f"{basename}_det.jpg")
        img.save(out_path)

    print(f"Done — annotated images saved to {DETECT_DIR}/")


if __name__ == "__main__":
    main()

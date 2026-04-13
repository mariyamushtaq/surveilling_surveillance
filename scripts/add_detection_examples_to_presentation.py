"""
Add detection example slides to thesis_update_presentation.pptx.
Inserts examples of correct and incorrect detections from both
surveillance camera data and ALPR data.

Usage (from repository root):
    pip install python-pptx
    python scripts/add_detection_examples_to_presentation.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Surveillance camera data (Philly Street View)
SURV_CORRECT = os.path.join(
    PROJECT_ROOT, "data/threshold_analysis/detections/A9nN3L4n-JSRSrY1jY_73g_9_det.jpg"
)  # 0.99 Directed Camera
SURV_INCORRECT = os.path.join(
    PROJECT_ROOT, "data/threshold_analysis/detections/tdpgzS0VfkX1as4AsPDkhw_336_det.jpg"
)  # 0.18 Dome - low confidence, likely false positive

# ALPR data (Street View at ALPR locations)
ALPR_CORRECT = os.path.join(
    PROJECT_ROOT, "alpr_data/threshold_analysis/detections/Tqp-CZrQnh78bwOjSdbo0A_90_det.jpg"
)  # 0.92 Dome Camera
ALPR_INCORRECT = os.path.join(
    PROJECT_ROOT, "alpr_data/threshold_analysis/detections/Ouf-LBbquy9SUew6_qXSKg_270_det.jpg"
)  # 0.097 Directed - very low confidence false positive


def move_slide(prs, old_index, new_index):
    """Move a slide from old_index to new_index (0-based)."""
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    xml_slides.remove(slides[old_index])
    xml_slides.insert(new_index, slides[old_index])


def add_slide_with_images(prs, title, examples, slide_layout_idx=6):
    """Add a slide with a title and 2x2 grid of images with captions."""
    blank_layout = prs.slide_layouts[slide_layout_idx]  # Blank
    slide = prs.slides.add_slide(blank_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    # Image dimensions - 2x2 grid, each image ~4" wide
    img_w, img_h = Inches(4.2), Inches(2.8)
    left_col = Inches(0.5)
    right_col = Inches(5)
    top_row = Inches(1.1)

    positions = [
        (left_col, top_row, "Correct"),
        (right_col, top_row, "Incorrect"),
    ]

    for i, (img_path, caption) in enumerate(examples):
        if not os.path.exists(img_path):
            print(f"  Warning: {img_path} not found, skipping")
            continue
        x, y, label = positions[i]
        # Add label above image
        label_box = slide.shapes.add_textbox(x, y - 0.35, img_w, Inches(0.3))
        tf = label_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"{label} detection"
        p.font.size = Pt(14)
        p.font.bold = True
        # Add image
        slide.shapes.add_picture(img_path, x, y, width=img_w, height=img_h)
        # Caption below image
        cap_box = slide.shapes.add_textbox(x, y + img_h + 0.05, img_w, Inches(0.4))
        tf = cap_box.text_frame
        p = tf.paragraphs[0]
        p.text = caption
        p.font.size = Pt(10)


def main():
    prs_path = os.path.join(PROJECT_ROOT, "thesis_update_presentation.pptx")
    if not os.path.exists(prs_path):
        print(f"Presentation not found: {prs_path}")
        return

    prs = Presentation(prs_path)

    # Slide 1: Surveillance camera detection examples
    surv_examples = [
        (SURV_CORRECT, "Directed Camera, score 0.99 (Philly Street View)"),
        (SURV_INCORRECT, "Dome Camera, score 0.18 — likely false positive"),
    ]
    add_slide_with_images(
        prs,
        "Detection Examples: Surveillance Camera Data (Philly Street View)",
        surv_examples,
    )

    # Slide 2: ALPR detection examples
    alpr_examples = [
        (ALPR_CORRECT, "Dome Camera, score 0.92 (Street View at ALPR location)"),
        (ALPR_INCORRECT, "Directed Camera, score 0.10 — false positive"),
    ]
    add_slide_with_images(
        prs,
        "Detection Examples: ALPR Data (Street View at Known ALPR Locations)",
        alpr_examples,
    )

    # Move new slides to positions 7 and 8 (after Manual Labeling at 6)
    n_slides = len(prs.slides)
    move_slide(prs, n_slides - 2, 7)  # Surveillance -> position 7
    move_slide(prs, n_slides - 1, 8)  # ALPR -> position 8

    prs.save(prs_path)
    print(f"Updated presentation saved: {prs_path}")
    print("Added 2 slides with correct/incorrect detection examples from:")
    print("  - Surveillance camera data (Philly Street View)")
    print("  - ALPR data (Street View at ALPR locations)")


if __name__ == "__main__":
    main()

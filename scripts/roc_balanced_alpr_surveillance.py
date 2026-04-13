"""
Balanced ALPR + general-surveillance subset for ROC analysis.

1) build_subset — Sample n_per_source rows (default 60) from each of ALPR and
   surveillance visual_inspection CSVs, using only binary-labeled rows, then copy
   (data/threshold_analysis/visual_inspection.csv). Creates that surveillance CSV
   from all_detections.csv if missing. Copies raw images and *_det.jpg into a
   single output folder.

2) plot_roc — Reads labeled rows from the two source CSVs or a combined CSV,
   draws the same balanced sample as build_subset, and saves three ROC figures:
   combined, ALPR-only, and surveillance-only (each with AUC + Youden annotation).

Ground truth: use is_correct in each visual_inspection CSV. Truthy: 1, y, yes,
true, tp, correct (case-insensitive). Falsy: 0, n, no, false, fp, incorrect.
Empty rows are skipped.

Import from Apple Numbers (.numbers):
  python scripts/roc_balanced_alpr_surveillance.py sync_alpr_labels_from_numbers
  python scripts/roc_balanced_alpr_surveillance.py sync_surveillance_labels_from_numbers

Usage (from repository root):
  python scripts/roc_balanced_alpr_surveillance.py build_subset --n_per_source 60 --seed 42
  python scripts/roc_balanced_alpr_surveillance.py plot_roc --n_per_source 60 --seed 42
"""

from __future__ import annotations

import json
import os
import shutil

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALPR_INSPECTION_DEFAULT = os.path.join(PROJECT_ROOT, "alpr_data", "visual_inspection.csv")
ALPR_IMG_DIR = os.path.join(PROJECT_ROOT, "alpr_data", "threshold_analysis", "images")
ALPR_DET_DIR = os.path.join(PROJECT_ROOT, "alpr_data", "threshold_analysis", "detections")

SURV_ALL_DET = os.path.join(PROJECT_ROOT, "data", "threshold_analysis", "all_detections.csv")
SURV_INSPECTION_DEFAULT = os.path.join(PROJECT_ROOT, "data", "threshold_analysis", "visual_inspection.csv")
SURV_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "threshold_analysis", "images")
SURV_DET_DIR = os.path.join(PROJECT_ROOT, "data", "threshold_analysis", "detections")

OUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "roc_balanced_subset")
OUTPUT_FIGURES_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "outputs", "figures")

ALPR_NUMBERS_DEFAULT = os.path.join(PROJECT_ROOT, "alpr_data", "visual_inspection.numbers")
SURV_NUMBERS_DEFAULT = os.path.join(
    PROJECT_ROOT, "data", "threshold_analysis", "visual_inspection_thesis.numbers"
)


def _numbers_table_to_dataframe(numbers_path: str) -> pd.DataFrame:
    try:
        from numbers_parser import Document
    except ImportError as e:
        raise ImportError(
            "Reading .numbers files requires: pip install numbers-parser"
        ) from e
    doc = Document(numbers_path)
    if not doc.sheets or not doc.sheets[0].tables:
        raise ValueError(f"No tables found in {numbers_path}")
    table = doc.sheets[0].tables[0]
    headers = [table.cell(0, c).value for c in range(table.num_cols)]
    rows = []
    for r in range(1, table.num_rows):
        rows.append([table.cell(r, c).value for c in range(table.num_cols)])
    return pd.DataFrame(rows, columns=headers)


def _coerce_label_from_numbers(val) -> float:
    """
    Map Apple Numbers inspection text to 1 / 0 / NaN for binary ROC.
    ALPR-style: correct / incorrect. Surveillance thesis: yes / no / probably not.
    Ambiguous phrases -> NaN (row kept; skipped by plot_roc until resolved).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().lower()
    if s == "":
        return np.nan
    positive = (
        "correct",
        "1",
        "true",
        "yes",
        "y",
        "tp",
        "pos",
        "positive",
    )
    negative = (
        "incorrect",
        "0",
        "false",
        "no",
        "n",
        "fp",
        "neg",
        "negative",
        "probably not",
    )
    ambiguous = (
        "yes and no",
        "no (maybe?)",
        "one yes one no",
        "maybe",
        "yes maybe",
    )
    if s in positive:
        return 1.0
    if s in negative:
        return 0.0
    if s in ambiguous:
        return np.nan
    return np.nan


def _merge_ground_truth_from_numbers_df(
    target_csv: str,
    gt_df: pd.DataFrame,
    label_col: str = "is_correct",
) -> None:
    """Merge is_correct + notes from a Numbers export into target_csv (matched rows)."""
    if label_col not in gt_df.columns:
        raise ValueError(f"Numbers table missing column {label_col!r}")
    base = pd.read_csv(target_csv)
    for col in ("image", "det_image"):
        if col not in gt_df.columns:
            raise ValueError(f"Numbers table missing column {col!r}")
    if "score" not in gt_df.columns:
        raise ValueError("Numbers table missing column 'score'")

    base["score_f"] = pd.to_numeric(base["score"], errors="coerce").round(6)
    gt = gt_df.copy()
    gt["score_f"] = pd.to_numeric(gt["score"], errors="coerce").round(6)

    merge_cols = ["image", "det_image", "score_f"]
    gt_sub = gt.rename(columns={label_col: "_gt_label"})
    if "notes" in gt_sub.columns:
        gt_sub = gt_sub.rename(columns={"notes": "_gt_notes"})
    keep = merge_cols + ["_gt_label"]
    if "_gt_notes" in gt_sub.columns:
        keep.append("_gt_notes")
    gt_sub = gt_sub[keep].drop_duplicates(subset=merge_cols)

    merged = base.merge(gt_sub, on=merge_cols, how="left")

    merged["is_correct"] = merged["_gt_label"].map(_coerce_label_from_numbers)

    n_ok = int(merged["is_correct"].notna().sum())
    n_amb = int((merged["_gt_label"].notna() & merged["is_correct"].isna()).sum())
    n_unm = int(merged["_gt_label"].isna().sum())
    if n_unm:
        print(f"  Warning: {n_unm} target rows had no matching row in the Numbers table.")
    if n_amb:
        print(
            f"  Note: {n_amb} rows have ambiguous is_correct text in Numbers "
            f"(left blank in CSV; resolve for full ROC coverage)."
        )
    print(f"Updated {target_csv} — {n_ok} rows with numeric is_correct (0/1).")

    drop_cols = ["_gt_label", "score_f"]
    if "_gt_notes" in merged.columns:
        if "notes" in merged.columns:
            merged["notes"] = merged["_gt_notes"].combine_first(merged["notes"])
        else:
            merged["notes"] = merged["_gt_notes"]
        drop_cols.append("_gt_notes")

    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    merged.to_csv(target_csv, index=False)


def sync_alpr_labels_from_numbers(
    numbers_path: str = ALPR_NUMBERS_DEFAULT,
    target_csv: str = ALPR_INSPECTION_DEFAULT,
):
    """Copy is_correct / notes from alpr_data/visual_inspection.numbers into the CSV."""
    if not os.path.isfile(numbers_path):
        raise FileNotFoundError(numbers_path)
    gt_df = _numbers_table_to_dataframe(numbers_path)
    _merge_ground_truth_from_numbers_df(target_csv, gt_df)


def sync_surveillance_labels_from_numbers(
    numbers_path: str = SURV_NUMBERS_DEFAULT,
    target_csv: str = SURV_INSPECTION_DEFAULT,
):
    """Same as sync_alpr_labels_from_numbers for the general surveillance inspection CSV."""
    if not os.path.isfile(numbers_path):
        raise FileNotFoundError(
            f"{numbers_path} not found. Save your Numbers sheet there or pass --numbers_path=…"
        )
    gt_df = _numbers_table_to_dataframe(numbers_path)
    _merge_ground_truth_from_numbers_df(target_csv, gt_df)


def _score_band(s: float) -> str:
    if s >= 0.8:
        return "0.8-1.0 (high)"
    if s >= 0.6:
        return "0.6-0.8"
    if s >= 0.4:
        return "0.4-0.6"
    if s >= 0.2:
        return "0.2-0.4"
    return "0.0-0.2 (low)"


def ensure_surveillance_visual_inspection(
    all_detections_csv: str = SURV_ALL_DET,
    out_csv: str = SURV_INSPECTION_DEFAULT,
    force: bool = False,
) -> str:
    """
    Create data/threshold_analysis/visual_inspection.csv from all_detections.csv
    if it does not exist. Does not overwrite if the file already exists unless
    force=True.
    """
    if os.path.isfile(out_csv) and not force:
        return out_csv

    if not os.path.isfile(all_detections_csv):
        raise FileNotFoundError(f"Missing {all_detections_csv}")

    df = pd.read_csv(all_detections_csv)
    df = df.sort_values(["image", "score"], ascending=[True, False]).reset_index(drop=True)
    df["det_index"] = df.groupby("image").cumcount()

    def split_pano_heading(name: str):
        stem = name.replace(".jpg", "")
        panoid, h = stem.rsplit("_", 1)
        return panoid, int(h)

    pano_h = df["image"].map(split_pano_heading)
    df["panoid"] = [t[0] for t in pano_h]
    df["heading"] = [t[1] for t in pano_h]
    df["det_image"] = df["image"].str.replace(".jpg", "_det.jpg", regex=False)
    df["score_band"] = df["score"].apply(_score_band)
    df["is_correct"] = ""
    df["notes"] = ""

    cols = [
        "panoid",
        "heading",
        "image",
        "det_image",
        "det_index",
        "class",
        "class_id",
        "score",
        "score_band",
        "bbox",
        "is_correct",
        "notes",
    ]
    df = df[cols]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote surveillance visual inspection template: {out_csv}")
    return out_csv


def _row_has_binary_label(row: pd.Series) -> bool:
    return _parse_label(row.get("is_correct", "")) is not None


def _df_binary_labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where is_correct maps to 0 or 1 (yes/no, correct/incorrect, etc.)."""
    m = df.apply(_row_has_binary_label, axis=1)
    return df.loc[m].reset_index(drop=True)


def _parse_label(val) -> int | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if val == 1:
            return 1
        if val == 0:
            return 0
    s = str(val).strip().lower()
    if s == "" or s == "nan":
        return None
    if s in ("1", "1.0", "true", "yes", "y", "tp", "correct", "pos", "positive"):
        return 1
    if s in ("0", "0.0", "false", "no", "n", "fp", "incorrect", "neg", "negative", "probably not"):
        return 0
    return None


def _copy_if_exists(src: str, dst_dir: str) -> bool:
    if not os.path.isfile(src):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
    return True


def build_subset(
    out_dir: str = OUT_DIR_DEFAULT,
    seed: int = 42,
    n_per_source: int = 60,
    alpr_csv: str = ALPR_INSPECTION_DEFAULT,
    surv_csv: str = SURV_INSPECTION_DEFAULT,
):
    """
    Copy n_per_source random rows from each source (ALPR + surveillance), using
    only rows with a binary label (is_correct → 0/1). Equal counts per source.
    """
    ensure_surveillance_visual_inspection()

    alpr_labeled = _df_binary_labeled_only(pd.read_csv(alpr_csv))
    surv_labeled = _df_binary_labeled_only(pd.read_csv(surv_csv))

    if len(alpr_labeled) < n_per_source:
        raise ValueError(
            f"Need at least {n_per_source} ALPR rows with yes/no (or 0/1) labels; "
            f"have {len(alpr_labeled)} in {alpr_csv}."
        )
    if len(surv_labeled) < n_per_source:
        raise ValueError(
            f"Need at least {n_per_source} surveillance rows with binary labels; "
            f"have {len(surv_labeled)} in {surv_csv}. Fill or fix ambiguous labels."
        )

    alpr_df = alpr_labeled.sample(n=n_per_source, random_state=seed).reset_index(drop=True)
    surv_sample = surv_labeled.sample(n=n_per_source, random_state=seed).reset_index(
        drop=True
    )

    alpr_out_img = os.path.join(out_dir, "alpr", "images")
    alpr_out_det = os.path.join(out_dir, "alpr", "detections")
    surv_out_img = os.path.join(out_dir, "surveillance", "images")
    surv_out_det = os.path.join(out_dir, "surveillance", "detections")

    for _, row in alpr_df.iterrows():
        img = row["image"]
        det_img = row.get("det_image", img.replace(".jpg", "_det.jpg"))
        _copy_if_exists(os.path.join(ALPR_IMG_DIR, img), alpr_out_img)
        _copy_if_exists(os.path.join(ALPR_DET_DIR, det_img), alpr_out_det)
        stem = os.path.splitext(img)[0]
        _copy_if_exists(os.path.join(ALPR_DET_DIR, f"{stem}_det.json"), alpr_out_det)

    for _, row in surv_sample.iterrows():
        img = row["image"]
        det_img = row.get("det_image", img.replace(".jpg", "_det.jpg"))
        _copy_if_exists(os.path.join(SURV_IMG_DIR, img), surv_out_img)
        _copy_if_exists(os.path.join(SURV_DET_DIR, det_img), surv_out_det)
        stem = os.path.splitext(img)[0]
        _copy_if_exists(os.path.join(SURV_DET_DIR, f"{stem}_det.json"), surv_out_det)

    alpr_df = alpr_df.copy()
    alpr_df["source"] = "alpr"
    surv_sample = surv_sample.copy()
    surv_sample["source"] = "surveillance"

    combined = pd.concat([alpr_df, surv_sample], ignore_index=True, sort=False)
    combined_path = os.path.join(out_dir, "combined_visual_inspection.csv")
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(combined_path, index=False)

    meta = {
        "n_per_source": int(n_per_source),
        "n_alpr": int(len(alpr_df)),
        "n_surveillance_sample": int(len(surv_sample)),
        "seed": seed,
        "alpr_csv": alpr_csv,
        "surv_csv": surv_csv,
        "combined_csv": combined_path,
    }
    with open(os.path.join(out_dir, "subset_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {combined_path}")
    print(json.dumps(meta, indent=2))


def _plot_single_roc(
    y: np.ndarray,
    scores: np.ndarray,
    title: str,
    plot_path: str,
    line_color: str = "#2c7fb8",
) -> dict | None:
    """
    One ROC figure with AUC, chance line, Youden optimum marker + text box.
    Returns summary dict, or None if y does not contain both classes.
    """
    if len(np.unique(y)) < 2:
        return None

    fpr, tpr, thr = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    j = tpr - fpr
    j_eff = np.asarray(j, dtype=float).copy()
    j_eff[~np.isfinite(thr)] = -np.inf
    ix = int(np.argmax(j_eff))
    if not np.isfinite(j_eff[ix]):
        ix = int(np.argmax(j))
    best_fpr = float(fpr[ix])
    best_tpr = float(tpr[ix])
    best_thr_raw = thr[ix]
    j_stat = float(best_tpr - best_fpr)
    if np.isfinite(best_thr_raw):
        best_thr_json = float(best_thr_raw)
        thr_str = f"{best_thr_json:.4f}"
    else:
        best_thr_json = None
        thr_str = "+∞" if np.isposinf(best_thr_raw) else (
            "−∞" if np.isneginf(best_thr_raw) else str(best_thr_raw)
        )

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(fpr, tpr, color=line_color, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Chance")
    ax.scatter(
        [best_fpr],
        [best_tpr],
        s=130,
        c="#c51b7d",
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
        label="Youden optimum (max TPR−FPR)",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10)
    note = (
        f"Youden optimum (max TPR−FPR)\n"
        f"Confidence threshold τ = {thr_str}\n"
        f"TPR = {best_tpr:.4f}    FPR = {best_fpr:.4f}\n"
        f"J = TPR − FPR = {j_stat:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "wheat",
            "alpha": 0.94,
            "edgecolor": "gray",
        },
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_total": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "auc": float(roc_auc),
        "youden_j": j_stat,
        "best_threshold": best_thr_json,
        "best_threshold_note": thr_str,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr,
        "plot": plot_path,
    }


def _plot_single_roc_with_visual_best(
    y: np.ndarray,
    scores: np.ndarray,
    title: str,
    plot_path: str,
    line_color: str = "#2c7fb8",
) -> dict | None:
    """
    Copy of ROC plot with both Youden point and a visual-best proxy point.
    The visual-best proxy is the ROC point closest to top-left (0,1), which
    mirrors common manual visual inspection of ROC trade-offs.
    """
    if len(np.unique(y)) < 2:
        return None

    fpr, tpr, thr = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    # Youden's J optimum.
    j = tpr - fpr
    j_eff = np.asarray(j, dtype=float).copy()
    j_eff[~np.isfinite(thr)] = -np.inf
    ix_j = int(np.argmax(j_eff))
    if not np.isfinite(j_eff[ix_j]):
        ix_j = int(np.argmax(j))

    j_fpr = float(fpr[ix_j])
    j_tpr = float(tpr[ix_j])
    j_thr_raw = thr[ix_j]
    j_stat = float(j_tpr - j_fpr)
    if np.isfinite(j_thr_raw):
        j_thr_json = float(j_thr_raw)
        j_thr_str = f"{j_thr_json:.4f}"
    else:
        j_thr_json = None
        j_thr_str = (
            "+∞"
            if np.isposinf(j_thr_raw)
            else ("−∞" if np.isneginf(j_thr_raw) else str(j_thr_raw))
        )

    # Visual-best proxy: nearest point to top-left corner.
    d2 = (fpr - 0.0) ** 2 + (1.0 - tpr) ** 2
    d2_eff = np.asarray(d2, dtype=float).copy()
    d2_eff[~np.isfinite(thr)] = np.inf
    ix_v = int(np.argmin(d2_eff))
    if not np.isfinite(d2_eff[ix_v]):
        ix_v = int(np.argmin(d2))

    v_fpr = float(fpr[ix_v])
    v_tpr = float(tpr[ix_v])
    v_thr_raw = thr[ix_v]
    if np.isfinite(v_thr_raw):
        v_thr_json = float(v_thr_raw)
        v_thr_str = f"{v_thr_json:.4f}"
    else:
        v_thr_json = None
        v_thr_str = (
            "+∞"
            if np.isposinf(v_thr_raw)
            else ("−∞" if np.isneginf(v_thr_raw) else str(v_thr_raw))
        )

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(fpr, tpr, color=line_color, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Chance")
    ax.scatter(
        [j_fpr],
        [j_tpr],
        s=130,
        c="#c51b7d",
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
        label="Youden J point",
    )
    ax.scatter(
        [v_fpr],
        [v_tpr],
        s=110,
        c="#1f9d55",
        marker="D",
        zorder=6,
        edgecolors="white",
        linewidths=1.5,
        label="Visual-best threshold point",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10)
    note = (
        f"Youden J point: τ = {j_thr_str}, TPR = {j_tpr:.4f}, FPR = {j_fpr:.4f}, J = {j_stat:.4f}\n"
        f"Visual-best point: τ = {v_thr_str}, TPR = {v_tpr:.4f}, FPR = {v_fpr:.4f}\n"
        f"(visual-best uses nearest-to-top-left proxy)"
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "wheat",
            "alpha": 0.94,
            "edgecolor": "gray",
        },
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_total": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "auc": float(roc_auc),
        "youden_j": j_stat,
        "best_threshold": j_thr_json,
        "best_threshold_note": j_thr_str,
        "best_tpr": j_tpr,
        "best_fpr": j_fpr,
        "visual_best_threshold": v_thr_json,
        "visual_best_threshold_note": v_thr_str,
        "visual_best_tpr": v_tpr,
        "visual_best_fpr": v_fpr,
        "plot": plot_path,
    }


def plot_roc(
    alpr_csv: str = ALPR_INSPECTION_DEFAULT,
    surv_csv: str = SURV_INSPECTION_DEFAULT,
    combined_csv: str | None = None,
    out_dir: str = OUT_DIR_DEFAULT,
    seed: int = 42,
    n_per_source: int = 60,
    output_figures_dir: str = OUTPUT_FIGURES_DIR_DEFAULT,
    plot_path: str | None = None,
    plot_alpr_path: str | None = None,
    plot_surveillance_path: str | None = None,
):
    """
    Merge equal-sized ALPR + surveillance labeled detections and plot ROC.
    Matches build_subset: n_per_source rows from each, drawn only from binary-
    labeled rows, same random seed.

    Writes three PNGs in out_dir: combined, ALPR-only, surveillance-only (same
    Youden / styling). Override paths with plot_path, plot_alpr_path,
    plot_surveillance_path if needed.
    """
    ensure_surveillance_visual_inspection()

    if combined_csv and os.path.isfile(combined_csv):
        combo = pd.read_csv(combined_csv)
        parts = [combo[combo["source"] == s] for s in ("alpr", "surveillance")]
        alpr_blk = _df_binary_labeled_only(parts[0])
        surv_blk = _df_binary_labeled_only(parts[1])
        if len(alpr_blk) < n_per_source or len(surv_blk) < n_per_source:
            raise ValueError(
                f"combined_csv must contain at least {n_per_source} binary-labeled "
                f"rows per source; got alpr={len(alpr_blk)}, surveillance={len(surv_blk)}."
            )
        alpr_labeled = alpr_blk.sample(n=n_per_source, random_state=seed)
        surv_labeled = surv_blk.sample(n=n_per_source, random_state=seed)
    else:
        alpr_pool = _df_binary_labeled_only(pd.read_csv(alpr_csv))
        surv_pool = _df_binary_labeled_only(pd.read_csv(surv_csv))
        if len(alpr_pool) < n_per_source:
            raise ValueError(
                f"Need ≥{n_per_source} labeled ALPR rows; have {len(alpr_pool)}."
            )
        if len(surv_pool) < n_per_source:
            raise ValueError(
                f"Need ≥{n_per_source} labeled surveillance rows; have {len(surv_pool)}."
            )
        alpr_labeled = alpr_pool.sample(n=n_per_source, random_state=seed)
        surv_labeled = surv_pool.sample(n=n_per_source, random_state=seed)

    def labels_scores(df: pd.DataFrame):
        ys, sc = [], []
        for _, row in df.iterrows():
            lab = _parse_label(row.get("is_correct", ""))
            if lab is None:
                continue
            ys.append(lab)
            sc.append(float(row["score"]))
        return np.array(ys), np.array(sc)

    y_a, s_a = labels_scores(alpr_labeled)
    y_s, s_s = labels_scores(surv_labeled)

    if len(y_a) == 0 and len(y_s) == 0:
        raise ValueError(
            "No labeled rows (is_correct). Fill is_correct in "
            f"{alpr_csv} and {surv_csv} (or use a combined CSV with source column), "
            "then re-run."
        )

    y = np.concatenate([y_a, y_s])
    scores = np.concatenate([s_a, s_s])

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"ROC needs both positives and negatives in merged labels; "
            f"got unique labels {np.unique(y)}. Check is_correct values."
        )

    os.makedirs(out_dir, exist_ok=True)
    if plot_path is None:
        plot_path = os.path.join(out_dir, "roc_alpr_surveillance_balanced.png")
    if plot_alpr_path is None:
        plot_alpr_path = os.path.join(out_dir, "roc_alpr_only.png")
    if plot_surveillance_path is None:
        plot_surveillance_path = os.path.join(out_dir, "roc_surveillance_only.png")

    comb = _plot_single_roc(
        y,
        scores,
        "ROC — balanced ALPR + surveillance (per-detection, human labels)",
        plot_path,
        line_color="#2c7fb8",
    )
    assert comb is not None

    alpr_sub = _plot_single_roc(
        y_a,
        s_a,
        f"ROC — ALPR only (n={len(y_a)} labeled detections, same sample as combined)",
        plot_alpr_path,
        line_color="#1b7837",
    )
    surv_sub = _plot_single_roc(
        y_s,
        s_s,
        f"ROC — surveillance only (n={len(y_s)} labeled detections, same sample as combined)",
        plot_surveillance_path,
        line_color="#762a83",
    )

    summary = {
        "n_per_source": int(n_per_source),
        "seed": int(seed),
        "n_alpr_labeled": int(len(y_a)),
        "n_surv_labeled": int(len(y_s)),
        "n_total": comb["n_total"],
        "n_positive": comb["n_positive"],
        "n_negative": comb["n_negative"],
        "auc": comb["auc"],
        "youden_j": comb["youden_j"],
        "best_threshold": comb["best_threshold"],
        "best_threshold_note": comb["best_threshold_note"],
        "best_tpr": comb["best_tpr"],
        "best_fpr": comb["best_fpr"],
        "plot": plot_path,
        "roc_alpr_only": alpr_sub,
        "roc_surveillance_only": surv_sub,
    }

    # Also write copied ROC figures under outputs/figures with visual-best markers.
    os.makedirs(output_figures_dir, exist_ok=True)
    visual_combined_path = os.path.join(
        output_figures_dir, "roc_alpr_surveillance_balanced_visual_best.png"
    )
    visual_alpr_path = os.path.join(output_figures_dir, "roc_alpr_only_visual_best.png")
    visual_surveillance_path = os.path.join(
        output_figures_dir, "roc_surveillance_only_visual_best.png"
    )
    comb_visual = _plot_single_roc_with_visual_best(
        y,
        scores,
        "ROC — balanced ALPR + surveillance (J point + visual-best threshold)",
        visual_combined_path,
        line_color="#2c7fb8",
    )
    assert comb_visual is not None
    alpr_visual = _plot_single_roc_with_visual_best(
        y_a,
        s_a,
        f"ROC — ALPR only (n={len(y_a)} labeled detections, visual-best + J)",
        visual_alpr_path,
        line_color="#1b7837",
    )
    surv_visual = _plot_single_roc_with_visual_best(
        y_s,
        s_s,
        f"ROC — surveillance only (n={len(y_s)} labeled detections, visual-best + J)",
        visual_surveillance_path,
        line_color="#762a83",
    )
    summary["output_figures_dir"] = output_figures_dir
    summary["roc_combined_visual_best"] = comb_visual
    summary["roc_alpr_only_visual_best"] = alpr_visual
    summary["roc_surveillance_only_visual_best"] = surv_visual
    with open(os.path.join(out_dir, "roc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved combined plot to {plot_path}")
    if alpr_sub:
        print(f"Saved ALPR-only plot to {alpr_sub['plot']}")
    else:
        print("Skipped ALPR-only ROC (need both classes in sample).")
    if surv_sub:
        print(f"Saved surveillance-only plot to {surv_sub['plot']}")
    else:
        print("Skipped surveillance-only ROC (need both classes in sample).")
    print(f"Saved visual-best annotated ROC copies to {output_figures_dir}")


def main():
    fire.Fire(
        {
            "build_subset": build_subset,
            "plot_roc": plot_roc,
            "sync_alpr_labels_from_numbers": sync_alpr_labels_from_numbers,
            "sync_surveillance_labels_from_numbers": sync_surveillance_labels_from_numbers,
        }
    )


if __name__ == "__main__":
    main()

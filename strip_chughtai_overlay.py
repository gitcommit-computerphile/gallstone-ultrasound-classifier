"""
Strip Chughtai text overlays and produce a clean stage1/ folder.

What this does:
- Chughtai frames: crop top 8% (date/name strip) + right 15% (machine params)
- Gulab Devi frames: copied as-is (no changes yet)
- Output mirrors processed/ structure under stage1/

Run:
    python strip_chughtai_overlay.py
    python strip_chughtai_overlay.py --top 0.14 --right 0.20   # adjust crop %
    python strip_chughtai_overlay.py --preview                  # show one before/after, don't write
"""

import argparse
import shutil
from pathlib import Path

import cv2
import pandas as pd


DATASET_ROOT = Path(__file__).parent
PROCESSED    = DATASET_ROOT / "processed"
STAGE1       = DATASET_ROOT / "stage1"
MANIFEST     = DATASET_ROOT / "manifest.csv"


def strip_chughtai(img, top_frac: float, right_frac: float):
    h, w = img.shape[:2]
    top   = int(h * top_frac)
    right = int(w * right_frac)
    return img[top:, :w - right]


def main(top_frac: float, right_frac: float, preview: bool):
    df = pd.read_csv(MANIFEST)
    rows = df[df["processed_relpath"].notna() & (df["processed_relpath"] != "")]

    ch_rows = rows[rows["source"] == "Chughtai"]
    gd_rows = rows[rows["source"] == "GulabDevi"]

    print(f"Chughtai frames to crop : {len(ch_rows)}")
    print(f"Gulab Devi frames to copy: {len(gd_rows)}")

    if preview:
        sample = ch_rows.iloc[0]
        src = DATASET_ROOT / sample["processed_relpath"]
        img = cv2.imread(str(src))
        cropped = strip_chughtai(img, top_frac, right_frac)
        print(f"\nPreview: {src.name}")
        print(f"  Before: {img.shape[1]}w x {img.shape[0]}h")
        print(f"  After : {cropped.shape[1]}w x {cropped.shape[0]}h")
        cv2.imshow("Before", img)
        cv2.imshow("After", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # create stage1/ mirror structure
    for label_dir in PROCESSED.iterdir():
        if label_dir.is_dir():
            (STAGE1 / label_dir.name).mkdir(parents=True, exist_ok=True)

    cropped_count = 0
    copied_count  = 0

    # Chughtai — crop
    for _, row in ch_rows.iterrows():
        src  = DATASET_ROOT / row["processed_relpath"]
        dst  = STAGE1 / Path(row["processed_relpath"]).relative_to("processed")
        img  = cv2.imread(str(src))
        if img is None:
            print(f"  WARN: could not read {src}")
            continue
        out = strip_chughtai(img, top_frac, right_frac)
        cv2.imwrite(str(dst), out)
        cropped_count += 1

    # Gulab Devi — copy as-is
    for _, row in gd_rows.iterrows():
        src = DATASET_ROOT / row["processed_relpath"]
        dst = STAGE1 / Path(row["processed_relpath"]).relative_to("processed")
        shutil.copy2(src, dst)
        copied_count += 1

    print(f"\nDone.")
    print(f"  Cropped  (Chughtai)  : {cropped_count}")
    print(f"  Copied   (Gulab Devi): {copied_count}")
    print(f"  Output   : {STAGE1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",     type=float, default=0.08,  help="Fraction of height to strip from top")
    parser.add_argument("--right",   type=float, default=0.15,  help="Fraction of width to strip from right")
    parser.add_argument("--preview", action="store_true",        help="Show before/after for one image, do not write")
    args = parser.parse_args()
    main(args.top, args.right, args.preview)
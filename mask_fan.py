"""
Mask the ultrasound fan and blacken everything outside it.

What this does:
- Detects the fan/dome boundary in each image (the curved sector region that
  contains actual ultrasound data)
- Blacks out all pixels outside that boundary
- Removes peripheral artifacts: T marker, depth scale numbers, dot row, etc.
- Reads from stage2/, outputs to stage3/ (non-destructive)

Run:
    python mask_fan.py
    python mask_fan.py --preview
    python mask_fan.py --preview --file normal/ch_adnan_03.png
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DATASET_ROOT = Path(__file__).parent
STAGE2       = DATASET_ROOT / "stage2"
STAGE3       = DATASET_ROOT / "stage3"
MANIFEST     = DATASET_ROOT / "manifest.csv"


def create_fan_mask(img: np.ndarray, threshold: int = 8) -> np.ndarray:
    """
    Returns a binary mask (255 = fan, 0 = outside).

    Strategy:
      1. Threshold to find all non-black pixels.
      2. Large morphological close to bridge interior dark regions
         (fluid-filled GB, acoustic shadows) into one solid blob.
      3. Find the largest connected component — that is the fan.
      4. Fill its contour to produce a solid mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # kernel ≈ 10 % of the shorter dimension; bridges most acoustic shadows
    h, w = binary.shape
    ksize = max(30, min(h, w) // 10)
    kernel = np.ones((ksize, ksize), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(binary.shape, 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def apply_fan_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = img.copy()
    result[mask == 0] = 0
    return result


def main(preview: bool, preview_file: str | None):
    df   = pd.read_csv(MANIFEST)
    rows = df[df["processed_relpath"].notna() & (df["processed_relpath"] != "")]
    rows = rows[rows["label"].isin(["stones", "normal"])]

    print(f"Binary frames to process: {len(rows)}")

    if preview:
        if preview_file:
            src = STAGE2 / preview_file
        else:
            # default: first Chughtai normal (most visible T + GB label)
            ch_normal = rows[(rows["source"] == "Chughtai") & (rows["label"] == "normal")]
            sample = ch_normal.iloc[0]
            src = STAGE2 / Path(sample["processed_relpath"]).relative_to("processed")

        img = cv2.imread(str(src))
        if img is None:
            print(f"Could not read: {src}")
            return

        mask   = create_fan_mask(img)
        result = apply_fan_mask(img, mask)

        print(f"\nPreview: {src.name}")
        print(f"  Image size : {img.shape[1]}w x {img.shape[0]}h")
        masked_px = int((mask == 0).sum())
        total_px  = mask.size
        print(f"  Pixels blacked out: {masked_px} / {total_px} ({masked_px/total_px:.1%})")

        # show mask boundary in green for inspection
        overlay = result.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        cv2.imshow("Original",     img)
        cv2.imshow("Fan mask",     mask)
        cv2.imshow("Masked",       result)
        cv2.imshow("Boundary",     overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # create stage3/ mirror structure
    for label_dir in STAGE2.iterdir():
        if label_dir.is_dir():
            (STAGE3 / label_dir.name).mkdir(parents=True, exist_ok=True)

    processed = 0
    for _, row in rows.iterrows():
        rel = Path(row["processed_relpath"]).relative_to("processed")
        src = STAGE2 / rel
        dst = STAGE3 / rel
        if not src.exists():
            print(f"  WARN: missing {src}")
            continue
        img    = cv2.imread(str(src))
        mask   = create_fan_mask(img)
        result = apply_fan_mask(img, mask)
        cv2.imwrite(str(dst), result)
        processed += 1

    print(f"\nDone. Processed {processed} frames → {STAGE3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Show before/after for one image, do not write")
    parser.add_argument("--file", type=str, default=None,
                        help="File to preview relative to stage2/ e.g. normal/ch_adnan_03.png")
    args = parser.parse_args()
    main(args.preview, args.file)
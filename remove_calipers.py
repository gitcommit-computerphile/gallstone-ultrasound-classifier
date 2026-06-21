"""
Remove caliper markers from stone images via OpenCV inpainting.

What this does:
- Reads from stage1/ (post-Chughtai-crop)
- Stone frames: detects bright/colored caliper pixels via HSV threshold, inpaints
- Non-stone frames: copied as-is (no calipers expected)
- Output goes to stage2/ (non-destructive, stage1/ unchanged)

Run:
    python remove_calipers.py
    python remove_calipers.py --preview           # show one before/after, don't write
    python remove_calipers.py --preview --file stones/gd_2025-10-01_khursheed-bb_01.png
"""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd


DATASET_ROOT = Path(__file__).parent
STAGE1       = DATASET_ROOT / "stage1"
STAGE2       = DATASET_ROOT / "stage2"
MANIFEST     = DATASET_ROOT / "manifest.csv"


def build_caliper_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # bright white/grey caliper markers (high value, any hue, low-mid saturation)
    mask_white = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))

    # colored markers — cyan, yellow, orange (high saturation)
    mask_color = cv2.inRange(hsv, (0, 100, 150), (180, 255, 255))

    mask = cv2.bitwise_or(mask_white, mask_color)

    # exclude large bright regions (fan highlight, not calipers)
    # calipers are small — remove connected components larger than 300px
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            filtered[labels == i] = 255

    # small dilation to cover anti-aliased edges
    kernel = np.ones((3, 3), np.uint8)
    filtered = cv2.dilate(filtered, kernel, iterations=1)
    return filtered


def remove_calipers(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask   = build_caliper_mask(img)
    result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result, mask


def main(preview: bool, preview_file: str | None):
    df    = pd.read_csv(MANIFEST)
    rows  = df[df["processed_relpath"].notna() & (df["processed_relpath"] != "")]

    stone_rows  = rows[rows["label"] == "stones"]
    other_rows  = rows[rows["label"] != "stones"]

    print(f"Stone frames to inpaint : {len(stone_rows)}")
    print(f"Other frames to inpaint : {len(other_rows)}  (removes GB label, depth numbers, colored markers)")

    if preview:
        if preview_file:
            src = STAGE1 / preview_file
        else:
            sample = stone_rows[stone_rows["source"] == "GulabDevi"].iloc[0]
            src = STAGE1 / Path(sample["processed_relpath"]).relative_to("processed")

        img = cv2.imread(str(src))
        if img is None:
            print(f"Could not read: {src}")
            return

        result, mask = remove_calipers(img)
        print(f"\nPreview: {src.name}")
        print(f"  Caliper pixels masked: {mask.sum() // 255}")
        cv2.imshow("Original", img)
        cv2.imshow("Mask",     mask)
        cv2.imshow("Inpainted", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # create stage2/ mirror structure
    for label_dir in STAGE1.iterdir():
        if label_dir.is_dir():
            (STAGE2 / label_dir.name).mkdir(parents=True, exist_ok=True)

    inpainted_count = 0

    # all frames — inpaint (catches calipers on stones, GB label + depth numbers on normals)
    for _, row in rows.iterrows():
        rel = Path(row["processed_relpath"]).relative_to("processed")
        src = STAGE1 / rel
        dst = STAGE2 / rel
        img = cv2.imread(str(src))
        if img is None:
            print(f"  WARN: could not read {src}")
            continue
        result, _ = remove_calipers(img)
        cv2.imwrite(str(dst), result)
        inpainted_count += 1

    print(f"\nDone.")
    print(f"  Inpainted (all frames) : {inpainted_count}")
    print(f"  Output                 : {STAGE2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Show before/after/mask for one image, do not write")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific file to preview relative to stage1/ e.g. stones/gd_2025-10-01_khursheed-bb_01.png")
    args = parser.parse_args()
    main(args.preview, args.file)
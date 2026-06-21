"""
Convert VIA polygon annotations → binary mask PNGs in fan_masks/.

Reads:  "Unique Image Outlines (3).json"  (VGG Image Annotator format)
Writes: fan_masks/mask_{W}x{H}.png  (white = fan, black = outside)

Run: python convert_outlines.py
"""

import json
import numpy as np
import cv2
from pathlib import Path

ROOT      = Path(__file__).parent
JSON_FILE = ROOT / "Unique Image Outlines (3).json"
MASKS_DIR = ROOT / "fan_masks"

MASKS_DIR.mkdir(exist_ok=True)

with open(JSON_FILE) as f:
    data = json.load(f)

img_metadata = data["_via_img_metadata"]
saved = 0

for entry in img_metadata.values():
    filename = entry["filename"]          # e.g. "221x194__gd_2025-10-02_safia_01.png"
    regions  = entry["regions"]

    # parse W x H from filename prefix
    size_str = filename.split("__")[0]    # "221x194"
    w, h     = map(int, size_str.split("x"))

    if not regions:
        print(f"  WARN: no polygon for {filename} — skipped")
        continue

    sa  = regions[0]["shape_attributes"]
    xs  = sa["all_points_x"]
    ys  = sa["all_points_y"]
    pts = np.array(list(zip(xs, ys)), dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    out = MASKS_DIR / f"mask_{w}x{h}.png"
    cv2.imwrite(str(out), mask)
    print(f"  {w}x{h}  →  {out.name}")
    saved += 1

print(f"\nDone. {saved}/{len(img_metadata)} masks saved to {MASKS_DIR}/")
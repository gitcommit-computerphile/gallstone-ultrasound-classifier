"""
Convert VIA polygon annotations to binary mask PNGs in fan_masks/.

Reads:  all "Unique Image Outlines*.json" files in the project root
Writes: fan_masks/mask_{W}x{H}.png  (white = fan, black = outside)

Run: python convert_outlines.py
"""

import json
import numpy as np
import cv2
from pathlib import Path

ROOT      = Path(__file__).parent
MASKS_DIR = ROOT / "fan_masks"

MASKS_DIR.mkdir(exist_ok=True)

json_files = sorted(ROOT.glob("Unique Image Outlines*.json"))
if not json_files:
    print("No 'Unique Image Outlines*.json' files found.")
    raise SystemExit(1)

print(f"Found {len(json_files)} annotation file(s):")
for jf in json_files:
    print(f"  {jf.name}")
print()

saved = 0
skipped = 0

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    img_metadata = data["_via_img_metadata"]

    for entry in img_metadata.values():
        filename = entry["filename"]
        regions  = entry["regions"]

        # handles "472x574__ch_name.png", "frame_472x574.png", "472x574.png"
        size_str = filename.split("__")[0]
        size_str = size_str.replace("frame_", "").replace(".png", "")
        w, h     = map(int, size_str.split("x"))

        if not regions:
            print(f"  WARN: no polygon for {filename} -- skipped")
            skipped += 1
            continue

        sa  = regions[0]["shape_attributes"]
        xs  = sa["all_points_x"]
        ys  = sa["all_points_y"]
        pts = np.array(list(zip(xs, ys)), dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        out = MASKS_DIR / f"mask_{w}x{h}.png"
        cv2.imwrite(str(out), mask)
        print(f"  {w}x{h}  ->  {out.name}")
        saved += 1

print(f"\nDone. {saved} masks saved, {skipped} skipped -> {MASKS_DIR}/")
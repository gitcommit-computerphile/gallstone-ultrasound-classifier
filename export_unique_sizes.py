"""
Copy one representative image per unique size from stage2/ into unique_sizes/.
Filename format: {W}x{H}__{original_name}.png
Run: python export_unique_sizes.py
"""

import shutil
from pathlib import Path
import cv2

STAGE2      = Path(__file__).parent / "stage2"
OUT_DIR     = Path(__file__).parent / "unique_sizes"

OUT_DIR.mkdir(exist_ok=True)

sizes: dict[tuple[int, int], Path] = {}
for f in sorted(STAGE2.rglob("*.png")):
    img = cv2.imread(str(f))
    if img is None:
        continue
    h, w = img.shape[:2]
    if (w, h) not in sizes:
        sizes[(w, h)] = f

for (w, h), src in sorted(sizes.items()):
    dst = OUT_DIR / f"{w}x{h}__{src.name}"
    shutil.copy2(src, dst)

print(f"Exported {len(sizes)} images to {OUT_DIR}/")
for (w, h), src in sorted(sizes.items()):
    print(f"  {w}x{h}  →  {src.name}")
"""
Generate one example image per augmentation, for documentation.

Takes a single frame from stage3/ and applies each augmentation from the training
pipeline in isolation, so the effect of each can be seen on its own.

IMPORTANT: panels use the STRONGEST setting in each transform's range, not a random
draw. A random draw often lands near the identity value (blur sigma 0.1, jitter
factor 1.0) and produces a panel indistinguishable from the original, which is
useless as documentation. Training still samples these ranges randomly; these
panels show the worst case the model is asked to cope with.

Every range mirrors dataset.py::get_transforms(augment=True) - keep the two in sync:

    RandomHorizontalFlip()                        p = 0.5
    RandomRotation(10)                            +/- 10 degrees        -> demo uses -10
    RandomResizedCrop(224, scale=(0.8, 1.0))      keep 80-100% of area  -> demo uses 0.80
    ColorJitter(brightness=0.3, contrast=0.3)     factor in [0.7, 1.3]  -> demo uses 1.3 / 0.7
    GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))                       -> demo uses 1.0
    AddGaussianNoise(std=0.02)                    applied AFTER Normalize

Outputs per augmentation:
    NN_<name>.png        the augmented image on its own
    diff/NN_<name>.png   |augmented - original|, amplified, to prove what moved

Usage:
    python make_augmentation_examples.py
    python make_augmentation_examples.py --image stage3/stones/<name>.png
    python make_augmentation_examples.py --variants        # random draws of full pipeline
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

# must match dataset.py
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
NOISE_STD = 0.02
SIZE = 224

OUT_DIR = Path("augmentation_examples")
DIFF_DIR = OUT_DIR / "diff"
DIFF_GAIN = 6  # amplify difference maps so subtle changes are visible


def pad_to_square(img: Image.Image) -> Image.Image:
    """Same as dataset.py::_pad_to_square."""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def base(img: Image.Image) -> Image.Image:
    return pad_to_square(img).resize((SIZE, SIZE), Image.BILINEAR)


def crop_at(img: Image.Image, scale: float) -> Image.Image:
    """Centre-biased crop keeping `scale` of the area, resized back to 224."""
    sq = pad_to_square(img)
    w, h = sq.size
    cw, ch = int(w * scale ** 0.5), int(h * scale ** 0.5)
    x, y = (w - cw) // 2, (h - ch) // 2
    return sq.crop((x, y, x + cw, y + ch)).resize((SIZE, SIZE), Image.BILINEAR)


def noise(ref: Image.Image, amplify: int = 1) -> Image.Image:
    """
    dataset.py adds noise AFTER Normalize, so std=0.02 there is 0.02 * STD in
    0-1 space: about 1.15 pixel levels out of 255. `amplify` is documentation only.
    """
    a = np.asarray(ref).astype(np.float32) / 255.0
    n = np.random.randn(*a.shape) * (NOISE_STD * STD[None, None, :] * amplify)
    return Image.fromarray((np.clip(a + n, 0, 1) * 255).astype(np.uint8))


def full_pipeline(img: Image.Image) -> Image.Image:
    """Everything, randomly sampled, in dataset.py's order."""
    out = pad_to_square(img)
    if random.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    out = out.rotate(random.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    sq = pad_to_square(out)
    w, h = sq.size
    s = random.uniform(0.8, 1.0)
    cw, ch = int(w * s ** 0.5), int(h * s ** 0.5)
    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
    out = sq.crop((x, y, x + cw, y + ch)).resize((SIZE, SIZE), Image.BILINEAR)
    out = ImageEnhance.Brightness(out).enhance(random.uniform(0.7, 1.3))
    out = ImageEnhance.Contrast(out).enhance(random.uniform(0.7, 1.3))
    out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
    return noise(out)


def diff_map(ref: Image.Image, aug: Image.Image) -> Image.Image:
    d = ImageChops.difference(ref.convert("RGB"), aug.convert("RGB"))
    a = np.asarray(d).astype(np.float32) * DIFF_GAIN
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def label_strip(text: str, width: int, colour=(40, 40, 40)) -> Image.Image:
    strip = Image.new("RGB", (width, 16), (250, 250, 250))
    ImageDraw.Draw(strip).text((2, 2), text, fill=colour)
    return strip


def clean_output_dirs() -> int:
    """
    Wipe PNGs from the output dirs before regenerating.

    Panel names change whenever the demo parameters change, so stale files from an
    earlier run would otherwise sit alongside the new ones and be mistaken for them.
    Scoped deliberately: only *.png, only in these two directories, no recursion.
    """
    removed = 0
    for d in (OUT_DIR, DIFF_DIR):
        if not d.is_dir():
            continue
        for f in d.glob("*.png"):
            f.unlink()
            removed += 1
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to a stage3 PNG")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.image:
        src = Path(args.image)
    else:
        stones = sorted(Path("stage3/stones").glob("*.png"))
        if not stones:
            raise SystemExit("no images found in stage3/stones/")
        src = stones[len(stones) // 2]

    random.seed(args.seed)
    np.random.seed(args.seed)

    img = Image.open(src).convert("RGB")
    OUT_DIR.mkdir(exist_ok=True)
    DIFF_DIR.mkdir(exist_ok=True)

    wiped = clean_output_dirs()
    if wiped:
        print(f"cleared {wiped} old PNG(s) from {OUT_DIR}/\n")

    ref = base(img)

    # ONE panel per augmentation TYPE, each at the strongest setting in its range.
    # Where a transform has two directions (brightness up/down, contrast up/down)
    # only the direction that moves pixels furthest is kept, measured empirically.
    panels = [
        ("00_original.png", "original (pad + resize)",
         ref),
        ("01_horizontal_flip.png", "horizontal flip",
         ref.transpose(Image.FLIP_LEFT_RIGHT)),
        ("02_rotation.png", "rotation -10 deg (range edge)",
         ref.rotate(-10, resample=Image.BILINEAR, fillcolor=(0, 0, 0))),
        ("03_random_crop.png", "crop to 80% area (range edge)",
         crop_at(img, 0.80)),
        ("04_brightness.png", "brightness x0.7 (strongest direction)",
         ImageEnhance.Brightness(ref).enhance(0.7)),
        ("05_contrast.png", "contrast x0.7 (strongest direction)",
         ImageEnhance.Contrast(ref).enhance(0.7)),
        ("06_gaussian_blur.png", "gaussian blur sigma=1.0 (range max)",
         ref.filter(ImageFilter.GaussianBlur(radius=1.0))),
        ("07_gaussian_noise.png", "gaussian noise std=0.02 (true strength)",
         noise(ref)),
        ("08_full_pipeline.png", "FULL pipeline, all combined",
         full_pipeline(img)),
        ("09_FORBIDDEN_vertical_flip.png", "VERTICAL FLIP - never use",
         ref.transpose(Image.FLIP_TOP_BOTTOM)),
    ]

    saved = []
    print(f"{'file':36s} {'mean |diff|':>11s}  {'max |diff|':>10s}   effect")
    print("-" * 92)
    for fname, label, out in panels:
        out.save(OUT_DIR / fname)
        d = np.asarray(ImageChops.difference(ref, out.convert("RGB"))).astype(float)
        if fname != "00_original.png":
            diff_map(ref, out).save(DIFF_DIR / fname)
        saved.append((OUT_DIR / fname, label, d.mean()))
        print(f"{fname:36s} {d.mean():11.2f}  {d.max():10.0f}   {label}")

    # contact sheet: augmented on top row-pair, diff underneath
    PAD, COLS, CELL, TXT = 6, 5, 200, 16
    rows = (len(saved) + COLS - 1) // COLS
    sheet = Image.new(
        "RGB",
        (COLS * CELL + PAD * (COLS + 1),
         rows * (CELL * 2 + TXT * 2) + PAD * (rows + 1)),
        (250, 250, 250),
    )
    d = ImageDraw.Draw(sheet)
    for i, (path, label, mdiff) in enumerate(saved):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL * 2 + TXT * 2 + PAD)
        sheet.paste(Image.open(path).resize((CELL, CELL)), (x, y))
        colour = (200, 0, 0) if "FORBIDDEN" in path.name else (40, 40, 40)
        d.text((x + 2, y + CELL + 3), label[:42], fill=colour)
        dp = DIFF_DIR / path.name
        if dp.exists():
            sheet.paste(Image.open(dp).resize((CELL, CELL)), (x, y + CELL + TXT))
            d.text((x + 2, y + CELL * 2 + TXT + 3),
                   f"diff x{DIFF_GAIN}  (mean {mdiff:.1f})", fill=(90, 90, 90))
        else:
            d.text((x + 2, y + CELL + TXT + 3), "(reference)", fill=(150, 150, 150))
    sheet.save(OUT_DIR / "_contact_sheet.png")

    if False:
        vs = [full_pipeline(img) for _ in range(8)]
        vsheet = Image.new("RGB", (4 * CELL + PAD * 5, 2 * CELL + PAD * 3), (250, 250, 250))
        for i, v in enumerate(vs):
            r, c = divmod(i, 4)
            vsheet.paste(v.resize((CELL, CELL)),
                         (PAD + c * (CELL + PAD), PAD + r * (CELL + PAD)))
            v.save(OUT_DIR / f"variant_{i:02d}.png")
        vsheet.save(OUT_DIR / "_variants_sheet.png")
        print(f"\nalso wrote 8 random full-pipeline variants + _variants_sheet.png")

    print(f"\nsource : {src}")
    print(f"seed   : {args.seed}")
    print(f"wrote  : {len(saved)} panels + {len(saved)-1} diff maps -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
"""
Two-step fan masking tool.

STEP 1 — Draw outlines (once):
    python create_fan_masks.py --draw

    Opens one representative image for every unique (width, height) found in
    stage2/. Click polygon vertices around the fan/dome. Keyboard controls:

        Left click  — add point
        Right click — undo last point
        Enter       — confirm and save this mask, move to next size
        R           — reset (clear all points)
        S           — skip this size (no mask saved, image copied as-is on apply)
        Q           — quit early (already-saved masks are kept)

    Masks are stored as binary PNGs in fan_masks/mask_{W}x{H}.png

STEP 2 — Apply masks:
    python create_fan_masks.py --apply

    For every binary frame in stage2/:
      - Looks up the mask for that image's (W x H)
      - Blacks out everything outside the mask
      - Writes to stage3/  (non-destructive, stage2/ unchanged)

Other:
    python create_fan_masks.py --list   # show unique sizes + frame counts
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
MASKS_DIR    = DATASET_ROOT / "fan_masks"
MANIFEST     = DATASET_ROOT / "manifest.csv"


# ─── helpers ────────────────────────────────────────────────────────────────

def scan_sizes(stage2_dir: Path) -> dict[tuple[int, int], list[Path]]:
    """Return {(w, h): [paths]} for every PNG in stage2/."""
    sizes: dict[tuple[int, int], list[Path]] = {}
    for f in sorted(stage2_dir.rglob("*.png")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes.setdefault((w, h), []).append(f)
    return sizes


def mask_path(w: int, h: int) -> Path:
    return MASKS_DIR / f"mask_{w}x{h}.png"


def load_mask(w: int, h: int) -> np.ndarray | None:
    p = mask_path(w, h)
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return m


# ─── list mode ──────────────────────────────────────────────────────────────

def cmd_list():
    sizes = scan_sizes(STAGE2)
    print(f"\n{'Size':>12}  {'Frames':>6}  {'Mask saved':>10}  Representative")
    print("─" * 70)
    total = 0
    for (w, h), files in sorted(sizes.items(), key=lambda x: -len(x[1])):
        saved = "YES" if mask_path(w, h).exists() else "-"
        print(f"{w:>5}x{h:<5}  {len(files):>6}  {saved:>10}  {files[0].name}")
        total += len(files)
    print(f"\nTotal: {len(sizes)} unique sizes, {total} frames")


# ─── draw mode ──────────────────────────────────────────────────────────────

INSTRUCTIONS = [
    "Left click  = add point",
    "Right click = undo last",
    "Enter = save & next",
    "R = reset  S = skip  Q = quit",
]


def draw_instructions(canvas: np.ndarray):
    y = 20
    for line in INSTRUCTIONS:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 100), 1, cv2.LINE_AA)
        y += 20


def draw_overlay(base: np.ndarray, points: list[tuple[int, int]],
                 size_label: str, idx: int, total: int) -> np.ndarray:
    canvas = base.copy()
    pts = np.array(points, dtype=np.int32)

    if len(points) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    for i, pt in enumerate(points):
        cv2.circle(canvas, pt, 5, (0, 80, 255), -1)
        if i > 0:
            cv2.line(canvas, points[i - 1], pt, (0, 255, 0), 2)
    if len(points) >= 2:
        # close the loop preview (thin dashed-ish line from last to first)
        cv2.line(canvas, points[-1], points[0], (0, 200, 0), 1)

    header = f"[{idx}/{total}]  {size_label}   pts: {len(points)}"
    cv2.putText(canvas, header, (10, base.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
    draw_instructions(canvas)
    return canvas


def interactive_draw(img: np.ndarray, size_label: str,
                     idx: int, total: int) -> np.ndarray | None:
    """
    Returns filled binary mask (uint8) or None if user skips/quits.
    Caller checks return value: None can mean skip (continue) or quit (stop).
    Sets a 'quit' attribute on the return sentinel.
    """
    points: list[tuple[int, int]] = []
    WIN = "Fan outline"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    # Resize window to fit screen reasonably
    disp_h = min(700, img.shape[0])
    disp_w = int(img.shape[1] * disp_h / img.shape[0])
    cv2.resizeWindow(WIN, disp_w, disp_h)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(WIN, on_mouse)

    action = None
    while action is None:
        frame = draw_overlay(img, points, size_label, idx, total)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if len(points) >= 3:
                action = "save"
            # else ignore — need at least 3 points
        elif key == ord("r"):
            points.clear()
        elif key == ord("s"):
            action = "skip"
        elif key == ord("q"):
            action = "quit"

    cv2.destroyWindow(WIN)

    if action == "save":
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
        return mask, False      # (mask, quit_flag)
    elif action == "skip":
        return None, False
    else:  # quit
        return None, True


def cmd_draw():
    MASKS_DIR.mkdir(exist_ok=True)
    sizes = scan_sizes(STAGE2)

    if not sizes:
        print("No PNG files found in stage2/. Run preprocessing steps first.")
        return

    sizes_list = sorted(sizes.items(), key=lambda x: -len(x[1]))
    total = len(sizes_list)
    print(f"Found {total} unique sizes. Will open one representative per size.")
    print("Already-saved masks will be shown — press Enter to keep or R to redraw.\n")

    for idx, ((w, h), files) in enumerate(sizes_list, start=1):
        rep   = files[0]
        label = f"{w}x{h}  ({len(files)} frame{'s' if len(files)>1 else ''})"
        img   = cv2.imread(str(rep))
        if img is None:
            print(f"  WARN: could not read {rep}, skipping")
            continue

        mp = mask_path(w, h)
        if mp.exists():
            # overlay existing mask so user can see what's saved
            existing = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            colored = cv2.cvtColor(existing, cv2.COLOR_GRAY2BGR)
            green_overlay = np.zeros_like(img)
            green_overlay[:, :, 1] = existing  # green channel = mask
            img_with_mask = cv2.addWeighted(img, 0.7, green_overlay, 0.3, 0)
            print(f"  [{idx}/{total}]  {label}  — mask already saved. "
                  f"Press Enter to keep, R to redraw, S to skip.")
            # Show existing; if user presses Enter immediately, we skip redraw
            cv2.namedWindow("Existing mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Existing mask", min(700, w), min(700, h))
            cv2.imshow("Existing mask", img_with_mask)
            k = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("Existing mask")
            if k in (13, 10) or k == ord("s"):
                print(f"    kept existing mask.")
                continue
            elif k == ord("q"):
                print("  Quit.")
                break
            # else 'r' or anything else → fall through to redraw

        result, quit_flag = interactive_draw(img, label, idx, total)
        if result is not None:
            cv2.imwrite(str(mp), result)
            print(f"  [{idx}/{total}]  saved  {mp.name}")
        else:
            print(f"  [{idx}/{total}]  skipped  {label}")

        if quit_flag:
            print("Quit early. Run again to continue from where you left off.")
            break

    cv2.destroyAllWindows()
    saved = sum(1 for (w, h), _ in sizes_list if mask_path(w, h).exists())
    print(f"\nDone. {saved}/{total} sizes have masks saved in fan_masks/")


# ─── apply mode ─────────────────────────────────────────────────────────────

def cmd_apply():
    df   = pd.read_csv(MANIFEST)
    rows = df[df["processed_relpath"].notna() & (df["processed_relpath"] != "")]
    rows = rows[rows["label"].isin(["stones", "normal"])]

    STAGE3.mkdir(exist_ok=True)
    for label_dir in STAGE2.iterdir():
        if label_dir.is_dir():
            (STAGE3 / label_dir.name).mkdir(parents=True, exist_ok=True)

    masked = copied = skipped = 0

    for _, row in rows.iterrows():
        rel = Path(row["processed_relpath"]).relative_to("processed")
        src = STAGE2 / rel
        dst = STAGE3 / rel
        if not src.exists():
            print(f"  WARN: missing {src}")
            skipped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        m = load_mask(w, h)
        if m is None:
            # no mask for this size — copy as-is
            shutil.copy2(src, dst)
            copied += 1
        else:
            result = img.copy()
            result[m == 0] = 0
            cv2.imwrite(str(dst), result)
            masked += 1

    print(f"\nDone.")
    print(f"  Masked  (fan mask applied) : {masked}")
    print(f"  Copied  (no mask for size) : {copied}")
    print(f"  Skipped (missing source)   : {skipped}")
    print(f"  Output : {STAGE3}")
    if copied:
        print(f"\n  NOTE: {copied} frames had no saved mask — run --draw to add them.")


# ─── entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--draw",  action="store_true", help="Interactively draw fan outline for each unique size")
    group.add_argument("--apply", action="store_true", help="Apply saved masks: stage2/ → stage3/")
    group.add_argument("--list",  action="store_true", help="List unique sizes and mask status")
    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.draw:
        cmd_draw()
    elif args.apply:
        cmd_apply()
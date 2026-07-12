"""
ingest_new_dataset.py

Stage 0 pipeline for new_dataset_12_july/ — sector crop, deduplication,
and manifest update.

What this does:
  0.1  Scan new_dataset_12_july/ folder structure → metadata per frame
  0.2  Sector crop: Chughtai dual-pane → right pane → largest-CC bounding box → processed/
  0.3  Perceptual hash deduplication (within new batch + against existing processed/)
  0.4  Append new rows to manifest.csv (existing rows untouched)

After running this, run the staging pipeline:
  python strip_chughtai_overlay.py --top 0.14 --right 0.20   → stage1/
  python remove_calipers.py                                    → stage2/
  python mask_fan.py                                           → stage3/

Usage:
  python ingest_new_dataset.py              # full run
  python ingest_new_dataset.py --dry-run    # show what would happen, write nothing
  python ingest_new_dataset.py --preview PATIENT_FOLDER_NAME  # show crop for one patient
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT       = Path(__file__).parent
NEW_DATA   = ROOT / "new_dataset_12_july"
PROCESSED  = ROOT / "processed"
MANIFEST   = ROOT / "manifest.csv"

# Chughtai dual-pane: GB is always in the right pane
SOURCE     = "Chughtai"
HOSPITAL   = "Chughtai Lab"

# Folder name → canonical label value
LABEL_MAP = {"normal": "normal", "stone": "stones", "stones": "stones"}

INSET = 4          # pixels to inset from bounding box edges when cropping
HAMMING_THRESH = 6  # near-duplicate threshold


# ─── Name normalisation ───────────────────────────────────────────────────────

def normalize_patient_name(folder: str) -> str:
    """Convert raw folder name to a stable, lowercase slug used in patient_id."""
    name = folder.strip()
    name = re.sub(r"\s*\(.*?\)", "", name)   # drop parenthetical notes
    name = re.sub(r"_20(?=[A-Za-z])", " ", name)  # _20 encoding → space
    name = name.replace(".", "-").replace("_", " ")
    name = name.lower().strip()
    name = re.sub(r"[\s]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def make_patient_id(slug: str) -> str:
    return f"ch_{slug}"


# ─── Frame number from BMP filename ──────────────────────────────────────────

def frame_num(bmp_path: Path) -> str:
    """Return 2-digit frame number extracted from the BMP filename."""
    digits = re.findall(r"\d+", bmp_path.stem)
    if digits:
        last = digits[-1][-3:]   # last 3 digits of last numeric group
        return f"{int(last):02d}"
    return "01"


# ─── Perceptual hash (DCT-based) ──────────────────────────────────────────────

def phash(img: np.ndarray, size: int = 32, keep: int = 8) -> np.ndarray:
    """
    DCT-based perceptual hash (64-bit bool array).
    Resize to size×size, apply row-wise + column-wise DCT via cv2.dct(),
    take top-left keep×keep coefficients, threshold at median.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct_row = cv2.dct(small)               # DCT along rows
    dct_2d  = cv2.dct(dct_row.T).T        # DCT along columns
    coeff   = dct_2d[:keep, :keep].flatten()
    return coeff > np.median(coeff)


def hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


# ─── Sector crop (Stage 0.2) ─────────────────────────────────────────────────

def sector_crop_chughtai(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Crop the Chughtai GB pane from a dual-pane frame.

    Steps:
      1. Take the right half (GB is always in the right pane for Chughtai).
      2. Grayscale threshold > 18 → content mask.
      3. Morphological open (3×3) to erase thin text, close (25×25) to merge fan.
      4. Largest connected component → bounding box crop with INSET.

    Returns (cropped_image, gb_view_string).
    """
    h, w = img.shape[:2]
    right = img[:, w // 2 :]              # right pane only

    gray    = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)

    open_k  = np.ones((3, 3),   np.uint8)
    close_k = np.ones((25, 25), np.uint8)
    opened  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
    closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        return right, "gb_dual"           # fallback: return raw right half

    areas      = stats[1:, cv2.CC_STAT_AREA]
    best       = int(np.argmax(areas)) + 1
    x, y       = stats[best, cv2.CC_STAT_LEFT],  stats[best, cv2.CC_STAT_TOP]
    bw, bh     = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]

    x1 = max(0, x + INSET);  y1 = max(0, y + INSET)
    x2 = min(right.shape[1], x + bw - INSET)
    y2 = min(right.shape[0], y + bh - INSET)

    if x2 <= x1 or y2 <= y1:
        return right, "gb_dual"

    return right[y1:y2, x1:x2], "gb_dual"


# ─── Scan new_dataset_12_july/ ───────────────────────────────────────────────

def scan_new_dataset() -> list[dict]:
    """
    Walk new_dataset_12_july/ and collect one entry per BMP/TIF image file.
    Returns list of dicts with keys:
      bmp_path, patient_folder, label, source_subfolder
    """
    entries = []
    image_exts = {".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}

    for src_dir in sorted(NEW_DATA.iterdir()):
        if not src_dir.is_dir():
            continue
        src_name = src_dir.name

        # Expect label subdirs: normal/, stone/, stones/
        for label_dir in sorted(src_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            raw_label = label_dir.name.lower()
            if raw_label not in LABEL_MAP:
                print(f"  SKIP unknown label dir: {label_dir.relative_to(ROOT)}")
                continue
            label = LABEL_MAP[raw_label]

            for patient_dir in sorted(label_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue

                for f in sorted(patient_dir.iterdir()):
                    if f.suffix.lower() not in image_exts:
                        continue
                    entries.append(
                        dict(
                            bmp_path=f,
                            patient_folder=patient_dir.name,
                            label=label,
                            source_subfolder=src_name,
                        )
                    )

    return entries


# ─── Build hashes of existing processed/ images ───────────────────────────────

def hash_existing_processed() -> dict[str, np.ndarray]:
    """
    Returns {processed_relpath: hash_array} for every PNG currently in processed/.
    Used for cross-batch duplicate detection.
    """
    hashes = {}
    for f in sorted(PROCESSED.rglob("*.png")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        hashes[rel] = phash(img)
    return hashes


# ─── Canonical output filename ────────────────────────────────────────────────

def output_path(label: str, patient_id: str, fn: str) -> Path:
    """processed/<label>/<patient_id>_<fn>.png"""
    return PROCESSED / label / f"{patient_id}_{fn}.png"


def find_free_filename(label: str, patient_id: str, fn_base: str) -> Path:
    """If <patient_id>_<fn>.png already exists, try _<fn>b, _<fn>c, … until free."""
    p = output_path(label, patient_id, fn_base)
    if not p.exists():
        return p
    for suffix in "bcdefghij":
        alt = output_path(label, patient_id, fn_base + suffix)
        if not alt.exists():
            return alt
    raise RuntimeError(f"Cannot find free filename for {patient_id}_{fn_base}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(dry_run: bool, preview_patient: str | None):
    df_existing = pd.read_csv(MANIFEST)
    existing_ids = dict(
        zip(df_existing["patient_id"], df_existing["label"])
    )  # {patient_id: label}

    print(f"Existing manifest rows : {len(df_existing)}")
    print(f"Existing processed PNGs: {len(list(PROCESSED.rglob('*.png')))}")

    entries = scan_new_dataset()
    print(f"New BMP files found    : {len(entries)}\n")

    if not entries:
        print("Nothing to process.")
        return

    # Hash all existing processed images for duplicate detection
    print("Hashing existing processed/ images …")
    existing_hashes = hash_existing_processed()   # {relpath: hash}
    print(f"  {len(existing_hashes)} hashes loaded.\n")

    # If --preview, show one crop and exit
    if preview_patient:
        hits = [e for e in entries if preview_patient.lower() in e["patient_folder"].lower()]
        if not hits:
            print(f"No patient matching '{preview_patient}' found.")
            return
        e = hits[0]
        img = cv2.imread(str(e["bmp_path"]))
        cropped, _ = sector_crop_chughtai(img)
        print(f"Preview: {e['bmp_path'].name}  →  {cropped.shape[1]}×{cropped.shape[0]}")
        cv2.imshow(f"BEFORE ({img.shape[1]}×{img.shape[0]})", img)
        cv2.imshow(f"AFTER  ({cropped.shape[1]}×{cropped.shape[0]})", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Ensure processed/ subdirs exist
    if not dry_run:
        for d in ["normal", "stones", "sludge", "unknown"]:
            (PROCESSED / d).mkdir(parents=True, exist_ok=True)

    # ── Build existing per-patient hashes for within-patient dedup ─────────────
    # Only compare frames against frames of the SAME patient_id.
    # Cross-patient comparison would flag different patients with similar anatomy
    # as duplicates (especially on the same Chughtai machine).
    existing_patient_hashes: dict[str, list[tuple[str, np.ndarray]]] = {}
    for _, row in df_existing.iterrows():
        if pd.isna(row.get("processed_relpath", "")) or row["processed_relpath"] == "":
            continue
        pid = row["patient_id"]
        rel = str(row["processed_relpath"]).replace("\\", "/")
        if rel in existing_hashes:
            existing_patient_hashes.setdefault(pid, []).append((rel, existing_hashes[rel]))

    # ── Per-frame processing ─────────────────────────────────────────────────
    new_rows      = []
    new_patient_hashes: dict[str, list[tuple[str, np.ndarray]]] = {}  # pid -> [(rel, hash)]
    conflicts     = []    # label conflicts with existing patients
    skip_count    = 0

    for e in entries:
        bmp_path   = e["bmp_path"]
        raw_folder = e["patient_folder"]
        label      = e["label"]

        slug       = normalize_patient_name(raw_folder)
        patient_id = make_patient_id(slug)
        fn_base    = frame_num(bmp_path)

        # ── Conflict check ───────────────────────────────────────────────────
        if patient_id in existing_ids and existing_ids[patient_id] != label:
            conflicts.append(
                f"  CONFLICT  {patient_id}:  existing={existing_ids[patient_id]!r}  "
                f"new={label!r}  ({raw_folder})"
            )
            # Disambiguate: append the label to avoid same patient_id for diff people
            patient_id = f"{patient_id}-{label[:3]}"

        # ── Load & crop ──────────────────────────────────────────────────────
        img = cv2.imread(str(bmp_path))
        if img is None:
            print(f"  WARN: could not read {bmp_path}, skipping")
            skip_count += 1
            continue

        cropped, gb_view = sector_crop_chughtai(img)
        h_new = phash(cropped)

        # ── Duplicate check WITHIN same patient only ──────────────────────────
        # Compare against (a) existing processed frames for this patient_id, and
        # (b) new frames already processed for this patient_id in this run.
        # Never compare across different patients — Chughtai images share anatomy
        # and machine layout, causing false-positive duplicates across patients.
        dup_of = ""
        is_dup = "no"

        all_same_patient_hashes = (
            existing_patient_hashes.get(patient_id, []) +
            new_patient_hashes.get(patient_id, [])
        )
        for rel, h_prev in all_same_patient_hashes:
            if hamming(h_new, h_prev) <= HAMMING_THRESH:
                dup_of = rel
                is_dup = "yes"
                break

        # ── Output path ──────────────────────────────────────────────────────
        out_path    = find_free_filename(label, patient_id, fn_base)
        out_relpath = str(out_path.relative_to(ROOT)).replace("\\", "/")

        if not dry_run:
            cv2.imwrite(str(out_path), cropped)
        # Always track hash for within-patient dedup, even in dry-run
        new_patient_hashes.setdefault(patient_id, []).append((out_relpath, h_new))

        orig_rel = str(bmp_path.relative_to(ROOT)).replace("\\", "/")

        row = dict(
            patient_id        = patient_id,
            source            = SOURCE,
            hospital          = HOSPITAL,
            scan_date         = "",
            patient_folder    = raw_folder,
            label             = label,
            finding_summary   = "",
            view_type         = "ultrasound",
            organ_hint        = "GB",
            original_relpath  = orig_rel,
            new_relpath       = orig_rel,
            gb_view           = gb_view,
            processed_relpath = out_relpath,
            is_duplicate      = is_dup,
            dup_of            = dup_of,
        )
        new_rows.append(row)

        action = f"[DUP -> {Path(dup_of).name}]" if is_dup == "yes" else f"-> {out_path.name}"
        tag    = "DRY" if dry_run else "   "
        print(f"  [{tag}] {patient_id:35s} {label:7s}  {action}")

    # ── Summary ──────────────────────────────────────────────────────────────
    new_frames = sum(1 for r in new_rows if r["is_duplicate"] == "no")
    dup_frames = sum(1 for r in new_rows if r["is_duplicate"] == "yes")

    print(f"\n{'-'*60}")
    print(f"  Total scanned : {len(entries)}")
    print(f"  New frames    : {new_frames}")
    print(f"  Duplicates    : {dup_frames}")
    print(f"  Skipped (bad) : {skip_count}")

    if conflicts:
        print(f"\n  !! LABEL CONFLICTS ({len(conflicts)}) -- check these manually:")
        for c in conflicts:
            print(c)
        print("  These patients were assigned a disambiguated patient_id (see above).")

    if dry_run:
        print("\n  DRY RUN -- nothing written.")
        return

    # ── Append to manifest.csv ───────────────────────────────────────────────
    df_new = pd.DataFrame(new_rows, columns=list(df_existing.columns))
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out.to_csv(MANIFEST, index=False)
    print(f"\n  manifest.csv updated: {len(df_existing)} -> {len(df_out)} rows (+{len(new_rows)})")
    print(f"\nNext steps:")
    print(f"  python strip_chughtai_overlay.py --top 0.14 --right 0.20")
    print(f"  python remove_calipers.py")
    print(f"  python mask_fan.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing any files")
    parser.add_argument("--preview", metavar="PATIENT_FOLDER",
                        help="Preview the crop for one patient folder and exit")
    args = parser.parse_args()
    main(args.dry_run, args.preview)
# Stage 0 — Data preparation methods

## Background (read this first)

**The project.** We are building an image classifier to detect **gallstones** in **abdominal
ultrasound** scans of the **gallbladder (GB)**. This document covers *data preparation* only —
no model is trained here. It is the groundwork that makes a later classifier trustworthy.

**The raw data.** Scans came from two sources in Pakistan, as folders of images:
- **Chughtai Lab** — already sorted by the lab into `Gb normal` / `Gb stones`.
- **Gulab Devi Teaching Hospital** — sorted by date/patient, *not* by diagnosis. The label for
  each patient comes from a phone photo of the typed radiology report kept in their folder.

The scans are *whole-abdomen* studies, so a patient's folder contains not just the gallbladder
but also kidney, liver, spleen, bladder, etc. — and the images have the hospital name, patient
name, MRN and date burned into the picture. Both facts are problems for machine learning.

**Key terms used below.**
- *Frame* = one ultrasound image. *Fan / sector* = the grey cone-shaped live-scan area.
- *Single-pane* = one fan in the image; *dual-pane* = two fans side-by-side (the machine shows
  two views at once). *Label leakage* = the model cheating off something other than the anatomy
  (e.g. the hospital banner, or the measurement calipers a sonographer draws on a stone).

**Where this fits.** Before Stage 0 the images were copied into a tidy, labelled folder tree
with a `manifest.csv` (see the top-level `README.md`). Stage 0 then cleans that set for training.

**Goal of Stage 0:** turn the raw, mixed scan dump into a clean, labelled, **gallbladder-only**
image set a classifier can actually learn from — and record every decision so the process is
auditable and reversible. **No original or reorganized file was deleted.** Everything is
additive: new columns in `manifest.csv` and a new `processed/` folder (train on `processed/`).

Pipeline: **0.1 GB-frame filtering → 0.2 sector-crop + de-identify → 0.3 de-duplication.**

---

## 0.1 — Gallbladder vs non-gallbladder frame filtering

**Why.** The scan folders are *whole-abdomen* studies. Many frames show kidney, liver,
spleen, bladder, prostate or pelvis — not the gallbladder. If left in, those frames inherit
the patient's report label (e.g. "stones") despite showing no gallbladder, injecting label
noise.

**Method.** All 139 ultrasound frames were rendered into labelled montages and visually
reviewed. The two sources differ:
- **Chughtai** frames are **dual-pane** (two organs side-by-side) with the organ name
  burned in ("Liver", "GB", "Right Kidney", "Spleen", "Bladder", "Prostate"). These labels
  were read directly to decide which pane(s) contain the gallbladder.
- **Gulab Devi** frames are **single-pane**, gallbladder-targeted scans; organ identity was
  confirmed by morphology (anechoic GB sac vs. bean-shaped kidney, etc.).

Two ambiguous frames were checked individually at full resolution:
- `nasir_*.tif` — confirmed **GB** (caliper reads 44.5 mm, matching the report's 4.4 cm stone).
- `yasmeen_02` — **not GB**: a 12L5 *linear-probe* superficial scan (GB uses the C6-1
  curvilinear probe). Excluded.

**Result, recorded in the `gb_view` column:**

| gb_view   | meaning                                                      | frames |
|-----------|--------------------------------------------------------------|-------:|
| `gb`      | single-pane gallbladder view                                 | 89 |
| `gb_dual` | dual-pane frame; GB is in the **right** pane (12 Chughtai + 6 Gulab) | 18 |
| `other`   | non-GB organ (kidney/liver/spleen/bladder/etc.)              | 28 |
| `uncertain`| liver-like, no GB label found (all 4 = patient HUMA)        | 4 |

→ **107 GB-containing frames** kept; 32 dropped. Notably, of Chughtai's 43 frames only
**12** show the gallbladder — the rest were survey views. One normal patient (HUMA) has no
identifiable GB frame and falls out of the trainable set.

**Dual-pane check.** Chughtai frames are dual-pane by design. Gulab frames were *assumed*
single-pane at first, which was wrong — **6 of them are dual-pane**. These were caught by an
objective test rather than by eye (visual review at thumbnail scale missed several): the
cropped output of a dual-pane frame retains near-full original width (the morphological close
bridges the two connected fans into one bounding box), so any crop with
`crop_width / original_width ≥ 0.80` is a two-pane crop. That test flagged exactly:
`gd_2025-10-03_farzana_02`, `gd_2025-10-03_panzi_02`, `gd_2025-10-03_shamim-bibi_01`,
`gd_2025-10-08_liaqat-ali_01`, `gd_2025-10-08_sobia_01`, `gd_2025-10-08_sobia_02`. All six
were reclassified `gb_dual` and re-cropped to the right pane (see 0.2). Post-fix, no crop
exceeds the width threshold. Note: column-projection / Otsu-bimodality detectors did **not**
work here because the two fans share near-field tissue at the top (no empty gutter), so the
width-of-output test is the reliable signal.

---

## 0.2 — Sector crop + de-identification

**Why.** Every frame has a burned-in border: hospital name, patient name, MRN, date/time,
and machine-parameter columns. That is **PHI** and also a **leakage shortcut** (a model can
read the banner style instead of the anatomy). We crop to just the grey ultrasound fan.

**Method** (`processed/` output, one PNG per GB frame):
1. Load grayscale; threshold at intensity > 18 to get a content mask.
2. Morphological **open** (3×3) to erase thin text strokes, then **close** (25×25) to merge
   the fan into one blob.
3. Take the **largest connected component** (the fan) via `cv2.connectedComponentsWithStats`
   and crop to its bounding box (4 px inset). Peripheral text/scale/colourbar are smaller
   components and are excluded automatically.
4. For `gb_dual` frames, the **right half** is isolated first, so the crop yields only the
   GB pane (and the Chughtai patient name, which sits in the left half, is discarded). The
   three Gulab dual-pane frames additionally have the top banner stripped (top 15%) before
   the right pane is isolated, since their banner spans the full width. (Six Gulab frames
   needed this — see the dual-pane check in 0.1.)

Crops were QC'd on contact sheets; 0 failed (no abnormal aspect ratios). Output path is in
the `processed_relpath` column.

**Limitations (important):**
- **In-fan annotations remain.** Measurement calipers / arrows the sonographer placed *on
  top of* the gallbladder cannot be removed by cropping. These are a real leakage risk
  (stone frames are more likely to carry calipers) and need inpainting/masking in a later
  step before serious training.
- Chughtai `gb_dual` crops keep a thin top header strip (the full-width Toshiba banner).
  It contains machine text and date — not the patient name — but should be trimmed later.

---

## 0.3 — De-duplication

**Why.** Patients have 1–6 frames, often captured seconds apart and nearly identical.
If near-duplicates land on both sides of a train/test split, they leak and inflate scores.

**Method.** A DCT-based **perceptual hash** (32×32 → 8×8 DCT, median-thresholded, 64-bit)
was computed on each cropped frame. Within each patient, frames within **Hamming distance ≤ 6**
of an already-kept frame are flagged. Flagged pairs were spot-checked side-by-side and
confirmed near-identical.

**Result:** 21 frames flagged (`is_duplicate = yes`, `dup_of` names the kept twin).
They are **flagged, not deleted** — drop them at training time if you want one frame/event.
→ **86 unique GB frames.**

---

## Final Stage-0 dataset

| stage | frames |
|-------|-------:|
| raw ultrasound frames | 139 |
| GB-containing (after 0.1) | 107 |
| unique GB (after 0.3) | **86** |

**Unique GB frames / patients by label**

| label   | frames | patients |
|---------|-------:|---------:|
| stones  | 64 | 48 |
| normal  | 12 | 11 |
| sludge  |  1 |  1 |
| unknown |  9 |  7 |

**Binary stones-vs-normal trainable set:** 76 unique frames · 59 patients (48 stones / 11 normal).
With duplicates included in train: 95 frames (83 stones / 12 normal). All 19 duplicates are in the stones class; normal has zero duplicates.

## Properties of processed/ output

**Image sizes:** 28 unique dimensions across 107 files. No two fans are the same size — each is a per-frame bounding box crop. The two sources produce systematically different aspect ratios:
- Chughtai: portrait (~472×577, ratio ≈ 0.82:1) — all `gb_dual` crops
- Gulab Devi: landscape (~533×364, ratio ≈ 1.46:1) — mostly `gb` single-pane crops

**Pixel format:** all PNGs are `Format24bppRgb` (3-channel) despite containing only grayscale data. R=G=B in every pixel. This is an artifact of how OpenCV writes PNGs from a grayscale array by default. No conversion needed at training time — pretrained models accept 3-channel input.

**Site distribution in binary set:**

| | Chughtai | Gulab Devi |
|---|---|---|
| normal | 8 (67%) | 4 (33%) |
| stones | 4 (5%) | 79 (95%) |

This is a severe site confound — the label is strongly correlated with the hospital source. A model trained on mixed data can learn Chughtai-style → normal, Gulab Devi-style → stones without inspecting the gallbladder. Cross-site validation is mandatory before any accuracy claim.

## New manifest columns added in Stage 0
`gb_view` · `processed_relpath` · `is_duplicate` · `dup_of`

## What's still open (next stages)
1. **In-fan caliper/annotation removal** (leakage) — biggest remaining data-quality task. Stone frames are more likely to carry calipers than normal frames, creating a direct leakage shortcut.
2. **Site confound** (Chughtai≈normal, Gulab≈stones) — design cross-site validation splits; evaluate Chughtai-only → Gulab Devi and vice versa.
3. **Chughtai dual-pane header** — `gb_dual` crops retain a thin top Toshiba banner (machine text, no patient name); trim before training.
4. **More normal cases** — only 11 normal patients vs 48 stones; external data or heavy augmentation needed.
5. Label the 7 `unknown` patients (needs their reports) and decide on `sludge`.

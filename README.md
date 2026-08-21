# Gallstone Ultrasound Dataset

Cleaned and labelled gallbladder (GB) ultrasound dataset for gallstone detection,
consolidated from two Pakistani sources: **Chughtai Lab** and **Gulab Devi Teaching Hospital**.

> This folder is a **non-destructive reorganized copy**. The original `Desktop/dataset`
> tree is untouched. Every file here maps back to its original via `manifest.csv`.

## Layout

```
gallstone_dataset/
├── manifest.csv          # one row per image (the source of truth)
├── README.md
├── docs/
│   └── STAGE0_METHODS.md # how the data was filtered/cropped/deduped + decisions
├── ultrasound/           # RAW scan frames as captured (all 139, by label)
│   ├── stones/  normal/  sludge/  unknown/
├── processed/            # ← Stage 0 output: GB-only, sector-cropped, de-identified
│   ├── stones/           # 64 unique-ish GB crops / 48 patients
│   ├── normal/           # 12 / 11 patients
│   ├── sludge/  unknown/
└── reports/              # photos of the radiology reports (the ground-truth source)
    └── orphan/           # report photos with no matching scan folder
```

**Train on `processed/`, not `ultrasound/`.** `ultrasound/` is the raw capture (includes
kidney/liver/other-organ frames and PHI overlays). `processed/` contains only gallbladder
views, cropped to the scan fan with the patient banner removed. See `docs/STAGE0_METHODS.md`.

## Labels

| label   | meaning                                                        |
|---------|----------------------------------------------------------------|
| stones  | report states calculus / calculi / stone(s) in the gallbladder |
| normal  | report states GB normal, no calculus                           |
| sludge  | report states sludge only, **no discrete stone**               |
| unknown | no report photo and no reliable label hint — needs review      |

**How labels were derived**
- **Chughtai Lab** images were pre-sorted by the lab into `Gb normal` / `Gb stones` folders — that folder is the label.
- **Gulab Devi** folders were *not* pre-sorted. Each patient folder usually contains a phone
  photo of the typed radiology report. Every one of those 52 report photos was read and the
  gallbladder finding transcribed into `finding_summary` in the manifest. That finding is the label.
- A few Gulab Devi folders have **no report photo**; labelled `unknown` unless the folder name
  itself said "normal gb" (then `normal`).

## File naming

`<src>_<scan-date>_<patient>_<nn>.<ext>`  e.g. `gd_2025-11-03_abdullah_01.bmp`
- `src`: `ch` = Chughtai, `gd` = Gulab Devi
- Chughtai has no scan date, so it is omitted: `ch_arshad_01.bmp`
- Report photos: `<src>_<date>_<patient>_report.<ext>`

## manifest.csv columns

`patient_id, source, hospital, scan_date, patient_folder, label, finding_summary,
view_type (ultrasound|report), organ_hint, gb_view, original_relpath, new_relpath,
processed_relpath, is_duplicate, dup_of`

Stage-0 columns: `gb_view` (gb | gb_dual | other | uncertain | na) marks which frames
actually show the gallbladder; `processed_relpath` points to the cleaned crop;
`is_duplicate`/`dup_of` flag near-identical frames within a patient.

## Stage 0 result (see docs/STAGE0_METHODS.md)

139 raw frames → **107 GB frames** (32 non-GB dropped) → **86 unique GB crops** (21
near-duplicates flagged). Binary stones-vs-normal trainable set: **76 frames / 59 patients
(48 stones, 11 normal)**.

## ⚠️ Important caveats before training

1. **Not every raw frame shows the gallbladder** — but this has now been resolved in Stage 0.
   The `ultrasound/` folder still contains the kidney/liver/other-organ frames; the cleaned
   gallbladder-only set is in `processed/` (use the `gb_view` column to filter). Remaining
   data-quality risk: **in-fan measurement calipers were not removed** and are a leakage
   shortcut — addressing that is the next step.
2. **Patient-level splits only.** Some patients recur across dates (e.g. two "Abdullah", two
   "Panzi"). Split train/val/test by `patient_id`, never mix the same patient across splits.
3. **Class imbalance.** 83 stone frames vs 12 normal frames (≈6.9:1). Augment and use weighted loss.
4. **`sludge` and `unknown`** are excluded from a binary stones-vs-normal task by default.
5. **PHI.** Scan overlays and report photos contain patient names, MRNs and dates.
   De-identify (crop the overlay / blur) before any external sharing.
6. **Site confound.** 79/83 stone frames are Gulab Devi; 8/12 normal frames are Chughtai.
   A model can learn hospital style instead of anatomy. Cross-site validation is mandatory.

## Counts

- Total files: 191 (139 ultrasound: 136 bmp + 3 tif; 52 report photos)
- Ultrasound patients: 48 stones · 12 normal · 1 sludge · 7 unknown
- Processed PNGs: 83 stones · 12 normal · 1 sludge · 11 unknown (107 total; 19 stone frames are near-duplicates)

## Training decisions

These decisions are fixed and should not be revisited without good reason.

**Splitting:** always split by `patient_id` (59 unique patients in the binary set). Never split by filename.

**Duplicate handling (Strategy 3):** keep all frames in the train split; drop `is_duplicate = yes` frames from val and test.

```python
binary = df[df["label"].isin(["stones", "normal"]) & df["processed_relpath"].notna()]
train_df = binary[binary["patient_id"].isin(train_patients)]
val_df   = binary[binary["patient_id"].isin(val_patients)  & (binary["is_duplicate"] != "yes")]
test_df  = binary[binary["patient_id"].isin(test_patients) & (binary["is_duplicate"] != "yes")]
```

**Preprocessing (all splits):** pad to square → resize to 224×224 → normalize with ImageNet mean/std. Pad with black pixels to preserve aspect ratio before resize — Chughtai crops are portrait (~472×577) and Gulab Devi are landscape (~533×364); direct resize without padding distorts anatomy.

**Augmentation (train only):** horizontal flip, brightness/contrast jitter, Gaussian noise, rotation ±10°, random crop+resize. Do not vertical flip (acoustic shadow below stone is a diagnostic cue). Apply 6–7 variants per normal frame vs 1–2 per stone frame to counteract the 6.9:1 imbalance. Combine with weighted loss `{stones: 1, normal: 6.9}`.

**Model:** DINOv2 frozen (ViT-S/14) + linear/MLP head is the primary approach. EfficientNet-B0 partial fine-tune as baseline. Do not train any model from scratch — dataset is too small.

**Evaluation must include cross-site validation** — Chughtai-only → Gulab Devi and vice versa. Mixed-split AUC alone is not sufficient. High AUC before caliper removal is not trustworthy.

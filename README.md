# Gallstone Ultrasound Dataset

A cleaned and labelled gallbladder ultrasound dataset for binary gallstone detection
(stones vs. normal), consolidated from two Pakistani sources: **Chughtai Lab** and
**Gulab Devi Teaching Hospital**.

Data cleaning is complete. Training code is in place and both baseline models have been run.

**Last updated:** 2026-08-21

---

## Quick facts

| | |
|---|---|
| Trainable frames | **194** (159 non-duplicate) |
| Patients | **136** |
| Class balance | 100 stones / 94 normal (≈1.06 : 1) |
| Train on | **`stage3/`** |
| Source of truth | `manifest.csv` (290 rows, one per image) |
| Best result | DINOv2, mixed-split AUC **0.952** |

---

## Where to read next

This README is an overview. The detail lives in four documents:

| Document | Covers |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Full working reference: constraints, methods, all results, open tasks |
| [`Data_Pipeline_Explained.md`](Data_Pipeline_Explained.md) | Step-by-step walkthrough of raw scans to trainable data, written for someone with no context |
| [`docs/STAGE0_METHODS.md`](docs/STAGE0_METHODS.md) | Stage 0 methodology and the decisions behind it |
| [`Open_Source_Datasets.md`](Open_Source_Datasets.md) | Plan for sourcing additional public data, with author contacts |

---

## Layout

```
gallstone_dataset/
├── manifest.csv               # one row per image (the source of truth, 290 rows)
│
├── dataset/                   # original donated dumps, exactly as received (191 files)
├── ultrasound/                # raw frames, renamed and label-sorted (139 files)
├── reports/                   # photos of radiology reports (52) - label evidence only
│   └── orphan/                # reports with no matching scan
│
├── processed/                 # Stage 0: fan-cropped, de-identified      (206 PNG)
├── stage1/                    # Stage 1: Chughtai text strips removed    (206 PNG)
├── stage2/                    # Stage 2: calipers inpainted              (206 PNG)
├── stage3/                    # Stage 3: fan-masked  <-- TRAIN ON THIS   (194 PNG)
│
├── fan_masks/                 # per-size fan polygons used by Stage 3
├── unique_sizes/              # one sample per image size, for annotation
│
├── dataset.py                 # loading, transforms, patient-level splits
├── train.py                   # training loop, mixed + cross-site evaluation
├── evaluate.py                # metrics and results table
├── ingest_new_dataset.py      # Stage 0 for newly donated batches
├── strip_chughtai_overlay.py  # processed/ -> stage1/
├── remove_calipers.py         # stage1/   -> stage2/
├── mask_fan.py                # stage2/   -> stage3/  (automatic)
├── create_fan_masks.py        # stage2/   -> stage3/  (manual polygons)
├── convert_outlines.py        # VIA JSON  -> fan_masks/
├── export_unique_sizes.py     # helper for the manual masking route
└── visualize_pipeline.py      # visual comparison across stages
```

**Train on `stage3/`.** Every earlier folder is kept for reference and is not safe to train on.
`ultrasound/` and `dataset/` still contain other-organ frames and burned-in patient details.

---

## The four cleaning stages

Each stage writes to a new folder rather than overwriting, so nothing is destructive and any
step can be inspected or redone.

| Stage | Output | What it removes |
|---|---|---|
| 0 | `processed/` | Non-gallbladder frames, patient name and MRN (via fan crop), duplicate flagging |
| 1 | `stage1/` | Chughtai text strips: top 14%, right 20% (Gulab Devi copied unchanged) |
| 2 | `stage2/` | Measurement calipers and organ labels, via HSV threshold and TELEA inpainting |
| 3 | `stage3/` | Everything outside the ultrasound fan |

Stage 2 matters most. Sonographers place measurement calipers on stones and not on normals,
which makes them a near-perfect giveaway that has nothing to do with anatomy. Any high score
recorded before that step was removed should not be trusted.

Full detail in [`Data_Pipeline_Explained.md`](Data_Pipeline_Explained.md).

---

## Labels

| label | meaning | frames |
|---|---|---:|
| `stones` | report states calculus / calculi / stone(s) in the gallbladder | 100 |
| `normal` | report states GB normal, no calculus | 94 |
| `sludge` | report states sludge only, no discrete stone | excluded |
| `unknown` | no report photo and no reliable label hint | excluded |

**How labels were derived**

- **Chughtai Lab** images arrived pre-sorted by the lab into `Gb normal` and `Gb stones`
  folders. The folder name is the label.
- **Gulab Devi** folders were sorted by date and patient, not by diagnosis. Most patient
  folders contain a phone photo of the typed radiology report. All 52 of those photos were
  read by hand and the gallbladder finding transcribed into `finding_summary`. That finding
  is the label.
- Folders with no report photo are labelled `unknown`, unless the folder name itself said
  "normal gb".

The 52 report photos are label evidence, not training data. They carry no `processed_relpath`
and can never be loaded by the training code.

---

## manifest.csv

15 columns, one row per image:

```
patient_id, source, hospital, scan_date, patient_folder, label, finding_summary,
view_type, organ_hint, original_relpath, new_relpath, gb_view,
processed_relpath, is_duplicate, dup_of
```

The three path columns form a chain tracking each image through the pipeline:
`original_relpath` points into `dataset/`, `new_relpath` into `ultrasound/` or `reports/`,
and `processed_relpath` into `processed/` (and by substitution into `stage1/` through
`stage3/`, since filenames are preserved).

**A blank `processed_relpath` means the image was excluded from training.** Nothing is ever
deleted, only excluded, so you can always ask what was dropped and why.

`gb_view` records the Stage 0 filtering decision:

| value | count | reaches training |
|---|---:|---|
| `gb_dual` | 117 | yes |
| `gb` | 89 | yes |
| `na` | 52 | no (report photos) |
| `other` | 28 | no (kidney, liver, other organs) |
| `uncertain` | 4 | no |

Note that path columns contain a mix of `\` and `/` separators. `dataset.py` normalises this,
but any new script reading them needs the same guard.

---

## File naming

`<src>_<scan-date>_<patient>_<nn>.<ext>` for example `gd_2025-11-03_patientname_01.bmp`

- `src`: `ch` = Chughtai, `gd` = Gulab Devi
- Chughtai has no scan date, so it is omitted: `ch_patientname_01.bmp`
- Report photos: `<src>_<date>_<patient>_report.<ext>`

Filenames are preserved across `processed/`, `stage1/`, `stage2/`, and `stage3/`, so any frame
can be traced through every stage by swapping the folder name.

---

## Hard constraints

**1. Patient-level splits only.** A `patient_id` must never appear in more than one of
train, validation, or test. Some first names recur across different dates and belong to
different people, so they must not be merged. Never split by filename.

**2. Duplicates: keep in train, drop from val and test.** 35 of the 194 binary frames are
flagged `is_duplicate = yes`. They carry real training signal but would inflate scores if
they sat opposite their twin in evaluation.

```python
binary = df[df["label"].isin(["stones", "normal"]) & df["processed_relpath"].notna()]
train_df = binary[binary["patient_id"].isin(train_patients)]
val_df   = binary[binary["patient_id"].isin(val_patients)  & (binary["is_duplicate"] != "yes")]
test_df  = binary[binary["patient_id"].isin(test_patients) & (binary["is_duplicate"] != "yes")]
```

**3. No class weighting.** The dataset is near-balanced at 100 stones to 94 normal, so
`CLASS_WEIGHTS` is `{stones: 1.0, normal: 1.0}`. **Do not reintroduce the old 6.9 weight.**
It was correct for the original 83:12 dataset and became actively harmful once the data
rebalanced, suppressing sensitivity from 0.786 down to 0.286.

**4. Cross-site validation is mandatory.** Mixed-split AUC alone is not sufficient evidence
of anything. See the site confound below.

**5. PHI.** `dataset/`, `ultrasound/`, `reports/`, and `manifest.csv` contain patient names,
MRNs, and dates. `dataset/` additionally carries names in its folder names. Do not share any
of these externally without de-identification. `stage3/` is safe: fan-masked with all
burned-in text removed.

---

## The site confound

This is the central weakness of the dataset and the reason cross-site testing is required.

| | Chughtai | Gulab Devi |
|---|---:|---:|
| normal | 90 (96%) | 4 (4%) |
| stones | 21 (21%) | 79 (79%) |

Normal cases come almost entirely from one hospital and stone cases almost entirely from the
other. The two sites use different machines, so their images differ in texture and grain even
after every overlay has been removed. A model can therefore score well by recognising hospital
style rather than anatomy.

No amount of image processing fixes this. It needs more normal cases from Gulab Devi, or a
fresh dataset where both classes came off the same machine. See
[`Open_Source_Datasets.md`](Open_Source_Datasets.md).

---

## Training

```bash
pip install torch torchvision scikit-learn pandas pillow opencv-python

python train.py --model efficientnet --data-root .
python train.py --model dinov2       --data-root .
```

Key flags: `--epochs` (default 50), `--lr` (1e-3), `--batch-size` (16), `--seed` (42),
`--num-workers` (set to 0 on Windows if the DataLoader errors).

**Preprocessing, applied to all splits:** pad the shorter side with black to square, resize to
224x224, convert to tensor, normalize with ImageNet statistics. Padding before resizing matters
because Chughtai crops are portrait (~472x577) and Gulab Devi are landscape (~533x364);
resizing without padding first distorts the anatomy.

**Augmentation, training split only:** horizontal flip, brightness and contrast jitter,
Gaussian noise, rotation up to 10 degrees, random crop and resize. Applied uniformly to both
classes, since the dataset is balanced.

Never vertical flip. A stone's acoustic shadow falls below it, and flipping puts the shadow
above the stone, which is physically impossible.

**Models:** DINOv2 (ViT-S/14) frozen with a linear head is the primary approach.
EfficientNet-B0 with partial fine-tuning is the baseline. Do not train from scratch, since
194 frames is nowhere near enough.

---

## Results

Best run: DINOv2, frozen backbone, evaluated on a held-out mixed-hospital test set of 26
images (14 stones, 12 normal). Threshold 0.234, chosen on the validation set and then applied
to test.

| Metric | Mixed split | GD to CH | CH to GD |
|---|---:|---:|---:|
| AUC-ROC | **0.952** | 0.475 | 0.358 |
| Sensitivity | **0.929** | 1.000 | 0.850 |
| Specificity | 0.500 | 0.051 | 0.000 |
| F1 | 0.788 | 0.299 | 0.887 |

Three caveats belong with any quotation of these numbers:

1. **The test set is 26 images.** One image flipping moves accuracy by four points. Treat
   0.952 as encouraging, not proven.
2. **Cross-site performance fails**, at or below random in both directions. The model learned
   hospital style, not anatomy. This is a data problem, not a model problem.
3. **Training ran too long.** Validation AUC peaked around epoch 7 and declined thereafter.
   The best checkpoint was correctly restored, so the result stands, but future runs should
   use early stopping.

Full result tables for both models, including the earlier runs, are in [`CLAUDE.md`](CLAUDE.md).

---

## Open tasks

1. **More normal cases from Gulab Devi.** Only 3 normal GD patients (4 frames) exist.
   Cross-site AUC will stay weak until this is fixed. Target: 15 to 20 more.
2. **GradCAM verification.** Confirm the model attends to gallbladder anatomy rather than
   residual artefacts. The caliper removal is believed to have worked, not yet proven.
3. **Shorter training runs.** Reduce to roughly 20 epochs or add early stopping with
   patience 5.
4. **Label the unknown patients.** 7 patients have no radiology report; sourcing those
   reports would add about 9 frames.
5. **External data.** See [`Open_Source_Datasets.md`](Open_Source_Datasets.md) for ranked
   sources and author contacts.

---

## Known gaps

**The raw archive is incomplete.** `dataset/` holds the original donation only. The 99
Chughtai frames ingested on 2026-07-12 arrived via a separate dump that was cropped straight
into `processed/` without archiving the originals, and that dump is gone. For those 99 frames
the earliest surviving copy is the already-cropped PNG, so Stage 0 cannot be re-run differently
on them.

**Stage 2 over-removes slightly.** The HSV threshold that finds calipers also catches some
genuinely bright tissue, which then gets inpainted. It reliably removes the calipers, which
was the priority, at the cost of lightly smoothing some real anatomy. Manual masking remains
the fallback if GradCAM suggests a problem.
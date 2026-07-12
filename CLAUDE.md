# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A curated medical imaging dataset for binary gallstone classification (stones vs. normal) from abdominal ultrasound scans. Two Pakistani sources: **Chughtai Lab** (`ch_`) and **Gulab Devi Teaching Hospital** (`gd_`). Data cleaning is complete (`stage3/` is the training source). Training code exists — see `dataset.py`, `train.py`, `evaluate.py`.

## Source of truth

`manifest.csv` is the authoritative index. One row per image (ultrasound frame or report photo). Key columns:

| column | meaning |
|--------|---------|
| `patient_id` | use this for train/val/test splitting — never split by filename |
| `label` | `stones` / `normal` / `sludge` / `unknown` |
| `gb_view` | `gb` / `gb_dual` / `other` / `uncertain` — only `gb` and `gb_dual` are in `processed/` |
| `processed_relpath` | path to the cleaned PNG crop (blank if frame was excluded) |
| `is_duplicate` / `dup_of` | near-identical frames flagged by perceptual hash; drop from val/test, keep in train |
| `finding_summary` | raw text from the radiology report (Gulab Devi only) |

## How labels were assigned

- **Chughtai Lab** — pre-sorted by the lab into `Gb normal` / `Gb stones` folders. The folder name is the label.
- **Gulab Devi Teaching Hospital** — sorted by date/patient, not by diagnosis. Each patient folder usually contains a phone photo of the typed radiology report (`reports/`). Every report was read and the gallbladder finding transcribed into `finding_summary` in the manifest. That finding is the label.
- Folders with no report photo are labelled `unknown` unless the folder name explicitly said "normal gb".

## Stage 0 — Data preparation (before any training)

Full methodology in `docs/STAGE0_METHODS.md`. Three steps:

### 0.1 — GB-frame filtering
Raw folders are whole-abdomen studies (kidney, liver, spleen, bladder, etc.). All 139 frames were reviewed visually on contact sheets. Chughtai frames are dual-pane with organ names burned in — read directly. Gulab Devi frames are single-pane — confirmed by morphology. Result: 107 GB-containing frames kept, 32 dropped (`gb_view` column).

**Dual-pane detection:** 6 Gulab Devi frames were incorrectly assumed single-pane. Caught by objective test: if `crop_width / original_width ≥ 0.80`, the crop spans two fans. Those 6 were reclassified `gb_dual` and re-cropped to the right pane.

### 0.2 — Sector crop + de-identification (`processed/`)
Every frame has PHI burned in (patient name, MRN, date). Method: grayscale threshold → morphological open/close → largest connected component (the fan) → crop to its bounding box. For `gb_dual` frames, right half isolated first. Output: `processed/`, one PNG per GB frame.

**Limitation:** in-fan measurement calipers survive this step — they sit inside the fan boundary and can't be removed by cropping. Fixed in Stages 1–2.

### 0.3 — De-duplication
DCT-based perceptual hash (32×32 → 8×8 DCT, 64-bit) computed per frame. Within each patient, frames within Hamming distance ≤ 6 are flagged as near-duplicates (`is_duplicate = yes`, `dup_of` names the kept twin). **Flagged, not deleted** — 21 duplicates, all in the stones class. Result: 86 unique GB frames.

**Stage 0 counts:**

| stage | frames |
|-------|-------:|
| Raw ultrasound | 139 |
| GB-containing (after 0.1) | 107 |
| Unique GB (after 0.3) | 86 |

## Site distribution (binary set)

After ingesting `new_dataset_12_july/` (2026-07-12):

| | Chughtai | Gulab Devi |
|---|---|---|
| normal | 90 (96%) | 4 (4%) |
| stones | 21 (21%) | 79 (79%) |

**Total: 194 frames · 136 patients · 100 stones · 94 normal** (159 non-duplicate).

The site confound persists but is reduced on the stones side: Chughtai now contributes 21 stone patients (was 3). Normal is still almost entirely Chughtai. Cross-site validation remains mandatory.

## Which data to use

- **Train on `stage3/`** — cleanest version: text strips removed, calipers inpainted, fan-masked (everything outside the ultrasound dome blacked out).
- `stage2/` = calipers inpainted, text strips removed (intermediate, kept for reference).
- `stage1/` = Chughtai text strips removed only (intermediate, kept for reference).
- `processed/` = Stage 0 output, untouched (kept for reference).
- **Do not train on `ultrasound/`** — raw BMPs/TIFs that include kidney/liver/other-organ frames and burned-in PHI.
- **Binary task filter**: `label IN ('stones', 'normal')` → 194 frames, 136 patients (159 non-duplicate).
- `sludge` and `unknown` are excluded from binary classification by default.

## Splitting and duplication strategy

**Always split by `patient_id` first**, then decide which frames to include per split. Never split by filename — the same patient's frames must stay in one split.

**Duplicate handling (Strategy 3):** keep all frames in the train split for maximum training signal; drop duplicates from val and test to keep evaluation clean.

```python
df = pd.read_csv("manifest.csv")
binary = df[df["label"].isin(["stones", "normal"]) & df["processed_relpath"].notna()]

train_df = binary[binary["patient_id"].isin(train_patients)]
val_df   = binary[binary["patient_id"].isin(val_patients)  & (binary["is_duplicate"] != "yes")]
test_df  = binary[binary["patient_id"].isin(test_patients) & (binary["is_duplicate"] != "yes")]
```

`train_patients`, `val_patients`, `test_patients` must be disjoint sets derived from `binary["patient_id"].unique()`.

## Augmentation strategy

Apply augmentation only to `stage3/` PNGs at training time (never to val/test).

**Always apply:**
- Horizontal flip — probe placement varies left/right; physically valid
- Brightness / contrast jitter — ultrasound gain settings differ by machine and sonographer
- Gaussian noise — simulates realistic speckle noise inherent to ultrasound
- Rotation ±10° — small probe angle variation; keep small to avoid black corners from fan crop
- Random crop + resize — simulates different sonographer framing/zoom

**Use with care:**
- Mild elastic deformation — low `alpha` only; aggressive deformation distorts stone shape and acoustic shadow
- Gamma correction — valid alternative to brightness jitter; mimics TGC curve differences

**Avoid:**
- Vertical flip — acoustic shadow (dark cone below a stone) is a primary diagnostic cue; flipping puts it above the stone, which is physically impossible
- Large rotations (>15°) — introduces black corners from the fan crop boundary
- Color jitter / channel operations — images are grayscale

**Class imbalance:** after the 2026-07-12 ingestion the dataset is near-balanced (100 stones vs 94 normal ≈ 1.06:1). Standard uniform augmentation and equal loss weighting are appropriate. The old 6.9:1 ratio and the heavy per-class augmentation strategy no longer apply — do not use `class_weight = {stones: 1, normal: 6.9}` with this dataset.

## Preprocessing pipeline

All images in `stage3/` must go through these steps before training. Val/test use only these steps — no augmentation.

```python
transforms.Compose([
    transforms.Pad(...),           # pad shorter side with black to make square
    transforms.Resize((224, 224)), # uniform resize after squaring
    transforms.ToTensor(),         # 0–255 → 0.0–1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Why pad before resize:** Two sources produce different aspect ratios — Chughtai is portrait (~472×577) and Gulab Devi is landscape (~533×364). Direct resize to 224×224 distorts anatomy (round GB becomes oval, acoustic shadows tilt). Padding the shorter side with black pixels squares the image first so resize is uniform. Black padding is safe because fan crops already have black edges.

**Why 224×224:** Standard input size for pretrained ImageNet models (ResNet, EfficientNet). Using a different size requires re-learning spatial weights from scratch.

**Why keep RGB, not convert to grayscale:** Images are stored as `Format24bppRgb` (3 channels) with R=G=B — grayscale data in an RGB container. Pretrained models expect 3 channels; keeping them as-is allows full use of pretrained weights. Converting to 1-channel grayscale would require reinitializing the first conv layer, losing pretrained benefit.

**Why ImageNet normalization:** Pretrained weights were learned with inputs normalized to these stats. Using them keeps activations in the expected range. Since all 3 channels are identical (grayscale), channel stats will be the same — the approximation still works in practice. Alternative: compute mean/std from the training split only for dataset-specific normalization, but never from the full dataset (leaks val/test distribution).

**Note on 28 unique image sizes:** Every image in `processed/` has a different size due to per-frame fan crop bounding boxes. Padding + resize handles this automatically.

## Remaining text overlay removal

Two types of text remain in `processed/` images after Stage 0 cropping. Both must be removed before training.

### Fix 1 — Chughtai text strips (simple crop) ✓ verified

All Chughtai images retain two burned-in strips from the Toshiba machine:
- **Top strip** (~14% of height): partial patient name, date/time, "Precision APure", dots row
- **Right strip** (~20% of width): depth scale numbers, organ label (`GB`), machine parameters (fps, gain, probe)

These are in fixed positions on every Chughtai image (same machine, same layout). Removal is a fixed percentage crop — no ML needed.

```python
def strip_chughtai_overlay(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    top   = int(h * 0.14)
    right = int(w * 0.20)
    return img[top:, :w - right]
```

Run via `strip_chughtai_overlay.py` — outputs to `stage1/` (non-destructive, `processed/` unchanged).

**Remaining after crop:** `GB` organ label and depth numbers partially inside the fan boundary — removed by Fix 2 inpainting. `T` marker in corners — removed by Fix 3 fan masking.

**Why this matters:** Chughtai = 8/12 normal frames. The machine parameter strip is a site confound signal — the model can learn "right-column text = normal" without looking at anatomy.

### Fix 2 — Caliper / text overlay removal (inpainting) ✓ done

Removed via OpenCV HSV threshold + TELEA inpainting on **all frames** (not just stones):
- **Stone frames:** measurement calipers (`+` markers, dashed lines) — direct leakage shortcut
- **Normal frames:** `GB` organ label, depth scale numbers (`5`, `10`), colored marker lines burned in by the Chughtai machine — site confound signals

Run via `remove_calipers.py` — reads from `stage1/`, outputs to `stage2/` (non-destructive).

**If results look suspicious after training** (model not generalising): fall back to manual masking — draw masks over calipers for all 83 stone frames and re-run inpainting.

### Fix 3 — Fan masking ✓ done

Blacks out everything outside the ultrasound dome — removes `T` marker, residual depth numbers, dot row, and any other peripheral machine artifacts in one step.

**Method:**
1. One representative image per unique size exported to `unique_sizes/` via `export_unique_sizes.py`.
2. All 28 polygons drawn in **VGG Image Annotator (VIA)** and saved as `Unique Image Outlines (3).json`.
3. `convert_outlines.py` reads the VIA JSON and writes `fan_masks/mask_{W}x{H}.png` for each size.
4. `create_fan_masks.py --apply` stamps the correct mask onto every image in `stage2/`, writing to `stage3/`.

```
python convert_outlines.py          # VIA JSON → fan_masks/*.png
python create_fan_masks.py --apply  # stage2/ → stage3/
```

**Why VIA over automatic detection:** automatic fan detection (thresholding + largest connected component) can absorb the `T` marker if it touches the fan edge. Manual annotation in VIA takes ~15 minutes for 28 sizes and guarantees correctness.

**What remains inside the fan after masking:** none — GB label and depth numbers removed by Fix 2 inpainting before masking is applied.

## Training code

Three files implement the full pipeline:

| file | purpose |
|------|---------|
| `dataset.py` | `GallstoneDataset`, transforms, patient-level splits, `WeightedRandomSampler`, per-sample loss weights |
| `train.py` | model setup, training loop, mixed + cross-site evaluation, saves checkpoints and `results_<model>.json` |
| `evaluate.py` | `compute_metrics` (AUC, sensitivity, specificity, F1, accuracy, Spec@Sens≥90%), `print_results_table` |

**Run (Google Colab):**
```
!pip install torch torchvision scikit-learn pandas pillow
!python train.py --model efficientnet --data-root /path/to/gallstone_dataset
!python train.py --model dinov2       --data-root /path/to/gallstone_dataset
```

Key flags: `--epochs` (default 30), `--lr` (default 1e-3), `--batch-size` (default 16), `--seed` (default 42), `--num-workers` (set to 0 on Windows if DataLoader errors).

**Known issue — Windows path separators in manifest.csv:** `processed_relpath` values use `\` (e.g. `processed\normal\ch_adnan_03.png`). On Linux/Colab, `Path(...).relative_to("processed")` fails. Fix: call `.replace("\\", "/")` on the relpath string before passing to `Path`. Both occurrences in `dataset.py` (file-exists check and `__getitem__`) need this fix.

## Model strategy

194 frames exist (159 non-duplicate). Overfitting is still the primary risk — do not train from scratch.

**The core idea:** use a large pretrained model as a frozen feature extractor. Only the small classifier head trains. This way, the backbone's weights (learned from millions of images) stay intact, and there are very few parameters left to overfit on 95 frames.

**Step 1 — Baseline: EfficientNet-B0**
- Freeze the entire backbone, train only the final classification head
- Once the head converges (~10 epochs), unfreeze the last 2 blocks only and fine-tune at a low learning rate
- Purpose: validate the full pipeline quickly before trying anything more complex

**Step 2 — Primary: DINOv2 frozen + head**
- DINOv2 (Meta's self-supervised ViT) produces richer, more transferable features than supervised CNNs
- Backbone is fully frozen — zero gradient flows through it
- Only a linear layer or tiny 2-layer MLP trains on top
- Consistently outperforms fine-tuned CNNs on small medical imaging datasets

```python
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
backbone.requires_grad_(False)           # frozen — nothing trains here
head = nn.Linear(384, 1)                 # only this trains
```

**What to avoid:**
- Training any model from scratch — needs 10k+ images minimum
- Fine-tuning large backbones fully (ResNet-50+, full ViT) — too many free parameters for 194 frames
- Reporting only mixed-split AUC — must also run cross-site validation (Chughtai → Gulab Devi and vice versa)
- Trusting high AUC before caliper leakage is fixed — the model may be reading calipers, not anatomy

## Evaluation metrics

**Primary metric: AUC-ROC.** Threshold-independent and imbalance-robust. The headline number for all comparisons.

**Report at a clinically chosen threshold** (e.g. sensitivity ≥ 0.90):
- **Sensitivity** — of all stone cases, how many did the model catch. A missed stone is more dangerous than a false alarm.
- **Specificity** — of all normal cases, how many did the model correctly clear.

**Also report (with disclaimer):**
- **F1 score** — handles imbalance better than accuracy but is threshold-dependent; use it as a secondary check, not a headline number.
- **Accuracy** — reported for completeness only. With the expanded dataset (100 stones vs 94 normal) a "always stones" baseline scores 51.5% — accuracy is now a meaningful signal, but AUC remains the primary metric.

**Minimum reporting table:**

| Metric | Mixed split | Chughtai → Gulab Devi | Gulab Devi → Chughtai |
|---|---|---|---|
| AUC-ROC | | | |
| Sensitivity | | | |
| Specificity | | | |
| F1 | | | |
| Accuracy ⚠️ | | | |

The cross-site columns are what reveal whether the model generalised to anatomy or memorised hospital style.

## Hard constraints

1. **Patient-level splits** — `patient_id` must not appear in more than one of train/val/test. Some names recur across dates (e.g. two "Abdullah", two "Panzi").
2. **Caliper leakage** — addressed in `stage2/` via OpenCV inpainting. Verify post-training with GradCAM — if activations fire on caliper locations, fall back to manual masking.
3. **Class imbalance** — 100 stones vs 94 normal (≈1.06:1 after 2026-07-12 ingestion). No special rebalancing needed. If retraining on the original smaller dataset, use `class_weight ≈ {stones: 1, normal: 6.9}`.
4. **Site confound** — see site distribution table above. Cross-site validation splits are mandatory, not optional.
5. **PHI** — `ultrasound/`, `reports/`, and `manifest.csv` contain patient names, MRNs, dates. Do not share externally without de-identification. All are gitignored.

## File naming convention

`<src>_<scan-date>_<patient>_<nn>.<ext>` e.g. `gd_2025-11-03_abdullah_01.bmp`

Chughtai has no scan date: `ch_arshad_01.bmp`. Report photos: `<src>_<date>_<patient>_report.<ext>`.

## Baseline results (2026-06-21, pre-expansion)

Both models trained on the original 95 frames (83 stones, 12 normal), 59 patients, patient-level 70/15/15 split. **These results are on the old smaller dataset — retrain after the 2026-07-12 ingestion.**

| Metric | EfficientNet | DINOv2 |
|---|---|---|
| Mixed split AUC | 0.83 | **0.94** |
| GD → CH AUC | 0.44 | 0.50 |
| CH → GD AUC | 0.86 | 0.69 |

**DINOv2 is the better model** — 0.94 mixed AUC with a clean training curve (val AUC rose steadily 0.45 → 1.0 over 30 epochs).

**Cross-site results are weak for both models** — this is a data problem, not a model problem. Gulab Devi has only 3 normal patients (4 frames); Chughtai had only 3 stone patients (now 21 after expansion). The CH→GD direction should improve with retraining.

**Threshold note:** at threshold 0.5, sensitivity is low (model outputs low probabilities for stones due to 6.9× normal loss weighting in the old run). With the balanced new dataset, threshold 0.5 should be more reliable.

## EfficientNet results (2026-07-12, post-expansion)

Trained on expanded dataset: 194 frames, 136 patients, ~1:1 class balance. Patient-level 70/15/15 split. Two-phase training: frozen backbone 30 epochs + unfreeze last 2 blocks 15 epochs (lr=1e-4).

| Metric | Mixed split | GD → CH | CH → GD |
|---|---|---|---|
| AUC-ROC | 1.0000 | 0.4691 | 0.3250 |
| Sensitivity | 0.6429 | 0.0000 | 0.0000 |
| Specificity | 1.0000 | 0.9114 | 1.0000 |
| Spec@Sens≥0.90 | 1.0000 | 0.0633 | 0.2500 |
| F1 | 0.7826 | 0.0000 | 0.0000 |
| Accuracy ⚠️ | 0.8077 | 0.7579 | 0.0625 |

Test set sizes: mixed (stones=14, normal=12), GD→CH (stones=16, normal=79), CH→GD (stones=60, normal=4).

**Mixed split AUC=1.0 is not trustworthy** — test set is only 26 samples. Val AUC plateaued at ~0.77–0.79 throughout training, which is the more honest generalisation signal.

**Cross-site failure is total** — sensitivity=0 in both directions at threshold 0.5. The model learned hospital style, not anatomy:
- GD images are 95% stones visually → model maps "GD style" to stones
- CH images are 81% normal visually → model maps "CH style" to normal
- Tested cross-site, probabilities are inverted relative to labels (GD→CH AUC < 0.5)

**Root cause: data imbalance per site.** GD cross-site train had only 3 normal patients; CH cross-site train had only 18 stone patients. No model can generalise cross-site from this distribution.

## DINOv2 results (2026-07-12, post-expansion)

Same dataset and split as EfficientNet above. Phase 1 only (frozen backbone, 30 epochs) — no fine-tuning phase.

| Metric | Mixed split | GD → CH | CH → GD |
|---|---|---|---|
| AUC-ROC | 0.9583 | 0.5676 | 0.3958 |
| Sensitivity | 0.2857 | 0.7500 | 0.2167 |
| Specificity | 1.0000 | 0.4177 | 0.7500 |
| Spec@Sens≥0.90 | 0.9167 | 0.0886 | 0.0000 |
| F1 | 0.4444 | 0.3243 | 0.3514 |
| Accuracy ⚠️ | 0.6154 | 0.4737 | 0.2500 |

Test set sizes: mixed (stones=14, normal=12), GD→CH (stones=16, normal=79), CH→GD (stones=60, normal=4).

**DINOv2 is the better model overall:**
- Val AUC rose steadily to 0.88 and plateaued (vs EfficientNet's 0.77–0.79 plateau) — healthier training curve
- GD→CH AUC 0.568 vs EfficientNet 0.469 — above random vs below random; sensitivity 0.750 vs 0.000
- CH→GD AUC 0.396 vs EfficientNet 0.325 — both fail, DINOv2 slightly less so
- Frozen ViT backbone does not memorise machine-specific patterns the way EfficientNet's fine-tuned blocks do

**Mixed split sensitivity 0.286 is a threshold artefact, not a model failure.** DINOv2 outputs conservative probabilities; threshold 0.5 is too high. Spec@Sens≥0.90 = 0.917 means at the right threshold the model catches 90% of stones while clearing 91.7% of normals — the best clinical number in either run.

**CH→GD val AUC peaked at 0.80 (epoch 11) then fell to 0.69** — overfitting. The model found a Chughtai-specific pattern that doesn't transfer to Gulab Devi. More GD normal cases remain the critical fix.

## Open tasks (next stages)

1. ~~**Chughtai text strip removal**~~ — **done**, output in `stage1/` (top 14% + right 20% crop, verified visually).
2. ~~**Caliper / text overlay removal**~~ — **done**, output in `stage2/` (OpenCV HSV inpainting on all frames — calipers on stones, GB label + depth numbers on normals).
3. ~~**Fan masking**~~ — **done**, 28 polygons annotated in VIA, masks in `fan_masks/`, output in `stage3/`. New frames use automatic contour detection via `mask_fan.py`.
4. ~~**Training code**~~ — **done**, `dataset.py` / `train.py` / `evaluate.py`; runs both models + cross-site evaluation.
5. ~~**Windows path fix**~~ — **done**, `dataset.py` patched to replace `\\` → `/` in `processed_relpath` before `Path.relative_to()`.
6. ~~**Baseline training run**~~ — **done**, EfficientNet AUC 0.83, DINOv2 AUC 0.94 on mixed split (2026-06-21). Pre-expansion dataset only.
7. ~~**More stone cases from Chughtai**~~ — **done** (2026-07-12), 21 Chughtai stone patients now (was 3). Run `ingest_new_dataset.py` to see how new data was added.
8. **Retrain on expanded dataset** — 194 frames, 136 patients, ~1:1 class balance. Expected improvement in CH→GD cross-site AUC.
9. **More normal cases from Gulab Devi** — still only 3 normal GD patients (4 frames); GD→CH cross-site AUC will remain weak until this is fixed. Target: 15–20 more normal GD scans.
10. **GradCAM verification** — confirm model attends to GB anatomy, not residual artefacts.
11. **Label unknown patients** — 7 patients have no radiology report; sourcing reports would add ~9 frames.
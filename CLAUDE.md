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

## Which data to use

- **Train on `stage3/`** — cleanest version: text strips removed, calipers inpainted, fan-masked (everything outside the ultrasound dome blacked out).
- `stage2/` = calipers inpainted, text strips removed (intermediate, kept for reference).
- `stage1/` = Chughtai text strips removed only (intermediate, kept for reference).
- `processed/` = Stage 0 output, untouched (kept for reference).
- **Do not train on `ultrasound/`** — raw BMPs/TIFs that include kidney/liver/other-organ frames and burned-in PHI.
- **Binary task filter**: `label IN ('stones', 'normal')` → 95 frames, 59 patients.
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

**Class imbalance via augmentation:** apply 6–7 augmented variants per normal frame vs 1–2 per stone frame. Combine with weighted loss (`class_weight ≈ {stones: 1, normal: 6.9}`).

**Why heavier augmentation on normal:** The normal class has only 12 unique frames across 11 patients — roughly 6.9× fewer than stones (83 frames, 48 patients). Without rebalancing, the model sees stones far more often per epoch and learns to predict stones by default, regardless of anatomy. Generating 6–7 variants per normal frame brings the effective per-epoch count closer to parity with stones.

**Why weighted loss on top of augmentation:** Augmentation increases data volume but doesn't change the loss signal weighting. A weighted loss (`class_weight ≈ {stones: 1, normal: 6.9}`) tells the model that a mistake on a normal frame costs 6.9× more than a mistake on a stone frame — directly counteracting the imbalance at the gradient level. The weight is derived from the inverse class frequency: 83 stone frames / 12 normal frames ≈ 6.9. Using both together addresses imbalance from two angles: data volume (augmentation) and optimization pressure (weighted loss).

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

Only 95 frames exist. The entire strategy flows from this — overfitting is the real problem, not architecture choice.

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
- Fine-tuning large backbones fully (ResNet-50+, full ViT) — too many free parameters for 95 frames
- Reporting only mixed-split AUC — must also run cross-site validation (Chughtai → Gulab Devi and vice versa)
- Trusting high AUC before caliper leakage is fixed — the model may be reading calipers, not anatomy

## Evaluation metrics

**Primary metric: AUC-ROC.** Threshold-independent and imbalance-robust. The headline number for all comparisons.

**Report at a clinically chosen threshold** (e.g. sensitivity ≥ 0.90):
- **Sensitivity** — of all stone cases, how many did the model catch. A missed stone is more dangerous than a false alarm.
- **Specificity** — of all normal cases, how many did the model correctly clear.

**Also report (with disclaimer):**
- **F1 score** — handles imbalance better than accuracy but is threshold-dependent; use it as a secondary check, not a headline number.
- **Accuracy** — reported for completeness only. ⚠️ With 83 stones vs 12 normal frames, a model that always predicts "stones" scores 87.4% accuracy while being clinically useless. Do not use accuracy to compare models or claim performance.

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
3. **Class imbalance** — 83 stone frames vs 12 normal frames (≈6.9:1). Augmentation or weighted loss required.
4. **Site confound** — Chughtai ≈ normal-heavy, Gulab Devi ≈ stones-heavy. Cross-site validation splits are important.
5. **PHI** — `ultrasound/` and `reports/` contain patient names, MRNs, dates. Do not share externally without de-identification.

## File naming convention

`<src>_<scan-date>_<patient>_<nn>.<ext>` e.g. `gd_2025-11-03_abdullah_01.bmp`

Chughtai has no scan date: `ch_arshad_01.bmp`. Report photos: `<src>_<date>_<patient>_report.<ext>`.

## Baseline results (2026-06-21)

Both models trained on 95 frames (83 stones, 12 normal), 59 patients, patient-level 70/15/15 split.

| Metric | EfficientNet | DINOv2 |
|---|---|---|
| Mixed split AUC | 0.83 | **0.94** |
| GD → CH AUC | 0.44 | 0.50 |
| CH → GD AUC | 0.86 | 0.69 |

**DINOv2 is the better model** — 0.94 mixed AUC with a clean training curve (val AUC rose steadily 0.45 → 1.0 over 30 epochs).

**Cross-site results are weak for both models** — this is a data problem, not a model problem. Gulab Devi has only 3 normal patients (4 frames); Chughtai has only 3 stone patients (4 frames). Neither site has enough of the minority class to train or evaluate cross-site properly.

**Threshold note:** at threshold 0.5, sensitivity is low (model outputs low probabilities for stones due to 6.9× normal loss weighting). Use Spec@Sens≥90 as the clinical operating point — DINOv2 achieves specificity 0.50 at 90% sensitivity on the mixed split.

## Open tasks (next stages)

1. ~~**Chughtai text strip removal**~~ — **done**, output in `stage1/` (top 14% + right 20% crop, verified visually).
2. ~~**Caliper / text overlay removal**~~ — **done**, output in `stage2/` (OpenCV HSV inpainting on all frames — calipers on stones, GB label + depth numbers on normals).
3. ~~**Fan masking**~~ — **done**, 28 polygons annotated in VIA, masks in `fan_masks/`, output in `stage3/`.
4. ~~**Training code**~~ — **done**, `dataset.py` / `train.py` / `evaluate.py`; runs both models + cross-site evaluation.
5. ~~**Windows path fix**~~ — **done**, `dataset.py` patched to replace `\\` → `/` in `processed_relpath` before `Path.relative_to()`.
6. ~~**Baseline training run**~~ — **done**, EfficientNet AUC 0.83, DINOv2 AUC 0.94 on mixed split (2026-06-21).
7. **More normal cases from Gulab Devi** — only 3 normal patients; cross-site GD→CH is random (AUC 0.50) until this is fixed. Target: 15–20 more normal GD scans.
8. **More stone cases from Chughtai** — only 3 stone patients; target: 10–15 more.
9. **GradCAM verification** — confirm model attends to GB anatomy, not residual artefacts.
10. **Label unknown patients** — 7 patients have no radiology report; sourcing reports would add ~9 frames.
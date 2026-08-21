# From Raw Scans to a Trainable Dataset

**Written:** 2026-08-20
**Who this is for:** anyone picking up this project cold. No prior context assumed.

This document walks through what came off the ultrasound machines, every transformation applied to it, and why each step exists. If you want the short version of the results instead, see the bottom of this file.

---

## 1. What we started with

Two Pakistani sources donated data, and they arrived in completely different shapes.

**Chughtai Lab** sent folders already sorted by diagnosis: one folder called "Gb normal", another called "Gb stones". Their machine is a Toshiba that outputs **dual-pane** images, meaning two ultrasound fans side by side in a single picture, with organ names burned into the image.

**Gulab Devi Teaching Hospital** sent folders sorted by date and patient, not by diagnosis. A folder was simply "3 November 2025, patient Abdullah" with no indication of what was wrong. Their images are single-pane.

Both sets shared the same three problems.

**They were whole-abdomen studies.** A sonographer scanning a patient photographs the kidney, liver, spleen, bladder, and gallbladder in one session. Only a minority of those images actually show a gallbladder. The rest are useless to us and actively harmful if left in.

**Every image had patient information burned into the pixels.** Not in metadata where it can be stripped cleanly, but printed into the image itself: patient name, medical record number, scan date, machine settings.

**Files were raw BMPs and TIFs** at inconsistent sizes with no consistent naming.

---

## 2. The labelling problem

Chughtai was easy. The lab had already sorted the folders, so the folder name is the label.

Gulab Devi was not. Their folders carried no diagnosis. What they did have was a **phone photograph of the typed radiology report** sitting inside most patient folders. So every one of those report photos was opened and read by hand, and the gallbladder finding was transcribed into a text field in our index. That transcribed finding became the label.

Folders with no report photo were marked `unknown` and excluded, unless the folder name itself said something explicit such as "normal gb". Seven patients are still stuck in this state.

This is why the index contains 52 report photographs alongside the ultrasound images. They are not training data. They are the evidence behind the labels.

---

## 3. The spine: manifest.csv

Before a single pixel was touched, everything was catalogued into one CSV file with one row per image and 15 columns. This file is the source of truth for the entire project. Nothing downstream reads the image folders directly. Every script reads the manifest.

The columns that matter most:

| Column | Purpose |
|---|---|
| `patient_id` | Which human this came from. Critical for splitting. |
| `label` | stones / normal / sludge / unknown |
| `source` | Chughtai or GulabDevi |
| `gb_view` | Whether this image actually contains a gallbladder |
| `processed_relpath` | Where the cleaned version lives |
| `is_duplicate` / `dup_of` | Near-identical frame flagging |
| `finding_summary` | The transcribed report text (Gulab Devi only) |

The manifest currently holds 290 rows: 238 ultrasound images plus 52 report photos. Of those, 194 survive the filter for binary stones-versus-normal training, spread across 136 patients.

---

## 4. Stage 0: three cleanup passes

### 4.1 Throwing out non-gallbladder images

All frames were reviewed visually on contact sheets. For Chughtai this was straightforward, because the organ name is printed on the image. For Gulab Devi there was no such label, so each frame was identified by anatomy.

In the original pass this cut **139 frames down to 107**.

One error was caught here, and how it was caught is worth recording. Six Gulab Devi frames were assumed to be single-pane but were actually dual-pane. Rather than re-examining everything by eye, an objective test was used: if the cropped width came out at 80% or more of the original width, the crop must be spanning two fans rather than one. That test flagged exactly those six, which were reclassified and re-cropped to the correct pane.

### 4.2 Cropping out the patient information

Every image had identifying text burned in. The removal method:

1. Convert to grayscale and threshold, separating bright pixels from dark background
2. Apply a morphological open, then a close, to clean up speckle
3. Find the largest connected blob, which is the ultrasound fan itself
4. Crop tightly to that blob's bounding box

Because the patient name and MRN sit in the black margin *outside* the fan, cropping to the fan removes them. For the dual-pane Chughtai images, the right half was isolated first, since the gallbladder sits consistently in the right pane on their machine.

This produced the `processed/` folder, one PNG per gallbladder frame.

**What this step could not fix:** measurement markers that sit *inside* the fan. Cropping cannot reach them. That becomes the most important problem later.

### 4.3 Finding near-duplicate frames

Sonographers often capture several near-identical shots of the same view seconds apart. If one lands in training and its twin lands in testing, the model has effectively already seen the answer, and the test score becomes fiction.

Detection used a **perceptual hash**: shrink each image to 32x32, run a discrete cosine transform, keep the top-left 8x8 block of coefficients, and threshold each against the median to produce a 64-bit fingerprint. Two images whose fingerprints differ in 6 bits or fewer are treated as near-identical.

Comparison was done **only within the same patient**, deliberately. Comparing across patients on a shared machine produces false positives, because two different patients scanned on the same Chughtai machine share so much layout that they hash similarly.

Duplicates were **flagged, not deleted**. They still carry real training signal. The rule is that they stay in training and are dropped from validation and test.

The original pass found 21 duplicates, leaving 86 unique frames. Currently 159 of the 194 frames are non-duplicate.

---

## 5. Stage 1: removing the Chughtai text strips

Every Chughtai image kept two bands of burned-in text that survived the fan crop:

- A **top strip**, roughly 14% of the height: partial patient name, timestamp, machine mode
- A **right strip**, roughly 20% of the width: depth scale numbers, the organ label "GB", frame rate, gain, probe settings

Because these come from one machine with one fixed layout, they land in the same place on every image. No machine learning is needed, just a fixed percentage crop.

This matters more than it sounds. Chughtai supplies the overwhelming majority of our normal cases. If a machine-parameter strip appears only on Chughtai images, and Chughtai images are mostly normal, then **the text strip itself becomes a giveaway for "normal"**. The model can score well by reading machine settings instead of anatomy.

Output: `stage1/`.

---

## 6. Stage 2: erasing the calipers

This is the step that separates a real result from a fake one.

When a sonographer spots a gallstone, they **measure it**. They place crosshair markers and dashed measurement lines directly on the image, and those get burned into the saved file. They do this because there is a stone. They do not do it when the gallbladder is normal.

So the calipers are a **near-perfect predictor of the label that has nothing to do with anatomy**. A model that learns "crosshairs present means stones" would score close to 100% in testing and be completely useless on a fresh scan that has not been annotated yet. This is called shortcut learning, and it is the most common way medical imaging results turn out to be worthless.

The calipers sit inside the fan, so cropping cannot remove them. Instead:

1. Threshold in HSV colour space to find both white markers (high brightness, low saturation) and coloured marker lines (high saturation)
2. Discard any blob larger than 300 pixels, since those are anatomy rather than annotation
3. Dilate the mask slightly so marker edges are fully covered
4. Run OpenCV **TELEA inpainting**, which reconstructs the covered pixels by extrapolating inward from the surrounding texture

The result is an image where the caliper is replaced by plausible surrounding tissue.

Crucially, this was run on **every image, not only the stone images**. Normal Chughtai images had their own giveaways: the "GB" organ label and the depth scale numbers burned in by the machine. Cleaning only the stone images would have removed one shortcut and left another one standing.

Output: `stage2/`.

---

## 7. Stage 3: fan masking

The final pass. Everything outside the ultrasound dome is blacked out entirely, which removes in one move any remaining corner markers, stray depth digits, dot rows, and machine artifacts hiding at the edges.

Two routes exist for this:

**Manual.** One representative image per unique pixel size was exported, all 28 polygons were hand-drawn in VGG Image Annotator, and those polygons were converted into reusable mask files keyed by image dimensions. This took about 15 minutes and guarantees correctness.

**Automatic.** Contour detection finds the fan boundary directly. This is what newly ingested images use.

Manual was chosen for the original set because automatic fan detection can accidentally swallow a corner marker if the marker happens to touch the fan edge.

Output: `stage3/`. **This is what actually gets trained on.**

---

## The stages side by side

| Folder | Contents | Status |
|---|---|---|
| `ultrasound/` | Raw BMPs, all organs, full PHI | Never train on this |
| `processed/` | Fan-cropped, name and MRN removed | Reference |
| `stage1/` | Chughtai text strips cropped off | Reference |
| `stage2/` | Calipers and organ labels inpainted away | Reference |
| **`stage3/`** | **Everything outside the fan blacked out** | **Training source** |

Each stage writes to a new folder rather than overwriting the previous one. Nothing is destructive, so any step can be inspected or redone independently.

---

## 8. Preprocessing at training time

The images are now clean but still inconsistent. Every one has a different pixel size, because each fan crop produced its own bounding box. There were 28 distinct sizes in the original set.

Worse, the two hospitals produce different **shapes**. Chughtai is portrait, roughly 472x577. Gulab Devi is landscape, roughly 533x364.

The fix, applied identically to training, validation, and test:

```python
transforms.Compose([
    transforms.Pad(...),            # black-pad the shorter side into a square
    transforms.Resize((224, 224)),  # now resize uniformly
    transforms.ToTensor(),          # 0-255 becomes 0.0-1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Why pad before resizing.** Squashing a portrait image straight into a square stretches the anatomy. A round gallbladder becomes an oval, and the acoustic shadow below a stone tilts. Padding to a square first means the resize is uniform in both directions, so the anatomy keeps its true proportions. Black padding is safe here because fan crops already have black edges, so nothing unfamiliar is introduced.

**Why 224x224.** That is what pretrained ImageNet models expect. Deviating from it forces the model to relearn spatial weights from scratch, which defeats the point of using a pretrained model at all.

**Why the images stay 3-channel RGB despite being grayscale.** They are stored with red, green, and blue channels all carrying the same value. Converting to true single-channel grayscale would force reinitialising the model's first convolutional layer, throwing away pretrained weights for no benefit.

**Why those specific normalization numbers.** They are the ImageNet channel statistics that the pretrained weights were originally learned under. Feeding data normalized the same way keeps activations in the range the network expects.

---

## 9. Augmentation

Applied to training images only, never to validation or test, because those need to stay a fixed and honest yardstick.

**What we apply and why:**

| Transform | Justification |
|---|---|
| Horizontal flip | Probe placement genuinely varies left to right |
| Brightness and contrast jitter | Gain settings differ by machine and operator |
| Gaussian noise | Simulates the speckle inherent to ultrasound |
| Rotation up to 10 degrees | Small probe angle variation |
| Random crop and resize | Different framing and zoom between sonographers |

**What we deliberately refuse to do:**

**No vertical flipping.** This is the interesting one. A gallstone casts an **acoustic shadow**, a dark cone extending *below* it, because the stone blocks sound from travelling deeper. That shadow is a primary diagnostic cue. Flipping vertically puts the shadow above the stone, which is physically impossible. You would be training the model on ultrasound images that could not exist.

**No large rotations** beyond roughly 15 degrees, since they drag black corners in from the fan boundary.

**No colour jitter or channel shuffling**, since the images are grayscale and there is no colour information to perturb.

---

## 10. Splitting rules

Two hard rules, both about preventing the model from cheating.

**Split by patient, never by file.** All of one person's frames must land in the same split. If patient Abdullah has four frames and three go to training while one goes to test, the model has already seen that gallbladder and the test score is inflated. This is fiddly in practice, because names repeat across dates: there are two different people called "Abdullah" and two called "Panzi" who must not be merged into one patient.

**Duplicates stay in training and are dropped from validation and test.** Maximum signal where it helps, clean measurement where it counts.

```python
train_df = binary[binary["patient_id"].isin(train_patients)]
val_df   = binary[binary["patient_id"].isin(val_patients)  & (binary["is_duplicate"] != "yes")]
test_df  = binary[binary["patient_id"].isin(test_patients) & (binary["is_duplicate"] != "yes")]
```

There is also a second, harder split: **cross-site**. Train entirely on one hospital, test entirely on the other. This is the split that reveals whether the model learned anatomy or hospital style, and it is mandatory rather than optional.

---

## Where this leaves us

The pipeline works. 139 raw frames became 86 clean ones in the first pass, and after a second data collection round the trainable set is 194 frames from 136 patients, close to evenly split at 100 stones against 94 normal.

Every known label shortcut has been closed. Patient names cropped, machine text stripped, calipers inpainted, everything outside the fan blacked out.

**The one thing all this cleaning cannot fix** is who supplied what. Roughly 96% of the normal images come from Chughtai and 79% of the stone images come from Gulab Devi. The two hospitals use different machines, which means their images carry different texture and grain even after every overlay is gone. That residual difference is still a usable shortcut.

This is exactly why cross-hospital AUC sits at 0.36 to 0.47 while mixed AUC sits at 0.95. The model performs well when both hospitals appear in training and testing, and collapses to worse than random when asked to generalise from one hospital to the other.

No amount of image processing solves that. It needs more normal cases from Gulab Devi, or a fresh dataset where both classes came off the same machine. See `Open_Source_Datasets.md` for the plan on sourcing that.

---

## Current best result, for reference

DINOv2 with a frozen backbone, evaluated on a held-out mixed-hospital test set of 26 images (14 stones, 12 normal):

| Metric | Score |
|---|---|
| AUC-ROC | 0.95 |
| Sensitivity | 0.93 |
| Specificity | 0.50 |
| F1 | 0.79 |
| Accuracy | 0.73 |

Reported at a decision threshold of 0.234 rather than the default 0.5. The threshold was chosen on the validation set and then applied to test, never chosen on test itself.

Three caveats belong with any quotation of these numbers. The test set is 26 images, so the error bars are wide. Cross-hospital performance fails completely, as described above. And training ran for 50 epochs when validation performance peaked around epoch 7, so the run should be shortened with early stopping.

---

## Outstanding verification

One check is still owed. **GradCAM has not yet been run**, so we do not have visual confirmation that the model attends to gallbladder anatomy rather than to some artifact the cleaning missed. Until that is done, the caliper removal is believed to have worked rather than proven to have worked.
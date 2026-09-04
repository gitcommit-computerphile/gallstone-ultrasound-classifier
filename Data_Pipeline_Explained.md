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

**Why the images stay 3-channel RGB.** Pretrained models expect three channels. Converting to single-channel grayscale would force reinitialising the model's first convolutional layer, which holds the universal edge and texture detectors and is the most transferable part of the network, then relearning it from 194 frames. There is no benefit to offset that cost.

Note that these images are **not** pure grayscale, despite looking it. Sampling 60 images from every stage found zero pure-grey frames: a representative in-fan pixel reads `R=107 G=113 B=126`. The machines apply a consistent blue tint of roughly 11 levels. That tint is near-identical across both hospitals (11.65 vs 11.17) and both classes (11.83 vs 11.08), so it carries no leakage risk, but it does mean a grayscale conversion would genuinely discard data.

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

**No colour jitter or channel shuffling.** The only colour present is a uniform machine tint that carries no diagnostic meaning, so perturbing channels adds noise without adding realism.

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

## 11. Choosing a model

### The constraint that decides everything

You have **194 images**. That single number eliminates most of the option space before architecture is even considered.

A model learns by adjusting parameters. If it has far more parameters than training examples, the cheapest way to reduce its error is to memorise the training set rather than learn anything general. With 194 images, essentially every modern vision architecture is orders of magnitude too large.

So the real question was never "which architecture is best?" It was **"how do I use a large model without letting it memorise?"**

### The options and why each was rejected or kept

| Option | Free parameters | Verdict |
|---|---|---|
| Train a CNN from scratch | millions | **Rejected.** Needs 10,000+ images minimum. Would memorise 194 instantly. |
| Fully fine-tune ResNet-50 or a full ViT | tens of millions | **Rejected.** Same memorisation problem, just with a better starting point. |
| Partially fine-tune EfficientNet-B0 | ~1 million | **Kept as baseline.** Freeze most, unfreeze the last two blocks. Reasonable middle road. |
| Frozen backbone, tiny head | **385** | **Chosen as primary.** DINOv2 with a linear head. |

### What "frozen backbone" actually means

From `train.py`:

```python
self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
self.backbone.requires_grad_(False)
self.head = nn.Linear(384, 1)

def forward(self, x):
    with torch.no_grad():
        feats = self.backbone(x)   # (B, 384) CLS token
    return self.head(feats)
```

The backbone holds roughly **21 million parameters, every one of them frozen**. The forward pass even wraps it in `torch.no_grad()`, so no gradient is computed through it at all.

The only thing that trains is the final line: **`nn.Linear(384, 1)`, which is 385 parameters.**

That is about two trainable parameters per training image, comparable in size to a logistic regression. It does not have the capacity to memorise 194 ultrasounds even if it wanted to.

The 21-million-parameter backbone still does all the visual work. It is simply a fixed function that converts an image into 384 numbers. You are not training a vision model; you are training a tiny classifier on top of a very good, unchangeable feature extractor.

### Why DINOv2 rather than any other frozen backbone

The decisive detail is **how DINOv2 was trained**. It is self-supervised: it learned from 142 million images with **no labels at all**, by learning to produce consistent representations of the same image seen under different crops and distortions.

Contrast a standard ImageNet-supervised backbone. That model was optimised to answer "is this one of these 1000 categories?", so its features became specialised toward telling dogs from cars. Useful, but shaped by a goal unrelated to ultrasound.

DINOv2 was never asked to name anything. Its features describe general visual structure: texture, edges, shape, spatial relationships. Those transfer far better to a domain that looks nothing like ImageNet photographs. This is a well-replicated result on small medical datasets and it is the core reason for the choice.

### What the two models actually scored

Both were run on the same data and the same patient-level split.

| | EfficientNet-B0 | DINOv2 frozen |
|---|---:|---:|
| Mixed-split test AUC | 1.000 | 0.952 |
| **Validation AUC plateau** | **0.77 – 0.79** | **0.88** |
| Gulab Devi → Chughtai AUC | 0.469 | 0.568 |
| Chughtai → Gulab Devi AUC | 0.325 | 0.396 |
| Mixed sensitivity @ 0.5 | 0.643 | 0.786 |

**EfficientNet's perfect 1.000 is not a win.** On a 26-image test set a perfect score is what overfitting looks like, not what success looks like. Its validation AUC never rose above 0.79, and validation is the honest signal here.

**DINOv2's validation curve is roughly ten points higher** and rose steadily rather than plateauing early.

**DINOv2 beat EfficientNet in both cross-site directions**, and there is a mechanical reason. EfficientNet's unfrozen blocks *can* adapt to machine-specific texture, so they do. A fully frozen backbone cannot: it can only reweight features that already existed before it ever saw this dataset, which makes it structurally harder to latch onto hospital style.

### Honest caveats

**The empirical comparison is weak.** Twenty-six test images cannot separate two models with confidence. The decision rests mainly on the a-priori argument about dataset size; the results are consistent with it rather than proof of it.

**DINOv2 still failed cross-site.** It lost less badly, but 0.396 and 0.568 are not usable numbers. The frozen backbone reduced the site confound. It did not solve it, because that is a data problem, not a model problem.

**The choice is robust either way.** Even had EfficientNet scored higher, 194 images would still argue for the frozen approach. If the dataset ever grows past a few thousand images, revisiting fine-tuning becomes reasonable.

---

## Where this leaves us

The pipeline works. 139 raw frames became 86 clean ones in the first pass, and after a second data collection round the trainable set is 194 frames from 136 patients, close to evenly split at 100 stones against 94 normal.

Every known label shortcut has been closed. Patient names cropped, machine text stripped, calipers inpainted, everything outside the fan blacked out.

**The one thing all this cleaning cannot fix** is who supplied what. Roughly 96% of the normal images come from Chughtai and 79% of the stone images come from Gulab Devi. The two hospitals use different machines, which means their images carry different texture and grain even after every overlay is gone. That residual difference is still a usable shortcut.

This is exactly why cross-hospital AUC sits at 0.36 to 0.47 while mixed AUC sits at 0.95. The model performs well when both hospitals appear in training and testing, and collapses to worse than random when asked to generalise from one hospital to the other.

No amount of image processing solves that. It needs more normal cases from Gulab Devi, or a fresh dataset where both classes came off the same machine. See `Open_Source_Datasets.md` for the plan on sourcing that.

---

## 12. All model results

Four training runs have been recorded. They are listed in full below, not just the best one, because the differences between them are where the useful lessons are.

### First, what the metrics actually mean

The tables below use seven numbers. Here is each one in plain terms, using the real mixed-split test set as the example: **26 images, of which 14 really have stones and 12 are really normal.**

**The model does not output "stones" or "normal".** It outputs a number between 0 and 1, a confidence score. You then pick a cutoff, called the **threshold**, above which you call it a stone. Everything below depends on where you put that cutoff.

---

**Sensitivity** = of the patients who really have stones, what fraction did we catch?

> 14 real stone cases. The model flags 13 of them. Sensitivity = 13/14 = **0.93**

This is the most important number in this project. A missed stone is the dangerous error: the patient goes home believing they are fine. The one it misses is called a **false negative**.

---

**Specificity** = of the patients who are really normal, what fraction did we correctly clear?

> 12 real normal cases. The model correctly clears 6. Specificity = 6/12 = **0.50**

The 6 it wrongly flags are **false positives**, or false alarms. Annoying and wasteful, since those people get an unnecessary follow-up, but not dangerous.

**Sensitivity and specificity pull against each other.** Lower the threshold and you catch more stones (sensitivity up) but flag more healthy people (specificity down). Raise it and the reverse happens. You cannot maximise both, so you choose which error you would rather make. Here, we deliberately accept false alarms to avoid missing stones.

---

**AUC-ROC** = how well the model separates the two groups, across *every possible threshold at once*.

Think of it as: pick one random stone patient and one random normal patient. AUC is the probability the model gives the stone patient the higher score.

| AUC | Meaning |
|---|---|
| 1.00 | Perfect separation |
| 0.95 | Very good |
| 0.50 | **Random guessing**, a coin flip |
| below 0.50 | Worse than random, meaning the scores are effectively backwards |

This is the headline metric because it does not depend on threshold choice. It measures whether the information is there at all, separately from whether the cutoff is set well.

**An AUC below 0.5 is not just "bad".** It means the model is systematically ranking stone cases *lower* than normal ones. That is what several cross-site runs did, and it is a signature of learning the wrong thing rather than learning nothing.

---

**Accuracy** = simply the fraction of all 26 predictions that were right.

> 19 of 26 correct = **0.73**

Easy to understand and weak as evidence. If a dataset were 90% normal, a model that blindly answered "normal" every time would score 90% accuracy while being clinically useless. Our data is close to balanced, so accuracy is not meaningless here, but it is still reported for completeness rather than treated as the headline.

---

**F1 score** = a single number blending sensitivity with how often a "stones" prediction was actually correct.

Useful as a tie-breaker when comparing two models. It is threshold-dependent, so it moves whenever the cutoff moves, and it should not be quoted on its own.

---

**Spec @ Sens ≥ 0.90** = "if we forced the model to catch at least 90% of stones, how many normals could it still clear?"

This is the clinically framed question. It asks what specificity is achievable at an acceptable safety level, regardless of what the default threshold happens to be.

It is the metric that exposed the run 3 problem. Sensitivity looked terrible at 0.286, but Spec@Sens≥0.90 was 0.917, which proved the model *could* catch 90% of stones while clearing 92% of normals. The ability was there; only the cutoff was wrong.

---

**Optimal threshold** = the cutoff chosen from the validation set to reach sensitivity of at least 0.90, then applied unchanged to the test set.

The default of 0.5 is an arbitrary convention, not a law. Choosing a better one is legitimate. What would *not* be legitimate is picking the threshold by looking at test results, which is why it is always selected on validation data the model was not scored on.

---

**One more distinction that matters throughout: validation AUC versus test AUC.**

- **Validation** data is checked repeatedly during training to decide when to stop and which checkpoint to keep.
- **Test** data is touched once, at the very end.

Test AUC is the formal result, but with only 26 test images it is extremely noisy. The validation curve, measured many times across many epochs, is often the more honest signal about whether a model is genuinely learning. That is exactly why EfficientNet's perfect test AUC of 1.000 is not believed: its validation AUC never rose above 0.79.

---

### The four runs at a glance

| # | Run | Date | Dataset | Mixed AUC | GD→CH | CH→GD |
|---|---|---|---|---:|---:|---:|
| 1a | EfficientNet baseline | 2026-06-21 | 95 frames, 59 patients | 0.83 | 0.44 | 0.86 |
| 1b | DINOv2 baseline | 2026-06-21 | 95 frames, 59 patients | 0.94 | 0.50 | 0.69 |
| 2 | EfficientNet post-expansion | 2026-07-12 | 194 frames, 136 patients | 1.000 | 0.469 | 0.325 |
| 3 | DINOv2 post-expansion | 2026-07-12 | 194 frames, 136 patients | 0.958 | 0.568 | 0.396 |
| 4 | **DINOv2 improved** | 2026-07-12 | 194 frames, 136 patients | **0.952** | 0.475 | 0.358 |

Test set sizes are the same for every post-expansion run: mixed 26 images (14 stones, 12 normal), GD→CH 95 images (16 stones, 79 normal), CH→GD 64 images (60 stones, 4 normal).

---

### Run 1: the baseline, before the dataset grew

Trained on the original 95 frames (83 stones, 12 normal) across 59 patients, with a 70/15/15 patient-level split.

| Metric | EfficientNet | DINOv2 |
|---|---:|---:|
| Mixed split AUC | 0.83 | **0.94** |
| GD → CH AUC | 0.44 | 0.50 |
| CH → GD AUC | 0.86 | 0.69 |

DINOv2 led on the mixed split from the start, with a clean training curve (validation AUC climbing steadily from 0.45 to 1.0 over 30 epochs).

The oddity here is EfficientNet's CH→GD score of 0.86, the single best cross-site number ever recorded in this project. It did not survive the expansion, and with only 3 Chughtai stone patients in that split it should be read as noise rather than a real capability.

---

### Run 2: EfficientNet on the expanded dataset

194 frames, 136 patients. Two-phase training: 30 epochs with the backbone frozen, then 15 epochs with the last two blocks unfrozen at lr=1e-4.

| Metric | Mixed split | GD → CH | CH → GD |
|---|---:|---:|---:|
| AUC-ROC | 1.0000 | 0.4691 | 0.3250 |
| Sensitivity | 0.6429 | 0.0000 | 0.0000 |
| Specificity | 1.0000 | 0.9114 | 1.0000 |
| Spec @ Sens≥0.90 | 1.0000 | 0.0633 | 0.2500 |
| F1 | 0.7826 | 0.0000 | 0.0000 |
| Accuracy | 0.8077 | 0.7579 | 0.0625 |

**The perfect 1.000 is the problem, not the achievement.** On a 26-image test set, a flawless score is what overfitting looks like. Validation AUC sat at 0.77 to 0.79 for the entire run, and that is the honest number.

**Cross-site failure is total.** Sensitivity is exactly 0.000 in both directions: the model never once called a stone case a stone when tested on the unseen hospital. GD→CH AUC below 0.5 means its probabilities are actually inverted relative to the labels.

The mechanism is visible in the data. Gulab Devi images are 95% stones, Chughtai images are 81% normal, so the model mapped "GD style → stones" and "CH style → normal" and never looked at anatomy.

---

### Run 3: DINOv2 on the expanded dataset

Same data and split. Phase 1 only, 30 epochs, backbone fully frozen.

| Metric | Mixed split | GD → CH | CH → GD |
|---|---:|---:|---:|
| AUC-ROC | 0.9583 | 0.5676 | 0.3958 |
| Sensitivity | 0.2857 | 0.7500 | 0.2167 |
| Specificity | 1.0000 | 0.4177 | 0.7500 |
| Spec @ Sens≥0.90 | 0.9167 | 0.0886 | 0.0000 |
| F1 | 0.4444 | 0.3243 | 0.3514 |
| Accuracy | 0.6154 | 0.4737 | 0.2500 |

Better than EfficientNet in every way that matters. Validation AUC rose to 0.88 and held, against EfficientNet's 0.77 to 0.79 plateau. GD→CH at 0.568 is above random where EfficientNet's 0.469 is below it, and sensitivity there was 0.750 against 0.000.

**But mixed sensitivity of 0.2857 looks alarming, and it was a red herring.** The model was catching only 4 of 14 stone cases at threshold 0.5. The giveaway is Spec@Sens≥0.90 = 0.9167, which says that at some *other* threshold the model could catch 90% of stones while still clearing 92% of normals. The information was there; the threshold was wrong.

The cause turned out to be a stale setting: `CLASS_WEIGHTS` still penalised normal cases 6.9 times more heavily, a value calculated for the old 83:12 imbalanced dataset. On a now-balanced dataset it was pushing every prediction toward "normal".

---

### Run 4: DINOv2 improved, the current best

Same data and split, 50 epochs. Three changes: `CLASS_WEIGHTS` corrected from 6.9 to 1.0, Gaussian noise augmentation added, and the decision threshold chosen from the validation set instead of defaulting to 0.5.

| Metric | Mixed split | GD → CH | CH → GD |
|---|---:|---:|---:|
| AUC-ROC | 0.9524 | 0.4747 | 0.3583 |
| Sensitivity @ 0.50 | 0.7857 | 0.8750 | 0.7833 |
| Specificity @ 0.50 | 1.0000 | 0.1646 | 0.0000 |
| Spec @ Sens≥0.90 | 1.0000 | 0.0633 | 0.0000 |
| F1 @ 0.50 | 0.8800 | 0.2917 | 0.8468 |
| Accuracy @ 0.50 | 0.8846 | 0.2842 | 0.7344 |
| **Optimal threshold** | **0.2344** | 0.2863 | 0.4391 |
| Sensitivity @ opt | **0.9286** | 1.0000 | 0.8500 |
| Specificity @ opt | 0.5000 | 0.0506 | 0.0000 |
| F1 @ opt | 0.7879 | 0.2991 | 0.8870 |
| Accuracy @ opt | 0.7308 | 0.2105 | 0.7969 |

**The class-weight fix was the single largest improvement in the project.** Mixed sensitivity at threshold 0.5 jumped from 0.286 to 0.786 with no change to the data or the architecture. The model had always been capable; it was miscalibrated.

**The best clinical operating point is the optimal threshold on the mixed split:** sensitivity 0.929 and specificity 0.500. That catches 13 of 14 stone cases while misclassifying 6 of 12 normals. For this task that trade is correct, since a missed stone is far more dangerous than a false alarm that triggers a second look.

**Cross-site did not improve, and GD→CH slightly worsened** (0.475 against 0.568 in run 3). The class-weight change shifted the probability distributions, which moved these numbers around, but the underlying cause never changed: only 3 normal Gulab Devi patients exist to train on.

**50 epochs was too many.** Validation AUC peaked at 0.8615 around epoch 7 or 8, then drifted down to 0.7846. The best checkpoint was correctly saved and restored so the reported numbers stand, but the run should be shortened to roughly 20 epochs or given early stopping with patience 5.

---

### What the four runs collectively show

**Expanding the dataset fixed the class imbalance but not the site confound.** Going from 95 to 194 frames took the class ratio from 6.9:1 to 1.06:1. Cross-site scores did not improve, because the new data was almost all Chughtai stones and the normal side stayed stuck at 4 Gulab Devi frames.

**A frozen backbone generalises better than a fine-tuned one here.** DINOv2 beat EfficientNet in both cross-site directions on the expanded dataset. Unfrozen blocks can adapt to machine-specific texture and therefore do; a frozen backbone can only reweight features that existed before it saw this data.

**Calibration mattered more than architecture.** The biggest single jump in usable performance came from correcting one stale constant, not from changing models.

**Mixed-split AUC has been between 0.83 and 1.00 across every run, and it has never once predicted cross-site performance.** Quoting it alone would misrepresent the project in all four runs.

### The caveats that belong with every number above

**The test set is 26 images.** One image changing class moves accuracy by about four points. Treat two-decimal precision as false confidence.

**Cross-site performance fails in every run.** The best post-expansion cross-site AUC is 0.568, and most are below random. Nothing here would work in a clinic that was not one of these two hospitals.

**No GradCAM has been run.** We have no visual confirmation that the model attends to gallbladder anatomy rather than a residual artefact the cleaning missed.

---

## Outstanding verification

One check is still owed. **GradCAM has not yet been run**, so we do not have visual confirmation that the model attends to gallbladder anatomy rather than to some artifact the cleaning missed. Until that is done, the caliper removal is believed to have worked rather than proven to have worked.
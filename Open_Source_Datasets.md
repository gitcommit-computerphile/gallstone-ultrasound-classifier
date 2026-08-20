# Open-Source Gallbladder Ultrasound Datasets — Where to Get More Data

**Written:** 2026-07-16
**Purpose:** Find a free public dataset with both normal and gallstone ultrasound images, so we can train without relying on our own 194 in-house frames.

---

## The short version

If you only read one paragraph: **email Dheeraj Kumar at `dheeraj.singh@paruluniversity.ac.in` today.** His dataset is exactly the task we want (normal vs. gallstones), and it only needs a polite email — no paperwork. The better dataset, GBCU, needs a faculty member's signature, so send that one too *if* you can find a professor willing to sign.

---

## Why we're looking outside our own data

Our current dataset has two problems that more of our own scans won't quickly fix:

1. **It's small.** 194 frames, of which only 159 are unique. That's tiny for deep learning, and the model memorises rather than learns.

2. **The two hospitals give the game away.** Almost all our *normal* scans come from Chughtai, and most of our *stone* scans come from Gulab Devi. So the model can score well by simply recognising "this looks like a Chughtai image → say normal" — without ever looking at the gallbladder. That's why our cross-hospital scores collapse to near-random (AUC 0.32–0.57). The model learned hospital style, not anatomy.

A fresh, better-balanced public dataset sidesteps both problems at once.

---

## The one requirement that matters most

**Normal and stone images must look alike.**

This sounds obvious but it's the whole ballgame. If we get our normal images from Hospital A and our stone images from Hospital B, the model will just learn to tell Hospital A from Hospital B — the exact mistake we're trying to escape. We'd have rebuilt our current problem with new data.

In practice this means: **prefer datasets from a single hospital using a single ultrasound machine**, where both classes were scanned the same way. That's the filter used to rank everything below.

---

## The uncomfortable finding

**There is no free, instantly-downloadable dataset that has both normal and gallstone images.**

We checked thoroughly. The single biggest free dataset, **UIdataGB** (10,692 images, downloadable right now from Mendeley with no permission needed), sounds perfect until you look at its nine categories — they are *all diseases*. Gallstones, cholecystitis, perforation, carcinoma, and so on. **There is no "normal" class at all.** It would hand us 1,326 more stone images we don't need, and zero of the normals we do need. It also ships no patient IDs, which means we couldn't split the data safely.

So every remaining route runs through an email. The good news: most of these are ordinary "just ask the author" requests, not formal paperwork.

---

## Your options, best fit first

| # | Dataset | Has both classes? | Do images match? | Paperwork? |
|---|---------|-------------------|------------------|------------|
| 1 | **Kumar et al. 2025** | ✅ Exactly normal vs. gallstones | ⚠️ Good — but 3 hospitals | Just an email |
| 2 | **GBCU** (IITD) | ✅ 432 normal + stones | ✅ Best — one hospital, one machine | ❌ Faculty signature |
| 3 | **Ge et al. 2025** | ✅ Normal / stones / cholecystitis | ✅ One institution | Just an email |
| 4 | **Yu et al. 2021** | ✅ Normal / stones / cholecystitis | ⚠️ Handheld scans vary | Just an email |
| — | ~~UIdataGB~~ | ❌ **No normal class** | — | None needed |

---

### Option 1 — Kumar et al. (start here)

**Paper:** *Computer-aided cholelithiasis diagnosis using explainable convolutional neural network*, Scientific Reports, 2025
**Contact:** Dheeraj Kumar — `dheeraj.singh@paruluniversity.ac.in` ✅ *verified*
**Institution:** Parul University, Vadodara, India

This is our exact task — the paper classifies *cholelithiasis (gallstones) vs. normal* from ultrasound. No relabeling or reshaping needed on our end. The paper states the data is available from the authors on request, and there's no formal agreement mentioned, so a student can simply ask.

**Two things to watch:**
- The images come from **three** hospitals (PGIMER Chandigarh, SIDS Surat, and Parul Sevashram Vadodara), so the hospital-style problem can creep back in. We'd need to check whether normal and stone cases are evenly spread across the three.
- Their paper mixes in **AI-generated synthetic images**. We want the real scans only — worth saying so explicitly in the email.

---

### Option 2 — GBCU (best images, but gated)

**Contacts:** Dr. Pankaj Gupta — `pankajgupta959@gmail.com` ✅ *verified*
Dr. Chetan Arora — `chetan@cse.iitd.ac.in` ✅ *verified*
**Institution:** IIT Delhi + PGIMER Chandigarh, India

This is the strongest dataset on paper: 1,255 images from 218 patients — 432 normal (from 71 patients), 558 benign (stones sit in here), and 265 malignant. Crucially, **everything was scanned at one hospital on one machine** (a GE Logiq S8), so normal and stone images genuinely look alike. It also has enough patient information to split the data properly.

**The catch:** it requires a signed license agreement, and they explicitly state *"We will only accept requests from permanent employees/faculty of the requesting institute."* A student cannot request it — a supervisor must send the form. Any faculty sponsor counts, including a course instructor, so it's worth one ask.

**Also note:** gallstones aren't a separate folder here. They're marked as boxes drawn inside the "benign" images, so we'd need to filter the benign set down to just the stone-marked ones.

---

### Option 3 — Ge et al. (worth a try)

**Paper:** *Exploring Deep Learning Applications using Ultrasound Single View Cines in Acute Gallbladder Pathologies*, Academic Radiology, 2025
**Authors:** Connie Ge, Junbong Jang, Patrick Svrcek, Victoria Fleming, Young H. Kim
**Contact:** ❌ Not verified — grab it from the corresponding-author footnote on the paper's first page

Separates *normal gallbladder*, *non-urgent gallstones*, and *acute cholecystitis*, all from one institution using the same protocol — so image consistency should be strong. The data is video clips (cines) rather than stills, which means extra work to pull frames out, but it also means more frames per patient.

---

### Option 4 — Yu et al. (backup)

**Paper:** *Lightweight deep neural networks for cholelithiasis and cholecystitis detection by point-of-care ultrasound*, Computer Methods and Programs in Biomedicine, 2021
**Authors:** Chih-Jui Yu, Hsing-Jung Yeh, Chun-Chao Chang et al.
**Institution:** Taipei Medical University, Taiwan
**Contact:** ❌ Not verified — check the paper's corresponding-author footnote

Has normal, gallstone, and cholecystitis images from a single emergency department. The downside is these are point-of-care scans taken by emergency doctors with handheld probes, so framing and quality vary more than in a radiology department. It's a reasonable fallback, not a first choice.

> **On the two unverified emails:** I couldn't confirm them (PubMed blocked automated access). They're printed on the first page of each paper under the corresponding author's name. Better to read them off the PDF than to guess — a wrong address just fails silently.

---

## The email template

> **Subject:** Data request — [paper title]
>
> Dear Dr. [Name],
>
> I'm a student at [university] working on an academic, non-commercial project on gallbladder ultrasound classification (stones vs. normal). I read your paper "[title]" ([journal, year]) and saw that the dataset is available on request.
>
> Would it be possible to access the ultrasound images? Specifically, I would need:
> - the **real (non-synthetic) images** with their normal/gallstone labels
> - **patient identifiers or per-patient grouping**, if available, as I split data by patient to avoid leakage
> - any note on whether normal and stone images were captured on the **same machine and protocol**
>
> I'm happy to sign any data-use agreement, and I will cite your paper in any resulting work.
>
> Thank you for your time,
> [Your name, university, supervisor's name if you have one]

---

## Two things to ask for in every single email

These are easy to forget and both are deal-breakers:

**1. Patient IDs.** If we don't know which images came from the same person, we can't split the data properly — the same patient's scans end up in both training and testing, and the model looks far better than it really is. This alone is why UIdataGB is unusable for us despite its size.

**2. Confirmation that both classes were scanned the same way.** Same machine, same department, same protocol. If normals and stones were captured differently, the model will spot that shortcut instead of learning anatomy — and we're back where we started.

---

## Licensing

Everything here is fine for academic, non-commercial use, which is what we're doing.

One thing to flag: UIdataGB's paper says **CC BY-NC** (non-commercial), while its Mendeley page says **CC BY 4.0**. Those contradict each other. It doesn't matter while this stays academic, but if this project ever goes commercial, that needs pinning down from the actual licence file in the download. Either way, attribution is required — cite the source paper.

---

## What to do next

1. **Today:** Email Dheeraj Kumar (Option 1). Lowest friction, exactly our task.
2. **Today, in parallel:** Ask around for any faculty member who'd sign the GBCU form (Option 2). It's the best data and the only one with guaranteed image consistency.
3. **If neither lands within ~2 weeks:** Fall back to Options 3 and 4, pulling the emails off the papers directly.
4. **Meanwhile:** Nothing here blocks the existing project. Our own data still needs more normal Gulab Devi scans regardless — no public dataset can supply *those*, since they'd have to come from that specific hospital.

**A realistic expectation:** researchers often take days or weeks to reply, and some never do. Send several requests at once rather than waiting on any one of them.

---

## Reference links

- [UIdataGB on Mendeley](https://data.mendeley.com/datasets/r6h24d2d3y/2) — the no-normal-class one, listed for completeness
- [GBCU dataset page](https://gbc-iitd.github.io/data/gbcu) — licence form and details
- [Kumar et al. 2025 (Scientific Reports)](https://www.nature.com/articles/s41598-025-85798-2)
- [Ge et al. 2025 (Academic Radiology)](https://www.academicradiology.org/article/S1076-6332(24)00648-2/fulltext)
- [Yu et al. 2021 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169260721004569)
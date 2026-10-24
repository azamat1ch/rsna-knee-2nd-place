# Method: how the members read the scans and how they are blended

Tags: OFFICIAL (Kaggle or the host), MEASURED (ours), CLAIM (someone else's report).
Labels: [labels.md](labels.md). Scores: [results.md](results.md).

## The data

4,407 training exams, 24,371 series (scan sequences), 819,078 DICOM files (one per slice). Up to 6 useful series
per exam: 3 planes (sagittal = side, coronal = front, axial = top) × fluid-sensitive or not. Field of view 70-320
mm; some images are 640 × 1280. MEASURED.

## Reading a scan (`knee/dicom.py`)

- **Decoding:** uncompressed files straight from the bytes, others through pydicom and the offline decoder wheels.
  The test may hold JPEG Lossless and JPEG 2000 (OFFICIAL), so the wheels self-test every run.
- **Pixels:** rescale slope and intercept, MONOCHROME1 inverted, anything else an error. One window per series at
  the 0.5th-99.5th percentiles over all slices, clipped to 0..1, stored as uint8.
- **Slice order** (`geometry_order`): position along the series normal. Filenames are random and `InstanceNumber`
  is wrong in most series.
- **Clean series only:** 3+ slices, one orientation, one size and spacing, no duplicate positions, no gap over 1.5 ×
  the median, CSV plane matches the geometry. Others are rejected, not repaired.
- **Scale:** millimetres from `PixelSpacing`; each slice resized on its own (bilinear, antialiased) and zero-padded.
  The 448 store fits the whole field of view into 448 px.

## Choosing series (`knee/series.py`)

- **6 slots** = plane × `Fluid_Sensitive`, one series each; empty slots are masked. A typical exam fills 5.
- **prefer-2d:** a 2D series beats a thin one (slices under 2 mm), then most slices, then finer pixels.
- **Tilted series dropped** (normal's largest component under 0.9, or within 0.2 of the second): 693 series; 71
  exams lose a slot.
- **Every knee as a right knee:** side from the slice centres' patient x if |x| > 20 mm, else one consistent R/L
  tag, else no flip (right 2,245, left 1,982, unknown 118). Coronal and axial images flip; sagittal slice order
  reverses. Labels never change. No mirror augmentation.
- **Slabs:** tile members average slices under 2 mm into about 3 mm slabs; window members, under 1.75 mm into 3.5 mm.

## The knee centre and crops (`knee/centre.py`)

- **Knee centre J**, a rule: up-down = mean field-of-view centre of sagittal and coronal series minus 11 mm;
  front-back = sagittal field-of-view centre; left-right = soft-tissue middle on a coronal slice. About 10 mm off,
  so boxes are generous.
- **crop-convnext:** square boxes at 384 px, cut from the 448 rows. Sagittal 120 mm, 15 mm above and in front of J;
  coronal 130 mm, 5 mm below J; axial 100 mm, 35 mm in front of J, slices -10..+72 mm from J.
- **lateral-effnet:** an 84 mm lateral square at 288 px plus the whole knee (160 mm at 224 px), both cut once from
  the file's pixels.
- **Window members:** per series a full view plus 100 mm local views around J (joint, kneecap, back; coronal medial
  and lateral), as 320 px rows cut once from the file's pixels.

## Inputs and models

Code: `knee/tiles.py`, `knee/windows.py`, `knee/models.py`.

- **Tiles** (crop-convnext, lateral-effnet): 3 neighbouring slices or slabs as colour channels, up to 16 per series,
  covering every slice, plus a sin/cos position embedding. Training draws a random offset 0/1/2; inference averages
  all three (+0.001 to +0.002, MEASURED). **TileModel:** a timm encoder per tile, then per-finding gated attention
  over all the exam's tiles. lateral-effnet pools its whole-knee branch apart; a per-exam gate (starts at 1) scales
  the close-up term.
- **Windows** (window-convnext, window-effnet): three slab rows of one series view as channels, at most 352 per
  exam. Training draws covering windows with slight slice jitter; inference uses fixed covers at three shifts.
  **WindowNet:** per finding, spatial attention inside each window, then attention across windows.

| Member | Encoder (timm, ImageNet start) | Input | Trained on | Seeds × epochs |
| --- | --- | --- | --- | --- |
| crop-convnext | `convnext_tiny.fb_in22k_ft_in1k_384` | knee boxes, 384 px | `teacher-mix-pf-58` | 3 × 10 |
| lateral-effnet | `tf_efficientnetv2_s.in21k_ft_in1k`, frozen batch norm | lateral box 288 px + whole knee 224 px | `expert-pf-lateral` (without the 58) | 3 × 15 |
| window-convnext | `convnext_tiny.fb_in22k_ft_in1k_384` | windows, 288 px crops of 320 px rows | `teacher-mix-pf-58` | 3 × 16 |
| window-effnet | `tf_efficientnetv2_s.in21k_ft_in1k`, frozen batch norm | windows, 320 px | `expert-reader-58` | 3 × 15 |

**Training** (full values in `configs/<member>.json`). Every member trains on all its exams, with no held-out fold,
and saves EMA weights (a slow running average of the weights, 0.999).

- Tiles: AdamW, lr 2e-4, weight decay 0.01, 5% warm-up then cosine, batch 4, fp16, binary cross-entropy on soft
  targets; small rotation, shift, zoom, gain and gamma, random series drop.
- Windows: AdamW, encoder lr 1e-4, head lr 1e-3, weight decay 0.05, gradient clip 5, batch 2, fp16; window-effnet in
  bf16 (fp16 hit NaN just past the learning-rate peak).

**Inference:** fp16 on T4, fp32 rerun where an output is not finite. Seeds averaged as probabilities (tiles) or
logits (windows). A failed exam keeps the member's prior (training-set rates), and the member is then treated as
failed, as in the scored notebook.

## Matthew's members (`external/`)

mil-reader (a 2.5D multiple-instance reader over knee-centred close-ups) and pf-specialist run his inference code unchanged:
`knee/external.py` checks each file against its pin, runs it and writes a member CSV. Their models and training:
[external/train/](../external/train/README.md).

## The blend (`knee/blend.py`, `configs/blend.json`)

- Per finding, each member's predictions are ranked over the whole test set (allowed: the notebook sees all test
  exams at once, OFFICIAL) and scaled to 0..1. Why ranks: [results.md](results.md#the-metric-in-practice).
- Core four on all 12 findings: mil-reader 0.35, crop-convnext 0.25, window-convnext 0.20, window-effnet 0.20.
- A specialist that finished takes 0.20 of its findings and the core four 0.80: lateral-effnet on Lateral Meniscus
  and Lateral OA, pf-specialist on PF OA.
- Weights set by hand, never fitted.

## Known traps still in the final members

| Trap | Effect |
| --- | --- |
| "Thin" chosen by thickness, slabs cut by spacing | 137 2D series with overlapping slices averaged in pairs (tiles) |
| At most 16 tiles per series | 231 series in 166 exams not fully seen |
| One knee centre projected into every series | Untested where series do not share coordinates |
| `Fluid_Sensitive`, `Fat_Suppression` differ at test (OFFICIAL) | Slots may fill differently |
| Not every plane promised at test (OFFICIAL) | Missing slots masked; a failed exam gets the prior |
| 2 corrupt training files; about 7 exams with both knees (CLAIM) | Guarded per exam; a centre crop can miss the knee |

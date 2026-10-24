# Results

Leaderboard scores are OFFICIAL (Kaggle). Member names: [method.md](method.md); older names:
[labels.md, Old names](labels.md#old-names).

## Final result

**Private 0.960, 2nd place** (57529331, 2026-10-22; public 0.969), tied with 1st on score; 3rd scored 0.959. Our best
private score of all 101 submissions. The clean notebook's late submission 57562482 scored the same.

| Public | Private | Submission | What |
| ---: | ---: | --- | --- |
| 0.969 | **0.960** | 57529331 | Final blend: core four plus both specialists |
| 0.968 | 0.959 | 57515377 | Core four at equal weights, no specialists |
| 0.968 | 0.959 | 57287145 | mil-reader / crop-convnext / window-convnext at 40 / 30 / 30 |
| 0.967 | 0.959 | 57247642 | mil-reader / crop-convnext at 55 / 45 |
| 0.966 | 0.957 | 57232571 | mil-reader alone |
| 0.965 | 0.954 | 57512157 | window-effnet alone |
| 0.964 | 0.955 | 57061664 | crop-convnext alone: our simplest strong model (images only, one architecture, one label table) |
| 0.964 | 0.955 | 57281157 | window-convnext alone (2 seeds) |

## What moved the score

| Public | Private | Submission | Step |
| ---: | ---: | --- | --- |
| 0.851 | 0.829 | 56853819 | ResNet-50 baseline on report labels |
| 0.920 | 0.904 | 56892316 | Image-read labels, every slice as tiles, about 3 mm slabs |
| 0.942 | 0.934 | 56894882 | CoAtNet instead of ResNet-50 |
| 0.952 | 0.944 | 56969209 | PF OA half from the report reader in the labels |
| 0.955 | 0.947 | 56958299 | ConvNeXt-T 224 |
| 0.960 | 0.951 | 56979045 | Detail crops around the knee centre, 288 px |
| 0.961 | 0.952 | 57017605 | Teacher-mix (student) labels |
| 0.962 | 0.953 | 57027506 | All training data, 3 seeds |
| 0.964 | 0.955 | 57061664 | crop-convnext: 384 px, new boxes, the PF OA rule, the 58 in training |
| 0.967 | 0.959 | 57247642 | Team merge: rank blend with mil-reader |
| 0.968 | 0.959 | 57287145 | Adds window-convnext |
| 0.969 | 0.960 | 57529331 | Adds window-effnet, lateral-effnet and pf-specialist; new weights |

IDEA: labels and pixels on the joint moved our own models most; after 0.965 only new members helped. 38 checkpoints
of one recipe scored 0.961, three seeds of a better recipe 0.964.

## Public and private

MEASURED from OFFICIAL scores:

- Public about 390 exams, private about 910 (OFFICIAL).
- Public to private: about −0.009 per file (76 submissions above 0.9, SD 0.0023).
- Order held: Spearman 0.977 over the 65 files at 0.955+; the best public file was the best private. 0.968 → 0.969
  public was 0.959 → 0.960 private.
- Standard error: about 0.004 public, 0.0026 private (SIM, our simulation).

## Per finding (public probes)

A probe keeps one finding and sets the other 11 to 0.5, so AUC = 12 × score − 5.5 (±0.006 from rounding).

| Finding | Best member probed | AUC |
| --- | --- | ---: |
| Lateral OA | lateral-effnet ×3 (57339251) | 0.932 |
| Lateral OA | mil-reader; an earlier two-branch crop model | 0.920 |
| PF OA | window-convnext ×2; the two-branch crop model | 0.920 |

PF OA and Lateral OA were the weakest findings for every model probed. lateral-effnet first lifted Lateral OA.

## The 58 expert exams (exploratory)

Known to about ±0.02 macro AUC; used to study errors, never to pick models. Models trained on the 58 are not scored
on them. MEASURED:

- **Fold models:** window-convnext 0.9488, crop-convnext 0.9445. lateral-effnet ×3 (all data, no 58) 0.9438, Lateral
  OA 0.915 (our best); least like the other members (Spearman about 0.92).
- **Experts tick more than reports:** 240 positives against 154, mostly mild (PF OA 21 vs 2, Effusion 35 vs 13,
  Synovitis 27 vs 11).
- **Weakest for the crop model:** Synovitis 0.882, Lateral OA 0.886, PF OA 0.902, Lateral Meniscus 0.919.
- **Labels vs model:** the final label table scores 0.956, the best fold model 0.9445.

## The metric in practice

- +0.012 on one finding is +0.001 overall.
- Rank mean 0.950 against probability mean 0.945 (SIM: two models of AUC 0.93, correlation 0.6).
- Ties count half: write float32, or ranks with 6+ digits.

Inference times: [verification.md](verification.md). The hidden run finished inside the limit.

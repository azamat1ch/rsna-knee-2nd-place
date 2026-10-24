# RSNA Knee Abnormality Detection: 2nd place solution

Team MURAD: Azamat ([azamat1ch](https://www.kaggle.com/azamat1ch)) and Matthew Liew
([matthewliew](https://www.kaggle.com/matthewliew)). **Private leaderboard 0.960, 2nd place**; public 0.969.

Better training labels, CNNs on knee-centred crops of every MRI series, and a rank blend of six models.

## Links

| What | Where |
| --- | --- |
| Code (this repository) | [github.com/azamat1ch/rsna-knee-2nd-place](https://github.com/azamat1ch/rsna-knee-2nd-place) |
| Inference notebook | [azamat1ch/rsna-knee-2nd-place-predict](https://www.kaggle.com/code/azamat1ch/rsna-knee-2nd-place-predict) |
| Model weights (six members, 21 files) | dataset [azamat1ch/rsna-knee-2nd-place-models](https://www.kaggle.com/datasets/azamat1ch/rsna-knee-2nd-place-models) |
| Code and offline DICOM decoder wheels | dataset [azamat1ch/rsna-knee-2nd-place-code](https://www.kaggle.com/datasets/azamat1ch/rsna-knee-2nd-place-code) |
| Competition | [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) |

This repository is a clean rewrite of the scored notebook (57529331): same weights, inputs and blend, bit-identical
predictions on 48 training exams, same score as a late submission (57562482):
[docs/verification.md](docs/verification.md).

## The solution

The test has images only, scored by the mean of 12 per-finding AUCs against expert labels. Training exams have a
report; 58 (**the 58**) also have expert labels.

1. **Labels** ([docs/labels.md](docs/labels.md)): language and vision models read the reports and images; teacher
   models and fixed rules make soft targets. Our largest gain.
2. **Six members** ([docs/method.md](docs/method.md)):

   | Member | Model | Blend weight |
   | --- | --- | --- |
   | mil-reader (Matthew) | ConvNeXt-T 2.5D reader, 5 seeds | 0.35 |
   | crop-convnext | ConvNeXt-T on knee-centred crops, 384 px, 3 seeds | 0.25 |
   | window-convnext | ConvNeXt-T on full and 100 mm local views, 3 seeds | 0.20 |
   | window-effnet | EfficientNetV2-S on the same views, 3 seeds | 0.20 |
   | lateral-effnet (specialist) | EfficientNetV2-S, lateral close-up + whole knee, 3 seeds | 0.20 on Lateral Meniscus and Lateral OA |
   | pf-specialist (Matthew) | PF OA model on his reader, 3 seeds | 0.20 on PF OA |

3. **Rank blend** (`knee/blend.py`, `configs/blend.json`): per finding, ranks scaled 0..1, summed with hand-set
   weights. Where a specialist runs, the core four share 0.80.

Scores: [docs/results.md](docs/results.md). What failed: [docs/experiments.md](docs/experiments.md).

## Predict

Open the notebook above, or build one from `predict.py` (`kaggle/predict/kernel-metadata.json`: competition and both
datasets attached, T4 ×2, internet off). Output: `/kaggle/working/submission.csv`.

1. Installs the decoder wheels offline, self-tests them, and checks every weight and code file against its SHA-256.
2. Runs the core four, each in its own process group with a time budget, and writes the core blend (all or nothing).
3. Runs the two specialists in the time left; if one fails, the core file stays.
4. Writes member predictions and `receipt.json` (times, exit codes, specialists used) to `/kaggle/working/blend/`.

Hidden test: several hours. 3 visible test exams: about 5 minutes.

## Train

All commands in order: [entry_points.md](entry_points.md). Stores, labels, one run per seed, then weights: [docs/training.md](docs/training.md). Matthew's members:
[external/train/](external/train/README.md).

## Repository map

| Path | What |
| --- | --- |
| `predict.py` | The inference notebook |
| `knee/` | Our package: scans and crops (`dicom`, `series`, `centre`), inputs (`tiles`, `windows`), `models`, `train`, `infer`, `stores`, `labels`, `panels`, `blend`, `score`, `external` (Matthew's members) |
| `configs/` | One recipe per member, `blend.json`, label prompts (`labels/`) |
| `external/` | Matthew's inference code, unchanged; his training code in `external/train/` |
| `kaggle/` | Metadata of the notebook, the datasets and the store kernel |
| `tests/` | Blend against the scored kernel's numbers; the metric |
| `docs/` | Method, labels, training, results, experiments, verification |

## Hardware and OS

- **Training:** rented vast.ai boxes, one GPU per run: RTX 5090 (32 GB), one window-effnet seed on an RTX PRO 6000.
  Boxes had 24-128 CPU cores and 126-515 GB RAM. Container `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py313-2026-09-07`
  (Ubuntu 24.04, CUDA 12.8.1, Python 3.13). Time and cost per member: [docs/training.md](docs/training.md).
- **Stores and label packs:** Kaggle CPU notebooks.
- **Inference:** Kaggle notebook, 2× T4 GPUs, internet off.

## Side effects and assumptions

- Nothing modifies the competition data; every command reads it and writes elsewhere.
- `knee.stores` and `knee.train` refuse an existing `--out` folder (`knee.train --resume` continues one).
- Inference checks every weight and code file against its SHA-256 and stops on a mismatch.
- `predict.py` assumes the Kaggle paths in `SETTINGS.json` and writes only under `/kaggle/working/`.
- Training needs the label tables (labels dataset, private, shared with the host on request, or rebuilt in step 2
  of [docs/training.md](docs/training.md)).

## Environment

- **Inference:** Docker image
  `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`,
  nothing installed beyond the decoder wheels.
- **Training:** [docs/training.md](docs/training.md).
- **Label tables and tests:** Python 3.12+, NumPy, pandas, SciPy. Run `python -m pytest -q tests`.
- Default paths: [SETTINGS.json](SETTINGS.json).

## Licence

CC-BY-NC 4.0 ([LICENSE](LICENSE)). Third-party code, weights, data and credits: [NOTICE.md](NOTICE.md).

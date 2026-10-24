# Verification: this repository equals the scored notebook

**Original** = the private code of the notebook that scored private 0.960. **Clean** = this repository. They give
the same predictions, bit for bit, on every member and both blends (MEASURED). Checks used training exams or
synthetic scans only.

## 1. Kaggle GPU, 48 training exams

One notebook (T4 ×2, internet off, competition Docker image) builds a stand-in test folder of 48 training exams (263
series, 10,469 DICOM files) and runs the original's six member stages, then the clean `predict.py`. Same weight
files (same SHA-256). No exam had a compressed DICOM.

| Compared | Result |
| --- | --- |
| Each member, 48 exams × 12 findings | **bit-equal** (max difference 0.0; Spearman 1.0 per finding) |
| Core blend (four members) | **bit-equal** |
| Full blend (with specialists) | **bit-equal** |
| Clean blend on the original's member files | bit-equal |

Run times (MEASURED, seconds):

| Member | Original | Clean |
| --- | ---: | ---: |
| crop-convnext | 271 | 224 |
| window-convnext | 192 | 189 |
| window-effnet | 168 | 171 |
| mil-reader | 197 | 156 |
| pf-specialist | 230 | 228 |
| lateral-effnet | 210 | 189 |
| **Clean notebook, end to end** | | **1,173** |

## 2. Fresh Kaggle machine

`kaggle/predict/kernel-metadata.json` (competition and our two datasets only), visible test (3 exams): all six
members ran, both specialists used, 291 s, valid file. Late submission 57562482: private 0.960, public 0.969
(OFFICIAL).

## 3. CPU, original against clean

All MEASURED.

| Part | How | Result |
| --- | --- | --- |
| Preprocessing (reading, series, knee centre, tiles, windows) | synthetic exams, every field | equal |
| Training stores | 8 synthetic exams | 448: 27/27 files byte-equal; lateral 21/21 series; window 8/8 exams |
| Store readers | both trainers on the clean stores | same items and shapes |
| Inference, CPU fp32 | 8 synthetic exams | crop-convnext, window-convnext, lateral-effnet bit-equal (window-effnet: section 1) |
| Weights | our 12 weight files in both codes | bit-equal outputs, 4 recipes |
| Model start | same seed and start weights | bit-equal |
| Training maths | lr schedule, EMA, loss, augmentation, exam order, window draws | bit-equal |
| Two training steps | 4 recipes | bit-equal weights (logged loss off by 3e-8, print only) |
| Resume | 2 epochs, resumed after epoch 1 | byte-equal weights |
| Full schedule | real step counts, 3 seeds, tiny model (crop-convnext 11,010, lateral-effnet 16,305, window-convnext 35,216, window-effnet 33,015) | lr, weight decay, EMA, batch order, saved weights equal at every step |
| Stores on real exams (Kaggle CPU) | 20 exams per kind vs the final models' store files | 448: 103 series, lateral 71 series, window 10,779 rows, all byte-equal; exam order equal per shard |
| Labels (real data) | `knee.labels export / folds / mix all` from the labels dataset | 15/15 tables byte-equal; prompts equal |
| Blend | 96 random cases, member subsets, rank scaling, the original's 3-exam run | equal; both submission files rebuilt |
| Metric | 3,600 AUC and 300 macro cases vs a reference | equal |
| Notebook flow | forced failures | failed specialist keeps core file; core failure writes none; time-out stops the process group |
| Matthew's drivers | his code via `knee/external.py` | mil-reader, pf-specialist bit-equal |
| Matthew's trainers, trimmed vs his release | 16 synthetic exams, fake OAI rows, tiny model, 0 and 2 loader workers | mil-reader, pf-specialist: 169/169 weight tensors bit-equal; OAI grade table equal on 3 fake MOAKS sets |
| Tests | `python -m pytest -q tests` | 18 passed |

## 4. Not checked

- **Compressed DICOMs on real files.** All training files were uncompressed; the test may use JPEG Lossless or JPEG
  2000 (OFFICIAL). The store builder skips the decoders, as in the original. The wheels match the scored notebook's
  (SHA-256) and self-test on each run.
- **A full retrain.** GPU training is not bit-repeatable; section 3 covers what decides training.
- **New label calls.** Not repeatable; tables are rebuilt from logged answers.
- **lateral-effnet's skipped fp16 steps** (1-2 of 16,305 per seed). Not replayable on CPU; same skip code (read).
- **Label panels on real scans.** They only feed new calls.
- **Teachers and Matthew's training.** Teachers are frozen inputs. `external/train/` was not rerun on real data or a
  GPU; his weights match our pins.

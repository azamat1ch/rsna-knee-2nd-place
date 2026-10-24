# Training: rebuild every member from the competition data

Stores, labels, one run per seed, weights. A retrain is close, not bit-identical (GPU training varies). Checked
equal: [verification.md](verification.md).

## What you need

| Input | Where |
| --- | --- |
| Competition data (`train.csv`, `train_series.csv`, `train_series/`) | Kaggle competition `rsna-knee-abnormality-detection` |
| 15 label tables and their frozen inputs | labels dataset (private, shared with the host on request), or step 2 |
| Start weights | Hugging Face, pinned by revision and SHA-256 in each recipe (`encoder.start_weights`) |
| One GPU per run | rented RTX 5090s (32 GB); one window-effnet seed on an RTX PRO 6000 |
| Environment | Python 3.13, CUDA 12.8, `pip install -r requirements.txt` |

## 1. Stores (Kaggle CPU)

Training scans cut into input rows by the inference code (`knee.dicom`, `knee.tiles`, `knee.windows`).
Deterministic.

```bash
python -m knee.stores --kind 448|lateral|window --start N --count 250 \
    --data COMPETITION_DIR --exclusions exclusions.json --out SHARD_DIR
python -m knee.stores --kind 448|window --expert ...    # the 58 expert exams as their own shard
```

- **Exams:** training exams minus `exclusions.json` (the 58 and 4 linked exams; step 2), fixed hashed order. 4,345
  exams, 18 shards: `--start 0, 250, ..., 4250`. Only `train_series.csv` and the DICOMs are read.
- **On Kaggle:** push `knee/stores.py` with `kaggle/stores/kernel-metadata.json` (competition, code dataset, a
  dataset with `exclusions.json`; inference Docker image). Set `KAGGLE_JOB` at the top of the file per shard.

| Store | Kind | Used by | Layout the trainer reads |
| --- | --- | --- | --- |
| 448 px rows, every usable series | `448` (18 + `--expert`) | crop-convnext, lateral-effnet | `DATA/448/<shard>/` (`images.zip`, `cache_manifest.json`, `series_metadata.jsonl`) |
| Lateral crops from the file's pixels | `lateral` (18) | lateral-effnet | `DATA/lateral/<shard>/` (`crops.u8`, `series_private.jsonl`) |
| Window rows, every chosen series | `window` (18 + `--expert`) | window-convnext, window-effnet | `DATA/<shard>/` (`px.u8.npy`, `meta.f32.npy`, `index.json`) |

Tile members also need `train_series.csv` in `DATA/`. They cache knee centres in `DATA/cache/centres.json`.

## 2. Labels

The 15 tables, rebuilt from the labels dataset with no model call, byte-equal (MEASURED). Rules:
[labels.md](labels.md).

```bash
python -m knee.labels export --train train.csv --partition partition.json --reader-log calls/reader.jsonl \
    --expert-log calls/expert.jsonl --blind-lateral-log calls/blind-lateral.jsonl --out tables
python -m knee.labels folds --train train.csv --partition partition.json \
    --reader-detail tables/reader-detail.jsonl --out tables
python -m knee.labels mix all --dir tables --train train.csv --teacher-root . --mil-reader-oof mil-reader-oof.csv \
    --teachers teachers/convnext-224.json teachers/convnext-detail-288.json teachers/coatnet-224.json \
               teachers/effnetv2s-320.json teachers/radimagenet-r50.json
```

`knee.labels reader` / `expert` (packs from `knee.panels`, Kaggle CPU) make new model calls and different tables.

## 3. Train each seed

```bash
python -m knee.train --recipe configs/<member>.json --seed SEED --data DATA --labels tables --out RUN_DIR \
    [--start-weights model.safetensors] [--workers 8] [--resume]
```

One seed per GPU. Downloads the pinned start weights unless `--start-weights` is given; checkpoints every epoch.
Writes `RUN_DIR/seed-SEED.pt` (EMA weights, as inference loads them) and `run.json` (recipe, labels, hashes).

| Member | `--data` | Label table | Seeds | Epochs |
| --- | --- | --- | --- | --- |
| crop-convnext | tile store (448 incl. the 58 shard) | `teacher-mix-pf-58.csv` | 20261014, 20261015, 20261016 | 10 |
| lateral-effnet | tile store (448 + lateral, no 58 shard needed) | `expert-pf-lateral.csv` | 20261101, 20261102, 20261103 | 15 |
| window-convnext | window store incl. the 58 shard | `teacher-mix-pf-58.csv` | 20261014, 20261015, 20261016 | 16 |
| window-effnet | window store incl. the 58 shard | `expert-reader-58.csv` | 20261121, 20261122, 20261123 | 15 |

### What it took (MEASURED, the shipped runs)

| Member | Hardware | Per epoch | Box time, all seeds | Cost |
| --- | --- | --- | --- | --- |
| crop-convnext | 1× RTX 5090 | 432-544 s | 4.6 h (one box) | $3.64 |
| lateral-effnet | 1× RTX 5090 per seed | 390-940 s (by host) | 11.4 h (3 boxes) | $11.32 |
| window-convnext | 1× RTX 5090 | about 1,000 s | 13.9 h (one box) | $14.71 (with one fold-0 run) |
| window-effnet | 1× RTX PRO 6000 + 2× RTX 5090 | 1,360-1,510 s | 22.6 h (3 boxes) | $29.14 |

Boxes had 24-128 CPU cores and 126-515 GB RAM. Loaders need many cores; window members need fast disk.

## 4. Weights for inference

Store each file as `<member>/seed-<seed>.pt` in the models dataset and pin its SHA-256 in the recipe's
`weights.files`. Inference refuses a mismatched file.

## Matthew's members

mil-reader, pf-specialist and their fastMRI encoder: [external/train/](../external/train/README.md).

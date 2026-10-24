# Entry points

The commands, in order. Each takes its paths as arguments; the defaults are in `SETTINGS.json`. Details:
[docs/training.md](docs/training.md) (our four members) and [external/train/](external/train/README.md) (Matthew's
two).

## 1. Prepare data

```bash
# Model inputs ("stores") from the training scans, one shard of 250 exams at a time
python -m knee.stores --kind 448|lateral|window --start N --count 250 \
    --data COMPETITION_DIR --exclusions exclusions.json --out SHARD_DIR
# The 15 label tables, rebuilt from the labels dataset (no model calls)
python -m knee.labels export ...
python -m knee.labels folds ...
python -m knee.labels mix all ...
```

Matthew's members: `build_cache.py`, `build_views.py`, `make_targets.py` (mil-reader) and `native.py`,
`oai_join.py`, `oai_inputs.py` (pf-specialist).

## 2. Train

```bash
python -m knee.train --recipe configs/<member>.json --seed SEED --data DATA --labels tables --out RUN_DIR
```

One run per seed and member. Writes `RUN_DIR/seed-SEED.pt` (the weights inference loads), `run.json`, and a
`resume.pt` checkpoint after every epoch (`--resume` continues). Matthew's: `external/train/encoder/pretrain_fastmri.py`,
`external/train/mil-reader/train.py`, `external/train/pf-specialist/train.py`.

## 3. Predict

```bash
python predict.py
```

Runs as the Kaggle notebook [azamat1ch/rsna-knee-2nd-place-predict](https://www.kaggle.com/code/azamat1ch/rsna-knee-2nd-place-predict)
with the competition, the code dataset and the models dataset attached. Reads the test scans from the competition
folder and the weights from the models dataset; writes `/kaggle/working/submission.csv`.

# Training code for Matthew's members

Matthew Liew's code that trained mil-reader, pf-specialist and their shared encoder. The models and data
code they use are the inference copies in `external/reader/` and `external/v731/`.

Trimmed from his own training code: copies of the inference files removed, paths made into flags, code the
final runs did not use removed. On synthetic data both trainers give bit-identical weights to his originals,
and `oai_join.py` the same grade table as his two OAI table scripts.

| Folder | Trains | Matthew's name | Weights |
| --- | --- | --- | --- |
| `encoder/` | ConvNeXt-T encoder, pretrained on fastMRI knee scans | fm_encoder7 | `fm_encoder7.pt` |
| `mil-reader/` | mil-reader, 12 findings, 5 seeds; also the left/right knee classifier | V7.2 | `v72full_s0..s4.pt`, `side_model.pt` |
| `pf-specialist/` | pf-specialist, PF OA with a kneecap-cartilage grade head, 3 seeds | E3 (V7.3.1) | `final_ema.pt` per seed |

Build in this order: side classifier, encoder, mil-reader inputs, mil-reader, pf-specialist. Not included:
data, labels, weights. **Environment:** Python 3.11, image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`,
pins in `requirements.txt`, including the DICOM decoders (JPEG Lossless, JPEG 2000) that `predict.py` installs;
`build_cache.py` and `native.py` stop if they are missing. Commands below run from the folder named in each section, inputs in `work/`.

## 1. Side classifier (`mil-reader/`)

ResNet-18, left versus right knee, trained in a private Kaggle notebook on competition exams whose side the
DICOM headers make certain; it then labels our unsure exams and the fastMRI validation scans (the encoder
uses those). `python side_notebook.py side.py <fastMRI val URL>` writes that notebook's script; it saves
`side_model.pt`. Its train/test split is now fixed; the original run's split was not, so a rerun gives a
slightly different model.

## 2. Encoder (`encoder/`)

Learns 7 fastMRI+ box findings: lateral and medial meniscus tear, lateral and medial cartilage loss, ACL
sprain, bone (fracture, contusion, oedema), MCL sprain. Inputs: the fastMRI v2.0 single-coil knee archives
(train and val, coronal `reconstruction_esc` only), fastMRI+ `Annotations/knee.csv` and
`knee_file_list.csv` (github.com/microsoft/fastmri-plus, fetched 2026-10-11), `side_model.pt`.

```bash
python pretrain_fastmri.py --work work --tars fm_raw/knee_singlecoil_val.tar.xz fm_raw/knee_singlecoil_train.tar.xz \
  --boxes work/fmplus/knee.csv --reviewed work/fmplus/knee_file_list.csv --side-model work/side_model.pt \
  --out work/fm_encoder.pt
python pretrain_fastmri.py --work work --tars none --boxes work/fmplus/knee.csv \
  --reviewed work/fmplus/knee_file_list.csv --side-model work/side_model.pt --out work/fm_encoder7.pt \
  --reuse --more-targets
```

The first run converts the volumes (its encoder is not used). 1,102 of 1,172 volumes kept, 4 epochs, 256 px
views. Held-out slice AUC 0.96-0.99 (his log).

## 3. mil-reader (`mil-reader/`)

| Input | Made by |
| --- | --- |
| `labels_D.csv` | `make_targets.py` from our `expert.csv` and fold and split files from Matthew's app (not included); sha256 `aabca615`, checked by both trainers (`--unpinned` accepts another file) |
| `cache384_train/` | `build_cache.py`: every series at 0.4 mm per pixel, 384 px, up to 64 slices; a series with no readable slice is not written, so a rerun tries it again |
| `views_train.csv` | `build_views.py` |
| `laterality_v2.csv` | script not kept: the scan's side tag, or, if the knee is within 20 mm of the scanner centre, a comparison with left and right knee templates (not the side classifier) |
| `fm_encoder7.pt` | the encoder |

One RTX 5090 per seed, `SEED` 0 to 4:

```bash
python train.py --labels work/labels_D.csv --data data --work work --views work/views_train.csv \
  --laterality work/laterality_v2.csv --init-enc work/fm_encoder7.pt --seed ${SEED} --out runs/v72full_s${SEED}
```

It saves EMA weights (a slow running average of the weights) as `fold-1.pt`.

## 4. pf-specialist (`pf-specialist/`)

Sees the whole knee, a 120 mm kneecap box, its 2× zoom from the original pixels, and the kneecap-cartilage
scan if any. Learns PF OA plus 8 OAI MOAKS kneecap-cartilage grades (one OAI row per 4 competition rows).
Band and box sizes: `external/v731/bands.py`.

Inputs: mil-reader's cache, laterality and encoder; `labels_D.csv`; PF OA labels `d4-train-4345.csv` (our
`expert-pf-floor.csv`); and two sets of arrays:

1. **Competition:** header tables, cartilage scans for all 4,403 exams, then the native arrays.
   `native.py` runs Matthew's `external/v731/prepare_native.py` unchanged with your paths (its own are
   folders on his Mac). `build --all` covers only the 4,345 non-gold exams; `build --gold` adds the 58
   expert-labelled exams to the same folder, and `train.py` stops if they are missing.

```bash
python ../../v8prep/headers.py --zip <archive.zip> --out work/headers --splits train
python ../../v731/kseries.py --headers work/headers/series_headers.parquet --labels work/labels_D.csv \
  --include-gold --out work/native/k_series.csv
python native.py --work work --headers work/headers --zip <archive.zip> counts
python native.py --work work --headers work/headers --zip <archive.zip> build --all --workers 22 --gate-tolerance 1
python native.py --work work --headers work/headers --zip <archive.zip> build --gold --workers 22 --gate-tolerance 1
```
2. **OAI:** MOAKS tables `kMRI_SQ_MOAKS_BICL{00,01,03,05,06,08,10}.txt` and the knee MRI of NDA package
   `submission_60350`:

```bash
python oai_join.py --moaks <MOAKS ASCII folder> --nda <NDA index> --out work/oai
python oai_inputs.py prepare --grades work/oai/grades.csv --series work/oai/series.csv \
  --out work/oai/native --source-dir <DICOM tarballs>
```

`oai_join.py` writes 7,706 knee visits. Then, on B200 GPUs, `SEED` 0 to 2:

```bash
python train.py --config e3.json --seed ${SEED} --out runs/E3_s${SEED} --labels work/labels_D.csv \
  --d4 work/d4-train-4345.csv --data data --work work --laterality work/laterality_v2.csv \
  --native-root work/native --encoder work/fm_encoder7.pt --oai-grades work/oai/grades.csv \
  --oai-native work/oai/native
```

`train.py` accepts only our `labels_D.csv`, `d4-train-4345.csv` and `fm_encoder7.pt` (sha256 checks). The
encoder is not in the public models dataset; with a retrained one or your own labels pass `--unpinned`.

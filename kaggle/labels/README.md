# RSNA Knee, 2nd place: training labels (private)

Frozen inputs of the training labels and the 15 tables rebuilt from them. Holds study IDs and report quotes: do not
publish. How they are built: `docs/labels.md` in the solution repo.

- `calls/`: logged model answers. `reader.jsonl` (gpt-6.1-sol, reports), `expert.jsonl` (gpt-6-astra, image panels
  + report), `blind-lateral.jsonl` (gpt-6-astra, lateral images only).
- `partition.json`: the 58 expert-labelled exams and linked exams, left out of the folds.
- `teachers/`: one out-of-fold prediction file per fold for each of five teachers, listed with sha256.
- `mil-reader-oof.csv`: mil-reader's out-of-fold logits (feeds `expert-reader`).
- `tables/`: every label table, as rebuilt from the files above and the competition `train.csv`.

Rebuild (no model call; byte-equal to `tables/`):

    python -m knee.labels export --train train.csv --partition partition.json --reader-log calls/reader.jsonl \
        --expert-log calls/expert.jsonl --blind-lateral-log calls/blind-lateral.jsonl --out tables
    python -m knee.labels folds --train train.csv --partition partition.json \
        --reader-detail tables/reader-detail.jsonl --out tables
    python -m knee.labels mix all --dir tables --train train.csv --teacher-root . --mil-reader-oof mil-reader-oof.csv \
        --teachers teachers/convnext-224.json teachers/convnext-detail-288.json teachers/coatnet-224.json \
                   teachers/effnetv2s-320.json teachers/radimagenet-r50.json

Licence: CC-BY-NC 4.0. Derived from the RSNA competition data; its terms apply.

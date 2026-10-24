# Training labels

The competition labels come from reports; the test is scored on what experts see in the images. So the training
labels are soft and image-aware, from three model passes: the **reader** (a language model reads each report), the
**expert** (a vision model reads image panels plus the report) and the **blind-lateral** read (lateral half only, no
report). The rest is arithmetic: `knee/labels.py` rebuilds every table byte for byte. Training only. Table per
member: [method.md](method.md#inputs-and-models); Matthew's: [external/train/](../external/train/README.md).

## Which table feeds which

```
train.csv ──> reader ──> reader.csv, reader-detail.jsonl ───────────────────────────┐
                │                                                                    │
                └──> folds ──> folds.csv, exclusions.json ──> folds-58.csv            │
                                                                                     │
DICOM ──> panels (Kaggle CPU) ──> expert packs ──> expert ──> expert.csv ────────────┤
                             └──> blind packs  ──> blind-lateral ──> blind-lateral.csv
                                                                                     │
expert.csv + reader p_expert ──> expert-pf ──(+ blind-lateral)──> expert-pf-lateral ─┤
                                                                                     │
expert-pf-lateral + 5 teacher OOF ──> teacher-mix ──> teacher-mix-pf ──> teacher-mix-pf-58
expert-pf-lateral ──> expert-pf-floor ──(+ mil-reader OOF)──> expert-reader ──> expert-reader-58
```

## Steps

Every table: `StudyInstanceUID` + the 12 findings, sorted by ID, values in [0, 1]. Commands are
`python -m knee.labels <command>`; prompts are in `configs/labels/`. "Unclear" = the reader filed that exam's PF OA
wording as `unclear`. Last column: the table scored as a model on the 58 (MEASURED).

| Step | Command | Rule | The 58 |
| --- | --- | --- | ---: |
| reader | `reader`, then `export` | One call per distinct report (4,199 for 4,345 exams): a category and a 0-100 score per finding | 0.899 |
| folds | `folds` | 5 folds; same report text together; balanced by size, language and per-finding yes/no counts; the 58 and 4 linked exams left out | |
| panels | `python -m knee.panels` | Image panels for the two vision passes | |
| expert | `expert`, then `export` | Images + report + reader answers; 0-100 per finding; a missing answer falls back to the reader | 0.942 |
| blind-lateral | `expert --blind-lateral`, then `export` | Lateral half only, no report; Lateral OA kept | |
| expert-pf | `mix expert-pf` | PF OA = 0.5 x expert + 0.5 x reader score / 100 | 0.944 |
| expert-pf-lateral | `mix expert-pf-lateral` | Lateral OA = 0.4 x expert-pf + 0.6 x blind-lateral | 0.948 |
| teacher-mix | `mix teacher-mix` | w x expert-pf-lateral + (1 - w) x mean of 5 teachers; w = 0.7 for ACL, MCL, both menisci, PF OA, Lateral OA, else 0.4 | 0.956 |
| teacher-mix-pf | `mix teacher-mix-pf` | PF OA = max(value, 0.8) where unclear | 0.956 (the 58 cannot see it) |
| teacher-mix-pf-58 | `mix teacher-mix-pf-58` | Adds the 58 with their train.csv 0/1 labels | not scorable |
| expert-pf-floor | `mix expert-pf-floor` | PF OA = 0.6 + 0.4 x value where unclear (order kept) | 0.948 |
| expert-reader | `mix expert-reader` | 0.7 x expert-pf-floor + 0.3 x sigmoid(mil-reader logit) | 0.955 (an earlier mil-reader) |
| expert-reader-58 | `mix expert-reader-58` | Adds the 58, as above | not scorable |

`folds-58.csv` adds the 58 as fold `expert58`, never held out. `mix all` runs every mix in order, in float64.

These scores are exploratory (the rule hunt opened all 58) and resolve about ±0.02 macro AUC (paired bootstrap,
2,000 draws). Out-of-fold scores cannot judge a label change: a key built from one table prefers it.

- **expert-pf:** reports describe kneecap cartilage damage without saying OA; experts call it PF OA, but the image
  read pushed it down. PF OA 0.931 → 0.955.
- **expert-pf-lateral:** the blind lateral read passed its pre-set rule for Lateral OA only (0.856 → 0.898).
- **teacher-mix:** every student beat its teacher on every fold (+0.0044 out of fold for the detail model,
  optimistic: the teachers saw the student's validation fold); public 0.960 → 0.961 (OFFICIAL). A teacher of a
  different architecture added more than a better one of the same kind.
- **Ceiling:** the labels outscore the models on the 58 ([results.md](results.md#the-58-expert-exams-exploratory)).
  Lateral OA labels rose +0.043; the models did not follow.

## Two rules set by looking at the 58

Scores on the 58 of models trained with these rules are optimistic.

1. **PF OA floor and lift** (`teacher-mix-pf`, `expert-pf-floor`): all 12 of the 58 with an `unclear` PF OA report
   were expert-positive (95% lower bound 78%). Floor 0.8 and lift 0.6 + 0.4 x value (group mean 0.54 → 0.82) come
   from that count, not from a score.
2. **The 58 as training rows** (`…-58` tables): final all-data seeds train on them, so are never scored on them.

## The teachers

Five 5-fold tile models on `expert-pf.csv` (gated-attention pooling, 16 views per slot, cosine, weight decay 0.01).
Their 25 out-of-fold files (each exam predicted by a model that never saw it) are frozen inputs, checked by
`teacher_mean` (sha256, one file per fold, full coverage).

| Teacher | Architecture and input |
| --- | --- |
| convnext-224 | ConvNeXt-T (ImageNet-1k), 224 px, 140 mm crop, 15 epochs, lr 2e-4 |
| convnext-detail-288 | ConvNeXt-T (ImageNet-22k, fine-tuned 1k at 384), 288 px detail crop at the knee centre, 10 epochs |
| coatnet-224 | CoAtNet-1 rmlp (ImageNet-12k), 224 px, 140 mm crop; 10 epochs on `expert.csv`, then 10 on `expert-pf.csv` |
| effnetv2s-320 | EfficientNetV2-S (ImageNet-21k), 320 px, 160 mm crop, 15 epochs |
| radimagenet-r50 | RadImageNet ResNet-50, 224 px, 140 mm crop, grey slab channels, 20 epochs, lr 5e-4 |

## The reader

- **Target:** what experts tick on the images. They tick more mild findings than reports state; "on the fence" is
  negative (host rubric, verbatim at the top of `reader.md`).
- **Per finding:** up to two verbatim quotes, a source, a **category**, a reason and **`p_expert`** (0-100, chance an
  expert marks it). Quotes come first, in a strict JSON schema: 0 parse failures on 4,199 reports.
- `not_mentioned` is its own state (silence is not "no"). Present but below the rubric's cut is `probably_no`.
- Code, not the model, maps the category to a value: band bottom + band width × `p_expert` / 100.

| Category | Band | Synovitis |
| --- | --- | --- |
| yes | 0.85-1.00 | same |
| probably_yes | 0.65-0.85 | same |
| unclear | 0.45-0.65 | same |
| probably_no | 0.25-0.45 | 0.25-0.35 |
| not_mentioned | 0.05-0.25 | 0.35-0.45 (most reports skip it) |
| no | 0.00-0.05 | same |

On the 58: category alone 0.874, band + score 0.899.

- Reports are read in their own language (nine), never translated.
- A silent finding moves only by a closed list of inferences (pivot-shift bruises → ACL, at most `probably_yes`;
  hemarthrosis → Effusion). Effusion never implies Synovitis, oedema never Fracture.
- Tuned on 24 of the 58, at most two rounds, rules seen in 2+ exams; the other 34 scored once, after freezing.

## The image expert

- **One call per exam** with the panels, the raw report and the reader's answers (`expert.md`,
  `expert-findings.md`): observations by region first, the report as a colleague's read, decide from the images,
  distinct 0-100 values.
- **Panels** (native size, one window): every slice at 240 px (up to 40 per series), the joint at 512 px, ligaments
  448 px, kneecap 384 px, whole fluid slices; each labelled with series, slice rank, mm position and directions.
  About 277 images and 44k image tokens per exam.

| Label source (MEASURED) | All 58 | Untouched 34 |
| --- | ---: | ---: |
| Report reader | 0.899 | 0.893 |
| Image read without the report | 0.923 | 0.910 |
| That read averaged with the reader | 0.932 | 0.923 |
| **Image expert: images + report + reader** | **0.942** | **0.935** |

Biggest gains where reports are vague: Effusion 0.806 → 0.975, Baker's 0.803 → 0.979. It lost only on PF OA (0.930
against the reader's 0.960), fixed by `expert-pf`. Five central slices instead of all dropped Contusion 1.00 → 0.86.

## Frozen inputs, privacy and calls

- **Frozen:** the three call logs, the 25 teacher files, the mil-reader fold logits (3,906 out-of-fold, 439 a mean of
  its five fold models) and the four prompts. Re-running `reader` or `expert` gives new tables; `export`, `folds` and
  `mix` make no call.
- **Private** (never in Git): train.csv, the expert partition, call logs, packs, every table. Frozen inputs and the
  15 tables: private Kaggle dataset `azamat1ch/rsna-knee-2nd-place-labels`.
- **Calls:** endpoint `CLIPROXY_BASE_URL` (default `http://127.0.0.1:18317/v1`), key `CLIPROXY_API_KEY` or
  `OPENAI_API_KEY`; reader `gpt-6.1-sol`, expert `gpt-6-astra`, effort high; one JSON line logged per call. Sending
  competition data to an outside API was allowed (OFFICIAL, thread 733965); test images never left Kaggle.
- **Panels:** Kaggle CPU, Pillow pinned at 12.3.0 (PNG bytes depend on it). Packs were deleted after labelling.
- **Cost:** subscription, no per-token bill. Reader 21.7M input + 5.4M output tokens; image expert 61,465 credits
  (about 80 s per call, 15 in parallel); blind lateral 33,616 credits (250 / 1,500 credits per 1M input / output).

## What failed

| Idea | Result (MEASURED unless tagged) |
| --- | --- |
| Zoom variants (bigger joint crops) | +0.002 to +0.005, below the pre-set +0.02 bar |
| Repeat reads; reviews shown the expert's answer | Errors repeat; reviews copy it |
| Blind specialists: Lateral Meniscus, Fracture, Contusion | LM +0.066 on the 24 tuning exams, −0.019 on the 34 |
| Synovitis specialist; a visual Synovitis rule | Blind 0.80 vs label 0.89; shown the answers, reads copy them (0.993) |
| Second round of students as teachers | +0.0004 |
| Rule hunt over all 12 findings | 225 hits vs 221 by chance; one real rule (PF OA) |
| Four more floor rules | Public 0.955 vs its matched partner's 0.957 (OFFICIAL) |
| Public label tables | +0.000 on the 34 |
| Adjudication pass (image, slice viewer, then report) | Won on 6 picked exams, lost on 6 fresh; about 7.4M tokens |
| Five-call image panel | Tied the single call on the 24 at 3.7× the tokens |

## Old names

| New | Old |
| --- | --- |
| `reader.csv`, `reader-detail.jsonl` | report-category-v1.1 (+ detail) |
| `expert.csv` | image-expert-d-v1 (arm D) |
| `blind-lateral.csv` | image-expert-sl-v1 (arm SL) |
| `expert-pf.csv` | image-expert-d2-v1 (D2) |
| `expert-pf-lateral.csv` | image-expert-d2sl6-v1 (D3) |
| `teacher-mix.csv`, `teacher-mix-pf.csv`, `teacher-mix-pf-58.csv` | dmix-teachers5-sl-v1, …-pfu80-v1, …-pfu80-e58-v1 |
| `expert-pf-floor.csv` | image-expert-d4-v1 (D4) |
| `expert-reader.csv`, `expert-reader-58.csv` | image-expert-d6-v1 (D6), image-expert-d6-e58-v1 |
| `folds.csv`, `folds-58.csv`, `exclusions.json` | folds-v2, folds-v2-e58, exclusions-v2 |
| prompts `reader.md`, `expert.md`, `expert-findings.md`, `blind-lateral.md` | report-category-v1.1.md, expert-v2.md, expert-v2-findings.md, expert-v4-specialists.md (FRAME, SL, SCALE) |
| teachers convnext-224, convnext-detail-288, coatnet-224, effnetv2s-320, radimagenet-r50 | cnxt224, detail288, coat, effv2s, rin_r50 |
| crop-convnext, lateral-effnet, window-convnext, window-effnet | r7, LAT1 (r18), S6-Tiny, S6-Eff |
| mil-reader, pf-specialist | V7.2, E3 (teammate's models) |

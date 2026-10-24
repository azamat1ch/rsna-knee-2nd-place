"""labels_D.csv, the label table of mil-reader and the split table of pf-specialist.

One row per training exam (index StudyInstanceUID): `is_gold`, the 12 targets, `fold` and
`gold_<finding>`.

- fold 0-4: the 3,909 weak_train + weak_development exams, targets from our image labels (`--team-d`).
- fold -2: the 440 correction_pool exams.
- fold -1, is_gold 1: the 58 exams with official labels (in `gold_<finding>`).

Exams without an image label are dropped (4). The folds, split manifest and study index come from
Matthew's app (not included).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
HOLDOUT_FOLD = -2
EXPECTED = {"folds": 3909, "holdout": 440, "gold": 58}
MAX_MISSING_D = 10


def study_assignments(folds_path: Path, manifest_path: Path, index_path: Path) -> dict[str, int]:
    """StudyInstanceUID -> fold (0-4) or HOLDOUT_FOLD; fails closed on any gap."""
    folds = {
        r["study_pseudonym"]: int(r["fold"]) for r in json.loads(folds_path.read_text())["records"]
    }
    cohorts = {
        r["study_pseudonym"]: r["cohort"] for r in json.loads(manifest_path.read_text())["records"]
    }
    alias_to_uid = {
        r["study_pseudonym"]: r["source_study_uid"]
        for r in json.loads(index_path.read_text())["records"]
    }
    out: dict[str, int] = {}
    for alias, fold in folds.items():
        if cohorts.get(alias) not in ("weak_train", "weak_development"):
            raise ValueError("a fold study is not in a weak training cohort")
        if not 0 <= fold <= 4:
            raise ValueError("fold outside 0-4")
        out[alias_to_uid[alias]] = fold
    for alias, cohort in cohorts.items():
        if cohort == "correction_pool":
            if alias in folds:
                raise ValueError("a correction_pool study has a fold")
            out[alias_to_uid[alias]] = HOLDOUT_FOLD
    counts = {
        "folds": sum(f >= 0 for f in out.values()),
        "holdout": sum(f == HOLDOUT_FOLD for f in out.values()),
    }
    for key, n in counts.items():
        if n != EXPECTED[key]:
            raise ValueError(f"{key}: expected {EXPECTED[key]} studies, found {n}")
    return out


def build(
    team_d: Path,
    train_csv: Path,
    folds: Path,
    manifest: Path,
    index: Path,
) -> pd.DataFrame:
    assign = study_assignments(folds, manifest, index)
    tr = pd.read_csv(train_csv).set_index("StudyInstanceUID")
    gold = tr[LABELS]
    is_gold = gold.notna().all(axis=1)
    if int(is_gold.sum()) != EXPECTED["gold"]:
        raise ValueError(f"gold: expected {EXPECTED['gold']} studies, found {int(is_gold.sum())}")
    if set(assign) & set(gold.index[is_gold]):
        raise ValueError("a gold study is in a fold or the holdout")

    d = (
        pd.read_csv(team_d)
        .drop_duplicates("StudyInstanceUID")
        .set_index("StudyInstanceUID")[LABELS]
    )
    d = d.astype(float).dropna()
    # team D has no answer for a handful of studies (4 on 2026-10-10); they are dropped, not guessed
    missing = sorted(set(assign) - set(d.index))
    if len(missing) > MAX_MISSING_D:
        raise ValueError(
            f"team D labels are missing for {len(missing)} studies (limit {MAX_MISSING_D})"
        )
    if missing:
        print(f"dropping {len(missing)} studies without a team D label", flush=True)
    ids = pd.Index(sorted(set(assign) - set(missing)), name="StudyInstanceUID")
    d = d.reindex(ids)
    weak = d.copy()
    weak.insert(0, "is_gold", 0)
    weak["fold"] = [assign[s] for s in ids]
    gold_ids = gold.index[is_gold]
    g = pd.DataFrame(np.nan, index=pd.Index(gold_ids, name="StudyInstanceUID"), columns=LABELS)
    g.insert(0, "is_gold", 1)
    g["fold"] = -1
    out = pd.concat([weak, g])
    for c in LABELS:
        out[f"gold_{c}"] = gold[c].reindex(out.index)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--team-d", type=Path, required=True, help="our image labels (expert.csv)")
    ap.add_argument(
        "--train-csv", type=Path, required=True, help="competition train.csv (official gold labels)"
    )
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--study-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = build(
        args.team_d,
        args.train_csv,
        args.folds,
        args.split_manifest,
        args.study_index,
    )
    out.to_csv(args.out)
    print(
        "wrote", args.out, out.shape, "| fold sizes", out.fold.value_counts().sort_index().to_dict()
    )
    print("mean target:", out.loc[out.is_gold == 0, LABELS].mean().round(3).to_dict())


if __name__ == "__main__":
    main()

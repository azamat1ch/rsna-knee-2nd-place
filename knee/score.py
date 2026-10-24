"""The competition metric (mean per-finding AUC) and the submission check.

AUC is the Mann-Whitney statistic: the share of positive/negative pairs ranked in the right order, ties counted
half. Rows are matched by StudyInstanceUID, never by position, so a reordered file scores the same.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from knee import LABELS

ID = "StudyInstanceUID"


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """AUC of scores p for binary targets y; None when one class is missing."""
    pos = int(np.count_nonzero(y == 1))
    neg = len(y) - pos
    if not pos or not neg:
        return None
    order = np.argsort(p, kind="stable")
    s = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    rank_sum = ranks[y[order] == 1].sum()
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def score(pred_ids: list[str], preds: np.ndarray, ids: list[str], targets: np.ndarray,
          mask: np.ndarray | None = None) -> dict:
    """Mean per-finding AUC. targets/mask are (n, 12) in `ids` order; masked-out cells are ignored.
    Returns {"macro": float | None, "per_finding": {finding: auc | None}}; macro is None if any finding is undefined."""
    if len(set(pred_ids)) != len(pred_ids) or set(pred_ids) != set(ids):
        raise ValueError("prediction and target study IDs differ")
    pos = {s: i for i, s in enumerate(pred_ids)}
    p = np.asarray(preds, dtype=float)[[pos[s] for s in ids]]
    y = np.asarray(targets, dtype=float)
    mask = np.ones(y.shape, bool) if mask is None else np.asarray(mask, bool)
    if p.shape != (len(ids), len(LABELS)) or y.shape != p.shape or mask.shape != p.shape:
        raise ValueError("predictions, targets and mask must be (studies, 12)")
    if not np.isin(y[mask], (0, 1)).all():
        raise ValueError("scored targets must be 0 or 1")
    per = {f: auc(y[mask[:, j], j], p[mask[:, j], j]) for j, f in enumerate(LABELS)}
    macro = None if None in per.values() else float(np.mean(list(per.values())))
    return {"macro": macro, "per_finding": per}


def validate(sub: pd.DataFrame, ids: list[str]) -> None:
    """Raise ValueError unless `sub` is a scoreable submission for `ids` (sample order): exact columns, the same
    ids in the same order, finite values in [0, 1], and no constant column when there is more than one exam."""
    if list(sub.columns) != [ID, *LABELS]:
        raise ValueError(f"columns must be {ID} + the 12 findings in sample order")
    if [str(s) for s in sub[ID]] != list(ids):
        raise ValueError("study IDs differ from the sample (set or order)")
    x = sub[list(LABELS)].to_numpy()
    if x.dtype.kind != "f" or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("predictions must be finite floats in [0, 1]")
    constant = [f for j, f in enumerate(LABELS) if len(x) > 1 and (x[:, j] == x[0, j]).all()]
    if constant:
        raise ValueError(f"constant columns: {constant}")

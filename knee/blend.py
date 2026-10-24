"""Rank blend of the member predictions and the atomic submission write.

Only the order of exams within each finding counts for the metric, so every member is turned into ranks scaled
to [0, 1] and the ranks are mixed with fixed weights from `configs/blend.json`. A specialist takes a fixed share
of its own findings; the core members split the rest in their usual ratios. Members are summed in config order,
which keeps the float result identical to the kernel that was scored.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from knee import LABELS

ID = "StudyInstanceUID"


def rank01(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share one value) scaled to [0, 1]; a single exam gets 0.5."""
    r = rankdata(x, method="average")
    return (r - 1) / max(1, len(r) - 1) if len(r) > 1 else np.full(1, 0.5)


def core(config: dict) -> dict[str, float]:
    """Core member -> weight, in config order."""
    return {m["name"]: m["weight"] for m in config["members"] if "weight" in m}


def specialists(config: dict) -> dict[str, tuple[str, ...]]:
    """Specialist member -> its findings, in config order."""
    return {m["name"]: tuple(m["findings"]) for m in config["members"] if "findings" in m}


def weights(present: list[str], finding: str, config: dict) -> dict[str, float]:
    """Weight of every present member on one finding (zero weights included)."""
    weight, spec = core(config), specialists(config)
    own = [k for k in present if k in spec and finding in spec[k]]
    assert len(own) <= 1, (finding, own)
    share = config["specialist_share"] if own else 0.0
    present_core = [k for k in present if k in weight]
    total = sum(weight[p] for p in present_core)
    return {k: share if k in own else 0.0 if k in spec else weight[k] / total * (1 - share) for k in present}


def blend(preds: dict[str, np.ndarray], config: dict, labels: tuple[str, ...] = LABELS) -> np.ndarray:
    """preds: member -> (n, 12) in sample order. Returns (n, 12): per finding, the weighted sum of rank01."""
    order = [m["name"] for m in config["members"]]
    assert set(preds) <= set(order), sorted(set(preds) - set(order))
    present = [k for k in order if k in preds]
    assert sum(k in core(config) for k in present) >= 2, present
    cols = []
    for j, finding in enumerate(labels):
        w = weights(present, finding, config)
        col = 0.0
        for k in present:
            if w[k]:
                col = col + w[k] * rank01(preds[k][:, j])
        cols.append(col)
    return np.stack(cols, 1)


def read_member(path: Path, ids: list[str], labels: tuple[str, ...] = LABELS) -> np.ndarray:
    """A member CSV as float64 (n, 12) in the order of `ids`; the id set must match exactly."""
    d = pd.read_csv(path, dtype={ID: str}).set_index(ID)
    assert d.index.is_unique and set(d.index) == set(ids) and len(d) == len(ids), (path.name, len(d), len(ids))
    x = d.loc[ids, list(labels)].to_numpy(np.float64)
    assert np.isfinite(x).all(), f"{path.name}: non-finite values"
    return x


def frame(ids: list[str], values: np.ndarray, labels: tuple[str, ...] = LABELS) -> pd.DataFrame:
    sub = pd.DataFrame(values, columns=list(labels))
    sub.insert(0, ID, ids)
    return sub


def write_csv(sub: pd.DataFrame, path: Path) -> None:
    """Write `<path>.part`, then rename: the target is either absent, the old file or the complete new one."""
    part = path.with_name(path.name + ".part")
    sub.to_csv(part, index=False)
    os.replace(part, path)

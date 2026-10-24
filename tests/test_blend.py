"""Rank blend: rank01, the weights (specialist share, core x0.8), and exact equality with the scored kernel's blend
on stored fixtures (tests/data/blend_fixture.npz holds member predictions and that kernel's outputs)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee import LABELS
from knee.blend import blend, rank01, read_member, weights, write_csv

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "blend.json").read_text())
CORE = ["crop-convnext", "window-convnext", "window-effnet", "mil-reader"]
ALL = CORE + ["pf-specialist", "lateral-effnet"]


def test_rank01_ties_and_single_exam():
    assert rank01(np.array([0.3])).tolist() == [0.5]
    assert rank01(np.array([0.1, 0.9, 0.5])).tolist() == [0.0, 1.0, 0.5]
    assert rank01(np.array([0.2, 0.2, 0.7, 0.2])).tolist() == [1 / 3, 1 / 3, 1.0, 1 / 3]
    assert rank01(np.array([4.0, 4.0])).tolist() == [0.5, 0.5]


def test_core_weights_without_specialists():
    w = weights(CORE, "ACL", CFG)
    assert w == {"crop-convnext": 0.25, "window-convnext": 0.2, "window-effnet": 0.2, "mil-reader": 0.35}
    assert sum(w.values()) == 1.0


def test_specialist_takes_its_share_and_core_keeps_its_ratios():
    w = weights(ALL, "PF OA", CFG)
    assert w["pf-specialist"] == 0.2 and w["lateral-effnet"] == 0.0
    assert w["crop-convnext"] == pytest.approx(0.25 * 0.8) and w["mil-reader"] == pytest.approx(0.35 * 0.8)
    assert sum(w.values()) == pytest.approx(1.0)
    lateral = weights(ALL, "Lateral OA", CFG)
    assert lateral["lateral-effnet"] == 0.2 and lateral["pf-specialist"] == 0.0
    assert weights(ALL, "ACL", CFG) == {**weights(CORE, "ACL", CFG), "pf-specialist": 0.0, "lateral-effnet": 0.0}


def test_missing_core_member_is_renormalised():
    w = weights(["crop-convnext", "window-convnext", "mil-reader"], "ACL", CFG)
    assert w["mil-reader"] == pytest.approx(0.35 / 0.8) and sum(w.values()) == pytest.approx(1.0)


def test_blend_is_monotone_in_one_member_and_bounded():
    rng = np.random.default_rng(0)
    preds = {k: rng.random((20, 12)) for k in CORE}
    out = blend(preds, CFG)
    assert out.shape == (20, 12) and out.min() >= 0 and out.max() <= 1 + 1e-12
    same = blend({k: (v ** 3 if k == "mil-reader" else v) for k, v in preds.items()}, CFG)
    assert np.array_equal(out, same)  # ranks only


def test_blend_equals_the_scored_kernel_bit_for_bit():
    data = np.load(ROOT / "tests" / "data" / "blend_fixture.npz")
    cases = sorted({k.split("_")[0] for k in data.files})
    assert cases
    for c in cases:
        preds = {m: data[f"{c}_{m}"] for m in ALL if f"{c}_{m}" in data.files}
        assert blend(preds, CFG).tobytes() == data[f"{c}_expected"].tobytes(), c


def test_member_csv_round_trip(tmp_path):
    ids = ["b", "a", "c"]
    values = np.arange(36, dtype=float).reshape(3, 12) / 40
    frame = pd.DataFrame(values[::-1], columns=list(LABELS))
    frame.insert(0, "StudyInstanceUID", ids[::-1])
    write_csv(frame, tmp_path / "m.csv")
    assert not (tmp_path / "m.csv.part").exists()
    assert np.array_equal(read_member(tmp_path / "m.csv", ids), values)
    with pytest.raises(AssertionError):
        read_member(tmp_path / "m.csv", ids + ["d"])

"""Metric (mean per-finding AUC) and submission validator on tiny synthetic data."""
import numpy as np
import pandas as pd
import pytest

from knee import LABELS
from knee.score import auc, score, validate

IDS = ["s1", "s2", "s3", "s4"]
Y = np.tile(np.array([[0], [0], [1], [1]], dtype=float), (1, 12))


def preds():
    p = np.tile(np.array([[0.1], [0.2], [0.7], [0.9]]), (1, 12))  # AUC 1
    p[:, 0] = [0.1, 0.4, 0.35, 0.8]  # 3 of 4 pairs in order: 0.75
    p[:, 1] = 0.5  # all tied: 0.5
    return p


def test_auc_values():
    assert auc(np.array([0, 1]), np.array([0.2, 0.1])) == 0.0
    assert auc(np.array([0, 0, 1]), np.array([0.5, 0.5, 0.5])) == 0.5
    assert auc(np.array([1, 1]), np.array([0.1, 0.2])) is None


def test_mean_per_finding_auc_and_row_order():
    out = score(IDS, preds(), IDS, Y)
    assert out["per_finding"]["ACL"] == pytest.approx(0.75) and out["per_finding"]["MCL"] == pytest.approx(0.5)
    assert out["macro"] == pytest.approx((0.75 + 0.5 + 10) / 12)
    order = [3, 0, 2, 1]
    assert score([IDS[i] for i in order], preds()[order], IDS, Y)["macro"] == out["macro"]


def test_mask_and_undefined_finding():
    mask = np.ones(Y.shape, bool)
    mask[1, 0] = False
    assert score(IDS, preds(), IDS, Y, mask)["per_finding"]["ACL"] == pytest.approx(1.0)
    y = Y.copy()
    y[:, 5] = 0
    out = score(IDS, preds(), IDS, y)
    assert out["macro"] is None and out["per_finding"]["Lateral OA"] is None


def submission(values=None, ids=IDS):
    sub = pd.DataFrame(np.linspace(0.1, 0.9, 48).reshape(4, 12) if values is None else values, columns=list(LABELS))
    sub.insert(0, "StudyInstanceUID", ids)
    return sub


def test_validator_accepts_a_good_file():
    validate(submission(), IDS)


@pytest.mark.parametrize("case", ["missing_id", "order", "range", "nan", "columns", "constant"])
def test_validator_rejects_a_bad_file(case):
    sub = submission()
    if case == "missing_id":
        sub = sub.iloc[:3]
    elif case == "order":
        sub = sub.iloc[::-1]
    elif case == "range":
        sub.loc[0, "ACL"] = 1.2
    elif case == "nan":
        sub.loc[2, "MCL"] = np.nan
    elif case == "columns":
        sub = sub[["StudyInstanceUID", *LABELS[::-1]]]
    elif case == "constant":
        sub["PF OA"] = 0.5
    with pytest.raises(ValueError):
        validate(sub, IDS)


def test_single_exam_may_be_constant():
    validate(submission(np.full((1, 12), 0.5), ["s1"]), ["s1"])

"""V7.3.1 K view: pick the dedicated kneecap-cartilage scan of a study from DICOM series
headers (no pixels).

The rule uses only header fields that the Kaggle test headers also carry (SeriesDescription,
SequenceName, MRAcquisitionType) plus the series plane from ImageOrientationPatient, so training
and inference apply it identically.

Usage: python v731/kseries.py --headers series_headers.parquet --out k_series.csv
           --labels labels_D.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

K_RULE_VERSION = "k-rule-1"
K_RULE = (
    "k-rule-1: plane in {Axial, Sagittal} (from IOP) AND one of: "
    "[cartilage] SeriesDescription contains 'cartilag' (cartilage/cartilago, any acquisition); "
    "[patella3d] description contains 'patella' and MRAcquisitionType=3D; "
    "[watsc] description contains 'watsc' (Philips 3D water-selective cartilage); "
    "[dess] description contains 'de3d' or SequenceName starts with '*de3d' (Siemens DESS); "
    "[vibe_we] description contains 'vibe_we' (Siemens VIBE water excitation). "
    "Generic PD/T2/T1 2D sequences and generic isotropic 3D PD/T2 (SPACE/VISTA/CUBE) are "
    "excluded; anonymised "
    "descriptions ('DummySeriesDesc!') qualify only through SequenceName '*de3d'. "
    "One per study: axial before sagittal, then smaller slice gap (0.01 mm), then more frames, "
    "then series id."
)

_RULES = [
    ("cartilage", lambda d, s, a: "cartilag" in d),
    ("patella3d", lambda d, s, a: "patella" in d and a == "3D"),
    ("watsc", lambda d, s, a: "watsc" in d),
    ("dess", lambda d, s, a: "de3d" in d or s.startswith("*de3d")),
    ("vibe_we", lambda d, s, a: "vibe_we" in d),
]
PLANE_ORDER = {"Axial": 0, "Sagittal": 1}


def _s(v):
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def k_tag(description, sequence_name, acquisition, plane):
    """Rule tag ('cartilage', 'patella3d', 'watsc', 'dess', 'vibe_we') when the series qualifies as
    K, else None."""
    if plane not in PLANE_ORDER:
        return None
    d = re.sub(r"\s+", " ", _s(description).lower())
    s, a = _s(sequence_name).lower(), _s(acquisition).upper()
    for tag, f in _RULES:
        if f(d, s, a):
            return tag
    return None


def select_k(headers):
    """headers: DataFrame with study, series, SeriesDescription, SequenceName, MRAcquisitionType,
    plane, slice_gap,
    n_frames_total. Returns one row per study that has a K candidate (deterministic), columns
    study, series, rule, tag, description, plane, gap_mm, n_frames, n_candidates."""
    h = headers.copy()
    h["tag"] = [
        k_tag(d, s, a, p)
        for d, s, a, p in zip(
            h.SeriesDescription, h.SequenceName, h.MRAcquisitionType, h.plane, strict=True
        )
    ]
    c = h[h.tag.notna()].copy()
    if c.empty:
        return pd.DataFrame(
            columns=[
                "study",
                "series",
                "rule",
                "tag",
                "description",
                "plane",
                "gap_mm",
                "n_frames",
                "n_candidates",
            ]
        )
    c["_po"] = c.plane.map(PLANE_ORDER)
    c["_gap"] = c.slice_gap.astype(float).round(2)
    c["_nf"] = -c.n_frames_total.astype(int)
    c = c.sort_values(["study", "_po", "_gap", "_nf", "series"], kind="mergesort")
    n_cand = c.groupby("study").size()
    best = c.drop_duplicates("study", keep="first")
    return (
        pd.DataFrame(
            {
                "study": best.study.values,
                "series": best.series.values,
                "rule": K_RULE_VERSION,
                "tag": best.tag.values,
                "description": best.SeriesDescription.values,
                "plane": best.plane.values,
                "gap_mm": best.slice_gap.astype(float).round(4).values,
                "n_frames": best.n_frames_total.astype(int).values,
                "n_candidates": n_cand.loc[best.study].values,
            }
        )
        .sort_values("study")
        .reset_index(drop=True)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--labels",
        help="labels_D.csv: keep only studies with is_gold == 0 (gold58 excluded, PF731-DATA-001); "
        "only the StudyInstanceUID/is_gold columns are read",
    )
    ap.add_argument(
        "--include-gold",
        action="store_true",
        help="with --labels: keep gold58 studies too (not round 1)",
    )
    a = ap.parse_args()
    h = pd.read_parquet(a.headers)
    h = h[h.split == a.split]
    if a.labels:
        lab = pd.read_csv(a.labels, usecols=["StudyInstanceUID", "is_gold"])
        keep = lab.StudyInstanceUID if a.include_gold else lab.StudyInstanceUID[lab.is_gold == 0]
        h = h[h.study.isin(set(keep))]
    k = select_k(h)
    if os.path.exists(a.out):
        old = pd.read_csv(a.out, dtype=str)
        new = k.astype(str)
        if not (
            list(old.columns) == list(new.columns)
            and len(old) == len(new)
            and (old.values == new.values).all()
        ):
            sys.exit(f"refusing to overwrite {a.out}: existing content differs")
        print("unchanged", a.out)
        return
    tmp = a.out + f".{os.getpid()}.tmp"
    k.to_csv(tmp, index=False)
    os.link(tmp, a.out)
    os.unlink(tmp)
    print("wrote", a.out, len(k), "studies")


if __name__ == "__main__":
    main()

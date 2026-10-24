"""radiant-kg F: is a series a plain (non-fat-suppressed, non-IR) T1-weighted series? Header fields only, no pixels.

One function, `is_t1`, decides. Training (build_t1_series.py writes the --t1-series CSV from it) and Kaggle inference
(infer_c.py classifies the hidden test series from their DICOM headers with it) both call it, so they agree by
construction. Change the rule here only, and bump T1_RULE: checkpoints record the version they were trained with and
infer_c.py refuses a mismatch.

PROVISIONAL rule (not yet validated against real headers; the supervisor tunes it after the header audit), in order:
1. suppressed -> not T1: ScanningSequence contains IR, InversionTime > 0, ScanOptions contains FS, ImageType contains
   WATER (Dixon water image), or the description says fat-sat / IR (fs not followed by e, fat, spair, spir, stir, tirm,
   chess, token ir, token water).
2. description has a T1 token (t1, t1w, t1_tse, sag_t1 ...; not t10) -> T1.
3. description has a PD / T2 / STIR token (pd, dp, proton, inter(mediate), t2) -> not T1.
4. pure gradient echo (ScanningSequence GR without SE) -> not T1 (knee GRE is mostly T2*/DESS/MEDIC).
5. RepetitionTime <= 800 ms and EchoTime <= 30 ms -> T1.
6. otherwise not T1 (missing parameters never count as T1).
A description starting with "dummyseriesdesc" (anonymised) counts as empty. SequenceVariant is read but not used yet.
"""
import re

T1_RULE = "t1-v0"
FIELDS = ["SeriesDescription", "ScanningSequence", "SequenceVariant", "ScanOptions", "RepetitionTime", "EchoTime",
          "InversionTime", "ImageType"]
TR_MAX, TE_MAX = 800.0, 30.0

_SUPPRESSED = re.compile(r"fs(?!e)|fat|spair|spir|stir|tirm|chess|(?<![a-z0-9])(ir|water)(?![a-z0-9])")
_T1 = re.compile(r"(?<![a-z0-9])t1(?![0-9])")
_OTHER = re.compile(r"(?<![a-z0-9])(pd|dp|proton|inter|t2)")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parts(v):
    """Multi-valued DICOM string (MultiValue, list, or 'SE\\IR') -> set of upper-case parts."""
    if v is None:
        return set()
    items = [v] if isinstance(v, str) else list(v)
    return {p.strip().upper() for s in items for p in str(s).replace("/", "\\").split("\\") if p.strip()}


def t1_header_fields(ds):
    """pydicom dataset (header only) -> {field: value} for is_t1."""
    return {k: ds.get(k) for k in FIELDS}


def is_t1(header_fields):
    """header_fields: dict with any of FIELDS (missing = unknown) -> bool. See the module docstring for the rule."""
    f = header_fields
    d = str(f.get("SeriesDescription") or "").lower()
    if d.startswith("dummyseriesdesc"):
        d = ""
    scan = _parts(f.get("ScanningSequence"))
    ti = _num(f.get("InversionTime"))
    if ("IR" in scan or (ti is not None and ti > 0) or "FS" in _parts(f.get("ScanOptions"))
            or "WATER" in _parts(f.get("ImageType")) or _SUPPRESSED.search(d)):
        return False
    if _T1.search(d):
        return True
    if _OTHER.search(d):
        return False
    if "GR" in scan and "SE" not in scan:
        return False
    tr, te = _num(f.get("RepetitionTime")), _num(f.get("EchoTime"))
    return tr is not None and te is not None and tr <= TR_MAX and te <= TE_MAX


def series_is_t1(headers):
    """headers: header-only pydicom datasets of one series, sorted by file name; the first one decides."""
    return bool(headers) and is_t1(t1_header_fields(headers[0]))


def nonfs_sag_cor(series_df):
    """Rows of a *_series.csv that can land in slots 3/4 (sagittal or coronal, not fat-suppressed): the only series
    whose T1 status matters to build_study_table."""
    keep = (series_df.Fat_Suppression.astype(int) == 0) & series_df.Anatomical_Plane.isin(["Sagittal", "Coronal"])
    return series_df[keep]

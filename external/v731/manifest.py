"""V7.3.1 native manifest: the ONE interpretation every reader uses (PF731-DATA-002).

`native/manifest.csv` is append-only: each build attempt appends one row per (series, kind)
and earlier rows (including failures) are never deleted. The LAST row per (series, kind) in
file order is the current state. A current row is usable only when its status is "ok", its
bands_version equals bands.BANDS_VERSION AND its
array exists with the recorded sha256, dtype uint8 and shape [n_slab, crop, crop]
(bands-3, 120 mm box: P 300, Z/K 600).

API:
  MANIFEST_COLS, TERMINAL_STATUSES, OK_STATUS
  read_manifest(path) -> DataFrame (all rows, file order; empty frame when the file is absent)
  current_rows(path_or_df) -> DataFrame (last row per (series, kind))
  MIN_BAND_SLABS (1 = rule disabled), BAND_TOO_THIN ("band_too_thin"), too_thin(n_slab)
  row_state(status, n_slab=None) -> "ok" | "terminal" | "failed" (ok with n_slab < MIN_BAND_SLABS
      -> terminal; never at MIN_BAND_SLABS = 1)
  same_bands(row) -> bool (row's bands_version == bands.BANDS_VERSION)
  usable_reason(row, root, check_sha=True) -> "" when usable, else the reason (status, bands
      version, "band_too_thin" (terminal), missing array, shape/dtype, sha256)
  is_usable(row, root, check_sha=True) -> bool
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from bands import BANDS_VERSION, P_CROP, Z_CROP

MANIFEST_COLS = [
    "study", "series", "kind", "plane", "slot", "gap_mm", "slab_m", "n_raw_total", "n_raw_band",
    "n_slab", "band_lo_mm", "band_hi_mm", "r0", "c0", "ps_row", "ps_col", "lo", "hi", "sha256",
    "status", "slab_members", "note", "bands_version",
]  # fmt: skip
OK_STATUS = "ok"
# terminal = the input is truly absent; never retried. Any other non-ok status (decode_error, which
# also records write/IO exceptions) is retryable and rebuilt on the next resume pass.
TERMINAL_STATUSES = frozenset({"missing_geometry", "empty_band", "missing_series"})
CROP = {"P": P_CROP, "Z": Z_CROP, "K": Z_CROP}  # bands-3: 300 / 600 / 600 (120 mm box)
# Codex SHEETS-003 proposed treating ok rows with fewer than 3 slabs as a terminal absence
# (band_too_thin). DISABLED per review (2026-10-16): a slab count alone does not establish absent
# PF anatomy and would mask valid 1-2 slab bands in K/OAI/unseen exams. MIN_BAND_SLABS = 1 means
# no ok row is ever band_too_thin; the constant, too_thin() and the reason name stay for
# compatibility. Disclosed limitation: 3 one-slab sagittal series of one bilateral study (anchor
# between the knees) remain usable.
MIN_BAND_SLABS = 1
BAND_TOO_THIN = "band_too_thin"


def read_manifest(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLS)
    return pd.read_csv(path, dtype={"series": str, "study": str, "slot": str, "kind": str})


def current_rows(manifest):
    """Last row per (series, kind) in file order (history stays in the file)."""
    man = manifest if isinstance(manifest, pd.DataFrame) else read_manifest(manifest)
    if not len(man):
        return man
    return man.drop_duplicates(["series", "kind"], keep="last")


def same_bands(row):
    """True when the row was built with the current bands.BANDS_VERSION (arrays of different band
    versions are never mixed; rows without a version are rejected)."""
    return row.get("bands_version") == BANDS_VERSION


def too_thin(n_slab):
    """True when a slab count is known and below MIN_BAND_SLABS."""
    return n_slab is not None and not pd.isna(n_slab) and int(n_slab) < MIN_BAND_SLABS


def row_state(status, n_slab=None):
    """'ok' | 'terminal' | 'failed'. Pass n_slab for ok rows: fewer than MIN_BAND_SLABS slabs ->
    'terminal' (band_too_thin)."""
    if status == OK_STATUS:
        return "terminal" if too_thin(n_slab) else "ok"
    return "terminal" if status in TERMINAL_STATUSES else "failed"


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def usable_reason(row, root, check_sha=True):
    """row: mapping with kind, series, status, n_slab, sha256 (a current row). root: native/."""
    if row["status"] != OK_STATUS:
        return f"status {row['status']}"
    if not same_bands(row):
        return f"bands version {row.get('bands_version')!r} != {BANDS_VERSION!r}"
    if too_thin(row.get("n_slab")):
        return BAND_TOO_THIN
    kind, sr = row["kind"], row["series"]
    p = Path(root) / kind / f"{sr}.npy"
    if not p.exists():
        return "array missing"
    try:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        shape, dtype = a.shape, a.dtype
        del a
    except Exception as e:
        return f"unreadable: {type(e).__name__}"
    if dtype != np.uint8 or shape != (int(row["n_slab"]), CROP[kind], CROP[kind]):
        return f"shape/dtype {shape} {dtype}"
    if check_sha and _sha256(p) != row["sha256"]:
        return "sha256 mismatch"
    return ""


def is_usable(row, root, check_sha=True):
    return usable_reason(row, root, check_sha) == ""

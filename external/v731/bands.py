"""V7.3.1 PF specialist: kneecap band, slab and canonical-frame helpers (pure, no I/O).

Rules (CONTRACT.md "Band and slab rules"):
- Band per plane from slice centres (LPS mm, centre of the slice's 0.4 mm canonical frame)
  and the study anchor X:
    axial    centre_z in [X_z - 25, X_z + 70]  (inclusive edges)
    sagittal centre_x in [X_x - 45, X_x + 45]  (inclusive edges)
  (BANDS_VERSION v731-bands-2; v731-bands-1 was -10..+70 / +-35 and is archived in native_bands1/)
- Series gap = median |diff| of the sorted positions of ALL raw slices. Thin series
  (gap < 1.75 mm): consecutive band slices (canonical sorted order) are averaged into slabs
  of m = max(1, round_half_up(3.0 / gap)) slices; the last slab may hold fewer; every band
  slice is in exactly one slab. Thick series: m = 1.
  Rounding: thin/thick uses the unrounded gap; for m the gap is rounded to 0.01 mm and
  3.0 / gap is rounded half-up, so gaps stored as 1.199997 or 1.200001 give the same m (= 3);
  Python's round() would give 2 or 3 depending on float noise.
- Slab averaging is done in float32 on the windowed [0, 1] intensities (same window as
  load_series), then *255 and rounded to uint8 once. A one-slice slab is the load_series
  value exactly.
- Canonical frames: preprocess._canon_slice (orientation, unchanged) + `resample` =
  preprocess._resample with pixel size and frame size as parameters. P frame 0.4 mm / 384 px;
  Z/K frame 0.2 mm / 768 px (same 153.6 mm field).
- Boxes (v731-bands-3, founder 2026-10-16 "switch to 120mm"): 120 mm squares with the same
  intended centre as the views_train.csv / views.crop_boxes 100 mm kneecap box (r0, c0) and the
  same clamp rule: box120(r0, c0) = clamp(r0 + 125 - 150, 0, 84), same for c0. P = 300 px at
  (r0', c0') in the 0.4 mm frame; Z/K = 600 px at (2*r0', 2*c0') in the 0.2 mm frame (the same
  physical 120 mm box). The 120 mm box always contains the 100 mm box. Arrays are stored
  unmirrored (left knees are mirrored at load by the dataset).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

_READER = str(Path(__file__).resolve().parents[1] / "reader")
if _READER not in sys.path:
    sys.path.insert(0, _READER)
os.environ.setdefault("BRANCH_B_MM_PER_PX", "0.4")
os.environ.setdefault("BRANCH_B_SIZE", "384")
import cv2  # noqa: E402
import preprocess  # noqa: E402

assert preprocess.MM_PER_PX == 0.4 and preprocess.SIZE == 384, (
    "bands.py needs the 0.4 mm / 384 px reader defaults"
)

# bands-2 (founder 2026-10-16): axial -25..+70 mm, sagittal +-45 mm; bands-3: same bands, 120 mm box
BANDS_VERSION = "v731-bands-3"
AXIAL_BAND_MM = (-25.0, 70.0)  # relative to the anchor's z (LPS +z = superior)
SAGITTAL_BAND_MM = (-45.0, 45.0)  # relative to the anchor's x
THIN_GAP_MM = 1.75
SLAB_MM = 3.0
BOX_MM = 120.0
VIEW_CROP = 250  # the views.py 100 mm box (views_train.csv kneecap r0, c0) in the 0.4 mm frame
P_MM, P_SIZE, P_CROP = 0.4, 384, 300  # 120 mm / 0.4 mm
Z_MM, Z_SIZE, Z_CROP = 0.2, 768, 600  # 120 mm / 0.2 mm


def band_axis(plane):
    """(world axis index, (lo, hi) offsets) for a plane, or None for coronal (no PF band)."""
    if plane == "Axial":
        return 2, AXIAL_BAND_MM
    if plane == "Sagittal":
        return 0, SAGITTAL_BAND_MM
    return None


def band_limits(plane, anchor):
    """Absolute (lo, hi) mm of the band along its axis for this anchor (LPS x, y, z)."""
    ax, (dlo, dhi) = band_axis(plane)
    return float(anchor[ax]) + dlo, float(anchor[ax]) + dhi


def band_mask(plane, centres, anchor):
    """centres [n, 3] LPS mm (slice frame centres) -> bool [n], inclusive at both edges."""
    ax, _ = band_axis(plane)
    lo, hi = band_limits(plane, anchor)
    v = np.asarray(centres, np.float64)[:, ax]
    return (v >= lo) & (v <= hi)


def series_gap(positions):
    """Median |diff| of sorted slice positions (mm); 0.0 for fewer than two slices (load_series
    'slice_gap')."""
    p = np.sort(np.asarray(positions, np.float64))
    return float(np.median(np.abs(np.diff(p)))) if len(p) > 1 else 0.0


def slab_size(gap):
    """Slices per slab: thick (gap >= 1.75 mm) or degenerate (gap <= 0) series -> 1, else
    round_half_up(3 / gap)."""
    raw = float(gap)
    if (
        not np.isfinite(raw) or raw <= 0 or raw >= THIN_GAP_MM
    ):  # thin/thick on the unrounded gap (contract literal)
        return 1
    g = max(round(raw, 2), 0.01)
    return max(1, math.floor(SLAB_MM / g + 0.5))


def slab_groups(band_idx, m):
    """Consecutive groups of m indices (input order = canonical sorted order); the last may be
    shorter.
    Every index appears in exactly one group."""
    idx = list(band_idx)
    return [idx[i : i + m] for i in range(0, len(idx), m)]


def window(x, lo, hi):
    """load_series intensity window on canonical float frames -> [0, 1] (same expression, same dtype
    rules)."""
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def window_bounds(sub_frames):
    """lo, hi = 1st / 99.5th percentile of the ::4-subsampled 0.4 mm frames of ALL native slices
    (load_series)."""
    lo, hi = np.percentile(np.asarray(sub_frames), [1, 99.5])
    return lo, hi


def slab_average(windowed, groups):
    """windowed [n, H, W] in [0, 1]; groups: lists of row indices into `windowed` -> uint8
    [len(groups), H, W].
    Multi-slice slabs are averaged in float32; single-slice slabs keep the load_series value
    exactly."""
    out = np.empty((len(groups), *tuple(windowed.shape[1:])), np.uint8)
    for k, g in enumerate(groups):
        s = (
            windowed[g[0]]
            if len(g) == 1
            else np.asarray(windowed[g], np.float32).mean(0, dtype=np.float32)
        )
        out[k] = (s * 255).round().astype(np.uint8)
    return out


def window_indices(n):
    """Token windows [j-1, j, j+1] clipped to 0..n-1, for j in 0..n-1 -> int [n, 3]."""
    j = np.arange(n)
    return np.clip(np.stack([j - 1, j, j + 1], 1), 0, max(n - 1, 0))


def token_positions(n):
    """Token position feature = slab index / max(n - 1, 1)."""
    return np.arange(n, dtype=np.float32) / max(n - 1, 1)


def canon_slice(img, iop, ps, plane):
    """preprocess._canon_slice unchanged (its missing-spacing default uses the 0.4 mm / 384 frame
    for every output)."""
    return preprocess._canon_slice(img, iop, ps, plane)


def resample(img, ps, mm_per_px, size):
    """preprocess._resample with pixel size and frame size as parameters (identical code path at 0.4
    / 384)."""
    out = np.zeros((size, size), np.float32)
    h, w = img.shape
    ch = min(h, int(round(size * mm_per_px / ps[0])))  # noqa: RUF046 (preprocess._resample verbatim)
    cw = min(w, int(round(size * mm_per_px / ps[1])))  # noqa: RUF046
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0 : y0 + ch, x0 : x0 + cw])
    nh = max(1, min(size, int(round(ch * ps[0] / mm_per_px))))  # noqa: RUF046
    nw = max(1, min(size, int(round(cw * ps[1] / mm_per_px))))  # noqa: RUF046
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    res = cv2.resize(crop, (nw, nh), interpolation=interp)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy : oy + nh, ox : ox + nw] = res
    return out


def frame_p(img_canon, ps_canon):
    return resample(img_canon, ps_canon, P_MM, P_SIZE)


def frame_z(img_canon, ps_canon):
    return resample(img_canon, ps_canon, Z_MM, Z_SIZE)


def box120(r0, c0):
    """100 mm views box top-left (r0, c0) in the 0.4 mm / 384 frame -> 120 mm box top-left with the
    same intended centre (r0 + 125) and the same clamp rule (inside the frame): clamp(centre - 150,
    0, 384 - 300). The result always contains the 100 mm box."""
    half_old, half_new, hi = VIEW_CROP // 2, P_CROP // 2, P_SIZE - P_CROP
    return tuple(int(min(max(int(v) + half_old - half_new, 0), hi)) for v in (r0, c0))


def cut_p(frame, r0, c0):
    """300 px (120 mm) kneecap box of a 0.4 mm / 384 frame at the box120 top-left (r0, c0)."""
    r0, c0 = int(r0), int(c0)
    assert 0 <= r0 <= P_SIZE - P_CROP and 0 <= c0 <= P_SIZE - P_CROP, (r0, c0)
    return frame[..., r0 : r0 + P_CROP, c0 : c0 + P_CROP]


def cut_z(frame, r0, c0):
    """600 px box at (2*r0, 2*c0) of a 0.2 mm / 768 frame: the same physical 120 mm box as cut_p."""
    r0, c0 = int(r0), int(c0)
    assert 0 <= r0 <= P_SIZE - P_CROP and 0 <= c0 <= P_SIZE - P_CROP, (r0, c0)
    return frame[..., 2 * r0 : 2 * r0 + Z_CROP, 2 * c0 : 2 * c0 + Z_CROP]

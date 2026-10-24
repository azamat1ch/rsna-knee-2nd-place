"""Canonical MRI volumes: one DICOM-series -> uint8 volume transform for training and inference.

Adapted from ``preprocess.py`` of the public 2.5D ConvNeXt knee reader (Kaggle notebook
"rsna-knee-stack-2-5d-convnext-mil-lb-0-944", Part B). The only intended difference is
resizing with Pillow float32 instead of OpenCV, so the module needs no extra dependency.

Canonical means: slices sorted along the slice normal, one in-plane orientation per plane
(from ImageOrientationPatient), a fixed physical pixel size, a centred fixed field of view
and one intensity window per series. Training caches and Kaggle inference must both call
``canonical_volume`` so the model never sees two different preprocessings.
"""

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

MM_PER_PX = 0.4
SIZE = 384  # 153.6 mm field of view at 0.4 mm/px
MAX_SLICES = 64
# v2 (all-data) safety bound: never silently subsample, but refuse pathologically deep series.
MAX_CANONICAL_SLICES = 512
# Below this median inter-slice gap a series is treated as thin and slab-averaged in v2.
SLAB_AVERAGE_GAP_MM = 2.0
PLANES = ("Sagittal", "Coronal", "Axial")

_KEEP_MIRRORED = np.ascontiguousarray
_LEFT_WORDS = frozenset({"L", "LT", "LEFT"})
_RIGHT_WORDS = frozenset({"R", "RT", "RIGHT"})
_WORD_RE = re.compile(r"[A-Z]+")

# Contrast-type tagging is a documented heuristic (docs/DECISIONS.md), never an accuracy claim.
# Values are ordered by description-matching priority; "unknown" is the explicit fallback.
CONTRAST_UNKNOWN = "unknown"
CONTRASTS = ("T1", "PD", "T2", "GRE", "STIR")
_CONTRAST_DESCRIPTION_FIELDS = ("SeriesDescription", "ProtocolName", "SequenceName")
_CONTRAST_DESCRIPTION_WORDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("T1", frozenset({"T1"})),
    ("PD", frozenset({"PD", "PROTON"})),
    ("T2", frozenset({"T2"})),
    ("GRE", frozenset({"GRE", "DESS", "FLASH", "MEDIC", "MERGE", "FFE", "SPGR", "WATS"})),
    ("STIR", frozenset({"STIR", "TIRM", "IR"})),
)
_CONTRAST_TOKEN_RE = re.compile(r"[A-Z0-9]+")
# Weighting/contrast suffixes glued to a base token in real protocol names (T1WI, PDW, T2FS).
_CONTRAST_SUFFIXES = frozenset({"W", "WI", "FS", "FAT", "WEIGHTED"})
# Physics thresholds: TR/TE in milliseconds, InversionTime in milliseconds.
_STIR_INVERSION_TIME_MS = 250.0

# Normal-axis index -> (slice-order sign, (column axis, sign), (row axis, sign)), LPS coordinates.
# Sagittal: slices right->left, columns anterior->posterior, rows superior->inferior.
# Coronal:  slices anterior->posterior, columns patient-right->left, rows superior->inferior.
# Axial:    slices superior->inferior, columns patient-right->left, rows anterior->posterior.
CANON: dict[int, tuple[int, tuple[int, int], tuple[int, int]]] = {
    0: (+1, (1, +1), (2, -1)),
    1: (+1, (0, +1), (2, -1)),
    2: (-1, (0, +1), (1, +1)),
}

_Slice = tuple[
    np.ndarray, list[float] | None, list[float] | None, list[float] | None, float, Any
]


def _read(payload: bytes) -> list[_Slice]:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(payload))
    arr = np.asarray(ds.pixel_array, dtype=np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, intercept = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and intercept is not None:
        arr = arr * float(slope) + float(intercept)
    iop = (
        [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    )
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    instance = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    # Keep the parsed dataset on each slice so side evidence reuses this parse (no second read).
    return [(frame, iop, ipp, ps, instance + i * 1e-3, ds) for i, frame in enumerate(frames)]


def _normalise_side(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper()
    if token in _LEFT_WORDS:
        return "L"
    if token in _RIGHT_WORDS:
        return "R"
    return None


def _description_side(ds: Any) -> str | None:
    """Whole-word side from the descriptive fields; matches both sides -> None."""
    left = right = False
    for field in ("SeriesDescription", "StudyDescription", "ProtocolName"):
        text = ds.get(field) if hasattr(ds, "get") else getattr(ds, field, None)
        if not text:
            continue
        for word in _WORD_RE.findall(str(text).upper()):
            left = left or word in _LEFT_WORDS
            right = right or word in _RIGHT_WORDS
    if left and right:
        return None
    if left:
        return "L"
    if right:
        return "R"
    return None


def _explicit_side(ds: Any) -> str | None:
    for tag in ("Laterality", "ImageLaterality"):
        side = _normalise_side(ds.get(tag) if hasattr(ds, "get") else getattr(ds, tag, None))
        if side is not None:
            return side
    return _description_side(ds)


def _image_center_x(ds: Any) -> float | None:
    try:
        ipp = [float(v) for v in ds.ImagePositionPatient]
        iop = [float(v) for v in ds.ImageOrientationPatient]
        columns = float(ds.Columns)
        rows = float(ds.Rows)
        spacing = [float(v) for v in ds.PixelSpacing]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return ipp[0] + iop[0] * columns * spacing[1] / 2 + iop[3] * rows * spacing[0] / 2


def series_side_evidence(headers: Sequence[Any]) -> tuple[str | None, float | None]:
    """Explicit knee side and median image-centre x (mm) for one series' DICOM headers.

    ``headers`` are pydicom datasets (parsed once, header-only for the table builder;
    already parsed by ``_read`` for the cache path). Conflicting explicit sides across
    slices yield ``None``.
    """
    explicit: set[str] = set()
    centre_xs: list[float] = []
    for ds in headers:
        side = _explicit_side(ds)
        if side is not None:
            explicit.add(side)
        centre_x = _image_center_x(ds)
        if centre_x is not None:
            centre_xs.append(centre_x)
    side = next(iter(explicit)) if len(explicit) == 1 else None
    centre = float(np.median(centre_xs)) if centre_xs else None
    return side, centre


def resolve_knee_side(evidence: Sequence[tuple[str | None, float | None]]) -> tuple[str, str]:
    """Resolve one study's side from its series' ``series_side_evidence`` values.

    Returns ``(side, source)`` with side in {"L", "R", "unknown"} and source in
    {"tag", "geometry", "unknown"}: unanimous explicit tags win; otherwise the sign of
    the median series centre-x (in LPS +x = patient LEFT); otherwise unknown.
    """
    explicit = {side for side, _ in evidence if side is not None}
    if len(explicit) == 1:
        return next(iter(explicit)), "tag"
    centre_xs = [centre for _, centre in evidence if centre is not None]
    if centre_xs:
        median = float(np.median(centre_xs))
        if median > 0:
            return "L", "geometry"
        if median < 0:
            return "R", "geometry"
    return "unknown", "unknown"


def _field(ds: Any, name: str) -> Any:
    return ds.get(name) if hasattr(ds, "get") else getattr(ds, name, None)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _token_matches(token: str, term: str) -> bool:
    if token == term:
        return True
    if token.startswith(term):
        return token[len(term) :] in _CONTRAST_SUFFIXES
    return False


def _contrast_from_description(ds: Any) -> str | None:
    """Whole-word contrast from the descriptive fields, by fixed priority order."""
    tokens: set[str] = set()
    for field in _CONTRAST_DESCRIPTION_FIELDS:
        text = _field(ds, field)
        if text:
            tokens.update(_CONTRAST_TOKEN_RE.findall(str(text).upper()))
    for contrast, vocabulary in _CONTRAST_DESCRIPTION_WORDS:
        if any(_token_matches(token, term) for token in tokens for term in vocabulary):
            return contrast
    return None


def _sequence_tokens(ds: Any) -> list[str]:
    value = _field(ds, "ScanningSequence")
    if value is None:
        return []
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return [str(item).upper() for item in value]
    return [str(value).upper()]


def _contrast_from_physics(ds: Any) -> str | None:
    """DICOM physics fallback: gradient-echo, STIR inversion, then a TR/TE weighting rule."""
    sequences = _sequence_tokens(ds)
    if any(token.startswith("GR") for token in sequences):
        return "GRE"
    inversion_time = _as_float(_field(ds, "InversionTime"))
    if (
        any(token.startswith("IR") for token in sequences)
        and inversion_time is not None
        and inversion_time < _STIR_INVERSION_TIME_MS
    ):
        return "STIR"
    repetition_time = _as_float(_field(ds, "RepetitionTime"))
    echo_time = _as_float(_field(ds, "EchoTime"))
    if repetition_time is not None and echo_time is not None:
        if repetition_time < 1000 and echo_time < 30:
            return "T1"
        if repetition_time >= 1500 and echo_time < 50:
            return "PD"
        if repetition_time >= 1500 and echo_time >= 60:
            return "T2"
    return None


def _slice_contrast(ds: Any) -> tuple[str, str]:
    described = _contrast_from_description(ds)
    if described is not None:
        return described, "description"
    physical = _contrast_from_physics(ds)
    if physical is not None:
        return physical, "physics"
    return CONTRAST_UNKNOWN, "unknown"


def series_contrast_evidence(headers: Sequence[Any]) -> tuple[str, str]:
    """Most common contrast type across a series' DICOM headers, with its resolution source.

    ``headers`` are pydicom datasets (or any object supporting ``.get``). Per-slice the
    description wins over the physics fallback; a tie for the most common contrast resolves
    to ``unknown``. Returns ``(contrast, source)`` with contrast in ``CONTRASTS`` plus
    ``unknown`` and source in {"description", "physics", "unknown"}.
    """
    counts: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    for ds in headers:
        contrast, source = _slice_contrast(ds)
        counts[contrast] += 1
        sources[contrast][source] += 1
    if not counts:
        return CONTRAST_UNKNOWN, "unknown"
    winner, top = counts.most_common(1)[0]
    if sum(1 for value in counts.values() if value == top) > 1:
        winner = CONTRAST_UNKNOWN
    source_counts = sources[winner]
    if not source_counts:
        return winner, "unknown"
    return winner, source_counts.most_common(1)[0][0]


def series_contrast(headers: Sequence[Any]) -> str:
    """Most common contrast type for one series, or ``unknown``.

    See ``series_contrast_evidence`` for the heuristic and its thresholds.
    """
    return series_contrast_evidence(headers)[0]


def mirror_to_right(volume: np.ndarray, plane_name: str, side: str) -> np.ndarray:
    """Mirror a LEFT-knee volume across the patient left-right axis so it looks right-sided.

    Volumes are canonical ``[slice, row, col]`` arrays: sagittal runs right->left along
    the slice axis, coronal and axial run patient-right->left along the column axis. Pure
    and its own inverse for side "L"; "R" and "unknown" pass through unchanged.
    """
    if side != "L":
        return np.ascontiguousarray(volume)
    if plane_name == "Sagittal":
        return np.ascontiguousarray(volume[::-1])
    if plane_name in ("Coronal", "Axial"):
        return np.ascontiguousarray(volume[:, :, ::-1])
    return np.ascontiguousarray(volume)


def _canon_slice(
    img: np.ndarray,
    iop: list[float] | None,
    ps: list[float] | None,
    plane: int,
    size: int,
    mm_per_px: float,
) -> tuple[np.ndarray, list[float]]:
    """Reorient one slice to the canonical in-plane orientation; returns (img, [row_mm, col_mm])."""
    spacing = ps or [mm_per_px * img.shape[0] / size] * 2  # no spacing: assume it spans the crop
    if iop is None:
        return img, spacing
    row_dir, col_dir = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(row_dir[cax]) < abs(row_dir[rax]):  # rows and columns swapped relative to canonical
        img, row_dir, col_dir, spacing = img.T, col_dir, row_dir, spacing[::-1]
    if row_dir[cax] * csgn < 0:
        img = img[:, ::-1]
    if col_dir[rax] * rsgn < 0:
        img = img[::-1]
    return img, spacing


def _resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    # Pillow widens the bilinear kernel when shrinking, so downscaling is antialiased.
    source = Image.fromarray(np.ascontiguousarray(img, dtype=np.float32))
    return np.asarray(source.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def _resample(img: np.ndarray, spacing: list[float], size: int, mm_per_px: float) -> np.ndarray:
    """Centre-crop to size*mm_per_px mm, resample to mm_per_px, zero-pad to size x size."""
    out = np.zeros((size, size), np.float32)
    h, w = img.shape
    ch = min(h, round(size * mm_per_px / spacing[0]))
    cw = min(w, round(size * mm_per_px / spacing[1]))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    nh = max(1, min(size, round(ch * spacing[0] / mm_per_px)))
    nw = max(1, min(size, round(cw * spacing[1] / mm_per_px)))
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy : oy + nh, ox : ox + nw] = _resize(crop, nh, nw)
    return out


def _slab_average(
    frames: list[np.ndarray], positions: Sequence[float], slab_mm: float
) -> list[np.ndarray]:
    """Average adjacent thin slices into ``slab_mm`` bins; thick series pass through.

    Thin series (median inter-slice gap below ``SLAB_AVERAGE_GAP_MM``) have their slices
    grouped into slabs of ``slab_mm`` and averaged so every slice enters exactly one slab
    and none is discarded. Thick series are returned unchanged, preserving 3 mm scans.
    """
    if slab_mm <= 0 or len(frames) < 2:
        return frames
    pos = np.asarray(positions, dtype=np.float64)
    gaps = np.abs(np.diff(pos))
    if gaps.size == 0 or float(np.median(gaps)) >= SLAB_AVERAGE_GAP_MM:
        return frames
    origin = float(pos.min())
    bins = np.floor((pos - origin) / slab_mm + 1e-6).astype(np.int64)
    grouped: dict[int, list[np.ndarray]] = {}
    for index, frame in enumerate(frames):
        grouped.setdefault(int(bins[index]), []).append(frame)
    return [
        np.mean(np.stack(grouped[bucket]), axis=0).astype(np.float32)
        for bucket in sorted(grouped)
    ]


def canonical_volume(
    payloads: Sequence[bytes],
    plane_name: str | None = None,
    *,
    size: int = SIZE,
    mm_per_px: float = MM_PER_PX,
    fov_mm: float | None = None,
    slab_mm: float | None = None,
    max_slices: int | None = MAX_SLICES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (uint8 volume [n, size, size], info). Unreadable slices are skipped and counted.

    Defaults reproduce the original fixed 153.6 mm / 64-slice behaviour byte-for-byte. The v2
    all-data path passes ``fov_mm`` (which overrides ``mm_per_px``), ``slab_mm`` for thin-series
    slab averaging, and ``max_slices=None`` so every slice is kept; series deeper than
    ``MAX_CANONICAL_SLICES`` then fail closed instead of subsampling silently.
    """
    if fov_mm is not None:
        if fov_mm <= 0:
            raise ValueError("fov_mm must be positive")
        mm_per_px = fov_mm / size
    slices: list[_Slice] = []
    bad = 0
    for payload in payloads:
        try:
            slices.extend(_read(payload))
        except Exception:
            bad += 1
    info: dict[str, Any] = {"n_raw": len(slices), "n_bad": bad}
    series_side, image_center_x = series_side_evidence([s[5] for s in slices])
    info["series_side"] = series_side
    info["image_center_x"] = image_center_x
    if not slices:
        return np.zeros((1, size, size), np.uint8), info

    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
    else:
        normal = None
        plane = PLANES.index(plane_name) if plane_name in PLANES else 0
    info["plane"] = PLANES[plane]

    positions: list[float] | None = None
    if normal is not None and all(s[2] is not None for s in slices):
        sign = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sign)  # type: ignore[arg-type]
        positions = [float(np.dot(s[2], normal)) for s in slices]  # type: ignore[arg-type]
        info["slice_gap"] = (
            float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0
        )
        info["ordering"] = "spatial"
    else:
        slices.sort(key=lambda s: s[4])
        info["ordering"] = "instance_number"
    if len(slices) > MAX_CANONICAL_SLICES:
        raise ValueError("series exceeds the maximum supported slice count")
    if max_slices is not None and len(slices) > max_slices:
        keep = np.linspace(0, len(slices) - 1, max_slices).round().astype(int)
        slices = [slices[i] for i in keep]
        if positions is not None:
            positions = [positions[i] for i in keep]

    frames = [
        _resample(*_canon_slice(s[0], s[1], s[3], plane, size, mm_per_px), size, mm_per_px)
        for s in slices
    ]
    if slab_mm is not None and positions is not None:
        frames = _slab_average(frames, positions, slab_mm)
        info["slab_mm"] = slab_mm
    volume = np.stack(frames)
    # One window per series (padding included, exactly as the reference reader does).
    lo, hi = np.percentile(volume[:, ::4, ::4], [1, 99.5])
    volume = np.clip((volume - lo) / max(float(hi - lo), 1e-6), 0, 1)
    info["n"] = len(volume)
    return (volume * 255).round().astype(np.uint8), info

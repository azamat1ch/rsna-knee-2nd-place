"""Series and slices: usable series, the 6 slots, prefer-2d, thin slabs, tiles, knee side, stack positions.

Pure functions of header records (the 448 store's series_metadata.jsonl rows, or dicom.Series.record()),
so the store reader and raw-DICOM inference choose the same series and slices. No pixels here.
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PLANES = ("sagittal", "coronal", "axial")
SLOTS = tuple((plane, fluid) for plane in PLANES for fluid in (0, 1))
PLANE_MIN_COSINE = 0.9          # |normal| along the plane's axis
PLANE_MARGIN = 0.2              # over the second largest axis
ORIENTATION_TOL = 0.001
SPACING_TOL_MM = 1e-6
DUPLICATE_MM = 1e-4
GAP_FACTOR = 1.5
PREFER_2D_THIN_MM = 2.0         # prefer-2d: a series thinner than this is "3D" (SliceThickness, else spacing)
TILE_STRIDE = 3                 # a tile is 3 neighbouring slices (or slabs); tiles start every 3
TILE_SAMPLER = "tile-random-offset-v1"   # part of the training draw's hash key
DEAD_ZONE_MM = 20.0             # knee side: geometry decides beyond 20 mm from x = 0
UNTAGGED_MIN_MM = 40.0          # with no Laterality tag geometry must be beyond 40 mm
AXIS_COSINE = 0.8               # an in-plane axis "runs along x" above this cosine
POSITION_FREQUENCIES = 8
SAGITTAL_DIRECTION = {"right": (1.0, 0.0, 0.0), "left": (-1.0, 0.0, 0.0)}
PLANE_DIRECTION = {"coronal": (0.0, 1.0, 0.0), "axial": (0.0, 0.0, -1.0)}


@dataclass(frozen=True)
class Candidate:
    """A usable series in its slot. sources: source indices (store rows, in slice order); triplets: three
    adjacent sources (or, after `slabs`, three adjacent slabs, each a tuple of sources); window: the crop
    window (cy, cx, side_mm) on the 448 canvas once centre.crop_boxes set it."""
    record: dict
    slot: tuple[str, int]
    signature: tuple[str, int, int, str]   # plane, fluid, fat suppression, contrast
    sources: tuple[int, ...]
    triplets: tuple[tuple, ...]
    spacing_area: float
    window: tuple[float, float, float] | None = None

    @property
    def series(self) -> str:
        return self.record["series"]

    @property
    def plane(self) -> str:
        return self.slot[0]


@dataclass(frozen=True)
class Selection:
    """One exam's 6 slots (SLOTS order); None = no usable series."""
    study_id: str
    slots: tuple[Candidate | None, ...]

    @property
    def slot_mask(self) -> np.ndarray:
        return np.asarray([v is not None for v in self.slots], dtype=bool)

    def with_slots(self, slots) -> "Selection":
        return dataclasses.replace(self, slots=tuple(slots))


def read_series_table(path: str | Path) -> dict[str, dict[str, dict]]:
    """train/test_series.csv as {study: {series: row}}."""
    table = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            table.setdefault(row["StudyInstanceUID"], {})[row["SeriesInstanceUID"]] = row
    return table


def _flag(value, name):
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value in ("0", "1"):
        return int(value)
    raise ValueError(f"{name}: expected 0 or 1")


def _indices(values, n, name):
    if not isinstance(values, list) or any(type(v) is not int for v in values):
        raise ValueError(f"{name}: integer source indices expected")
    if len(set(values)) != len(values) or any(v < 0 or v >= n for v in values):
        raise ValueError(f"{name}: duplicate or out-of-range source indices")
    return values


def candidate(record: dict, metadata: dict) -> Candidate:
    """Check one series record against its CSV row and make its triplets; ValueError when unusable.

    Refused: unresolved or irregular geometry, under 3 slices, mixed shape/spacing/orientation/contrast/frame,
    an uncertain plane or one the CSV disagrees with, duplicate positions or a gap over 1.5 x the median.
    A decode hole is allowed but never bridged: triplets are consecutive slices that all decoded."""
    if record.get("order_rule") != "geometry" or record.get("issues") != []:
        raise ValueError("unresolved or irregular geometry")
    headers = record["headers"]
    n = len(headers)
    if n < 3:
        raise ValueError("fewer than three slices")
    order = _indices(record["ordered_source_indices"], n, "order")
    if len(order) != n:
        raise ValueError("order must be a full permutation")
    iops = np.asarray([h["iop"] for h in headers], dtype=float)
    ipps = np.asarray([h["ipp"] for h in headers], dtype=float)
    positions = np.asarray(record["positions_source_order"], dtype=float)
    spacing = np.asarray([h["spacing"] for h in headers], dtype=float)
    shapes = np.asarray([h["shape"] for h in headers])
    if iops.shape != (n, 6) or ipps.shape != (n, 3) or positions.shape != (n,):
        raise ValueError("invalid geometry shape")
    if not all(np.isfinite(v).all() for v in (iops, ipps, positions, spacing)):
        raise ValueError("nonfinite geometry")
    if spacing.shape != (n, 2) or np.any(spacing <= 0):
        raise ValueError("invalid in-plane spacing")
    if shapes.shape != (n, 2) or shapes.dtype.kind not in "iu" or np.any(shapes <= 0):
        raise ValueError("invalid slice shape")
    if (not np.array_equal(shapes, np.broadcast_to(shapes[0], shapes.shape))
            or not np.allclose(spacing, spacing[0], atol=SPACING_TOL_MM, rtol=0)):
        raise ValueError("mixed slice shapes or spacing")
    if not np.allclose(iops, iops[0], atol=ORIENTATION_TOL, rtol=0):
        raise ValueError("mixed orientation")
    r, c = iops[0, :3], iops[0, 3:]
    if (abs(np.linalg.norm(r) - 1) > ORIENTATION_TOL or abs(np.linalg.norm(c) - 1) > ORIENTATION_TOL
            or abs(r @ c) > ORIENTATION_TOL):
        raise ValueError("invalid orientation basis")
    normal = np.cross(r, c)
    axes = np.sort(np.abs(normal))
    if axes[-1] < PLANE_MIN_COSINE or axes[-1] - axes[-2] < PLANE_MARGIN:
        raise ValueError("ambiguous plane")
    plane = PLANES[int(np.argmax(np.abs(normal)))]
    csv_plane = metadata["Anatomical_Plane"]
    if not isinstance(csv_plane, str) or csv_plane.lower() != plane:
        raise ValueError("CSV plane disagrees with the geometry")
    fluid = _flag(metadata["Fluid_Sensitive"], "Fluid_Sensitive")
    fat = _flag(metadata["Fat_Suppression"], "Fat_Suppression")
    if not np.allclose(positions, ipps @ normal, atol=ORIENTATION_TOL, rtol=0):
        raise ValueError("position metadata mismatch")
    gaps = np.diff(positions[order])
    if np.any(gaps <= DUPLICATE_MM) or np.any(gaps > GAP_FACTOR * np.median(gaps)):
        raise ValueError("duplicate positions or large gap")
    if not np.isclose(record["spacing_z"], np.median(gaps), atol=ORIENTATION_TOL, rtol=0):
        raise ValueError("through-plane spacing mismatch")
    contrasts = {h.get("contrast", "unknown") for h in headers}
    frames = {h.get("frame_uid", "") for h in headers}
    if len(contrasts) != 1 or len(frames) != 1:
        raise ValueError("mixed contrast or frame of reference")
    contrast = next(iter(contrasts))
    if not isinstance(contrast, str) or not contrast:
        raise ValueError("invalid contrast")
    decoded = _indices(record["decoded_source_indices"], n, "decoded")
    sources = _indices(record["full_fov_source_indices"], n, "stored")
    if not set(sources).issubset(decoded):
        raise ValueError("stored sources not decoded")
    rank = {s: k for k, s in enumerate(order)}
    for name, mapping in (("decoded", decoded), ("stored", sources)):
        if any(rank[a] >= rank[b] for a, b in zip(mapping, mapping[1:])):
            raise ValueError(f"{name} order mismatch")
    available = set(sources)
    triplets = tuple(tuple(order[k - 1:k + 2]) for k in range(1, n - 1) if set(order[k - 1:k + 2]) <= available)
    if not triplets:
        raise ValueError("no consecutive triplet")
    return Candidate(record, (plane, fluid), (plane, fluid, fat, contrast), tuple(sources), triplets,
                     float(np.prod(spacing[0])))


def is_thin(value: Candidate, below_mm: float = PREFER_2D_THIN_MM) -> bool:
    """Median SliceThickness under below_mm, or the slice spacing when no thickness is recorded."""
    thickness = [float(h["thickness"]) for h in value.record["headers"] if h.get("thickness") is not None]
    measure = float(np.median(thickness)) if thickness else float(value.record["spacing_z"])
    return measure < below_mm


def select(records: list[dict], metadata: dict[str, dict], study_id: str, rule: str = "prefer-2d") -> Selection:
    """One series per slot: most usable slices, then smaller pixel area, then UID; with prefer-2d a 2D series
    is taken over a thin 3D one when the slot has both. metadata: {series UID: CSV row} of this exam."""
    if rule not in ("prefer-2d", None):
        raise ValueError("series rule must be prefer-2d or None")
    groups = {}
    for record in records:
        if record["study"] != study_id:
            raise ValueError("a record of another exam")
        try:
            value = candidate(record, metadata[record["series"]])
        except (ValueError, KeyError, TypeError):
            continue
        groups.setdefault(value.slot, []).append(value)
    chosen = []
    for slot in SLOTS:
        options = sorted(groups.get(slot, []), key=lambda v: (-len(v.sources), v.spacing_area, v.series))
        if rule is not None:
            options = [v for v in options if not is_thin(v)] or options
        chosen.append(options[0] if options else None)
    return Selection(study_id, tuple(chosen))


# ---------------------------------------------------------------- thin slabs (tile family)

def slab_starts(m: int, k: int) -> list[int]:
    """Slab starts 0, k, 2k, ... and a last slab ending on slice m - 1 (it may overlap the one before)."""
    starts = list(range(0, m - k + 1, k))
    return starts if starts[-1] == m - k else starts + [m - k]


def slabs(value: Candidate, slab_mm: float, thin_below_mm: float) -> Candidate:
    """A series thinner than thin_below_mm as triplets of three adjacent slabs of k = round(slab_mm / spacing)
    slices; unchanged when k < 2, the series is under 3 slabs, or it has a decode hole."""
    spacing = float(value.record["spacing_z"])
    if spacing >= thin_below_mm:
        return value
    k, m, sources = int(round(slab_mm / spacing)), len(value.sources), tuple(value.sources)
    if k < 2 or m < 3 * k or tuple(value.triplets) != tuple(zip(sources, sources[1:], sources[2:])):
        return value
    groups = [sources[start:start + k] for start in slab_starts(m, k)]
    return dataclasses.replace(value, triplets=tuple(zip(groups, groups[1:], groups[2:])))


def slab_selection(selection: Selection, slab_mm: float, thin_below_mm: float) -> Selection:
    return selection.with_slots(None if v is None else slabs(v, slab_mm, thin_below_mm) for v in selection.slots)


def flat(triple) -> list[int]:
    """Source indices of a tile: three slices, or three slabs of several slices each."""
    return [s for part in triple for s in (part if isinstance(part, tuple) else (part,))]


def middle(triple) -> int:
    """Source index of a tile's middle slice (slabs: the middle slice of the middle slab)."""
    mid = triple[1]
    return mid[len(mid) // 2] if isinstance(mid, tuple) else mid


# ---------------------------------------------------------------- tiles

def tile_indices(n: int, r: int) -> list[int]:
    """Tiles of a series with n triplets: starts -r, 3-r, 6-r, ... until the last slice is covered, each
    clamped into [0, n-1], duplicates kept once. r is the tile offset (0, 1 or 2)."""
    if type(n) is not int or n <= 0 or r not in (0, 1, 2):
        raise ValueError("need n > 0 triplets and r in {0, 1, 2}")
    starts, start = [], -r
    while True:
        starts.append(min(max(start, 0), n - 1))
        if start + 2 >= n + 1:
            break
        start += TILE_STRIDE
    return sorted(set(starts))


def spaced(tiles: list[int], count: int, phase: float) -> list[int]:
    """`count` of the tiles evenly spaced: tile index int((k + phase) * K / count)."""
    last = len(tiles) - 1
    return [tiles[min(int((k + phase) * len(tiles) / count), last)] for k in range(count)]


def eval_tiles(n: int, cap: int, shift: int = 0) -> list[int]:
    """Evaluation tiles at offset `shift`; above `cap`, cap of them evenly spaced at phase 0.5."""
    tiles = tile_indices(n, shift)
    return tiles if len(tiles) <= cap else spaced(tiles, cap, 0.5)


def train_tiles(n: int, count: int, seed: int, epoch: int, key: str) -> list[int]:
    """Training tiles of one series: the offset r, then (if over `count` tiles) the phase, both drawn from
    a generator keyed by sha256 of (seed, epoch, key = "study/series"), so workers do not matter."""
    digest = hashlib.sha256(json.dumps([seed, epoch, key, TILE_SAMPLER]).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    tiles = tile_indices(n, int(rng.integers(0, 3)))
    return spaced(tiles, count, float(rng.random())) if len(tiles) > count else tiles


# ---------------------------------------------------------------- knee side (show as a right knee)

def slice_centre_x(h: dict) -> float:
    """Patient x (mm, LPS) of a slice's centre."""
    iop = np.asarray(h["iop"], dtype=float)
    rows_mm, cols_mm = (float(v) for v in h["spacing"])
    rows, cols = (int(v) for v in h["shape"])
    return float(h["ipp"][0]) + iop[0] * cols_mm * cols / 2 + iop[3] * rows_mm * rows / 2


def knee_side(selection: Selection) -> dict:
    """dict(side right|left|unknown, how, centre_x_mm, tags) from the chosen series' headers.

    x = median slice-centre x over every source of every slot (LPS: a right knee lies at x < 0); beyond
    20 mm geometry decides, else a single consistent Laterality tag; untagged and within 40 mm: unknown."""
    xs, tags = [], set()
    for value in selection.slots:
        if value is None:
            continue
        headers = value.record["headers"]
        xs += [slice_centre_x(headers[s]) for s in value.sources]
        tags.add(str(headers[value.sources[0]].get("laterality") or "").strip().upper())
    tags.discard("")
    x = float(np.median(xs)) if xs else 0.0
    tag = {"R": "right", "L": "left"}.get(next(iter(tags)) if len(tags) == 1 else "")
    if abs(x) > DEAD_ZONE_MM:
        side, how = ("right" if x < 0 else "left"), "geometry"
    elif tag is not None:
        side, how = tag, "tag"
    else:
        side, how = "unknown", "none"
    if not tags and abs(round(x, 1)) < UNTAGGED_MIN_MM:
        side, how = "unknown", "none"
    return dict(side=side, how=how, centre_x_mm=round(x, 1), tags=sorted(tags))


def series_action(value: Candidate, side: str) -> str | None:
    """How to show a series as a right knee: "reverse" (sagittal: tile channel order), "w" or "h" (flip the
    image width or height), or None (already so, unknown side, or no in-plane axis along x)."""
    iop = np.asarray(value.record["headers"][value.sources[0]]["iop"], dtype=float)
    row, column = iop[:3], iop[3:]
    if value.plane == "sagittal":
        d, action = float(np.cross(row, column)[0]), "reverse"
    elif abs(row[0]) >= AXIS_COSINE:
        d, action = float(row[0]), "w"
    elif abs(column[0]) >= AXIS_COSINE:
        d, action = float(column[0]), "h"
    else:
        return None
    if side not in ("right", "left"):
        return None
    return action if (d > 0) != (side == "right") else None


def as_right_knee(selection: Selection) -> tuple[Selection, list[str | None], str]:
    """(selection with reversed sagittal tiles where needed, action per slot, side)."""
    side = knee_side(selection)["side"]
    actions = [None if v is None else series_action(v, side) for v in selection.slots]
    slots = [dataclasses.replace(v, triplets=tuple(t[::-1] for t in v.triplets)) if a == "reverse" else v
             for v, a in zip(selection.slots, actions)]
    return selection.with_slots(slots), actions, side


def flip(image: np.ndarray, action: str | None) -> np.ndarray:
    """Mirror the last axis ("w") or the one before ("h") of an image or a stack of images."""
    if action == "w":
        return image[..., ::-1]
    if action == "h":
        return image[..., ::-1, :]
    return image


# ---------------------------------------------------------------- stack positions

def slice_positions(value: Candidate, side: str) -> dict[int, float]:
    """{source: p in [0, 1]} along the canonical direction (sagittal +x for a right knee, -x left, as stored
    when unknown; coronal +y; axial -z)."""
    iop = np.asarray(value.record["headers"][value.sources[0]]["iop"], dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    t = {s: float(np.dot(np.asarray(value.record["headers"][s]["ipp"], dtype=float), normal)) for s in value.sources}
    low, high = min(t.values()), max(t.values())
    if not high - low > 1e-6:
        raise ValueError("series slices share one position")
    if value.plane == "sagittal":
        direction = SAGITTAL_DIRECTION.get(side)
    else:
        direction = PLANE_DIRECTION[value.plane]
    if direction is None:
        sign = 1.0
    else:
        dot = float(np.dot(normal, direction))
        if abs(dot) < 0.5:
            raise ValueError("series normal is not along its plane's axis")
        sign = 1.0 if dot > 0 else -1.0
    return {s: (v - low) / (high - low) if sign > 0 else (high - v) / (high - low) for s, v in t.items()}


def tile_position(table: dict[int, float], triple) -> np.float32:
    """Mean stack position of a tile's sources (in the tile's own order: the float sum depends on it)."""
    return np.float32(np.mean([table[s] for s in flat(triple)]))


def position_features(view_position):
    """(B, V) positions in [0, 1] -> (B, V, 16) fp32 torch: sin(pi k p), cos(pi k p), k = 1..8."""
    import torch
    k = torch.arange(1, POSITION_FREQUENCIES + 1, device=view_position.device, dtype=torch.float32)
    angle = math.pi * view_position.float()[..., None] * k
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

"""Knee centre and crops: the rule-based knee centre J, crop boxes, crop cuts and the window-family local views.

J (patient LPS mm) comes from the 448 px rows and the headers, the same in training and inference. Boxes are
mm squares placed around J per plane; crops are cut from the 448 rows, or once from the file's own pixels.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

import numpy as np

from knee.dicom import CANVAS_PX, physical_fit, quantize, resize
from knee.series import Candidate, Selection, knee_side, middle

JOINT_Z_SHIFT_MM = -11.0         # the field centres sit above the joint line
TISSUE_LEVEL = 0.12 * 255        # pixel level counted as tissue
TISSUE_COLUMN_SHARE = 0.15       # a column is tissue when this share of the tissue rows holds tissue
TISSUE_MIN_COLUMNS = 10
TISSUE_Z_BAND_MM = 20.0          # coronal rows within 20 mm of the joint line
SAGITTAL_AGREE_MM = 30.0         # coronal x further than this from the sagittal range middle is overruled
FALLBACK_MM = 140.0              # no centre: a square this size on the field centre
# crop "knee" boxes: (width, height) mm and the box centre relative to J (LPS: front = -y, up = +z)
SAG_BOX, SAG_UP_MM, SAG_FRONT_MM = (120.0, 120.0), 15.0, 15.0
COR_BOX, COR_UP_MM = (130.0, 110.0), -5.0
AX_BOX, AX_FRONT_MM, AX_PROBE_MM, AX_RANGE_MM = (100.0, 80.0), 35.0, 30.0, (-10.0, 72.0)
# crop "lateral" boxes: one 84 mm square on the outer compartment, sagittal and coronal only
LAT_BOX_MM, LAT_OUT_MM, LAT_UP_MM = 84.0, 30.0, -5.0
LAT_SAG_RANGE_MM, LAT_COR_RANGE_MM = (0.0, 60.0), (-30.0, 34.0)
LATERAL_X = {"right": -1.0, "left": 1.0}   # sign of patient x towards lateral
MAX_SIDE_CANVASES = 8            # a crop over 8 canvases wide means broken geometry
# window-family local views
LOCAL_MM = 100.0
CORONAL_OFFSET_MM = 30.0
PF_OFFSET_MM = (40.0, 25.0)          # (in front of, above) J
POSTERIOR_OFFSET_MM = (35.0, -5.0)   # (behind, above) J
GAP_MIN_MM = 1e-3


# ---------------------------------------------------------------- one series' slices in patient space

class _Stack:
    """A chosen series' slices (value.sources order): orientation, spacing and centres rounded as stored."""

    def __init__(self, value: Candidate):
        headers = [value.record["headers"][s] for s in value.sources]
        h0 = headers[0]
        iop = np.asarray(h0["iop"], dtype=float)
        self.rv, self.cv = iop[:3], iop[3:]          # directions of increasing column and of increasing row
        self.rsp, self.csp = (float(v) for v in h0["spacing"])
        self.rows, self.cols = h0["shape"]
        self.plane = value.plane
        centres, ipps = [], []
        for h in headers:
            ipp = np.asarray(h["ipp"], dtype=float)
            rows_mm, cols_mm = h["spacing"]
            centres.append((ipp + self.rv * cols_mm * h["shape"][1] / 2 + self.cv * rows_mm * h["shape"][0] / 2)
                           .round(2).tolist())
            ipps.append(ipp.round(3).tolist())
        self.ipp, self.centres = np.asarray(ipps, dtype=float), np.asarray(centres, dtype=float)

    def pixel(self, k: int, point) -> tuple[float, float]:
        """(col, row) of a patient point projected on slice k."""
        d = np.asarray(point, dtype=float) - self.ipp[k]
        return d @ self.rv / self.csp, d @ self.cv / self.rsp

    def centre(self) -> np.ndarray:
        return self.centres.mean(axis=0)


def _geometry(value: Candidate) -> tuple[tuple[int, int], tuple[float, float]]:
    h = value.record["headers"][0]
    return tuple(int(x) for x in h["shape"]), tuple(float(x) for x in h["spacing"])


# ---------------------------------------------------------------- knee centre

def _native_image(rows448, value: Candidate, k: int) -> np.ndarray:
    """Row k back on the file's pixel grid: the fitted part of the 448 row, PIL bilinear resize."""
    from PIL import Image
    shape, spacing = _geometry(value)
    (fh, fw), (top, left) = physical_fit(shape, spacing, (CANVAS_PX, CANVAS_PX))
    crop = np.asarray(rows448[[k]][0])[top:top + fh, left:left + fw]
    return np.asarray(Image.fromarray(crop).resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)


def _tissue_x(image: np.ndarray, s: _Stack, k: int, z_band: float | None) -> float | None:
    """Patient x of the middle of the tissue on one coronal or axial slice, or None."""
    mask = image > TISSUE_LEVEL
    if z_band is not None:
        z = s.ipp[k][2] + np.arange(s.rows) * s.rsp * s.cv[2]
        mask[np.abs(z - z_band) > TISSUE_Z_BAND_MM] = False
    cols = np.where(mask.mean(axis=0) > TISSUE_COLUMN_SHARE * mask.any(axis=1).mean())[0] if mask.any() else []
    if len(cols) < TISSUE_MIN_COLUMNS:
        return None
    mid = (cols[0] + cols[-1]) / 2
    return float((s.ipp[k] + mid * s.csp * s.rv)[0])


def _joint_centre(stacks: list[tuple[_Stack, Callable[[int], np.ndarray]]]) -> np.ndarray:
    """z and y from the sagittal/coronal field centres, x from the tissue width on a coronal (else axial)
    slice, else the sagittal slice-range middle, else the mean series centre."""
    sag = [s for s, _ in stacks if s.plane == "sagittal"]
    cor = [(s, image) for s, image in stacks if s.plane == "coronal"]
    ax = [(s, image) for s, image in stacks if s.plane == "axial"]
    zs = [s.centre()[2] for s in sag + [s for s, _ in cor]] or [s.centre()[2] for s, _ in ax]
    ys = [s.centre()[1] for s in sag] or [s.centre()[1] for s, _ in ax + cor]
    z, y = float(np.mean(zs)) + JOINT_Z_SHIFT_MM, float(np.mean(ys))
    x = None
    for s, image in cor:
        k = int(np.argmin(np.abs(s.centres[:, 1] - y)))
        x = _tissue_x(image(k), s, k, z)
        if x is not None:
            break
    if x is None:
        for s, image in ax:
            k = int(np.argmin(np.abs(s.centres[:, 2] - z)))
            x = _tissue_x(image(k), s, k, None)
            if x is not None:
                break
    if x is None and sag:
        x = float((sag[0].centres[:, 0].min() + sag[0].centres[:, 0].max()) / 2)
    if x is None:
        x = float(np.mean([s.centre()[0] for s, _ in stacks]))
    return np.array([x, y, z])


def knee_centre(selection: Selection, rows448: Callable[[Candidate], object]) -> list[float] | None:
    """J = [x, y, z] patient mm, or None (no sagittal and no coronal series, or any failure: FOV crops).

    rows448(value) -> that series' uint8 (n, 448, 448) rows (an array, or dicom.LazyRows)."""
    try:
        loaded = {}

        def image_of(value):
            def image(k):
                if value.series not in loaded:
                    loaded[value.series] = rows448(value)
                return _native_image(loaded[value.series], value, k)
            return image

        stacks = [(_Stack(v), image_of(v)) for v in selection.slots if v is not None]
        sag = [s for s, _ in stacks if s.plane == "sagittal"]
        cor = [s for s, _ in stacks if s.plane == "coronal"]
        if not sag and not cor:
            return None
        J = _joint_centre(stacks)
        if sag:
            xs = np.concatenate([s.centres[:, 0] for s in sag])
            mid = float((xs.min() + xs.max()) / 2)
            if not cor or abs(J[0] - mid) > SAGITTAL_AGREE_MM:
                J = np.array([mid, J[1], J[2]])
        return [float(v) for v in J]
    except Exception:
        return None


# ---------------------------------------------------------------- crop boxes

def _fov_window(value: Candidate, side_mm: float) -> tuple[float, float, float]:
    shape, spacing = _geometry(value)
    (fh, fw), (top, left) = physical_fit(shape, spacing, (CANVAS_PX, CANVAS_PX))
    return top + fh / 2, left + fw / 2, side_mm


def _canvas_point(value: Candidate, s: _Stack, k: int, point) -> tuple[float, float]:
    """(cy, cx) on the 448 canvas of a patient point projected on slice k."""
    shape, spacing = _geometry(value)
    (fh, fw), (top, left) = physical_fit(shape, spacing, (CANVAS_PX, CANVAS_PX))
    col, row = s.pixel(k, point)
    return top + (row + 0.5) * fh / shape[0], left + (col + 0.5) * fw / shape[1]


def _knee_boxes(selection: Selection, J) -> Selection:
    """crop "knee": one window per series around J; axial tiles limited to -10..72 mm from J."""
    slots = []
    for value in selection.slots:
        if value is None:
            slots.append(None)
            continue
        if J is None:
            slots.append(dataclasses.replace(value, window=_fov_window(value, FALLBACK_MM)))
            continue
        s, J = _Stack(value), np.asarray(J, dtype=float)
        triplets = value.triplets
        if value.plane == "sagittal":
            k = int(np.argmin(np.abs(s.centres[:, 0] - J[0])))
            point, box = J + [0, -SAG_FRONT_MM, SAG_UP_MM], SAG_BOX
        elif value.plane == "coronal":
            k = int(np.argmin(np.abs(s.centres[:, 1] - J[1])))
            point, box = J + [0, 0, COR_UP_MM], COR_BOX
        else:
            dz = s.centres[:, 2] - J[2]
            k = int(np.argmin(np.abs(dz - AX_PROBE_MM)))
            point, box = J + [0, -AX_FRONT_MM, dz[k]], AX_BOX
            z = {source: s.centres[row, 2] - J[2] for row, source in enumerate(value.sources)}
            triplets = tuple(t for t in triplets if AX_RANGE_MM[0] <= z[middle(t)] <= AX_RANGE_MM[1]) or triplets
        cy, cx = _canvas_point(value, s, k, point)
        slots.append(dataclasses.replace(value, triplets=triplets, window=(float(cy), float(cx), float(max(box)))))
    return selection.with_slots(slots)


def _lateral_boxes(selection: Selection, J) -> Selection:
    """crop "lateral": 84 mm on the outer compartment, sagittal and coronal (axial slots dropped), tiles kept
    in range (all if none is); falls back to the no-centre FOV crop of every series when there is no J, no
    sagittal or coronal series, or the side is unknown."""
    kept = selection.with_slots(None if v is None or v.plane == "axial" else v for v in selection.slots)
    side = knee_side(kept)["side"]
    if J is None or not kept.slot_mask.any() or side not in LATERAL_X:
        return _knee_boxes(selection, None)
    J, sign, slots = np.asarray(J, dtype=float), LATERAL_X[side], []
    for value in kept.slots:
        if value is None:
            slots.append(None)
            continue
        s = _Stack(value)
        if value.plane == "sagittal":
            offset, (low, high) = sign * (s.centres[:, 0] - J[0]), LAT_SAG_RANGE_MM
            k, point = int(np.argmin(np.abs(offset - LAT_OUT_MM))), J + [0, 0, LAT_UP_MM]
        else:
            offset, (low, high) = s.centres[:, 1] - J[1], LAT_COR_RANGE_MM
            k, point = int(np.argmin(np.abs(offset))), J + [sign * LAT_OUT_MM, 0, LAT_UP_MM]
        cy, cx = _canvas_point(value, s, k, point)
        where = {source: offset[row] for row, source in enumerate(value.sources)}
        inside = tuple(t for t in value.triplets if low <= where[middle(t)] <= high)
        slots.append(dataclasses.replace(value, triplets=inside or value.triplets,
                                         window=(float(cy), float(cx), LAT_BOX_MM)))
    return kept.with_slots(slots)


def crop_boxes(selection: Selection, J, kind: str = "knee") -> Selection:
    """The selection with each series' crop window (Candidate.window) and its tiles limited as the box needs.

    kind "knee" (every plane, 120/130/100 mm) or "lateral" (outer compartment, axial dropped)."""
    if kind == "knee":
        return _knee_boxes(selection, J)
    if kind == "lateral":
        return _lateral_boxes(selection, J)
    raise ValueError(f"unknown crop kind {kind!r}")


def centre_boxes(selection: Selection, side_mm: float) -> Selection:
    """Every series with a side_mm window on its field centre (the context view)."""
    return selection.with_slots(None if v is None else dataclasses.replace(v, window=_fov_window(v, side_mm))
                                for v in selection.slots)


# ---------------------------------------------------------------- cuts

def canvas_window(value: Candidate, canvas: int = CANVAS_PX) -> tuple[int, int, int]:
    """(top, left, side) in 448 canvas pixels of a series' window."""
    shape, (rsp, csp) = _geometry(value)
    cy, cx, side_mm = value.window
    mm_per_pixel = max(shape[0] * rsp, shape[1] * csp) / canvas
    side = max(1, math.floor(side_mm / mm_per_pixel + 0.5))
    if side > MAX_SIDE_CANVASES * canvas:
        raise ValueError("field of view below an eighth of the crop: geometry looks wrong")
    return math.floor(cy - side / 2 + 0.5), math.floor(cx - side / 2 + 0.5), side


def crop(stack: np.ndarray, value: Candidate, size: int) -> np.ndarray:
    """(n, 448, 448) unit float32 rows -> (n, size, size) float32: the window, zero padded past the canvas."""
    stack = np.asarray(stack, dtype=np.float32)
    n, rows, cols = stack.shape
    top, left, side = canvas_window(value, canvas=rows)
    out = np.zeros((n, side, side), dtype=np.float32)
    y0, y1, x0, x1 = max(top, 0), min(top + side, rows), max(left, 0), min(left + side, cols)
    if y1 > y0 and x1 > x0:
        out[:, y0 - top:y1 - top, x0 - left:x1 - left] = stack[:, y0:y1, x0:x1]
    return resize(out, (size, size))


def native_box(value: Candidate) -> tuple[int, int, int, int]:
    """(top, left, rows, cols) in the file's own pixels of a series' window: the 448 window carried over by
    the canvas scale, in whole pixels."""
    shape, spacing = _geometry(value)
    top, left, side = canvas_window(value)
    (fitted_rows, fitted_cols), (pad_top, pad_left) = physical_fit(shape, spacing, (CANVAS_PX, CANVAS_PX))
    per_row, per_col = shape[0] / fitted_rows, shape[1] / fitted_cols
    return (math.floor((top - pad_top) * per_row + 0.5), math.floor((left - pad_left) * per_col + 0.5),
            max(1, math.floor(side * per_row + 0.5)), max(1, math.floor(side * per_col + 0.5)))


def native_cut(unit_slice: np.ndarray, box, size: int) -> np.ndarray:
    """One unit slice's box at size x size uint8: zero padded past the picture, one antialiased resize."""
    top, left, rows, cols = box
    out = np.zeros((1, rows, cols), dtype=np.float32)
    y0, y1 = max(top, 0), min(top + rows, unit_slice.shape[0])
    x0, x1 = max(left, 0), min(left + cols, unit_slice.shape[1])
    if y1 > y0 and x1 > x0:
        out[0, y0 - top:y1 - top, x0 - left:x1 - left] = unit_slice[y0:y1, x0:x1]
    return quantize(resize(out, (size, size))[0])


# ---------------------------------------------------------------- window-family local views

@dataclasses.dataclass(frozen=True)
class Grid:
    """One series' pixel grid in patient space; slices in the order of the headers given."""
    col_dir: np.ndarray
    row_dir: np.ndarray
    normal: np.ndarray
    row_mm: float
    col_mm: float
    rows: int
    cols: int
    origins: np.ndarray
    positions: np.ndarray

    @property
    def gap_mm(self) -> float:
        steps = np.diff(np.sort(self.positions))
        steps = steps[steps > GAP_MIN_MM]
        return float(np.median(steps)) if len(steps) else 0.0


def grid(headers: list[dict]) -> Grid:
    iops = np.asarray([h["iop"] for h in headers], dtype=float)
    ipps = np.asarray([h["ipp"] for h in headers], dtype=float)
    spacings = np.asarray([h["spacing"] for h in headers], dtype=float)
    shapes = np.asarray([h["shape"] for h in headers], dtype=float)
    if (not np.allclose(iops, iops[0], atol=.001, rtol=0) or not np.allclose(spacings, spacings[0], atol=1e-4, rtol=0)
            or not (shapes == shapes[0]).all()):
        raise ValueError("slices of one series differ in orientation, spacing or shape")
    col_dir, row_dir = iops[0, :3], iops[0, 3:]
    normal = np.cross(col_dir, row_dir)
    return Grid(col_dir, row_dir, normal, float(spacings[0, 0]), float(spacings[0, 1]),
                int(shapes[0, 0]), int(shapes[0, 1]), ipps, ipps @ normal)


def _project(g: Grid, point) -> tuple[float, float, bool]:
    """(row, col) of a patient point on its nearest slice, and whether it lies inside the series."""
    point = np.asarray(point, dtype=float)
    along = float(point @ g.normal)
    index = int(np.argmin(np.abs(g.positions - along)))
    delta = point - g.origins[index]
    row, col = float(delta @ g.row_dir / g.row_mm), float(delta @ g.col_dir / g.col_mm)
    off = along - float(g.positions[index])
    reach = max(g.gap_mm, g.row_mm, g.col_mm)
    inside = -.5 <= row <= g.rows - .5 and -.5 <= col <= g.cols - .5 and abs(off) <= reach
    return row, col, bool(inside)


def _field_centre(g: Grid) -> np.ndarray:
    middle_px = g.origins[0] + g.col_dir * g.col_mm * ((g.cols - 1) / 2) + g.row_dir * g.row_mm * ((g.rows - 1) / 2)
    return middle_px + g.normal * ((g.positions.min() + g.positions.max()) / 2 - g.positions[0])


def _view(g: Grid, name: str, point, size_mm: float) -> dict | None:
    row, col, _ = _project(g, point)
    if not (-.5 <= row <= g.rows - .5 and -.5 <= col <= g.cols - .5):
        return None
    rows, cols = max(1, round(size_mm / g.row_mm)), max(1, round(size_mm / g.col_mm))
    box = (math.floor(row - (rows - 1) / 2 + 0.5), math.floor(col - (cols - 1) / 2 + 0.5), rows, cols)
    return dict(name=name, box=box, size_mm=float(size_mm))


def local_views(headers: list[dict], plane: str, J, side: str) -> list[dict]:
    """The 100 mm local views of one series: [dict(name, box (top, left, rows, cols) file px, size_mm)].

    Sagittal and axial: joint (J), pf, posterior; coronal: medial and lateral, 30 mm either side of J (an
    unknown side is cut as a right knee). A view whose centre falls outside the picture is left out. With
    no J, J outside the series, or no view left: one "wide" 100 mm view on the field centre."""
    g = grid(headers)
    views = []
    if J is not None and _project(g, J)[2]:
        joint = np.asarray([float(v) for v in J], dtype=float)
        if plane == "coronal":
            medial = np.array([-1.0 if side == "left" else 1.0, 0.0, 0.0]) * CORONAL_OFFSET_MM
            centres = dict(medial=joint + medial, lateral=joint - medial)
        else:
            centres = dict(joint=joint, pf=joint + [0.0, -PF_OFFSET_MM[0], PF_OFFSET_MM[1]],
                           posterior=joint + [0.0, POSTERIOR_OFFSET_MM[0], POSTERIOR_OFFSET_MM[1]])
        views = [v for v in (_view(g, name, point, LOCAL_MM) for name, point in centres.items()) if v]
    return views or [_view(g, "wide", _field_centre(g), LOCAL_MM)]


def local_cut(unit_slice: np.ndarray, box, size: int) -> np.ndarray:
    """A local view of one unit slice: float32 (size, size), the box zero padded past the picture."""
    unit_slice = np.asarray(unit_slice, dtype=np.float32)
    top, left, rows, cols = box
    out = np.zeros((rows, cols), dtype=np.float32)
    y0, y1 = max(top, 0), min(top + rows, unit_slice.shape[0])
    x0, x1 = max(left, 0), min(left + cols, unit_slice.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = unit_slice[y0:y1, x0:x1]
    return resize(out[None], (size, size))[0]

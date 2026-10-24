"""radiant-kg V3: patient-space geometry of the reader's canonical images, from DICOM headers only.

load_series (preprocess.py) reorients each slice to a canonical in-plane orientation, centre-crops it to
SIZE * MM_PER_PX millimetres, resamples to MM_PER_PX and zero-pads to SIZE x SIZE. This module replays exactly the
same steps on the header fields (no pixels) and returns, for every slice that load_series keeps (same sort, same
subsampling), an affine in patient LPS millimetres: world(r, c) = O + r * DR + c * DC for canonical pixel (r, c).
That lets a point found in one series (e.g. the knee joint centre) be placed in any other series of the study.
"""

import numpy as np

from preprocess import CANON, MAX_SLICES, MM_PER_PX, PLANES, SIZE


def _slice_affine(rows, cols, iop, ipp, ps, plane):
    """Affine (O, DR, DC) of one canonical slice. Mirrors preprocess._canon_slice and preprocess._resample."""
    ps = list(ps) if ps else [MM_PER_PX * rows / SIZE] * 2
    rcos, ccos = np.array(iop[:3], float), np.array(iop[3:], float)
    O = np.array(ipp, float)
    DR, DC = ps[0] * ccos, ps[1] * rcos  # +1 row moves along the column cosine, +1 column along the row cosine
    h, w = rows, cols
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    r, c = rcos, ccos
    if abs(r[cax]) < abs(r[rax]):  # transpose
        DR, DC = DC, DR
        h, w = w, h
        r, c, ps = c, r, ps[::-1]
    if r[cax] * csgn < 0:  # flip columns
        O, DC = O + (w - 1) * DC, -DC
    if c[rax] * rsgn < 0:  # flip rows
        O, DR = O + (h - 1) * DR, -DR
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
    cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    O = O + y0 * DR + x0 * DC
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
    nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    sy, sx = ch / nh, cw / nw  # cv2.resize: new pixel centre i maps to old (i + 0.5) * s - 0.5
    O = O + (0.5 * sy - 0.5) * DR + (0.5 * sx - 0.5) * DC
    DR, DC = DR * sy, DC * sx
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    O = O - oy * DR - ox * DC
    return O, DR, DC


def series_geometry(headers, plane_name=None):
    """headers: pydicom datasets of every file of one series (header-only is enough).
    Returns (plane, [(O, DR, DC) per kept slice in load_series order]) or (plane, []) when geometry is missing."""
    sl = []
    for i, ds in enumerate(headers):
        try:
            n_frames = int(ds.get("NumberOfFrames", 1) or 1)
            iop = [float(v) for v in ds.ImageOrientationPatient]
            ipp = [float(v) for v in ds.ImagePositionPatient]
            ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
            inst = float(ds.get("InstanceNumber") or 0)
            for f in range(n_frames):
                sl.append((iop, ipp, ps, inst + f * 1e-3, int(ds.Rows), int(ds.Columns)))
        except Exception:
            continue
    if not sl:
        return None, []
    iop = sl[0][0]
    normal = np.cross(iop[:3], iop[3:])
    plane = int(np.abs(normal).argmax())
    sgn = CANON[plane][0] * np.sign(normal[plane])
    sl.sort(key=lambda s: float(np.dot(s[1], normal)) * sgn)
    if len(sl) > MAX_SLICES:
        keep = np.linspace(0, len(sl) - 1, MAX_SLICES).round().astype(int)
        sl = [sl[i] for i in keep]
    return PLANES[plane], [_slice_affine(s[4], s[5], s[0], s[1], s[2], plane) for s in sl]


def to_pixel(aff, X):
    """World point X -> (row, col, out-of-plane mm) on the slice with affine (O, DR, DC)."""
    O, DR, DC = aff
    M = np.stack([DR, DC], 1)
    rc, *_ = np.linalg.lstsq(M, np.asarray(X, float) - O, rcond=None)
    n = np.cross(DR, DC)
    n = n / (np.linalg.norm(n) + 1e-9)
    return float(rc[0]), float(rc[1]), float(abs(np.dot(np.asarray(X, float) - O, n)))


def centre(aff):
    O, DR, DC = aff
    return O + (SIZE - 1) / 2 * DR + (SIZE - 1) / 2 * DC

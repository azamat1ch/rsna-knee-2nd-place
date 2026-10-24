"""radiant-kg V3: knee-centred close-up views (after the 0.954 notebook's view set v4), in the reader's canonical cache.

Joint centre = the header-only "slab prior" of heliosli's knee_anchor_bank_v2: left-right from the sagittal stack,
front-back from the coronal stack, up-down from the sagittal and coronal field centres. Close-ups are 100 mm squares
placed in patient millimetres around it (LPS: +x patient left, +y posterior, +z superior), per plane:
  sagittal: joint; kneecap (40 mm anterior, 25 mm superior); behind the knee (35 mm posterior, 5 mm inferior)
  axial:    joint; kneecap (40 mm anterior); behind the knee (35 mm posterior)
  coronal:  lateral and medial compartments (30 mm either side; lateral = patient right for a right knee)
The view id says what the square shows, so after left knees are mirrored a "lateral" crop is still lateral.
"""

import numpy as np

import geo
from preprocess import MM_PER_PX, SIZE

LOCAL_MM = 100.0
CROP = int(round(LOCAL_MM / MM_PER_PX))  # 250 px in the 0.4 mm cache
VIEWS = ["full", "joint", "kneecap", "posterior", "lateral", "medial"]


def slab_prior(geoms):
    """geoms: {series: (plane, [affines])}. World joint-centre estimate (header-only), or None."""
    cen = {p: [] for p in ("Sagittal", "Coronal", "Axial")}
    for plane, affs in geoms.values():
        if affs:
            cen[plane].append(np.mean([geo.centre(a) for a in affs], 0))  # stack middle
    sag, cor, ax = (np.mean(cen[p], 0) if cen[p] else None for p in ("Sagittal", "Coronal", "Axial"))
    if sag is None and cor is None:
        return ax
    if sag is None:
        return cor
    if cor is None:
        return sag
    return np.array([sag[0], cor[1], (sag[2] + cor[2]) / 2])


def series_dx(affs, X):
    """radiant-kg V8 arm P: per-slice x displacement (mm, LPS) of each slice centre from the anchor X, in the order of
    affs. The reader's dataset (knee.phys_values) turns it into d = lateral-positive after left/right normalisation."""
    return np.array([geo.centre(a)[0] for a in affs], np.float64) - float(X[0])


def view_centres(plane, X, side):
    """World centres of the close-ups for one plane. side: 'R' or 'L' (the knee scanned)."""
    ant, sup = np.array([0, -1.0, 0]), np.array([0, 0, 1.0])
    lat = np.array([-1.0 if side == "R" else 1.0, 0, 0])  # LPS +x = patient left
    if plane == "Sagittal":
        return {"joint": X, "kneecap": X + 40 * ant + 25 * sup, "posterior": X - 35 * ant - 5 * sup}
    if plane == "Axial":
        return {"joint": X, "kneecap": X + 40 * ant, "posterior": X - 35 * ant}
    return {"lateral": X + 30 * lat, "medial": X - 30 * lat}


def crop_boxes(plane, affs, X, side):
    """{view: (row0, col0)} top-left corners of CROP x CROP squares in the canonical image (as stored, unmirrored),
    from the middle slice's geometry, kept inside the image."""
    mid = affs[len(affs) // 2]
    out = {}
    for v, W in view_centres(plane, X, side).items():
        r, c, _ = geo.to_pixel(mid, W)
        out[v] = (int(np.clip(round(r - CROP / 2), 0, SIZE - CROP)), int(np.clip(round(c - CROP / 2), 0, SIZE - CROP)))
    return out


def mirror_box(plane, box):
    """Top-left corner after the reader's left-knee mirroring (coronal/axial columns flip; sagittal is unchanged
    in-plane because only its slice order is reversed)."""
    r0, c0 = box
    return (r0, c0) if plane == "Sagittal" else (r0, SIZE - CROP - c0)

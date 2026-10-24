"""Image panels the image expert sees, drawn from the six chosen series at native DICOM resolution.

Two packs per training exam: `expert` (overview + joint, ligament, patellofemoral, fluid detail; 12 findings) and
`blind-lateral` (short overview + enlarged lateral half). A pack is the request content minus report and reader
text. Runs as a Kaggle CPU kernel; Pillow is pinned because resize, default font and PNG bytes depend on it.
"""
from __future__ import annotations

import argparse
import base64
import csv
import functools
import io
import json
import math
import multiprocessing
import zipfile
from pathlib import Path

import numpy as np

from knee import LABELS

PILLOW = "12.3.0"
SLOTS = ["sagittal-0", "sagittal-1", "coronal-1", "coronal-0", "axial-1", "axial-0"]  # overview order
SERIES_NAMES = {"sagittal-0": "SAG nonFS", "sagittal-1": "SAG FS", "coronal-1": "COR FS", "coronal-0": "COR nonFS",
                "axial-1": "AX FS", "axial-0": "AX nonFS"}
CONTRAST = {"PD": "PD", "T1": "T1", "T2": "T2", "gradient_echo": "GRE"}  # label names; "unknown" prints nothing
REGIONS = ["fluid_and_synovium", "menisci", "cruciates_and_collaterals", "bone", "cartilage"]
BLIND_FINDINGS = ["Lateral Meniscus", "Lateral OA"]
BLIND_ROLE = "the lateral-compartment expert (lateral meniscus and lateral tibiofemoral cartilage)"
PNG_PREFIX = "data:image/png;base64,"

# Geometry
SIDE_DEAD_ZONE_MM = 20.0  # render: median slice-centre x beyond this decides the side, else the laterality tag
SIDE_MIN_MM = 40.0  # packs: without a laterality tag, geometry decides only beyond this
JOINT_Z_SHIFT_MM = -11.0  # joint line sits below the field-of-view centre
TISSUE_LEVEL, TISSUE_COLUMN_SHARE, TISSUE_BAND_MM = 0.12, 0.15, 20  # soft-tissue width on one slice
RUN_SPACING_MM = 2.9  # consecutive slices: about every 3 mm
PATCH_PX = 28  # one image token per 28 x 28 px patch (estimate)

# Expert pack
OVERVIEW_PX, OVERVIEW_MAX_SLICES = 240, 40
EXPERT_TOKEN_CAP, CAP_MARGIN = 80_000, 4_000  # overview series dropped from the end while over cap - margin
DETAIL_SECTIONS = ["joint", "ligament", "patellofemoral", "fluid"]

# Blind-lateral pack (boxes are (width, height) in mm along columns and rows; sizes are the longest side in px)
BL_OVERVIEW_PX, BL_OVERVIEW_PER_PLANE = 240, 7  # one series per plane, evenly spaced over the middle 80 %
SAG_RANGE_MM, SAG_MAX, SAG_BOX_MM, SAG_UP_MM = (-4, 40), 15, (80, 64), 4.0
COR_RANGE_MM, COR_MAX, COR_BOX_MM, COR_SHIFT_MM = (-28, 26), 16, (76, 50), 18
HALF_PX = {"lateral": 768, "medial": 640}  # side unknown: both halves 768
BLIND_TOKEN_CAP = 60_000


# ---------------------------------------------------------------- image sources

class Renders:
    """Native renders: {study: manifest entry} and the PNG bytes of each slice by member name."""

    def __init__(self, studies: dict, read):
        self.studies, self.read = studies, read

    @classmethod
    def folder(cls, root: Path) -> "Renders":
        """A render folder: manifest.json ({"studies": {id: entry}}) + images.zip (one PNG per slice)."""
        archive = zipfile.ZipFile(Path(root) / "images.zip")
        return cls(json.loads((Path(root) / "manifest.json").read_text())["studies"], archive.read)

    def study(self, study_id: str) -> dict | None:
        return self.studies.get(study_id)

    def image(self, member: str):
        from PIL import Image
        return Image.open(io.BytesIO(self.read(member))).convert("L")


def native_study(study, dead_zone_mm: float = SIDE_DEAD_ZONE_MM) -> tuple[dict, list[tuple[str, bytes]]]:
    """A knee.dicom.Study -> (manifest entry, [(member, PNG bytes)]): every slice of each chosen series at native
    resolution, u8 = rint(clip((pixels - lo) / (hi - lo), 0, 1) * 255) in the series window."""
    import pydicom
    from PIL import Image
    from knee.dicom import physical_pixels
    centres_x, entries, tags, files = [], {}, set(), []
    for cand in study.selection.slots:
        if cand is None:
            continue
        r, series = cand.record, study.series[cand.series]
        plane, fluid = cand.slot
        lo, hi = float(series.lo), float(series.hi)  # Python floats keep float32 pixels float32, as the packs ran
        h0 = r["headers"][cand.sources[0]]
        description = str(getattr(pydicom.dcmread(series.paths[cand.sources[0]], stop_before_pixels=True),
                                  "SeriesDescription", ""))
        iop = np.asarray(h0["iop"], dtype=float)
        rv, cv = iop[:3], iop[3:]
        rows_mm, cols_mm = h0["spacing"]  # DICOM PixelSpacing: row spacing, column spacing
        centres, ipps, members = [], [], []
        for k, src in enumerate(cand.sources):
            h = r["headers"][src]
            pix = physical_pixels(series.native_pixels(src), h["slope"], h["intercept"], h["photometric"])
            u8 = np.rint(np.clip((pix - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            ipp = np.asarray(h["ipp"], dtype=float)
            centres.append((ipp + rv * cols_mm * h["shape"][1] / 2 + cv * rows_mm * h["shape"][0] / 2).round(2).tolist())
            ipps.append(ipp.round(3).tolist())
            name = f"{study.study_id}/{plane}-{fluid}/{k:03d}.png"
            buffer = io.BytesIO()
            Image.fromarray(u8).save(buffer, "PNG", optimize=True)
            files.append((name, buffer.getvalue()))
            members.append(name)
        centres_x += [c[0] for c in centres]
        tags.add(h0.get("laterality", ""))
        entries[f"{plane}-{fluid}"] = dict(
            series=cand.series, plane=plane, fluid=fluid, fat=cand.signature[2], contrast=cand.signature[3],
            tr=h0.get("tr"), te=h0.get("te"), thickness=h0.get("thickness"), description=description,
            spacing_z=r["spacing_z"], pixel_spacing=h0["spacing"], shape=h0["shape"], window=[lo, hi],
            iop=h0["iop"], ipp=ipps, slice_centres=centres, members=members)
    mx = float(np.median(centres_x)) if centres_x else 0.0
    tag = {t.strip().upper() for t in tags} - {""}
    side = "right" if mx < -dead_zone_mm else "left" if mx > dead_zone_mm else \
        {"R": "right", "L": "left"}.get(next(iter(tag)) if len(tag) == 1 else "", "unknown")
    return dict(side=side, centre_x_mm=round(mx, 1), laterality_tag=sorted(tag), native=True, slots=entries), files


# ---------------------------------------------------------------- geometry

class Series:
    """One slotted series: native pixel grid, patient geometry (LPS mm), display name."""

    def __init__(self, slot: str, entry: dict):
        self.slot, self.e = slot, entry
        iop = np.asarray(entry["iop"], dtype=float)
        self.rv, self.cv = iop[:3], iop[3:]  # direction of increasing column, of increasing row
        self.rsp, self.csp = (float(v) for v in entry["pixel_spacing"])  # row spacing, column spacing (mm)
        self.rows, self.cols = entry["shape"]
        self.ipp = np.asarray(entry["ipp"], dtype=float)
        self.centres = np.asarray(entry["slice_centres"], dtype=float)
        self.plane = entry["plane"]
        self.n = len(entry["members"])
        self.name = f"{SERIES_NAMES[slot]} {CONTRAST.get(entry['contrast'], '')}".strip()

    def pixel(self, k: int, point) -> tuple[float, float]:
        """(column, row) of a patient point projected on slice k."""
        d = np.asarray(point, dtype=float) - self.ipp[k]
        return d @ self.rv / self.csp, d @ self.cv / self.rsp

    def centre(self) -> np.ndarray:
        return self.centres.mean(axis=0)


def knee_side(entry: dict) -> str:
    """Render side, but 'unknown' without a laterality tag when |median slice-centre x| < SIDE_MIN_MM."""
    if not entry.get("laterality_tag") and abs(entry["centre_x_mm"]) < SIDE_MIN_MM:
        return "unknown"
    return entry["side"]


def tissue_centre_x(renders, s: Series, k: int, z_band: float | None) -> float | None:
    """Patient x of the middle of the soft-tissue extent on one slice (coronal: rows within 20 mm of z_band)."""
    img = np.asarray(renders.image(s.e["members"][k]), dtype=np.float32)
    mask = img > TISSUE_LEVEL * 255
    if z_band is not None:
        z = s.ipp[k][2] + np.arange(s.rows) * s.rsp * s.cv[2]
        mask[np.abs(z - z_band) > TISSUE_BAND_MM] = False
    cols = np.where(mask.mean(axis=0) > TISSUE_COLUMN_SHARE * mask.any(axis=1).mean())[0] if mask.any() else []
    if len(cols) < 10:
        return None
    mid = (cols[0] + cols[-1]) / 2
    return float((s.ipp[k] + mid * s.csp * s.rv)[0])


def joint_centre(renders, series: dict, side: str) -> tuple[np.ndarray, str]:
    """Knee centre J (LPS mm) and how x was found. z, y: mean field-of-view centre of the sagittal and coronal
    series (z shifted down). x (notch): soft-tissue middle on the coronal slice nearest y, else the axial slice
    nearest z, else the middle of the sagittal slice range, else the mean series centre."""
    sag = [s for s in series.values() if s.plane == "sagittal"]
    cor = [s for s in series.values() if s.plane == "coronal"]
    ax = [s for s in series.values() if s.plane == "axial"]
    zs = [s.centre()[2] for s in sag + cor] or [s.centre()[2] for s in ax]
    ys = [s.centre()[1] for s in sag] or [s.centre()[1] for s in ax + cor]
    z, y = float(np.mean(zs)) + JOINT_Z_SHIFT_MM, float(np.mean(ys))
    x, how = None, None
    for s in cor:
        k = int(np.argmin(np.abs(s.centres[:, 1] - y)))
        x, how = tissue_centre_x(renders, s, k, z), f"tissue width, {s.name}"
        if x is not None:
            break
    if x is None:
        for s in ax:
            k = int(np.argmin(np.abs(s.centres[:, 2] - z)))
            x, how = tissue_centre_x(renders, s, k, None), f"tissue width, {s.name}"
            if x is not None:
                break
    if x is None and sag:
        x, how = float((sag[0].centres[:, 0].min() + sag[0].centres[:, 0].max()) / 2), "sagittal range middle"
    if x is None:
        x, how = float(np.mean([s.centre()[0] for s in series.values()])), "series centre"
    return np.array([x, y, z]), how


def lateral_mm(x: float, J: np.ndarray, side: str) -> float:
    """Signed distance from the notch: + = lateral (side known) or patient right (side unknown)."""
    return (x - J[0]) if side == "left" else (J[0] - x)


def position_text(s: Series, k: int, J: np.ndarray, side: str) -> str:
    c = s.centres[k]
    if s.plane == "sagittal":
        d = lateral_mm(c[0], J, side)
        if abs(d) < 2.5:
            return "notch"
        if side in ("left", "right"):
            return f"{abs(d):.0f}mm {'LAT' if d > 0 else 'MED'}"
        return f"{abs(d):.0f}mm pt-{'R' if d > 0 else 'L'}"
    if s.plane == "coronal":
        d = c[1] - J[1]
        return "plateau ctr" if abs(d) < 2.5 else f"{abs(d):.0f}mm {'post' if d > 0 else 'ant'}"
    d = c[2] - J[2]
    return "joint line" if abs(d) < 2.5 else f"{abs(d):.0f}mm {'above' if d > 0 else 'below'} JL"


def anatomical_order(s: Series, J: np.ndarray, side: str) -> list[int]:
    """Sagittal lateral -> medial (side unknown: patient right -> left); coronal anterior -> posterior;
    axial superior -> inferior."""
    if s.plane == "sagittal":
        key = [-lateral_mm(x, J, side) for x in s.centres[:, 0]]
    elif s.plane == "coronal":
        key = list(s.centres[:, 1])
    else:
        key = list(-s.centres[:, 2])
    return [int(i) for i in np.argsort(key, kind="stable")]


def first(series: dict, *slots: str) -> Series | None:
    return next((series[s] for s in slots if s in series), None)


def pick(indices: list, n: int) -> list:
    """At most n evenly spaced items, order kept."""
    if len(indices) <= n:
        return list(indices)
    return [indices[int(round(i))] for i in np.linspace(0, len(indices) - 1, n)]


def run_slices(s: Series, ks: list[int], n: int, centre=None) -> list[int]:
    """Consecutive slices about 3 mm apart (every slice of a 3-5 mm series, every k-th of a thin 3D series), the
    central n if more. centre = (values per slice, value to centre on); default the middle of ks."""
    step = max(1, math.ceil(RUN_SPACING_MM / max(float(s.e["spacing_z"]), 0.1)))
    if not ks:
        return []
    mid = len(ks) // 2 if centre is None else int(np.argmin([abs(v - centre[1]) for v in centre[0]]))
    sub = ks[mid % step::step]
    if len(sub) <= n:
        return sub
    j = min(range(len(sub)), key=lambda t: abs(mid % step + t * step - mid))
    lo = min(max(j - n // 2, 0), len(sub) - n)
    return sub[lo:lo + n]


def tokens(w: float, h: float) -> int:
    return math.ceil(w / PATCH_PX) * math.ceil(h / PATCH_PX)


# ---------------------------------------------------------------- drawing

def edge_text(entry: dict, side: str) -> tuple[str, str]:
    """Patient direction at the right and bottom image edges ('pt-L (MED)', ANT, POST, SUP, INF)."""
    iop = np.asarray(entry["iop"], dtype=float)

    def name(v):
        axis = int(np.argmax(np.abs(v)))
        pos = v[axis] > 0  # LPS: +x patient left, +y posterior, +z superior
        if axis == 0:
            text = "pt-L" if pos else "pt-R"
            if side in ("left", "right"):
                text += " (LAT)" if (pos == (side == "left")) else " (MED)"
            return text
        return ("POST" if pos else "ANT") if axis == 1 else ("SUP" if pos else "INF")
    return name(iop[:3]), name(iop[3:])


@functools.lru_cache(maxsize=None)
def font(px: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=px)


def draw_image(renders, s: Series, k: int, label: str, side: str, size: int, box_mm=None, centre=None):
    """Native slice -> optional crop of box_mm around a patient point, physical aspect with the longest side
    `size` px (LANCZOS), black top band with the label, edge letters right and bottom."""
    from PIL import Image, ImageDraw
    img = renders.image(s.e["members"][k])
    if box_mm:
        col, row = s.pixel(k, centre)
        w_px, h_px = box_mm[0] / s.csp, box_mm[1] / s.rsp
        x0 = min(max(col - w_px / 2, 0), max(s.cols - w_px, 0))
        y0 = min(max(row - h_px / 2, 0), max(s.rows - h_px, 0))
        img = img.crop((int(round(x0)), int(round(y0)), int(round(min(x0 + w_px, s.cols))),
                        int(round(min(y0 + h_px, s.rows)))))
    w_mm, h_mm = img.size[0] * s.csp, img.size[1] * s.rsp
    scale = size / max(w_mm, h_mm)
    img = img.resize((max(1, round(w_mm * scale)), max(1, round(h_mm * scale))), Image.LANCZOS)
    fpx = 11 if size <= 260 else 13 if size <= 400 else 15
    band = fpx + 6
    canvas = Image.new("L", (img.size[0], img.size[1] + band), 0)
    canvas.paste(img, (0, band))
    draw = ImageDraw.Draw(canvas)
    lpx = fpx
    while lpx > 8 and draw.textlength(label, font=font(lpx)) > canvas.size[0] - 6:
        lpx -= 1  # long label on a narrow image: shrink rather than clip
    draw.text((3, 2), label, fill=255, font=font(lpx))
    right, bottom = edge_text(s.e, side)
    f = font(fpx)
    draw.text((canvas.size[0] - draw.textlength(right, font=f) - 3, band + img.size[1] // 2), right, fill=255, font=f)
    draw.text((canvas.size[0] // 2 - draw.textlength(bottom, font=f) // 2, canvas.size[1] - fpx - 3), bottom,
              fill=255, font=f)
    return canvas


def png_url(img) -> str:
    buffer = io.BytesIO()
    img.save(buffer, "PNG", optimize=True)
    return PNG_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_label(s: Series, k: int, J: np.ndarray, side: str, zoom: bool) -> str:
    rank = anatomical_order(s, J, side).index(k) + 1
    return f"{s.name} | {rank}/{s.n} | {position_text(s, k, J, side)}" + (" | zoom" if zoom else "")


def series_list(series: dict, side: str) -> str:
    order = {"sagittal": "lateral to medial" if side in ("left", "right") else
             "patient right to patient left (knee side unknown: find the fibular head, it is lateral)",
             "coronal": "anterior to posterior", "axial": "superior to inferior"}
    lines = [f"Knee side from DICOM geometry: {side}." + ("" if side in ("left", "right") else
             " Unknown: MED/LAT cannot be printed; sagittal positions say pt-R / pt-L (patient right / left)."),
             "Positions in labels are millimetres from an estimated knee centre (sagittal: the notch; coronal: the "
             "tibial plateau centre; axial: the joint line, JL), found from the field of view and soft-tissue width; "
             "they can be off by about 1 cm. Slice rank/total is exact.",
             "Series acquired in this exam (a call may show only some of them; the section list says which):"]
    for slot in SLOTS:
        if slot not in series:
            continue
        s = series[slot]
        e = s.e
        lines.append(f"- {s.name}: plane {e['plane']}, {'fat-suppressed fluid-sensitive' if e['fluid'] else 'non-fat-suppressed'}"
                     f"{' (fat-suppressed flag)' if e['fat'] and not e['fluid'] else ''}, contrast class "
                     f"{e['contrast'] if e['contrast'] != 'unknown' else 'not determined (see description)'} "
                     f"(TR {e['tr']}, TE {e['te']}), description '{e.get('description', '')}', {s.n} slices, slice "
                     f"spacing {e['spacing_z']:.1f} mm (thickness {e['thickness']}), pixel {s.rsp:.2f} x {s.csp:.2f} mm, "
                     f"{s.rows} x {s.cols} px, order {order[e['plane']]}")
    missing = [SERIES_NAMES[s] for s in SLOTS if s not in series]
    if missing:
        lines.append("Not available in this exam: " + ", ".join(missing))
    return "\n".join(lines)


def _setup(renders, study_id: str):
    entry = renders.study(study_id)
    if entry is None or not entry.get("native"):
        raise KeyError("study not rendered natively")
    side = knee_side(entry)
    series = {slot: Series(slot, e) for slot, e in entry["slots"].items()}
    J, how = joint_centre(renders, series, side)
    return side, series, J, how


# ---------------------------------------------------------------- expert pack

SECTION_TEXT = {
    "overview": "Overview: every slice of every series in this call (a series with more than 40 slices: 40 evenly "
                "spaced) at {px} px, series by series, in anatomical order",
    "joint": "Joint detail: in each sagittal series, consecutive slices {compartments}; then "
             "consecutive coronal slices from the anterior horns to the posterior roots (512 "
             "px), then about the same coronal range in the other coronal series (384 px); all cropped to the "
             "tibiofemoral joint (condyles, menisci, plateaus). Consecutive = every acquired slice, or about every "
             "3 mm in a thin 3D series",
    "ligament": "Ligament detail: consecutive central sagittal slices through the notch (ACL; both sagittal "
                "series), consecutive coronal slices of the medial side from the MCL origin toward its tibial "
                "insertion (as far as the field of view reaches), and consecutive axial slices from about 1 cm "
                "below to 2 cm above the estimated joint line, cropped and enlarged (448 px)",
    "patellofemoral": "Patellofemoral detail: consecutive axial slices through the patella and trochlea, cropped "
                      "and enlarged (384 px)",
    "fluid": "Fluid detail: whole sagittal slices, evenly sampled (not consecutive), {fluid_range} "
             "(suprapatellar recess, Hoffa fat pad, popliteal fossa, posteromedial corner; 448 px), and whole "
             "axial slices, evenly sampled, from the top of the series to below the joint line (384 px)",
}


def section_text(sec: str, side: str) -> str:
    known = side in ("left", "right")
    compartments = ("from the lateral edge of the knee to the notch, then from the notch to the medial edge (lateral "
                    "to medial; slices within 2 mm of the notch are left out)" if known else
                    "from the patient-right edge of the knee to the notch, then from the notch to the patient-left "
                    "edge (patient right to left; slices within 2 mm of the notch are left out)")
    fluid_range = ("from about 15 mm lateral to 35 mm medial of the notch, shown lateral to medial" if known else
                   "from about 30 mm patient-right to 30 mm patient-left of the notch, shown patient right to left")
    return SECTION_TEXT[sec].format(compartments=compartments, fluid_range=fluid_range)


def detail_specs(series: dict, J: np.ndarray, side: str) -> dict:
    """Detail sections: {section: [(series, slice, box_mm or None = whole slice, centre, size px)]}."""
    out = {sec: [] for sec in DETAIL_SECTIONS}
    known = side in ("left", "right")
    lat = lambda s, k: lateral_mm(s.centres[k, 0], J, side)  # noqa: E731
    # Joint: each sagittal series 2-38 mm lateral, then medial, of the notch, <= 12 consecutive per side; coronal
    # (FS first) 28 mm anterior to 24 mm posterior of the plateau centre, <= 16; all 110 x 82.5 mm at J, 512 px;
    # the other coronal series over the same range, <= 12, 384 px.
    for slot in ("sagittal-0", "sagittal-1"):
        s = series.get(slot)
        if s is None:
            continue
        order = anatomical_order(s, J, side)
        for sign in (1, -1):
            ks = [k for k in order if 2 <= sign * lat(s, k) <= 38]
            out["joint"] += [(s, k, (110, 82.5), J, 512) for k in run_slices(s, ks, 12)]
    s = first(series, "coronal-1", "coronal-0")
    if s:
        ks = [k for k in anatomical_order(s, J, side) if -28 <= s.centres[k, 1] - J[1] <= 24]
        out["joint"] += [(s, k, (110, 82.5), J, 512) for k in run_slices(s, ks, 16)]
        other = series.get("coronal-0" if s.slot == "coronal-1" else "coronal-1")
        if other is not None:
            ks = [k for k in anatomical_order(other, J, side) if -28 <= other.centres[k, 1] - J[1] <= 24]
            out["joint"] += [(other, k, (110, 82.5), J, 384) for k in run_slices(other, ks, 12)]
    # Ligament: both sagittal series 8 mm medial to 16 mm lateral of the notch (side unknown +-12), <= 7, 100 x
    # 100 mm centred 8 mm above J; coronal (FS first) within 15 mm of the plateau centre, <= 6, 90 x 170 mm 25 mm
    # medial and 15 mm below J (side unknown 130 x 170 on the notch); axial (FS first) 10 mm below to 22 mm above
    # the joint line, <= 10, 110 x 110 mm; 448 px.
    for slot in ("sagittal-0", "sagittal-1"):
        s = series.get(slot)
        if s is None:
            continue
        lo, hi = (-8, 16) if known else (-12, 12)
        ks = [k for k in anatomical_order(s, J, side) if lo <= lat(s, k) <= hi]
        out["ligament"] += [(s, k, (100, 100), J + np.array([0, 0, 8.0]), 448)
                            for k in run_slices(s, ks, 7, ([lat(s, k) for k in ks], (lo + hi) / 2))]
    s = first(series, "coronal-1", "coronal-0")
    if s:
        ks = [k for k in anatomical_order(s, J, side) if abs(s.centres[k, 1] - J[1]) <= 15]
        if known:
            medial = J + np.array([25.0 if side == "right" else -25.0, 0, -15.0])  # right knee: medial = +x
            out["ligament"] += [(s, k, (90, 170), medial, 448) for k in run_slices(s, ks, 6)]
        else:
            out["ligament"] += [(s, k, (130, 170), J + np.array([0, 0, -15.0]), 448) for k in run_slices(s, ks, 6)]
    s = first(series, "axial-1", "axial-0")
    if s:
        ks = [k for k in anatomical_order(s, J, side) if -10 <= s.centres[k, 2] - J[2] <= 22]
        out["ligament"] += [(s, k, (110, 110), J, 448) for k in run_slices(s, ks, 10)]
    # Patellofemoral: axial (FS first) 8 to 64 mm above the joint line, <= 14, 120 x 90 mm centred 52 mm
    # anterior of J, 384 px.
    s = first(series, "axial-1", "axial-0")
    if s:
        ks = [k for k in anatomical_order(s, J, side) if 8 <= s.centres[k, 2] - J[2] <= 64]
        out["patellofemoral"] += [(s, k, (120, 90), J + np.array([0, -52.0, 0]), 384) for k in run_slices(s, ks, 14)]
    # Fluid, whole slices: sagittal FS (else nonFS) 35 mm medial to 15 mm lateral of the notch (side unknown +-30),
    # <= 10 evenly spaced, 448 px; axial FS (else nonFS) from the top down to 25 mm below the joint line, <= 10, 384 px.
    s = first(series, "sagittal-1", "sagittal-0")
    if s:
        lo, hi = (-35, 15) if known else (-30, 30)
        ks = [k for k in anatomical_order(s, J, side) if lo <= lat(s, k) <= hi]
        out["fluid"] += [(s, k, None, None, 448) for k in pick(ks, 10)]
    s = first(series, "axial-1", "axial-0")
    if s:
        ks = [k for k in anatomical_order(s, J, side) if s.centres[k, 2] - J[2] >= -25]
        out["fluid"] += [(s, k, None, None, 384) for k in pick(ks, 10)]
    return out


def expert_pack(renders, study_id: str) -> dict:
    """{sections, content, info} of the expert pack (all 12 findings)."""
    side, series, J, how = _setup(renders, study_id)
    details = detail_specs(series, J, side)
    overview = [s for s in SLOTS if s in series]

    def est_overview(slots):
        total = 0
        for slot in slots:
            s = series[slot]
            w, h = s.cols * s.csp, s.rows * s.rsp
            sc = OVERVIEW_PX / max(w, h)
            total += min(s.n, OVERVIEW_MAX_SLICES) * tokens(round(w * sc), round(h * sc) + 17)
        return total

    def est_detail(s, box, size):
        w, h = box if box else (s.cols * s.csp, s.rows * s.rsp)
        return tokens(size * w / max(w, h), size * h / max(w, h) + 21)

    detail_est = sum(est_detail(s, b, size) for sec in DETAIL_SECTIONS for (s, _, b, _, size) in details[sec])
    dropped = []
    while len(overview) > 1 and est_overview(overview) + detail_est > EXPERT_TOKEN_CAP - CAP_MARGIN:
        dropped.append(overview.pop())  # lowest-priority overview series first
    parts = []  # (section, text or None, (series, slice, box, centre, size, label) or None)
    for slot in overview:
        s = series[slot]
        parts.append(("overview", f"Overview series {s.name}", None))
        for k in pick(anatomical_order(s, J, side), OVERVIEW_MAX_SLICES):
            parts.append(("overview", None, (s, k, None, None, OVERVIEW_PX, image_label(s, k, J, side, False))))
    for sec in DETAIL_SECTIONS:
        if details[sec]:
            parts.append((sec, section_text(sec, side), None))
        for s, k, box, centre, size in details[sec]:
            parts.append((sec, None, (s, k, box, centre, size, image_label(s, k, J, side, box is not None))))
    counts = {sec: sum(1 for p in parts if p[0] == sec and p[2]) for sec in ["overview", *DETAIL_SECTIONS]}
    info = dict(side=side, J=[round(float(v), 1) for v in J], J_from=how, overview_slots=overview,
                dropped_overview=dropped, sections=DETAIL_SECTIONS, overview_px=OVERVIEW_PX, counts=counts)
    lines = [f"   - {SECTION_TEXT['overview'].format(px=OVERVIEW_PX)} ({counts['overview']} images: "
             + ", ".join(series[s].name for s in overview) + ")"]
    lines += [f"   - {section_text(sec, side)} ({counts[sec]} images)" for sec in DETAIL_SECTIONS]
    head = [f"Findings to score: {', '.join(LABELS)}.", "Observations required for: " + ", ".join(REGIONS) + ".",
            series_list(series, side)]
    content = [dict(type="input_text", text="\n".join(head))]
    est = 0
    for _, text, image in parts:
        if text:
            content.append(dict(type="input_text", text=f"[{text}]"))
        if image:
            s, k, box, centre, size, label = image
            img = draw_image(renders, s, k, label, side, size, box, centre)
            est += tokens(*img.size)
            content.append(dict(type="input_image", image_url=png_url(img), detail="high"))
    info.update(images=sum(1 for p in parts if p[2]), est_image_tokens=est)
    return dict(sections="\n".join(lines), content=content, info=info)


# ---------------------------------------------------------------- blind-lateral pack

def half_specs(series: dict, J: np.ndarray, side: str, sign: int) -> list:
    """sign +1: lateral half (side unknown: patient right), -1: medial half (patient left)."""
    known = side in ("left", "right")
    px = HALF_PX["lateral" if sign > 0 or not known else "medial"]
    out = []
    for slot in ("sagittal-0", "sagittal-1"):
        s = series.get(slot)
        if s is None:
            continue
        ks = [k for k in anatomical_order(s, J, side)
              if SAG_RANGE_MM[0] <= sign * lateral_mm(s.centres[k, 0], J, side) <= SAG_RANGE_MM[1]]
        out += [(s, k, SAG_BOX_MM, J + np.array([0, 0, SAG_UP_MM]), px) for k in run_slices(s, ks, SAG_MAX)]
    s = first(series, "coronal-1", "coronal-0")
    if s:
        xdir = 1.0 if side == "left" else -1.0  # patient-x direction of +lateral_mm (LPS: +x = patient left)
        centre = J + np.array([sign * xdir * COR_SHIFT_MM, 0, 0])
        ks = [k for k in anatomical_order(s, J, side) if COR_RANGE_MM[0] <= s.centres[k, 1] - J[1] <= COR_RANGE_MM[1]]
        out += [(s, k, COR_BOX_MM, centre, px) for k in run_slices(s, ks, COR_MAX)]
    return out


def overview_specs(series: dict, J: np.ndarray, side: str) -> list:
    out = []
    for slots in (("sagittal-0", "sagittal-1"), ("coronal-1", "coronal-0"), ("axial-1", "axial-0")):
        s = first(series, *slots)
        if s:
            order = anatomical_order(s, J, side)
            cut = len(order) // 10
            out += [(s, k, None, None, BL_OVERVIEW_PX) for k in pick(order[cut:len(order) - cut], BL_OVERVIEW_PER_PLANE)]
    return out


def half_text(side: str, sign: int) -> str:
    known = side in ("left", "right")
    if sign > 0:
        name = "Lateral half, enlarged" if known else ("Patient-right half, enlarged (knee side unknown: this is the "
                                                       "lateral half if the knee is right, else the medial half)")
        direction, px = ("lateral" if known else "patient-right"), HALF_PX["lateral"]
    else:
        name = "Medial half, enlarged" if known else "Patient-left half, enlarged (the lateral half if the knee is left)"
        direction, px = ("medial" if known else "patient-left"), HALF_PX["medial"] if known else HALF_PX["lateral"]
    order = "lateral to medial" if known else "patient right to patient left"
    return (f"{name}: each sagittal series, consecutive slices from {-SAG_RANGE_MM[0]} mm on the other side of the "
            f"notch to {SAG_RANGE_MM[1]} mm {direction} of it (every acquired slice, or about every 3 mm in a thin 3D "
            f"series, at most {SAG_MAX} per series; the slices nearest the notch overlap it, so the roots are "
            f"included), shown {order}, cropped to the meniscus, the femoral condyle and the tibial plateau "
            f"({SAG_BOX_MM[0]} x {SAG_BOX_MM[1]} mm); then up to {COR_MAX} consecutive slices of one coronal series "
            f"(fat-suppressed if acquired) within about {-COR_RANGE_MM[0]} mm anterior to {COR_RANGE_MM[1]} mm "
            f"posterior of the tibial plateau centre (anterior horn, body, posterior horn and root), cropped to this "
            f"compartment ({COR_BOX_MM[0]} x {COR_BOX_MM[1]} mm); {px} px")


def blind_lateral_pack(renders, study_id: str) -> dict:
    """{sections, content, info} of the blind-lateral pack: overview + lateral half (+ medial half if side unknown)."""
    side, series, J, _ = _setup(renders, study_id)
    secs = [(f"Overview for orientation: {BL_OVERVIEW_PER_PLANE} evenly spaced whole slices of one series per plane "
             f"({BL_OVERVIEW_PX} px)", overview_specs(series, J, side)),
            (half_text(side, 1), half_specs(series, J, side, 1))]
    if side not in ("left", "right"):
        secs.append((half_text(side, -1), half_specs(series, J, side, -1)))
    secs = [(text, items) for text, items in secs if items]
    head = [f"Focused call: you are {BLIND_ROLE}; other readers score the other findings. "
            f"Findings to score: {', '.join(BLIND_FINDINGS)}.", series_list(series, side)]
    content = [dict(type="input_text", text="\n".join(head))]
    est = 0
    for text, items in secs:
        content.append(dict(type="input_text", text=f"[{text}]"))
        for s, k, box, centre, size in items:
            img = draw_image(renders, s, k, image_label(s, k, J, side, bool(box)), side, size, box, centre)
            est += tokens(*img.size)
            content.append(dict(type="input_image", image_url=png_url(img), detail="high"))
    if est > BLIND_TOKEN_CAP:
        raise ValueError(f"pack over cap: {est}")
    info = dict(side=side, J=[round(float(v), 1) for v in J], images=sum(len(i) for _, i in secs),
                est_image_tokens=est, sections=[len(items) for _, items in secs])
    return dict(sections="\n".join(f"   - {t} ({len(items)} images)" for t, items in secs), content=content, info=info)


PACKS = {"expert": expert_pack, "blind-lateral": blind_lateral_pack}


# ---------------------------------------------------------------- pack files

def encode_pack(study_id: str, kind: str, pack: dict, slots_present: int) -> list[tuple[str, bytes]]:
    """Zip members: one PNG per image (the bytes png_url made) and pack.json last."""
    files, items = [], []
    for c in pack["content"]:
        if c["type"] == "input_image":
            name = f"{study_id}/{len(files):03d}.png"
            files.append((name, base64.b64decode(c["image_url"][len(PNG_PREFIX):])))
            items.append(dict(type="input_image", file=name, detail=c["detail"]))
        else:
            items.append(c)
    record = dict(study_id=study_id, kind=kind, sections=pack["sections"], content=items, info=pack["info"],
                  slots_present=slots_present)
    files.append((f"{study_id}/pack.json", json.dumps(record).encode()))
    return files


def read_pack(archive: zipfile.ZipFile, study_id: str) -> dict:
    """pack.json with image content restored as data URLs, each PNG re-encoded by the local png_url."""
    from PIL import Image
    record = json.loads(archive.read(f"{study_id}/pack.json"))
    content = []
    for c in record["content"]:
        if c["type"] == "input_image":
            img = Image.open(io.BytesIO(archive.read(c["file"])))
            img.load()
            if img.mode != "L":
                raise ValueError("pack image is not 8-bit grayscale")
            c = dict(type="input_image", image_url=png_url(img), detail=c["detail"])
        content.append(c)
    return dict(record, content=content)


# ---------------------------------------------------------------- Kaggle kernel

_JOB: dict = {}


def render_exam(study_id: str, study_dir: Path, series_rows: list[dict], kinds) -> dict:
    """One exam from raw DICOMs -> {kind: zip members}. Series choice and window as the training store, with the
    plain rule (most usable slices, then finer pixels, then UID), not prefer-2d."""
    from knee import dicom
    study = dicom.raw_study(study_dir, series_rows, rule=None)
    entry, files = native_study(study)
    renders = Renders({study_id: entry}, dict(files).__getitem__)
    return {kind: encode_pack(study_id, kind, PACKS[kind](renders, study_id), len(entry["slots"])) for kind in kinds}


def _render(study_id: str) -> dict:
    try:
        return dict(study=study_id, files=render_exam(study_id, _JOB["root"] / study_id, _JOB["rows"][study_id],
                                                      _JOB["kinds"]))
    except Exception as error:  # one broken exam must not stop the run
        return dict(study=study_id, error=f"{type(error).__name__}: {error}"[:300])


def pin_pillow(wheels: Path | None) -> str:
    """Install Pillow PILLOW into a private folder when the image has another version (from local wheels if
    given, else from the package index), before anything imports PIL."""
    import importlib.metadata
    import subprocess
    import sys
    if importlib.metadata.version("pillow") == PILLOW:
        return PILLOW
    target = Path("/tmp/pinned-pillow")
    source = ["--no-index", "--find-links", str(wheels)] if wheels else []
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--quiet",
                    "--disable-pip-version-check", "--target", str(target), *source, f"pillow=={PILLOW}"], check=True)
    sys.path.insert(0, str(target))
    import PIL
    if PIL.__version__ != PILLOW:
        raise SystemExit(f"Pillow {PIL.__version__}: packs must be drawn with Pillow {PILLOW}")
    return PIL.__version__


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dicom-root", type=Path, required=True, help="competition train_series/ folder")
    parser.add_argument("--series-csv", type=Path, required=True, help="competition train_series.csv")
    parser.add_argument("--ids", type=Path, required=True, help="training exam IDs, one per line")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=sorted(PACKS), default=sorted(PACKS))
    parser.add_argument("--per-part", type=int, default=50, help="exams per packs-NNN.zip")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decoder-wheels", type=Path, help="offline JPEG decoder wheels (knee.dicom)")
    parser.add_argument("--pillow-wheels", type=Path, help="folder holding the pinned Pillow wheel")
    args = parser.parse_args(argv)
    pin_pillow(args.pillow_wheels)
    if args.decoder_wheels:
        from knee import dicom
        dicom.install_decoders(args.decoder_wheels)
    ids = args.ids.read_text().split()
    rows: dict[str, list] = {}
    with args.series_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.setdefault(row["StudyInstanceUID"], []).append(row)
    _JOB.update(root=args.dicom_root, rows=rows, kinds=args.kinds)
    for kind in args.kinds:
        (args.out / kind).mkdir(parents=True, exist_ok=False)
    failures, archives = {}, {}
    with multiprocessing.get_context("fork").Pool(args.workers, maxtasksperchild=50) as pool:
        for n, res in enumerate(pool.imap(_render, ids)):
            if n % args.per_part == 0:
                for archive in archives.values():
                    archive.close()
                archives = {kind: zipfile.ZipFile(args.out / kind / f"packs-{n // args.per_part:03d}.zip", "x")
                            for kind in args.kinds}
            if "error" in res:
                failures[res["study"]] = res["error"]
                continue
            for kind, files in res["files"].items():
                for name, data in files:
                    archives[kind].writestr(name, data)
    for archive in archives.values():
        archive.close()
    (args.out / "failures.json").write_text(json.dumps(failures, indent=1))
    print(json.dumps(dict(exams=len(ids), failed=len(failures))))


if __name__ == "__main__":
    main()

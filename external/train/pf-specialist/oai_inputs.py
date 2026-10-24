"""radiant-kg V7.3.1 E3: OAI image inputs (W / P / Z; no K for OAI) and the OAI item source.

Native directory (--out; competition native array formats):
  W/<series>.npy  uint8 [n<=64, 384, 384]: preprocess.load_series replica (0.4 mm canonical,
                  64-slice cap) on the series' slices AFTER the plane filter (V7.1 W cache format)
  P/<series>.npy  uint8 [n_slab, 250, 250]: kneecap box of the 0.4 mm frame, all band slices, slabs
  Z/<series>.npy  uint8 [n_slab, 500, 500]: same box resampled once from original pixels to 0.2 mm
  manifest.csv    append-only, one row per (series, kind W|P|Z): competition columns + OAI extras
  anchors.csv     append-only, one row per knee-visit: recomputed anchor, frozen V8 anchor, distance
  done.jsonl      append-only resume log (one line per finished knee-visit)

OAI series -> V7.1 slots (v8prep/oai_convert.SERIES_MAP): SAG_IW_TSE slot 0 (sagFS), COR_MPR
slot 1 (corFS), AX_MPR slot 2 (axFS; OAI's axial reformat of the sagittal 3D DESS WE), COR_IW_TSE
slot 4 (corNFS). P/Z exist for SAG_IW_TSE (groups 6/10) and AX_MPR (groups 7/11); OAI has no
sagNFS/axNFS series and no K.

Why rebuild from DICOM instead of using the V8 cache: every OAI *_MPR tarball holds one extra
reference image (ImageType 'PROJECTION IMAGE', sagittal orientation). load_series and
geo.series_geometry take the plane from the first file they read, so V8's cache, cache-meta plane,
views boxes and anchors are wrong for many MPR series (frozen tables: AX_MPR plane 'Sagittal' for
1255 series in oai_cache_meta.csv and 4747 in oai_views.csv). Plane rule: keep only slices whose
IOP normal's dominant axis equals the plane of the series type; excluded slices are counted in the
manifest. Geometry, anchor and boxes are then recomputed as for the competition (views.slab_prior
over the study's series with 64-capped geometry; views.crop_boxes(...)['kneecap']).

Identity rule: every kept DICOM must carry PatientID == participant and SeriesDescription ==
<TYPE>_<SIDE>, and, when present, ClinicalTrialTimePointID == visit and ImageComments / Laterality
== side; otherwise the series is 'identity_mismatch' and nothing is written for it.

Band and slab rules come from v731/bands.py (the data worker's helpers for the competition arrays).

Run: python oai_inputs.py prepare --grades grades.csv --series series.csv --out DIR --source-dir NDA_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "v731"), str(HERE.parents[1] / "reader")]
import bands  # noqa: E402  (puts reader/ on sys.path, pins 0.4 mm / 384)
import geo  # noqa: E402
import manifest as native_manifest  # noqa: E402  (data worker: the one manifest interpretation)
import oai_join  # noqa: E402
import preprocess  # noqa: E402
import views  # noqa: E402

EXPECTED_PLANE = {
    "SAG_IW_TSE": "Sagittal",
    "COR_MPR": "Coronal",
    "AX_MPR": "Axial",
    "COR_IW_TSE": "Coronal",
}
PZ_TYPES = ("SAG_IW_TSE", "AX_MPR")
MIN_W_SLICES = 3  # knee.build_study_table drops series with fewer cached slices
# the competition manifest columns (manifest.MANIFEST_COLS, incl. bands_version) + OAI extras
MANIFEST_COLS = [
    *native_manifest.MANIFEST_COLS,
    "series_type",
    "n_files",
    "n_excluded",
    "n_projection_excluded",
    "identity_checked",
]
ANCHOR_COLS = ["study", "x", "y", "z", "n_series", "frozen_x", "frozen_y", "frozen_z", "d_mm"]
assert MANIFEST_COLS[: len(native_manifest.MANIFEST_COLS)] == native_manifest.MANIFEST_COLS


# ---- reading
def read_series_dir(d):
    """All readable slices of a directory of DICOM files, via preprocess._read (pixels exactly as
    load_series).
    Returns (slices, n_bad, n_files); slice = (frame, iop, ipp, ps, inst, rows, cols, tags)."""
    import pydicom

    slices, bad, n_files = [], 0, 0
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            continue
        n_files += 1
        try:
            frames = preprocess._read(p)
            h = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:
            bad += 1
            continue
        tags = {
            "pid": str(h.get("PatientID", "")).strip(),
            "tp": str(h.get("ClinicalTrialTimePointID", "")).strip(),
            "desc": str(h.get("SeriesDescription", "")).strip(),
            "comments": str(h.get("ImageComments", "")).strip().upper(),
            "lat": str(h.get("Laterality", "")).strip().upper(),
            "itype": [str(x).upper() for x in (h.get("ImageType") or [])],
        }
        for f, iop, ipp, ps, inst in frames:
            slices.append((f, iop, ipp, ps, inst, f.shape[0], f.shape[1], tags))
    return slices, bad, n_files


def plane_of(iop):
    if iop is None:
        return None
    return preprocess.PLANES[int(np.abs(np.cross(iop[:3], iop[3:])).argmax())]


def filter_plane(slices, series_type):
    """Plane rule -> (kept, n_excluded, n_projection_excluded)."""
    want = EXPECTED_PLANE[series_type]
    kept = [s for s in slices if plane_of(s[1]) == want and s[2] is not None]
    exc = [s for s in slices if not (plane_of(s[1]) == want and s[2] is not None)]
    return kept, len(exc), sum("PROJECTION IMAGE" in s[7]["itype"] for s in exc)


def identity_problems(slices, participant, visit, side, series_type):
    """Identity rule -> list of problem strings (empty = ok)."""
    side_lr = {"RIGHT": "R", "LEFT": "L"}[side]
    probs = set()
    for s in slices:
        t = s[7]
        if t["pid"] != str(participant):
            probs.add("PatientID")
        if t["desc"] != f"{series_type}_{side}":
            probs.add("SeriesDescription")
        if t["tp"] and t["tp"] != visit:
            probs.add("ClinicalTrialTimePointID")
        if t["comments"] in ("LEFT", "RIGHT") and t["comments"] != side:
            probs.add("ImageComments")
        if t["lat"] in ("L", "R") and t["lat"] != side_lr:
            probs.add("Laterality")
    return sorted(probs)


def sort_slices(slices):
    """load_series ordering (all slices share one plane after filter_plane) -> (plane_i, sorted,
    gap)."""
    iop = slices[0][1]
    normal = np.cross(iop[:3], iop[3:])
    plane = int(np.abs(normal).argmax())
    sgn = preprocess.CANON[plane][0] * np.sign(normal[plane])
    ss = sorted(slices, key=lambda s: float(np.dot(s[2], normal)) * sgn)
    pos = [float(np.dot(s[2], normal)) for s in ss]
    return plane, ss, bands.series_gap(pos)


def cap_index(n):
    """load_series / geo.series_geometry 64-slice subsampling."""
    if n > preprocess.MAX_SLICES:
        return np.linspace(0, n - 1, preprocess.MAX_SLICES).round().astype(int)
    return np.arange(n)


def w_volume(sorted_slices, plane):
    """preprocess.load_series replica on already sorted slices: 64 cap, canonical 0.4 mm frames,
    1/99.5 window."""
    ss = [sorted_slices[i] for i in cap_index(len(sorted_slices))]
    vol = np.stack(
        [preprocess._resample(*preprocess._canon_slice(s[0], s[1], s[3], plane)) for s in ss]
    )
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8), float(lo), float(hi)


def affines(sorted_slices, plane):
    return [geo._slice_affine(s[5], s[6], s[1], s[2], s[3], plane) for s in sorted_slices]


# ---- writes
def npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(arr), allow_pickle=False)
    return b.getvalue()


def write_new(dest, data):
    """No-overwrite write (stage + fsync + os.link). Identical existing bytes accepted; different
    bytes refused."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    if dest.exists():
        if oai_join.sha256_file(dest) == digest:
            return digest
        raise FileExistsError(f"{dest} exists with different content; refusing to overwrite")
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    with open(tmp, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, dest)
    except FileExistsError:
        if oai_join.sha256_file(dest) != digest:
            raise
    finally:
        tmp.unlink()
    return digest


def safe_write(dest, arr, note=""):
    """write_new for one array -> (sha, status, note). A different existing file is preserved and
    reported as status 'conflict' (retried on resume, never overwritten)."""
    try:
        return write_new(dest, npy_bytes(arr)), "ok", note
    except FileExistsError as e:
        return None, "conflict", f"{note} {e}".strip()[:200]


def append_rows(path, rows, cols):
    """Append-only CSV (header written once); flushed and fsynced."""
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        if new:
            f.write(",".join(cols) + "\n")
        pd.DataFrame(rows, columns=cols).to_csv(f, header=False, index=False)
        f.flush()
        os.fsync(f.fileno())


# ---- sources


def local_fetch(root):
    """Source for already-downloaded data: <root>/<series key>/ (DICOM files), or <root>/<tarball
    basename>."""
    root = Path(root)

    def fetch(tarball, dest, key):
        os.makedirs(dest, exist_ok=True)
        d = root / key
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    os.link(f, Path(dest) / f.name)
            return
        tb = root / Path(tarball).name
        if tb.exists():
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(dest, filter="data")
            return
        raise FileNotFoundError(f"{key} not in {root}")

    return fetch


def _series_dir(root):
    """Directory holding the most files (tarballs are one flat folder; nesting tolerated)."""
    best, n_best = root, -1
    for d, _sub, files in os.walk(root):
        if len(files) > n_best:
            best, n_best = d, len(files)
    return best


# ---- one knee-visit
def prepare_study(task, out, fetch):
    """task: study, participant, visit, side (RIGHT|LEFT), series {type: (key, tarball)},
    frozen_anchor or None.
    Writes W/P/Z arrays; returns (manifest rows, anchor row)."""
    out = Path(out)
    st, side = task["study"], task["side"]
    side_lr = {"RIGHT": "R", "LEFT": "L"}[side]
    tmp = out / ".work" / "tmp" / st
    shutil.rmtree(tmp, ignore_errors=True)
    rows, ok = [], {}

    def row(key, stype, kind, **kw):
        r = {c: None for c in MANIFEST_COLS}
        r.update(
            {
                "study": st,
                "series": key,
                "kind": kind,
                "series_type": stype,
                "slot": oai_join.SLOT_OF_TYPE[stype],
                "bands_version": bands.BANDS_VERSION,  # every row, W included
            }
        )
        r.update(kw)
        return r

    def kinds_of(stype):
        return ["W", "P", "Z"] if stype in PZ_TYPES else ["W"]

    try:
        for stype, (key, tarball) in sorted(task["series"].items()):
            d = tmp / key
            try:
                fetch(tarball, str(d), key)
                slices, bad, n_files = read_series_dir(_series_dir(d))
            except Exception as e:
                rows += [
                    row(key, stype, k, status="fetch_error", note=f"{type(e).__name__}: {e}"[:200])
                    for k in kinds_of(stype)
                ]
                continue
            finally:
                shutil.rmtree(d, ignore_errors=True)
            base = {"n_files": n_files}
            if not slices:
                rows += [
                    row(key, stype, k, status="decode_error", note=f"bad_files={bad}", **base)
                    for k in kinds_of(stype)
                ]
                continue
            kept, n_exc, n_proj = filter_plane(slices, stype)
            base.update(n_excluded=n_exc, n_projection_excluded=n_proj)
            if len(kept) < 1:
                rows += [
                    row(key, stype, k, status="plane_mismatch", **base) for k in kinds_of(stype)
                ]
                continue
            probs = identity_problems(kept, task["participant"], task["visit"], side, stype)
            if probs:
                rows += [
                    row(key, stype, k, status="identity_mismatch", note=";".join(probs), **base)
                    for k in kinds_of(stype)
                ]
                continue
            plane_i, ss, gap = sort_slices(kept)
            affs = affines(ss, plane_i)
            ok[stype] = {
                "key": key,
                "plane_i": plane_i,
                "plane": preprocess.PLANES[plane_i],
                "slices": ss,
                "gap": gap,
                "affs": affs,
                "affs_cap": [affs[i] for i in cap_index(len(affs))],
                "base": {
                    **base,
                    "identity_checked": 1,
                    "plane": preprocess.PLANES[plane_i],
                    "gap_mm": round(gap, 4),
                    "n_raw_total": len(ss),
                    "ps_row": ss[0][3][0] if ss[0][3] else None,
                    "ps_col": ss[0][3][1] if ss[0][3] else None,
                    "note": f"bad_files={bad}" if bad else "",
                },
            }
        X = (
            views.slab_prior({v["key"]: (v["plane"], v["affs_cap"]) for v in ok.values()})
            if ok
            else None
        )
        fa = task.get("frozen_anchor")
        anchor_row = {"study": st, "n_series": len(ok)}
        if X is not None:
            anchor_row.update(
                x=round(float(X[0]), 3), y=round(float(X[1]), 3), z=round(float(X[2]), 3)
            )
        if fa is not None:
            anchor_row.update(frozen_x=fa[0], frozen_y=fa[1], frozen_z=fa[2])
            if X is not None:
                anchor_row["d_mm"] = round(float(np.linalg.norm(np.asarray(fa, float) - X)), 3)
        for stype, v in sorted(ok.items()):
            key, b = v["key"], v["base"]
            vol, lo, hi = w_volume(v["slices"], v["plane_i"])
            status = "ok" if len(vol) >= MIN_W_SLICES else "too_few_slices"
            sha, note = None, b["note"]
            if status == "ok":
                sha, status, note = safe_write(out / "W" / f"{key}.npy", vol, note)
            b = {**b, "note": note}
            rows.append(
                row(
                    key,
                    stype,
                    "W",
                    **b,
                    n_slab=len(vol),
                    slab_m=1,
                    lo=lo,
                    hi=hi,
                    sha256=sha,
                    status=status,
                )
            )
            if stype in PZ_TYPES:
                rows += pz_arrays(v, stype, X, side_lr, out, row)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows, anchor_row


def pz_arrays(v, stype, X, side_lr, out, row):
    """P and Z arrays of one plane-filtered series (bands.py rules; mirrors
    prepare_native.build_series)."""
    key, plane, plane_i, ss, b = v["key"], v["plane"], v["plane_i"], v["slices"], v["base"]
    if X is None:
        return [row(key, stype, k, **b, status="missing_geometry", note="no anchor") for k in "PZ"]
    cen = np.array([geo.centre(a) for a in v["affs"]])
    band_idx = np.flatnonzero(bands.band_mask(plane, cen, X))
    lo_mm, hi_mm = bands.band_limits(plane, X)
    m = bands.slab_size(v["gap"])
    groups = bands.slab_groups(range(len(band_idx)), m)
    # bands-3: 120 mm box with the same centre as the views 100 mm kneecap box (bands.box120)
    r0, c0 = bands.box120(*views.crop_boxes(plane, v["affs_cap"], X, side_lr)["kneecap"])
    info = {
        **b,
        "slab_m": m,
        "n_raw_band": len(band_idx),
        "n_slab": len(groups),
        "band_lo_mm": round(lo_mm, 3),
        "band_hi_mm": round(hi_mm, 3),
        "r0": int(r0),
        "c0": int(c0),
        "slab_members": json.dumps([[int(band_idx[i]) for i in g] for g in groups]),
    }
    if len(band_idx) == 0:
        return [row(key, stype, k, **info, status="empty_band") for k in "PZ"]
    band = set(int(i) for i in band_idx)
    subs, crops = [], {"P": [], "Z": []}
    for i, s in enumerate(ss):
        img, ps = bands.canon_slice(s[0], s[1], s[3], plane_i)
        f04 = bands.frame_p(img, ps)
        subs.append(f04[::4, ::4])
        if i in band:
            crops["P"].append(bands.cut_p(f04, r0, c0).copy())
            crops["Z"].append(bands.cut_z(bands.frame_z(img, ps), r0, c0).copy())
    lo, hi = bands.window_bounds(np.stack(subs))
    out_rows = []
    for k in "PZ":
        arr = bands.slab_average(bands.window(np.stack(crops[k]), lo, hi), groups)
        sha, status, note = safe_write(Path(out) / k / f"{key}.npy", arr, info["note"])
        out_rows.append(
            row(
                key,
                stype,
                k,
                **{**info, "note": note},
                lo=float(lo),
                hi=float(hi),
                sha256=sha,
                status=status,
            )
        )
    return out_rows


# ---- planning
def frozen_anchors(path):
    if not path or not Path(path).exists():
        return {}
    v = pd.read_csv(path, usecols=["study", "view", "r0", "c0", "z"])
    v = v[v.view == "anchor"]
    return {
        s: (float(a), float(b), float(c))
        for s, a, b, c in zip(v.study, v.r0, v.c0, v.z, strict=True)
    }


def make_tasks(grades, series, studies=None, anchors=None):
    """Grade-table rows -> prepare tasks (series key + tarball per type)."""
    path = dict(zip(series.key, series.tarball, strict=True))
    g = grades if studies is None else grades[grades.study.isin(set(studies))]
    tasks = []
    for r in g.itertuples():
        ser = {}
        for t, col in oai_join.SERIES_COLS.items():
            k = getattr(r, col)
            if isinstance(k, str) and k:
                ser[t] = (k, path[k])
        tasks.append(
            {
                "study": r.study,
                "participant": str(r.participant),
                "visit": r.visit,
                "side": r.side,
                "series": ser,
                "frozen_anchor": (anchors or {}).get(r.study),
            }
        )
    return tasks


def _init_worker():
    import cv2

    cv2.setNumThreads(1)


def _job(args):
    task, out, source_dir = args
    t0 = time.time()
    fetch = local_fetch(source_dir)
    try:
        rows, arow = prepare_study(task, out, fetch)
    except Exception as e:  # unexpected: record and keep going
        rows = [
            {
                **{c: None for c in MANIFEST_COLS},
                "study": task["study"],
                "series": k,
                "kind": "W",
                "series_type": t,
                "status": "error",
                "note": f"{type(e).__name__}: {e}"[:200],
            }
            for t, (k, _) in task["series"].items()
        ]
        arow = {"study": task["study"], "n_series": 0}
    return task["study"], rows, arow, round(time.time() - t0, 2)


# Resume (PF731-OAI-002): a knee-visit is skipped only when the CURRENT manifest row (last row per
# (series, kind), manifest.current_rows) of every expected (series, kind) is either a usable ok
# array or a recorded terminal absence. Anything else (no row, fetch/decode/error/conflict, ok row
# whose array is missing, unreadable, wrong shape or, with check_sha, wrong sha256) is redone.
# done.jsonl is a log only. Earlier rows and arrays are never deleted.
OAI_TERMINAL = frozenset(
    native_manifest.TERMINAL_STATUSES | {"identity_mismatch", "plane_mismatch", "too_few_slices"}
)
W_PX = 384


def expected_outputs(task):
    return [
        (key, kind)
        for t, (key, _) in sorted(task["series"].items())
        for kind in (["W", "P", "Z"] if t in PZ_TYPES else ["W"])
    ]


def usable_reason(row, root, check_sha=False):
    """'' when the current row is an ok, readable array of the recorded shape (and sha256)."""
    if row["kind"] != "W":
        return native_manifest.usable_reason(row, root, check_sha)
    if row["status"] != "ok":
        return f"status {row['status']}"
    if not native_manifest.same_bands(row):
        return f"bands version {row.get('bands_version')!r} != {bands.BANDS_VERSION!r}"
    p = Path(root) / "W" / f"{row['series']}.npy"
    if not p.exists():
        return "array missing"
    try:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        shape, dtype = a.shape, a.dtype
        del a
    except Exception as e:
        return f"unreadable: {type(e).__name__}"
    if dtype != np.uint8 or shape != (int(row["n_slab"]), W_PX, W_PX):
        return f"shape/dtype {shape} {dtype}"
    if check_sha and oai_join.sha256_file(p) != row["sha256"]:
        return "sha256 mismatch"
    return ""


def study_complete(task, cur, root, check_sha=False):
    """cur: {(series, kind): current manifest row dict}."""
    for key, kind in expected_outputs(task):
        r = cur.get((key, kind))
        if r is None or not native_manifest.same_bands(r):  # other band version: rebuild
            return False
        if r["status"] == "ok":
            why = usable_reason(r, root, check_sha)
            if why and why != native_manifest.BAND_TOO_THIN:  # too thin = terminal absence
                return False
        elif r["status"] not in OAI_TERMINAL:
            return False
    return True


def current_index(out):
    cur = native_manifest.current_rows(read_manifest(Path(out) / "manifest.csv"))
    return {(r["series"], r["kind"]): r for r in cur.to_dict("records")}


def cmd_prepare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grades = oai_join.load_table(a.grades)
    series = pd.read_csv(a.series, dtype=str)
    want = list(grades.study)
    cur = current_index(out)
    tasks = [
        t
        for t in make_tasks(grades, series, want, frozen_anchors(a.frozen_views))
        if not study_complete(t, cur, out, a.check_sha)
    ]
    print(
        f"prepare: {len(want)} studies requested, {len(want) - len(tasks)} already done, "
        f"{len(tasks)} to do",
        flush=True,
    )
    t0, n_ok, nbytes = time.time(), 0, 0
    with (
        open(out / "prepare_progress.jsonl", "a") as log,
        Pool(a.workers, initializer=_init_worker) as p,
    ):
        for i, (st, rows, arow, dt) in enumerate(
            p.imap_unordered(_job, [(t, str(out), a.source_dir) for t in tasks], chunksize=1)
        ):
            append_rows(out / "manifest.csv", rows, MANIFEST_COLS)
            append_rows(out / "anchors.csv", [arow], ANCHOR_COLS)
            with open(out / "done.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {"study": st, "seconds": dt, "status": sorted({r["status"] for r in rows})}
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
            for r in rows:
                if r["status"] == "ok":
                    n_ok += 1
                    nbytes += (out / r["kind"] / f"{r['series']}.npy").stat().st_size
            if i % 20 == 0 or i == len(tasks) - 1:
                el = time.time() - t0
                msg = {
                    "done": i + 1,
                    "total": len(tasks),
                    "ok_arrays": n_ok,
                    "gb_written": round(nbytes / 1e9, 3),
                    "elapsed_s": round(el),
                    "studies_per_min": round((i + 1) / el * 60, 2),
                }
                log.write(json.dumps(msg) + "\n")
                log.flush()
                print(json.dumps(msg), flush=True)


MANIFEST_DTYPES = {
    "series": str,
    "study": str,
    "kind": str,
    "status": str,
    "sha256": str,
    "bands_version": str,
}


def read_manifest(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=MANIFEST_COLS)
    return pd.read_csv(path, dtype=MANIFEST_DTYPES)


# ---- item source
class OAISource:
    """E3 OAI rows in the contract's item format, backed by the native directory.

    A knee-visit is usable when at least one of its series has an ok W array; its P/Z groups are
    filled from the
    native P/Z arrays of the series picked for slots 0 and 2 (masked when absent). K is always
    masked. y_pf is NaN, y_grades the [8, 3] cumulative targets, is_oai True. Left knees are
    mirrored at load (dataset.PFDataset)."""

    def __init__(
        self,
        grades_csv,
        native_root,
        k=8,
        k_eval=12,
        check_sha=False,
    ):
        from dataset import (
            NativeStore,
        )  # model worker's module (same directory); imported lazily for the preparer

        self.k, self.k_eval = int(k), int(k_eval)
        self.root = Path(native_root)
        self.grades = oai_join.load_table(grades_csv).set_index("study", drop=False)
        cur = current_index(self.root)  # LAST row per (series, kind) (PF731-DATA-002)
        self.ok = {
            (k, s) for (s, k), r in cur.items() if not usable_reason(r, self.root, check_sha)
        }
        self.native = NativeStore(self.root, ok={x for x in self.ok if x[0] in ("P", "Z")})
        self.table = {}
        for r in self.grades.itertuples():
            slots = [[] for _ in range(6)]
            for t, col in oai_join.SERIES_COLS.items():
                key = getattr(r, col)
                if isinstance(key, str) and ("W", key) in self.ok:
                    slots[oai_join.SLOT_OF_TYPE[t]].append(key)
            if any(slots):
                self.table[r.study] = slots
        self.mirror = set(self.grades.study[self.grades.side == "LEFT"])
        self.targets = dict(
            zip(
                self.grades.study,
                oai_join.cumulative_targets(
                    self.grades[list(oai_join.FIELD_NAMES)].to_numpy(np.float64)
                ),
                strict=True,
            )
        )

    def ready(self, split):
        g = self.grades[self.grades.split == split]
        return [s for s in g.study if s in self.table]

    def _items(self, split, train):
        studies = self.ready(split)
        return oai_items_class()(
            studies,
            self.table,
            str(self.root / "W"),
            self.native,
            y_pf=None,
            k=self.k if train else self.k_eval,
            train=train,
            mirror=self.mirror,
            k_series={},
            groups=range(15),
            grades={s: self.targets[s] for s in studies},
            w_loader=NpyLoader(self.root / "W"),
        )

    def train_items(self):
        return self._items("train", True)

class NpyLoader:
    """Picklable W loader (memory-mapped <dir>/<series>.npy) for DataLoader worker processes."""

    def __init__(self, d):
        self.d = str(d)

    def __call__(self, series):
        return np.load(f"{self.d}/{series}.npy", mmap_mode="r")


_OAI_ITEMS = None


def oai_items_class():
    """dataset.PFDataset subclass (built once, module-level so it pickles): sets y_grades and
    is_oai."""
    global _OAI_ITEMS
    if _OAI_ITEMS is None:
        from dataset import PFDataset

        class OAIItems(PFDataset):
            def __init__(self, *args, grades=None, **kw):
                super().__init__(*args, **kw)
                self.grades = grades or {}

            def __getitem__(self, i):
                it = super().__getitem__(i)
                it["y_grades"] = np.asarray(self.grades[self.studies[i]], np.float32).copy()
                it["is_oai"] = True
                return it

        OAIItems.__qualname__ = OAIItems.__name__ = "OAIItems"
        globals()["OAIItems"] = OAIItems
        _OAI_ITEMS = OAIItems
    return _OAI_ITEMS


def __getattr__(
    name,
):  # PEP 562: lets pickle find oai_inputs.OAIItems in DataLoader worker processes
    if name == "OAIItems":
        return oai_items_class()
    raise AttributeError(name)


def main():
    ap = argparse.ArgumentParser(description="OAI knee visits -> W/P/Z arrays (prepare)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--grades", required=True, help="oai_join.py grades.csv")
    p.add_argument("--series", required=True, help="oai_join.py series.csv")
    p.add_argument("--out", required=True, help="native directory (W/ P/ Z/ manifest.csv)")
    p.add_argument("--source-dir", required=True, help="the NDA DICOM tarballs or one folder per series")
    p.add_argument("--frozen-views", default="", help="optional: his V8 views table, only for anchors.csv distances")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--check-sha", action="store_true", help="resume: re-hash ok arrays too")
    a = ap.parse_args()
    cmd_prepare(a)


if __name__ == "__main__":
    main()

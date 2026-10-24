"""V7.3.1 native P / Z / K view arrays from the original DICOM archive
(CONTRACT.md "Native view arrays").

Commands:
  counts  header-only: band raw / slab counts per series and study for the 4345 train+develop
          labels_D.csv studies (gold58 excluded, PF731-DATA-001; no label columns are read)
          -> counts.csv, counts_series.csv, caps.json, targets.csv
          counts --gold --out DIR: the same tables for the gold58 studies only (final-stage
          retrain on 4403; NOT round 1), caps.json not written (caps stay over the 4345)
  build   pixels: for every target series write P/<series>.npy and Z/<series>.npy (V7.1
          sagittal/axial slot candidates) and K/<series>.npy (k_series.csv) with no-overwrite
          atomic writes; manifest.csv is append-only, one row per (series, kind) per attempt
          (the last row is current). Resume (PF731-DATA-002) skips only series whose needed
          kinds are terminal (missing_geometry / empty_band / missing_series) or ok with the
          array present and matching sha256 / shape; failed rows and missing or changed arrays
          are rebuilt. Before any output, 5 random target series replay the original 64-slice
          cache path and must equal cache384_train exactly (np.array_equal), through
          preprocess.load_series AND through this module's canonical-frame code; slice centres
          must match slice_geometry.parquet. --gate-tolerance 1 ('tol1', for other platforms):
          load_series == frame code stays exact, shape/dtype/centres exact, and vs the Mac cache
          max |diff| <= 1 with <= 1e-4 of voxels differing; gate_mode is written in every row
          and per-series max_diff/n_diff/frac in gate_receipt-passN.json. Each pass ends with
          coverage-passN.csv/.json (every needed series/kind: ok / terminal(reason) / failed /
          missing).
  build selection: --studies FILE (one study id per line) | --all (train + develop studies)
          | --gold (gold58 only; NOT part of round 1 - do not run before the final
          exploratory evaluation).

Run: python v731/prepare_native.py counts
     python v731/prepare_native.py build --studies native/sample20.txt
     python v731/prepare_native.py build --all --workers 10
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import struct
import sys
import tempfile
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bands  # noqa: E402  (puts reader/ on sys.path and pins 0.4 mm / 384)
import cv2  # noqa: E402
import geo  # noqa: E402
import knee  # noqa: E402
import manifest as nm  # noqa: E402
import preprocess  # noqa: E402
import pydicom  # noqa: E402
import views  # noqa: E402

cv2.setNumThreads(1)

R = Path("/Users/user/projects/radiant-kg")
WORK = R / "restricted_local/branch_b_work"
HEADERS = R / "restricted_local/v8-recovery/prep/headers"
ZIP = R / "rsna-knee-abnormality-detection.zip"
OUT = R / "restricted_local/v731-inputs/native"
CACHE = WORK / "cache384_train"
SLOT_GROUPS = {
    0: "sagFS",
    2: "axFS",
    3: "sagNFS",
    5: "axNFS",
}  # P groups 6..9 / Z groups 10..13 in this order
GATE_N, GATE_SEED = 5, 20261016
# PF731-DATA-002: manifest semantics (append-only, last row per (series, kind) is current, usable =
# ok + array with matching sha256/shape, terminal statuses) live in manifest.py for every reader.
MANIFEST_COLS = nm.MANIFEST_COLS
TERMINAL_STATUSES = nm.TERMINAL_STATUSES
read_manifest = nm.read_manifest
latest_rows = nm.current_rows
row_state = nm.row_state


# ---------------------------------------------------------------------------------- archive access
def index_archive(zip_path, series_ids):
    """{series: [(name, header_offset, compress_size, file_size, crc, compress_type)] in archive
    member order}."""
    import zipfile

    want = set(series_ids)
    out = {s: [] for s in want}
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            p = i.filename.split("/")
            if len(p) == 4 and p[0] == "train_series" and p[2] in want and p[3].endswith(".dcm"):
                out[p[2]].append(
                    (
                        i.filename,
                        i.header_offset,
                        i.compress_size,
                        i.file_size,
                        i.CRC,
                        i.compress_type,
                    )
                )
    return out


def read_member(fh, m, max_out=None):
    """Bytes of one archive member (stored or deflated) via its local header; CRC-checked when read
    in full.
    max_out: stop after about this many output bytes (header-only reads)."""
    name, off, csize, usize, crc, ctype = m
    fh.seek(off)
    hdr = fh.read(30)
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", hdr)
    assert sig == 0x04034B50, f"bad local header for {name}"
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    fh.seek(off + 30 + nlen + xlen)
    if ctype == 0:
        data = fh.read(csize if max_out is None else min(csize, max_out))
    elif ctype == 8:
        d = zlib.decompressobj(-15)
        if max_out is None:
            data = d.decompress(fh.read(csize)) + d.flush()
        else:
            parts, got, left = [], 0, csize
            while left > 0 and got < max_out:
                chunk = fh.read(min(65536, left))
                left -= len(chunk)
                o = d.decompress(chunk)
                parts.append(o)
                got += len(o)
            data = b"".join(parts)
    else:
        raise ValueError(f"unsupported compression {ctype} for {name}")
    if max_out is None:
        assert len(data) == usize and (zlib.crc32(data) & 0xFFFFFFFF) == crc, (
            f"CRC/size mismatch for {name}"
        )
    return data


# ---------------------------------------------------------------------------------------- geometry
def sorted_slices(slices):
    """load_series ordering (no 64 cap). slices: (frame_or_None, iop, ipp, ps, inst, rows, cols)
    tuples.
    Returns (plane_index, plane_name, sorted slices, positions along the normal (raw dot), gap) or
    raises ValueError
    when IOP/IPP is missing."""
    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is None or any(s[2] is None for s in slices):
        raise ValueError("missing IOP/IPP")
    normal = np.cross(iop[:3], iop[3:])
    plane = int(np.abs(normal).argmax())
    sgn = preprocess.CANON[plane][0] * np.sign(normal[plane])
    slices = sorted(slices, key=lambda s: float(np.dot(s[2], normal)) * sgn)
    pos = [float(np.dot(s[2], normal)) for s in slices]
    gap = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0
    return plane, preprocess.PLANES[plane], slices, pos, gap


def affines(slices, plane):
    """geo._slice_affine of every sorted slice (0.4 mm / 384 canonical frame)."""
    return [geo._slice_affine(s[5], s[6], s[1], s[2], s[3], plane) for s in slices]


def header_slices(fh, members):
    """Header-only slice tuples (frame None) for a series, in archive member order."""
    out = []
    for m in members:
        ds = None
        for cap in (1 << 20, None):
            try:
                ds = pydicom.dcmread(io.BytesIO(read_member(fh, m, cap)), stop_before_pixels=True)
                _ = ds.ImagePositionPatient, ds.Rows
                break
            except Exception:
                ds = None
        if ds is None:
            continue
        iop = (
            [float(v) for v in ds.ImageOrientationPatient]
            if "ImageOrientationPatient" in ds
            else None
        )
        ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
        ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
        out.append(
            (
                None,
                iop,
                ipp,
                ps,
                float(ds.get("InstanceNumber") or 0),
                int(ds.Rows),
                int(ds.Columns),
            )
        )
    return out


# ------------------------------------------------------------------------------------------ inputs
def load_inputs():
    lab = pd.read_csv(
        WORK / "labels_D.csv", usecols=["StudyInstanceUID", "is_gold", "fold"]
    )  # no label columns
    ser = pd.read_csv(WORK / "data/train_series.csv")
    meta = pd.read_csv(WORK / "cache384_train_meta.csv")
    table = knee.build_study_table(
        ser, dict(zip(meta.series, meta.n.fillna(0), strict=False))
    )  # V7.1 exactly (no t1 list)
    v = pd.read_csv(WORK / "views_train.csv")
    anc = v[v.view == "anchor"].set_index("study")
    anchor = {s: np.array([r.r0, r.c0, r.z], float) for s, r in anc.iterrows()}
    kp = v[v.view == "kneecap"]
    # bands-3: the 120 mm box with the same intended centre as the views_train 100 mm kneecap box
    box = {s: bands.box120(r0, c0) for s, r0, c0 in zip(kp.series, kp.r0, kp.c0, strict=True)}
    hdr = pd.read_parquet(HEADERS / "series_headers.parquet")
    hdr = hdr[hdr.split == "train"].set_index("series")
    kdf = pd.read_csv(OUT / "k_series.csv")
    return lab, table, anchor, box, hdr, kdf


def target_table(lab, table, kdf):
    """One row per (study, series): slots (sag/ax V7.1 candidates, ';'-joined), is_k, subset."""
    sub = {}
    for st, g, f in zip(lab.StudyInstanceUID, lab.is_gold, lab.fold, strict=False):
        sub[st] = "gold" if g == 1 else ("develop" if f == -2 else ("train" if f >= 0 else "other"))
    kmap = dict(zip(kdf.study, kdf.series, strict=False))
    rows = []
    for st in lab.StudyInstanceUID:
        slots = table.get(st, [[] for _ in range(knee.N_SLOTS)])
        ser = {}
        for s in SLOT_GROUPS:
            for sr in slots[s]:
                ser.setdefault(sr, []).append(s)
        if st in kmap:
            ser.setdefault(kmap[st], [])
        for sr, sl in ser.items():
            rows.append(
                {
                    "study": st,
                    "series": sr,
                    "slots": ";".join(str(x) for x in sl),
                    "is_k": sr == kmap.get(st),
                    "subset": sub[st],
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ counts
_FH = None


def _init_worker():
    global _FH
    cv2.setNumThreads(1)
    _FH = open(ZIP, "rb")  # noqa: SIM115 (one handle per worker process for its lifetime)


def _count_job(task):
    series, members = task
    try:
        sl = header_slices(_FH, members)
        plane_i, plane, sl, _pos, gap = sorted_slices(sl)
        cen = np.array([geo.centre(a) for a in affines(sl, plane_i)])
        return series, {
            "plane_hdr": plane,
            "n_raw": len(sl),
            "gap": gap,
            "centres": cen,
            "src": "zip_headers",
        }
    except Exception as e:
        return series, {"error": f"{type(e).__name__}: {e}"[:200]}


def series_count_row(plane, centres, gap, anchor):
    if anchor is None:
        return {
            "n_raw_band": 0,
            "n_slab": 0,
            "slab_m": bands.slab_size(gap),
            "status": "missing_anchor",
        }
    if bands.band_axis(plane) is None:
        return {
            "n_raw_band": 0,
            "n_slab": 0,
            "slab_m": bands.slab_size(gap),
            "status": "coronal_geometry",
        }
    mask = bands.band_mask(plane, centres, anchor)
    m = bands.slab_size(gap)
    nb = int(mask.sum())
    lo, hi = bands.band_limits(plane, anchor)
    return {
        "n_raw_band": nb,
        "n_slab": len(bands.slab_groups(range(nb), m)),
        "slab_m": m,
        "band_lo_mm": round(lo, 3),
        "band_hi_mm": round(hi, 3),
        "status": "ok" if nb else "empty_band",
    }


def counts_studies(lab, gold=False):
    """Studies of a counts run: default the 4345 train+develop (PF731-DATA-001: the 58 are
    excluded from counts and caps); gold=True: the gold58 only (no caps are written for them)."""
    return lab[lab.is_gold == 1] if gold else lab[lab.is_gold == 0]


def cmd_counts(a):
    t0 = time.time()
    gold = bool(getattr(a, "gold", False))
    lab, table, anchor, box, hdr, kdf = load_inputs()
    lab = counts_studies(lab, gold)
    tt = target_table(lab, table, kdf)
    geom = pd.read_parquet(
        HEADERS / "slice_geometry.parquet",
        columns=["series", "raw_rank", "pos_canon_mm", "centre_x", "centre_y", "centre_z"],
    )
    geom = geom[geom.series.isin(set(tt.series))]
    gcount = geom.groupby("series").size()
    need_zip = [
        s for s in tt.series.unique() if gcount.get(s, 0) != int(hdr.loc[s, "n_frames_total"])
    ]
    print(
        f"targets {len(tt)} rows / {tt.series.nunique()} series; slice_geometry complete for "
        f"{tt.series.nunique() - len(need_zip)}, zip headers needed for {len(need_zip)}",
        flush=True,
    )
    info = {}
    for s, g in geom[~geom.series.isin(set(need_zip))].groupby("series"):
        g = g.sort_values("raw_rank")
        pos = g.pos_canon_mm.to_numpy()
        info[s] = {
            "plane_hdr": hdr.loc[s, "plane"],
            "n_raw": len(g),
            "gap": bands.series_gap(pos),
            "centres": g[["centre_x", "centre_y", "centre_z"]].to_numpy(),
            "src": "slice_geometry",
        }
    if need_zip:
        idx = index_archive(ZIP, need_zip)
        with Pool(a.workers, initializer=_init_worker) as p:
            for i, (s, r) in enumerate(
                p.imap_unordered(_count_job, [(s, idx[s]) for s in need_zip], chunksize=2)
            ):
                info[s] = r
                if i % 100 == 0:
                    print(f"zip headers {i}/{len(need_zip)} {time.time() - t0:.0f}s", flush=True)
    rows = []
    for r in tt.itertuples():
        inf = info.get(r.series, {"error": "no geometry"})
        base = {
            "study": r.study,
            "series": r.series,
            "slots": r.slots,
            "is_k": r.is_k,
            "subset": r.subset,
            "csv_plane": None,
            "plane": inf.get("plane_hdr"),
            "src": inf.get("src"),
            "n_raw": inf.get("n_raw", 0),
            "gap_mm": round(inf.get("gap", float("nan")), 4),
            "n_frames_total": int(hdr.loc[r.series, "n_frames_total"]),
            "has_box": r.series in box,
            "bands_version": bands.BANDS_VERSION,
        }
        if "error" in inf:
            base.update(
                {"n_raw_band": 0, "n_slab": 0, "status": "missing_geometry", "note": inf["error"]}
            )
        else:
            base.update(
                series_count_row(inf["plane_hdr"], inf["centres"], inf["gap"], anchor.get(r.study))
            )
        rows.append(base)
    cs = pd.DataFrame(rows)
    ser = pd.read_csv(WORK / "data/train_series.csv").set_index("SeriesInstanceUID")
    cs["csv_plane"] = ser.loc[cs.series, "Anatomical_Plane"].values
    # per study summary
    st_rows = []
    for st in lab.StudyInstanceUID:
        g = cs[cs.study == st]
        row = {
            "study": st,
            "bands_version": bands.BANDS_VERSION,
            "subset": g.subset.iloc[0] if len(g) else None,
            "has_anchor": st in anchor,
        }
        for s, name in SLOT_GROUPS.items():
            c = g[g.slots.str.split(";").apply(lambda x, s=s: str(s) in x)]
            row[f"{name}_n_cand"] = len(c)
            for col in ("n_raw", "n_raw_band", "n_slab"):
                row[f"{name}_{col}_max"] = int(c[col].max()) if len(c) else 0
            row[f"{name}_n_raw_band_min"] = int(c.n_raw_band.min()) if len(c) else 0
            row[f"{name}_n_slab_min"] = int(c.n_slab.min()) if len(c) else 0
        k = g[g.is_k]
        row["has_k"] = bool(len(k))
        row["k_series"] = k.series.iloc[0] if len(k) else ""
        for col in ("n_raw", "n_raw_band", "n_slab"):
            row[f"K_{col}"] = int(k[col].iloc[0]) if len(k) else 0
        row["k_gap_mm"] = float(k.gap_mm.iloc[0]) if len(k) else float("nan")
        st_rows.append(row)
    cst = pd.DataFrame(st_rows)
    for name, df in (("counts_series.csv", cs), ("counts.csv", cst), ("targets.csv", tt)):
        write_new(OUT / name, df.to_csv(index=False).encode())
    if gold:  # caps are defined over the 4345 only (PF731-DATA-001): never from the 58
        print(f"gold58 counts done in {time.time() - t0:.0f}s (caps.json not written)", flush=True)
        return
    caps = {
        "rule": f"{bands.BANDS_VERSION}; max slab count over the {len(cst)} train+develop "
        "studies (gold58 excluded, PF731-DATA-001), V7.1 slot candidates and K series",
        "per_group": {name: int(cst[f"{name}_n_slab_max"].max()) for name in SLOT_GROUPS.values()},
        "per_group_raw_band": {
            name: int(cst[f"{name}_n_raw_band_max"].max()) for name in SLOT_GROUPS.values()
        },
        "K": int(cst.K_n_slab.max()),
        "K_raw_band": int(cst.K_n_raw_band.max()),
    }
    caps["P"] = caps["Z"] = max(caps["per_group"].values())
    caps["bands_version"] = bands.BANDS_VERSION
    caps["axial_band_mm"], caps["sagittal_band_mm"] = bands.AXIAL_BAND_MM, bands.SAGITTAL_BAND_MM
    write_new(OUT / "caps.json", (json.dumps(caps, indent=2) + "\n").encode())
    print(json.dumps(caps, indent=1))
    print(f"counts done in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ writes
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_new(dest, data):
    """No-overwrite atomic write: stage, fsync, os.link into a fresh path. An existing destination
    with identical
    bytes is accepted (idempotent resume); different bytes raise FileExistsError. Returns sha256."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    if dest.exists():
        if sha256_file(dest) == digest:
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
        if sha256_file(dest) != digest:
            raise
    finally:
        tmp.unlink()
    return digest


def npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(arr), allow_pickle=False)
    return b.getvalue()


# ------------------------------------------------------------------------------------------- build
def decode_series(fh, members):
    """All readable slices (pixels + headers) in archive member order, via preprocess._read;
    (slices, n_bad)."""
    slices, bad = [], 0
    for m in members:
        try:
            data = read_member(fh, m)
            ds_rows = None
            for f, iop, ipp, ps, inst in preprocess._read(io.BytesIO(data)):
                ds_rows = f.shape
                slices.append((f, iop, ipp, ps, inst, ds_rows[0], ds_rows[1]))
        except Exception:
            bad += 1
    return slices, bad


def build_series(task, fh=None, out=None):
    """_build_series_rows with every row stamped bands_version = bands.BANDS_VERSION."""
    out = Path(out or task.get("out") or OUT)
    return [{**r, "bands_version": bands.BANDS_VERSION} for r in _build_series_rows(task, fh, out)]


def _build_series_rows(task, fh, out):
    """task: dict(study, series, members, kinds {'P','Z','K'} subset, slots, anchor, box (r0,c0) or
    None, side).
    Writes arrays, returns manifest rows (list of dicts)."""
    fh = fh or _FH
    st, sr, kinds = task["study"], task["series"], task["kinds"]
    rows = []

    def fail(status, note, extra=None):
        for k in sorted(kinds):
            r = {c: None for c in MANIFEST_COLS}
            r.update(
                {
                    "study": st,
                    "series": sr,
                    "kind": k,
                    "slot": task["slots"],
                    "status": status,
                    "note": note,
                }
            )
            r.update(extra or {})
            rows.append(r)
        return rows

    if not task["members"]:
        return fail("missing_series", "series absent from the archive")
    slices, bad = decode_series(fh, task["members"])
    if not slices:
        return fail("decode_error", f"no readable slices ({bad} bad)")
    try:
        plane_i, plane, slices, _pos, gap = sorted_slices(slices)
    except ValueError as e:
        return fail("missing_geometry", str(e))
    affs = affines(slices, plane_i)
    cen = np.array([geo.centre(a) for a in affs])
    base = {
        "plane": plane,
        "gap_mm": round(gap, 4),
        "n_raw_total": len(slices),
        "ps_row": slices[0][3][0] if slices[0][3] else None,
        "ps_col": slices[0][3][1] if slices[0][3] else None,
    }
    if task["anchor"] is None:
        return fail("missing_geometry", "no study anchor", base)
    if bands.band_axis(plane) is None:
        return fail("missing_geometry", "coronal geometry: no kneecap band", base)
    anchor = np.asarray(task["anchor"], float)
    mask = bands.band_mask(plane, cen, anchor)
    band_idx = np.flatnonzero(mask)
    lo_mm, hi_mm = bands.band_limits(plane, anchor)
    m = bands.slab_size(gap)
    groups = bands.slab_groups(range(len(band_idx)), m)
    base.update(
        {
            "slab_m": m,
            "n_raw_band": len(band_idx),
            "n_slab": len(groups),
            "band_lo_mm": round(lo_mm, 3),
            "band_hi_mm": round(hi_mm, 3),
            "slab_members": json.dumps([[int(band_idx[i]) for i in g] for g in groups]),
        }
    )
    boxes = {}
    if kinds & {"P", "Z"}:
        if task["box"] is None:
            for k in sorted(kinds & {"P", "Z"}):
                r = {c: None for c in MANIFEST_COLS}
                r.update(base)
                r.update(
                    {
                        "study": st,
                        "series": sr,
                        "kind": k,
                        "slot": task["slots"],
                        "status": "missing_geometry",
                        "note": "no views_train kneecap box",
                    }
                )
                rows.append(r)
            kinds = kinds - {"P", "Z"}
        else:
            boxes["P"] = boxes["Z"] = task["box"]
    if "K" in kinds:
        side = task.get("side", "R")
        # own full geometry (all raw slices), then the same 120 mm transform as P/Z
        boxes["K"] = bands.box120(*views.crop_boxes(plane, affs, anchor, side)["kneecap"])
    if not kinds:
        return rows
    if len(band_idx) == 0:
        for k in sorted(kinds):
            r = {c: None for c in MANIFEST_COLS}
            r.update(base)
            r.update(
                {
                    "study": st,
                    "series": sr,
                    "kind": k,
                    "slot": task["slots"],
                    "status": "empty_band",
                    "r0": boxes[k][0],
                    "c0": boxes[k][1],
                    "note": "",
                }
            )
            rows.append(r)
        return rows
    band_set = set(int(i) for i in band_idx)
    subs = []
    crops = {k: [] for k in kinds}
    zbox_keys = {}
    for i, s in enumerate(slices):
        img, ps = bands.canon_slice(s[0], s[1], s[3], plane_i)
        f04 = bands.frame_p(img, ps)
        subs.append(f04[::4, ::4])
        if i not in band_set:
            continue
        if "P" in kinds:
            crops["P"].append(bands.cut_p(f04, *boxes["P"]).copy())
        zk = [k for k in ("Z", "K") if k in kinds]
        if zk:
            f02 = bands.frame_z(img, ps)
            for k in zk:
                key = boxes[k]
                if key in zbox_keys and zbox_keys[key] != k:
                    crops[k].append(crops[zbox_keys[key]][-1])
                else:
                    zbox_keys.setdefault(key, k)
                    crops[k].append(bands.cut_z(f02, *key).copy())
    lo, hi = bands.window_bounds(np.stack(subs))
    sub = {"P": "P", "Z": "Z", "K": "K"}
    for k in sorted(kinds):
        arr = bands.slab_average(bands.window(np.stack(crops[k]), lo, hi), groups)
        digest = write_new(out / sub[k] / f"{sr}.npy", npy_bytes(arr))
        r = {c: None for c in MANIFEST_COLS}
        r.update(base)
        r.update(
            {
                "study": st,
                "series": sr,
                "kind": k,
                "slot": task["slots"],
                "r0": boxes[k][0],
                "c0": boxes[k][1],
                "lo": float(lo),
                "hi": float(hi),
                "sha256": digest,
                "status": "ok",
                "note": f"bad_files={bad}" if bad else "",
            }
        )
        rows.append(r)
    return rows


GATE_MODES = {0: "exact", 1: "tol1"}
GATE_TOL1_MAX_FRAC = 1e-4


def gate_compare(vol, mine, old, cen, geom_centres, tolerance=0):
    """Gate decision on arrays (pure). vol = preprocess.load_series on THIS platform, mine = this
    module's frame code (64 cap), old = cache384_train (built on the Mac), cen / geom_centres =
    capped slice centres here / slice_geometry.parquet.
    Always required: vol == mine exactly (platform self-consistency); shape and dtype of vol and
    mine equal old; centres equal (atol 1e-3 mm).
    tolerance 0 ('exact'): vol == old exactly.
    tolerance 1 ('tol1', cross-platform float rounding, integrator decision 2026-10-16): max |vol -
    old| <= 1 grey level AND fraction of differing voxels <= 1e-4.
    Returns a dict with mode, max_diff, n_diff, frac and the booleans; raises AssertionError."""
    mode = GATE_MODES[int(tolerance)]
    shape_ok = bool(vol.dtype == old.dtype == mine.dtype and vol.shape == old.shape == mine.shape)
    self_ok = bool(shape_ok and np.array_equal(vol, mine))
    cen_ok = (
        geom_centres is not None
        and geom_centres.shape == cen.shape
        and bool(np.allclose(cen, geom_centres, atol=1e-3))
    )
    if shape_ok:
        d = np.abs(vol.astype(np.int16) - old.astype(np.int16))
        max_diff, n_diff = int(d.max()) if d.size else 0, int(np.count_nonzero(d))
        frac = n_diff / max(d.size, 1)
    else:
        max_diff, n_diff, frac = None, None, None
    if mode == "exact":
        cache_ok = shape_ok and n_diff == 0
    else:
        cache_ok = shape_ok and max_diff <= 1 and frac <= GATE_TOL1_MAX_FRAC
    res = {
        "gate_mode": mode,
        "cache_shape": list(old.shape),
        "shape_dtype_equal": shape_ok,
        "load_series_equals_bands_frame": self_ok,
        "load_series_equal": bool(shape_ok and n_diff == 0),  # vs cache, exact
        "bands_frame_equal": bool(shape_ok and np.array_equal(mine, old)),  # vs cache, exact
        "centres_equal": cen_ok,
        "cache_within_tolerance": bool(cache_ok),
        "max_diff": max_diff,
        "n_diff": n_diff,
        "frac": frac,
    }
    assert self_ok and cen_ok and cache_ok, f"64-cap replay gate ({mode}) failed: {res}"
    return res


def gate_series(task, geom_centres, fh=None, cache_dir=CACHE, tmp_root=None, tolerance=0):
    """64-cap replay gate: (1) preprocess.load_series on the series extracted to a temp dir; (2)
    this module's canonical frame code with the 64 cap; (3) centres of the capped slices vs
    slice_geometry centre_x/y/z; compared with cache384_train/<series>.npy by gate_compare
    (tolerance 0 = exact, 1 = tol1). Returns a dict; raises AssertionError on any failure."""
    fh = fh or _FH
    sr = task["series"]
    old = np.load(cache_dir / f"{sr}.npy", allow_pickle=False)
    tmp_root = Path(tmp_root or task.get("out") or OUT)
    with tempfile.TemporaryDirectory(prefix=".gate-", dir=tmp_root) as tmp:
        for j, m in enumerate(task["members"]):
            with open(Path(tmp) / f"{j:05d}.dcm", "xb") as f:
                f.write(read_member(fh, m))
        assert preprocess.MAX_SLICES == 64
        vol, _info = preprocess.load_series(tmp, task.get("csv_plane"))
    slices, _bad = decode_series(fh, task["members"])
    plane_i, _plane, slices, _pos, _gap = sorted_slices(slices)
    keep = np.arange(len(slices))
    if len(slices) > preprocess.MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, preprocess.MAX_SLICES).round().astype(int)
    frames = np.stack(
        [
            bands.frame_p(*bands.canon_slice(slices[i][0], slices[i][1], slices[i][3], plane_i))
            for i in keep
        ]
    )
    lo, hi = bands.window_bounds(frames[:, ::4, ::4])
    mine = (bands.window(frames, lo, hi) * 255).round().astype(np.uint8)
    cen = np.array([geo.centre(a) for a in affines([slices[i] for i in keep], plane_i)])
    res = {"series": sr, "n_raw": len(slices)}
    res.update(gate_compare(vol, mine, old, cen, geom_centres, tolerance))
    return res


def _build_job(task):
    t = time.time()
    try:
        rows = build_series(task)
    except Exception as e:
        rows = [
            {
                **{c: None for c in MANIFEST_COLS},
                "study": task["study"],
                "series": task["series"],
                "kind": k,
                "slot": task["slots"],
                "status": "decode_error",
                "note": f"{type(e).__name__}: {e}"[:200],
                "bands_version": bands.BANDS_VERSION,
            }
            for k in sorted(task["kinds"])
        ]
    return task["series"], rows, time.time() - t


class GateSelectionError(RuntimeError):
    pass


def select_gate_ids(target_series, cache_dir, geom_series, n=GATE_N, seed=GATE_SEED):
    """PF731-DATA-003: deterministic reference series for the 64-cap replay gate, drawn from the
    invocation's whole target set (independent of what is still to do, so retry-only runs gate on
    the same fixed references): sorted targets that have a cache384_train array and slice_geometry
    rows, then random.Random(seed).sample of min(n, pool). Raises GateSelectionError when the pool
    is empty. Returns (ids, selection record)."""
    targets = sorted({str(s) for s in target_series})
    pool = [s for s in targets if s in geom_series and (Path(cache_dir) / f"{s}.npy").exists()]
    sel = {"targets": len(targets), "pool": len(pool), "seed": seed, "n": n}
    if not pool:
        raise GateSelectionError(
            f"empty gate reference pool: none of {len(targets)} target series has both a "
            f"cache384_train array in {cache_dir} and slice_geometry rows; refusing to build"
        )
    ids = random.Random(seed).sample(pool, min(n, len(pool)))
    sel["selected"] = ids
    return ids, sel


def refuse_gate(reason, selection):
    """Fail closed before any array is written: record the refused selection, then exit."""
    rec = {"status": "refused", "reason": reason, "selection": selection, "time": time.time()}
    write_new(next_pass_path("gate_receipt", ".json"), (json.dumps(rec, indent=1) + "\n").encode())
    with open(OUT / "build_progress.jsonl", "a") as f:
        f.write(json.dumps({"event": "gate_refused", **rec}) + "\n")
    sys.exit(f"gate refused: {reason}")


def _gate_job(args):
    task, cen, tolerance = args
    return gate_series(task, cen, tolerance=tolerance)


BUILD_COLS = [*MANIFEST_COLS, "gate_mode"]  # gate_mode: 'exact' | 'tol1' (provenance of the gate)


def manifest_write_cols(path, gate_mode):
    """Columns to append with. A new manifest gets BUILD_COLS. An existing manifest keeps its own
    header: BUILD_COLS as is; the legacy MANIFEST_COLS header (rows before gate_mode existed, all
    gated exact) only for exact builds; anything else, or tol1 on a legacy header, is refused so
    columns can never be misaligned or the gate provenance lost."""
    path = Path(path)
    if not path.exists():
        return BUILD_COLS
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    if header == BUILD_COLS:
        return BUILD_COLS
    if header == MANIFEST_COLS and gate_mode == "exact":
        return MANIFEST_COLS
    sys.exit(
        f"{path}: header {header[-3:]}... cannot record gate_mode={gate_mode}; build into a fresh "
        "native dir (manifests are append-only and never rewritten)"
    )


def needed_kinds(g):
    """Kinds a target series needs: P and Z for a V7.1 sagittal/axial slot candidate, K for the
    study's K series."""
    slots = any(str(s) not in ("", "nan", "None") for s in g.slots)
    return ({"P", "Z"} if slots else set()) | ({"K"} if g.is_k.any() else set())


def verify_ok_row(args):
    """(kind, series, n_slab, sha256, out) -> (series, kind, manifest.usable_reason)."""
    kind, sr, n_slab, digest, version, out = args
    row = {
        "kind": kind,
        "series": sr,
        "status": nm.OK_STATUS,
        "n_slab": n_slab,
        "sha256": digest,
        "bands_version": version,
    }
    return sr, kind, nm.usable_reason(row, out)


def verify_ok(man, out=None, pool=None):
    """{(series, kind): '' | reason} for every latest 'ok' row (pool.imap_unordered when given)."""
    out = out or OUT
    lr = latest_rows(man)
    ok = lr[lr.status == "ok"]
    args = [
        (k, s, n, d, v, str(out))
        for k, s, n, d, v in zip(
            ok.kind, ok.series, ok.n_slab, ok.sha256, ok.bands_version, strict=True
        )
    ]
    it = (
        pool.imap_unordered(verify_ok_row, args, chunksize=16) if pool else map(verify_ok_row, args)
    )
    return {(s, k): why for s, k, why in it}


def completed_series(tt, man, verified=None):
    """Series skipped on resume: every needed kind's latest row is terminal, or ok AND verified
    (array present with the
    recorded sha256/shape; `verified` from verify_ok, computed here when None). Failed rows and ok
    rows whose array is
    missing or changed are rebuilt."""
    if not len(man):
        return set()
    lr = latest_rows(man)
    if verified is None:
        verified = verify_ok(lr)
    state = {}
    for s, k, st, bv in zip(lr.series, lr.kind, lr.status, lr.bands_version, strict=True):
        terminal = row_state(st) == "terminal" and bv == bands.BANDS_VERSION
        why = verified.get((s, k))
        good = terminal or (st == "ok" and why in ("", nm.BAND_TOO_THIN))  # too thin = terminal
        state[(s, k)] = good
    return {
        sr
        for sr, g in tt.groupby("series")
        if all(state.get((sr, k), False) for k in needed_kinds(g))
    }


def coverage_table(tt, man, verified):
    """One row per needed (study, series, kind) of the targets: state ok / terminal / failed /
    missing, status, reason."""
    lr = latest_rows(man).set_index(["series", "kind"])
    rows = []
    for sr, g in tt.groupby("series"):
        for k in sorted(needed_kinds(g)):
            base = {"study": g.study.iloc[0], "series": sr, "kind": k, "slots": g.slots.iloc[0]}
            if (sr, k) not in lr.index:
                rows.append({**base, "state": "missing", "status": "", "reason": "no manifest row"})
                continue
            r = lr.loc[(sr, k)]
            st = row_state(r.status)
            why = "" if pd.isna(r.note) else str(r.note)
            if st == "terminal" and r.bands_version != bands.BANDS_VERSION:
                st, why = "failed", f"terminal row from bands version {r.bands_version!r}"
            v = verified.get((sr, k), "not verified")
            if r.status == "ok" and v == nm.BAND_TOO_THIN:
                st, why = "terminal", nm.BAND_TOO_THIN
            elif r.status == "ok" and v != "":
                st, why = "failed", f"ok row but {v}"
            rows.append({**base, "state": st, "status": r.status, "reason": why})
    return pd.DataFrame(rows)


def next_pass_path(stem, suffix):
    n = 1
    while (OUT / f"{stem}-pass{n}{suffix}").exists():
        n += 1
    return OUT / f"{stem}-pass{n}{suffix}"


def write_coverage(tt, pool=None):
    """coverage-passN.csv / .json for the given targets from the current manifest (fresh names,
    never overwritten)."""
    man = read_manifest(OUT / "manifest.csv")
    verified = verify_ok(man, OUT, pool)
    cov = coverage_table(tt, man, verified)
    csv_path = next_pass_path("coverage", ".csv")
    write_new(csv_path, cov.to_csv(index=False).encode())
    summ = {
        "manifest_rows": len(man),
        "targets_series": int(tt.series.nunique()),
        "targets_studies": int(tt.study.nunique()),
        "needed_rows": len(cov),
        "state": cov.state.value_counts().to_dict(),
        "by_kind_state": {
            f"{k}/{s}": int(n) for (k, s), n in cov.groupby(["kind", "state"]).size().items()
        },
        "terminal_reasons": {
            f"{s}: {r}": int(n)
            for (s, r), n in cov[cov.state == "terminal"]
            .groupby(["status", "reason"])
            .size()
            .items()
        },
        "failed": cov[cov.state.isin(["failed", "missing"])][
            ["series", "kind", "status", "reason"]
        ].to_dict("records"),
        "csv": str(csv_path),
    }
    write_new(csv_path.with_suffix(".json"), (json.dumps(summ, indent=1) + "\n").encode())
    return summ


def build_tasks(tt, anchor, box, side, kdf, members, csv_plane):
    tasks = []
    for sr, g in tt.groupby("series", sort=False):
        kinds = set()
        slots = sorted({x for s in g.slots for x in str(s).split(";") if x and x != "nan"})
        if slots:
            kinds |= {"P", "Z"}
        if g.is_k.any():
            kinds.add("K")
        st = g.study.iloc[0]
        tasks.append(
            {
                "study": st,
                "series": sr,
                "kinds": kinds,
                "slots": ";".join(slots),
                "members": members.get(sr, []),
                "anchor": anchor.get(st),
                "box": box.get(sr),
                "side": side.get(st, "R"),
                "csv_plane": csv_plane.get(sr),
                "out": str(OUT),  # workers are spawned: the output dir travels with the task
            }
        )
    return tasks


def cmd_build(a):
    t0 = time.time()
    lab, table, anchor, box, _hdr, kdf = load_inputs()
    tt = target_table(lab, table, kdf)
    if a.studies:
        want = [x.strip() for x in Path(a.studies).read_text().splitlines() if x.strip()]
        gold = set(lab[lab.is_gold == 1].StudyInstanceUID)
        assert not (set(want) & gold) or a.allow_gold, (
            "--studies includes gold58 studies; pass --gold separately"
        )
        tt = tt[tt.study.isin(set(want))]
    elif a.gold:
        tt = tt[tt.subset == "gold"]
    elif a.all:
        tt = tt[tt.subset.isin(["train", "develop"])]
    else:
        sys.exit("choose --studies FILE, --all or --gold")
    manifest_path = OUT / "manifest.csv"
    gate_mode = GATE_MODES[a.gate_tolerance]
    cols = manifest_write_cols(manifest_path, gate_mode)
    man = read_manifest(manifest_path)
    with Pool(
        a.workers
    ) as vp:  # PF731-DATA-002: resume skips only verified ok rows and terminal rows
        verified = verify_ok(man, OUT, vp)
    bad_ok = {key: why for key, why in verified.items() if why}
    done = completed_series(tt, man, verified)
    todo = tt[~tt.series.isin(done)]
    lr = latest_rows(man)
    retry = lr[lr.series.isin(set(todo.series)) & (lr.status != "ok")]
    print(
        f"resume: {len(bad_ok)} ok rows failed verification {list(bad_ok.items())[:5]}; "
        f"{len(retry)} failed rows to retry "
        f"{retry[['series', 'kind', 'status', 'note']].to_dict('records')[:5]}",
        flush=True,
    )
    side = pd.read_csv(WORK / "laterality_v2.csv").set_index("study")["side"].to_dict()
    ser = pd.read_csv(WORK / "data/train_series.csv")
    csv_plane = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane, strict=False))
    print(
        f"build: {tt.series.nunique()} target series ({tt.study.nunique()} studies); "
        f"done {len(done)}; "
        f"todo {todo.series.nunique()}",
        flush=True,
    )
    if todo.empty:
        print(
            "coverage:",
            json.dumps({k: v for k, v in write_coverage(tt).items() if k != "failed"}),
            flush=True,
        )
        return
    # PF731-DATA-003: the reference pool is fixed by the invocation's target set (not the todo list)
    # and must be non-empty and fully indexable before any output (fail closed).
    geom_series = set(
        pd.read_parquet(HEADERS / "slice_geometry.parquet", columns=["series"]).series
    )
    try:
        gate_ids, gate_sel = select_gate_ids(tt.series.unique(), CACHE, geom_series)
    except GateSelectionError as e:
        refuse_gate(
            str(e), {"targets": int(tt.series.nunique()), "todo": int(todo.series.nunique())}
        )
    members = index_archive(ZIP, set(todo.series.unique()) | set(gate_ids))
    print(f"archive indexed in {time.time() - t0:.0f}s", flush=True)
    no_members = [s for s in gate_ids if not members.get(s)]
    if no_members:
        refuse_gate(f"reference series absent from the archive: {no_members}", gate_sel)
    tasks = build_tasks(todo, anchor, box, side, kdf, members, csv_plane)
    tasks.sort(key=lambda t: -len(t["members"]))  # largest first for load balance
    ref_tasks = build_tasks(
        tt[tt.series.isin(set(gate_ids))], anchor, box, side, kdf, members, csv_plane
    )
    geom = pd.read_parquet(
        HEADERS / "slice_geometry.parquet",
        columns=["series", "cache_idx", "centre_x", "centre_y", "centre_z"],
    )
    geom = geom[geom.series.isin(set(gate_ids))].sort_values(["series", "cache_idx"])
    gcen = {
        s: g[["centre_x", "centre_y", "centre_z"]].to_numpy() for s, g in geom.groupby("series")
    }
    by_id = {t["series"]: t for t in ref_tasks}
    log = open(OUT / "build_progress.jsonl", "a")  # noqa: SIM115 (open for the whole build)
    with Pool(a.workers, initializer=_init_worker) as p:
        gres = p.map(_gate_job, [(by_id[s], gcen.get(s), a.gate_tolerance) for s in gate_ids])
        write_new(
            next_pass_path("gate_receipt", ".json"),
            (
                json.dumps(
                    {
                        "status": "ok",
                        "gate_mode": gate_mode,
                        "selection": gate_sel,
                        "results": gres,
                    },
                    indent=1,
                )
                + "\n"
            ).encode(),
        )
        log.write(json.dumps({"event": "gate", "time": time.time(), "results": gres}) + "\n")
        log.flush()
        print("gate ok:", json.dumps(gres), flush=True)
        new_file = not manifest_path.exists()
        with open(manifest_path, "a", newline="") as mf:
            if new_file:
                mf.write(",".join(cols) + "\n")
            n_ok = n_rows = 0
            nbytes = 0
            for i, (sr, rows, _dt) in enumerate(p.imap_unordered(_build_job, tasks, chunksize=1)):
                rows = [{**r, "gate_mode": gate_mode} for r in rows]
                pd.DataFrame(rows, columns=cols).to_csv(mf, header=False, index=False)
                mf.flush()
                os.fsync(mf.fileno())
                n_rows += len(rows)
                for r in rows:
                    if r["status"] == "ok":
                        n_ok += 1
                        nbytes += (OUT / r["kind"] / f"{sr}.npy").stat().st_size
                if i % 50 == 0 or i == len(tasks) - 1:
                    el = time.time() - t0
                    msg = {
                        "event": "progress",
                        "series_done": i + 1,
                        "series_total": len(tasks),
                        "rows": n_rows,
                        "ok_rows": n_ok,
                        "gb_written": round(nbytes / 1e9, 3),
                        "elapsed_s": round(el),
                        "series_per_min": round((i + 1) / el * 60, 1),
                    }
                    log.write(json.dumps(msg) + "\n")
                    log.flush()
                    print(json.dumps(msg), flush=True)
        summ = write_coverage(tt, p)
    print("coverage:", json.dumps({k: v for k, v in summ.items() if k != "failed"}), flush=True)
    print("coverage failed/missing:", json.dumps(summ["failed"][:20]), flush=True)
    print(f"build done in {time.time() - t0:.0f}s", flush=True)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counts")
    c.add_argument("--workers", type=int, default=10)
    c.add_argument(
        "--gold",
        action="store_true",
        help="gold58 only (final-stage retrain, not round 1); requires --out; no caps.json",
    )
    c.add_argument("--out", help=f"output dir (default {OUT}); must hold k_series.csv")
    b = sub.add_parser("build")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--studies")
    g.add_argument("--all", action="store_true")
    g.add_argument("--gold", action="store_true")
    b.add_argument(
        "--allow-gold",
        action="store_true",
        help="allow gold58 ids inside --studies (geometry only)",
    )
    b.add_argument("--workers", type=int, default=10)
    b.add_argument("--out", help=f"output dir (default {OUT}); must hold k_series.csv")
    b.add_argument(
        "--gate-tolerance",
        type=int,
        choices=sorted(GATE_MODES),
        default=0,
        help="64-cap replay gate vs cache384_train: 0 = exact (default); 1 = tol1 for other "
        "platforms (load_series == frame code still exact; shape/dtype/centres exact; "
        "max |diff| <= 1 and differing fraction <= 1e-4); recorded as gate_mode in every row",
    )
    a = ap.parse_args()
    if a.cmd == "counts" and a.gold and not a.out:
        ap.error("counts --gold needs --out (a fresh dir, never the round-1 native dir)")
    if getattr(a, "out", None):
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    cmd_counts(a) if a.cmd == "counts" else cmd_build(a)


if __name__ == "__main__":
    main()

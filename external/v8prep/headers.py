"""V8 prep: header-only DICOM pass over the competition data -> two parquet tables.

1. ``series_headers.parquet`` -- one row per series: file/frame counts, plane, the acquisition
   header fields of the first readable header, and *hints* (not labels) for T1 / PD / T2 / STIR /
   fat saturation, each with the token that triggered it so the tags can be audited later.
2. ``slice_geometry.parquet`` -- one row per slice the training cache keeps, in cache order
   (``cache_idx``), with IPP / IOP / spacing, signed position along the slice normal and the
   canonical-pixel affine from ``reader/geo._slice_affine``.

Slice order mirrors ``reader/preprocess.load_series`` (the function that builds the cache, also
via ``build_cache_from_zip.py``), not ``geo.series_geometry``:

* files are visited in directory-listing order (``os.listdir`` for ``--root``; zip member order
  for ``--zip``); a multi-frame file expands to frames with instance ``inst + i * 1e-3`` that all
  share the file-level IPP;
* the slice normal comes from the first slice that has an IOP; the plane is its largest normal
  component, else the csv ``Anatomical_Plane``, else Sagittal;
* slices are sorted by ``dot(IPP, normal) * canonical sign`` only when *every* slice has an IPP,
  otherwise by instance number (stable sort, so ties keep listing order);
* more than ``MAX_SLICES`` slices are subsampled with ``linspace(...).round()``.

Divergences of ``geo.series_geometry`` from ``load_series`` (this module follows load_series):

* geo silently drops files lacking IOP or IPP (or with unparsable values), so with any missing IPP
  it keeps fewer slices, subsamples a different list and still sorts by position, whereas
  load_series keeps every readable slice and falls back to instance-number order;
* geo returns no geometry when no file has IOP+IPP; load_series still builds a volume (plane from
  the csv), and returns a single blank slice for a series with no readable file at all
  (no slice rows here; ``n_kept`` = 0);
* geo takes the normal from the first file with IOP *and* IPP, load_series from the first with IOP
  -- only matters when IOP differs inside a series (``iop_consistent`` /
  ``iop_max_dev`` flag that; 1e-3 tolerance, since header IOPs carry rounding noise).

What a header pass cannot reproduce:

* files whose header reads but whose pixels fail are skipped by load_series (``n_bad`` in the
  cache meta); ``n_header_bad`` here counts only files whose header (or IOP/IPP/spacing values)
  fails, so the kept list can differ -- cross-check against the cache meta's ``n_raw``/``n``;
* ``build_cache_from_zip.py`` copies a series into a temp dir and calls load_series there, so the
  file order the cache saw was that temp dir's ``os.listdir`` hash order. Order only matters when
  sort keys tie (same position or instance) or IOP is inconsistent; ``tie_with_neighbor`` /
  ``n_sort_ties`` flag the ambiguous slices.

Usage:
    python3 -I experiments/branch_b/v8prep/headers.py --zip /workspace/comp_raw/<archive>.zip
    python3 -I experiments/branch_b/v8prep/headers.py --root DIR  # DIR/{split}_series/...
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reader"))
import geo
from preprocess import CANON, MAX_SLICES, PLANES

HEADER_FIELDS = [
    "SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence", "SequenceVariant",
    "ScanOptions", "MRAcquisitionType", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
    "EchoTrainLength", "MagneticFieldStrength", "Manufacturer", "ManufacturerModelName",
    "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns", "ImageType",
]  # fmt: skip
NUMERIC_FIELDS = {
    "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "EchoTrainLength",
    "MagneticFieldStrength", "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns",
}  # fmt: skip

_ZIP: zipfile.ZipFile | None = None
_ROOT: str | None = None


# ----------------------------------------------------------------------------- header reading


def _open_member(name: str) -> Any:
    """File-like for one DICOM file: a zip member (streamed) or a path in the root dir."""
    if _ZIP is not None:
        return _ZIP.open(name)
    assert _ROOT is not None
    return open(os.path.join(_ROOT, name), "rb")


def _read_header(name: str) -> pydicom.Dataset:
    # Same reader settings as preprocess._read (no force) but stop before the pixel data.
    with _open_member(name) as fp:
        return pydicom.dcmread(fp, stop_before_pixels=True)


def _n_frames(ds: pydicom.Dataset) -> int:
    """Frames preprocess._read produces: `list(arr) if arr.ndim == 3 else [arr]`."""
    nf = int(ds.get("NumberOfFrames", 1) or 1)
    spp = int(ds.get("SamplesPerPixel", 1) or 1)
    ndim = 2 + (nf > 1) + (spp > 1)
    if ndim != 3:
        return 1
    return nf if nf > 1 else int(ds.Rows)  # single-frame colour: load_series iterates rows (sic)


def _floats(ds: pydicom.Dataset, kw: str) -> list[float] | None:
    return [float(v) for v in ds[kw].value] if kw in ds else None


def _value(ds: pydicom.Dataset, kw: str) -> Any:
    if kw not in ds or ds[kw].value is None or ds[kw].value == "":
        return None
    v = ds[kw].value
    if kw == "PixelSpacing":
        return "\\".join(str(float(x)) for x in v)
    if kw in NUMERIC_FIELDS:
        try:
            return float(v[0] if isinstance(v, pydicom.multival.MultiValue) else v)
        except (TypeError, ValueError, IndexError):
            return None
    if isinstance(v, pydicom.multival.MultiValue | list | tuple):
        return "\\".join(str(x) for x in v)
    return str(v)


# ----------------------------------------------------------------------------- contrast hints

_FS_OPTS = ("FS", "SFS", "FAT_SAT", "FATSAT", "SPAIR", "CHESS", "SPECIAL_FS")
_FS_DESC = re.compile(
    r"fat[\s_\-]*sat|fatsat|(?<![a-z])(?:t1|t2|pd|dp|tse|fse)?fs(?![a-z])|f/s|stir|spair|spir"
    r"|tirm|ideal|dixon[\s_\-]*w|(?<![a-z])water|(?<![a-z])w(?:ater)?[\s_\-]*only"
    r"|(?<![a-z])we(?![a-z])"
)
_T1_DESC = re.compile(r"(?<![a-z0-9])t1(?![0-9])")
_T2_DESC = re.compile(r"(?<![a-z0-9])t2(?![0-9*])")
_PD_DESC = re.compile(r"(?<![a-z])(?:pd|dp)(?![a-z])|proton|dens|(?<![a-z])(?:pd|dp)(?=fs|w)")
_STIR_DESC = re.compile(r"stir|tirm|(?<![a-z])tir(?![a-z])")


def _desc_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(k) or "") for k in ("SeriesDescription", "ProtocolName")).lower()


def contrast_hints(row: dict[str, Any]) -> dict[str, Any]:
    """Hints only (never final labels): each `<x>_hint` has an `<x>_token` naming what matched."""
    desc = _desc_text(row)
    opts = [o.strip().upper() for o in str(row.get("ScanOptions") or "").split("\\") if o.strip()]
    seq = str(row.get("ScanningSequence") or "").upper()
    tr, te, ti = row.get("RepetitionTime"), row.get("EchoTime"), row.get("InversionTime")
    is_ir = "IR" in seq.split("\\") or "IR" in opts
    out: dict[str, Any] = {}

    def put(name: str, token: str | None) -> None:
        out[f"{name}_hint"] = token is not None
        out[f"{name}_token"] = token

    opt_fs = next((o for o in opts if o in _FS_OPTS), None)
    m = _FS_DESC.search(desc)
    put("fat_sat", f"opt:{opt_fs}" if opt_fs else (f"desc:{m.group(0)}" if m else None))

    m = _STIR_DESC.search(desc)
    short_ti = is_ir and ti is not None and ti < 300
    put("stir", f"desc:{m.group(0)}" if m else (f"phys:IR TI={ti:g}" if short_ti else None))

    m = _T1_DESC.search(desc)
    phys_t1 = tr is not None and te is not None and tr < 800 and te < 30 and not is_ir
    put("t1", f"desc:{m.group(0)}" if m else (f"phys:TR={tr:g} TE={te:g}" if phys_t1 else None))

    m = _PD_DESC.search(desc)
    phys_pd = tr is not None and te is not None and tr >= 1000 and te < 45 and not is_ir
    put("pd", f"desc:{m.group(0)}" if m else (f"phys:TR={tr:g} TE={te:g}" if phys_pd else None))

    m = _T2_DESC.search(desc)
    phys_t2 = tr is not None and te is not None and tr >= 1500 and te >= 60 and not is_ir
    put("t2", f"desc:{m.group(0)}" if m else (f"phys:TR={tr:g} TE={te:g}" if phys_t2 else None))
    return out


# ----------------------------------------------------------------------------- slice ordering


def cache_order(
    slices: list[dict[str, Any]], plane_name: str | None
) -> tuple[int, str, list[int], list[int], np.ndarray | None, float]:
    """Mirror of load_series' plane / sort / subsample on header slices (each has iop, ipp, inst).

    Returns (plane index, plane source, sorted order of input indices, kept positions in the
    sorted list, normal or None, canonical sign)."""
    iop = next((s["iop"] for s in slices if s["iop"] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane, source = int(np.abs(normal).argmax()), "iop"
    else:
        normal = None
        plane = PLANES.index(plane_name) if plane_name in PLANES else 0
        source = "csv" if plane_name in PLANES else "default"
    idx = list(range(len(slices)))
    sgn = 1.0
    if normal is not None and all(s["ipp"] is not None for s in slices):
        sgn = float(CANON[plane][0] * np.sign(normal[plane]))
        idx.sort(key=lambda i: float(np.dot(slices[i]["ipp"], normal)) * sgn)
    else:
        idx.sort(key=lambda i: slices[i]["inst"])
    keep = list(range(len(idx)))
    if len(idx) > MAX_SLICES:
        keep = [int(k) for k in np.linspace(0, len(idx) - 1, MAX_SLICES).round().astype(int)]
    return plane, source, idx, keep, normal, sgn


# ----------------------------------------------------------------------------- per series job


def _job(task: tuple[str, str, str, list[str], str | None]) -> tuple[dict[str, Any], list[dict]]:
    split, study, series, members, csv_plane = task
    srow: dict[str, Any] = {"study": study, "series": series, "split": split,
                            "n_files": len(members), "csv_plane": csv_plane}  # fmt: skip
    slices: list[dict[str, Any]] = []
    first: pydicom.Dataset | None = None
    bad, errs = 0, []
    for name in members:
        try:
            ds = _read_header(name)
            # the same conversions preprocess._read does; a failure there makes the file "bad"
            iop, ipp = _floats(ds, "ImageOrientationPatient"), _floats(ds, "ImagePositionPatient")
            ps = _floats(ds, "PixelSpacing")
            inst = float(ds.get("InstanceNumber") or 0)
            nfr = _n_frames(ds)
            rows, cols = int(ds.Rows), int(ds.Columns)
        except Exception as e:
            bad += 1
            errs.append(f"{type(e).__name__}: {e}"[:120])
            continue
        if first is None:
            first = ds
        rel = name.split("/", 1)[1]  # "{study}/{series}/{file}", identical in zip and dir mode
        for f in range(nfr):
            slices.append({"file": rel, "frame": f, "iop": iop, "ipp": ipp, "ps": ps,
                           "inst": inst + f * 1e-3, "rows": rows, "cols": cols})  # fmt: skip
    srow["n_header_bad"] = bad
    srow["header_error"] = errs[0] if errs else None
    srow["n_frames_total"] = len(slices)
    if first is not None:
        for kw in HEADER_FIELDS:
            srow[kw] = _value(first, kw)
        srow.update(contrast_hints(srow))
    if not slices:  # load_series returns one blank slice here
        srow.update(plane=csv_plane, plane_source="none", sort_mode="none", n_kept=0)
        return srow, []

    plane, source, order, keep, normal, sgn = cache_order(slices, csv_plane)
    by_pos = normal is not None and all(s["ipp"] is not None for s in slices)
    iops = np.array([s["iop"] for s in slices if s["iop"] is not None], float).reshape(-1, 6)
    iop_dev = float(np.abs(iops - iops[0]).max()) if len(iops) else np.nan
    keys = [
        float(np.dot(slices[i]["ipp"], normal)) * sgn if by_pos else slices[i]["inst"]
        for i in order
    ]
    ties = [
        (k > 0 and keys[k] == keys[k - 1]) or (k + 1 < len(keys) and keys[k] == keys[k + 1])
        for k in range(len(keys))
    ]
    srow.update(
        plane=PLANES[plane], plane_source=source, sort_mode="position" if by_pos else "instance",
        n_kept=len(keep), n_missing_ipp=sum(s["ipp"] is None for s in slices),
        n_missing_iop=sum(s["iop"] is None for s in slices), iop_max_dev=iop_dev,
        iop_consistent=bool(iop_dev < 1e-3) and len(iops) == len(slices),
        n_sort_ties=int(sum(ties)),
        slice_gap=float(np.median(np.abs(np.diff(sorted(keys))))) if by_pos and len(keys) > 1
        else None,
    )  # fmt: skip

    out = []
    nan3 = [np.nan] * 3
    for cache_idx, rank in enumerate(keep):
        s = slices[order[rank]]
        r: dict[str, Any] = {
            "study": study, "series": series, "cache_idx": cache_idx, "raw_rank": rank,
            "file": s["file"], "frame": s["frame"], "instance_number": s["inst"],
            "tie_with_neighbor": ties[rank],
        }  # fmt: skip
        ipp = s["ipp"] if s["ipp"] is not None else nan3
        iop = s["iop"] if s["iop"] is not None else [np.nan] * 6
        ps = s["ps"] if s["ps"] is not None else [np.nan] * 2
        r.update({f"ipp_{a}": float(v) for a, v in zip("xyz", ipp, strict=True)})
        r.update({f"iop_{k}": float(v) for k, v in enumerate(iop)})
        r.update(ps_row=float(ps[0]), ps_col=float(ps[1]), rows=s["rows"], cols=s["cols"])
        if normal is not None and s["ipp"] is not None:
            r["pos_normal_mm"] = float(np.dot(s["ipp"], normal))
            r["pos_canon_mm"] = r["pos_normal_mm"] * float(CANON[plane][0] * np.sign(normal[plane]))
        else:
            r["pos_normal_mm"] = r["pos_canon_mm"] = np.nan
        if s["iop"] is not None and s["ipp"] is not None:
            aff = geo._slice_affine(s["rows"], s["cols"], s["iop"], s["ipp"], s["ps"], plane)
            cen = geo.centre(aff)
        else:
            aff, cen = (nan3, nan3, nan3), nan3
        for nm, vec in zip(("O", "DR", "DC", "centre"), (*aff, cen), strict=True):
            r.update({f"{nm}_{a}": float(v) for a, v in zip("xyz", vec, strict=True)})
        out.append(r)
    return srow, out


def _job_safe(task: tuple[str, str, str, list[str], str | None]) -> tuple[dict, list[dict]]:
    try:
        return _job(task)
    except Exception as e:  # never lose a series silently
        split, study, series, members, csv_plane = task
        return {"study": study, "series": series, "split": split, "n_files": len(members),
                "csv_plane": csv_plane, "error": f"{type(e).__name__}: {e}"[:200]}, []  # fmt: skip


# ----------------------------------------------------------------------------- task discovery


def _init(zip_path: str | None, root: str | None) -> None:
    global _ZIP, _ROOT
    _ZIP = zipfile.ZipFile(zip_path) if zip_path else None
    _ROOT = root


def _worker_init(zip_path: str | None, root: str | None) -> None:
    """Forked workers reuse the parent's parsed central directory (819k entries take ~8 CPU-s to
    parse; 48 workers each parsing it took ~60 s on a 15-CPU quota) but need their own file handle,
    since a forked handle shares its read offset with the parent and siblings."""
    global _ZIP
    if zip_path and _ZIP is not None and _ZIP.filename == zip_path:
        _ZIP.fp = open(zip_path, "rb")  # noqa: SIM115 -- lives as long as the worker
    else:
        _init(zip_path, root)


def _csv_planes(read_csv: Any) -> dict[str, str]:
    try:
        se = read_csv()
    except (FileNotFoundError, KeyError):
        return {}
    return dict(zip(se.SeriesInstanceUID.astype(str), se.Anatomical_Plane, strict=True))


def discover(
    zip_path: str | None, root: str | None, splits: list[str]
) -> list[tuple[str, str, str, list[str], str | None]]:
    """Tasks (split, study, series, member names in listing order, csv plane). Opens the input in
    this process (``_init``) so forked workers inherit the parsed zip directory."""
    _init(zip_path, root)
    tasks = []
    if _ZIP is not None:
        zf = _ZIP
        names = zf.namelist()
        for split in splits:
            prefix = f"{split}_series/"
            members: dict[tuple[str, str], list[str]] = defaultdict(list)
            for name in names:  # same grouping as build_cache_from_zip.series_members
                if name.startswith(prefix) and not name.endswith("/"):
                    parts = name[len(prefix) :].split("/")
                    if len(parts) >= 3:
                        members[(parts[0], parts[1])].append(name)
            planes = _csv_planes(lambda s=split: pd.read_csv(zf.open(f"{s}_series.csv")))
            for (study, series), ms in members.items():
                tasks.append((split, study, series, ms, planes.get(series)))
    else:
        assert root is not None
        for split in splits:
            base = os.path.join(root, f"{split}_series")
            if not os.path.isdir(base):
                continue
            csv = os.path.join(root, f"{split}_series.csv")
            planes = _csv_planes(lambda csv=csv: pd.read_csv(csv))
            for study in sorted(os.listdir(base)):
                for series in sorted(os.listdir(os.path.join(base, study))):
                    rel = os.path.join(f"{split}_series", study, series)
                    ms = [os.path.join(rel, fn) for fn in os.listdir(os.path.join(root, rel))]
                    tasks.append((split, study, series, ms, planes.get(series)))
    return tasks


def run(
    zip_path: str | None, root: str | None, out_dir: str | None, splits: list[str],
    workers: int, limit: int = 0, verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    t0 = time.time()
    tasks = discover(zip_path, root, splits)
    if limit:
        tasks = tasks[:limit]
    nfiles = sum(len(t[3]) for t in tasks)
    if verbose:
        print(f"{len(tasks)} series, {nfiles} files (listing {time.time() - t0:.0f}s)", flush=True)
    srows, grows = [], []
    if workers <= 1:
        results: Any = map(_job_safe, tasks)
    else:
        ctx = multiprocessing.get_context("fork")
        pool = ctx.Pool(workers, initializer=_worker_init, initargs=(zip_path, root))
        results = pool.imap_unordered(_job_safe, tasks, chunksize=2)
    t1 = time.time()
    for i, (s, g) in enumerate(results, 1):
        srows.append(s)
        grows.extend(g)
        if verbose and (i % 500 == 0 or i == len(tasks)):
            el = time.time() - t1
            rate = i / max(el, 1e-9)
            print(f"{i}/{len(tasks)} series {el:.0f}s ({rate:.1f} series/s)", flush=True)
    if workers > 1:
        pool.close()
        pool.join()
    series_df = pd.DataFrame(srows).sort_values(["split", "study", "series"], ignore_index=True)
    geom_df = pd.DataFrame(grows)
    if len(geom_df):
        geom_df = geom_df.sort_values(["study", "series", "cache_idx"], ignore_index=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        series_df.to_parquet(os.path.join(out_dir, "series_headers.parquet"), index=False)
        geom_df.to_parquet(os.path.join(out_dir, "slice_geometry.parquet"), index=False)
    if verbose:
        err = int(series_df["error"].notna().sum()) if "error" in series_df else 0
        print(
            f"done: {len(series_df)} series, {len(geom_df)} kept slices, "
            f"{int(series_df['n_header_bad'].fillna(0).sum())} unreadable headers in "
            f"{int((series_df['n_header_bad'].fillna(0) > 0).sum())} series, {err} series errors, "
            f"{int((series_df.get('n_kept', 0) == 0).sum())} series with no slices, "
            f"{time.time() - t0:.0f}s total",
            flush=True,
        )
    return series_df, geom_df


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="competition archive")
    src.add_argument("--root", help="directory with {split}_series/{study}/{series}/*")
    ap.add_argument("--out", default="/workspace/v8prep_out/headers/")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N series only (smoke runs)")
    a = ap.parse_args()
    run(a.zip, a.root, a.out, a.splits, a.workers, a.limit)


if __name__ == "__main__":
    main()

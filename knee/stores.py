"""The training stores, cut from the train DICOMs on a Kaggle CPU kernel, one shard of exams per run.

  448      every usable series as 448 px uint8 rows: images.zip, cache_manifest.json, series_metadata.jsonl (tiles.Store448)
  lateral  the lateral crop's series cut once from the file pixels: crops.u8, series_private.jsonl (tiles.NativeStore)
  window   window rows of every chosen series: px.u8.npy, meta.f32.npy, index.json (windows.WindowStore)
Exams: the train exams minus exclusions.json (the 58 expert exams and the exams linked to them) in
sha256("20261004:<id>") order, items --start .. --start+--count of that list; --expert takes the 58 instead.
Reads train_series.csv and the DICOMs; no report, label or test file. Rows are made by the same code as
inference (dicom.raw_study, tiles, windows), so training sees the bytes inference sees.

python -m knee.stores --kind window --start 0 --count 250 --data COMPETITION_DIR --exclusions exclusions.json --out DIR
As a Kaggle script (no arguments) it runs KAGGLE_JOB with the attached competition, the code dataset and the one
exclusions.json found under /kaggle/input (knee.labels folds output).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zipfile

ORDER_SEED = 20261004            # exam order of every store: sha256(f"{ORDER_SEED}:{study}")
KAGGLE_JOB = dict(kind="window", start=0, count=250, expert=False)   # set before pushing the store kernel
KAGGLE_INPUT, KAGGLE_OUT = Path("/kaggle/input"), Path("/kaggle/working")
COMPETITION = "rsna-knee-abnormality-detection"
ZIP_LEVEL = 6                    # images.zip: deflate level 6
ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # fixed member time, so a rebuilt archive has the same bytes
NPY_HEADER_BYTES = 128           # px.u8.npy header, written first as a placeholder and rewritten at the end
TASKS_PER_WORKER = 25            # exams per worker process before it is replaced


def _code_root() -> Path:
    """The folder holding knee/: the attached code dataset when this file runs as a Kaggle script."""
    hits = sorted({p.parents[1] for pat in ("datasets/*/*/knee/stores.py", "*/knee/stores.py")
                   for p in KAGGLE_INPUT.glob(pat)})
    if len(hits) != 1:
        raise FileNotFoundError(f"code dataset: found {len(hits)}")
    return hits[0]


if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(_code_root()))

import numpy as np  # noqa: E402

from knee import read_json, sha256, write_json  # noqa: E402
from knee.series import read_series_table  # noqa: E402
from knee.tiles import NATIVE_BLOB, NATIVE_INDEX, STORE_MEMBER  # noqa: E402

STATE = {}   # the worker processes' view of the job, set before the pool forks


# ---------------------------------------------------------------- which exams

def hash_order(ids) -> list[str]:
    return sorted(ids, key=lambda s: (hashlib.sha256(f"{ORDER_SEED}:{s}".encode()).hexdigest(), s))


def shard_exams(table: dict, exclusions: dict, start: int, count: int, expert: bool) -> list[str]:
    """The shard's exams: the 58 (sorted) with expert, else items start..start+count of the fitting exams."""
    if expert:
        ids = sorted(exclusions["expert_study_ids"])
        if not set(ids) <= set(table):
            raise ValueError("an expert exam has no train_series.csv rows")
        return ids
    skip = set(exclusions["expert_study_ids"]) | set(exclusions["linked_study_ids"])
    fitting = hash_order(s for s in table if s not in skip)
    if start < 0 or count <= 0 or start >= len(fitting):
        raise ValueError(f"start {start} / count {count} outside the {len(fitting)} fitting exams")
    return fitting[start:start + count]


# ---------------------------------------------------------------- one exam per kind (worker processes)

def rows448(study: str) -> list:
    """[(zip member, npy bytes, series record)] of every usable series, in SeriesInstanceUID order, as the 448
    store cut them: header checks, decode, window, then the rows; a series that fails is left out."""
    from knee.dicom import decode, read_series
    from knee.series import candidate
    out = []
    for uid, row in sorted(STATE["table"][study].items()):
        paths = sorted((STATE["root"] / study / uid).glob("*.dcm"))
        if not paths:
            raise ValueError("empty listed series")
        try:
            series = read_series(study, uid, paths)
            candidate(series.record(), row)
            decode(series)
            candidate(series.record(), row)
        except Exception:
            continue
        record = series.record()
        record["source_sha256"] = {str(i): sha256(paths[i]) for i in series.decoded}
        stream = io.BytesIO()
        np.save(stream, np.stack([series.row448(i) for i in series.decoded]), allow_pickle=False)
        out.append((STORE_MEMBER.format(study=study, series=uid), stream.getvalue(), record))
    return out


def lateral_rows(study: str) -> list:
    """[(index entry without its offset, uint8 rows)] of the lateral crop's series (tiles.NativeStore format)."""
    from knee.dicom import raw_study
    from knee.centre import native_box
    from knee.tiles import TileRecipe, raw_exam
    tile = TileRecipe.from_recipe(STATE["recipe"])
    plan, pixels = raw_exam(raw_study(STATE["root"] / study, list(STATE["table"][study].values()), tile.rule), tile)
    out = []
    for value in plan.detail.selection.slots:
        if value is None:
            continue
        rows = np.ascontiguousarray(pixels.native_rows(value, tile.size)[list(range(len(value.sources)))])
        h = value.record["headers"][0]
        cy, cx, side_mm = value.window
        out.append((dict(study=study, series=value.series, slot=f"{value.slot[0]}-{value.slot[1]}",
                         sources=list(value.sources), shape=[int(x) for x in h["shape"]],
                         spacing=[float(x) for x in h["spacing"]][:2], window=list(value.record["window"]),
                         cy=cy, cx=cx, side_mm=side_mm, box=list(native_box(value)), rows=len(rows),
                         sha256=hashlib.sha256(rows.tobytes()).hexdigest()), rows))
    return out


def window_rows(study: str):
    """(px uint8 [N, 320, 320], meta float32 [N, 12], blocks) of windows.study_rows."""
    from knee.dicom import raw_study
    from knee.windows import study_rows
    inputs = STATE["recipe"]["inputs"]
    exam = raw_study(STATE["root"] / study, list(STATE["table"][study].values()), inputs["series_rule"])
    return study_rows(exam, int(inputs["store_size"]))


CUT = {"448": rows448, "lateral": lateral_rows, "window": window_rows}
RECIPE = {"448": None, "lateral": "lateral-effnet.json", "window": "window-convnext.json"}


def one_exam(study: str):
    """Worker: (study, result | None, error type | None)."""
    import torch
    torch.set_num_threads(1)
    try:
        return study, CUT[STATE["kind"]](study), None
    except Exception as error:
        return study, None, type(error).__name__


# ---------------------------------------------------------------- writers (main process, exams in shard order)

def write_448(out: Path, results) -> dict:
    manifest, records, done = {}, [], []
    with zipfile.ZipFile(out / "images.zip", "x", allowZip64=True) as archive:
        for study, series in results:
            for member, data, record in series:
                info = zipfile.ZipInfo(member, date_time=ZIP_TIME)
                info.external_attr = 0o644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL)
                manifest[member] = dict(archive="images.zip", member_sha256=hashlib.sha256(data).hexdigest())
                records.append(record)
            done.append(study)
    with zipfile.ZipFile(out / "images.zip") as archive:   # read every member back against its pin
        for member, entry in manifest.items():
            if hashlib.sha256(archive.read(member)).hexdigest() != entry["member_sha256"]:
                raise ValueError("a stored series does not read back as it was written")
    write_json(out / "cache_manifest.json", dict(files=manifest), indent=2)
    (out / "series_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return dict(done=done, series=len(records))


def write_lateral(out: Path, results) -> dict:
    offset, done, series = 0, [], 0
    with (out / NATIVE_BLOB).open("xb") as blob, (out / NATIVE_INDEX).open("x") as index:
        for study, cut in results:
            for entry, rows in cut:
                blob.write(rows.tobytes())
                index.write(json.dumps(dict(entry, offset=offset)) + "\n")
                offset += rows.nbytes
                series += 1
            done.append(study)
    return dict(done=done, series=series, bytes=offset)


def npy_header(n: int, size: int) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"descr": "|u1", "fortran_order": False, "shape": (n, size, size)})
    if len(stream.getvalue()) != NPY_HEADER_BYTES:
        raise ValueError("unexpected npy header length")
    return stream.getvalue()


def write_window(out: Path, results, order: list[str], errors: dict) -> dict:
    from knee.windows import META_COLUMNS
    size = int(STATE["recipe"]["inputs"]["store_size"])
    index = dict(order=order, done={}, blocks={}, cursor=0, errors=errors, meta_cols=list(META_COLUMNS))
    metas = []
    with (out / "px.u8.npy").open("xb") as px:
        px.write(npy_header(0, size))
        for study, (rows, meta, blocks) in results:
            start = index["cursor"]
            px.write(rows.tobytes())
            metas.append(meta)
            index["done"][study] = [start, len(rows), len(blocks)]
            index["blocks"][study] = [dict(b, views=[[v, start + off] for v, off in b["views"]]) for b in blocks]
            index["cursor"] = start + len(rows)
        px.seek(0)
        px.write(npy_header(index["cursor"], size))
    meta = np.concatenate(metas) if metas else np.zeros((0, len(META_COLUMNS)), np.float32)
    np.save(out / "meta.f32.npy", meta.astype(np.float32))
    (out / "index.json").write_text(json.dumps(index))   # errors is complete once results is used up
    if np.load(out / "px.u8.npy", mmap_mode="r").shape != (index["cursor"], size, size) or len(meta) != index["cursor"]:
        raise ValueError("px or meta does not hold the indexed rows")
    return dict(done=list(index["done"]), series=sum(len(b) for b in index["blocks"].values()), rows=index["cursor"])


# ---------------------------------------------------------------- main

def build(kind: str, data: Path, exclusions: Path, out: Path, start: int, count: int, expert: bool,
          code: Path, workers: int) -> dict:
    """Cut one shard into `out` (created); returns its summary (no exam IDs)."""
    started = time.time()
    table = read_series_table(data / "train_series.csv")
    ids = shard_exams(table, read_json(exclusions), start, count, expert)
    STATE.update(kind=kind, table=table, root=data / "train_series",
                 recipe=read_json(code / "configs" / RECIPE[kind]) if RECIPE[kind] else None)
    out.mkdir(parents=True, exist_ok=False)
    order = sorted(ids) if kind == "448" else ids   # the 448 store went exam by exam in ID order
    failures, errors = {}, {}

    def results():
        with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=TASKS_PER_WORKER) as pool:
            for study, result, error in pool.imap(one_exam, order, chunksize=1):
                if error is None:
                    yield study, result
                else:
                    failures[error] = failures.get(error, 0) + 1
                    errors[study] = error

    if kind == "448":
        got = write_448(out, results())
    elif kind == "lateral":
        got = write_lateral(out, results())
    else:
        got = write_window(out, results(), ids, errors)
    done = got.pop("done")
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    summary = dict(kind=kind, exams=len(ids), done=len(done), failures=failures, expert=expert, start=start,
                   count=count, ids_sha256=hashlib.sha256("\n".join(sorted(done)).encode()).hexdigest(), **got,
                   seconds=round(time.time() - started, 1), files={name: sha256(out / name) for name in files})
    summary["status"] = "passed" if len(done) == len(ids) else "incomplete"
    write_json(out / "summary.json", summary)
    return summary


def one_file(name: str) -> Path:
    hits = sorted(KAGGLE_INPUT.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {len(hits)} under {KAGGLE_INPUT}")
    return hits[0]


def competition_dir() -> Path:
    for d in (KAGGLE_INPUT / "competitions" / COMPETITION, KAGGLE_INPUT / COMPETITION):
        if (d / "train_series.csv").is_file():
            return d
    raise FileNotFoundError("competition data not attached")


def main(argv=None) -> int:
    code = Path(__file__).resolve().parents[1]
    kaggle = not (argv if argv is not None else sys.argv[1:]) and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    if kaggle:
        code = _code_root()
        job = KAGGLE_JOB
        args = argparse.Namespace(kind=job["kind"], start=job["start"], count=job["count"], expert=job["expert"],
                                  data=competition_dir(), exclusions=one_file("exclusions.json"),
                                  out=KAGGLE_OUT / job["kind"], workers=os.cpu_count() or 1)
    else:
        ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        ap.add_argument("--kind", choices=sorted(CUT), required=True)
        ap.add_argument("--start", type=int, default=0, help="first exam of the shard in the fitting-exam order")
        ap.add_argument("--count", type=int, default=250, help="exams in the shard")
        ap.add_argument("--expert", action="store_true", help="the 58 expert exams instead (images only)")
        ap.add_argument("--data", type=Path, required=True, help="folder of train_series.csv and train_series/")
        ap.add_argument("--exclusions", type=Path, required=True, help="exclusions.json (knee.labels folds)")
        ap.add_argument("--out", type=Path, required=True, help="shard folder to create")
        ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="exam worker processes")
        args = ap.parse_args(argv)
    summary = build(args.kind, args.data, args.exclusions, args.out, args.start, args.count, args.expert, code,
                    args.workers)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

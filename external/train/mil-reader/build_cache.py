"""Build the reader's training cache straight from the competition zip, one series at a time.

The archive is too large to unpack next to the cache, so each worker extracts one series into a
temporary directory, runs the reader's own `load_series` on it (byte-identical preprocessing to
Kaggle inference), saves `<series>.npy` and deletes the temporary files.

Output matches `reader/preprocess.py`: `<out>/cache384_train/<series>.npy` and
`<out>/cache384_train_meta.csv`. Re-running skips series already written. A series with no readable
slice is recorded as an error and not written, so a later run tries it again.

Usage: python build_cache.py --zip <archive.zip> --out <work_dir>
       [--workers 14] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import require_decoders  # noqa: E402  (also puts reader/ on the path)
from preprocess import SIZE, load_series  # noqa: E402

_ZIP: zipfile.ZipFile | None = None


def _init(zip_path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def _job(task: tuple[str, str, str, list[str], str]) -> dict[str, object] | None:
    study, series, plane, members, out_dir = task
    out = f"{out_dir}/{series}.npy"
    info: dict[str, object] = {"study": study, "series": series}
    if os.path.exists(
        out
    ):  # already built: keep its row so an interrupted run's meta stays complete
        info["n"] = int(np.load(out, mmap_mode="r").shape[0])
        return info
    tmp = tempfile.mkdtemp(prefix="bb_")
    try:
        assert _ZIP is not None
        for i, name in enumerate(members):
            with _ZIP.open(name) as src, open(f"{tmp}/{i:05d}.dcm", "wb") as dst:
                shutil.copyfileobj(src, dst)
        vol, extra = load_series(tmp, plane)
        if not extra["n_raw"]:  # load_series returns a blank placeholder: keep it out of the cache
            raise ValueError(f"no readable slice ({extra['n_bad']} files failed)")
        np.save(out + ".tmp.npy", vol)
        os.replace(out + ".tmp.npy", out)
        info.update(extra)
    except Exception as e:  # recorded per series, as in the reference preprocess.py
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return info


def series_members(zf: zipfile.ZipFile, split: str) -> dict[str, list[str]]:
    prefix = f"{split}_series/"
    members: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if name.startswith(prefix) and not name.endswith("/"):
            parts = name[len(prefix) :].split("/")
            if len(parts) >= 3:
                members[parts[1]].append(name)
    return members


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True, help="work dir; the cache goes to <out>/cache384_train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    require_decoders()

    out_dir = f"{args.out}/cache{SIZE}_{args.split}"
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        with zf.open(f"{args.split}_series.csv") as f:
            se = pd.read_csv(f)
        members = series_members(zf, args.split)
    if args.limit:
        se = se.sample(args.limit, random_state=0)
    tasks = [
        (a, b, p, members.get(b, []), out_dir)
        for a, b, p in zip(
            se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane, strict=True
        )
    ]
    print(f"{len(tasks)} series, {sum(len(t[3]) for t in tasks)} files", flush=True)

    rows, t0 = [], time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.zip,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, tasks, chunksize=4)):
            if r:
                rows.append(r)
            if i % 500 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    meta = f"{args.out}/cache{SIZE}_{args.split}_meta.csv"
    if os.path.exists(meta) and len(df):
        df = pd.concat([pd.read_csv(meta), df]).drop_duplicates("series", keep="last")
    if len(df):
        df.to_csv(meta, index=False)
        print(
            "done",
            len(df),
            "errors:",
            int(df["error"].notna().sum()) if "error" in df else 0,
            flush=True,
        )


if __name__ == "__main__":
    main()

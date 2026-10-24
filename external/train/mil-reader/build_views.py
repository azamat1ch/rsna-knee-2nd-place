"""V3: close-up boxes (views.py) for every series of the competition archive, header-only.

Writes CSV rows (study, series, plane, view, r0, c0) plus one 'anchor' row per study with the joint centre.
Usage: python build_views.py --zip <archive.zip> --laterality <laterality_v2.csv> --out views.csv
"""
import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reader"))
import geo  # noqa: E402
import views  # noqa: E402

_Z = None


def _init(p):
    global _Z
    _Z = zipfile.ZipFile(p)


def job(task):
    study, series, side = task
    geoms = {}
    for sr, members in series.items():
        hs = []
        for m in members:
            try:
                hs.append(pydicom.dcmread(io.BytesIO(_Z.read(m)), stop_before_pixels=True))
            except Exception:
                pass
        geoms[sr] = geo.series_geometry(hs)
    X = views.slab_prior(geoms)
    rows = []
    if X is None:
        return rows
    rows.append((study, "", "", "anchor", round(float(X[0]), 2), round(float(X[1]), 2), round(float(X[2]), 2)))
    for sr, (plane, affs) in geoms.items():
        if not affs:
            continue
        for v, (r0, c0) in views.crop_boxes(plane, affs, X, side).items():
            rows.append((study, sr, plane, v, r0, c0, ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--laterality", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    side = {}
    for r in csv.DictReader(open(a.laterality)):
        side[r["study"]] = r["side"]
    mem = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(a.zip) as z:
        for n in z.namelist():
            if n.startswith(f"{a.split}_series/") and n.endswith(".dcm"):
                _, st, sr, _f = n.split("/")
                mem[st][sr].append(n)
    tasks = [(st, dict(s), side.get(st, "R")) for st, s in mem.items()]
    with Pool(a.workers, initializer=_init, initargs=(a.zip,)) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "series", "plane", "view", "r0", "c0", "z"])
        for i, rows in enumerate(p.imap_unordered(job, tasks, chunksize=4)):
            w.writerows(rows)
            if i % 500 == 0:
                print(i, "studies", flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()

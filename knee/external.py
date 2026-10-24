"""Drivers for Matthew Liew's two members: mil-reader (all findings) and pf-specialist (PF OA).

His code is in `external/` (unchanged, his own folder names) and his weights in our models dataset under
`mil-reader/` and `pf-specialist/`. Every file is checked against the sha256 pins in `configs/blend.json`, his entry
point runs with our PYTHONPATH removed (his modules must win), and his output is written as a member CSV
(StudyInstanceUID + 12 findings, sample order).

    python -m knee.external mil-reader --data COMPETITION_DIR --out CSV --work DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knee import LABELS, sha256
from knee.blend import ID, frame, write_csv

INPUT = Path("/kaggle/input")
CODE = Path(__file__).resolve().parents[1]
CONFIG = CODE / "configs" / "blend.json"
EXTERNAL = CODE / "external"


def dataset_dir(slug: str, root: Path = INPUT) -> Path:
    """Mount folder of a Kaggle dataset `owner/name` (new or old mount layout)."""
    owner, name = slug.split("/")
    for d in (root / "datasets" / owner / name, root / name):
        if d.is_dir():
            return d
    raise FileNotFoundError(f"dataset {slug} is not attached")


def check_files(folder: Path, pins: dict[str, str], what: str) -> None:
    for name, want in pins.items():
        if not (folder / name).is_file() or sha256(folder / name) != want:
            raise RuntimeError(f"{what}: {name} is missing or differs from its pin")


def his_env(src: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return dict(env, PYTHONPATH=str(src), **extra)


def verify(member: dict, models: Path) -> None:
    """Check his pinned code and weights before any GPU work."""
    ext = member["external"]
    check_files(EXTERNAL, ext["code"], member["name"])
    check_files(models, {**ext["models"], **ext["side_model"]}, member["name"])


def stage(ext: dict, models: Path, src: Path, flat: str | None = None) -> Path:
    """Copy his code (or one folder of it, flattened) and the side model into a writable folder."""
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    for rel in ext["code"]:
        dst = src / (Path(rel).relative_to(flat) if flat else rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXTERNAL / rel, dst)
    (side_rel, side_sha), = ext["side_model"].items()
    shutil.copyfile(models / side_rel, src / "side_model.pt")
    check_files(src, {"side_model.pt": side_sha}, "side model")
    return src


def sample_ids(data: Path) -> list[str]:
    sample = pd.read_csv(data / "sample_submission.csv", dtype={ID: str})
    assert list(sample.columns) == [ID, *LABELS], "sample_submission columns changed"
    return sample[ID].tolist()


def mil_reader(member: dict, data: Path, work: Path, models: Path) -> pd.DataFrame:
    ext = member["external"]
    verify(member, models)
    src = stage(ext, models, work / "c_src", flat="reader")
    preds, record = work / "preds.csv", work / "infer_runtime_record.json"
    ckpts = [models / n for n in sorted(ext["models"])]
    env = his_env(src, RADIANT_INFER_DTYPE=ext["dtype"], RADIANT_RUNTIME_RECORD=str(record))
    cmd = [sys.executable, str(src / Path(ext["entry"]).name), str(data), "test", str(preds),
           str(src / "side_model.pt"), *map(str, ckpts)]
    subprocess.run(cmd, check=True, cwd=src, env=env)
    rec = json.loads(record.read_text())
    assert {c["scoring_dtype"] for c in rec["checkpoints"]} == {ext["dtype"]} == {rec["infer_dtype_env"]}, rec
    assert {c["file"]: c["sha256"] for c in rec["checkpoints"]} == {Path(k).name: v for k, v in ext["models"].items()}
    ids = sample_ids(data)
    pred = pd.read_csv(preds, dtype={ID: str}).set_index(ID)
    x = pred.loc[ids, list(LABELS)].to_numpy(np.float64)
    assert len(x) == len(ids) and np.isfinite(x).all(), "mil-reader output incomplete"
    return frame(ids, x)


def pf_specialist(member: dict, data: Path, work: Path, models: Path) -> pd.DataFrame:
    ext = member["external"]
    verify(member, models)
    src = stage(ext, models, work / "v731_src")
    raw = work / "specialist_raw.csv"
    cmd = [sys.executable, str(src / ext["entry"]), "predict", str(data), "test", str(raw),
           str(src / "side_model.pt"), *[str(models / n) for n in sorted(ext["models"])],
           "--native-root", str(work / "native_test"), "--drop-native",
           "--record", str(work / "specialist_record.json"), "--workers", str(ext["workers"]), "--dtype", ext["dtype"]]
    subprocess.run(cmd, check=True, cwd=src, env=his_env(src))
    spec = importlib.util.spec_from_file_location("pf_specialist_round2", src / "v731" / "round2.py")
    round2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(round2)
    pf = round2.read_specialist(raw.read_bytes())
    ids = sample_ids(data)
    assert set(pf) == set(ids), "pf-specialist output misses exams"
    x = np.full((len(ids), len(LABELS)), 0.5)
    x[:, LABELS.index("PF OA")] = [pf[i] for i in ids]
    assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all(), "pf-specialist output outside [0, 1]"
    return frame(ids, x)


DRIVERS = {"mil-reader": mil_reader, "pf-specialist": pf_specialist}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("member", choices=sorted(DRIVERS))
    ap.add_argument("--data", type=Path, required=True, help="competition folder (test_series.csv, test_series/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--inputs", type=Path, default=INPUT, help="where the datasets are mounted")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    member = next(m for m in cfg["members"] if m["name"] == a.member)
    a.work.mkdir(parents=True, exist_ok=True)
    models = dataset_dir(cfg["models_dataset"], a.inputs)
    write_csv(DRIVERS[a.member](member, a.data.resolve(), a.work.resolve(), models), a.out)


if __name__ == "__main__":
    main()

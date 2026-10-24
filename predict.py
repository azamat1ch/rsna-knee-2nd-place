"""RSNA Knee, 2nd place: the inference kernel (Kaggle script, T4 x2, internet off).

Runs the members of configs/blend.json one at a time, each in its own process group, then rank-blends them into
/kaggle/working/submission.csv. The four core members are all or nothing and their blend is written first. The two
specialists then run in what is left of the time limit; if one fails, the file keeps the core blend.
Member CSVs and receipt.json go to /kaggle/working/blend/.
"""
import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.time()
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
OUT = WORKING / "blend"
SUBMISSION = WORKING / "submission.csv"


def find_code() -> Path:
    """The folder holding knee/ and configs/: next to this file, else the attached code dataset."""
    here = Path(globals().get("__file__", "")).resolve().parent
    if (here / "knee" / "blend.py").is_file():
        return here
    hits = sorted({p.parents[1] for pat in ("datasets/*/*/knee/blend.py", "*/knee/blend.py") for p in INPUT.glob(pat)})
    assert len(hits) == 1, f"code dataset: found {len(hits)}"
    return hits[0]


CODE = find_code()
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from knee import LABELS, sha256  # noqa: E402
from knee import external  # noqa: E402
from knee.blend import ID, blend, frame, read_member, write_csv  # noqa: E402
from knee.score import validate  # noqa: E402


def install_decoders(pins: dict[str, str]) -> None:
    """Offline JPEG Lossless / JPEG 2000 plugins for pydicom from the pinned wheels, then a decode self-test."""
    wheels = [CODE / "wheels" / name for name in pins]
    for w in wheels:
        assert sha256(w) == pins[w.name], f"wheel {w.name} differs from its pin"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet", *map(str, wheels)],
                   check=True)
    importlib.invalidate_caches()
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.pixels import get_decoder, pixel_array
    from pydicom.uid import JPEG2000, ExplicitVRLittleEndian, JPEG2000Lossless, JPEGLossless, JPEGLosslessSV1
    syntaxes = (JPEGLossless, JPEGLosslessSV1, JPEG2000Lossless, JPEG2000)
    missing = [u.name for u in syntaxes if not get_decoder(u).is_available]
    assert not missing, f"no decoder for {missing}"
    px = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 37) % 4096  # synthetic 12-bit image
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.update(dict(Rows=64, Columns=64, SamplesPerPixel=1, PhotometricInterpretation="MONOCHROME2", BitsAllocated=16,
                   BitsStored=12, HighBit=11, PixelRepresentation=0))
    ds.compress(JPEG2000Lossless, px)
    for plugin in get_decoder(JPEG2000Lossless).available_plugins:
        assert (pixel_array(ds, decoding_plugin=plugin) == px).all(), f"JPEG 2000 round trip failed ({plugin})"


def competition_dir(root: Path, slug: str) -> Path:
    for d in (root, root / "competitions" / slug, root / slug):
        if (d / "test_series.csv").is_file() and (d / "sample_submission.csv").is_file():
            return d
    raise FileNotFoundError(f"no competition data under {root}")


def run_group(cmd: list[str], budget: float, cwd: Path) -> int | str:
    """Run cmd in a new session; at most `budget` s; then kill its whole process group (frees the GPUs)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(CODE), os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        rc = "timeout"
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if rc == "timeout":
        proc.wait()
    return rc


class Members:
    """Runs members on one competition folder, each within its time budget, and records them in the receipt."""

    def __init__(self, cfg: dict, comp: Path, ids: list[str], work_root: Path, receipt: dict):
        self.cfg, self.time, self.comp, self.ids = cfg, cfg["time"], comp, ids
        self.work_root, self.receipt = work_root, receipt
        self.models = external.dataset_dir(cfg["models_dataset"])

    def remaining(self) -> float:
        return self.time["limit_s"] - self.time["reserve_s"] - (time.time() - STARTED)

    def run(self, m: dict, budget: float) -> np.ndarray:
        name, work, out = m["name"], self.work_root / m["name"], OUT / f"{m['name']}.csv"
        if budget < self.time["min_budget_s"]:
            raise RuntimeError(f"{name}: only {budget:.0f} s left")
        work.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        common = ["--data", str(self.comp), "--out", str(out)]
        if "recipe" in m:
            cmd = ["knee.infer", "--recipe", str(CODE / m["recipe"]), "--weights", str(self.models), *common,
                   "--deadline", str(int(budget - self.time["deadline_margin_s"]))]
        else:
            cmd = ["knee.external", name, *common, "--work", str(work)]
        print(f"[predict] {name}: budget {budget:.0f} s", flush=True)
        start = time.time()
        rc = run_group([sys.executable, "-m", *cmd], budget, work)
        self.receipt["members"][name] = {"budget_s": round(budget), "seconds": round(time.time() - start, 1), "rc": rc}
        print(f"[predict] {name}: rc={rc} in {time.time() - start:.0f} s", flush=True)
        if rc != 0:
            raise RuntimeError(f"{name} failed (rc={rc})")
        return read_member(out, self.ids)

    def write(self, preds: dict[str, np.ndarray]) -> None:
        sub = frame(self.ids, blend(preds, self.cfg))
        validate(sub, self.ids)
        write_csv(sub, SUBMISSION)


def predict(cfg: dict, data_root: Path, chosen: list[str], work_root: Path, receipt: dict) -> None:
    install_decoders(cfg["wheels"])
    comp = competition_dir(data_root, cfg["competition"])
    sample = pd.read_csv(comp / "sample_submission.csv", dtype={ID: str})
    assert list(sample.columns) == [ID, *LABELS], "sample_submission columns changed"
    run = Members(cfg, comp, sample[ID].tolist(), work_root, receipt)
    core = [m for m in cfg["members"] if "weight" in m and m["name"] in chosen]
    for m in core:
        if "external" in m:
            external.verify(m, run.models)  # wrong external files fail the run before any GPU work
    preds = {m["name"]: run.run(m, run.remaining()) for m in core}  # all or nothing
    if len(preds) >= 2:
        run.write(preds)
        shutil.copy(SUBMISSION, OUT / "core-submission.csv")
        receipt["status"] = "core written"
    for m in (m for m in cfg["members"] if "findings" in m and m["name"] in chosen):
        try:
            preds[m["name"]] = run.run(m, min(m["cap_s"], run.remaining()))
            receipt["specialists"][m["name"]] = "used"
        except Exception as error:
            receipt["specialists"][m["name"]] = f"skipped: {type(error).__name__}: {str(error)[:300]}"
    if receipt["status"] == "core written" and "used" in receipt["specialists"].values():
        try:
            run.write(preds)
            receipt["status"] = "completed"
        except Exception as error:  # the core file stays in place
            receipt["status"] = f"core kept: {type(error).__name__}: {str(error)[:300]}"
    if receipt["status"] == "running":
        receipt["status"] = "members only (fewer than 2 core members: no blend)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=INPUT, help="competition folder, or where it is mounted")
    ap.add_argument("--members", default="", help="comma-separated subset of members (default: all)")
    ap.add_argument("--keep-members", action="store_true", help="keep member work folders under blend/work")
    args = ap.parse_args()
    cfg = json.loads((CODE / "configs" / "blend.json").read_text())
    chosen = args.members.split(",") if args.members else [m["name"] for m in cfg["members"]]
    assert set(chosen) <= {m["name"] for m in cfg["members"]}, chosen
    receipt = {"status": "running", "members": {}, "specialists": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for p in (SUBMISSION, SUBMISSION.with_name(SUBMISSION.name + ".part")):
            p.unlink(missing_ok=True)
        predict(cfg, args.data_root, chosen, OUT / "work" if args.keep_members else Path("/tmp/members"), receipt)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {str(error)[:1000]}")
        SUBMISSION.unlink(missing_ok=True)  # never leave a file from a failed run
        raise
    finally:
        receipt["elapsed_s"] = round(time.time() - STARTED, 1)
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
        print(json.dumps(receipt, default=str), flush=True)


if __name__ == "__main__":
    main()

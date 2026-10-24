"""V7.3.1 Kaggle-side inference for the PF specialist, and its private bundle builder (no upload).

``predict``: from the native DICOM test series (``<data>/<split>_series/<study>/<series>/*.dcm``)
build, per study, exactly what training saw and score the specialist checkpoints:

* W   = V7.1's inference path (infer_c): study table from ``<split>_series.csv``
        (``knee.build_study_table``, slot candidates), ``preprocess.load_series`` of each slot's
        first
        candidate, side from ``infer_c.study_side`` (tags / position / side classifier).
* anchor X and kneecap boxes = build_views.py's rule: ``geo.series_geometry`` of every file header
of
        every series of the study, ``views.slab_prior`` -> X, ``views.crop_boxes(...)['kneecap']``.
* K   = ``kseries.select_k`` on ``v8prep/headers._job`` rows (the code that wrote
        series_headers.parquet, run here in directory mode).
* P/Z/K arrays = ``prepare_native.build_series`` UNCHANGED (same band/slab/box/window code as the
        training arrays): each series' files are packed, in sorted file-name order, into an
        in-memory stored ZIP so build_series reads them through its own ``read_member``.
* item = ``dataset.PFDataset`` (eval mode, k = checkpoint k_eval, left knees mirrored), model =
        ``model.build_model(cfg)`` + the checkpoint's final EMA weights. PF probability =
        sigmoid(pf_combined) for a grade-head (E3) model, else sigmoid(pf_direct); seeds'
        probabilities
        are averaged.

Known non-parity with training (recorded, not hidden): the side comes from the V7.1 inference rule,
whereas the training arrays/boxes used laterality_v2.csv; the W table counts files (capped at 64)
as infer_c does, training used the cache's slice count. Both are V7.1's own train/inference
conventions.

``bundle``: stage the private Kaggle dataset folders + kernel like build_v81_kaggle.py
(ds_v731_models/, ds_v731_assets/, kernel/, MANIFEST.json) and PRINT the kaggle commands. Nothing
is uploaded or pushed.

Usage:
  python kaggle_infer.py predict <data_dir> <split> <out_csv> <side_model.pt> <ckpt> [<ckpt> ...]
  python kaggle_infer.py bundle --ckpts c1.pt c2.pt --side-model side_model.pt --out DIR
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent  # .../branch_b/v731
BRANCH = HERE.parent
for _p in (BRANCH / "reader", BRANCH, BRANCH / "v8prep", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_CANON_DIR = (
    BRANCH.parents[1] / "radiant/adapters/mri"
)  # canonical.py in the repo; bundled into reader/
if _CANON_DIR.is_dir() and str(_CANON_DIR) not in sys.path:
    sys.path.append(str(_CANON_DIR))

ID = "StudyInstanceUID"
PF = "PF OA"
SLOTS_PZ = (0, 2, 3, 5)
MODELS_ID = "matthewliew/radiant-kg-c-v731-models"
ASSETS_ID = "matthewliew/radiant-kg-c-v731-assets"
KERNEL_ID = "matthewliew/radiant-kg-c-v731-specialist"
COMPETITION = "rsna-knee-abnormality-detection"


class InferError(RuntimeError):
    pass


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- DICOM access
def series_files(root: str, st: str, sr: str) -> list[str]:
    return sorted(glob.glob(f"{root}/{st}/{sr}/*.dcm"))


def mem_zip(files: list[str]) -> tuple[io.BytesIO, list[tuple]]:
    """Pack files (in the given order) into an in-memory STORED zip; return (fh, members) in
    prepare_native.index_archive's member format, so build_series reads them unchanged."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for i, f in enumerate(files):
            z.write(f, arcname=f"s/{i:05d}_{os.path.basename(f)}")
    buf.seek(0)
    with zipfile.ZipFile(buf) as z:
        members = [
            (i.filename, i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type)
            for i in z.infolist()
        ]
    buf.seek(0)
    return buf, members


def header_rows(
    data_dir: str, split: str, st: str, series_ids: list[str], csv_plane: dict[str, str]
) -> list[dict[str, Any]]:
    """series_headers.parquet rows for these series, from v8prep/headers._job in directory mode."""
    import headers as H

    H._init(None, data_dir)
    rows = []
    for sr in series_ids:
        members = [
            f"{split}_series/{st}/{sr}/{os.path.basename(f)}"
            for f in series_files(f"{data_dir}/{split}_series", st, sr)
        ]
        srow, _ = H._job_safe((split, st, sr, members, csv_plane.get(sr)))
        rows.append(srow)
    return rows


def k_choice(rows: list[dict[str, Any]]) -> tuple[str, str] | None:
    """(series, plane) of the study's K series via kseries.select_k, or None."""
    import kseries
    import pandas as pd

    cols = [
        "study",
        "series",
        "SeriesDescription",
        "SequenceName",
        "MRAcquisitionType",
        "plane",
        "slice_gap",
        "n_frames_total",
    ]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows if "error" not in r], columns=cols)
    df = df[df.n_frames_total.fillna(0).astype(int) > 0]
    if df.empty:
        return None
    k = kseries.select_k(df)
    return (str(k.series.iloc[0]), str(k.plane.iloc[0])) if len(k) else None


def anchor_and_boxes(headers: dict[str, list], side: str):
    """build_views.job's rule from in-memory headers -> (X or None, {series: (plane, (r0, c0))}).
    Boxes use the unrounded X (as build_views); the band anchor is X rounded like views_train.csv
    (see stored_anchor)."""
    import geo
    import views

    geoms = {sr: geo.series_geometry(hs) for sr, hs in headers.items()}
    X = views.slab_prior(geoms)
    boxes = {}
    if X is None:
        return None, boxes
    for sr, (plane, affs) in geoms.items():
        if affs and plane in ("Sagittal", "Axial"):
            boxes[sr] = (plane, views.crop_boxes(plane, affs, X, side)["kneecap"])
    return X, boxes


# ----------------------------------------------------------------------------- per study prep
def pz_box(views_box):
    """P/Z crop top-left from the views (build_views) 100 mm kneecap box: the SAME shared
    transform prepare_native.load_inputs applies to views_train.csv (bands.box120: same intended
    centre, clamped inside the frame). K boxes get it inside prepare_native.build_series."""
    import bands

    return bands.box120(*views_box)


def stored_anchor(X):
    """X as training read it back from views_train.csv: build_views writes round(float(v), 2)."""
    import numpy as np

    return None if X is None else np.array([round(float(v), 2) for v in X], float)


def prepare_study(
    data_dir: str,
    split: str,
    st: str,
    table: dict,
    csv_plane: dict[str, str],
    native_root: str,
    side_net=None,
    side: str | None = None,
) -> dict[str, Any]:
    """Everything one study needs: W volumes, side, X, boxes, K choice, and the native P/Z/K arrays
    written under native_root by prepare_native.build_series. ``side`` overrides the rule
    (tests)."""
    import infer_c
    import knee
    import prepare_native as PN
    from preprocess import load_series

    root = f"{data_dir}/{split}_series"
    all_series = sorted(d for d in os.listdir(f"{root}/{st}") if os.path.isdir(f"{root}/{st}/{d}"))
    slots = table.get(st, [[] for _ in range(knee.N_SLOTS)])
    vols, chosen = {}, [[] for _ in range(knee.N_SLOTS)]
    for s, cands in enumerate(slots):  # infer_c.DicomStudyDataset: first candidate per slot
        if cands:
            try:
                vols[cands[0]] = load_series(f"{root}/{st}/{cands[0]}")[0]
                chosen[s] = [cands[0]]
            except Exception:
                pass
    hdrs = infer_c.read_headers(root, st, all_series, True)
    if side is None:
        cor_ids = chosen[1] or chosen[4]
        side = infer_c.study_side(hdrs, side_net, vols[cor_ids[0]] if cor_ids else None)
    X, boxes = anchor_and_boxes(hdrs, side)
    kc = k_choice(header_rows(data_dir, split, st, all_series, csv_plane))
    # P/Z for the series the eval sample picks (first candidate of slots 0, 2, 3, 5); K series
    kinds: dict[str, set[str]] = {}
    slot_of: dict[str, list[int]] = {}
    for s in SLOTS_PZ:
        if chosen[s]:
            kinds.setdefault(chosen[s][0], set()).update({"P", "Z"})
            slot_of.setdefault(chosen[s][0], []).append(s)
    if kc is not None:
        kinds.setdefault(kc[0], set()).add("K")
    manifest = []
    for sr, ks in kinds.items():
        fh, members = mem_zip(series_files(root, st, sr))
        task = {
            "study": st,
            "series": sr,
            "kinds": set(ks),
            "slots": ";".join(str(x) for x in slot_of.get(sr, [])),
            "members": members,
            "anchor": stored_anchor(X),
            "box": pz_box(boxes[sr][1]) if sr in boxes else None,
            "side": side,
            "csv_plane": csv_plane.get(sr),
        }
        try:
            manifest += PN.build_series(task, fh=fh, out=Path(native_root))
        except Exception as e:  # recorded, the group is then masked as missing (never substituted)
            manifest += [
                {
                    "study": st,
                    "series": sr,
                    "kind": k,
                    "status": "decode_error",
                    "note": f"{type(e).__name__}: {e}"[:200],
                }
                for k in sorted(ks)
            ]
    return {
        "study": st,
        "vols": vols,
        "chosen": chosen,
        "side": side,
        "anchor": stored_anchor(X),
        "boxes_views": {k: v[1] for k, v in boxes.items()},  # build_views' 100 mm kneecap box
        "boxes": {k: pz_box(v[1]) for k, v in boxes.items()},  # the box P/Z are cut from
        "k_series": kc,
        "manifest": manifest,
    }


def usable_keys(rows: list[dict[str, Any]], native_root: str) -> set[tuple[str, str]]:
    """{(kind, series)} the item may use: the data worker's single rule, manifest.usable_reason
    (status ok, current bands version, at least MIN_BAND_SLABS slabs (thinner bands are a terminal
    absence, masked), array present with the recorded shape), as for the training arrays."""
    import manifest as M

    return {
        (str(r["kind"]), str(r["series"]))
        for r in rows
        if M.usable_reason(r, native_root, check_sha=False) == ""
    }


def make_item(prep: dict[str, Any], native_root: str, groups, k_eval: int) -> dict[str, Any]:
    """dataset.PFDataset item (eval mode) for one prepared study."""
    from v731.dataset import NativeStore, PFDataset

    st = prep["study"]
    store = NativeStore(native_root, ok=usable_keys(prep["manifest"], native_root))
    table = {st: prep["chosen"]}
    ds = PFDataset(
        [st],
        table,
        None,
        store,
        None,
        k=k_eval,
        train=False,
        mirror={st} if prep["side"] == "L" else None,
        k_series={st: prep["k_series"]} if prep["k_series"] else None,
        groups=groups,
        w_loader=lambda sr: prep["vols"][sr],
    )
    return ds[0]


# ----------------------------------------------------------------------------- models
def load_checkpoint(path: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """(model in eval mode, cfg, raw checkpoint) from train.py's final_ema.pt: {"model": final EMA
    state dict, "config": cfg, "identity", "groups", ...}. A rolling training state (has "opt") is
    refused: only the final EMA checkpoint is ever scored."""
    import torch

    from v731 import model as M

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "opt" in ck:
        raise InferError(f"{path}: a training state, not train.py's final_ema.pt")
    cfg, sd = ck.get("config"), ck.get("model")
    if cfg is None or sd is None:
        raise InferError(f"{path}: checkpoint lacks config / model (keys {sorted(ck)})")
    if "groups" in ck and [int(g) for g in ck["groups"]] != [int(g) for g in cfg["groups"]]:
        raise InferError(
            f"{path}: checkpoint groups {ck['groups']} != config groups {cfg['groups']}"
        )
    m = M.build_model(cfg)
    m.load_state_dict(sd, strict=True)
    return m.eval(), cfg, ck


def pf_prob(out: dict[str, Any]):
    """sigmoid(pf_combined) when the model has the grade head (E3), else sigmoid(pf_direct)."""
    logit = out["pf_combined"] if "pf_combined" in out else out["pf_direct"]
    return logit.float().sigmoid()


def predict_items(
    models: list, items: list[dict[str, Any]], device: str = "cpu", dtype: str = "float32"
):
    """Mean over models of each item's PF probability -> tensor [len(items)]."""
    import torch

    from v731.dataset import collate, to_device

    batch = to_device(collate(items), device)
    probs = []
    with torch.no_grad():
        for m in models:
            with torch.autocast(
                "cuda",
                dtype=getattr(torch, dtype),
                enabled=device.startswith("cuda") and dtype != "float32",
            ):
                probs.append(pf_prob(m(batch, train=False)))
    return torch.stack(probs).mean(0).cpu()


def model_meta(cfg: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    groups = tuple(int(g) for g in cfg["groups"])
    k_eval = int(cfg.get("recipe", {}).get("k_eval", cfg.get("k_eval", 12)))
    return groups, k_eval


def drop_study_arrays(rows: list[dict[str, Any]], native_root: str) -> int:
    """Delete one study's <native_root>/<kind>/<series>.npy after its item is built, so disk use
    stays flat over the hidden test (~33 MB per study; ~1,300 studies would exceed Kaggle's disk).
    Returns the number of files removed."""
    n = 0
    for r in rows:
        p = Path(native_root) / str(r.get("kind")) / f"{r.get('series')}.npy"
        if p.is_file():
            p.unlink()
            n += 1
    return n


class StudyPrepDataset:
    """torch Dataset: one prepared item per study (prep is CPU work, run in DataLoader workers)."""

    def __init__(
        self,
        data_dir,
        split,
        studies,
        table,
        csv_plane,
        native_root,
        side_model,
        groups,
        k_eval,
        drop_native=False,
    ):
        self.a = (data_dir, split, table, csv_plane, native_root)
        self.studies, self.side_model, self.side_net = list(studies), side_model, None
        self.groups, self.k_eval, self.drop_native = groups, k_eval, drop_native

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        import timm
        import torch

        if self.side_net is None and self.side_model:
            self.side_net = timm.create_model("resnet18", pretrained=False, num_classes=2)
            self.side_net.load_state_dict(torch.load(self.side_model, map_location="cpu"))
            self.side_net.eval()
        data_dir, split, table, csv_plane, native_root = self.a
        prep = prepare_study(
            data_dir, split, self.studies[i], table, csv_plane, native_root, self.side_net
        )
        item = make_item(prep, native_root, self.groups, self.k_eval)
        if self.drop_native:  # arrays are fully in memory now (NativeStore.get: np.load, no mmap)
            drop_study_arrays(prep["manifest"], native_root)
        item["_record"] = {
            "study": prep["study"],
            "side": prep["side"],
            "k_series": prep["k_series"],
            "manifest": [
                {
                    k: r.get(k)
                    for k in ("series", "kind", "status", "n_slab", "sha256", "bands_version")
                }
                for r in prep["manifest"]
            ],
        }
        return item


def run(
    data_dir: str,
    split: str,
    side_model: str,
    ckpts: list[str],
    native_root: str,
    studies: list[str] | None = None,
    bs: int = 2,
    workers: int = 4,
    dtype: str = "float16",
    drop_native: bool = False,
):
    """-> (DataFrame StudyInstanceUID, PF OA; per-study records). ``drop_native``: delete each
    study's arrays once its item is built (flat disk use on the hidden test)."""
    import knee
    import pandas as pd
    import torch

    se = pd.read_csv(f"{data_dir}/{split}_series.csv", dtype={ID: str, "SeriesInstanceUID": str})
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv", dtype={ID: str})[ID].tolist()
    se = se[se[ID].isin(set(studies))]
    n_files = {}
    for st, sr in zip(se[ID], se.SeriesInstanceUID, strict=True):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), 64) if os.path.isdir(d) else 0  # infer_c (V7.1)
    table = knee.build_study_table(se, n_files)
    csv_plane = dict(zip(se.SeriesInstanceUID, se.Anatomical_Plane, strict=True))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = [load_checkpoint(p) for p in ckpts]
    metas = {model_meta(cfg) for _, cfg, _ in loaded}
    variants = {str(cfg.get("variant")) for _, cfg, _ in loaded}
    if len(metas) != 1 or len(variants) != 1:
        raise InferError(f"checkpoints disagree on variant / groups / k_eval: {variants} {metas}")
    groups, k_eval = metas.pop()
    models = [m.to(dev) for m, _, _ in loaded]
    ds = StudyPrepDataset(
        data_dir,
        split,
        studies,
        table,
        csv_plane,
        native_root,
        side_model,
        groups,
        k_eval,
        drop_native=drop_native,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=list)
    probs, records = [], []
    for items in dl:
        records += [it.pop("_record") for it in items]
        probs.append(predict_items(models, items, dev, dtype))
    p = torch.cat(probs).numpy() if probs else []
    df = pd.DataFrame({ID: [r["study"] for r in records], PF: p})
    return df, records


# ----------------------------------------------------------------------------- bundle
ASSET_ROOTS = ("v731/kaggle_infer.py", "v731/round2.py")  # what the kernels run
# planning-only modules reached through lazy imports (round2.retrain_plan -> launch): never run on
# Kaggle, so the walk stops there (keeps OAI preparation code out of the inference package)
ASSET_STOP = ("v731/launch.py",)
# kaggle_infer's sys.path order (last insert first): v731/, v8prep/, branch_b/ (packages), reader/
SEARCH = ("v731", "v8prep", "", "reader")


def _module_names(path: Path) -> set[str]:
    """Dotted module names imported anywhere in the file (top level and inside functions)."""
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)  # from pkg import module
    return names


def _resolve(name: str, branch: Path) -> list[str]:
    """branch_b-relative files a dotted import resolves to (empty for third-party / stdlib)."""
    parts = name.split(".")
    for base in SEARCH:
        root = branch / base if base else branch
        if len(parts) == 1 and (root / f"{parts[0]}.py").is_file():
            return [str(Path(base) / f"{parts[0]}.py") if base else f"{parts[0]}.py"]
        pkg = root / parts[0]
        if (pkg / "__init__.py").is_file():
            out = [
                str(Path(base) / parts[0] / "__init__.py") if base else f"{parts[0]}/__init__.py"
            ]
            if len(parts) > 1 and (pkg / f"{parts[1]}.py").is_file():
                out.append(
                    str(Path(base) / parts[0] / f"{parts[1]}.py")
                    if base
                    else f"{parts[0]}/{parts[1]}.py"
                )
            return out
    return []


def asset_sources(branch: Path = BRANCH, roots: tuple[str, ...] = ASSET_ROOTS) -> dict[str, str]:
    """{published path: branch_b-relative source}: the kernels' entry modules plus every local
    module they import, transitively (derived from the code, so a new import is staged too).
    ``canonical`` (radiant/adapters/mri) is added separately as reader/canonical.py."""
    todo, seen = list(roots), set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for name in _module_names(branch / rel):
            todo += [r for r in _resolve(name, branch) if r not in seen and r not in ASSET_STOP]
    return {r: r for r in sorted(seen)}


CANONICAL = BRANCH.parents[1] / "radiant/adapters/mri/canonical.py"

NOTEBOOK = """# radiant-kg V7.3.1 PF specialist (private kernel; staged by kaggle_infer.py bundle).
# Writes /kaggle/working/specialist_pf.csv (StudyInstanceUID, PF OA) in sample_submission order and
# runtime_receipt.json. It never submits; the round-2 candidate combines this with the team blend.
import glob, hashlib, json, os, shutil, subprocess, sys, time
import pandas as pd
INPUT, WORKING = "/kaggle/input", "/kaggle/working"
RUN_NONCE = "__V731_RUN_NONCE__"
def find(pattern):
    for pat in [f"{INPUT}/*/", f"{INPUT}/*/*/", f"{INPUT}/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if glob.glob(os.path.join(d, pattern)):
                return d.rstrip("/")
    raise FileNotFoundError(pattern)
assets, data = find("v731/kaggle_infer.py"), find("test_series.csv")
ckpts = sorted(glob.glob(f"{find('v731_*.pt')}/v731_*.pt"))
src = f"{WORKING}/v731_src"
shutil.copytree(assets, src, dirs_exist_ok=True)
out = f"{WORKING}/specialist_raw.csv"
subprocess.run([sys.executable, f"{src}/v731/kaggle_infer.py", "predict", data, "test", out,
                f"{src}/side_model.pt", *ckpts, "--native-root", f"{WORKING}/native_test",
                "--record", f"{WORKING}/infer_record.json"], check=True, cwd=src)
sample = pd.read_csv(f"{data}/sample_submission.csv", dtype={"StudyInstanceUID": str})
pred = pd.read_csv(out, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
assert set(pred.index) == set(sample.StudyInstanceUID), "study set differs from sample_submission"
sub = pred.loc[sample.StudyInstanceUID, ["PF OA"]].reset_index()
assert sub["PF OA"].notna().all()
sub.to_csv(f"{WORKING}/specialist_pf.csv", index=False)
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"run_nonce": RUN_NONCE, "checkpoints_sha256": {os.path.basename(c): h(c) for c in ckpts},
           "specialist_pf_sha256": h(f"{WORKING}/specialist_pf.csv"), "finished_unix": time.time()},
          open(f"{WORKING}/runtime_receipt.json", "w"), indent=2)
print("wrote specialist_pf.csv", len(sub))
"""


LABEL_SHA256 = {  # train.py D4_SHA256 / D3_SHA256: D4 by default, D3 only as the round-2 fallback
    "d4": "5e3f6797be07300ce3584715ac31e43c0a2c602947040a4d6443cee72771b7e4",
    "d3": "16adf4fcdc25375b2a5a8fff685946b368033374959660254c9c78cbe5ebecff",
}
N_ALL = 4345  # round 2: 3906 train + 439 develop, gold58 never
# final fit (train.py --cohort final4403): 4345 + the 58 on official labels (labels_D sha256)
FINAL_COHORT, N_FINAL, N_GOLD = "final4403", 4403, 58
LABELS_D_SHA256 = "aabca6154133502d6d087ec3190e543652b5b1c1c344f4160a93817df1867e42"
DEBUG_OFF = ("limit", "limit_eval", "max_steps", "only_studies")
RUN_KEYS = (
    "variant",
    "cohort",
    "cohort_n",
    "cohort_sha256",
    "label_source",
    "label_sha256",
    "seed",
    "develop_n",
)  # train.py run_info = receipt["run"] = final_ema.pt["args"]


def checkpoint_record(path: Path) -> dict[str, Any]:
    """What a final_ema.pt was trained on, from train.py's own records (never the file name):
    final_ema.pt["args"] == receipt.json["run"] (train.py run_info), identity chain checkpoint ==
    receipt == COMPLETE.json, file sha256 == both receipts' final_ema.pt sha256, config equal."""
    import torch

    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "opt" in ck or "config" not in ck or "args" not in ck:
        raise InferError(f"{path}: not train.py's final_ema.pt (needs config + args)")
    try:
        rec = json.loads((path.parent / "receipt.json").read_text())
        done = json.loads((path.parent / "COMPLETE.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise InferError(f"{path}: receipt.json / COMPLETE.json missing or unreadable: {e}") from e
    sha = sha256_file(path)
    ident = ck.get("identity")
    if not ident or not (ident == rec.get("identity") == done.get("identity")):
        raise InferError(f"{path}: identity chain broken (checkpoint/receipt/COMPLETE)")
    outs = [x.get("outputs", {}).get("final_ema.pt", {}).get("sha256") for x in (rec, done)]
    if not sha == outs[0] == outs[1]:
        raise InferError(f"{path}: sha256 differs from the run's receipts")
    if rec.get("config") != ck["config"]:
        raise InferError(f"{path}: receipt config differs from the checkpoint's")
    args, run = ck["args"], rec.get("run")
    if args != run or any(k not in args for k in RUN_KEYS):
        raise InferError(f"{path}: checkpoint args != receipt run record")
    cfg = ck["config"]
    if args["variant"] != cfg.get("variant") or int(args["seed"]) != int(cfg["recipe"]["seed"]):
        raise InferError(f"{path}: args variant/seed disagree with the config")
    return {
        "path": str(path),
        "sha256": sha,
        "identity": ident,
        **{k: args[k] for k in RUN_KEYS},
        **{k: args[k] for k in ("gold_n", "gold_label_sha256") if k in args},
        "debug": rec.get("debug") or {},
        "inputs_sha256": dict(rec.get("input_sha256") or {}),
    }


def verify_final_set(
    records: list[dict[str, Any]],
    n_all: int = N_ALL,
    label_sha256: dict[str, str] | None = None,
    debug_off: tuple[str, ...] = DEBUG_OFF,
) -> dict[str, Any]:
    """The final package: 2-3 distinct seeds of ONE variant, each a round-2 run on the all-4345
    cohort (cohort == round2, cohort_n == 4345, nothing held out, no debug limits) with ONE active
    label source whose recorded sha256 is the pinned D4 (or D3 fallback) file; one cohort id hash
    and identical pinned inputs across seeds. Raises InferError otherwise."""
    shas = label_sha256 or LABEL_SHA256
    if not 2 <= len(records) <= 3:
        raise InferError(f"final bundle needs 2-3 seed checkpoints, got {len(records)}")
    variants = {r["variant"] for r in records}
    if len(variants) != 1:
        raise InferError(f"final bundle mixes variants {sorted(variants)}")
    seeds = [int(r["seed"]) for r in records]
    if len(set(seeds)) != len(seeds):
        raise InferError(f"final bundle repeats a seed: {seeds}")
    for r in records:
        final = r["cohort"] == FINAL_COHORT
        want_n = N_FINAL if final else n_all
        if r["cohort"] not in ("round2", FINAL_COHORT) or r["cohort_n"] != want_n or r["develop_n"]:
            raise InferError(
                f"{r['path']}: cohort {r['cohort']} n={r['cohort_n']} "
                f"develop={r['develop_n']}, not round2 on all {n_all} / final4403 on {N_FINAL}"
            )
        if final and (
            r.get("gold_n") != N_GOLD
            or r.get("gold_label_sha256") != LABELS_D_SHA256
            or r["label_source"] != "d4"
        ):
            raise InferError(f"{r['path']}: final4403 needs D4 + the 58 on pinned labels_D")
        if any(r["debug"].get(k) for k in debug_off):
            raise InferError(f"{r['path']}: debug-limited run")
        if r["label_source"] not in shas or r["label_sha256"] != shas[r["label_source"]]:
            raise InferError(
                f"{r['path']}: label {r['label_source']} sha256 {r['label_sha256']} "
                "is not the pinned D4/D3 file"
            )
    for key in ("cohort", "label_source", "cohort_sha256"):
        if len({r[key] for r in records}) != 1:
            raise InferError(f"final bundle mixes {key}")
    if len({json.dumps(r["inputs_sha256"], sort_keys=True) for r in records}) != 1:
        raise InferError("final bundle mixes pinned inputs")
    return {
        "variant": variants.pop(),
        "seeds": sorted(seeds),
        "label_source": records[0]["label_source"],
        "cohort": records[0]["cohort"],
        "n_train": records[0]["cohort_n"],
        "cohort_sha256": records[0]["cohort_sha256"],
    }


def _bands_version() -> str:
    import bands

    return str(bands.BANDS_VERSION)


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise InferError(f"{dst} exists; stage into a fresh directory")
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def bundle(
    ckpts: list[Path],
    side_model: Path,
    out: Path,
    nonce: str | None = None,
    branch: Path = BRANCH,
    canonical: Path = CANONICAL,
    mode: str = "final",
    n_all: int = N_ALL,
    label_sha256: dict[str, str] | None = None,
    debug_off: tuple[str, ...] = DEBUG_OFF,
) -> dict[str, Any]:
    """Stage ds_v731_models/, ds_v731_assets/, kernel/ and MANIFEST.json under a FRESH ``out``.
    mode "final": the verified round-2 package (verify_final_set). mode "dev": exactly ONE
    checkpoint (smoke / pipeline check), private "-dev" kernel, never a submission candidate."""
    if out.exists():
        raise InferError(f"{out} exists; bundles go to a fresh directory")
    if mode == "final":
        package = verify_final_set(
            [checkpoint_record(c) for c in ckpts], n_all, label_sha256, debug_off
        )
        kernel_id, label = KERNEL_ID, "final round-2 package"
    elif mode == "dev":
        if len(ckpts) != 1:
            raise InferError(f"dev bundle takes exactly one checkpoint, got {len(ckpts)}")
        package = {"mode": "dev"}
        kernel_id, label = KERNEL_ID + "-dev", "dev/smoke bundle (NOT a submission candidate)"
    else:
        raise InferError(f"unknown bundle mode {mode!r}")
    names = {}
    for c in ckpts:
        _, cfg, _ck = load_checkpoint(str(c))
        run = str(cfg["variant"]).upper()
        seed = int(cfg["recipe"]["seed"])
        name = f"v731_{run.lower()}_s{seed}.pt"
        if name in names:
            raise InferError(f"two checkpoints map to {name}")
        names[name] = c
    groups = {model_meta(load_checkpoint(str(c))[1]) for c in ckpts}
    if len(groups) != 1:
        raise InferError(f"checkpoints disagree on groups / k_eval: {groups}")
    md, ad, kd = out / "ds_v731_models", out / "ds_v731_assets", out / "kernel"
    for n, c in names.items():
        place(c, md / n)
    for pub, rel in asset_sources(branch).items():
        place(branch / rel, ad / pub)
    if canonical.is_file():
        place(canonical, ad / "reader/canonical.py")
    else:
        raise InferError(f"canonical.py missing: {canonical}")
    place(side_model, ad / "side_model.pt")
    import secrets

    nonce = nonce or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(8)}"
    kd.mkdir(parents=True)
    (kd / "v731_notebook.py").write_text(NOTEBOOK.replace("__V731_RUN_NONCE__", nonce))
    for d, i in ((md, MODELS_ID), (ad, ASSETS_ID)):
        (d / "dataset-metadata.json").write_text(
            json.dumps(
                {"title": i.split("/")[1], "id": i, "licenses": [{"name": "other"}]}, indent=2
            )
        )
    (kd / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[1],
                "code_file": "v731_notebook.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": [MODELS_ID, ASSETS_ID],
                "competition_sources": [COMPETITION],
                "kernel_sources": [],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
    )
    files = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "schema": "v731-kaggle-1",
        "mode": mode,
        "bands_version": _bands_version(),
        "label": label,
        "package": package,
        "run_nonce": nonce,
        "files_sha256": files,
        "checkpoints": {n: str(c) for n, c in names.items()},
        "commands": commands(out),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def commands(out: Path) -> list[str]:
    """Printed for the founder; never executed here."""
    return [
        f"kaggle datasets create -p {out / 'ds_v731_models'} -q",
        # assets hold sub-folders (reader/, v731/, v8prep/): without --dir-mode the CLI skips
        # them and the dataset would contain no code
        f"kaggle datasets create -p {out / 'ds_v731_assets'} --dir-mode zip -q",
        f"kaggle kernels push -p {out / 'kernel'}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("data_dir")
    p.add_argument("split")
    p.add_argument("out_csv")
    p.add_argument("side_model")
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--native-root")
    p.add_argument(
        "--drop-native",
        action="store_true",
        help="delete each study's P/Z/K arrays after use (Kaggle: disk stays flat)",
    )
    p.add_argument("--record")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2, help="studies per forward (T4: 1 if OOM)")
    p.add_argument("--dtype", default="float16")
    b = sub.add_parser("bundle")
    b.add_argument("--ckpts", nargs="+", type=Path, required=True)
    b.add_argument("--side-model", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--mode", choices=("final", "dev"), default="final")
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        native_root = a.native_root or tempfile.mkdtemp(prefix="v731_native_")
        t0 = time.time()
        df, rec = run(
            a.data_dir,
            a.split,
            a.side_model,
            a.ckpts,
            native_root,
            workers=a.workers,
            bs=a.batch_size,
            dtype=a.dtype,
            drop_native=a.drop_native,
        )
        df.to_csv(a.out_csv, index=False)
        if a.record:
            Path(a.record).write_text(
                json.dumps(
                    {
                        "bands_version": _bands_version(),
                        "studies": rec,
                        "ckpts_sha256": {os.path.basename(c): sha256_file(c) for c in a.ckpts},
                        "seconds": time.time() - t0,
                    },
                    indent=1,
                )
            )
        print(f"wrote {a.out_csv}: {len(df)} studies, {time.time() - t0:.0f}s")
    else:
        m = bundle(a.ckpts, a.side_model, a.out, mode=a.mode)
        print(json.dumps({k: v for k, v in m.items() if k != "files_sha256"}, indent=2))
        print("NOT executed (founder approval needed):")
        for c in m["commands"]:
            print("  " + c)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""radiant-kg C-line inference: predict the 12 findings straight from DICOM folders, with left-knee mirroring.

Same preprocessing, study table and windows as training (knee.py / preprocess.py, unchanged), plus the side rule the
C-line models were trained with (docs/intent/c_plan.md):
- unanimous DICOM laterality tags, else the sign of the median series centre-x (radiant canonical.resolve_knee_side);
- a single tag is overruled by the position when the knee lies >= 20 mm on the other side of the scanner centre;
- no tag and the knee within 20 mm of the centre: the side classifier (side_model.pt) on the central coronal slices
  decides when it is confident (|p - 0.5| * 2 >= 0.9).
Left knees are mirrored to look like right knees (sagittal: slice order reversed; coronal/axial: columns reversed).

Close-ups (V3) are placed exactly as in training (geo.py, views.py) from every file's header.
V8 arm P (checkpoint arg phys_pos): sagittal slice displacement from the same header anchor (phys_dx), the rule
v8prep/build_phys_pos.py applies to the training cache.

Arm F (checkpoints trained with --t1-series): there is no precomputed T1 list for the hidden test set, so every
non-fat-suppressed sagittal/coronal series is classified from its first file's header with seqtype.is_t1, the same
function that wrote the training list, and the study table prefers T1 series exactly as in training.

Arm O (checkpoints trained with --oai): the separate OAI heads are rebuilt so the state dict loads strictly; the
model's default forward returns only the twelve competition logits, so the OAI outputs are never computed here.

V8.1 (Kaggle T4): the scoring dtype is resolve_dtype()'s, not blindly the checkpoint's. V8 checkpoints save
inference_dtype=bfloat16 (the training boxes' scoring dtype); a T4 (compute capability 7.5) has no native bf16, and bf16
autocast there was reported catastrophic. Order: the RADIANT_INFER_DTYPE environment variable when set (the V8.1 Kaggle
notebook sets float16); else bfloat16 becomes float16 on a CUDA device below compute capability 8; else the
checkpoint's own value (float16 for legacy V7 checkpoints that saved none).

V8.1 runtime binding (Codex kaggle-runtime-bundle-binding): when RADIANT_RUNTIME_RECORD names a file, the run writes
there what it ACTUALLY used: the sha256 of every checkpoint file it loaded with its saved and scoring dtype and whether
autocast was on, the side model's sha256, the sha256 of every module it imported from its own directory, the device and
the torch/CUDA versions. v81_notebook.py folds this into runtime_receipt.json; nothing changes when the variable is unset.

Usage: python infer_c.py <data_dir> <split> <out_csv> <side_model.pt> <ckpt> [<ckpt> ...]
"""
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch

import geo
import seqtype
import views as V
from canonical import resolve_knee_side, series_side_evidence
from knee import LABELS, N_SLOTS, KneeNet, StudyDataset, build_study_table, parse_fx_bias, parse_pf_groups
from preprocess import MAX_SLICES, load_series

POSITION_MM, SIDE_CONF = 20.0, 0.9
INFER_DTYPE_ENV = "RADIANT_INFER_DTYPE"  # radiant-kg V8.1: float16 | bfloat16 | float32 override
RUNTIME_RECORD_ENV = "RADIANT_RUNTIME_RECORD"  # radiant-kg V8.1: write what was actually loaded/used here
LOADED = []  # radiant-kg V8.1: one entry per checkpoint load_models() loaded (filled only when the record is on)
DTYPES = ("float16", "bfloat16", "float32")
# radiant-kg V8.1: checkpoint args passed to KneeNet only when its signature takes them (fx_bias = W2's Fracture bias)


def bf16_native(device):
    """True when `device` runs bf16 natively: CUDA compute capability >= 8 (Ampere and later). A T4 is 7.5;
    torch.cuda.is_bf16_supported() can report True there through emulation, so the capability decides."""
    return device == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def resolve_dtype(saved, device):
    """radiant-kg V8.1: the autocast dtype for one checkpoint (see the module docstring)."""
    forced = os.environ.get(INFER_DTYPE_ENV, "").strip()
    if forced:
        if forced not in DTYPES:
            raise ValueError(f"{INFER_DTYPE_ENV}={forced!r}; expected one of {DTYPES}")
        return forced
    saved = saved or "float16"
    if saved == "bfloat16" and device == "cuda" and not bf16_native(device):
        return "float16"
    return saved


def read_headers(root, st, series_ids, all_files):
    """{series: [headers]}: first and last file only, or every file when close-up geometry is needed."""
    out = {}
    for sr in series_ids:
        files = sorted(glob.glob(f"{root}/{st}/{sr}/*.dcm"))
        pick = files if all_files else (files[:1] + files[-1:])
        hs = []
        for f in pick:
            try:
                hs.append(pydicom.dcmread(f, stop_before_pixels=True))
            except Exception:
                pass
        out[sr] = hs
    return out


def t1_series_from_dicom(root, se):
    """radiant-kg F: {series ids judged T1} among the series that can fill slots 3/4, from the first file's header
    (read_headers' choice). Same rule as build_t1_series.py, which wrote the training list."""
    out, c = set(), seqtype.nonfs_sag_cor(se)
    for st, sr in zip(c.StudyInstanceUID, c.SeriesInstanceUID):
        if seqtype.series_is_t1(read_headers(root, st, [sr], False)[sr]):
            out.add(str(sr))
    return out


def phys_dx(geoms, headers, X, vols, sagittal):
    """radiant-kg V8 arm P: {series: dx per loaded slice} for the chosen sagittal series, the rule of
    v8prep/build_phys_pos.py: header plane sagittal, every header has IOP and IPP (otherwise load_series and
    geo.series_geometry may order or keep different slices) and geometry for every loaded slice; else no entry (zero)."""
    out = {}
    if X is None:
        return out
    for sr in sagittal:
        plane, affs = geoms.get(sr, (None, []))
        hs = headers.get(sr, [])
        complete = all("ImageOrientationPatient" in h and "ImagePositionPatient" in h for h in hs)
        if plane == "Sagittal" and complete and sr in vols and len(affs) == len(vols[sr]):
            out[sr] = V.series_dx(affs, X)
    return out


def study_side(headers, side_net, cor_vol):
    """'L' or 'R' for one study, following the training rule. cor_vol: canonical coronal volume or None."""
    ev = []
    for hs in headers.values():
        if hs:
            ev.append(series_side_evidence(hs[:1] + hs[-1:]))
    side, _ = resolve_knee_side(ev)
    tags = {s for s, _ in ev if s}
    xs = [c for _, c in ev if c is not None]
    x = float(np.median(xs)) if xs else None
    pos = None if x is None or x == 0 else ("L" if x > 0 else "R")
    if len(tags) == 1 and pos and pos != next(iter(tags)) and abs(x) >= POSITION_MM:
        return pos
    if not tags and (x is None or abs(x) < POSITION_MM) and side_net is not None and cor_vol is not None:
        n = len(cor_vol)
        img = np.stack([cv2.resize(cor_vol[min(max(j, 0), n - 1)], (256, 256), interpolation=cv2.INTER_AREA)
                        for j in (n // 2 - 2, n // 2, n // 2 + 2)]).astype(np.float32) / 255
        with torch.no_grad():
            p = float(torch.sigmoid(side_net(torch.from_numpy(img)[None]))[0, 0])
        if abs(p - 0.5) * 2 >= SIDE_CONF:
            return "R" if p > 0.5 else "L"
    return side if side in ("L", "R") else "R"


class DicomStudyDataset(torch.utils.data.Dataset):
    """Loads a study's selected series from DICOM, decides its side, then builds the model input with the training
    StudyDataset itself (same windows, mirroring and close-ups), so inference cannot drift from training."""

    def __init__(self, studies, table, root, k, all_series, side_model, use_views=False, use_phys=False):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k
        self.all_series, self.side_model, self.side_net, self.use_views = all_series, side_model, None, use_views
        self.use_phys = use_phys  # radiant-kg V8 arm P

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        if self.side_net is None and self.side_model:
            self.side_net = timm.create_model("resnet18", pretrained=False, num_classes=2)
            self.side_net.load_state_dict(torch.load(self.side_model, map_location="cpu"))
            self.side_net.eval()
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        vols, chosen = {}, [[] for _ in range(N_SLOTS)]
        for s, cands in enumerate(slots):
            if cands:
                try:
                    vols[cands[0]] = load_series(f"{self.root}/{st}/{cands[0]}")[0]
                    chosen[s] = [cands[0]]
                except Exception:
                    pass
        cor_ids = chosen[1] or chosen[4]  # fat-suppressed coronal first, as for the classifier
        cor = vols[cor_ids[0]] if cor_ids else None
        headers = read_headers(self.root, st, self.all_series.get(st, []), self.use_views or self.use_phys)
        side = study_side(headers, self.side_net, cor)
        boxes = phys = None
        if self.use_views or self.use_phys:
            geoms = {sr: geo.series_geometry(hs) for sr, hs in headers.items()}
            X = V.slab_prior(geoms)
        if self.use_phys:
            phys = phys_dx(geoms, headers, X, vols, chosen[0] + chosen[3])
        if self.use_views:
            boxes = {}
            if X is not None:
                for sr in vols:
                    plane, affs = geoms.get(sr, (None, []))
                    if affs:
                        boxes[sr] = V.crop_boxes(plane, affs, X, side)
        ds = StudyDataset([st], {st: chosen}, None, k=self.k, train=False, loader=lambda sr: vols[sr],
                          mirror={st} if side == "L" else None, views=boxes, phys=phys)
        item = ds[0]
        x, mask, pos = item[:3]
        return (x, mask, pos, side) + ((item[4],) if self.use_phys else ())


def load_models(ckpts):
    """-> (models [(model, res)], k_eval, views, T1 rule or None, phys_pos)."""
    models, ks, vs, t1, ps = [], set(), set(), set(), set()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = KneeNet(a["backbone"], pretrained=False, spatial=bool(a.get("spatial", False)),
                    views=bool(a.get("views")),
                    pf_groups=parse_pf_groups(a.get("pf_groups")),  # radiant-kg V8 arm R
                    oai_heads=len(ck.get("oai_heads") or []),  # radiant-kg V8 arm O: OAI outputs built, not used
                    phys_pos=bool(a.get("phys_pos")),  # radiant-kg V8 arm P
                    fx_bias=parse_fx_bias(a.get("fx_bias")))  # radiant-kg V8.1: learned values come from the state dict
        vs.add(bool(a.get("views")))
        t1.add(a.get("t1_rule") if a.get("t1_series") else None)  # radiant-kg F
        ps.add(bool(a.get("phys_pos")))  # radiant-kg V8 arm P
        m.load_state_dict(ck["model"])
        m.inference_dtype = resolve_dtype(a.get("inference_dtype", "float16"), dev)  # legacy V7: float16; V8.1 above
        if os.environ.get(RUNTIME_RECORD_ENV):  # radiant-kg V8.1: hash the exact file torch.load read
            LOADED.append({"file": os.path.basename(p), "sha256": file_sha256(p),
                           "saved_dtype": a.get("inference_dtype"), "scoring_dtype": m.inference_dtype,
                           "device": dev, "autocast": dev == "cuda" and m.inference_dtype != "float32"})
        print(f"{os.path.basename(p)}: saved inference dtype {a.get('inference_dtype')}, "
              f"scoring dtype {m.inference_dtype} on {dev}", flush=True)
        models.append((m.cuda().eval() if torch.cuda.is_available() else m.eval(), int(a["res"])))
        ks.add(int(a["k_eval"]))
    assert len(ks) == 1 and len(vs) == 1 and len(ps) == 1, f"checkpoints disagree on k_eval {ks}, views {vs} or phys {ps}"
    assert len(t1) == 1, f"checkpoints disagree on the T1 rule {t1}"
    t1_rule = t1.pop()
    assert t1_rule in (None, seqtype.T1_RULE), f"checkpoints trained with T1 rule {t1_rule}, seqtype has {seqtype.T1_RULE}"
    return models, ks.pop(), vs.pop(), t1_rule, ps.pop()


def run(data_dir, split, side_model, ckpts, studies=None, bs=2, workers=4):
    se = pd.read_csv(f"{data_dir}/{split}_series.csv")
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv").StudyInstanceUID.tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
    models, k, use_views, t1_rule, use_phys = load_models(ckpts)
    t1 = None
    if t1_rule is not None:  # radiant-kg F
        t1 = t1_series_from_dicom(root, se)
        print(f"t1-series ({t1_rule}): {len(t1)} T1 series of {len(seqtype.nonfs_sag_cor(se))} non-FS sag/cor", flush=True)
    table = build_study_table(se, n_files, t1)
    all_series = se.groupby("StudyInstanceUID").SeriesInstanceUID.apply(list).to_dict()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = torch.utils.data.DataLoader(DicomStudyDataset(studies, table, root, k, all_series, side_model, use_views, use_phys),
                                     batch_size=bs, num_workers=workers, pin_memory=dev == "cuda")
    out, sides = [], []
    for batch in dl:
        x, mask, pos, side = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev), batch[3]
        phys = batch[4].to(dev) if use_phys else None  # radiant-kg V8 arm P
        predictions = []
        with torch.no_grad():
            for m, res in models:
                dtype = m.inference_dtype
                with torch.autocast(dev, dtype=getattr(torch, dtype), enabled=dev == "cuda" and dtype != "float32"):
                    predictions.append(m(x, mask, pos, res=res, phys=phys).float().sigmoid())
        p = torch.stack(predictions).mean(0)
        out.append(p.cpu())
        sides += list(side)
    df = pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)
    print("sides:", pd.Series(sides).value_counts().to_dict(), flush=True)
    return df


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_record(side_model):
    """radiant-kg V8.1: what this process actually loaded (see the module docstring)."""
    here = os.path.dirname(os.path.abspath(__file__))
    mods = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)  # some extension modules report a bare relative name: skip those
        if f and os.path.isabs(f) and os.path.dirname(f) == here and f.endswith(".py") and os.path.isfile(f):
            mods[os.path.basename(f)] = file_sha256(f)
    cuda = torch.cuda.is_available()
    return {"schema": "v81-infer-runtime-1", "checkpoints": LOADED,
            "side_model": {"file": os.path.basename(side_model), "sha256": file_sha256(side_model)},
            "imported_sources_sha256": dict(sorted(mods.items())),
            "infer_dtype_env": os.environ.get(INFER_DTYPE_ENV, ""),
            "device": "cuda" if cuda else "cpu", "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda}


if __name__ == "__main__":
    data_dir, split, out_csv, side_model, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    run(data_dir, split, side_model, ckpts).to_csv(out_csv)
    print("wrote", out_csv)
    if os.environ.get(RUNTIME_RECORD_ENV):  # radiant-kg V8.1
        with open(os.environ[RUNTIME_RECORD_ENV], "w") as f:
            json.dump(runtime_record(side_model), f, indent=2)

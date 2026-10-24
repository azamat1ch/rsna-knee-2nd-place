"""V4 encoder: teach the reader's 2D encoder what meniscus tears and cartilage loss look like, from fastMRI+ boxes.

Slice-level targets (lateral tear, medial tear, lateral cartilage loss, medial cartilage loss) from fastMRI+ boxes
(drawn on the 320 px reconstruction, flipped up/down only), on fastMRI single-coil coronal volumes rendered like our
0.4 mm cache (384 px) and oriented to the canonical right-knee view with the side classifier (prep_fastmri.py; global
up/down by majority, per-volume left/right, confident volumes only). In that frame image columns run patient right ->
left, so the lateral half is the low-column half; boxes in a central band are left unlabelled for that slice. The
encoder sees the reader's own 3-slice windows and augmentation (knee.make_views), then its weights are saved for
`train.py --init-enc`.

  python pretrain_fastmri.py --work work --tars <val.tar.xz> <train.tar.xz> --boxes ... --reviewed ... \
      --side-model work/side_model.pt --out work/fm_encoder.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reader"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knee import make_views  # noqa: E402
from prep_fastmri import canvas, side_input  # noqa: E402

BAND = 0.06
TARGETS = ["lat_tear", "med_tear", "lat_cart", "med_cart"]
# radiant-kg V6: unsided abnormalities also taught (a box on the slice = present), when --more-targets is given
MORE = {"acl": {"Ligament - ACL High Grade Sprain", "Ligament - ACL Low Grade sprain", "Ligament - ACL Low Grade Sprain"},
        "bone": {"Bone-Fracture/Contusion/dislocation", "Bone- Subchondral edema"},
        "mcl": {"Ligament - MCL Low-Mod Grade Sprain", "Ligament - MCL High Grade sprain", "Ligament - MCL High Grade Sprain"}}
TEAR = {"Meniscus Tear", "Displaced Meniscal Tissue"}
CART = {"Cartilage - Partial Thickness loss/defect", "Cartilage - Full Thickness loss/defect"}


def convert(args):
    """One archive -> <out>/fm_<file>.npy (uint8 [z, 384, 384], as stored) + side inputs for the classifier."""
    tar, out = args
    rows = []
    with tarfile.open(tar, "r|xz") as tf:
        for m in tf:
            if not m.name.endswith(".h5"):
                continue
            fid = Path(m.name).stem
            try:
                import io
                with h5py.File(io.BytesIO(tf.extractfile(m).read())) as h:
                    rec = h["reconstruction_esc"][()]
                np.save(f"{out}/fm_{fid}.npy", canvas(rec, 0.4, 384))
                np.save(f"{out}/side_{fid}.npy", side_input(rec))
                rows.append(fid)
            except Exception as e:  # one bad file never stops the run
                print("skip", fid, repr(e)[:120], flush=True)
            if len(rows) % 100 == 0:
                print(Path(tar).name, "converted", len(rows), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--side-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="dry runs: use only this many volumes")
    ap.add_argument("--more-targets", action="store_true", help="V6: also teach ACL sprain, bone contusion/fracture/oedema, MCL sprain")
    ap.add_argument("--reuse", action="store_true", help="reuse volumes already converted into <work>/fm_vols")
    a = ap.parse_args()
    vdir = f"{a.work}/fm_vols"
    os.makedirs(vdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    global TARGETS
    if a.more_targets:
        TARGETS = TARGETS + list(MORE)
    # 1. convert both archives in parallel (xz decompression is single-threaded per archive), or reuse
    if a.reuse:
        files = sorted(Path(f).stem[3:] for f in os.listdir(vdir) if f.startswith("fm_") and f.endswith(".npy"))
    else:
        with Pool(len(a.tars)) as p:
            files = [f for part in p.map(convert, [(t, vdir) for t in a.tars]) for f in part]
    if a.limit:
        files = files[:a.limit]
    print("volumes", len(files), flush=True)

    # 2. orientation with the side classifier
    sm = timm.create_model("resnet18", pretrained=False, num_classes=2)
    sm.load_state_dict(torch.load(a.side_model, map_location="cpu"))
    sm.eval().to(dev)
    q = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            x = np.stack([np.load(f"{vdir}/side_{f}.npy") for f in files[i:i + 64]]).astype(np.float32) / 255
            xs = torch.from_numpy(np.concatenate([x, x[:, :, ::-1].copy()])).to(dev)
            o = torch.sigmoid(sm(xs)).cpu().numpy()
            n = len(x)
            q += [(o[j, 0], o[j, 1], o[n + j, 0]) for j in range(n)]
    S = pd.DataFrame(q, index=files, columns=["p_right", "p_upright", "p_right_upside_down"])
    vflip = (S.p_upright > 0.5).mean() < 0.5
    pr = S.p_right_upside_down if vflip else S.p_right
    S["mirror"], S["conf"] = pr < 0.5, (pr - 0.5).abs() * 2
    reviewed = set(pd.read_csv(a.reviewed, header=None)[0])
    keep = [f for f in files if S.loc[f, "conf"] >= a.min_conf and f in reviewed]
    print(f"upside down: {bool(vflip)}; confident and reviewed: {len(keep)} of {len(files)}", flush=True)

    # 3. slice-level targets (NaN = unknown for that slice)
    boxes = pd.read_csv(a.boxes)
    boxes["label"] = boxes["label"].str.strip()
    items, Y = [], []
    for f in keep:
        n = np.load(f"{vdir}/fm_{f}.npy", mmap_mode="r").shape[0]
        y = np.zeros((n, len(TARGETS)), np.float32)
        b = boxes[boxes.file == f]
        for _, r in b.iterrows():
            z = int(r.slice)
            if not 0 <= z < n:
                continue
            cx = (r.x + r.width / 2) / 320.0
            cx = 1 - cx if S.loc[f, "mirror"] else cx
            side = "lat" if cx < 0.5 - BAND else "med" if cx > 0.5 + BAND else None
            unsided = next((t for t, labs in MORE.items() if r.label in labs), None) if a.more_targets else None
            if unsided is not None:
                y[z, TARGETS.index(unsided)] = 1.0
                continue
            kind = "tear" if r.label in TEAR else "cart" if r.label in CART else None
            if kind is None:
                continue
            if side is None:
                y[z, [TARGETS.index(f"lat_{kind}"), TARGETS.index(f"med_{kind}")]] = np.nan
            elif not np.isnan(y[z, TARGETS.index(f"{side}_{kind}")]):
                y[z, TARGETS.index(f"{side}_{kind}")] = 1.0
        for z in range(1, n - 1):
            items.append((f, z))
            Y.append(y[z])
    Y = np.array(Y)
    print("windows", len(items), "positives", dict(zip(TARGETS, np.nansum(Y, 0).astype(int).tolist())), flush=True)
    rng = np.random.default_rng(0)
    vols = np.array(sorted(set(f for f, _ in items)))
    val_vols = set(rng.choice(vols, max(1, len(vols) // 10), replace=False))
    tr = [i for i, (f, _) in enumerate(items) if f not in val_vols]
    va = [i for i, (f, _) in enumerate(items) if f in val_vols]
    cache = {}

    def window(i):
        f, z = items[i]
        if f not in cache:
            v = np.load(f"{vdir}/fm_{f}.npy")
            v = v[:, ::-1] if vflip else v
            v = v[:, :, ::-1] if S.loc[f, "mirror"] else v
            cache[f] = np.ascontiguousarray(v)
        return cache[f][z - 1:z + 2]

    enc = timm.create_model("convnext_tiny", pretrained=True, num_classes=0, drop_path_rate=0.1).to(dev)
    headl = torch.nn.Linear(enc.num_features, len(TARGETS)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(headl.parameters()), lr=1e-4, weight_decay=0.02)
    bs = 32
    steps = a.epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=max(1, steps))
    pos = np.nansum(Y[tr], 0)
    neg = np.sum(~np.isnan(Y[tr]), 0) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50), dtype=torch.float32, device=dev)

    def predict(idx):
        enc.eval()
        out = []
        with torch.no_grad():
            for j in range(0, len(idx), 64):
                x = torch.from_numpy(np.stack([window(i) for i in idx[j:j + 64]])).to(dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    out.append(torch.sigmoid(headl(enc(make_views(x, 256, False, 1)))).float().cpu())
        return torch.cat(out).numpy()

    report = {}
    for ep in range(a.epochs):
        enc.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for j in range(0, len(perm) - bs + 1, bs):
            idx = perm[j:j + bs]
            x = torch.from_numpy(np.stack([window(i) for i in idx])).to(dev)
            y = torch.from_numpy(Y[idx]).to(dev)
            known = ~torch.isnan(y)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logit = headl(enc(make_views(x, 256, True, 1)))
            loss = (F.binary_cross_entropy_with_logits(logit.float(), torch.nan_to_num(y), pos_weight=pw,
                                                       reduction="none") * known).sum() / known.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        p = predict(va)
        aucs = {}
        for k, t in enumerate(TARGETS):
            yk = Y[va, k]
            m = ~np.isnan(yk)
            aucs[t] = float(roc_auc_score(yk[m], p[m, k])) if m.any() and 0 < yk[m].sum() < m.sum() else None
        report[f"epoch{ep}"] = {"loss": tot / max(1, len(perm) // bs), "val_auc": aucs}
        print(f"epoch {ep}", json.dumps(report[f"epoch{ep}"]), flush=True)
    torch.save({k: v for k, v in enc.state_dict().items()}, a.out)
    report.update({"windows": len(items), "val_volumes": len(val_vols), "upside_down": bool(vflip),
                   "volumes_used": len(keep), "positives_train": dict(zip(TARGETS, pos.astype(int).tolist()))})
    json.dump(report, open(a.out.replace(".pt", "_report.json"), "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()

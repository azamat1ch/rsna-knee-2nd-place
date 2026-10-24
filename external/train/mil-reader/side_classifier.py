# Left/right knee classifier for V4 (docs/intent/c_plan.md), run as a private Kaggle notebook.
# 1. Trusted sides from DICOM headers (one tag agreeing with the scanner position, or no tag, knee >= 40 mm from centre).
# 2. A small CNN on canonical coronal slices learns "looks like a right knee as stored" and "upright".
# 3. Held-out accuracy gate (>= 99 %), then predictions on our uncertain studies and on fastMRI validation scans,
#    plus image sheets for a visual fibula check. No label is read; nothing is submitted.
import glob, io, json, os, subprocess, sys, tarfile, time, urllib.request
import multiprocessing as mp
from multiprocessing import Pool

mp.set_start_method("fork", force=True)  # workers inherit state; the script has no __main__ guard

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "h5py"], check=False)
SRC = os.environ.get("SIDE_SRC", "/kaggle/working/src")
os.makedirs(SRC, exist_ok=True)
for name, text in EMBEDDED.items():  # noqa: F821 (filled in by the builder)
    open(f"{SRC}/{name}", "w").write(text)
sys.path.insert(0, SRC)

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn.functional as F
from canonical import series_side_evidence, resolve_knee_side  # our side logic, unchanged
import preprocess as PP  # the reader's canonical orientation and 0.4 mm resampling, unchanged

T0 = time.time()
OUT = os.environ.get("SIDE_OUT", "/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = int(os.environ.get("SIDE_LIMIT", "0"))  # dry runs only
EP = int(os.environ.get("SIDE_EPOCHS", "10"))
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
DATA = os.environ.get("SIDE_DATA") or os.path.dirname((glob.glob("/kaggle/input/*/train_series.csv") + glob.glob("/kaggle/input/*/*/train_series.csv"))[0])
SER = pd.read_csv(f"{DATA}/train_series.csv")
log("data at", DATA, "series", len(SER))


def study_info(args):
    study, rows = args
    ev, cor = [], []
    for sr, plane, fs in rows:
        d = f"{DATA}/train_series/{study}/{sr}"
        files = sorted(glob.glob(f"{d}/*.dcm"))
        if not files:
            continue
        hs = []
        for f in (files[0], files[-1]):
            try:
                hs.append(pydicom.dcmread(f, stop_before_pixels=True))
            except Exception:
                pass
        if hs:
            ev.append(series_side_evidence(hs))
        if plane == "Coronal":
            cor.append((len(files), fs, d))
    side, source = resolve_knee_side(ev)
    tags = {s for s, _ in ev if s}
    xs = [c for _, c in ev if c is not None]
    x = float(np.median(xs)) if xs else None
    img = None
    if cor:
        cor.sort(key=lambda t: (-t[1], -t[0]))  # prefer fat-suppressed, then the longest
        try:
            vol, _ = PP.load_series(cor[0][2], "Coronal")
            n = len(vol)
            img = np.stack([cv2.resize(vol[min(max(j, 0), n - 1)], (256, 256), interpolation=cv2.INTER_AREA)
                            for j in (n // 2 - 2, n // 2, n // 2 + 2)])
        except Exception:
            img = None
    return {"study": study, "tags": "".join(sorted(tags)), "x": x, "side": side, "source": source, "img": img}


tasks = [(st, list(zip(g.SeriesInstanceUID, g.Anatomical_Plane, g.Fat_Suppression))) for st, g in SER.groupby("StudyInstanceUID")]
if LIMIT:
    tasks = tasks[:LIMIT]
rows = []
with Pool(os.cpu_count()) as p:
    for i, r in enumerate(p.imap_unordered(study_info, tasks, chunksize=4)):
        rows.append(r)
        if i % 500 == 0:
            log("studies", i)
S = pd.DataFrame([{k: v for k, v in r.items() if k != "img"} for r in rows]).set_index("study")
IMG = {r["study"]: r["img"] for r in rows if r["img"] is not None}
pos = np.where(S.x > 0, "L", np.where(S.x < 0, "R", ""))
far = S.x.abs() >= 40
trusted = far & (((S.tags.isin(["L", "R"])) & (pos == S.tags)) | (S.tags == ""))
S["truth"] = np.where(trusted, pos, "")
ids = sorted(s for s in S.index[trusted] if s in IMG)  # imap_unordered order varies: sort before the split
log("trusted with a coronal image:", len(ids), "of", len(S))
rng = np.random.default_rng(0)
rng.shuffle(ids)
cut = int(0.8 * len(ids))
tr_ids, te_ids = ids[:cut], ids[cut:]


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, train):
        self.ids, self.train = ids, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = IMG[self.ids[i]].astype(np.float32) / 255.0
        right = float(S.loc[self.ids[i], "truth"] == "R")  # "looks like a right knee as stored"
        upright = 1.0
        if self.train:
            if rng.random() < 0.5:
                x, right = x[:, :, ::-1], 1 - right  # mirroring turns a right knee into a left one
            if rng.random() < 0.5:
                x, upright = x[:, ::-1, :], 0.0
            s = rng.uniform(0.7, 1.0)  # zoom: fastMRI uses a 140 mm field, ours 153.6 mm
            c = int(256 * s)
            y0, x0 = rng.integers(0, 256 - c + 1, 2)
            x = np.stack([cv2.resize(ch[y0:y0 + c, x0:x0 + c], (256, 256)) for ch in x])
            x = np.clip(x ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.tensor([right, upright])


def make():
    m = timm.create_model("resnet18", pretrained=True, num_classes=2)
    return m.to(DEV)


def predict(m, arrs):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), 64):
            x = torch.from_numpy(np.stack(arrs[i:i + 64]).astype(np.float32) / 255.0).to(DEV)
            out.append(torch.sigmoid(m(x)).cpu().numpy())
    return np.concatenate(out)


torch.manual_seed(0)
m = make()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
dl = torch.utils.data.DataLoader(DS(tr_ids, True), batch_size=min(64, max(2, len(tr_ids))), shuffle=True, num_workers=0, drop_last=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EP * len(dl))
for ep in range(EP):
    m.train()
    tot = 0
    for x, y in dl:
        loss = F.binary_cross_entropy_with_logits(m(x.to(DEV)), y.to(DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    p = predict(m, [IMG[i] for i in te_ids])
    t = (S.loc[te_ids, "truth"] == "R").values
    acc = float(((p[:, 0] > 0.5) == t).mean())
    log(f"epoch {ep} loss {tot / len(dl):.4f} held-out side accuracy {acc:.4f} upright {float((p[:, 1] > 0.5).mean()):.4f}")
torch.save(m.state_dict(), f"{OUT}/side_model.pt")

# held-out report, including mirrored and upside-down copies (the model must get those right too)
te = [IMG[i] for i in te_ids]
t = (S.loc[te_ids, "truth"] == "R").values
p0 = predict(m, te)
p_h = predict(m, [a[:, :, ::-1] for a in te])
p_v = predict(m, [a[:, ::-1, :] for a in te])
conf = np.abs(p0[:, 0] - 0.5) * 2
rep = {"n_trusted": len(ids), "n_test": len(te_ids),
       "side_acc": float(((p0[:, 0] > 0.5) == t).mean()),
       "side_acc_mirrored": float(((p_h[:, 0] > 0.5) == ~t).mean()),
       "upright_acc": float((p0[:, 1] > 0.5).mean()), "upside_down_acc": float((p_v[:, 1] < 0.5).mean())}
for c in (0.5, 0.8, 0.9, 0.98):
    k = conf >= c
    rep[f"side_acc_conf{c}"] = [float(((p0[k, 0] > 0.5) == t[k]).mean()) if k.any() else None, float(k.mean())]
rep["gate_99_passed"] = rep["side_acc"] >= 0.99
log(json.dumps(rep))

# our studies whose side the header rule is unsure about
unsure = S.index[~trusted]
pu = predict(m, [IMG[i] for i in unsure if i in IMG])
pd.DataFrame({"study": [i for i in unsure if i in IMG], "p_right": pu[:, 0], "p_upright": pu[:, 1]}).merge(
    S.reset_index()[["study", "tags", "x", "side", "source"]], on="study").to_csv(f"{OUT}/ours_unsure_sides.csv", index=False)


# fastMRI validation scans, streamed (never stored whole)
def fm_canonical(rec):
    n = int(round(140.0 / 0.4))
    out = np.zeros((len(rec), 384, 384), np.float32)
    o = (384 - n) // 2
    for i, s in enumerate(rec):
        out[i, o:o + n, o:o + n] = cv2.resize(s.astype(np.float32), (n, n))
    lo, hi = np.percentile(out[:, ::4, ::4], [1, 99.5])
    v = (np.clip((out - lo) / max(hi - lo, 1e-9), 0, 1) * 255).round().astype(np.uint8)
    k = len(v)
    return np.stack([cv2.resize(v[min(max(j, 0), k - 1)], (256, 256), interpolation=cv2.INTER_AREA) for j in (k // 2 - 2, k // 2, k // 2 + 2)])


import h5py  # noqa: E402

fm, fm_img = [], {}
try:
    if os.environ.get("SIDE_SKIP_FM"):
        raise RuntimeError("skipped (dry run)")
    with urllib.request.urlopen(FASTMRI_VAL_URL, timeout=120) as r, tarfile.open(fileobj=r, mode="r|xz") as tf:  # noqa: F821
        for mbr in tf:
            if not mbr.name.endswith(".h5"):
                continue
            b = tf.extractfile(mbr).read()
            with h5py.File(io.BytesIO(b)) as h:
                a = fm_canonical(h["reconstruction_esc"][()])
                acq = str(h.attrs.get("acquisition", ""))
            fid = os.path.basename(mbr.name)[:-3]
            fm_img[fid] = a
            fm.append({"file": fid, "acq": acq})
            if len(fm) % 25 == 0:
                log("fastMRI", len(fm))
            if time.time() - T0 > 10.5 * 3600 or len(fm) >= int(os.environ.get("SIDE_FM_MAX", "100000")):
                break
except Exception as e:
    log("fastMRI stream stopped:", repr(e)[:200])
if fm:
    F_ = pd.DataFrame(fm)
    arrs = [fm_img[f] for f in F_.file]
    for tag, fn in (("as_stored", lambda a: a), ("upside_down", lambda a: a[:, ::-1, :])):
        q = predict(m, [fn(a) for a in arrs])
        F_[f"p_right_{tag}"], F_[f"p_upright_{tag}"] = q[:, 0], q[:, 1]
    vote = float((F_.p_upright_as_stored > 0.5).mean())
    F_["global_upright_vote_as_stored"] = vote
    F_.to_csv(f"{OUT}/fastmri_val_sides.csv", index=False)
    flip_v = vote < 0.5
    side_col = "p_right_upside_down" if flip_v else "p_right_as_stored"
    # image sheet: 24 scans shown upright, with the model's call; a right knee has the fibula on the image's left
    pick = F_.sample(min(24, len(F_)), random_state=0)
    tiles = []
    for _, rw in pick.iterrows():
        a = fm_img[rw.file][1]
        a = a[::-1] if flip_v else a
        t_ = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        call = "R" if rw[side_col] > 0.5 else "L"
        cv2.putText(t_, f"{rw.file[-7:]} {call} {rw[side_col]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t_)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
    cv2.imwrite(f"{OUT}/fastmri_side_sheet.png", sheet)
    rep.update({"fastmri_scans": len(F_), "fastmri_upright_vote_as_stored": vote,
                "fastmri_called_right": float((F_[side_col] > 0.5).mean()),
                "fastmri_confident_0.9": float((np.abs(F_[side_col] - 0.5) * 2 >= 0.9).mean())})
# a matching sheet of our own trusted test studies, for comparison
tiles = []
for sid in te_ids[:24]:
    t_ = cv2.cvtColor(IMG[sid][1], cv2.COLOR_GRAY2BGR)
    cv2.putText(t_, f"truth {S.loc[sid, 'truth']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(t_)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
cv2.imwrite(f"{OUT}/ours_side_sheet.png", sheet)
json.dump(rep, open(f"{OUT}/side_report.json", "w"), indent=1)
log("done", json.dumps(rep))

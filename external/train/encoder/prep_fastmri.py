"""V4 data: fastMRI single-coil knee scans as extra Lateral Meniscus examples (docs/intent/c_plan.md).

1. The left/right + upright classifier trained on our own sided scans (Kaggle notebook
   `matthewliew/radiant-kg-side-classifier`, `side_model.pt`, ResNet-18) scores each fastMRI volume's central
   coronal slices, rendered exactly as in that notebook (0.4 mm/px, 384 px canvas, slices n//2-2, n//2, n//2+2,
   resized to 256). The up/down choice is decided once for the whole set by majority (the reconstructions share one
   pipeline); the left-right choice per volume, keeping only volumes the classifier is confident about.
2. Each fastMRI volume is also resampled into the V3 grid (0.3 mm/px, 512 px, percentile window) for training and
   flipped into the canonical right-knee orientation.
3. Labels from fastMRI+ boxes (drawn on the 320 px reconstruction flipped up/down only, per fastMRI+
   `example.ipynb`), mapped through the same left-right flip. In the canonical frame image columns run patient
   right -> left, so for a right-looking knee the lateral compartment is the low-column half. Lateral Meniscus = 1
   when a "Meniscus Tear" / "Displaced Meniscal Tissue" box centre lies in the lateral half (outside a central
   no-call band), 0 when the volume was reviewed and has no meniscus box in the lateral half or the band. Every
   other finding is left empty (masked in training).
Writes <work>/cache512_train/fm_<file>.npy, <work>/fm_meta.csv, <work>/fm_series.csv, <work>/fm_labels.csv,
<work>/fm_sides.csv and <work>/fm_audit.json.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import timm
import torch

BAND = 0.06  # no-call band around the image midline, fraction of width
TEAR = {"Meniscus Tear", "Displaced Meniscal Tissue"}


def canvas(rec, mm, size):
    """reconstruction_esc [z, 320, 320] over 140 mm -> uint8 [z, size, size] at `mm` per pixel, centred."""
    n = int(round(140.0 / mm))
    out = np.zeros((len(rec), size, size), np.float32)
    o = (size - n) // 2
    for i, s in enumerate(rec):
        out[i, o:o + n, o:o + n] = cv2.resize(s.astype(np.float32), (n, n), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(out[:, ::4, ::4], [1, 99.5])
    return (np.clip((out - lo) / max(hi - lo, 1e-9), 0, 1) * 255).round().astype(np.uint8)


def side_input(rec):
    """Exactly the side notebook's fastMRI rendering (fm_canonical)."""
    v = canvas(rec, 0.4, 384)
    k = len(v)
    return np.stack([cv2.resize(v[min(max(j, 0), k - 1)], (256, 256), interpolation=cv2.INTER_AREA)
                     for j in (k // 2 - 2, k // 2, k // 2 + 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--side-model", required=True)
    ap.add_argument("--min-conf", type=float, default=0.9, help="keep volumes with |p - 0.5| * 2 at least this")
    ap.add_argument("--tmp", default="/workspace/fm_tmp")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model("resnet18", pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(a.side_model, map_location="cpu"))
    m.eval().to(dev)
    boxes = pd.read_csv(a.boxes)
    boxes["label"] = boxes["label"].str.strip()
    reviewed = set(pd.read_csv(a.reviewed, header=None)[0])
    os.makedirs(a.tmp, exist_ok=True)
    os.makedirs(f"{a.work}/cache512_train", exist_ok=True)
    rows = []
    for tar in a.tars:
        with tarfile.open(tar, "r|xz") as tf:
            for mbr in tf:
                if not mbr.name.endswith(".h5"):
                    continue
                tf.extract(mbr, a.tmp)
                p = f"{a.tmp}/{mbr.name}"
                fid = Path(mbr.name).stem
                try:
                    with h5py.File(p) as h:
                        rec = h["reconstruction_esc"][()]
                        acq = str(h.attrs.get("acquisition", ""))
                    x = side_input(rec)
                    batch = np.stack([x, x[:, ::-1].copy()]).astype(np.float32) / 255
                    with torch.no_grad():
                        q = torch.sigmoid(m(torch.from_numpy(batch).to(dev))).cpu().numpy()
                    np.save(f"{a.work}/cache512_train/fm_{fid}.npy", canvas(rec, 0.3, 512))
                    rows.append({"file": fid, "fs": int("FS" in acq), "n": len(rec),
                                 "p_right": float(q[0, 0]), "p_upright": float(q[0, 1]),
                                 "p_right_upside_down": float(q[1, 0]), "p_upright_upside_down": float(q[1, 1])})
                except Exception as e:  # one bad file never stops the run
                    rows.append({"file": fid, "error": repr(e)[:200]})
                os.remove(p)
                if len(rows) % 100 == 0:
                    print("converted", len(rows), flush=True)
    R = pd.DataFrame(rows).set_index("file")
    ok = R[R["error"].isna()] if "error" in R else R
    vote = float((ok.p_upright > 0.5).mean())
    vflip = vote < 0.5
    pr = ok.p_right_upside_down if vflip else ok.p_right
    conf = (pr - 0.5).abs() * 2
    keep = conf >= a.min_conf
    hflip = pr < 0.5  # looks like a left knee: mirror it to look like a right knee
    ok = ok.assign(p_right_used=pr, conf=conf, keep=keep, mirror=hflip)
    ok.to_csv(f"{a.work}/fm_sides.csv")
    labels, series = [], []
    for fid in ok.index[keep]:
        if fid not in reviewed:
            continue
        b = boxes[(boxes.file == fid) & boxes.label.isin(TEAR)]
        cx = (b.x + b.width / 2) / 320.0
        if hflip[fid]:
            cx = 1 - cx
        lateral = bool((cx < 0.5 - BAND).any())
        band = bool(((cx >= 0.5 - BAND) & (cx <= 0.5 + BAND)).any())
        if not lateral and band:
            continue  # cannot tell the side of that tear: leave the file out
        v = np.load(f"{a.work}/cache512_train/fm_{fid}.npy")
        v = v[:, ::-1] if vflip else v
        v = v[:, :, ::-1] if hflip[fid] else v
        np.save(f"{a.work}/cache512_train/fm_{fid}.npy", np.ascontiguousarray(v))
        labels.append({"StudyInstanceUID": f"fm_{fid}", "lm": 1.0 if lateral else 0.0})
        series.append({"StudyInstanceUID": f"fm_{fid}", "SeriesInstanceUID": f"fm_{fid}", "Fluid_Sensitive": 1,
                       "Fat_Suppression": int(ok.loc[fid, "fs"]), "Anatomical_Plane": "Coronal"})
    pd.DataFrame(series).to_csv(f"{a.work}/fm_series.csv", index=False)
    pd.DataFrame(labels).to_csv(f"{a.work}/fm_labels.csv", index=False)
    pd.DataFrame({"series": [s["SeriesInstanceUID"] for s in series],
                  "n": [int(ok.loc[s["SeriesInstanceUID"][3:], "n"]) for s in series]}).to_csv(f"{a.work}/fm_meta.csv", index=False)
    audit = {"converted": int(len(ok)), "errors": int(len(R) - len(ok)), "upright_vote_as_stored": vote,
             "flipped_upside_down": bool(vflip), "confident": int(keep.sum()), "min_conf": a.min_conf,
             "mirrored_of_confident": int((hflip & keep).sum()), "labelled": len(labels),
             "lm_positive": int(sum(r["lm"] for r in labels))}
    json.dump(audit, open(f"{a.work}/fm_audit.json", "w"), indent=1)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()

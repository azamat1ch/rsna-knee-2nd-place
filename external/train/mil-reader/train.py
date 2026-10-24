"""Train mil-reader (Matthew's V7.2) on all 4,403 labelled exams; nothing is held out.

  python train.py --labels work/labels_D.csv --data data --work work --views work/views_train.csv \
      --laterality work/laterality_v2.csv --init-enc work/fm_encoder7.pt --seed 0 --out runs/v72full_s0

Writes <out>/fold-1.pt (EMA weights + args, read by external/reader/infer_c.py) and log.txt; a per-epoch
state.pt lets an interrupted run resume where it stopped.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import loader_settings, masked_bce, seed_epoch  # noqa: E402
from knee import LABELS, KneeNet, StudyDataset, build_study_table  # noqa: E402

LABELS_D_SHA256 = "aabca6154133502d6d087ec3190e543652b5b1c1c344f4160a93817df1867e42"
N_EXAMS = 4403  # 3,906 + 439 on Team D labels, 58 on their official labels


def final_fit_table(lab):
    """Every exam in labels_D.csv, in file order; the 58 gold exams train on their gold_<finding> labels."""
    tr = lab.copy()
    gold = tr.index[tr.is_gold == 1]
    tr.loc[gold, LABELS] = tr.loc[gold, [f"gold_{c}" for c in LABELS]].to_numpy()
    if len(tr) != N_EXAMS or not tr.index.is_unique or tr[LABELS].isna().any().any():
        raise ValueError("labels_D.csv is not the 4,403-exam final-fit table")
    return tr


def param_groups(model, lr, lr_head):
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    return [{"params": list(model.enc.parameters()), "lr": lr}, {"params": head, "lr": lr_head}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="labels_D.csv (make_targets.py --variant D)")
    ap.add_argument("--data", required=True, help="competition folder (train_series.csv)")
    ap.add_argument("--work", required=True, help="holds cache384_train/ and cache384_train_meta.csv")
    ap.add_argument("--views", required=True, help="close-up boxes (build_views.py)")
    ap.add_argument("--laterality", required=True, help="study,side; side L is mirrored to look like R")
    ap.add_argument("--init-enc", required=True, help="fm_encoder7.pt (encoder/pretrain_fastmri.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_eval", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--inference-dtype", default="bfloat16")
    ap.add_argument("--unpinned", action="store_true", help="accept a labels file other than ours (skip the sha256 check)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(json.dumps(vars(args)))
    if not args.unpinned and hashlib.sha256(Path(args.labels).read_bytes()).hexdigest() != LABELS_D_SHA256:
        raise ValueError(f"{args.labels} is not the pinned labels_D.csv (--unpinned skips this check)")
    tr = final_fit_table(pd.read_csv(args.labels, index_col=0))
    meta = pd.read_csv(f"{args.work}/cache384_train_meta.csv")
    series = pd.read_csv(f"{args.data}/train_series.csv")
    table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0))))
    vb = pd.read_csv(args.views)
    vb = vb[vb.view != "anchor"]
    views = {}
    for sr, v, r0, c0 in zip(vb.series, vb.view, vb.r0, vb.c0):
        views.setdefault(sr, {})[v] = (int(r0), int(c0))
    side = pd.read_csv(args.laterality).set_index("study")["side"]
    mirror = set(side.index[side == "L"])
    say(f"train {len(tr)}, close-ups for {len(views)} series, mirroring {len(mirror & set(tr.index))} exams")

    dl_kw = loader_settings(args.workers)
    train_ds = StudyDataset(tr.index, table, f"{args.work}/cache384_train", tr[LABELS].values.astype(np.float32),
                            args.k, train=True, mirror=mirror, views=views)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs // args.accum, shuffle=True, drop_last=True,
                                           **dl_kw)
    os.environ["BRANCH_B_GRAD_CKPT"] = "1"  # gradient checkpointing: same numbers, less memory
    model = KneeNet(args.backbone, pretrained=False, spatial=True, views=True)
    model.enc.load_state_dict(torch.load(args.init_enc, map_location="cpu"))
    model = model.cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.lr_head), weight_decay=args.wd)
    steps = args.epochs * (len(train_dl) // args.accum)
    warm = len(train_dl) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))

    state_path = f"{args.out}/state.pt"
    step, start_ep = 0, 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, start_ep = st["step"], st["epoch"] + 1
        torch.set_rng_state(st["rng"]["torch"].cpu())
        np.random.set_state(st["rng"]["numpy"])
        torch.cuda.set_rng_state_all([x.cpu() for x in st["rng"]["cuda"]])
        say(f"resumed after epoch {st['epoch']}")
    for ep in range(start_ep, args.epochs):
        seed_epoch(dl_kw, args.seed, ep)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for x, mask, pos, y in train_dl:
            x, mask, pos, y = x.cuda(non_blocking=True), mask.cuda(), pos.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, mask, pos, res=args.res, train=True)
            loss = masked_bce(logit, y)
            (loss / args.accum).backward()
            tot += loss.item()
            n += 1
            if n % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        say(f"ep {ep:2d} loss {tot / n:.4f} time {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "epoch": ep,
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "cuda": torch.cuda.get_rng_state_all()}}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    torch.save({"model": ema.state_dict(), "args": {**vars(args), "spatial": True}}, f"{args.out}/fold-1.pt")
    os.remove(state_path)


if __name__ == "__main__":
    main()

"""Train pf-specialist (Matthew's E3, V7.3.1) on all 4,403 labelled exams plus OAI rows; nothing is held out.

  python train.py --config e3.json --seed 0 --out runs/E3_s0 --labels work/labels_D.csv \
      --d4 work/d4-train-4345.csv --data data --work work --laterality work/laterality_v2.csv \
      --native-root work/native --encoder work/fm_encoder7.pt --oai-grades work/oai/grades.csv \
      --oai-native work/oai/native

Competition exams train PF OA: the 4,345 on D4 (our expert-pf-floor table), the 58 on their official label.
OAI knee visits (oai_join.py, oai_inputs.py) train the kneecap-cartilage grade head:
one OAI row after every 4 competition rows. Writes <out>/final_ema.pt (read by external/v731/kaggle_infer.py)
and log.txt; a per-epoch state.pt lets an interrupted run resume where it stopped.
"""

from __future__ import annotations

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
from knee import build_study_table  # noqa: E402

from v731.dataset import FIELDS, NativeStore, PFDataset, collate, load_k_series, to_device  # noqa: E402
from v731.model import FM_ENCODER7_SHA256, build_model, grade_loss, load_encoder, param_groups  # noqa: E402

D4_SHA256 = "5e3f6797be07300ce3584715ac31e43c0a2c602947040a4d6443cee72771b7e4"
LABELS_D_SHA256 = "aabca6154133502d6d087ec3190e543652b5b1c1c344f4160a93817df1867e42"
COUNTS = (3906, 439, 58)  # labels_D folds 0-4, the old holdout (fold -2), the gold exams


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_fit_labels(labels_d, d4, pinned=True):
    """(exam ids, PF OA targets): D4 for folds 0-4 then fold -2 (file order), then the 58 gold exams' gold_PF OA."""
    for path, want in ((labels_d, LABELS_D_SHA256), (d4, D4_SHA256)):
        if pinned and sha256(path) != want:
            raise ValueError(f"{path} is not the pinned file (--unpinned skips this check)")
    lab = pd.read_csv(labels_d, usecols=["StudyInstanceUID", "is_gold", "fold", "gold_PF OA"],
                      dtype={"StudyInstanceUID": str})
    weak, gold = lab[lab.is_gold == 0], lab[lab.is_gold == 1].set_index("StudyInstanceUID")["gold_PF OA"]
    folds, old_holdout = weak.StudyInstanceUID[weak.fold >= 0].tolist(), weak.StudyInstanceUID[weak.fold == -2].tolist()
    if (len(folds), len(old_holdout), len(gold)) != COUNTS:
        raise ValueError("labels_D is not the 4,403-exam final-fit table")
    ids = folds + old_holdout
    y = pd.read_csv(d4, usecols=["StudyInstanceUID", "PF OA"], dtype={"StudyInstanceUID": str})
    y = y.set_index("StudyInstanceUID")["PF OA"].astype(np.float64)
    if set(y.index) != set(ids) or not y.index.is_unique:
        raise ValueError("D4 exams differ from labels_D's non-gold exams")
    return ids + list(gold.index), pd.concat([y.loc[ids], gold.astype(np.float64)])


def check_gold_native(store, series, gold, root):
    """Stop when prepare_native.py was not also run with --gold: the 58 would train with no native views."""
    if store.ok is None:
        return
    built = {s for _, s in store.ok}
    have = set(series.StudyInstanceUID[series.SeriesInstanceUID.isin(built)])
    if not have & set(gold):
        raise SystemExit(f"{root}: none of the 58 gold exams has native arrays; run prepare_native build --gold into it")


class MixedDataset(torch.utils.data.Dataset):
    """Competition rows (index < n_comp), then OAI rows."""

    def __init__(self, comp, oai):
        self.comp, self.oai, self.n_comp = comp, oai, len(comp)

    def __len__(self):
        return self.n_comp + len(self.oai)

    def __getitem__(self, i):
        return self.comp[i] if i < self.n_comp else self.oai[i - self.n_comp]


class PFBatchSampler(torch.utils.data.Sampler):
    """Micro-batches of competition rows in DataLoader(shuffle=True, drop_last=True) order, each followed by
    `k_oai` OAI rows drawn with replacement from a numpy Generator seeded by (seed, epoch)."""

    def __init__(self, n_comp, batch_size, generator, seed, n_oai, k_oai):
        self.n_comp, self.bs, self.generator, self.seed = n_comp, batch_size, generator, seed
        self.n_oai, self.k_oai = n_oai, k_oai
        w = np.ones(n_oai)
        self.p = w / w.sum()
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_comp // self.bs

    def __iter__(self):
        comp = list(torch.utils.data.RandomSampler(range(self.n_comp), generator=self.generator))
        rng = np.random.default_rng([self.seed, self.epoch, 0x731])
        for i in range(len(self)):
            batch = comp[i * self.bs : (i + 1) * self.bs]
            yield batch + (self.n_comp + rng.choice(self.n_oai, size=self.k_oai, p=self.p)).tolist()


def compute_loss(out, batch, lambda_grade):
    """Competition rows: BCE on pf_direct and pf_combined; OAI rows (y_pf NaN, masked there): the grade loss."""
    y = batch["y_pf"]
    return (masked_bce(out["pf_direct"], y) + masked_bce(out["pf_combined"], y)
            + grade_loss(out["grade_logits"], batch["y_grades"]) * lambda_grade)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="e3.json")
    ap.add_argument("--seed", type=int, required=True, help="0, 1 or 2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", required=True, help="labels_D.csv (split table and the 58's official labels)")
    ap.add_argument("--d4", required=True, help="d4-train-4345.csv (our expert-pf-floor.csv)")
    ap.add_argument("--data", required=True, help="competition folder (train_series.csv)")
    ap.add_argument("--work", required=True, help="holds cache384_train/ and cache384_train_meta.csv")
    ap.add_argument("--laterality", required=True)
    ap.add_argument("--native-root", required=True, help="prepare_native.py output (manifest.csv, P/ Z/ K/, k_series.csv)")
    ap.add_argument("--encoder", required=True, help="fm_encoder7.pt")
    ap.add_argument("--oai-grades", required=True, help="oai_join.py grade table")
    ap.add_argument("--oai-native", required=True, help="oai_inputs.py prepare output")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unpinned", action="store_true",
                    help="accept labels and an encoder other than ours (skip the sha256 checks)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["recipe"]["seed"] = a.seed
    r = cfg["recipe"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU: smoke tests only
    from oai_inputs import OAISource

    oai = OAISource(a.oai_grades, a.oai_native, k=r["k"], k_eval=r["k_eval"])
    ids, y = final_fit_labels(a.labels, a.d4, pinned=not a.unpinned)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    torch.manual_seed(r["seed"])
    np.random.seed(r["seed"])
    series = pd.read_csv(f"{a.data}/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    meta = pd.read_csv(f"{a.work}/cache384_train_meta.csv", dtype={"series": str})
    table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0), strict=True)))
    side = pd.read_csv(a.laterality, dtype={"study": str}).set_index("study")["side"]
    groups = cfg["groups"]
    store = NativeStore.from_root(a.native_root)
    check_gold_native(store, series, ids[-COUNTS[2]:], a.native_root)
    train_ds = PFDataset(ids, table, f"{a.work}/cache384_train", store,
                         y.loc[ids].to_numpy(np.float32), k=r["k"], train=True, mirror=set(side.index[side == "L"]),
                         k_series=load_k_series(f"{a.native_root}/k_series.csv"), groups=groups)
    oai_train = oai.train_items()
    micro = r["bs"] // r["accum"]
    dl_kw = loader_settings(a.workers)
    # 0.2 mm 600 px windows make big batches: no pinned staging, one batch in flight per worker (memory only)
    dl_kw["pin_memory"] = False
    if a.workers > 0:
        dl_kw["prefetch_factor"] = 1
    k_oai = int(micro * cfg["oai"]["oai_per_competition_row"])
    sampler = PFBatchSampler(len(train_ds), micro, dl_kw["generator"], r["seed"], len(oai_train), k_oai)
    train_dl = torch.utils.data.DataLoader(MixedDataset(train_ds, oai_train), batch_sampler=sampler,
                                           collate_fn=collate, **dl_kw)
    say(f"seed {r['seed']}: {len(train_ds)} competition exams, {len(oai_train)} OAI rows ({k_oai} per micro-batch)")

    model = build_model(cfg)
    enc_sha = load_encoder(model, a.encoder, None if a.unpinned else FM_ENCODER7_SHA256)
    model = model.to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, r["lr_enc"], r["lr_head"]), weight_decay=r["wd"])
    per_epoch = len(train_dl) // r["accum"]
    steps = r["epochs"] * per_epoch
    warm = max(1, r["warmup_epochs"] * per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))

    state_path = out / "state.pt"
    step, start_ep = 0, 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, start_ep = st["step"], st["epoch"] + 1
        torch.set_rng_state(st["rng"]["torch"].cpu())
        np.random.set_state(st["rng"]["numpy"])
        torch.cuda.set_rng_state_all([x.cpu() for x in st["rng"]["cuda"]])
        say(f"resumed after epoch {st['epoch']}")
    for ep in range(start_ep, r["epochs"]):
        seed_epoch(dl_kw, r["seed"], ep)
        sampler.set_epoch(ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, n, tot = time.time(), 0, 0.0
        for batch in train_dl:
            b = to_device(batch, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = model(b, train=True)
            loss = compute_loss(o, b, cfg["oai"]["lambda_grade"])
            (loss / r["accum"]).backward()
            tot += float(loss.detach())
            n += 1
            if n % r["accum"]:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), r["clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            d = min(r["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers(), strict=True):
                    be.copy_(bm)
        say(f"ep {ep:2d} loss {tot / max(n, 1):.4f} time {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "epoch": ep,
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "cuda": torch.cuda.get_rng_state_all()}}, str(state_path) + ".tmp")
        os.replace(str(state_path) + ".tmp", state_path)

    ck = {"model": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "config": cfg, "fields": list(FIELDS),
          "groups": list(groups), "encoder_sha256": enc_sha, "args": {"variant": cfg["variant"], "seed": r["seed"]}}
    torch.save(ck, out / "final_ema.pt")
    state_path.unlink()


if __name__ == "__main__":
    main()

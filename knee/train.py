"""Train one member seed: `python -m knee.train --recipe configs/X.json --seed S --data DIR --labels DIR --out DIR`.

One loop for both families: pinned start weights, keyed or fixed-seed draws, one optimizer step per batch, an EMA
copy, a resumable checkpoint after every epoch (`--resume` continues it) and the final EMA weights in the format the
inference kernel loads. The family pieces (augmentation, schedule, loss, step rules) repeat the training runs of the
shipped weights; every random draw has its own seeded stream, so a rerun on the same software makes the same draws.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from knee import LABELS, models, sha256

ID = "StudyInstanceUID"
MAX_SKIPS_IN_A_ROW = 30        # tile: non-finite fp16 gradients skipped in a row before the run fails
TILE_AUGMENT = dict(rotate_degrees=10.0, shift_fraction=0.06, zoom_in=0.10, gain=0.10, gamma=(0.8, 1.2), drop_slot=0.12)
WINDOW_STREAMS = dict(torch=1000, augment=5000, shift=9500, data=9000)   # window seeds: base + 17 * seed
WINDOW_STREAM_STEP = 17
WINDOW_ROTATE_DEGREES, WINDOW_SCALE, WINDOW_GAMMA, WINDOW_CONTRAST = 10.0, (0.9, 0.2), (0.8, 0.45), (0.9, 0.2)
WINDOW_EMA_WARM = (1.0, 10.0)  # EMA decay min(ema, (1 + t) / (10 + t)) at optimizer step t
RESUME_FILE, RUN_FILE = "resume.pt", "run.json"


# ---------------------------------------------------------------- shared pieces

def read_labels(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(sorted IDs, float32 targets (N, 12), observed (N, 12) bool); a blank cell is unobserved."""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: row[ID])
    cells = [[row[label] for label in LABELS] for row in rows]
    targets = np.array([[np.nan if c == "" else float(c) for c in row] for row in cells], dtype=np.float32)
    observed = ~np.isnan(targets)
    if np.any((targets[observed] < 0) | (targets[observed] > 1)):
        raise ValueError(f"{path}: targets outside [0, 1]")
    return [row[ID] for row in rows], targets, observed


def start_weights(recipe: dict, path: str | None) -> Path:
    """The encoder's pinned start weights: a local file, or the pinned Hugging Face revision; sha256 checked."""
    pin = recipe["encoder"]["start_weights"]
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=pin["repo"], filename=pin["file"], revision=pin["revision"])
    if sha256(path) != pin["sha256"]:
        raise ValueError(f"{path}: sha256 differs from the recipe's start-weights pin")
    return Path(path)


def autocast(device: torch.device, amp: str):
    """fp16/bf16 autocast on CUDA; plain fp32 elsewhere (CPU checks)."""
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda")


def save_atomic(value, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_states() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])


def versions() -> dict:
    import safetensors
    import timm
    return dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__,
                numpy=np.__version__, safetensors=safetensors.__version__,
                cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


# ---------------------------------------------------------------- tile family

def tile_lr(base: float, step: int, total: int, warmup_fraction: float) -> float:
    """Linear warmup over the first steps, then cosine to zero; set before every batch."""
    warmup = max(1, int(warmup_fraction * total))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return float(base * 0.5 * (1 + np.cos(np.pi * progress)))


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Mean BCE-with-logits over the observed cells only, in fp32."""
    return F.binary_cross_entropy_with_logits(logits[observed].float(), targets[observed].float())


def micro_batches(batch: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous near-equal row ranges of a batch."""
    edges = [round(i * batch / parts) for i in range(parts + 1)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class TileEma:
    """EMA of every float tensor of the state (buffers included); integer buffers are copied. No warmup."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.updates = decay, 0
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.lerp_(current[name].to(value.dtype), 1.0 - self.decay)
            else:
                value.copy_(current[name])
        self.updates += 1


def tile_draw(seed: int, epoch: int, study_id: str, present: list[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot augmentation (angle, shift x, shift y, zoom, gain, gamma) float32 (6, 6) and keep (6,) bool.

    A pure function of (seed, epoch, exam), so worker count and resume do not change it. Now and then one present
    slot is dropped, never the last one.
    """
    a = TILE_AUGMENT
    digest = hashlib.sha256(json.dumps([seed, epoch, "augment-v1", study_id]).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    params = np.empty((models.N_SLOTS, 6), dtype=np.float32)
    for i in range(models.N_SLOTS):   # the same draws for every slot, present or not
        angle = math.radians(rng.uniform(-a["rotate_degrees"], a["rotate_degrees"]))
        shift = rng.uniform(-a["shift_fraction"], a["shift_fraction"], size=2)
        zoom = 1.0 + rng.uniform(0.0, a["zoom_in"])
        gain = 1.0 + rng.uniform(-a["gain"], a["gain"])
        gamma = math.exp(rng.uniform(math.log(a["gamma"][0]), math.log(a["gamma"][1])))
        params[i] = (angle, shift[0], shift[1], zoom, gain, gamma)
    slots = [i for i, here in enumerate(present) if here]
    keep = np.array(present, dtype=bool)
    drop, pick = rng.random(), rng.integers(max(1, len(slots)))
    if len(slots) >= 2 and drop < a["drop_slot"]:
        keep[slots[int(pick)]] = False
    return params, keep


@torch.no_grad()
def tile_augment(images: torch.Tensor, mask: torch.Tensor, params: torch.Tensor, mean, std) -> torch.Tensor:
    """Resample the used views of normalised (B, V, 3, H, W) images: rotation, shift and zoom-in (bilinear, zero
    fill), then gamma and gain in unit range, then normalised again. params (B, 6, 6) from tile_draw."""
    batch, views = mask.shape
    with torch.autocast(images.device.type, enabled=False):
        mean = torch.as_tensor(mean, dtype=torch.float32, device=images.device)[:, None, None]
        std = torch.as_tensor(std, dtype=torch.float32, device=images.device)[:, None, None]
        slot = torch.arange(views, device=images.device) // (views // models.N_SLOTS)
        p = params.to(images.device, torch.float32)[:, slot][mask]                  # (N, 6)
        unit = images[mask].float() * std + mean
        cos, sin = torch.cos(p[:, 0]) / p[:, 3], torch.sin(p[:, 0]) / p[:, 3]
        theta = torch.stack([torch.stack([cos, -sin, 2 * p[:, 1]], dim=1),
                             torch.stack([sin, cos, 2 * p[:, 2]], dim=1)], dim=1)   # shift: a fraction of the size
        grid = F.affine_grid(theta, list(unit.shape), align_corners=False)
        out = F.grid_sample(unit, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        gain, gamma = p[:, 4, None, None, None], p[:, 5, None, None, None]
        out = (out.clamp(0, 1) ** gamma * gain).clamp(0, 1)
        result = images.clone()
        result[mask] = ((out - mean) / std).to(images.dtype)
    return result


class TileTrainItems(torch.utils.data.Dataset):
    """Training items of a tile dataset plus targets and the augmentation draw (a dropped slot loses its views in
    both branches; a slot the detail branch lacks keeps its context views). Keys (epoch, index)."""

    def __init__(self, data, ids: list[str], targets: np.ndarray, observed: np.ndarray, seed: int):
        self.data, self.ids, self.targets, self.observed, self.seed = data, ids, targets, observed, seed

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key):
        epoch, index = key
        item = dict(self.data[key])
        per_slot = len(item["view_mask"]) // models.N_SLOTS
        if item["study_id"] != self.ids[index]:
            raise ValueError("the dataset's exam order differs from the label table's")
        present = [bool(b) for b in item["slot_mask"]]
        params, keep = tile_draw(self.seed, epoch, item["study_id"], present)
        item["view_mask"] = np.asarray(item["view_mask"]) & np.repeat(keep, per_slot)
        if "context_mask" in item:
            spare = keep | ~np.array(present)
            item["context_mask"] = np.asarray(item["context_mask"]) & np.repeat(spare, len(item["context_mask"]) // models.N_SLOTS)
        item.update(augment=params, targets=self.targets[index], observed=self.observed[index])
        return item


TILE_KEYS = ("images", "view_mask", "view_position", "context_images", "context_mask", "context_position",
             "augment", "targets", "observed")
TILE_INPUTS = ("view_position", "context_images", "context_mask", "context_position")


def tile_collate(items: list[dict]) -> dict:
    return {k: torch.stack([torch.as_tensor(np.asarray(x[k])) for x in items]) for k in TILE_KEYS if k in items[0]}


def tile_step(model, batch: dict, optimizer, scaler, ema: TileEma, parts: int, device, amp: str) -> float | None:
    """One optimizer update over micro-batches, each piece weighted by its share of the observed cells.
    Returns the batch loss, or None when the update was skipped (no observed cell, or non-finite fp16 gradients)."""
    observed = batch["observed"]
    count = int(observed.sum())
    if count == 0:
        return None
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for a, b in micro_batches(len(observed), parts):
        part = int(observed[a:b].sum())
        if part == 0:
            continue
        inputs = {k: batch[k][a:b].to(device, non_blocking=True) for k in TILE_INPUTS if k in batch}
        with autocast(device, amp):
            logits = model(batch["images"][a:b].to(device, non_blocking=True),
                           batch["view_mask"][a:b].to(device, non_blocking=True), **inputs)
            piece = masked_bce(logits, batch["targets"][a:b].to(device), observed[a:b].to(device))
        if not torch.isfinite(piece):
            raise FloatingPointError("non-finite loss; no update made")
        weighted = piece * (part / count)
        scaler.scale(weighted).backward()
        total += float(weighted.detach())
    scaler.unscale_(optimizer)
    grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not bool(torch.stack([g.isfinite().all() for g in grads]).all()):
        if not scaler.is_enabled():
            raise FloatingPointError("non-finite gradient; no update made")
        scaler.step(optimizer)   # finds the inf and skips the update
        scaler.update()          # and lowers the scale
        optimizer.zero_grad(set_to_none=True)
        return None
    scaler.step(optimizer)
    scaler.update()
    ema.update(model)
    return total


def train_tile(recipe: dict, seed: int, data: Path, ids, targets, observed, out: Path, start: Path,
               resume: bool, workers: int, device: torch.device) -> dict:
    from knee import tiles
    t = recipe["train"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = models.build(recipe, start).to(device)
    frozen = models.frozen_batchnorm(model.encoder) if recipe["inputs"].get("frozen_batchnorm") else []
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=t["grad_scale"], enabled=t["amp"] == "fp16" and device.type == "cuda")
    ema = TileEma(model, t["ema"])
    items = TileTrainItems(tiles.training_dataset(data, recipe, ids, seed, workers=workers), ids, targets, observed, seed)
    sampler = tiles.EpochSampler(len(ids), [seed, "all"])
    loader = torch.utils.data.DataLoader(items, batch_size=t["batch"], sampler=sampler, num_workers=workers,
                                         persistent_workers=workers > 0, collate_fn=tile_collate,
                                         pin_memory=device.type == "cuda")
    steps = -(-len(ids) // t["batch"])
    total = steps * t["epochs"]
    mean, std = recipe["encoder"]["mean"], recipe["encoder"]["std"]
    record = dict(epochs=[], updates=0, skipped=0)
    begin = 0
    if resume:
        state = torch.load(out / RESUME_FILE, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        ema.module.load_state_dict(state["ema"])
        ema.updates, record, begin = state["ema_updates"], state["record"], state["epoch"]
        set_rng_states(state["rng"])
    for epoch in range(begin, t["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        for module in frozen:
            module.eval()
        loss_sum, cells, in_a_row = 0.0, 0, 0
        for index, batch in enumerate(loader):
            for group in optimizer.param_groups:
                group["lr"] = tile_lr(t["lr"], epoch * steps + index, total, t["warmup"])
            params = batch.pop("augment")
            for images, mask in (("images", "view_mask"), ("context_images", "context_mask")):
                if images in batch:
                    batch[mask] = batch[mask].to(device, non_blocking=True)
                    batch[images] = tile_augment(batch[images].to(device, non_blocking=True), batch[mask], params, mean, std)
            loss = tile_step(model, batch, optimizer, scaler, ema, t["micro_batches"], device, t["amp"])
            if loss is None:
                if batch["observed"].any():
                    in_a_row += 1
                    record["skipped"] += 1
                    if in_a_row > MAX_SKIPS_IN_A_ROW:
                        raise FloatingPointError("fp16 gradients non-finite too many times in a row")
                continue
            in_a_row = 0
            count = int(batch["observed"].sum())
            loss_sum, cells = loss_sum + loss * count, cells + count
            record["updates"] += 1
        record["epochs"].append(round(loss_sum / max(1, cells), 5))
        print(json.dumps(dict(epoch=epoch + 1, loss=record["epochs"][-1], updates=record["updates"])), flush=True)
        save_atomic(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         ema=ema.module.state_dict(), ema_updates=ema.updates, record=record, epoch=epoch + 1,
                         rng=rng_states()), out / RESUME_FILE)
    final = out / f"seed-{seed}.pt"
    save_atomic({"format": models.TILE_FORMAT, "recipe": recipe["name"], "seed": seed, "completed_epoch": t["epochs"],
                 "weights_only": f"EMA of the model weights, decay {ema.decay}, {ema.updates} updates",
                 "states": {"model": ema.module.state_dict()}}, final)
    return dict(record, weights=final.name)


# ---------------------------------------------------------------- window family

def window_lr_lambda(total: int, warm: int):
    """LambdaLR factor at optimizer step s: linear warmup, then cosine to zero."""
    return lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))


def window_bce(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits summed over the batch per finding, divided by the batch size, then the mean over findings."""
    return (F.binary_cross_entropy_with_logits(z, target, reduction="none").sum(0) / len(z)).mean()


@torch.no_grad()
def window_ema_update(ema: dict, model: torch.nn.Module, step: int, decay: float) -> None:
    d = min(decay, (WINDOW_EMA_WARM[0] + step) / (WINDOW_EMA_WARM[1] + step))
    for name, p in model.named_parameters():
        if name in ema:
            ema[name].mul_(d).add_(p.detach(), alpha=1 - d)


def window_augment(img: torch.Tensor, crop: int, shift_px: int, gen: torch.Generator,
                   shift_gen: torch.Generator | None) -> torch.Tensor:
    """uint8 (B, W, 3, S, S) -> uint8 (B, W, 3, crop, crop). Draw order: crop offset (gen), batch shift (shift_gen),
    rotation and scale per exam (gen), gamma and contrast per exam (gen). Per-exam values apply to all its windows."""
    batch, windows, channels, size, _ = img.shape
    oy = int(torch.randint(0, size - crop + 1, (1,), generator=gen))
    ox = int(torch.randint(0, size - crop + 1, (1,), generator=gen))
    img = img[..., oy:oy + crop, ox:ox + crop]
    if shift_gen is not None:   # one integer translation for the batch, zero fill: out[y, x] = in[y + dy, x + dx]
        n = shift_px
        dy = int(torch.randint(-n, n + 1, (1,), generator=shift_gen))
        dx = int(torch.randint(-n, n + 1, (1,), generator=shift_gen))
        img = F.pad(img, (n, n, n, n))[..., n + dy:n + dy + crop, n + dx:n + dx + crop]
    x = img.float().reshape(batch * windows, channels, crop, crop)
    angle = (torch.rand(batch, generator=gen) * (2 * WINDOW_ROTATE_DEGREES) - WINDOW_ROTATE_DEGREES) * math.pi / 180
    scale = torch.rand(batch, generator=gen) * WINDOW_SCALE[1] + WINDOW_SCALE[0]
    cos, sin, zero = torch.cos(angle) / scale, torch.sin(angle) / scale, torch.zeros(batch)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    theta = theta.repeat_interleave(windows, 0).to(x.device)
    with torch.autocast(x.device.type, enabled=False):   # the grid in fp32 even inside the training autocast
        grid = F.affine_grid(theta.float(), x.shape, align_corners=False)
        x = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = (torch.rand(batch, generator=gen) * WINDOW_GAMMA[1] + WINDOW_GAMMA[0]).repeat_interleave(windows, 0)
    contrast = (torch.rand(batch, generator=gen) * WINDOW_CONTRAST[1] + WINDOW_CONTRAST[0]).repeat_interleave(windows, 0)
    x = (x / 255.).clamp_(0, 1).pow(gamma.to(x.device).view(-1, 1, 1, 1))
    x = ((x - 0.5) * contrast.to(x.device).view(-1, 1, 1, 1) + 0.5).clamp_(0, 1) * 255.
    return x.round_().to(torch.uint8).reshape(batch, windows, channels, crop, crop)


def window_step(model, batch, target, optimizer, scaler, schedule, ema: dict, streams, recipe: dict, device) -> float | None:
    """Augment, forward, loss, clipped step, schedule and EMA for one batch (img, meta, mask); None = non-finite
    loss, nothing changed."""
    t, inputs = recipe["train"], recipe["inputs"]
    img, meta, mask = batch
    with autocast(device, t["amp"]):
        img = window_augment(img.to(device, non_blocking=True), inputs["crop"], inputs["shift_px"], *streams)
        z = model(img, meta.to(device), mask.to(device))
    loss = window_bce(z, target)
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss.detach()):
        return None
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip_grad"])
    scaler.step(optimizer)
    scaler.update()
    schedule.step()
    window_ema_update(ema, model, schedule.last_epoch, t["ema"])
    return float(loss.detach())


class NonceSampler(torch.utils.data.RandomSampler):
    """A shuffle that also yields each draw's position in the epoch: (index, nonce) keys for keyed window draws."""

    def __iter__(self):
        for position, index in enumerate(super().__iter__()):
            yield int(index), position


def window_stream(name: str, seed: int) -> int:
    return WINDOW_STREAMS[name] + WINDOW_STREAM_STEP * seed


def train_window(recipe: dict, seed: int, data: Path, ids, targets, observed, out: Path, start: Path,
                 resume: bool, workers: int, device: torch.device) -> dict:
    from knee import windows
    t, inputs = recipe["train"], recipe["inputs"]
    if not observed.all():
        raise ValueError("window training needs every cell observed")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(window_stream("torch", seed))
    gen = torch.Generator().manual_seed(window_stream("augment", seed))
    shift_gen = torch.Generator().manual_seed(window_stream("shift", seed)) if inputs["shift_px"] > 0 else None
    model = models.build(recipe, start).to(device)
    torch.manual_seed(window_stream("torch", seed))   # dropout and drop path
    model.enc.set_grad_checkpointing(True)
    # targets as the training table stored them (6 significant digits); the prior starts the bias
    target = torch.from_numpy(np.array([[float(f"{v:.6g}") for v in row] for row in targets], dtype=np.float32)).to(device)
    with torch.no_grad():
        model.bias.copy_(torch.logit(target.mean(0).clamp(1e-3, 1 - 1e-3)))
    frozen = models.frozen_batchnorm(model.enc) if t["frozen_batchnorm"] else []
    encoder = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": t["lr"]}, {"params": head, "lr": t["head_lr"]}],
                                  weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] == "fp16" and device.type == "cuda")
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    data_seed = window_stream("data", seed)
    dataset = windows.training_dataset(data, recipe, ids, data_seed)
    order_gen = torch.Generator().manual_seed(data_seed)   # per epoch: the worker base seed, then the permutation
    loader = torch.utils.data.DataLoader(dataset, batch_size=t["batch"], sampler=NonceSampler(dataset, generator=order_gen),
                                         num_workers=workers, collate_fn=windows.collate, drop_last=True,
                                         pin_memory=device.type == "cuda", generator=order_gen,
                                         **({"prefetch_factor": 2} if workers else {}))
    total = len(loader) * t["epochs"]
    warm = max(1, int(t["warmup"] * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, window_lr_lambda(total, warm))
    record = dict(epochs=[], updates=0, skipped=0)
    begin = 0
    if resume:
        state = torch.load(out / RESUME_FILE, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        ema = {n: v.to(device) for n, v in state["ema"].items()}
        for g, s in ((gen, "gen"), (shift_gen, "shift_gen"), (order_gen, "order_gen")):
            if g is not None:
                g.set_state(state[s])
        record, begin = state["record"], state["epoch"]
        set_rng_states(state["rng"])
    for epoch in range(begin, t["epochs"]):
        dataset.epoch = epoch
        model.train()
        for module in frozen:
            module.eval()
        loss_sum, count = 0.0, 0
        for img, meta, mask, idx in loader:
            loss = window_step(model, (img, meta, mask), target[idx.to(device)], optimizer, scaler, schedule, ema,
                               (gen, shift_gen), recipe, device)
            if loss is None:
                if record["skipped"] >= t["skip_nonfinite"]:
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
                record["skipped"] += 1
                continue
            loss_sum, count = loss_sum + loss * len(idx), count + len(idx)
            record["updates"] += 1
        record["epochs"].append(round(loss_sum / max(1, count), 5))
        print(json.dumps(dict(epoch=epoch + 1, loss=record["epochs"][-1], updates=record["updates"])), flush=True)
        save_atomic(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                         scaler=scaler.state_dict(), ema={n: v.cpu() for n, v in ema.items()}, gen=gen.get_state(),
                         shift_gen=None if shift_gen is None else shift_gen.get_state(), order_gen=order_gen.get_state(),
                         record=record, epoch=epoch + 1, rng=rng_states()), out / RESUME_FILE)
    with torch.no_grad():   # ship the EMA weights
        for name, p in model.named_parameters():
            if name in ema:
                p.copy_(ema[name])
    final = out / f"seed-{seed}.pt"
    save_atomic({k: v.detach().cpu() for k, v in model.state_dict().items()}, final)
    return dict(record, weights=final.name)


# ---------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data", required=True, type=Path, help="the family's training store")
    parser.add_argument("--labels", required=True, type=Path, help="folder holding the recipe's label table")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start-weights", help="local copy of the pinned encoder safetensors (default: download)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="continue from the last finished epoch in --out")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (tile family)
    recipe = json.loads(args.recipe.read_text())
    labels = args.labels / recipe["train"]["labels"]
    ids, targets, observed = read_labels(labels)
    start = start_weights(recipe, args.start_weights)
    args.out.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = {"tile": train_tile, "window": train_window}[recipe["family"]]
    record = train(recipe, args.seed, args.data, ids, targets, observed, args.out, start, args.resume, args.workers, device)
    (args.out / RESUME_FILE).unlink(missing_ok=True)
    run = dict(recipe=recipe["name"], recipe_sha256=sha256(args.recipe), seed=args.seed, labels=labels.name,
               labels_sha256=sha256(labels), exams=len(ids), start_weights_sha256=recipe["encoder"]["start_weights"]["sha256"],
               versions=versions(), **record, weights_sha256=sha256(args.out / record["weights"]))
    (args.out / RUN_FILE).write_text(json.dumps(run, indent=1) + "\n")


if __name__ == "__main__":
    main()

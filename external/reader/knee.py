"""Shared dataset and model code for training and inference."""
import math
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PF_INDEX = LABELS.index("PF OA")  # radiant-kg V8 arm R
FX_INDEX = LABELS.index("Fracture")  # radiant-kg V8.1 fx-bias
PLANES = ["Sagittal", "Coronal", "Axial"]
# slot = plane index + 3 * (1 - fat_suppressed): fat-suppressed planes first
N_SLOTS = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def slot_of(plane, fat_sup):
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


T1_SLOTS = (3, 4)  # radiant-kg F: sagittal and coronal non-fat-suppressed


def build_study_table(series_df, n_slices, t1_series=None):
    """series_df: StudyInstanceUID, SeriesInstanceUID, Fat_Suppression, Anatomical_Plane.
    n_slices: dict series -> cached slice count. Returns {study: [[series ids of slot 0], ...]}
    with each slot's candidates sorted by slice count (largest first).
    radiant-kg F: t1_series (set of str series ids judged T1 by seqtype.is_t1, or None = V7): in slots 3 and 4 only, a
    study with >= 1 T1 candidate keeps only its T1 candidates (same order); otherwise the slot is unchanged."""
    table = {}
    for st, sr, fs, pl in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID,
                              series_df.Fat_Suppression, series_df.Anatomical_Plane):
        if n_slices.get(sr, 0) < 3:
            continue
        table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_slices[s])
        if t1_series is not None:  # radiant-kg F
            for j in T1_SLOTS:
                t1 = [s for s in slots[j] if str(s) in t1_series]
                if t1:
                    slots[j] = t1
    return table


def parse_pf_groups(value):
    """radiant-kg V8 arm R: '0,2,8' (CLI or checkpoint args) -> (0, 2, 8).
    None or '' -> None (V7 behaviour)."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return tuple(int(g) for g in value.split(",") if g.strip())
    return tuple(int(g) for g in value)


def parse_fx_bias(value):
    """radiant-kg V8.1: Fracture attention bias. 'view0=1.0,nonfs=0.5' (CLI or checkpoint args) -> 24 floats, one per
    token group g = slot * 4 + view: +view0 on every whole-knee group (view 0: 0, 4, ..., 20), +nonfs on every group of
    the non-fat-suppressed sagittal and coronal slots (3, 4: groups 12..19), added where both apply.
    None or '' -> None (V7 behaviour)."""
    if value is None or value == "":
        return None
    terms = {}
    for part in str(value).split(","):
        if not part.strip():
            continue
        key, sep, num = part.partition("=")
        key = key.strip()
        if not sep or key not in ("view0", "nonfs") or key in terms:
            raise ValueError(f"fx-bias {value!r}: expected 'view0=X,nonfs=Y' (each key at most once)")
        terms[key] = float(num)
        if not math.isfinite(terms[key]):
            raise ValueError(f"fx-bias {value!r}: {key} is not finite")
    if not terms:
        raise ValueError(f"fx-bias {value!r}: no terms")
    bias = [0.0] * (N_SLOTS * VIEWS_PER_SLOT)
    for g in range(len(bias)):
        slot, view = divmod(g, VIEWS_PER_SLOT)
        if view == 0:
            bias[g] += terms.get("view0", 0.0)
        if slot in FX_NONFS_SLOTS:
            bias[g] += terms.get("nonfs", 0.0)
    return tuple(bias)


def window_centres(n, k, train, lo=0.04, hi=0.96):
    """k slice indices (centres of 3-slice windows) spread over the series."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


PHYS_SCALE_MM = 50.0  # radiant-kg V8 arm P: token input = displacement / 50 mm


def phys_values(dx, c, n, left):
    """radiant-kg V8 arm P. dx: per-cache-slice x displacement (mm, LPS, slice centre minus the study's header anchor,
    cache order before mirroring) or None. c: window centre indices in the volume as fed (mirrored for left knees);
    n: slice count. Returns [k] float32 = d / 50 mm with d positive = lateral after left/right normalisation: right
    knee lateral = patient right = -x, so d = -dx; a left knee is mirrored to look like a right knee, its lateral is
    patient left = +x, so d = +dx, and window index j maps back to cache index n - 1 - j. Missing or mismatched
    geometry (no entry, length != n, non-finite value) gives exactly 0."""
    out = np.zeros(len(c), np.float32)
    if dx is None or len(dx) != n:
        return out
    idx = (n - 1 - np.asarray(c)) if left else np.asarray(c)
    d = np.asarray(dx, np.float64)[idx] * (1.0 if left else -1.0)
    ok = np.isfinite(d)
    out[ok] = (d[ok] / PHYS_SCALE_MM).astype(np.float32)
    return out


def load_phys_table(path):
    """radiant-kg V8 arm P: table from v8prep/build_phys_pos.py (parquet or csv; columns series, cache_idx, dx_mm,
    valid) -> {series: float64 [n] dx in cache order}. Only valid series with a complete 0..n-1 index are kept; every
    other series gets no entry and therefore a zero contribution."""
    import pandas as pd
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df = df[df["valid"].astype(bool)].sort_values(["series", "cache_idx"])
    out = {}
    for sr, g in df.groupby("series", sort=False):
        idx, dx = g["cache_idx"].to_numpy(), g["dx_mm"].to_numpy(np.float64)
        if np.array_equal(idx, np.arange(len(idx))) and np.isfinite(dx).all():
            out[str(sr)] = dx
    return out


class StudyDataset(torch.utils.data.Dataset):
    """One item = one study: uint8 windows [N_SLOTS, k, 3, H, W], slot mask, slice positions, target."""

    def __init__(self, studies, table, cache_dir, targets=None, k=8, train=False, loader=None, mirror=None, views=None,
                 phys=None):
        self.studies, self.table, self.cache_dir = list(studies), table, cache_dir
        self.targets, self.k, self.train = targets, k, train
        self.mirror = mirror or set()  # radiant-kg: left-knee studies, mirrored to look like right knees
        self.views = views  # radiant-kg V3: {series: {view: (row0, col0)}} close-up corners (views.py), or None
        self.loader = loader or (lambda s: np.load(f"{self.cache_dir}/{s}.npy", mmap_mode="r"))
        # radiant-kg V8 arm P: {series: per-cache-slice x displacement in mm (phys_values)}, or None. When given, every
        # item gets a fifth tensor [groups, k] (d / 50 mm, zero outside sagittal slots); when None, items are unchanged.
        self.phys = phys

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        if self.views is not None:  # radiant-kg V3
            return self._get_with_views(i, st, slots)
        x, mask, pos = None, np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        ph = np.zeros((N_SLOTS, self.k), np.float32)  # radiant-kg V8 arm P
        for s, cands in enumerate(slots):
            if not cands:
                continue
            sr = cands[np.random.randint(len(cands))] if self.train else cands[0]
            vol = self.loader(sr)
            if st in self.mirror:  # radiant-kg: flip patient left-right (cache is LPS, see preprocess.CANON)
                vol = vol[::-1] if s % 3 == 0 else vol[:, :, ::-1]  # sagittal: slice order; else: columns
            n = len(vol)
            c = window_centres(n, self.k, self.train)
            win = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])  # [k, 3, H, W]
            if x is None:
                x = np.zeros((N_SLOTS, self.k) + win.shape[1:], np.uint8)
            x[s], mask[s], pos[s] = win, True, c / max(n - 1, 1)
            if self.phys is not None and s % 3 == 0:  # radiant-kg V8 arm P: sagittal slots only
                ph[s] = phys_values(self.phys.get(sr), c, n, st in self.mirror)
        if x is None:
            size = int(os.environ.get("BRANCH_B_SIZE", "384"))  # radiant-kg: match the cache size
            x = np.zeros((N_SLOTS, self.k, 3, size, size), np.uint8)
        y = self.targets[i] if self.targets is not None else np.zeros(len(LABELS), np.float32)
        out = (torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(np.asarray(y, np.float32)))
        return out + (torch.from_numpy(ph),) if self.phys is not None else out


def _get_with_views_impl(self, i, st, slots):
    """radiant-kg V3: groups = slot * VIEWS_PER_SLOT + view (0 = full image, then the plane's close-ups in
    PLANE_VIEWS order); every group holds the same k windows (same slices), close-ups resized to the cache size."""
    import cv2
    cv2.setNumThreads(1)  # radiant-kg: one thread per loader worker (the box caps total threads)
    n_g = N_SLOTS * VIEWS_PER_SLOT
    x, mask, pos = None, np.zeros(n_g, bool), np.zeros((n_g, self.k), np.float32)
    ph = np.zeros((n_g, self.k), np.float32)  # radiant-kg V8 arm P
    left = st in self.mirror
    for s, cands in enumerate(slots):
        if not cands:
            continue
        sr = cands[np.random.randint(len(cands))] if self.train else cands[0]
        vol = self.loader(sr)
        if left:
            vol = vol[::-1] if s % 3 == 0 else vol[:, :, ::-1]
        n = len(vol)
        c = window_centres(n, self.k, self.train)
        win = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])  # [k, 3, H, W]
        if x is None:
            x = np.zeros((n_g, self.k) + win.shape[1:], np.uint8)
        size = win.shape[-1]
        g = s * VIEWS_PER_SLOT
        x[g], mask[g], pos[g] = win, True, c / max(n - 1, 1)
        if self.phys is not None and s % 3 == 0:  # radiant-kg V8 arm P: the sagittal full view and its close-ups share
            ph[g:g + VIEWS_PER_SLOT] = phys_values(self.phys.get(sr), c, n, left)  # slices
        boxes = self.views.get(sr, {})
        for vi, name in enumerate(PLANE_VIEWS[s % 3], start=1):
            if name not in boxes:
                continue
            r0, c0 = boxes[name]
            if left and s % 3 != 0:
                c0 = size - CROP_PX - c0  # coronal/axial columns were flipped
            crop = win[:, :, r0:r0 + CROP_PX, c0:c0 + CROP_PX]
            x[g + vi] = np.stack([np.stack([cv2.resize(ch, (size, size), interpolation=cv2.INTER_LINEAR) for ch in w])
                                  for w in crop])
            mask[g + vi], pos[g + vi] = True, c / max(n - 1, 1)
    if x is None:
        size = int(os.environ.get("BRANCH_B_SIZE", "384"))
        x = np.zeros((n_g, self.k, 3, size, size), np.uint8)
    ph[~mask] = 0  # radiant-kg V8 arm P: close-ups without a box stay exactly zero
    y = self.targets[i] if self.targets is not None else np.zeros(len(LABELS), np.float32)
    out = (torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(np.asarray(y, np.float32)))
    return out + (torch.from_numpy(ph),) if self.phys is not None else out


StudyDataset._get_with_views = _get_with_views_impl
# radiant-kg V3: close-up views per plane (views.py names), 100 mm = 250 px in the 0.4 mm cache
PLANE_VIEWS = {0: ["joint", "kneecap", "posterior"], 1: ["lateral", "medial"], 2: ["joint", "kneecap", "posterior"]}
VIEWS_PER_SLOT = 4
CROP_PX = 250
FX_NONFS_SLOTS = (3, 4)  # radiant-kg V8.1 fx-bias: non-fat-suppressed sagittal and coronal


def make_views(x, res, train, group, fov=0.92):
    """x: uint8 [N, 3, H, W] with N = n_series * group. Crops a centred `fov` fraction of the cached
    field of view and resizes to res; in training adds rotation/scale/shift and gain/gamma jitter
    that is shared within a series."""
    n = x.shape[0]
    ns = n // group
    dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev)
        sc = torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev)
        gamma = gain = None
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:  # avoid aliasing when shrinking a lot
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if train:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


class KneeNet(nn.Module):
    """2D backbone on 3-slice windows -> tokens tagged with series slot and slice position ->
    small transformer over all windows of the study -> one attention pooling per finding."""

    def __init__(self, backbone="convnext_tiny", pretrained=True, dim=384, depth=2, drop=0.2, drop_path=0.1, spatial=False,
                 views=False, pf_groups=None, oai_heads=0, phys_pos=False, fx_bias=None):
        super().__init__()
        self.spatial = bool(spatial)  # radiant-kg V2: per-finding focus inside each window (see spatial_pool)
        # radiant-kg V8 arm R: PF OA reads only these token groups (see head_per_finding).
        # None = V7 unchanged.
        self.pf_groups = tuple(pf_groups) if pf_groups is not None else None
        if self.pf_groups is not None:
            n_g = N_SLOTS * (VIEWS_PER_SLOT if views else 1)
            if not self.spatial:
                raise ValueError("pf_groups needs spatial=True "
                                 "(head() shares one token sequence across findings)")
            if not self.pf_groups or any(g < 0 or g >= n_g for g in self.pf_groups):
                raise ValueError(f"pf_groups {self.pf_groups} outside 0..{n_g - 1}")
            allowed = torch.zeros(n_g, dtype=torch.bool)
            allowed[list(self.pf_groups)] = True
            self.register_buffer("pf_allowed", allowed, persistent=False)  # not in the state dict
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        if os.environ.get("BRANCH_B_GRAD_CKPT") == "1":  # radiant-kg: trade training speed for memory
            self.enc.set_grad_checkpointing(True)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS * (VIEWS_PER_SLOT if views else 1), dim)  # radiant-kg V3: one per (slot, view)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))
        if self.spatial:  # radiant-kg V2, after the classifier made public in the 0.954 notebook (compact_train.spatial_pool)
            d = self.enc.num_features
            self.sp_key = nn.Linear(d, 128)  # keys from per-position normalised features
            self.sp_pos = nn.Linear(16, 128, bias=False)  # fixed 2-D position features -> keys (zero at init)
            nn.init.zeros_(self.sp_pos.weight)
            self.sp_q = nn.Parameter(torch.zeros(len(LABELS), 128))  # zero query = uniform weights = the plain average
        # radiant-kg V8 arm O: separate OAI outputs (own attention-pooling columns and classifier) on the shared backbone,
        # tokens and transformer. Built last, from a fixed seed inside fork_rng, so V7's parameters and every later random
        # number are unchanged; 0 heads = V7 (no new parameters).
        self.oai_heads = int(oai_heads or 0)
        if self.oai_heads:
            with torch.random.fork_rng(devices=[]):  # CPU stream saved and restored
                torch.random.default_generator.manual_seed(20261014)  # CPU only: CUDA's stream is not touched
                self.oai_att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, self.oai_heads))
                self.oai_cls_w = nn.Parameter(torch.randn(self.oai_heads, dim) * 0.02)
                self.oai_cls_b = nn.Parameter(torch.zeros(self.oai_heads))
        if phys_pos:  # radiant-kg V8 arm P: physical mediolateral position (d / 50 mm) added next to pos_emb
            with torch.random.fork_rng(devices=[]):  # Linear's default init draws random numbers; keep V7's stream
                self.phys = nn.Linear(1, dim, bias=False)
            nn.init.zeros_(self.phys.weight)  # zero at init: logits equal V7's for the same weights and input
        # radiant-kg V8.1 fx-bias: learned additive bias on Fracture's attention logits, one per token group, started
        # from parse_fx_bias. None = V7 (no parameter, no random numbers drawn).
        self.fx_bias = None
        if fx_bias is not None:
            n_g = N_SLOTS * (VIEWS_PER_SLOT if views else 1)
            if not (self.spatial and views):
                raise ValueError("fx_bias needs spatial=True and views=True (24 token groups)")
            if len(fx_bias) != n_g:
                raise ValueError(f"fx_bias has {len(fx_bias)} values, the model has {n_g} groups")
            self.fx_bias = nn.Parameter(torch.tensor([float(v) for v in fx_bias], dtype=torch.float32))

    def spatial_pool(self, fmap):
        """radiant-kg V2. fmap [N, d, h, w] (ConvNeXt features before its head) -> [N, C, d]: finding c weighs the
        window's positions with its own query, then the encoder's head norm. Zero queries give the plain average, so at
        initialisation every finding gets exactly the baseline feature norm(mean_p f_p)."""
        n, d, h, w = fmap.shape
        f = fmap.flatten(2).transpose(1, 2)  # [N, P, d]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=f.device), torch.linspace(-1, 1, w, device=f.device),
                                indexing="ij")
        fr = torch.tensor([1.0, 2.0, 4.0, 8.0], device=f.device) * (math.pi / 2)
        yy, xx = yy.reshape(-1, 1) * fr, xx.reshape(-1, 1) * fr
        pe = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1).to(f.dtype)  # [P, 16]
        kin = F.layer_norm(f.float(), (d,)).to(f.dtype)
        k = torch.tanh(self.sp_key(kin) + self.sp_pos(pe)[None])  # [N, P, 128]
        att = torch.einsum("npk,ck->npc", k, self.sp_q.to(k.dtype)).float().softmax(1).to(f.dtype)
        g = torch.einsum("npc,npd->ncd", att, f)  # [N, C, d]
        nrm = self.enc.head.norm
        return F.layer_norm(g.float(), (d,), nrm.weight.float(), nrm.bias.float(), nrm.eps).to(f.dtype)

    def _phys(self, phys, dtype):
        """radiant-kg V8 arm P: [B, S, K] displacement / 50 mm -> [B, S, K, dim] residual, or None when the arm is off."""
        if phys is None or not hasattr(self, "phys"):
            assert phys is None, "phys input given to a model built without phys_pos"
            return None
        return self.phys(phys.unsqueeze(-1).to(dtype))

    def head(self, feat, mask, pos, phys=None, oai=False):
        """feat [B, S, K, D] (or [B, S, K, C, D] with spatial focus), mask [B, S] bool, pos [B, S, K] in 0..1 -> logits [B, 12].
        phys [B, S, K] (radiant-kg V8 arm P) or None.
        radiant-kg V8 arm O: oai=True returns (logits [B, 12], OAI logits [B, H])."""
        if feat.dim() == 5:  # radiant-kg V2: one token sequence per finding, sharing every weight
            return self.head_per_finding(feat, mask, pos, phys, oai)
        if self.fx_bias is not None:  # radiant-kg V8.1: the shared sequence has no Fracture-only attention column path
            raise ValueError("fx_bias needs per-finding features (spatial=True)")
        b, s, k, _ = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        ph = self._phys(phys, feat.dtype)
        if ph is not None:
            h = h + ph
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)  # a study with no series at all attends to everything
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)  # [B, T, C]
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        out = (pooled * self.cls_w).sum(-1) + self.cls_b
        return (out, self.oai_pool(h, pad)) if oai else out

    def oai_pool(self, h, pad):
        """radiant-kg V8 arm O. h [B, T, dim] tokens after the transformer, pad [B, T] -> OAI logits [B, H]: the same
        attention pooling and classifier form as the competition head, with the OAI head's own parameters."""
        a = self.oai_att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)  # [B, T, H]
        pooled = torch.einsum("bth,btd->bhd", a.to(h.dtype), h)
        return (pooled * self.oai_cls_w).sum(-1) + self.oai_cls_b

    def head_per_finding(self, feat, mask, pos, phys=None, oai=False):
        """radiant-kg V2. feat [B, S, K, C, D]: finding c runs the shared head on its own window features and reads only
        its own attention column. With identical features for every finding this equals head() exactly."""
        b, s, k, c, _ = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        add = self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))  # [B, S, K, dim]
        ph = self._phys(phys, feat.dtype)  # radiant-kg V8 arm P
        if ph is not None:
            add = add + ph
        h = self.proj(feat) + add.unsqueeze(3)  # [B, S, K, C, dim]
        h = h.permute(0, 3, 1, 2, 4).reshape(b * c, s * k, -1)  # [B*C, T, dim]
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        padc = pad.repeat_interleave(c, 0)
        pool_pad = pad.unsqueeze(-1)  # [B, T, 1]
        if self.pf_groups is not None:  # radiant-kg V8 arm R
            padc, pool_pad = self.pf_route(mask, k, pad, padc)
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=padc)
        a = self.att(h).float().view(b, c, s * k, c)  # [B, C(sequence), T, C(column)]
        a = torch.diagonal(a, dim1=1, dim2=3)  # [B, T, C]
        if self.fx_bias is not None:  # radiant-kg V8.1: Fracture column only, before padding (pads stay at -1e4)
            assert s == len(self.fx_bias), f"fx_bias has {len(self.fx_bias)} groups, input has {s}"
            fx = a[..., FX_INDEX:FX_INDEX + 1] + self.fx_bias.repeat_interleave(k).view(1, -1, 1)  # token t = g * k + j
            a = torch.cat([a[..., :FX_INDEX], fx, a[..., FX_INDEX + 1:]], -1)
        a = a.masked_fill(pool_pad, -1e4).softmax(1)
        pooled = torch.einsum("btc,bctd->bcd", a.to(h.dtype), h.view(b, c, s * k, -1))
        out = (pooled * self.cls_w).sum(-1) + self.cls_b
        if not oai:
            return out
        # radiant-kg V8 arm O: OAI heads read the mean of the per-finding token sequences (no extra transformer pass);
        # with --pf-groups the routed PF OA sequence is left out, so routing stays a competition-PF-only change
        hs = h.view(b, c, s * k, -1)
        if self.pf_groups is not None:
            hs = hs[:, [i for i in range(c) if i != PF_INDEX]]
        return out, self.oai_pool(hs.mean(1), pad)

    def pf_route(self, mask, k, pad, padc):
        """radiant-kg V8 arm R. PF OA's sequence keeps only valid tokens of the allowed groups,
        both as transformer key padding and at attention pooling; a study with no allowed group
        valid keeps its original padding. Returns (padc [B*C, T], pool_pad [B, T, C]); the other
        findings' rows are the unchanged `pad`."""
        b, s = mask.shape
        c = len(LABELS)
        keep = (mask & self.pf_allowed.view(1, s)).unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pf_pad = torch.where(keep.any(1, keepdim=True), ~keep, pad)
        padc = padc.view(b, c, s * k).clone()
        padc[:, PF_INDEX] = pf_pad
        pool_pad = pad.unsqueeze(-1).expand(b, s * k, c).clone()
        pool_pad[:, :, PF_INDEX] = pf_pad
        return padc.view(b * c, s * k), pool_pad

    def forward(self, x, mask, pos, res=256, train=False, phys=None, oai=False):
        """x uint8 [B, S, K, 3, H, W]. phys [B, S, K] (radiant-kg V8 arm P) or None.
        radiant-kg V8 arm O: oai=True returns (logits, OAI logits)."""
        if oai and not self.oai_heads:
            raise ValueError("oai=True needs a model built with oai_heads > 0")
        b, s, k = x.shape[:3]
        keep = mask.view(-1)  # run the backbone only on series that exist
        xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        views = make_views(xv, res, train, k).contiguous(memory_format=torch.channels_last)
        if self.spatial:  # radiant-kg V2
            g = self.spatial_pool(self.enc.forward_features(views))  # [N, C, d]
            feat = g.new_zeros(b * s, k, *g.shape[1:])
            feat[keep] = g.view(-1, k, *g.shape[1:])
            return self.head(feat.view(b, s, k, *g.shape[1:]), mask, pos, phys, oai)
        f = self.enc(views)
        feat = f.new_zeros(b * s, k, f.shape[-1])
        feat[keep] = f.view(-1, k, f.shape[-1])
        return self.head(feat.view(b, s, k, -1), mask, pos, phys, oai)

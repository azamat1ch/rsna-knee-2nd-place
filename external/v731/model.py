"""V7.3.1 PF specialist (contract: v731/CONTRACT.md, "Model outputs").

One timm ConvNeXt encoder (initialised from fm_encoder7 only) reads every valid window of every
enabled group; V7.1's spatial_pool gives one feature per output query; tokens = proj(feature) +
group embedding (15 groups) + slice-position MLP; a 2-layer pre-norm transformer runs one token
sequence per output (V7.1 head_per_finding); attention pooling reads that output's own column.

Outputs ("queries"):
  E1/E2: pf                    -> pf_direct logit [B]
  E3   : pf, PM, PL, FMA, FLA  -> pf_direct, grade_logits [B, 8, 3], pf_combined [B]
E3 grade head: per region a linear map to (size latent, conditional full-thickness latent);
cumulative logits = latent - increasing cut points, so P(>=1) >= P(>=2) >= P(>=3); full-thickness
probability = P(size) * P(conditional), then min with size and cumulative min over thresholds, so
ft <= size at every threshold by construction (after V7.3 SeverityHead version 2).
pf_combined = pf_direct + sum(softplus(w) * P(grade >= t)) + b with w [8, 3] and ONE scalar b.

View preparation reuses knee.make_views (V7.1 augmentation, shared within one group's series):
W/P at res 256, Z/K at res 512, fov 0.92. P crops (native size from v731.manifest.CROP, any
size) are resized to p_resize = 384 px first, as V7.1 resized its close-ups to the 384 px cache
size; Z/K crops go to make_views at their native size. Encoder input resolutions therefore do not
depend on the box size. The encoder runs over all valid windows in chunks (default 64); an exam of
any size is processed in full.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

BRANCH_B = Path(__file__).resolve().parents[1]
if str(BRANCH_B / "reader") not in sys.path:
    sys.path.insert(0, str(BRANCH_B / "reader"))
from knee import KneeNet, make_views  # noqa: E402

if str(BRANCH_B) not in sys.path:
    sys.path.insert(0, str(BRANCH_B))
from v731.dataset import FIELDS, K_GROUP, N_GROUPS, N_THRESH, P_GROUPS, Z_GROUPS  # noqa: E402

REGIONS = ("PM", "PL", "FMA", "FLA")
FM_ENCODER7_SHA256 = "404e487853915dd0bb7b72ccb390412acff61b46f5091bc963abbf65b92a5628"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class GradeHead(nn.Module):
    """Eight ordered cumulative MOAKS fields + the PF fusion (one scalar bias)."""

    def __init__(self, dim: int):
        super().__init__()
        self.regional = nn.ModuleList([nn.Linear(dim, 2) for _ in REGIONS])
        self.first_cut = nn.Parameter(torch.zeros(len(FIELDS), 1))
        self.cut_gaps = nn.Parameter(torch.zeros(len(FIELDS), N_THRESH - 1))
        self.fuse_w = nn.Parameter(torch.full((len(FIELDS), N_THRESH), -2.0))
        self.fuse_b = nn.Parameter(torch.zeros(()))  # the ONE fusion bias

    def cuts(self) -> torch.Tensor:
        return torch.cat(
            [self.first_cut, self.first_cut + F.softplus(self.cut_gaps).cumsum(-1)], -1
        )

    def ordinal_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled [B, 4, dim] (one per region, REGIONS order) -> [B, 8, 3] cumulative logits, FIELDS
        order."""
        b = pooled.shape[0]
        latent = torch.stack([lin(pooled[:, r]) for r, lin in enumerate(self.regional)], 1)
        logits = latent.float().reshape(b, len(FIELDS)).unsqueeze(-1) - self.cuts().float()
        size, cond = logits[:, 0::2], logits[:, 1::2]
        size = size.cummin(-1).values  # already ordered (increasing cuts); guards float ties
        # logit(sigmoid(size) * sigmoid(cond)), stable at extreme logits
        ft = size + cond - torch.logsumexp(torch.stack([torch.zeros_like(size), size, cond]), 0)
        ft = torch.minimum(ft, size).cummin(-1).values
        return torch.stack([size, ft], 2).flatten(1, 2)

    def fuse(self, pf_direct: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """pf_direct [B] + sum(softplus(w) * P) (no bias inside the sum) + b."""
        return (
            pf_direct.float()
            + (probs.float() * F.softplus(self.fuse_w.float())).sum((1, 2))
            + self.fuse_b
        )


def ordinal_targets(grades: torch.Tensor) -> torch.Tensor:
    """[B, 8] MOAKS grades 0..3 (NaN unknown) -> [B, 8, 3] cumulative targets (grade >= t), NaN
    kept."""
    known = ~torch.isnan(grades)
    bad = known & ((grades < 0) | (grades > 3) | (grades != grades.round()))
    if bool(bad.any()):
        raise ValueError("known grades must be integers 0..3")
    t = torch.arange(1, N_THRESH + 1, device=grades.device, dtype=grades.dtype)
    return torch.where(known.unsqueeze(-1), (grades.unsqueeze(-1) >= t).float(), float("nan"))


def grade_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Masked cumulative BCE: mean over known cells per field (over the batch's rows and
    thresholds), then mean over fields with >= 1 known cell. Unknown cells contribute exactly 0
    (value and gradient)."""
    known = ~torch.isnan(y)
    bce = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y), reduction="none")
    bce = torch.where(known, bce, torch.zeros_like(bce))
    n_field = known.sum((0, 2))
    per_field = bce.sum((0, 2)) / n_field.clamp_min(1)
    has = n_field > 0
    return (per_field * has).sum() / has.sum().clamp_min(1)


class PFSpecialist(nn.Module):
    spatial_pool = (
        KneeNet.spatial_pool
    )  # V7.1 per-output spatial queries (uses sp_key, sp_pos, sp_q, enc)

    def __init__(
        self,
        groups: Any = range(10),
        grade_head: bool = False,
        backbone: str = "convnext_tiny",
        dim: int = 384,
        depth: int = 2,
        heads: int = 8,
        ffn: int = 768,
        drop: float = 0.2,
        tf_dropout: float = 0.1,
        drop_path: float = 0.1,
        grad_ckpt: bool = False,
        chunk: int = 64,
        pad_chunks: bool = False,
        chunk_ckpt: bool = False,
        res_wp: int = 256,
        res_zk: int = 512,
        fov: float = 0.92,
        p_resize: int = 384,
    ):
        super().__init__()
        self.groups = tuple(sorted(int(g) for g in groups))
        if not set(range(6)) <= set(self.groups) <= set(range(N_GROUPS)):
            raise ValueError(f"groups {self.groups}: must include 0..5 and lie in 0..14")
        self.has_grades = bool(grade_head)
        if self.has_grades and set(self.groups) != set(range(N_GROUPS)):
            raise ValueError("the grade head (E3) is built on E2's groups 0..14")
        self.outputs = ("pf", *REGIONS) if self.has_grades else ("pf",)
        self.chunk, self.res_wp, self.res_zk = int(chunk), int(res_wp), int(res_zk)
        # pad_chunks: every encoder call sees exactly `chunk` windows (zero windows appended and
        # discarded), so conv shapes never vary; on CUDA each new batch size costs ~10 s of cuDNN
        # set-up (measured on an RTX 5090). Real outputs are unchanged (per-sample encoder).
        self.pad_chunks = bool(pad_chunks)
        # chunk_ckpt: checkpoint each encoder chunk (recomputed in backward with the same RNG
        # state) instead of timm's block checkpointing; memory only, the gradients are the same.
        self.chunk_ckpt = bool(chunk_ckpt)
        self.fov, self.p_resize = float(fov), int(p_resize)
        enabled = torch.zeros(N_GROUPS, dtype=torch.bool)
        enabled[list(self.groups)] = True
        self.register_buffer("enabled", enabled, persistent=False)
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=False, num_classes=0, **kw)
        if grad_ckpt and not chunk_ckpt:  # chunk_ckpt replaces block-level checkpointing
            self.enc.set_grad_checkpointing(True)
        nf, c = self.enc.num_features, len(self.outputs)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(nf, dim), nn.LayerNorm(dim))
        self.group_emb = nn.Embedding(N_GROUPS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, ffn, dropout=tf_dropout, batch_first=True, norm_first=True
        )
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Sequential(
            nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, c)
        )
        self.cls_w = nn.Parameter(torch.randn(dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(1))
        self.sp_key = nn.Linear(nf, 128)
        self.sp_pos = nn.Linear(16, 128, bias=False)
        nn.init.zeros_(self.sp_pos.weight)
        self.sp_q = nn.Parameter(torch.zeros(c, 128))  # zero queries = plain average at init (V7.1)
        self.grade = GradeHead(dim) if self.has_grades else None

    # ---- encoder ---------------------------------------------------------------------------
    def encode(self, views: torch.Tensor) -> torch.Tensor:
        """views [N, 3, r, r] -> [N, C, nf] in chunks of self.chunk windows (identical to one
        pass)."""
        c, nf = len(self.outputs), self.enc.num_features
        n, size = len(views), max(1, self.chunk)
        if n and self.pad_chunks and n % size:
            views = torch.cat([views, views.new_zeros((size - n % size, *views.shape[1:]))])

        def run(v: torch.Tensor) -> torch.Tensor:
            fmap = self.enc.forward_features(v.contiguous(memory_format=torch.channels_last))
            return self.spatial_pool(fmap)

        ckpt = self.chunk_ckpt and torch.is_grad_enabled() and self.training
        outs = [
            torch.utils.checkpoint.checkpoint(run, v, use_reentrant=False) if ckpt else run(v)
            for v in views.split(size)
        ]
        return torch.cat(outs)[:n] if outs else views.new_zeros((0, c, nf))

    def _resize_p(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.p_resize:
            return x
        y = F.interpolate(
            x.float(), size=(self.p_resize, self.p_resize), mode="bilinear", align_corners=False
        )
        return y.round_().clamp_(0, 255).to(torch.uint8)

    def tokens(self, batch: dict[str, Any], train: bool = False):
        """-> feat [B, T, C, nf], valid [B, T], group id [T], pos [B, T]. Layout per exam:
        W (6 x kmax) | P (4 x Np) | Z (4 x Nz) | K (1 x Nk), padded to the batch maximum."""
        w = batch["w"]
        dev, b, kmax = w.device, w.shape[0], w.shape[2]
        n_p, n_z, n_k = batch["p"].shape[2], batch["z"].shape[2], batch["k"].shape[2]
        off_p = 6 * kmax
        off_z = off_p + 4 * n_p
        off_k = off_z + 4 * n_z
        t_all = off_k + n_k
        rows = torch.arange(b, device=dev) * t_all
        # W: V7.1's call (all valid series at once, augmentation shared per series of k windows)
        wmask = batch["w_mask"] & self.enabled[:6]
        keep = wmask.reshape(-1)
        xv = w.reshape(b * 6, kmax, *w.shape[3:])[keep].flatten(0, 1)
        tok_w = batch["w_tok"].reshape(b * 6, kmax)[keep].reshape(-1)
        idx_w = (rows[:, None, None] + torch.arange(6 * kmax, device=dev).view(1, 6, kmax)).reshape(
            b * 6, kmax
        )
        idx_w = idx_w[keep].reshape(-1)[tok_w]
        views_lo = [make_views(xv, self.res_wp, train, kmax, self.fov)[tok_w]] if len(xv) else []
        idx_lo, views_hi, idx_hi = [idx_w], [], []
        for kind, gids, off, n_max in (
            ("p", P_GROUPS, off_p, n_p),
            ("z", Z_GROUPS, off_z, n_z),
            ("k", (K_GROUP,), off_k, n_k),
        ):
            if n_max == 0:
                continue
            counts = batch[f"{kind}_tok"].sum(-1).tolist()  # one host sync per kind
            x = batch[kind]
            for i in range(b):
                for g, gid in enumerate(gids):
                    n = int(counts[i][g])
                    if n == 0 or gid not in self.groups:
                        continue
                    xi = x[i, g, :n]
                    if kind == "p":
                        xi = self._resize_p(xi)
                    res = self.res_wp if kind == "p" else self.res_zk
                    v = make_views(
                        xi, res, train, n, self.fov
                    )  # one augmentation per group's series
                    idx = rows[i] + off + g * n_max + torch.arange(n, device=dev)
                    (views_lo if kind == "p" else views_hi).append(v)
                    (idx_lo if kind == "p" else idx_hi).append(idx)
        c, nf = len(self.outputs), self.enc.num_features
        feats, idxs = [], []
        for vs, ix in ((views_lo, idx_lo), (views_hi, idx_hi)):
            if vs:
                feats.append(self.encode(torch.cat(vs)))
                idxs.append(torch.cat(ix))
        dtype = feats[0].dtype if feats else torch.float32
        feat = torch.zeros((b * t_all, c, nf), device=dev, dtype=dtype)
        valid = torch.zeros(b * t_all, device=dev, dtype=torch.bool)
        for f, ix in zip(feats, idxs, strict=True):
            feat = feat.index_put((ix,), f.to(dtype))
            valid[ix] = True
        gid = torch.cat(
            [
                torch.arange(6, device=dev).repeat_interleave(kmax),
                torch.tensor(P_GROUPS, device=dev).repeat_interleave(n_p),
                torch.tensor(Z_GROUPS, device=dev).repeat_interleave(n_z),
                torch.full((n_k,), K_GROUP, device=dev, dtype=torch.long),
            ]
        )
        pos = torch.cat([batch[f"{k}_pos"].reshape(b, -1) for k in ("w", "p", "z", "k")], 1)
        return feat.view(b, t_all, c, nf), valid.view(b, t_all), gid, pos

    # ---- head ------------------------------------------------------------------------------
    def head(self, feat: torch.Tensor, valid: torch.Tensor, gid: torch.Tensor, pos: torch.Tensor):
        """V7.1 head_per_finding over C outputs -> pooled [B, C, dim]."""
        b, t, c, _ = feat.shape
        h = self.proj(feat)  # [B, T, C, dim]
        add = self.group_emb(gid)[None] + self.pos_emb(pos.unsqueeze(-1).to(h.dtype))
        h = (h + add.unsqueeze(2)).permute(0, 2, 1, 3).reshape(b * c, t, -1)
        pad = ~valid
        pad = pad & ~pad.all(1, keepdim=True)  # an exam with no valid token attends to everything
        h = self.tf(h, src_key_padding_mask=pad.repeat_interleave(c, 0))
        a = self.att(h).float().view(b, c, t, c)
        a = torch.diagonal(a, dim1=1, dim2=3)  # [B, T, C]: output c reads its own column
        a = a.masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        return torch.einsum("btc,bctd->bcd", a.to(h.dtype), h.view(b, c, t, -1))

    def forward(self, batch: dict[str, Any], train: bool = False) -> dict[str, torch.Tensor]:
        feat, valid, gid, pos = self.tokens(batch, train)
        pooled = self.head(feat, valid, gid, pos)
        pf_direct = ((pooled[:, 0] * self.cls_w).sum(-1) + self.cls_b).float()
        out = {"pf_direct": pf_direct}
        if self.grade is not None:
            logits = self.grade.ordinal_logits(pooled[:, 1:])
            out["grade_logits"] = logits
            out["pf_combined"] = self.grade.fuse(pf_direct, logits.sigmoid())
        return out


def build_model(
    cfg: dict[str, Any], backbone: str | None = None, chunk_ckpt: bool = False
) -> PFSpecialist:
    r = cfg["recipe"]
    return PFSpecialist(
        groups=cfg["groups"],
        grade_head=bool(cfg["grade_head"]),
        backbone=backbone or r["backbone"],
        dim=r["dim"],
        depth=r["depth"],
        heads=r["heads"],
        ffn=r["ffn"],
        drop=r["drop"],
        tf_dropout=r["tf_dropout"],
        drop_path=r["drop_path"],
        grad_ckpt=bool(r["grad_ckpt"]),
        chunk=r["chunk"],
        pad_chunks=bool(r.get("pad_chunks", False)),
        chunk_ckpt=chunk_ckpt,
        res_wp=r["res_wp"],
        res_zk=r["res_zk"],
        fov=r["fov"],
        p_resize=r["p_resize"],
    )


def load_encoder(
    model: PFSpecialist, path: str | Path, sha256: str | None = FM_ENCODER7_SHA256
) -> str:
    """Load the encoder state dict (strict) from fm_encoder7.pt; refuse a different file."""
    got = sha256_file(path)
    if sha256 is not None and got != sha256:
        raise ValueError(f"encoder {path}: sha256 {got} != expected {sha256}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.enc.load_state_dict(sd)
    return got


def param_groups(model: PFSpecialist, lr_enc: float, lr_head: float) -> list[dict[str, Any]]:
    """V7.1 AdamW groups: encoder at lr_enc, everything else at lr_head (global weight decay on
    both)."""
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    return [{"params": list(model.enc.parameters()), "lr": lr_enc}, {"params": head, "lr": lr_head}]

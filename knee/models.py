"""The two model families and how they are built and loaded.

TileModel: one timm encoder per view (a 3-slice tile), a slot and stack-position embedding, then per-finding gated
attention over the views. With `context_size` in the recipe it has a second, separately pooled whole-knee branch and a
per-exam gate on the detail term. WindowNet: an encoder feature map per window, per-finding spatial attention inside
the window, then per-finding attention across the windows. Parameter names, shapes and the order in which layers are
made (and so the random draws of a same-seed start) follow the shipped weight files.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from knee import LABELS, sha256

N_LABELS = len(LABELS)
N_SLOTS = 6                 # plane (sag, cor, ax) x fluid-sensitive (0/1), slot-major view blocks
ATTENTION_HIDDEN = 128      # gated attention width (tile family)
POSITION_FREQUENCIES = 8    # sin/cos(pi k p), k = 1..8, per view
TILE_FORMAT = "knee-epoch-checkpoint-v1"

WINDOW_HIDDEN = 384         # window token width
WINDOW_DROPOUT = 0.1
WINDOW_SLOTS = 10           # slot embedding rows (meta column 0, clamped)
WINDOW_VIEWS = 6            # view embedding rows (meta column 1, clamped)
SPATIAL_KEY = 128           # spatial attention key width
SPATIAL_FREQUENCIES = (1.0, 2.0, 4.0, 8.0)  # times pi/2, per image axis
WINDOW_PAD = 32             # encoder batch padded to a multiple of this


def _no_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


# ---------------------------------------------------------------- tile family

def _add_pooling(owner: nn.Module, dim: int) -> None:
    """Slot embedding, gated attention and position projection on `owner`, in this order (random draws),
    with the zero starts that make a new model plain mean pooling."""
    owner.slot_embedding = nn.Embedding(N_SLOTS, dim)
    owner.attention_v = nn.Linear(dim, ATTENTION_HIDDEN)
    owner.attention_u = nn.Linear(dim, ATTENTION_HIDDEN)
    owner.attention_w = nn.Linear(ATTENTION_HIDDEN, N_LABELS)
    nn.init.zeros_(owner.slot_embedding.weight)
    nn.init.zeros_(owner.attention_w.weight)
    nn.init.zeros_(owner.attention_w.bias)


def _add_position(owner: nn.Module, dim: int) -> None:
    owner.position_projection = nn.Linear(N_SLOTS * 2 * POSITION_FREQUENCIES, dim, bias=False)
    nn.init.zeros_(owner.position_projection.weight)


def _slot_one_hot(views: int, device: torch.device) -> Tensor:
    slots = torch.arange(views, device=device) // (views // N_SLOTS)
    return F.one_hot(slots, N_SLOTS).float()                                      # (V, 6)


def _position_features(position: Tensor) -> Tensor:
    """(B, V) stack positions in [0, 1] -> (B, V, 16) fp32: sin(pi k p), cos(pi k p)."""
    k = torch.arange(1, POSITION_FREQUENCIES + 1, device=position.device, dtype=torch.float32)
    angle = math.pi * position.float()[..., None] * k
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def _position_embedding(owner: nn.Module, position: Tensor, mask: Tensor) -> Tensor:
    """(B, V, D) fp32: the view's position features placed in its slot block, projected; zero on unused views."""
    batch, views = mask.shape
    blocks = _slot_one_hot(views, mask.device)[None, :, :, None] * _position_features(position)[:, :, None, :]
    embedding = blocks.reshape(batch, views, -1) @ owner.position_projection.weight.float().t()
    return embedding * mask[..., None].float()


def _pool(owner: nn.Module, embeddings: Tensor, mask: Tensor) -> Tensor:
    """Per-finding gated attention over the views, fp32: (B, V, D) -> (B, 12, D)."""
    with _no_autocast(embeddings.device):
        h = embeddings + (_slot_one_hot(embeddings.shape[1], embeddings.device) @ owner.slot_embedding.weight.float())[None]
        x = h.to(owner.attention_v.weight.dtype)
        gated = torch.tanh(owner.attention_v(x)) * torch.sigmoid(owner.attention_u(x))
        scores = owner.attention_w(gated).float().masked_fill(~mask[..., None], float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bvj,bvd->bjd", weights, h)


class _Gate(nn.Module):
    """Per-finding detail gate, g = 2 sigmoid(w . [zA, zB] + b) in (0, 2). Zero start (g = 1), no random draw."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(N_LABELS, 2 * dim))
        self.bias = nn.Parameter(torch.zeros(N_LABELS))

    def forward(self, detail: Tensor, context: Tensor) -> Tensor:
        with _no_autocast(detail.device):
            x = torch.cat([detail.float(), context.float()], dim=-1).detach()   # the gate reads, never trains through
            return 2.0 * torch.sigmoid((x * self.weight.float()[None]).sum(dim=-1) + self.bias.float())


class TileModel(nn.Module):
    """Views (B, V, 3, H, W) normalised, view_mask (B, V) bool, view_position (B, V) -> (B, 12) logits.

    V is 6 equal slot blocks. two_branch adds a context branch (context_images/mask/position, its own pooling and
    a zero-start context head) and the detail gate. encoder_pad > 0 (inference only) pads each encoder batch with
    zero images to a multiple of it; eval-mode encoders treat every image on its own.
    """

    def __init__(self, encoder: nn.Module, dim: int, two_branch: bool = False):
        super().__init__()
        self.encoder, self.dim, self.two_branch, self.encoder_pad = encoder, dim, two_branch, 0
        self.head = nn.Linear(dim, N_LABELS)
        _add_pooling(self, dim)
        _add_position(self, dim)
        if two_branch:
            self.context = nn.Module()
            _add_pooling(self.context, dim)
            _add_position(self.context, dim)
            self.context_head = nn.Linear(dim, N_LABELS, bias=False)
            nn.init.zeros_(self.context_head.weight)
            self.gate = _Gate(dim)

    def _encode(self, images: Tensor) -> Tensor:
        count = images.shape[0]
        extra = -count % self.encoder_pad if self.encoder_pad else 0
        if extra:
            images = torch.cat([images, images.new_zeros((extra, *images.shape[1:]))])
        return self.encoder(images)[:count]

    def _embed(self, images: Tensor, mask: Tensor) -> Tensor:
        """fp32 (B, V, D) view features, zero on unused views; only used views reach the encoder."""
        features = self._encode(images[mask])
        embeddings = features.new_zeros((*mask.shape, self.dim), dtype=torch.float32)
        embeddings[mask] = features.float()
        return embeddings

    def _branch(self, owner: nn.Module, images: Tensor, mask: Tensor, position: Tensor) -> Tensor:
        """Pooled (B, 12, D) of one branch; an exam with no view in it gets zeros."""
        used = mask.any(dim=1)
        if used.all():
            embeddings = self._embed(images, mask)
        else:
            embeddings = images.new_zeros((*mask.shape, self.dim), dtype=torch.float32)
            if used.any():
                embeddings[used] = self._embed(images[used], mask[used])
        with _no_autocast(embeddings.device):
            embeddings = embeddings + _position_embedding(owner, position, mask)
        if used.all():
            return _pool(owner, embeddings, mask)
        safe = mask.clone()
        safe[~used, 0] = True   # a finite softmax for an empty row; its result is zeroed
        return _pool(owner, embeddings, safe) * used[:, None, None]

    def forward(self, images: Tensor, view_mask: Tensor, view_position: Tensor, context_images: Tensor | None = None,
                context_mask: Tensor | None = None, context_position: Tensor | None = None) -> Tensor:
        dtype = self.head.weight.dtype
        if not self.two_branch:
            embeddings = self._embed(images, view_mask)
            with _no_autocast(embeddings.device):
                embeddings = embeddings + _position_embedding(self, view_position, view_mask)
            pooled = _pool(self, embeddings, view_mask)
            return (pooled.to(dtype) * self.head.weight[None]).sum(dim=-1) + self.head.bias
        detail = self._branch(self, images, view_mask, view_position)
        context = self._branch(self.context, context_images, context_mask, context_position)
        if not (view_mask.any(dim=1) | context_mask.any(dim=1)).all():
            raise ValueError("every exam needs at least one view")
        term = (detail.to(dtype) * self.head.weight[None]).sum(dim=-1)
        logits = self.gate(detail, context).to(dtype) * term + self.head.bias
        return logits + (context.to(dtype) * self.context_head.weight[None]).sum(dim=-1)


def _tile_encoder(arch: str, start_weights: Path | None) -> nn.Module:
    """timm classifier model, strict load of the start weights, then the classifier removed (pooled features)."""
    import timm
    encoder = timm.create_model(arch, pretrained=False, num_classes=1000)
    if start_weights is not None:
        from safetensors.torch import load_file
        state = load_file(str(start_weights), device="cpu")
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("non-finite start weights")
        encoder.load_state_dict(state, strict=True)
    encoder.reset_classifier(0)
    return encoder


# ---------------------------------------------------------------- window family

def _window_encoder(arch: str, start_weights: Path | None, drop_path: float) -> nn.Module:
    """timm model returning its last feature map; start weights loaded strictly except the ImageNet classifier."""
    import timm
    encoder = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, drop_path_rate=drop_path,
                                global_pool="")
    if start_weights is not None:
        from safetensors.torch import load as load_bytes
        state = load_bytes(Path(start_weights).read_bytes())
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        classifier = encoder.pretrained_cfg.get("classifier") or "head.fc"
        allowed = {"head.fc.weight", "head.fc.bias", classifier + ".weight", classifier + ".bias"}
        if missing or set(unexpected) - allowed:
            raise ValueError(f"start weights do not fit {arch}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return encoder


class WindowNet(nn.Module):
    """Windows img uint8 (B, W, 3, S, S), meta (B, W, 6), mask (B, W) bool -> (B, 12) logits.

    meta columns: slot, view id, rel_pos, log_spacing, extent_130, fluid. A ConvNeXt head norm is moved out of the
    encoder and applied to each finding's pooled vector (pool first, then norm, as the classifier does).
    """

    def __init__(self, encoder: nn.Module, mean: list[float], std: list[float]):
        super().__init__()
        norm = getattr(getattr(encoder, "head", None), "norm", None)
        self.sp_head_norm = None
        if isinstance(norm, nn.Module) and not isinstance(norm, nn.Identity):
            self.sp_head_norm = norm
            encoder.head.norm = nn.Identity()
        self.enc = encoder
        d, h = encoder.num_features, WINDOW_HIDDEN
        self.proj = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h))
        self.slot = nn.Embedding(WINDOW_SLOTS, h)
        self.view = nn.Embedding(WINDOW_VIEWS, h)
        self.geom = nn.Linear(4, h, bias=False)
        for weight in (self.slot.weight, self.view.weight, self.geom.weight):
            nn.init.zeros_(weight)
        self.drop = nn.Dropout(WINDOW_DROPOUT)
        self.win_proj = nn.Linear(h, h)
        self.q_win = nn.Parameter(torch.randn(N_LABELS, h) * 0.02)
        self.out_norm = nn.LayerNorm(h)
        self.cls_w = nn.Parameter(torch.randn(N_LABELS, h) * 0.02)
        self.bias = nn.Parameter(torch.zeros(N_LABELS))
        self.register_buffer("mean", torch.tensor([float(v) for v in mean]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([float(v) for v in std]).view(1, 3, 1, 1))
        self.sp_key = nn.Linear(d, SPATIAL_KEY)
        self.sp_pos = nn.Linear(4 * len(SPATIAL_FREQUENCIES), SPATIAL_KEY, bias=False)
        nn.init.zeros_(self.sp_pos.weight)
        self.sp_q = nn.Parameter(torch.zeros(N_LABELS, SPATIAL_KEY))   # zero query = plain average over positions

    def spatial_pool(self, fmap: Tensor) -> Tensor:
        """[N, d, h, w] map -> [N, 12, d]: each finding weighs the positions with its own query."""
        h, w = fmap.shape[-2:]
        f = fmap.flatten(2).transpose(1, 2)                                         # [N, P, d], p = y * w + x
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=f.device), torch.linspace(-1, 1, w, device=f.device),
                                indexing="ij")
        freq = torch.tensor(SPATIAL_FREQUENCIES, device=f.device) * (math.pi / 2)
        yy, xx = yy.reshape(-1, 1) * freq, xx.reshape(-1, 1) * freq
        pe = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1)                # [P, 16]
        keys_in = F.layer_norm(f.float(), (f.shape[-1],)).to(f.dtype)
        k = torch.tanh(self.sp_key(keys_in) + self.sp_pos(pe.to(f.dtype))[None])   # [N, P, 128]
        att = torch.einsum("npk,ck->npc", k, self.sp_q.to(k.dtype)).float().softmax(1).to(f.dtype)
        g = torch.einsum("npc,npd->ncd", att, f)
        if self.sp_head_norm is not None:
            n = self.sp_head_norm
            g = F.layer_norm(g.float(), (g.shape[-1],), n.weight, n.bias, n.eps).to(f.dtype)
        return g

    def head(self, tokens: Tensor, meta: Tensor, mask: Tensor) -> Tensor:
        """Per-finding window tokens of the real windows [n, 12, d] -> logits [B, 12]."""
        batch, windows = mask.shape
        f = self.proj(tokens.float())
        emb = (self.slot(meta[..., 0].long().clamp(0, WINDOW_SLOTS - 1))
               + self.view(meta[..., 1].long().clamp(0, WINDOW_VIEWS - 1)) + self.geom(meta[..., 2:6]))
        t = f.new_zeros((batch, windows) + tuple(f.shape[1:]))
        t[mask] = f
        t = self.drop(t + emb[:, :, None, :])
        scores = torch.einsum("bwch,ch->bwc", torch.tanh(self.win_proj(t)), self.q_win).float()
        weights = scores.masked_fill(~mask[:, :, None], float("-inf")).softmax(1).to(t.dtype)
        study = self.out_norm(torch.einsum("bwc,bwch->bch", weights, t))
        return (study * self.cls_w).sum(-1).float() + self.bias

    def forward(self, img: Tensor, meta: Tensor, mask: Tensor, chunk: int = 0) -> Tensor:
        x = img[mask]
        n = len(x)
        if n % WINDOW_PAD:   # a few batch shapes only; every op is per window, so pad rows change nothing
            x = torch.cat([x, x.new_zeros((WINDOW_PAD - n % WINDOW_PAD,) + tuple(x.shape[1:]))])
        step = chunk or len(x)
        tokens = []
        for s in range(0, len(x), step):
            xs = x[s:s + step].float().div_(255.)
            xs = ((xs - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            tokens.append(self.spatial_pool(self.enc(xs)))
        return self.head(torch.cat(tokens)[:n], meta, mask)


def frozen_batchnorm(encoder: nn.Module) -> list[nn.Module]:
    """The encoder's batch-norm layers, which training keeps in eval mode (pretrained statistics)."""
    found = [m for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not found:
        raise ValueError("frozen_batchnorm needs an encoder with batch-norm layers")
    return found


# ---------------------------------------------------------------- build and load

def build(recipe: dict, start_weights: Path | None = None) -> nn.Module:
    """A new model of the recipe. start_weights: the pinned encoder safetensors (None = random encoder)."""
    arch = recipe["encoder"]["timm"]
    if recipe["family"] == "tile":
        encoder = _tile_encoder(arch, start_weights)
        return TileModel(encoder, encoder.num_features, two_branch="context_size" in recipe["inputs"])
    if recipe["family"] == "window":
        encoder = _window_encoder(arch, start_weights, recipe["encoder"]["drop_path"])
        model = WindowNet(encoder, recipe["encoder"]["mean"], recipe["encoder"]["std"])
        return model.to(memory_format=torch.channels_last)
    raise ValueError("unknown family " + str(recipe["family"]))


def load(recipe: dict, weight_file: str | Path, sha256_pin: str | None = None) -> nn.Module:
    """A trained member in eval mode, loaded strictly from a shipped (or newly trained) weight file."""
    if sha256_pin is not None and sha256(weight_file) != sha256_pin:
        raise ValueError(f"{weight_file}: sha256 does not match the pin")
    saved = torch.load(str(weight_file), map_location="cpu", weights_only=True)
    if recipe["family"] == "tile":
        if saved.get("format") != TILE_FORMAT:
            raise ValueError(f"{weight_file}: not a {TILE_FORMAT} file")
        saved = {k.removeprefix("_orig_mod."): v for k, v in saved["states"]["model"].items()}
    model = build(recipe)
    model.load_state_dict(saved, strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise ValueError(f"{weight_file}: non-finite weights")
    return model.eval()

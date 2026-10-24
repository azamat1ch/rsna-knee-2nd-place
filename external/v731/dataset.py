"""V7.3.1 PF specialist dataset (contract: v731/CONTRACT.md, "Dataset item").

One item = one exam:
  W groups 0..5: V7.1's whole-knee windows, sampled exactly as V7.1 (knee.build_study_table
      candidates, random candidate per slot in training / first in evaluation,
      knee.window_centres with k = 8 train / 12 eval, 3-slice windows, left knees mirrored).
  P groups 6..9: every band slab of native/P/<series>.npy for slots 0, 2, 3, 5 (sagFS, axFS,
      sagNFS, axNFS), the SAME series this sample picked for that slot's W.
  Z groups 10..13: native/Z/<series>.npy of the same series as P.
  K group 14: native/K/<series>.npy of the study's dedicated cartilage series; masked when the
      study has none (or its array is absent). When the K series is the series picked for a
      slot's P/Z in this sample, that slot's Z group is masked instead (K replaces it; flagged
      in `k_replaces_z` / `z_replaced_group`); that slot's P stays.
P/Z/K token j = slabs [j-1, j, j+1] clipped to the band; position = j / max(n - 1, 1).
Left knees (laterality side L): sagittal arrays reverse slab order, axial/coronal arrays flip crop
columns. A picked series whose native array is absent masks that group and is flagged in
`g_missing` (never substituted). Evaluation mode (train=False) draws no random numbers.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch

BRANCH_B = Path(__file__).resolve().parents[1]
if str(BRANCH_B / "reader") not in sys.path:
    sys.path.insert(0, str(BRANCH_B / "reader"))
from knee import N_SLOTS, window_centres  # noqa: E402

N_GROUPS = 15
W_GROUPS = tuple(range(6))
P_GROUPS = (6, 7, 8, 9)
Z_GROUPS = (10, 11, 12, 13)
K_GROUP = 14
PZ_SLOTS = (0, 2, 3, 5)  # sagFS, axFS, sagNFS, axNFS (slot = plane + 3 * (1 - fat_sup))
VARIANT_GROUPS = {"e1": tuple(range(10)), "e2": tuple(range(15)), "e3": tuple(range(15))}
FIELDS = ("PM_size", "PM_ft", "PL_size", "PL_ft", "FMA_size", "FMA_ft", "FLA_size", "FLA_ft")
N_THRESH = 3


def native_crop_sizes() -> dict[str, int]:
    """Native crop sizes in px {P, Z, K}: the ONE definition is v731.manifest.CROP (data worker;
    P at 0.4 mm, Z/K at 0.2 mm of the kneecap box). Never hard-coded here."""
    for p in (BRANCH_B, BRANCH_B / "v731"):  # v731.manifest imports `bands` top-level
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from v731.manifest import CROP

    return {k: int(CROP[k]) for k in ("P", "Z", "K")}


W_PX = 384


def slot_is_sagittal(slot: int) -> bool:
    return slot % 3 == 0


def mirror_native(arr: np.ndarray, sagittal: bool) -> np.ndarray:
    """Left knee -> looks like a right knee: sagittal reverses slab order, other planes flip
    columns."""
    return arr[::-1] if sagittal else arr[:, :, ::-1]


def slab_windows(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n, H, W] slabs -> ([n, 3, H, W] windows [j-1, j, j+1] clipped to the band, [n]
    positions)."""
    n = len(arr)
    if n == 0:
        return np.zeros((0, 3, *arr.shape[1:]), np.uint8), np.zeros(0, np.float32)
    # contiguous copies instead of a fancy-index gather on a (possibly flipped) memmap: the
    # gather cost ~0.2 s per 600 px series on the GPU boxes (page faults + strided reads)
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    win = np.empty((n, 3, *a.shape[1:]), np.uint8)
    win[:, 1] = a
    win[1:, 0], win[0, 0] = a[:-1], a[0]
    win[:-1, 2], win[-1, 2] = a[1:], a[-1]
    return win, (np.arange(n) / max(n - 1, 1)).astype(np.float32)


class NativeStore:
    """Lazy mmap reader of native/{P,Z,K}/<series>.npy. `get` returns None when the array is
    absent or, with a manifest, when the CURRENT manifest row (last row per (series, kind),
    v731.manifest.current_rows) is not status ok. A present array whose shape disagrees with the
    crop size or the current row's n_slab raises (never silently used)."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None,
        sizes: dict[str, int] | None = None,
        ok: set[tuple[str, str]] | None = None,
        n_slab: dict[tuple[str, str], int] | None = None,
    ):
        self.root = Path(root) if root is not None else None
        self.sizes = dict(sizes) if sizes else native_crop_sizes()
        self.ok = ok  # {(kind, series)} usable per the manifest, or None = trust file presence
        self.n_slab = n_slab or {}

    @classmethod
    def from_root(
        cls, root: str | os.PathLike[str], sizes: dict[str, int] | None = None
    ) -> NativeStore:
        """Manifest rule (PF731-DATA-002, v731.manifest): the last row per (series, kind) is
        current; only a current row with manifest.row_state(status, n_slab) == "ok" (band_too_thin
        excluded) of the current bands version (manifest.same_bands) is usable. Without a
        manifest every present array is trusted."""
        man = Path(root) / "manifest.csv"
        if not man.exists():
            return cls(root, sizes)
        for p in (BRANCH_B, BRANCH_B / "v731"):  # v731.manifest imports `bands` top-level
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from v731.manifest import current_rows, row_state, same_bands

        cur = current_rows(man)
        # usable = manifest.row_state(status, n_slab) == "ok" (so band_too_thin rows, ok with
        # n_slab < MIN_BAND_SLABS, are terminal absences) and the current bands version
        ok = cur.apply(
            lambda r: row_state(r["status"], r.get("n_slab")) == "ok" and bool(same_bands(r)),
            axis=1,
        )
        cur = cur[ok] if len(cur) else cur
        keys = list(zip(cur.kind.astype(str), cur.series.astype(str), strict=True))
        n = dict(zip(keys, cur.n_slab.astype(int), strict=True)) if "n_slab" in cur else {}
        return cls(root, sizes, set(keys), n)

    def get(self, kind: str, series: str) -> np.ndarray | None:
        if self.root is None:
            return None
        key = (kind, str(series))
        if self.ok is not None and key not in self.ok:
            return None
        path = self.root / kind / f"{series}.npy"
        if not path.exists():
            return None
        arr = np.load(path)  # whole array (every slab is used): one sequential read, no mmap
        px = self.sizes[kind]
        want_n = self.n_slab.get(key)
        if (
            arr.ndim != 3
            or arr.shape[1:] != (px, px)
            or arr.dtype != np.uint8
            or len(arr) == 0
            or (want_n is not None and len(arr) != want_n)
        ):
            raise ValueError(
                f"native {kind}/{series}.npy: shape {arr.shape} {arr.dtype}, "
                f"expected uint8 [{want_n or 'n>=1'}, {px}, {px}]"
            )
        return arr


def load_k_series(path: str | os.PathLike[str] | None) -> dict[str, tuple[str, str]]:
    """k_series.csv (study, series, rule, description, plane, gap_mm) -> {study: (series,
    plane)}."""
    if path is None or not Path(path).exists():
        return {}
    import pandas as pd

    t = pd.read_csv(path, dtype=str)
    if t.study.duplicated().any():
        raise ValueError("k_series.csv: more than one K series for a study")
    return {st: (sr, pl) for st, sr, pl in zip(t.study, t.series, t.plane, strict=True)}


class CacheLoader:
    """Picklable W loader: memory-mapped <cache_dir>/<series>.npy (V7.1 cache)."""

    def __init__(self, cache_dir: str | os.PathLike[str] | None):
        self.cache_dir = str(cache_dir)

    def __call__(self, series: str) -> np.ndarray:
        return np.load(f"{self.cache_dir}/{series}.npy", mmap_mode="r")


def _empty(px: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 3, px, px), np.uint8), np.zeros(0, np.float32)


class PFDataset(torch.utils.data.Dataset):
    """Competition exams in the contract's item format.

    studies: exam ids; table: knee.build_study_table output; cache_dir: V7.1 W cache (<series>.npy);
    native: NativeStore; y_pf: [n] float (NaN = unknown) or None; mirror: set of left-knee studies;
    k_series: {study: (series, plane)}; groups: enabled group ids (others stay empty and masked)."""

    def __init__(
        self,
        studies: Sequence[str],
        table: dict[str, list[list[str]]],
        cache_dir: str | None,
        native: NativeStore,
        y_pf: Sequence[float] | np.ndarray | None = None,
        k: int = 8,
        train: bool = False,
        mirror: Iterable[str] | None = None,
        k_series: dict[str, tuple[str, str]] | None = None,
        groups: Iterable[int] = range(15),
        w_loader: Callable[[str], np.ndarray] | None = None,
        w_px: int = W_PX,
    ):
        self.studies = [str(s) for s in studies]
        self.table, self.cache_dir, self.native = table, cache_dir, native
        self.y_pf = None if y_pf is None else np.asarray(y_pf, np.float32)
        if self.y_pf is not None and len(self.y_pf) != len(self.studies):
            raise ValueError("y_pf length != studies")
        self.k, self.train = int(k), bool(train)
        self.mirror = set(mirror or ())
        self.k_series = k_series or {}
        self.groups = frozenset(int(g) for g in groups)
        if not self.groups <= set(range(N_GROUPS)) or not self.groups >= set(W_GROUPS):
            raise ValueError(f"groups {sorted(self.groups)} must include 0..5 and lie in 0..14")
        self.w_loader = w_loader or CacheLoader(cache_dir)
        self.w_px = int(w_px)

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, i: int) -> dict[str, object]:
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        left = st in self.mirror
        # --- W: V7.1 sampling, same random-number use (knee._get_with_views_impl, view 0) ---
        w, w_mask = None, np.zeros(N_SLOTS, bool)
        w_pos = np.zeros((N_SLOTS, self.k), np.float32)
        picked: list[str | None] = [None] * N_SLOTS
        for s, cands in enumerate(slots):
            if not cands:
                continue
            sr = cands[np.random.randint(len(cands))] if self.train else cands[0]
            picked[s] = sr
            vol = self.w_loader(sr)
            if left:
                vol = vol[::-1] if slot_is_sagittal(s) else vol[:, :, ::-1]
            n = len(vol)
            c = window_centres(n, self.k, self.train)
            win = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            if w is None:
                w = np.zeros((N_SLOTS, self.k, *win.shape[1:]), np.uint8)
            w[s], w_mask[s], w_pos[s] = win, True, c / max(n - 1, 1)
        if w is None:
            w = np.zeros((N_SLOTS, self.k, 3, self.w_px, self.w_px), np.uint8)
        missing = np.zeros(N_GROUPS, bool)
        out: dict[str, object] = {"w": w, "w_mask": w_mask, "w_pos": w_pos}
        # --- K first: dedicated cartilage series. When it is the series picked for a slot's P/Z
        # in this sample, K replaces that slot's Z (the same 0.2 mm kneecap box of the same
        # slices); that slot's P stays. A K series without an array keeps Z and flags K missing.
        kv, kp = _empty(self.native.sizes["K"])
        k_mask, z_replaced = np.zeros(1, bool), -1
        if K_GROUP in self.groups and st in self.k_series:
            ksr, kplane = self.k_series[st]
            raw = self.native.get("K", ksr)
            if raw is None:
                missing[K_GROUP] = True
            else:
                if left:
                    raw = mirror_native(raw, str(kplane).lower().startswith("sag"))
                kv, kp = slab_windows(raw)
                k_mask[0] = True
                for gid, s in zip(Z_GROUPS, PZ_SLOTS, strict=True):
                    if picked[s] == ksr and gid in self.groups:
                        z_replaced = gid
                        break
        # --- P / Z: same series as the slot's W pick ---
        for kind, gids in (("p", P_GROUPS), ("z", Z_GROUPS)):
            px = self.native.sizes[kind.upper()]
            arrs, masks, poss = [], np.zeros(4, bool), []
            for gi, (gid, s) in enumerate(zip(gids, PZ_SLOTS, strict=True)):
                a, p = _empty(px)
                sr = picked[s]
                if gid in self.groups and sr is not None and gid != z_replaced:
                    raw = self.native.get(kind.upper(), sr)
                    if raw is None:
                        missing[gid] = True
                    else:
                        if left:
                            raw = mirror_native(raw, slot_is_sagittal(s))
                        a, p = slab_windows(raw)
                        masks[gi] = True
                arrs.append(a)
                poss.append(p)
            out[kind], out[f"{kind}_mask"], out[f"{kind}_pos"] = arrs, masks, poss
        out.update(
            k_view=kv,
            k_mask=k_mask,
            k_pos=kp,
            g_missing=missing,
            k_replaces_z=z_replaced >= 0,
            z_replaced_group=z_replaced,
        )
        y = math.nan if self.y_pf is None else float(self.y_pf[i])
        out.update(
            y_pf=np.float32(y),
            y_grades=np.full((len(FIELDS), N_THRESH), np.nan, np.float32),
            is_oai=False,
            study=st,
        )
        return out


def _stack_ragged(
    lists: list[list[np.ndarray]], px: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """B lists of G arrays [n, 3, px, px] -> (uint8 [B, G, Nmax, 3, px, px], token mask [B, G,
    Nmax], positions [B, G, Nmax]); Nmax = batch maximum (no truncation)."""
    b, g = len(lists), len(lists[0])
    nmax = max((len(a) for row in lists for a in row), default=0)
    x = torch.zeros((b, g, nmax, 3, px, px), dtype=torch.uint8)
    tok = torch.zeros((b, g, nmax), dtype=torch.bool)
    return x, tok, torch.zeros((b, g, nmax), dtype=torch.float32)


def collate(items: list[dict[str, object]]) -> dict[str, object]:
    """Pad every group to the batch maximum token count with explicit token masks. Never
    truncates."""
    b = len(items)
    kmax = max(np.asarray(it["w"]).shape[1] for it in items)
    w_px = np.asarray(items[0]["w"]).shape[-1]
    w = torch.zeros((b, N_SLOTS, kmax, 3, w_px, w_px), dtype=torch.uint8)
    w_tok = torch.zeros((b, N_SLOTS, kmax), dtype=torch.bool)
    w_pos = torch.zeros((b, N_SLOTS, kmax), dtype=torch.float32)
    w_mask = torch.zeros((b, N_SLOTS), dtype=torch.bool)
    for i, it in enumerate(items):
        x = torch.from_numpy(np.ascontiguousarray(it["w"]))
        m = torch.from_numpy(np.asarray(it["w_mask"], bool))
        kk = x.shape[1]
        w[i, :, :kk] = x
        w_mask[i] = m
        w_tok[i, :, :kk] = m[:, None]
        w_pos[i, :, :kk] = torch.from_numpy(np.asarray(it["w_pos"], np.float32))
    out: dict[str, object] = {"w": w, "w_mask": w_mask, "w_tok": w_tok, "w_pos": w_pos}
    for kind in ("p", "z", "k"):
        if kind == "k":
            rows = [[it["k_view"]] for it in items]
            prow = [[it["k_pos"]] for it in items]
            mrow = [np.asarray(it["k_mask"], bool).reshape(1) for it in items]
        else:
            rows = [list(it[kind]) for it in items]
            prow = [list(it[f"{kind}_pos"]) for it in items]
            mrow = [np.asarray(it[f"{kind}_mask"], bool) for it in items]
        px = np.asarray(rows[0][0]).shape[-1]
        x, tok, pos = _stack_ragged(rows, px)
        mask = torch.zeros(tok.shape[:2], dtype=torch.bool)
        for i in range(b):
            for g, (a, p) in enumerate(zip(rows[i], prow[i], strict=True)):
                n = len(a)
                if bool(mrow[i][g]) != (n > 0):
                    raise ValueError(
                        f"{items[i]['study']}: {kind}[{g}] mask disagrees with token count {n}"
                    )
                if n:
                    x[i, g, :n] = torch.from_numpy(np.ascontiguousarray(a))
                    tok[i, g, :n] = True
                    pos[i, g, :n] = torch.from_numpy(np.asarray(p, np.float32))
                    mask[i, g] = True
        out.update({kind: x, f"{kind}_tok": tok, f"{kind}_pos": pos, f"{kind}_mask": mask})
    out["y_pf"] = torch.tensor([float(it["y_pf"]) for it in items], dtype=torch.float32)
    out["y_grades"] = torch.from_numpy(
        np.stack([np.asarray(it["y_grades"], np.float32) for it in items])
    )
    out["is_oai"] = torch.tensor([bool(it["is_oai"]) for it in items], dtype=torch.bool)
    out["g_missing"] = torch.from_numpy(
        np.stack([np.asarray(it.get("g_missing", np.zeros(N_GROUPS, bool)), bool) for it in items])
    )
    out["k_replaces_z"] = torch.tensor(
        [bool(it.get("k_replaces_z", False)) for it in items], dtype=torch.bool
    )
    out["study"] = [str(it["study"]) for it in items]
    return out


def token_counts(batch: dict[str, object]) -> torch.Tensor:
    """[B, 4] valid tokens per exam: W, P, Z, K."""
    return torch.stack([batch[f"{k}_tok"].flatten(1).sum(1) for k in ("w", "p", "z", "k")], 1)


TENSOR_KEYS = (
    "w",
    "w_mask",
    "w_tok",
    "w_pos",
    "p",
    "p_tok",
    "p_pos",
    "p_mask",
    "z",
    "z_tok",
    "z_pos",
    "z_mask",
    "k",
    "k_tok",
    "k_pos",
    "k_mask",
    "y_pf",
    "y_grades",
    "is_oai",
)


def to_device(batch: dict[str, object], device: torch.device | str) -> dict[str, object]:
    out = dict(batch)
    for key in TENSOR_KEYS:
        out[key] = batch[key].to(device, non_blocking=True)
    return out

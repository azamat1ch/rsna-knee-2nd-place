"""Window-family inputs: one exam -> 320 px rows (whole field + 100 mm local views per slab) and their meta rows.

study_rows serves the store build and inference alike. A window is three slab rows of one (series, view) as
channels around a centre slab; window_rows lists them (training: covering draws, evaluation: fixed covers).
Dataset items are (img uint8 [W, 3, S, S], meta6 float32 [W, 6], index); collate pads a batch. No augmentation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from knee.centre import grid, knee_centre, local_cut, local_views
from knee.dicom import quantize, resize_physical
from knee.series import SLOTS, flip, knee_side, series_action

ROW_PX = 320                     # stored px per view (the models crop 288 or read all 320)
THIN_MM, SLAB_MM = 1.75, 3.5     # stacks spaced under 1.75 mm are averaged into 3.5 mm slabs
EXTENT_UNIT_MM = 130.0
MIN_RANGE_MM = 1e-6
META_COLUMNS = ("series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid",
                "fatsat", "slice_order", "n_slices", "real_series_idx")
META6 = (1, 3, 4, 5, 6, 7)       # slot, view_id, rel_pos, log_spacing, extent_130, fluid: the model's meta
PLANE_ID = {"sagittal": 0, "coronal": 1, "axial": 2}
PLANE_NAME = {"sagittal": "Sagittal", "coronal": "Coronal", "axial": "Axial"}
VIEW_ID = {"full": 0, "joint": 1, "medial": 2, "lateral": 3, "pf": 4, "posterior": 5, "wide": 1}
MIN_SPACING_MM = 0.1             # floor of the slab spacing in phys_step


# ---------------------------------------------------------------- rows of one exam

def slabs(positions) -> tuple[list[list[int]], float]:
    """(groups of row indices, spacing mm) of positions in stored order: a stack spaced under THIN_MM becomes
    groups of g = round(SLAB_MM / spacing); a last partial group stays when it holds at least max(1, g // 2)."""
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    spacing = float(np.median(np.diff(pos))) if n > 1 else 0.0
    groups = [[i] for i in range(n)]
    if 0 < spacing < THIN_MM:
        g = max(1, int(round(SLAB_MM / spacing)))
        count = n // g + (1 if n % g >= max(1, g // 2) else 0)
        if count >= 2:
            groups = [list(range(j * g, min(n, (j + 1) * g))) for j in range(count)]
            spacing = float(np.median(np.diff(np.array([pos[grp].mean() for grp in groups]))))
    return groups, spacing


def series_rows(units, headers: list[dict], plane: str, J, side: str, action, size: int = ROW_PX):
    """One series -> (uint8 rows [views x slabs, size, size], [(view_id, extent_mm)], slab positions, spacing).

    units(i): unit slice of headers[i] (slice order); action: series.series_action's, "reverse" turns the stack."""
    g = grid(headers)
    n = len(headers)
    order = list(range(n))[::-1] if action == "reverse" else list(range(n))
    pos = g.positions[order] * (-1.0 if action == "reverse" else 1.0)
    if np.any(np.diff(pos) <= 0):
        raise ValueError("slice positions are not strictly increasing in stored order")
    groups, spacing = slabs(pos)
    stored = [[order[i] for i in grp] for grp in groups]
    slab_pos = np.array([pos[grp].mean() for grp in groups])
    views = [("full", None, max(g.rows * g.row_mm, g.cols * g.col_mm))]
    views += [(v["name"], v["box"], v["size_mm"]) for v in local_views(headers, plane, J, side)]
    rows = np.empty((len(views) * len(stored), size, size), dtype=np.uint8)
    for k, grp in enumerate(stored):
        unit = units(grp[0]) if len(grp) == 1 else np.mean(np.stack([units(i) for i in grp]).astype(np.float64), axis=0)
        for v, (name, box, _) in enumerate(views):
            if box is None:
                picture = resize_physical(np.asarray(unit, dtype=np.float32), (g.row_mm, g.col_mm), (size, size))
            else:
                picture = local_cut(unit, box, size)
            rows[v * len(stored) + k] = quantize(flip(picture, action))
    return rows, [(VIEW_ID[name], extent) for name, _, extent in views], slab_pos, spacing


def meta_rows(series_idx: int, slot: int, plane: str, fluid: int, fat: int, views, slab_pos, spacing) -> np.ndarray:
    """float32 [views x slabs, 12] META_COLUMNS rows of one series."""
    n = len(slab_pos)
    rel = 2 * (slab_pos - slab_pos[0]) / max(slab_pos[-1] - slab_pos[0], MIN_RANGE_MM) - 1
    out = []
    for view_id, extent in views:
        for k in range(n):
            out.append([series_idx, slot, PLANE_ID[plane], view_id, rel[k], np.log1p(abs(spacing)),
                        extent / EXTENT_UNIT_MM, fluid, fat, k, n, slot])
    return np.asarray(out, dtype=np.float32)


def _in_position_order(units, headers, plane, J, side, action, size):
    """series_rows; a series not strictly increasing in its stored order is cut again in position order."""
    try:
        return series_rows(units, headers, plane, J, side, action, size)
    except ValueError as error:
        if "not strictly increasing" not in str(error):
            raise
    order = [int(i) for i in np.argsort(grid(headers).positions, kind="stable")]
    return series_rows(lambda i: units(order[i]), [headers[i] for i in order], plane, J, side, action, size)


def study_rows(study, size: int = ROW_PX) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """One exam (dicom.Study, series rule prefer-2d) -> (px uint8 [N, size, size], meta float32 [N, 12], blocks).

    blocks: per series dict(series, plane, n_slices, views [[view_id, first row]]), rows counted from the exam's
    first row. The knee centre and side are the tile family's."""
    selection = study.selection
    J = knee_centre(selection, study.rows448)
    side = knee_side(selection)["side"]
    px, meta, blocks, start = [], [], [], 0
    for value in selection.slots:
        if value is None:
            continue
        headers = [value.record["headers"][i] for i in value.sources]
        plane, fluid = value.slot
        fat = value.signature[2]
        rows, views, slab_pos, spacing = _in_position_order(lambda i, v=value: study.unit(v, i), headers, plane, J,
                                                            side, series_action(value, side), size)
        n = len(slab_pos)
        blocks.append(dict(series=value.series, plane=PLANE_NAME[plane], n_slices=n,
                           views=[[view_id, start + v * n] for v, (view_id, _) in enumerate(views)]))
        meta.append(meta_rows(len(blocks) - 1, SLOTS.index(value.slot), plane, int(fluid),
                              -1 if fat is None else int(fat), views, slab_pos, spacing))
        px.append(rows)
        start += len(rows)
    if not px:
        raise ValueError("no selected series")
    return np.concatenate(px), np.concatenate(meta), blocks


# ---------------------------------------------------------------- windows

def phys_step(spacing_mm: float, unit_mm: float) -> int:
    """Slabs per unit_mm of anatomy (1 when unit_mm <= 0)."""
    if unit_mm <= 0:
        return 1
    return max(1, int(round(unit_mm / max(float(spacing_mm), MIN_SPACING_MM))))


def covering_centers(n: int, phase: int = 1, stride: int = 3) -> list[int]:
    """Centres phase, phase + stride, ... below n, plus 0 and n - 1 when the ends are not covered."""
    if stride <= 1:
        return list(range(n))
    c = list(range(min(phase, n - 1), n, stride))
    if c[0] > 1:
        c.insert(0, 0)
    if c[-1] < n - 2:
        c.append(n - 1)
    return c


def window_rows(blocks, meta_of, mode: str, slices: int = 3, stride: int = 3, phys_mm: float = SLAB_MM,
                rng=None, jitter: float = 0.0, offset: int = 0) -> list[tuple[int, ...]]:
    """(file, centre row, row of slab -r, ..., row of slab +r) for every window of one exam.

    blocks: [dict(n_slices, views [(file, view_id, first row)])]; meta_of(file, row) -> its meta row. Per view the
    step is phys_step of its slab spacing. train: stride x step with a phase drawn per view, and per window the
    neighbours two steps apart with probability jitter; eval: stride x step at phase (1 + offset x step)."""
    out, r = [], slices // 2
    for block in blocks:
        n = block["n_slices"]
        for f, _, start in block["views"]:
            ps = phys_step(math.expm1(float(meta_of(f, start)[5])), phys_mm)
            if mode == "train":
                ks = covering_centers(n, int(rng.integers(0, stride * ps)), stride * ps)
            else:
                ks = covering_centers(n, (1 + offset * ps) % (stride * ps), stride * ps)
            for k in ks:
                step = ps * (2 if (mode == "train" and jitter > 0 and rng.random() < jitter) else 1)
                out.append((f, start + k) + tuple(start + min(n - 1, max(0, k + d * step)) for d in range(-r, r + 1)))
    return out


def gather(px_of, meta_of, rows: np.ndarray, slices: int) -> tuple[np.ndarray, np.ndarray]:
    """(img uint8 [W, slices, S, S], meta6 float32 [W, 6]) of window rows; px_of(f) / meta_of(f) are row arrays."""
    S = px_of(0).shape[-1]
    img = np.empty((len(rows), slices, S, S), dtype=np.uint8)
    m = np.empty((len(rows), meta_of(0).shape[1]), dtype=np.float32)
    for f in np.unique(rows[:, 0]):
        sel = np.flatnonzero(rows[:, 0] == f)
        r = rows[sel]
        img[sel] = np.stack([np.asarray(px_of(int(f))[r[:, 2 + d]]) for d in range(slices)], 1)
        m[sel] = np.asarray(meta_of(int(f))[r[:, 1]])
    return img, np.stack([m[:, c] for c in META6], 1).astype(np.float32)


def eval_windows(px: np.ndarray, meta: np.ndarray, blocks: list[dict], offset: int = 0, slices: int = 3,
                 stride: int = 3, phys_mm: float = SLAB_MM) -> tuple[np.ndarray, np.ndarray]:
    """Inference: the evaluation windows of one exam's study_rows at offset 0, 1 or 2."""
    views = [dict(n_slices=b["n_slices"], views=[(0, v, st) for v, st in b["views"]]) for b in blocks]
    rows = window_rows(views, lambda f, r: meta[r], "eval", slices, stride, phys_mm, offset=offset)
    if not rows:
        raise ValueError("no usable window")
    return gather(lambda f: px, lambda f: meta, np.array(rows, np.int64), slices)


# ---------------------------------------------------------------- store and dataset

class WindowStore:
    """Window store shards (px.u8.npy, meta.f32.npy, index.json); every exam in exactly one shard."""

    def __init__(self, shards):
        self.shards, self.blocks = [str(s) for s in shards], {}
        for f, shard in enumerate(self.shards):
            for study, blocks in json.loads((Path(shard) / "index.json").read_bytes())["blocks"].items():
                if study in self.blocks:
                    raise ValueError("an exam is in two window store shards")
                self.blocks[study] = [dict(b, views=[(f, v, st) for v, st in b["views"]]) for b in blocks]
        self._px = self._meta = None

    def px(self, f: int) -> np.ndarray:
        if self._px is None:
            self._px = [np.load(Path(d) / "px.u8.npy", mmap_mode="r") for d in self.shards]
        return self._px[f]

    def meta(self, f: int) -> np.ndarray:
        if self._meta is None:
            self._meta = [np.load(Path(d) / "meta.f32.npy", mmap_mode="r") for d in self.shards]
        return self._meta[f]


class WindowDataset:
    """Training items of the exams `ids`, keyed (index, nonce): the draw uses default_rng([seed, epoch, index,
    nonce]); set .epoch before each epoch. Over max_windows windows, a sorted random subset is kept."""

    def __init__(self, store: WindowStore, ids, inputs: dict, seed: int):
        missing = [s for s in ids if s not in store.blocks]
        if missing:
            raise ValueError(f"{len(missing)} exams are in no window store shard")
        self.store, self.ids, self.seed, self.epoch = store, list(ids), int(seed), 0
        self.slices, self.stride = int(inputs["slices"]), int(inputs["stride"])
        self.max_windows = int(inputs["max_windows"])
        self.phys_mm, self.jitter = float(inputs["slab_mm"]), float(inputs["slice_jitter"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, key):
        import torch
        i, nonce = key
        rng = np.random.default_rng([self.seed, int(self.epoch), int(i), int(nonce)])
        rows = window_rows(self.store.blocks[self.ids[i]], lambda f, r: self.store.meta(f)[r], "train", self.slices,
                           self.stride, self.phys_mm, rng, self.jitter)
        if len(rows) > self.max_windows:
            rows = [rows[j] for j in np.sort(rng.choice(len(rows), self.max_windows, replace=False))]
        img, meta6 = gather(self.store.px, self.store.meta, np.array(rows, np.int64), self.slices)
        return torch.from_numpy(img), torch.from_numpy(meta6), i


def training_dataset(data: Path, recipe: dict, ids, seed: int) -> WindowDataset:
    """data: a folder of window store shards (sub-folders with index.json)."""
    shards = sorted(p for p in Path(data).iterdir() if (p / "index.json").is_file())
    return WindowDataset(WindowStore(shards), ids, recipe["inputs"], seed)


def collate(batch):
    """(img uint8 [B, W, 3, S, S], meta float32 [B, W, 6], mask bool [B, W], index int64 [B]), zero padded."""
    import torch
    W = max(len(b[0]) for b in batch)
    S = batch[0][0].shape[-1]
    img = torch.zeros((len(batch), W, batch[0][0].shape[1], S, S), dtype=torch.uint8)
    meta = torch.zeros((len(batch), W, batch[0][1].shape[1]), dtype=torch.float32)
    mask = torch.zeros((len(batch), W), dtype=torch.bool)
    for j, (x, m, _) in enumerate(batch):
        img[j, :len(x)] = x
        meta[j, :len(x)] = m
        mask[j, :len(x)] = True
    return img, meta, mask, torch.tensor([b[2] for b in batch])

"""Tile-family inputs: one exam -> crop tiles for crop-convnext and lateral-effnet, one code path for both.

plan_exam fixes the series, crop windows, tiles' sources and the right-knee side; tile_item turns a plan and a
pixel source into model inputs. RawPixels cuts from a dicom.Study (inference); StorePixels reads the 448 store,
the lateral native store and a local cache of crop rows (training). Both make crop rows three source rows at a
time from the 448 rows / 255, or once from the file pixels (lateral), then round them to uint8.

Item (dict, numpy): study_id; images float32 [6V, 3, S, S] normalised by the encoder mean/std (unused views 0);
view_mask bool [6V]; view_position float32 [6V] (stack position 0..1); slots int64 [6V] (slot of each view);
slot_mask bool [6] (slots the detail branch holds); with a context branch also context_images [6V, 3, S2, S2],
context_mask, context_position. Views are slot-major, V per slot. No augmentation here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from knee import read_json, write_json
from knee.centre import centre_boxes, crop, crop_boxes, knee_centre, native_box, native_cut
from knee.dicom import CANVAS_PX, LazyRows, dequantize, quantize
from knee.series import (SLOTS, Candidate, Selection, as_right_knee, eval_tiles, flip, read_series_table, select,
                         slab_selection, slice_positions, tile_position, train_tiles)

CHUNK_ROWS = 3                       # crop rows are resized three source rows at a time, as training stored them
CONTEXT_SEED_KEY = "context_branch"  # the context branch draws its tiles from its own seed
STORE_MEMBER = "cache_private/{study}/{series}_full448.npy"
STORE_DIR, NATIVE_DIR, SERIES_CSV = "448", "lateral", "train_series.csv"
NATIVE_BLOB, NATIVE_INDEX = "crops.u8", "series_private.jsonl"


@dataclass(frozen=True)
class TileRecipe:
    """The inputs block of a tile recipe. crop "knee" or "lateral"; lateral detail tiles are cut once from the
    file pixels; context_size / context_mm add the whole-knee context branch."""
    crop: str
    size: int
    views: int
    slab_mm: float
    thin_below_mm: float
    rule: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    context_size: int | None = None
    context_mm: float | None = None

    @classmethod
    def from_recipe(cls, recipe: dict) -> "TileRecipe":
        i, e = recipe["inputs"], recipe["encoder"]
        return cls(i["crop"], int(i["size"]), int(i["views_per_slot"]), float(i["slab_mm"]), float(i["thin_below_mm"]),
                   i["series_rule"], tuple(e["mean"]), tuple(e["std"]), i.get("context_size"),
                   None if i.get("context_mm") is None else float(i["context_mm"]))

    @property
    def native(self) -> bool:
        return self.crop == "lateral"


@dataclass(frozen=True)
class Branch:
    """One branch of an exam: its selection as a right knee (crop windows set), the flip per slot and the side."""
    selection: Selection
    actions: tuple
    side: str


@dataclass(frozen=True)
class TilePlan:
    study_id: str
    detail: Branch
    context: Branch | None


def context_seed(seed: int) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps([seed, CONTEXT_SEED_KEY]).encode()).digest()[:4], "little")


def plan_exam(selection: Selection, J, recipe: TileRecipe) -> TilePlan:
    """Slabs, then the crop windows around J (None: field-centre crops), then the right-knee side per branch.
    selection: the exam's series before slabs (series.select); J: centre.knee_centre of that selection."""
    slabbed = slab_selection(selection, recipe.slab_mm, recipe.thin_below_mm)
    detail = Branch(*as_right_knee(crop_boxes(slabbed, J, recipe.crop)))
    context = None
    if recipe.context_size:
        context = Branch(*as_right_knee(centre_boxes(slabbed, recipe.context_mm)))
    return TilePlan(selection.study_id, detail, context)


# ---------------------------------------------------------------- crop rows

def crop_chunks(rows448, value: Candidate, size: int, starts) -> dict[int, np.ndarray]:
    """{row: uint8 crop} for the 3-row chunks starting at `starts` (a series under 3 rows: one chunk)."""
    n, out = len(value.sources), {}
    for s in starts:
        picked = list(range(s, min(s + CHUNK_ROWS, n)))
        unit = np.asarray(rows448[picked], dtype=np.float32) / 255
        for row, image in zip(picked, quantize(crop(unit, value, size))):
            out[row] = image
    return out


def chunk_start(row: int, n: int) -> int:
    return 0 if n < CHUNK_ROWS else min(CHUNK_ROWS * (row // CHUNK_ROWS), n - CHUNK_ROWS)


def crop_series(rows448, value: Candidate, size: int) -> np.ndarray:
    """uint8 (n, size, size) crop rows of a whole series."""
    n = len(value.sources)
    starts = sorted({chunk_start(r, n) for r in range(n)})
    done = crop_chunks(rows448, value, size, starts)
    return np.stack([done[r] for r in range(n)])


class CropRows:
    """A series' crop rows, made one 3-row chunk at a time on first read; indexed by an int or a list."""

    def __init__(self, rows448, value: Candidate, size: int):
        self.rows448, self.value, self.size, self.done = rows448, value, size, {}
        self.shape = (len(value.sources), size, size)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if isinstance(rows, (int, np.integer)):
            return self[[rows]][0]
        missing = sorted({chunk_start(r, self.shape[0]) for r in rows if r not in self.done})
        self.done.update(crop_chunks(self.rows448, self.value, self.size, missing))
        return np.stack([self.done[r] for r in rows])


# ---------------------------------------------------------------- pixel sources

class RawPixels:
    """Rows of one exam read from DICOM (dicom.raw_study), made on first read."""

    def __init__(self, study):
        self.study, self._crops = study, {}

    def rows448(self, value: Candidate):
        return self.study.rows448(value)

    def crop_rows(self, value: Candidate, size: int):
        key = (value.series, value.window, size)
        if key not in self._crops:
            self._crops[key] = CropRows(self.rows448(value), value, size)
        return self._crops[key]

    def native_rows(self, value: Candidate, size: int):
        key = ("native", value.series, value.window, size)
        if key not in self._crops:
            box = native_box(value)
            self._crops[key] = LazyRows(len(value.sources), lambda k: native_cut(self.study.unit(value, k), box, size),
                                        (size, size))
        return self._crops[key]


class Store448:
    """The 448 store: shard folders of images.zip, cache_manifest.json (member SHA-256) and series_metadata.jsonl."""

    def __init__(self, shards):
        self.members, self.records = {}, {}
        for shard in map(Path, shards):
            for member, entry in read_json(shard / "cache_manifest.json")["files"].items():
                if member in self.members:
                    raise ValueError("a 448 store member is in two shards")
                self.members[member] = (shard / entry["archive"], entry["member_sha256"])
            with (shard / "series_metadata.jsonl").open() as lines:
                for line in lines:
                    record = json.loads(line)
                    self.records.setdefault(record["study"], []).append(record)

    def member(self, value: Candidate) -> tuple[Path, str]:
        return self.members[STORE_MEMBER.format(study=value.record["study"], series=value.series)]

    def rows448(self, value: Candidate) -> np.ndarray:
        """uint8 (n, 448, 448), checked against its pinned SHA-256 and shape."""
        archive, digest = self.member(value)
        name = STORE_MEMBER.format(study=value.record["study"], series=value.series)
        with zipfile.ZipFile(archive) as z:
            raw = z.read(name)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("448 store member differs from its pinned SHA-256")
        rows = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
        if rows.dtype != np.uint8 or rows.shape != (len(value.sources), CANVAS_PX, CANVAS_PX):
            raise ValueError("448 store member has the wrong shape or type")
        return rows


class NativeStore:
    """The lateral native store: shard folders of crops.u8 (uint8 rows back to back) and series_private.jsonl."""

    def __init__(self, shards):
        self.entries = {}
        for shard in map(Path, shards):
            with (shard / NATIVE_INDEX).open() as lines:
                for line in lines:
                    entry = json.loads(line)
                    key = (entry["study"], entry["series"])
                    if key in self.entries:
                        raise ValueError("a native store series is in two shards")
                    self.entries[key] = dict(entry, blob=str(shard / NATIVE_BLOB))

    def rows(self, value: Candidate, size: int) -> np.memmap:
        """The stored rows of a series, after checking they were cut with this series' slices, window and box."""
        entry = self.entries.get((value.record["study"], value.series))
        shape = [int(x) for x in value.record["headers"][0]["shape"]]
        if (entry is None or entry["sources"] != list(value.sources) or entry["rows"] != len(value.sources)
                or entry["window"] != value.record.get("window") or entry["box"] != list(native_box(value))
                or entry["shape"] != shape):
            raise ValueError("the native store does not hold this series as it would be cut now")
        return np.memmap(entry["blob"], dtype=np.uint8, mode="r", offset=entry["offset"],
                         shape=(entry["rows"], size, size))


class StorePixels:
    """Training pixels: 448 store rows, crop rows cached once per series under `cache`, lateral native rows."""

    def __init__(self, store: Store448, cache: Path, native: NativeStore | None = None):
        self.store, self.cache, self.native = store, Path(cache), native

    def rows448(self, value: Candidate):
        return self.store.rows448(value)

    def crop_rows(self, value: Candidate, size: int):
        digest = self.store.member(value)[1]
        key = hashlib.sha256(json.dumps([digest, list(value.window), size]).encode()).hexdigest()[:32]
        path = self.cache / f"{key}.npy"
        if not path.is_file():
            rows = crop_series(self.rows448(value), value, size)
            self.cache.mkdir(parents=True, exist_ok=True)
            part = path.with_name(f"{path.name}.{os.getpid()}.part")
            with part.open("wb") as stream:
                np.save(stream, rows, allow_pickle=False)
            os.replace(part, path)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def native_rows(self, value: Candidate, size: int):
        return self.native.rows(value, size)


# ---------------------------------------------------------------- items

def _branch(branch: Branch, rows_of, size: int, views: int, mean, std, draw, study_id: str):
    """(images, view_mask, view_position) of one branch; draw(value, key) -> the tile indices of a series."""
    images = np.zeros((len(SLOTS) * views, 3, size, size), dtype=np.float32)
    mask = np.zeros(len(SLOTS) * views, dtype=bool)
    position = np.zeros(len(SLOTS) * views, dtype=np.float32)
    for slot, value in enumerate(branch.selection.slots):
        if value is None:
            continue
        rows, table = rows_of(value), slice_positions(value, branch.side)
        index = {source: row for row, source in enumerate(value.sources)}
        for j, i in enumerate(draw(value, f"{study_id}/{value.series}")):
            triple = value.triplets[i]
            if isinstance(triple[0], tuple):   # thin slabs: mean of the k cut slices per channel
                unit = np.mean([dequantize(rows[[index[a], index[b], index[c]]]) for a, b, c in zip(*triple)],
                               axis=0, dtype=np.float32)
            else:
                unit = dequantize(rows[[index[s] for s in triple]])
            view = slot * views + j
            images[view] = (unit - mean[:, None, None]) / std[:, None, None]
            mask[view] = True
            position[view] = tile_position(table, triple)
    for slot, action in enumerate(branch.actions):
        block = slice(slot * views, (slot + 1) * views)
        images[block] = flip(images[block], action).copy()
    return images, mask, position


def tile_item(plan: TilePlan, pixels, recipe: TileRecipe, *, training: bool = False, seed: int = 0,
              epoch: int = 0, shift: int = 0) -> dict:
    """Model inputs of one exam: a training draw (seed, epoch) or the evaluation tiles at offset `shift`."""
    mean, std = np.asarray(recipe.mean, dtype=np.float32), np.asarray(recipe.std, dtype=np.float32)
    views = recipe.views

    def draw_with(draw_seed):
        if training:
            return lambda value, key: train_tiles(len(value.triplets), views, draw_seed, epoch, key)
        return lambda value, key: eval_tiles(len(value.triplets), views, shift)

    if recipe.native:
        detail_rows = lambda value: pixels.native_rows(value, recipe.size)
    else:
        detail_rows = lambda value: pixels.crop_rows(value, recipe.size)
    images, mask, position = _branch(plan.detail, detail_rows, recipe.size, views, mean, std, draw_with(seed),
                                     plan.study_id)
    item = dict(study_id=plan.study_id, images=images, view_mask=mask, view_position=position,
                slots=np.repeat(np.arange(len(SLOTS)), views), slot_mask=plan.detail.selection.slot_mask)
    if plan.context is not None:
        context_rows = lambda value: pixels.crop_rows(value, recipe.context_size)
        images, mask, position = _branch(plan.context, context_rows, recipe.context_size, views, mean, std,
                                         draw_with(context_seed(seed)), plan.study_id)
        item.update(context_images=images, context_mask=mask, context_position=position)
    return item


def raw_exam(study, recipe: TileRecipe) -> tuple[TilePlan, RawPixels]:
    """Inference: the plan and pixels of one exam read by dicom.raw_study (same selection rule as the recipe)."""
    pixels = RawPixels(study)
    return plan_exam(study.selection, knee_centre(study.selection, pixels.rows448), recipe), pixels


# ---------------------------------------------------------------- training

class TileDataset:
    """Items of many exams. Training keys are (epoch, index); evaluation keys are an index (offset `shift`)."""

    def __init__(self, plans: list[TilePlan], pixels, recipe: TileRecipe, *, training: bool, seed: int = 0,
                 shift: int = 0):
        self.plans, self.pixels, self.recipe = list(plans), pixels, recipe
        self.training, self.seed, self.shift = training, seed, shift

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, key) -> dict:
        epoch, index = key if isinstance(key, tuple) else (0, key)
        return tile_item(self.plans[index], self.pixels, self.recipe, training=self.training, seed=self.seed,
                         epoch=int(epoch), shift=self.shift)


class EpochSampler:
    """(epoch, index) keys in a shuffled order that is a pure function of (seed, epoch)."""

    def __init__(self, length: int, seed):
        self.length, self.seed, self.epoch = length, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        digest = hashlib.sha256(json.dumps([self.seed, self.epoch]).encode()).digest()
        order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(self.length)
        return iter([(self.epoch, int(i)) for i in order])


class _Centres:
    def __init__(self, selections, pixels):
        self.selections, self.pixels = selections, pixels

    def __len__(self):
        return len(self.selections)

    def __getitem__(self, i):
        return self.selections[i].study_id, knee_centre(self.selections[i], self.pixels.rows448)


def store_plans(data: Path, recipe: TileRecipe, ids, cache: Path, workers: int = 0):
    """(plans, StorePixels) of the training exams `ids`. data holds 448/<shards>, train_series.csv and, for the
    lateral crop, lateral/<shards>. Knee centres are found once and kept in cache/centres.json."""
    import torch
    data, cache = Path(data), Path(cache)
    store = Store448(sorted(p for p in (data / STORE_DIR).iterdir() if p.is_dir()))
    native = NativeStore(sorted(p for p in (data / NATIVE_DIR).iterdir() if p.is_dir())) if recipe.native else None
    pixels = StorePixels(store, cache, native)
    table = read_series_table(data / SERIES_CSV)
    selections = [select(store.records[s], table[s], s, recipe.rule) for s in ids]
    path = cache / "centres.json"
    centres = read_json(path) if path.is_file() else {}
    todo = [s for s in selections if s.study_id not in centres]
    if todo:
        loader = torch.utils.data.DataLoader(_Centres(todo, pixels), batch_size=None, num_workers=workers,
                                             collate_fn=lambda x: x)
        centres.update(dict(loader))
        cache.mkdir(parents=True, exist_ok=True)
        write_json(path, centres)
    return [plan_exam(s, centres[s.study_id], recipe) for s in selections], pixels


def training_dataset(data: Path, recipe: dict, ids, seed: int, cache: Path | None = None,
                     workers: int = os.cpu_count() or 1) -> TileDataset:
    """The training items of `ids` (in that order), keyed (epoch, index)."""
    tile = TileRecipe.from_recipe(recipe)
    plans, pixels = store_plans(data, tile, ids, Path(data) / "cache" if cache is None else cache, workers)
    return TileDataset(plans, pixels, tile, training=True, seed=seed)

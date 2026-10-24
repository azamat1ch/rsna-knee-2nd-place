"""One member on the test exams: DICOM -> the recipe's inputs -> every seed model -> one probability CSV.

Tile recipes: one worker process per GPU on interleaved exams; loader processes build each exam's tiles at every
offset. Window recipes: reader processes cut each exam's rows; one thread per GPU runs every seed on them. fp16
autocast on CUDA, with an fp32 rerun where an output is not finite; fp32 on CPU. An exam that fails keeps the
recipe's prior, and then (as on a deadline or a failed worker) the CSV is still written but the exit code is 1.

python -m knee.infer --recipe configs/X.json --weights MODELS_DIR --data COMPETITION_DIR --out X.csv --deadline S
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from knee import LABELS, read_json, write_json

ID = "StudyInstanceUID"
ENCODER_PAD = 16          # tile encoders see batches padded to a multiple of 16 views (few kernel plans per GPU)
LOADER_PREFETCH = 2       # exams each tile loader process builds ahead
CPU_LOADERS = 1           # tile loader processes without a GPU (memory: an exam's tiles are ~0.5 GB)
KILL_GRACE_S = 300        # a tile worker still running this long after the deadline is killed
WINDOW_READERS = 4        # window row reader processes
READER_TASKS = 25         # exams per reader process before it is replaced
AHEAD_PER_READER = 2      # exams handed to each reader and not yet taken by a GPU thread
WINDOW_CHUNK = 512        # windows per encoder call
WINDOW_THREADS = 2        # torch threads of the window process (the readers use 1 each)
CSV_FORMAT = {"tile": lambda v: format(float(v), ".9g"), "window": lambda v: "%.6f" % v}


# ---------------------------------------------------------------- exams and output

def exam_order(data: Path) -> tuple[list[str], list[str]]:
    """(exams in run order: test.csv first when present, then the rest of sample_submission; sample order)."""
    with (data / "sample_submission.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    if rows[0] != [ID, *LABELS]:
        raise ValueError("sample_submission.csv columns differ from LABELS")
    sample = [r[0] for r in rows[1:]]
    listed = sample
    if (data / "test.csv").is_file():
        with (data / "test.csv").open(newline="") as stream:
            listed = [r[ID] for r in csv.DictReader(stream)]
    order = listed + [s for s in sample if s not in set(listed)]
    if len(set(order)) != len(order) or set(order) != set(sample):
        raise ValueError("test.csv and sample_submission.csv list different exams")
    return order, sample


def series_rows(data: Path, ids: list[str]) -> dict[str, list[dict]]:
    """{exam: its test_series.csv rows in SeriesInstanceUID order}."""
    grouped = {s: [] for s in ids}
    with (data / "test_series.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row[ID] in grouped:
                grouped[row[ID]].append(row)
    return {s: sorted(rows, key=lambda r: r["SeriesInstanceUID"]) for s, rows in grouped.items()}


def write_member(path: Path, ids: list[str], preds: dict[str, np.ndarray], prior, family: str) -> None:
    """CSV in sample_submission order, the prior where an exam has no prediction; atomic replace."""
    fmt, part = CSV_FORMAT[family], path.with_name(f".{path.name}.part")
    with part.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([ID, *LABELS])
        for study in ids:
            writer.writerow([study, *(fmt(v) for v in preds.get(study, prior))])
    os.replace(part, path)


def load_models(recipe: dict, weights: Path, device) -> list:
    """The recipe's seed models (strict, sha-pinned) on `device`, in the recipe's file order."""
    import torch
    from knee import models
    out = []
    for rel, digest in recipe["weights"]["files"].items():
        model = models.load(recipe, weights / rel, digest).requires_grad_(False).to(device)
        if recipe["family"] == "tile":
            model.encoder_pad = ENCODER_PAD
        else:
            model = model.to(memory_format=torch.channels_last)
        out.append(model)
    return out


def autocast(fp16: bool):
    import torch
    return torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()


# ---------------------------------------------------------------- tile family

class TileExams:
    """Dataset of exams -> dict(study, passes: one tiles.tile_item per eval offset) or dict(study, error)."""

    def __init__(self, ids, rows, image_root, recipe: dict):
        from knee.tiles import TileRecipe
        self.ids, self.rows, self.root = list(ids), rows, Path(image_root)
        self.tile, self.offsets = TileRecipe.from_recipe(recipe), list(recipe["infer"]["offsets"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from knee.dicom import raw_study
        from knee.tiles import raw_exam, tile_item
        study = self.ids[i]
        try:
            plan, pixels = raw_exam(raw_study(self.root / study, self.rows[study], self.tile.rule), self.tile)
            return dict(study=study, passes=[tile_item(plan, pixels, self.tile, shift=o) for o in self.offsets])
        except Exception as error:
            return dict(study=study, error=f"inputs:{type(error).__name__}")


def _identity(item):
    return item


def tile_probabilities(model, inputs: dict, fp16: bool, counts: dict):
    """float32 (12,) sigmoid of one pass, or None when it stays non-finite (fp16 first, then fp32)."""
    import torch
    for half in ([True, False] if fp16 else [False]):
        with torch.no_grad(), autocast(half):
            logits = model(**inputs).float()
        if torch.isfinite(logits).all():
            p = logits.sigmoid()[0]
            if torch.isfinite(p).all():
                if not half and fp16:
                    counts["fp32_rescued"] += 1
                return p.cpu().numpy()
        if half:
            counts["fp16_nonfinite"] += 1
    return None


def tile_exam(passes: list[dict], models: list, device, fp16: bool, counts: dict) -> list:
    """Per seed model: the float32 mean over the offset passes, or None when a pass failed."""
    import torch
    keys = ("images", "view_mask", "view_position", "context_images", "context_mask", "context_position")
    inputs = [{k: torch.from_numpy(np.ascontiguousarray(p[k]))[None].to(device) for k in keys if k in p} for p in passes]
    out = []
    for model in models:
        runs = []
        for x in inputs:
            try:
                runs.append(tile_probabilities(model, x, fp16, counts))
            except Exception:
                runs.append(None)
        out.append(None if any(r is None for r in runs) else
                   np.stack([np.asarray(r, np.float32) for r in runs]).mean(axis=0, dtype=np.float32).tolist())
    return out


def tile_worker(spec_path: str) -> None:
    """One device: load the seeds, stream the spec's exams through loader processes, append one JSON line each."""
    import torch
    spec = read_json(spec_path)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_num_threads(1)
    device = torch.device("cuda:0" if cuda else "cpu")
    recipe = read_json(spec["recipe"])
    fp16 = cuda and recipe["infer"]["precision"] == "fp16"
    models = load_models(recipe, Path(spec["weights"]), device)
    data = TileExams(spec["studies"], spec["rows"], spec["image_root"], recipe)
    loader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=False, num_workers=spec["loaders"],
                                         collate_fn=_identity,
                                         prefetch_factor=LOADER_PREFETCH if spec["loaders"] else None)
    counts, status = dict(fp16_nonfinite=0, fp32_rescued=0), "finished"
    with Path(spec["results"]).open("a") as sink:
        for item in loader:
            if time.time() > spec["deadline_unix"]:
                status = "deadline"
                break
            line = dict(study=item["study"])
            if "error" in item:
                line["error"] = item["error"]
            else:
                try:
                    line["members"] = tile_exam(item["passes"], models, device, fp16, counts)
                except Exception as error:
                    line["error"] = f"model:{type(error).__name__}"
            sink.write(json.dumps(line) + "\n")
            sink.flush()
    write_json(spec["status"], dict(status=status, device=str(device), fp16=fp16, **counts))


def run_tiles(recipe: dict, recipe_path: Path, weights: Path, data: Path, order: list[str], rows: dict,
              deadline_unix: float, work: Path) -> tuple[dict, dict]:
    """(exam -> float64 (12,) seed mean, receipt part). One worker process per GPU (interleaved exams), else one
    in-process worker on the CPU."""
    import torch
    gpus = torch.cuda.device_count()
    shards = [order[g::gpus] for g in range(gpus)] if gpus else [order]
    loaders = max(1, (os.cpu_count() or 4) // gpus) if gpus else CPU_LOADERS
    specs, procs = [], []
    for g, shard in enumerate(shards):
        spec = dict(recipe=str(recipe_path), weights=str(weights), image_root=str(data / "test_series"),
                    studies=shard, rows={s: rows[s] for s in shard}, loaders=loaders, deadline_unix=deadline_unix,
                    results=str(work / f"worker{g}.jsonl"), status=str(work / f"worker{g}-status.json"))
        write_json(work / f"worker{g}.json", spec, indent=None)
        specs.append(spec)
    if not gpus:
        tile_worker(str(work / "worker0.json"))
        codes = [0]
    else:
        package = str(Path(__file__).resolve().parents[1])
        for g in range(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1",
                       PYTHONPATH=os.pathsep.join(filter(None, [package, os.environ.get("PYTHONPATH")])))
            with (work / f"worker{g}.log").open("w") as log:
                procs.append(subprocess.Popen(
                    [sys.executable, "-c", "import sys; from knee.infer import tile_worker; tile_worker(sys.argv[1])",
                     str(work / f"worker{g}.json")], env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = []
        for proc in procs:
            try:
                codes.append(proc.wait(timeout=max(1.0, deadline_unix + KILL_GRACE_S - time.time())))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                codes.append("killed")
    results, workers = {}, []
    for g, (spec, code) in enumerate(zip(specs, codes)):
        path = Path(spec["results"])
        lines = path.read_text().splitlines() if path.is_file() else []
        for line in lines:
            record = json.loads(line)
            results[record["study"]] = record
        status = read_json(spec["status"]) if Path(spec["status"]).is_file() else {}
        log = work / f"worker{g}.log"
        workers.append(dict(code=code, exams=len(spec["studies"]), results=len(lines), **status,
                            log_tail=None if code == 0 or not log.is_file() else log.read_text()[-2000:]))
    n_models = len(recipe["weights"]["files"])
    raw = np.full((n_models, len(order), len(LABELS)), np.nan)
    errors = {}
    for i, study in enumerate(order):
        line = results.get(study, {"error": "no result"})
        if "error" in line:
            errors[line["error"]] = errors.get(line["error"], 0) + 1
            continue
        for m, value in enumerate(line["members"]):
            if value is not None:
                raw[m, i] = value
    some = np.isfinite(raw).all(axis=2).any(axis=0)
    mean = np.full((len(order), len(LABELS)), np.nan)
    mean[some] = np.nanmean(raw[:, some], axis=0)
    preds = {study: mean[i] for i, study in enumerate(order) if some[i]}
    part = dict(workers=workers, failures=errors,
                exams_partial_seeds=int((~np.isfinite(raw).all(axis=2) & some).any(axis=0).sum()),
                worker_failed=any(c != 0 or "status" not in w for c, w in zip(codes, workers)),
                deadline_hit=any(w.get("status") == "deadline" for w in workers))
    return preds, part


# ---------------------------------------------------------------- window family

READER = {}   # the reader processes' view of the job, set before the pool forks


def _reader_init() -> None:
    import torch
    torch.set_num_threads(1)


def window_rows(study: str):
    """Reader: (study, (px, meta, blocks) | None, error type | None)."""
    from knee.dicom import raw_study
    from knee.windows import study_rows
    try:
        exam = raw_study(READER["root"] / study, READER["rows"][study], READER["rule"])
        return study, study_rows(exam, READER["size"]), None
    except Exception as error:
        return study, None, f"rows:{type(error).__name__}"


def bounded(pool, fn, jobs, ahead: int):
    """AsyncResults of fn over jobs in job order, with at most `ahead` jobs handed out and not yet taken."""
    from collections import deque
    jobs, waiting = iter(jobs), deque()
    for job in jobs:
        waiting.append(pool.apply_async(fn, (job,)))
        if len(waiting) >= ahead:
            break
    while waiting:
        yield waiting.popleft()
        for job in jobs:
            waiting.append(pool.apply_async(fn, (job,)))
            break


def window_exam(rows, models: list, device, inputs: dict, offsets, fp16: bool, counts: dict, lock) -> np.ndarray:
    """float64 (12,): per offset the sigmoid of the seed-mean logits, then the mean over offsets."""
    import torch
    from knee.windows import eval_windows
    px, meta, blocks = rows
    crop = int(inputs["crop"])
    off = (int(inputs["store_size"]) - crop) // 2
    ps = []
    for offset in offsets:
        img, meta6 = eval_windows(px, meta, blocks, offset, int(inputs["slices"]), int(inputs["stride"]),
                                  float(inputs["slab_mm"]))
        with torch.no_grad():
            x = torch.from_numpy(img)[None, :, :, off:off + crop, off:off + crop].to(device)
            mt = torch.from_numpy(meta6)[None].to(device)
            mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
            zs = []
            for model in models:
                with autocast(fp16):
                    z = model(x, mt, mask, chunk=WINDOW_CHUNK).float()
                if fp16 and not torch.isfinite(z).all():
                    with lock:
                        counts["fp16_nonfinite"] += 1
                    z = model(x, mt, mask, chunk=WINDOW_CHUNK).float()
                    with lock:
                        counts["fp32_rescued"] += 1
                zs.append(z[0])
            z = torch.stack(zs).mean(0).double().cpu().numpy()
        if not np.isfinite(z).all():
            raise ValueError("non-finite after the fp32 rerun")
        ps.append(1 / (1 + np.exp(-z)))
    return ps[0] if len(ps) == 1 else np.mean(ps, 0)


def run_windows(recipe: dict, weights: Path, data: Path, order: list[str], rows: dict,
                deadline_unix: float) -> tuple[dict, dict]:
    """(exam -> float64 (12,), receipt part): reader processes -> a queue of one slot per device -> one thread per
    device holding every seed. On the deadline the exams not yet done keep the prior."""
    import torch
    inputs = recipe["inputs"]
    READER.update(root=data / "test_series", rows=rows, rule=inputs["series_rule"], size=int(inputs["store_size"]))
    gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{g}") for g in range(gpus)] or [torch.device("cpu")]
    fp16 = bool(gpus) and recipe["infer"]["precision"] == "fp16"
    torch.backends.cudnn.benchmark = False
    models = [load_models(recipe, weights, d) for d in devices]
    torch.set_num_threads(WINDOW_THREADS)
    offsets = list(recipe["infer"]["offsets"])
    lock, stop, todo = threading.Lock(), threading.Event(), queue.Queue(maxsize=len(devices))
    preds, errors, fatal = {}, {}, []
    counts = dict(fp16_nonfinite=0, fp32_rescued=0)

    def fail(kind):
        with lock:
            errors[kind] = errors.get(kind, 0) + 1

    def device_loop(g):
        try:
            while (job := todo.get()) is not None:
                if stop.is_set():
                    continue
                study, exam = job
                try:
                    with torch.cuda.device(devices[g]) if gpus else nullcontext():
                        p = window_exam(exam, models[g], devices[g], inputs, offsets, fp16, counts, lock)
                        if gpus:
                            torch.cuda.synchronize(devices[g])
                    with lock:
                        preds[study] = p
                except Exception as error:
                    fail(f"model:{type(error).__name__}")
        except BaseException as error:
            fatal.append(f"device {g}: {type(error).__name__}: {error}"[:500])
            stop.set()

    deadline_hit = False
    readers = max(1, min(WINDOW_READERS, os.cpu_count() or 1))
    with multiprocessing.get_context("fork").Pool(readers, initializer=_reader_init,
                                                  maxtasksperchild=READER_TASKS) as pool:
        threads = [threading.Thread(target=device_loop, args=(g,), daemon=True) for g in range(len(devices))]
        for t in threads:
            t.start()
        try:
            for result in bounded(pool, window_rows, order, AHEAD_PER_READER * readers):
                left = deadline_unix - time.time()
                try:
                    if left <= 0:
                        raise multiprocessing.TimeoutError
                    study, exam, error = result.get(timeout=left)
                except multiprocessing.TimeoutError:
                    deadline_hit = True
                    break
                if error is not None:
                    fail(error)
                    continue
                while not fatal:
                    try:
                        todo.put((study, exam), timeout=1)
                        break
                    except queue.Full:
                        if time.time() > deadline_unix:
                            deadline_hit = True
                            break
                if fatal or deadline_hit:
                    break
        finally:
            if deadline_hit or fatal:
                stop.set()
            for _ in threads:
                while any(t.is_alive() for t in threads):
                    try:
                        todo.put(None, timeout=1)
                        break
                    except queue.Full:
                        pass
            for t in threads:
                t.join()
    left_out = len(order) - len(preds) - sum(errors.values())
    if left_out:
        errors["not reached"] = left_out
    part = dict(devices=[str(d) for d in devices], fp16=fp16, readers=readers, failures=errors, **counts,
                worker_failed=bool(fatal), fatal=fatal, deadline_hit=deadline_hit)
    return preds, part


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", type=Path, required=True, help="configs/<member>.json")
    ap.add_argument("--weights", type=Path, required=True, help="models dataset root (the recipe's files are under it)")
    ap.add_argument("--data", type=Path, required=True, help="folder of test_series.csv, test_series/, sample_submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="member CSV; the receipt goes beside it")
    ap.add_argument("--deadline", type=float, required=True, help="seconds from start; then the rest keeps the prior")
    args = ap.parse_args(argv)
    started = time.time()
    deadline_unix = started + args.deadline
    recipe = read_json(args.recipe)
    order, sample = exam_order(args.data)
    rows = series_rows(args.data, order)
    prior = np.asarray(recipe["infer"]["prior"], dtype=np.float64)
    work = args.out.with_name(f".{args.out.stem}-work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if recipe["family"] == "tile":
        preds, part = run_tiles(recipe, args.recipe, args.weights, args.data, order, rows, deadline_unix, work)
    else:
        preds, part = run_windows(recipe, args.weights, args.data, order, rows, deadline_unix)
    write_member(args.out, sample, preds, prior, recipe["family"])
    fallback = len(sample) - len(preds)
    receipt = dict(member=recipe["name"], exams=len(sample), predicted=len(preds), prior_fallback=fallback,
                   seconds=round(time.time() - started, 1), deadline_s=args.deadline, **part)
    receipt["ok"] = not (fallback or part["deadline_hit"] or part["worker_failed"])
    write_json(args.out.with_name(f"{args.out.stem}-receipt.json"), receipt)
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: receipt[k] for k in ("member", "exams", "predicted", "prior_fallback", "failures",
                                              "deadline_hit", "worker_failed", "seconds", "ok")}), flush=True)
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

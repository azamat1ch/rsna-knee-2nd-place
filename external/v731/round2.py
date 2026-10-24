"""V7.3.1 round 2 plumbing (``docs/intent/v7_3_1_plan.md`` "Round 2"). Never submits.

(a) ``retrain-plan``: the round-1 winner's unchanged config retrained on ALL 4345 exams (3906
    train + 439 develop; gold58 never) with 2-3 seeds through train.py's own interface
    (--cohort round2 --seed S [--label-source d3]); D4 labels, D3 only as the fallback after a
    public-score drop. Prints the jobs; writes and runs nothing.
(b) ``replace-pf``: given the TEAM BLEND submission CSV (StudyInstanceUID + 12 findings; PF column
    ``PF OA``) and the specialist's test PF predictions, writes
      candidate.csv = the team blend with ONLY ``PF OA`` replaced by
                      0.5 * pct_rank(team PF) + 0.5 * pct_rank(specialist PF)
                      (average ranks for ties, pct = rank / n), every other cell byte-identical;
      control.csv   = the same team blend without the specialist (byte copy of the input);
      REPLACE_PF.json receipt (sha256 of inputs/outputs, id/order/shape checks).
    Verifies: 13 columns exactly, unique ids, same id set in both files (and, with --sample, the
    sample_submission order and columns), finite PF values; output ids/order/shape equal the
    input's.
    The team blend is a code-competition notebook output: for a real submission this function runs
    INSIDE the Kaggle notebook after the team blend and the specialist inference
    (``kaggle_infer.py``)
    have written their CSVs (see ``commands``). The team blend's source kernel is not local
    (BLOCKER recorded by ``commands``).
(c) ``commands``: prints the exact (not executed) Kaggle steps for the founder.

Pure stdlib for (b) so it runs unchanged inside a Kaggle kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

ID = "StudyInstanceUID"
PF = "PF OA"
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
COLUMNS = [ID, *LABELS]
COMPETITION = "rsna-knee-abnormality-detection"
SEEDS_DEFAULT = (0, 1, 2)
N_ALL = 4345  # 3906 train + 439 develop (gold58 excluded)
N_FINAL = 4403  # final fit: N_ALL + the 58 on official labels (train.py --cohort final4403)
TEAM_BLEND_BLOCKER = (
    "team blend source kernel unavailable locally: public 0.967 submissions 57247642 'Team blend A "
    "(conservative)' and 57247653 'Team blend B' come from a kernel outside matthewliew's account "
    "(likely azamat's); round 2 cannot run until its notebook source (or a kernel we can attach) "
    "is in hand"
)


class Round2Error(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- ranks
def pct_rank(values: list[float]) -> list[float]:
    """Average-rank percentiles: rank (1..n, ties share their mean rank) / n. Same as
    scipy.stats.rankdata(v, method='average') / n."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1  # mean of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return [r / n for r in ranks]


def rank_blend(team: list[float], spec: list[float]) -> list[float]:
    if len(team) != len(spec):
        raise Round2Error("length mismatch")
    a, b = pct_rank(team), pct_rank(spec)
    return [0.5 * x + 0.5 * y for x, y in zip(a, b, strict=True)]


# PF replacement rules: "rank50" = 0.5*pct_rank(team PF) + 0.5*pct_rank(specialist PF) (the
# default, unchanged); "specialist" = the specialist's PF probability itself (team PF unused).
PF_RULES = ("rank50", "specialist")


def new_pf(team: list[float], spec: list[float], rule: str = "rank50") -> list[float]:
    if rule == "rank50":
        return rank_blend(team, spec)
    if rule == "specialist":
        if len(team) != len(spec):
            raise Round2Error("length mismatch")
        return [float(v) for v in spec]
    raise Round2Error(f"unknown PF rule {rule!r} (one of {PF_RULES})")


# ----------------------------------------------------------------------------- CSV I/O
def read_rows(data: bytes) -> tuple[list[str], list[list[str]], str]:
    """(header, rows of raw cell strings, line terminator) without any float conversion."""
    text = data.decode("utf-8")
    term = "\r\n" if "\r\n" in text.split("\n", 1)[0] + "\n" else "\n"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    rows = [r for r in rows if r]  # a trailing blank line is not a row
    if not rows:
        raise Round2Error("empty CSV")
    return rows[0], rows[1:], term


def finite(cell: str, what: str) -> float:
    try:
        v = float(cell)
    except ValueError as e:
        raise Round2Error(f"{what}: not a number {cell!r}") from e
    if not math.isfinite(v):
        raise Round2Error(f"{what}: non-finite {cell!r}")
    return v


def check_team(header: list[str], rows: list[list[str]]) -> None:
    if header != COLUMNS:
        raise Round2Error(f"team blend columns {header} != expected 13 columns {COLUMNS}")
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise Round2Error("team blend has duplicate StudyInstanceUID")
    for r in rows:
        if len(r) != len(COLUMNS):
            raise Round2Error(f"team blend row {r[0]}: {len(r)} cells")
        for c, cell in zip(COLUMNS[1:], r[1:], strict=True):
            finite(cell, f"team {r[0]} {c}")


def read_specialist(data: bytes) -> dict[str, float]:
    """Specialist predictions: StudyInstanceUID + one PF column ('PF OA', 'pf' or 'pf_prob')."""
    header, rows, _ = read_rows(data)
    if ID not in header:
        raise Round2Error(f"specialist CSV lacks {ID}")
    col = next((c for c in (PF, "pf", "pf_prob", "PF") if c in header), None)
    if col is None:
        raise Round2Error(f"specialist CSV has no PF column (header {header})")
    i, j = header.index(ID), header.index(col)
    out: dict[str, float] = {}
    for r in rows:
        if r[i] in out:
            raise Round2Error(f"specialist duplicate id {r[i]}")
        out[r[i]] = finite(r[j], f"specialist {r[i]}")
    return out


def fmt(v: float) -> str:
    return repr(float(v))  # shortest round-trip text


def replace_pf(
    team_bytes: bytes,
    spec: dict[str, float],
    sample_bytes: bytes | None = None,
    rule: str = "rank50",
) -> tuple[bytes, dict[str, Any]]:
    """Team blend CSV bytes -> (candidate CSV bytes, checks). Only the PF OA cells change, to
    new_pf(team PF, specialist PF, rule)."""
    header, rows, term = read_rows(team_bytes)
    check_team(header, rows)
    ids = [r[0] for r in rows]
    if set(ids) != set(spec):
        raise Round2Error(
            f"id sets differ: team {len(ids)}, specialist {len(spec)}, "
            f"only team {len(set(ids) - set(spec))}, only specialist {len(set(spec) - set(ids))}"
        )
    checks: dict[str, Any] = {"n": len(rows), "columns": header, "id_sets_equal": True}
    if sample_bytes is not None:
        sh, srows, _ = read_rows(sample_bytes)
        if sh != COLUMNS:
            raise Round2Error(f"sample_submission columns {sh} != {COLUMNS}")
        if [r[0] for r in srows] != ids:
            raise Round2Error("team blend rows are not in sample_submission order")
        checks["sample_order_equal"] = True
    j = COLUMNS.index(PF)
    team_pf = [float(r[j]) for r in rows]
    new = new_pf(team_pf, [spec[i] for i in ids], rule)
    out_rows = [[*r[:j], fmt(v), *r[j + 1 :]] for r, v in zip(rows, new, strict=True)]
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator=term)
    w.writerow(header)
    w.writerows(out_rows)
    out = buf.getvalue().encode("utf-8")
    verify_candidate(team_bytes, out, spec, rule)
    checks.update(
        pf_rule=rule,
        pf_min=min(new),
        pf_max=max(new),
        other_columns_identical=True,
        order_identical=True,
    )
    return out, checks


def verify_candidate(
    team_bytes: bytes, cand_bytes: bytes, spec: dict[str, float], rule: str = "rank50"
) -> None:
    """Re-read both files: same header, ids, order, shape; all 11 other columns string-identical;
    PF equals new_pf(team PF, specialist PF, rule) (default: the rank blend)."""
    th, trows, _ = read_rows(team_bytes)
    ch, crows, _ = read_rows(cand_bytes)
    if th != ch or len(trows) != len(crows):
        raise Round2Error("candidate header/shape differs from team blend")
    j = COLUMNS.index(PF)
    expect = new_pf([float(r[j]) for r in trows], [spec[r[0]] for r in trows], rule)
    for t, c, e in zip(trows, crows, expect, strict=True):
        if t[0] != c[0]:
            raise Round2Error("candidate id order differs")
        if t[:j] + t[j + 1 :] != c[:j] + c[j + 1 :]:
            raise Round2Error(f"{t[0]}: a non-PF cell changed")
        if float(c[j]) != e:
            raise Round2Error(f"{t[0]}: PF {c[j]} != {rule} {e}")


def write_fresh(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise Round2Error(f"{path} exists with different content; never overwritten")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def run_replace(
    team: Path, spec_csv: Path, out_dir: Path, sample: Path | None = None
) -> dict[str, Any]:
    tb, sb = team.read_bytes(), spec_csv.read_bytes()
    smp = sample.read_bytes() if sample else None
    cand, checks = replace_pf(tb, read_specialist(sb), smp)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fresh(out_dir / "candidate.csv", cand)
    write_fresh(out_dir / "control.csv", tb)  # the same team blend without the specialist
    rec = {
        "schema": "v731-replace-pf-1",
        "rule": "PF OA = 0.5*pct_rank(team PF) + 0.5*pct_rank(specialist PF), "
        "average ranks for ties, pct = rank/n; other 11 columns unchanged",
        "team_blend": {"path": str(team), "sha256": sha256_bytes(tb)},
        "specialist": {"path": str(spec_csv), "sha256": sha256_bytes(sb)},
        "sample": {"path": str(sample), "sha256": sha256_bytes(smp)} if smp else None,
        "candidate_sha256": sha256_bytes(cand),
        "control_sha256": sha256_bytes(tb),
        "checks": checks,
    }
    write_fresh(out_dir / "REPLACE_PF.json", (json.dumps(rec, indent=2) + "\n").encode())
    return rec


# ----------------------------------------------------------------------------- seed averaging
def average_seeds(csvs: list[bytes], col: str = PF) -> dict[str, float]:
    """Mean probability per study over seed CSVs (StudyInstanceUID + col). All seeds must cover the
    same study set exactly; NaN/inf refused."""
    if not csvs:
        raise Round2Error("no seed predictions")
    per = []
    for b in csvs:
        h, rows, _ = read_rows(b)
        if ID not in h or col not in h:
            raise Round2Error(f"seed CSV lacks {ID}/{col}")
        i, j = h.index(ID), h.index(col)
        d: dict[str, float] = {}
        for r in rows:
            if r[i] in d:
                raise Round2Error(f"duplicate id {r[i]}")
            d[r[i]] = finite(r[j], f"seed {r[i]}")
        per.append(d)
    keys = set(per[0])
    if any(set(d) != keys for d in per):
        raise Round2Error("seed predictions cover different studies")
    return {k: sum(d[k] for d in per) / len(per) for k in per[0]}


# ----------------------------------------------------------------------------- retrain plan
ROUND2_SEEDS = (0, 1, 2)  # train.py ROUND2_SEEDS


def round2_argv(base_argv: list[str], seed: int, d3: bool, cohort: str = "round2") -> list[str]:
    """train.py's own round-2 interface: --cohort round2|final4403 --seed S [--label-source d3].
    The D3 file comes from the config's inputs.d3 (train.py Inputs.d3, sha-pinned by train.py)."""
    if seed not in ROUND2_SEEDS:
        raise Round2Error(f"round-2 seed {seed} not in {ROUND2_SEEDS}")
    if cohort not in ("round2", "final4403") or (d3 and cohort != "round2"):
        raise Round2Error(f"cohort {cohort!r} (D3 is the round-2 fallback only)")
    out = [*base_argv, "--cohort", cohort, "--seed", str(seed)]
    if d3:
        out += ["--label-source", "d3"]
    return out


def retrain_plan(
    winner: str,
    seeds: tuple[int, ...] = SEEDS_DEFAULT,
    d3: bool = False,
    runs_root: Path | None = None,
    base_configs: dict[str, Path] | None = None,
    final: bool = False,
) -> dict[str, Any]:
    """Round-2 jobs: the round-1 winner's UNCHANGED config on all 4345 (train + develop, never
    gold58; train.py --cohort round2 holds nothing out), 2-3 distinct seeds, final EMA, D4 labels
    (D3 with ``d3``: the fallback after a public-score drop). ``final``: the founder-approved
    V7.2-style final fit on 4403 = 4345 + the 58 on official labels (--cohort final4403, D4 only).
    Nothing is run or written."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import launch as L

    if winner not in L.RUNS:
        raise Round2Error(f"winner must be one of {L.RUNS}")
    if not 2 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise Round2Error("round 2 uses 2-3 distinct seeds")
    base = (base_configs or L.CONFIGS)[winner]
    tag = "d3" if d3 else "d4"
    cohort, n_train = ("final4403", N_FINAL) if final else ("round2", N_ALL)
    if final and d3:
        raise Round2Error("the final fit uses D4 (+ official gold labels) only")
    runs_root = runs_root or (L.V731_IN / "runs" / (f"final_{tag}" if final else f"round2_{tag}"))
    steps = round(L.EPOCHS * L.STEPS_PER_EPOCH_3906 * n_train / 3906)
    jobs = []
    for s in seeds:
        out = runs_root / f"{winner}_s{s}"
        argv = round2_argv(L.train_argv(base, out, s, "cuda"), s, d3, cohort)
        jobs.append(
            {"run": winner, "seed": s, "out_dir": str(out), "config": str(base), "argv": argv}
        )
    return {
        "winner": winner,
        "config": str(base),
        "label_source": tag,
        "seeds": list(seeds),
        "cohort": cohort,
        "n_train": n_train,
        "approx_steps_per_seed": steps,
        "jobs": jobs,
        "average": "mean of the seeds' sigmoid probabilities per study (kaggle_infer predict)",
        "package": "kaggle_infer.py bundle --mode final verifies variant/seeds/cohort/label from "
        "each run's receipt.json + final_ema.pt args",
    }


# ----------------------------------------------------------------------------- commands
def commands(owner: str = "matthewliew", version_tag: str = "v731") -> list[str]:
    k_cand = f"{owner}/radiant-kg-{version_tag}-teamblend-pf"
    k_ctrl = "<TEAM BLEND KERNEL (owner/slug) -- not available locally>"
    return [
        f"# BLOCKER: {TEAM_BLEND_BLOCKER}",
        "# 1. Stage the specialist bundle (no upload):",
        "python experiments/branch_b/v731/kaggle_infer.py bundle --mode final --ckpts "
        "<2-3 round-2 final_ema.pt of the winner> --side-model <side_model.pt> --out <fresh dir>",
        "# 2. Founder-approved uploads (private datasets) and the candidate kernel push:",
        "kaggle datasets create -p <dir>/ds_v731_models -q",
        # assets hold sub-folders: --dir-mode zip as in kaggle_infer.commands (LAUNCH-012)
        "kaggle datasets create -p <dir>/ds_v731_assets --dir-mode zip -q",
        "#    the candidate kernel = team blend notebook source + specialist inference + "
        "round2.replace_pf:",
        "#      team blend writes team.csv; kaggle_infer.py predict writes specialist_pf.csv;",
        "#      python round2.py replace-pf --team team.csv --specialist specialist_pf.csv "
        "--sample <comp>/sample_submission.csv --out . && cp candidate.csv submission.csv",
        "kaggle kernels push -p <dir>/kernel_candidate",
        "# 3. After both private runs validate, the founder submits (ONE public candidate vs the "
        "control):",
        f"kaggle competitions submit {COMPETITION} -k {k_cand} -v <VERSION> -f submission.csv "
        "-m 'V7.3.1 team blend + PF specialist (rank 50/50)'",
        f"# control (same team blend, no specialist) -- already scored publicly as 0.967 if "
        f"unchanged: {k_ctrl}",
        "# 4. Only if the candidate's public score drops: retrain the final seeds on D3 "
        "(round2.py retrain-plan --d3) and submit once more.",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("retrain-plan")
    p.add_argument("--winner", required=True, choices=("E1", "E2", "E3"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--d3", action="store_true", help="round-2 fallback: D3 labels")
    p.add_argument("--final", action="store_true", help="final fit on 4403 (+ the 58, official)")
    r = sub.add_parser("replace-pf")
    r.add_argument("--team", type=Path, required=True)
    r.add_argument("--specialist", type=Path, required=True)
    r.add_argument("--sample", type=Path)
    r.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("average-seeds")
    v.add_argument("csvs", nargs="+", type=Path)
    v.add_argument("--out", type=Path, required=True)
    sub.add_parser("commands")
    a = ap.parse_args(argv)
    if a.cmd == "retrain-plan":
        plan = retrain_plan(
            a.winner, tuple(int(s) for s in a.seeds.split(",")), a.d3, final=a.final
        )
        print(json.dumps(plan, indent=2))
    elif a.cmd == "replace-pf":
        print(json.dumps(run_replace(a.team, a.specialist, a.out, a.sample), indent=2))
    elif a.cmd == "average-seeds":
        avg = average_seeds([c.read_bytes() for c in a.csvs])
        buf = io.StringIO(newline="")
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([ID, PF])
        w.writerows([k, fmt(x)] for k, x in avg.items())
        write_fresh(a.out, buf.getvalue().encode())
        print(f"wrote {a.out}: {len(avg)} studies, mean of {len(a.csvs)} seeds")
    else:
        print("\n".join(commands()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

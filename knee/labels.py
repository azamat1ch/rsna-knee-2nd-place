"""Training labels: report reader and image expert (LLM calls, logged), then deterministic mixes and the folds.

Every model call is appended to a JSONL log, so `export`, `folds` and `mix` rebuild every table from the logs
and frozen inputs with no call. Tables are written once (never overwritten), rows sorted by study ID.
No report text reaches inference: these labels exist only for training.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from knee import LABELS
from knee import panels
from knee.panels import BLIND_FINDINGS, BLIND_ROLE, REGIONS

ID = "StudyInstanceUID"
PROMPTS = Path(__file__).resolve().parents[1] / "configs/labels"
ENDPOINT = os.environ.get("CLIPROXY_BASE_URL", "http://127.0.0.1:18317/v1")  # an OpenAI Responses-API endpoint
KEY_VARIABLES = ("CLIPROXY_API_KEY", "OPENAI_API_KEY")

# Report reader
EFFORT = "high"  # reasoning effort of every call
READER_MODEL, READER_SCHEMA_NAME = "gpt-6.1-sol", "knee_report_labels"
READER_ATTEMPTS, READER_TIMEOUT_S = 4, 600
CATEGORIES = ("yes", "probably_yes", "unclear", "probably_no", "no", "not_mentioned")
# The category picks a band, p_expert (0-100) places the exam inside it. Only order within a finding matters.
BANDS = dict(no=(0.00, 0.05), not_mentioned=(0.05, 0.25), probably_no=(0.25, 0.45),
             unclear=(0.45, 0.65), probably_yes=(0.65, 0.85), yes=(0.85, 1.00))
SYNOVITIS_BANDS = dict(BANDS, probably_no=(0.25, 0.35), not_mentioned=(0.35, 0.45))  # silence above probably_no

# Image expert
EXPERT_MODEL, EXPERT_SCHEMA_NAME = "gpt-6-astra", "knee_expert_read"
EXPERT_ATTEMPTS, EXPERT_TIMEOUT_S, BACKOFF_S, BACKOFF_MAX_S = 6, 1200, 15, 300
TIMEOUT_TRIES = 3  # blind-lateral: the proxy can answer 408; try the whole call again
BLIND_EXTRA = {
    "Lateral Meniscus": dict(images_with_surface_contact=dict(type="integer", minimum=0, maximum=60),
                             morphology=dict(type="string", enum=["normal", "blunted_or_truncated",
                                                                  "displaced_fragment", "root_absent",
                                                                  "post_surgical_remnant", "other"])),
    "Lateral OA": dict(worst_depth=dict(type="string", enum=["normal", "signal_or_fraying", "partial_under_half",
                                                             "over_half", "full_thickness"]),
                       worst_extent_mm=dict(type="integer", minimum=0, maximum=80)),
}
REPORT_HEADING = "REPORT (original text, as written):\n"
MAX_EXPERT_FALLBACK = 100  # exams without an expert answer that may take the reader row instead

# Mixes
PF, LATERAL = "PF OA", "Lateral OA"
PF_READER_SHARE = 0.5  # expert-pf: PF OA = 0.5 expert + 0.5 reader p_expert
LATERAL_BLIND_SHARE = 0.6  # expert-pf-lateral: Lateral OA = 0.4 expert-pf + 0.6 blind-lateral
TEACHER_EXPERT_SHARE = {x: 0.7 if x in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", PF, LATERAL) else 0.4
                        for x in LABELS}
TEACHER_FAMILIES = 5
PF_UNCLEAR_FLOOR = 0.8  # teacher-mix-pf: PF OA = max(p, 0.8) where the reader calls PF OA "unclear"
PF_UNCLEAR_LIFT = 0.6  # expert-pf-floor: PF OA = 0.6 + 0.4 p there (keeps the order inside the group)
MIL_EXPERT_SHARE = 0.7  # expert-reader: 0.7 expert-pf-floor + 0.3 sigmoid(mil-reader out-of-fold logit)
RULES = ["expert-pf", "expert-pf-lateral", "teacher-mix", "teacher-mix-pf", "teacher-mix-pf-58", "expert-pf-floor",
         "expert-reader", "expert-reader-58"]

# Folds
FOLDS, FOLD_SEED, EXPERT_FOLD = 5, 20261005, "expert58"
EXPERTS, LINKED = 58, 4
LANGUAGES = (("de", r"(knie|meniskus|erguss|kreuzband)"), ("tr", r"\b(diz|menisk\w*|eklem)\b"),
             ("pt", r"\b(joelho|derrame articular|ligamento cruzado anterior [ií]ntegro)\b"),
             ("es", r"\b(rodilla|menisco|derrame)\b"), ("fr", r"\b(genou|m[ée]nisque|[ée]panchement)\b"),
             ("it", r"\b(ginocchio|versamento)\b"), ("en", r"\b(knee|meniscus|effusion|ligament)\b"))
PROVENANCE = "expert-dev24-eval34-v1 partition: 58 expert + 4 report-linked companions"


# ---------------------------------------------------------------- files

def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def read_train(path: Path) -> dict[str, dict]:
    """train.csv rows by study ID (report text and the 12 expert label cells)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    out = {r[ID]: r for r in rows}
    if len(out) != len(rows) or "" in out:
        raise ValueError("missing or duplicate study ID in train.csv")
    return out


def read_partition(path: Path) -> tuple[list[str], list[str]]:
    """(58 expert exams, 4 report-linked companions), each sorted: kept out of every training table and fold."""
    rows = json.loads(Path(path).read_text())["rows"]
    expert = sorted(r["study_id"] for r in rows if r["expert_present"] in (True, "True"))
    linked = sorted(r["study_id"] for r in rows if r["expert_present"] not in (True, "True"))
    if len(expert) != EXPERTS or len(linked) != LINKED:
        raise ValueError("partition must hold 58 expert and 4 linked exams")
    return expert, linked


def training_ids(train: dict, partition: Path) -> list[str]:
    expert, linked = read_partition(partition)
    return sorted(set(train) - set(expert) - set(linked))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def read_table(source: Path | bytes, columns=LABELS) -> tuple[list[str], np.ndarray]:
    """ID + columns (a file or its bytes), every cell finite in [0, 1] -> (ids, float64 array)."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    if not rows or rows[0] != [ID, *columns]:
        raise ValueError(f"table columns must be {[ID, *columns]}")
    ids = [r[0] for r in rows[1:]]
    values = np.asarray([[float(c) for c in r[1:]] for r in rows[1:]], dtype=np.float64).reshape(len(ids), -1)
    if len(set(ids)) != len(ids) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("table: duplicate ID or cell outside [0, 1]")
    return ids, values


def write_once(path: Path, data: bytes) -> None:
    with Path(path).open("xb") as stream:  # tables are written once
        stream.write(data)


def table_bytes(ids, values, columns=LABELS, cell=lambda v: format(float(v), ".6g"), lineterminator="\n") -> bytes:
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator=lineterminator)
    writer.writerow((ID, *columns))
    for i in order:
        writer.writerow((ids[i], *(cell(v) for v in values[i])))
    return stream.getvalue().encode()


def write_table(path: Path, ids, values) -> None:
    data = table_bytes(ids, values)
    read_table(data)  # the table contract, checked on the bytes written
    write_once(path, data)


def read_detail(path: Path) -> dict[str, dict]:
    return {r["study_id"]: r for r in read_jsonl(path)}


# ---------------------------------------------------------------- model calls

def api_key() -> str:
    for name in KEY_VARIABLES:
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit("set CLIPROXY_API_KEY or OPENAI_API_KEY")


def post(body: dict, timeout: int) -> dict:
    """One Responses-API call -> parsed JSON answer, served model and usage."""
    request = urllib.request.Request(ENDPOINT + "/responses", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + api_key(),
                                              "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    text = "".join(c.get("text", "") for item in reply.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []))
    return dict(parsed=json.loads(text), model=reply.get("model"), usage=reply.get("usage") or {})


def body(model: str, instructions: str, user_input, name: str, schema: dict) -> dict:
    return dict(model=model, reasoning=dict(effort=EFFORT), instructions=instructions, input=user_input,
                text=dict(format=dict(type="json_schema", name=name, strict=True, schema=schema)))


def call_reader(request: dict) -> dict:
    """Any failure is retried; after the last attempt the error is recorded."""
    for attempt in range(READER_ATTEMPTS):
        try:
            return post(request, READER_TIMEOUT_S)
        except Exception as error:  # network, proxy or unparsable answer
            last = repr(error)
            time.sleep(5 * (attempt + 1))
    return dict(error=last)


def call_expert(request: dict) -> dict:
    """429, 5xx and network or parse failures are retried with exponential backoff and jitter; other 4xx stop.
    Every failed attempt stays in the row."""
    size, attempts = len(json.dumps(request).encode()), []
    for attempt in range(EXPERT_ATTEMPTS):
        started = time.monotonic()
        try:
            got = post(request, EXPERT_TIMEOUT_S)
            usage = got.pop("usage")
            return dict(got, seconds=round(time.monotonic() - started, 1), request_bytes=size, attempts=attempts,
                        usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")},
                        reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
        except urllib.error.HTTPError as error:
            attempts.append(dict(status=error.code, seconds=round(time.monotonic() - started, 1)))
            if error.code != 429 and error.code < 500:
                break
        except Exception as error:
            attempts.append(dict(error=repr(error)[:200], seconds=round(time.monotonic() - started, 1)))
        time.sleep(min(BACKOFF_MAX_S, BACKOFF_S * 2 ** attempt) * (0.5 + random.random()))
    return dict(error=attempts[-1] if attempts else "no attempt", attempts=attempts, request_bytes=size)


def run_pool(jobs: list, work, log: Path, workers: int) -> None:
    """Calls in threads; each row is appended to the log as soon as it finishes."""
    with cf.ThreadPoolExecutor(workers) as pool, Path(log).open("a", encoding="utf-8") as stream:
        for n, future in enumerate(cf.as_completed([pool.submit(work, job) for job in jobs]), 1):
            row = future.result()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{n}/{len(jobs)}] {'ok' if 'parsed' in row else 'error'}", flush=True)


# ---------------------------------------------------------------- report reader

def norm(text: str) -> str:
    """Reports equal after NFKC, whitespace collapse and casefold are read once."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def reader_schema() -> dict:
    finding = dict(type="object", additionalProperties=False, required=["quotes", "source", "category", "why",
                                                                        "p_expert"],
                   properties=dict(quotes=dict(type="array", items=dict(type="string"), maxItems=2),
                                   source=dict(type="string", enum=["stated", "global_normal", "inferred", "none"]),
                                   category=dict(type="string", enum=list(CATEGORIES)), why=dict(type="string"),
                                   p_expert=dict(type="integer", minimum=0, maximum=100)))
    return dict(type="object", additionalProperties=False,
                required=["language", "side_in_report", "prior_surgery", "global_normal", "global_normal_quote",
                          "flags", "findings"],
                properties=dict(
                    language=dict(type="string"),
                    side_in_report=dict(type="string", enum=["left", "right", "both", "none", "conflicting"]),
                    prior_surgery=dict(type="array", items=dict(type="string", enum=[
                        "acl_reconstruction", "meniscectomy", "meniscal_repair", "arthroplasty", "hardware",
                        "other"])),
                    global_normal=dict(type="string", enum=["none", "whole_knee", "rest_normal"]),
                    global_normal_quote=dict(type=["string", "null"]),
                    flags=dict(type="array", items=dict(type="string")),
                    findings=dict(type="object", additionalProperties=False, required=list(LABELS),
                                  properties={label: finding for label in LABELS})))


def representatives(train: dict, ids: list[str]) -> dict[str, str]:
    """norm(report) -> the first study (sorted) with that report."""
    first = {}
    for i in sorted(ids):
        first.setdefault(norm(train[i]["Report"]), i)
    return first


def reader_request(report: str) -> dict:
    prompt = (PROMPTS / "reader.md").read_text(encoding="utf-8")
    return body(READER_MODEL, prompt, "REPORT:\n" + report, READER_SCHEMA_NAME, reader_schema())


def cmd_reader(args) -> None:
    train = read_train(args.train)
    reps = representatives(train, training_ids(train, args.partition))
    done = {r["study_id"] for r in read_jsonl(args.log) if "parsed" in r} if args.log.exists() else set()
    todo = [s for s in reps.values() if s not in done]

    def work(study):
        report = train[study]["Report"]
        return dict(study_id=study, report_sha256=sha(report), **call_reader(reader_request(report)))
    print(f"reader: {len(todo)} reports to read", flush=True)
    run_pool(todo, work, args.log, args.workers)


def band_value(label: str, finding: dict) -> float:
    low, high = (SYNOVITIS_BANDS if label == "Synovitis" else BANDS)[finding["category"]]
    return low + (high - low) * (finding["p_expert"] / 100)


def export_reader(train: dict, ids: list[str], log: Path, out: Path) -> None:
    """reader.csv (CRLF, 4 decimals) and reader-detail.jsonl (category and p_expert per finding)."""
    answers = {r["study_id"]: r["parsed"] for r in read_jsonl(log) if "parsed" in r}  # the last answer wins
    reps = representatives(train, ids)
    picked = {i: answers.get(reps[norm(train[i]["Report"])]) for i in ids}
    if missing := [i for i in ids if picked[i] is None]:
        raise SystemExit(f"{len(missing)} exams have no read report; run `reader` again")
    table, detail = io.StringIO(newline=""), io.StringIO(newline="")
    writer = csv.writer(table)  # default dialect: CRLF line ends
    writer.writerow([ID, *LABELS])
    for i in ids:
        parsed = picked[i]
        found = parsed["findings"]
        writer.writerow([i, *(f"{band_value(x, found[x]):.4f}" for x in LABELS)])
        detail.write(json.dumps(dict(study_id=i, language=parsed["language"],
                                     categories={x: found[x]["category"] for x in LABELS},
                                     p_expert={x: found[x].get("p_expert") for x in LABELS}), ensure_ascii=False) + "\n")
    write_once(out / "reader.csv", table.getvalue().encode())
    write_once(out / "reader-detail.jsonl", detail.getvalue().encode())


# ---------------------------------------------------------------- image expert

def guides() -> dict[str, str]:
    """One '## Finding' block per finding from expert-findings.md."""
    text = (PROMPTS / "expert-findings.md").read_text(encoding="utf-8")
    blocks = {c.split("\n", 1)[0].strip(): "## " + c.strip() for c in re.split(r"^## ", text, flags=re.M)[1:]}
    if set(blocks) != set(LABELS):
        raise ValueError("expert-findings.md must hold one block per finding")
    return blocks


def fill(text: str, values: dict) -> str:
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def expert_instructions(sections: str) -> str:
    g = guides()
    return fill((PROMPTS / "expert.md").read_text(encoding="utf-8"),
                dict(FINDINGS="\n\n".join(g[f] for f in LABELS), SECTIONS=sections))


def blind_instructions(sections: str) -> str:
    return fill((PROMPTS / "blind-lateral.md").read_text(encoding="utf-8"),
                dict(ROLE=BLIND_ROLE, FINDINGS=", ".join(BLIND_FINDINGS), SECTIONS=sections))


def finding_schema(props: dict, findings: list[str], observations: dict) -> dict:
    item = lambda p: dict(type="object", additionalProperties=False, required=list(p), properties=p)  # noqa: E731
    return dict(type="object", additionalProperties=False, required=["observations", "findings", "image_quality"],
                properties=dict(observations=observations,
                                findings=dict(type="object", additionalProperties=False, required=list(findings),
                                              properties={f: item(props[f]) for f in findings}),
                                image_quality=dict(type="string")))


def answer_fields(extra: dict | None = None, informed: bool = False) -> dict:
    props = dict(image_evidence=dict(type="string"), **(extra or {}))
    if informed:
        props.update(report_says=dict(type="string"), agree=dict(type="string", enum=["yes", "no", "silent"]))
    props.update(probability=dict(type="integer", minimum=0, maximum=100),
                 key_images=dict(type="array", items=dict(type="string"), maxItems=4))
    return props


def expert_schema() -> dict:
    obs = dict(type="object", additionalProperties=False, required=REGIONS,
               properties={r: dict(type="string") for r in REGIONS})
    return finding_schema({f: answer_fields(informed=True) for f in LABELS}, list(LABELS), obs)


def blind_schema() -> dict:
    return finding_schema({f: answer_fields(BLIND_EXTRA[f]) for f in BLIND_FINDINGS}, BLIND_FINDINGS,
                          dict(type="string"))


def reader_block(detail: dict) -> str:
    lines = ["SECOND READER (report text only, no images): category and estimated probability (0-100) that experts "
             "mark the finding positive."]
    lines += [f"- {f}: {detail['categories'][f]}, {detail['p_expert'][f]}" for f in LABELS]
    return "\n".join(lines)


def expert_request(pack: dict, report: str | None = None, detail: dict | None = None) -> dict:
    """Request body for one pack (knee.panels). Expert packs get the report and the reader's answers appended;
    blind-lateral packs are sent as they are."""
    if pack["kind"] == "blind-lateral":
        return body(EXPERT_MODEL, blind_instructions(pack["sections"]), [dict(role="user", content=pack["content"])],
                    EXPERT_SCHEMA_NAME, blind_schema())
    content = [*pack["content"], dict(type="input_text", text=REPORT_HEADING + report),
               dict(type="input_text", text=reader_block(detail))]
    return body(EXPERT_MODEL, expert_instructions(pack["sections"]), [dict(role="user", content=content)],
                EXPERT_SCHEMA_NAME, expert_schema())


def checked(result: dict, findings: list[str]) -> dict:
    """An answer counts only with exactly the asked findings and integer probabilities 0-100."""
    if "parsed" in result:
        got = result["parsed"]["findings"]
        error = "findings mismatch" if set(got) != set(findings) else \
            None if all(isinstance(v["probability"], int) and 0 <= v["probability"] <= 100 for v in got.values()) \
            else "probability out of 0-100"
        if error:
            result = {k: v for k, v in result.items() if k != "parsed"} | dict(error=error)
    return result


def cmd_expert(args) -> None:
    kind = "blind-lateral" if args.blind_lateral else "expert"
    findings = BLIND_FINDINGS if args.blind_lateral else list(LABELS)
    train = read_train(args.train)
    allowed = set(training_ids(train, args.partition))
    detail = read_detail(args.reader_detail) if not args.blind_lateral else {}
    done = {r["study_id"] for r in read_jsonl(args.log) if "parsed" in r} if args.log.exists() else set()
    jobs = []
    for part in sorted(Path(args.packs).glob("packs-*.zip")):
        with zipfile.ZipFile(part) as archive:
            studies = sorted({n.split("/")[0] for n in archive.namelist() if n.endswith("/pack.json")})
        jobs += [(part, s) for s in studies if s not in done]
    if {s for _, s in jobs} - allowed:
        raise SystemExit("a pack is not a training exam")

    def work(job):
        part, study = job
        base = dict(study_id=study, kind=kind, part=part.name)
        try:
            with zipfile.ZipFile(part) as archive:
                pack = panels.read_pack(archive, study)
            if pack["kind"] != kind:
                raise ValueError("pack is of another kind")
            request = expert_request(pack, train[study]["Report"], detail.get(study))
            for _ in range(TIMEOUT_TRIES if args.blind_lateral else 1):
                result = checked(call_expert(request), findings)
                if not (isinstance(result.get("error"), dict) and result["error"].get("status") == 408):
                    break
            base.update(slots_present=pack["slots_present"], instructions_sha256=sha(request["instructions"]),
                        pack=pack["info"])
        except Exception as error:  # one broken exam must not stop the pool; a rerun retries it
            result = dict(error=f"{type(error).__name__}: {error}"[:300])
        return dict(base, **result, finished=datetime.now(timezone.utc).isoformat())
    print(f"{kind}: {len(jobs)} exams to read", flush=True)
    run_pool(jobs, work, args.log, args.workers)


def expert_answers(log: Path, ids: list[str]) -> dict[str, dict]:
    """study -> parsed answer; exactly one answer per exam, training exams only."""
    got = {}
    for r in read_jsonl(log):
        if "parsed" in r:
            if r["study_id"] in got:
                raise SystemExit("two answers for one exam")
            got[r["study_id"]] = r["parsed"]["findings"]
    if set(got) - set(ids):
        raise SystemExit("an answered exam is not a training exam")
    return got


def export_expert(ids: list[str], log: Path, out: Path) -> None:
    """expert.csv: probability / 100 (two decimals); an exam without an answer takes its reader row."""
    got = expert_answers(log, ids)
    _, reader = read_table(out / "reader.csv")
    reader = dict(zip(ids, reader))
    if len(set(ids) - set(got)) > MAX_EXPERT_FALLBACK:
        raise SystemExit("too many exams lack an expert answer")
    values = [[float(f"{got[i][x]['probability'] / 100:.2f}") for x in LABELS] if i in got else reader[i] for i in ids]
    write_table(out / "expert.csv", ids, values)


def export_blind(ids: list[str], log: Path, out: Path) -> None:
    """blind-lateral.csv: Lateral OA only (its Lateral Meniscus answer is not used)."""
    got = expert_answers(log, ids)
    present = sorted(got)
    data = table_bytes(present, [[got[i][LATERAL]["probability"]] for i in present], columns=[LATERAL],
                       cell=lambda p: f"{p / 100:.2f}")
    write_once(out / "blind-lateral.csv", data)


def cmd_export(args) -> None:
    train = read_train(args.train)
    ids = training_ids(train, args.partition)
    args.out.mkdir(parents=True, exist_ok=True)
    export_reader(train, ids, args.reader_log, args.out)
    export_expert(ids, args.expert_log, args.out)
    export_blind(ids, args.blind_lateral_log, args.out)


# ---------------------------------------------------------------- folds

def group_key(study: str, report: str) -> str:
    """Exams with the same normalised report share a group (and so a fold)."""
    text = re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", report).lower()).strip()
    raw = ("report\0" + text) if text else ("empty\0" + study)
    return sha(raw)[:20]


def language(report: str) -> str:
    text = report.lower()
    return next((name for name, pattern in LANGUAGES if re.search(pattern, text)), "other")


def assign_folds(reports: dict[str, str], excluded: set, key: dict[str, np.ndarray]) -> dict[str, tuple[str, str]]:
    """Greedy balanced folds over report groups: size, language and per-finding report-key positives and
    negatives, largest and rarest groups first, seeded hash tie-break. -> {study: (fold, group)}."""
    groups: dict[str, list[str]] = {}
    for study, report in reports.items():
        groups.setdefault(group_key(study, report), []).append(study)
    for members in groups.values():
        if excluded.intersection(members) and not excluded.issuperset(members):
            raise ValueError("an excluded exam shares a report with a training exam")
    fitting = {k: sorted(v) for k, v in groups.items() if not excluded.intersection(v)}
    langs = sorted({language(reports[m[0]]) for m in fitting.values()})
    vectors = {}
    for k, members in fitting.items():
        rows = np.array([key[s] for s in members], dtype=float)
        lang = np.zeros(len(langs))
        lang[langs.index(language(reports[members[0]]))] = len(members)
        vectors[k] = np.concatenate([[len(members)], lang, (rows == 1).sum(0), (rows == 0).sum(0)])
    total = sum(vectors.values())
    target = np.maximum(total / FOLDS, 1.0)
    rarity = {k: float((v / np.maximum(total, 1))[1 + len(langs):].sum()) for k, v in vectors.items()}
    order = sorted(fitting, key=lambda k: (-len(fitting[k]), -rarity[k], sha(f"{FOLD_SEED}\0{k}")))
    counts = np.zeros((FOLDS, len(target)))
    result = {}
    for k in order:
        g = vectors[k]
        fold = int(np.argmin((((counts + g) ** 2 - counts ** 2) / target ** 2).sum(1)))  # first minimum
        counts[fold] += g
        result.update({s: (str(fold), k) for s in fitting[k]})
    return result


def fold_bytes(assignments: dict) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow((ID, "fold", "group"))
    for study in sorted(assignments):
        writer.writerow((study, *assignments[study]))
    return stream.getvalue().encode()


def read_folds(path: Path) -> dict[str, str]:
    with Path(path).open(newline="") as stream:
        return {r[ID]: r["fold"] for r in csv.DictReader(stream)}


def cmd_folds(args) -> None:
    train = read_train(args.train)
    expert, linked = read_partition(args.partition)
    key = {s: np.array([{"yes": 1.0, "no": 0.0}.get(r["categories"][x], np.nan) for x in LABELS])
           for s, r in read_detail(args.reader_detail).items()}
    folds = assign_folds({s: r["Report"] or "" for s, r in train.items()}, set(expert) | set(linked), key)
    args.out.mkdir(parents=True, exist_ok=True)
    write_once(args.out / "folds.csv", fold_bytes(folds))
    with58 = folds | {s: (EXPERT_FOLD, sha("expert\0" + s)[:20]) for s in expert}  # the 58: training side only
    write_once(args.out / "folds-58.csv", fold_bytes(with58))
    write_once(args.out / "exclusions.json", (json.dumps(dict(expert_study_ids=expert, linked_study_ids=linked,
                                                              provenance=PROVENANCE), indent=1) + "\n").encode())


# ---------------------------------------------------------------- mixes

def unclear_pf(ids: list[str], detail: dict) -> np.ndarray:
    return np.array([detail[s]["categories"][PF] == "unclear" for s in ids])


def teacher_mean(sources: list[Path], root: Path, folds: dict[str, str]) -> dict[str, np.ndarray]:
    """Equal mean of the teacher families' out-of-fold predictions. Each source JSON lists one file per fold
    (path relative to root, sha256); each file holds only the exams of the one fold its model held out."""
    families = []
    for source in sources:
        one, seen = {}, set()
        for item in json.loads(Path(source).read_text())["sources"]:
            path = root / item["path"]
            if sha(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"{Path(source).name}: a teacher file changed")
            ids, values = read_table(path)
            held = {folds[s] for s in ids}
            if len(held) != 1 or held & seen:
                raise ValueError("a teacher file is not the out-of-fold predictions of one fold")
            seen |= held
            one.update(zip(ids, values))
        if seen != set(folds.values()):
            raise ValueError("a teacher family lacks a fold")
        families.append(one)
    shared = set.intersection(*(set(f) for f in families))
    return {s: np.mean([f[s] for f in families], axis=0) for s in shared}


def expert_rows(train: dict, expert: list[str]) -> np.ndarray:
    gold = np.array([[float(train[s][x]) for x in LABELS] for s in expert])
    if not np.isin(gold, (0.0, 1.0)).all():
        raise ValueError("expert labels must be 0 or 1")
    return gold


def mix(rule: str, args) -> None:
    d = args.dir
    detail = read_detail(d / "reader-detail.jsonl")
    if rule == "expert-pf":
        ids, base = read_table(d / "expert.csv")
        j, w = LABELS.index(PF), PF_READER_SHARE
        base[:, j] = (1 - w) * base[:, j] + w * np.array([detail[s]["p_expert"][PF] / 100 for s in ids])
    elif rule == "expert-pf-lateral":
        ids, base = read_table(d / "expert-pf.csv")
        sl_ids, sl = read_table(d / "blind-lateral.csv", [LATERAL])
        sl, j, w = dict(zip(sl_ids, sl[:, 0])), LABELS.index(LATERAL), LATERAL_BLIND_SHARE
        if set(ids) - set(sl):
            raise SystemExit("an exam lacks a blind-lateral answer")
        base[:, j] = [(1 - w) * base[i, j] + w * sl[s] for i, s in enumerate(ids)]
    elif rule == "teacher-mix":
        ids, base = read_table(d / "expert-pf-lateral.csv")
        folds = read_folds(d / "folds.csv")
        if set(ids) != set(folds) or len(args.teachers) != TEACHER_FAMILIES:
            raise SystemExit("teacher-mix needs the fold exams and five teacher families")
        teacher = teacher_mean(args.teachers, args.teacher_root, folds)
        if set(ids) - set(teacher):
            raise SystemExit("an exam lacks a teacher prediction")
        w = np.array([TEACHER_EXPERT_SHARE[x] for x in LABELS])
        base = np.array([w * base[i] + (1 - w) * teacher[s] for i, s in enumerate(ids)])
    elif rule == "teacher-mix-pf":
        ids, base = read_table(d / "teacher-mix.csv")
        hit, j = unclear_pf(ids, detail), LABELS.index(PF)
        base[hit, j] = np.maximum(base[hit, j], PF_UNCLEAR_FLOOR)
    elif rule == "expert-pf-floor":
        ids, base = read_table(d / "expert-pf-lateral.csv")
        hit, j = unclear_pf(ids, detail), LABELS.index(PF)
        base[hit, j] = PF_UNCLEAR_LIFT + (1 - PF_UNCLEAR_LIFT) * base[hit, j]
    elif rule == "expert-reader":
        ids, base = read_table(d / "expert-pf-floor.csv")
        with Path(args.mil_reader_oof).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        oof = {r[ID]: r for r in rows}
        if list(rows[0]) != [ID, "source", *LABELS] or len(oof) != len(rows) or set(oof) != set(ids):
            raise SystemExit("mil-reader OOF must hold one logit row per exam")
        logit = np.array([[float(oof[s][x]) for x in LABELS] for s in ids])
        base = MIL_EXPERT_SHARE * base + (1 - MIL_EXPERT_SHARE) * (1 / (1 + np.exp(-logit)))
    elif rule in ("teacher-mix-pf-58", "expert-reader-58"):
        ids, base = read_table(d / f"{rule.removesuffix('-58')}.csv")
        expert = json.loads((d / "exclusions.json").read_text())["expert_study_ids"]
        if set(expert) & set(ids):
            raise SystemExit("an expert exam is already in the table")
        ids, base = ids + expert, np.concatenate([base, expert_rows(read_train(args.train), expert)])
    else:
        raise SystemExit(f"unknown rule {rule}")
    write_table(d / f"{rule}.csv", ids, base)


def cmd_mix(args) -> None:
    for rule in RULES if args.rules == ["all"] else args.rules:
        mix(rule, args)
        print(rule, sha((args.dir / f"{rule}.csv").read_bytes())[:16], flush=True)


# ---------------------------------------------------------------- CLI

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--train", type=Path, required=True, help="competition train.csv")
    common.add_argument("--partition", type=Path, required=True, help="expert partition JSON (58 + 4 exams)")

    p = sub.add_parser("reader", parents=[common], help="read each distinct report once (LLM calls)")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_reader)

    p = sub.add_parser("expert", parents=[common], help="image expert on panel packs (LLM calls)")
    p.add_argument("--packs", type=Path, required=True, help="folder of packs-NNN.zip (knee.panels)")
    p.add_argument("--reader-detail", type=Path, help="reader-detail.jsonl (expert packs)")
    p.add_argument("--blind-lateral", action="store_true", help="images only, Lateral Meniscus and Lateral OA")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--workers", type=int, default=20)
    p.set_defaults(func=cmd_expert)

    p = sub.add_parser("export", parents=[common], help="logs -> reader.csv, reader-detail.jsonl, expert.csv, "
                                                        "blind-lateral.csv")
    p.add_argument("--reader-log", type=Path, required=True)
    p.add_argument("--expert-log", type=Path, required=True)
    p.add_argument("--blind-lateral-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("folds", parents=[common], help="folds.csv, folds-58.csv, exclusions.json")
    p.add_argument("--reader-detail", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_folds)

    p = sub.add_parser("mix", help="deterministic label tables, in order: " + ", ".join(RULES))
    p.add_argument("rules", nargs="+", choices=[*RULES, "all"])
    p.add_argument("--dir", type=Path, required=True, help="folder holding the exported tables and folds")
    p.add_argument("--train", type=Path, help="train.csv (the 58 expert rows)")
    p.add_argument("--teachers", type=Path, nargs="*", default=[], help="five teacher source JSONs")
    p.add_argument("--teacher-root", type=Path, help="folder the teacher paths are relative to")
    p.add_argument("--mil-reader-oof", type=Path, help="mil-reader out-of-fold logits CSV")
    p.set_defaults(func=cmd_mix)

    args = parser.parse_args(argv)
    if args.command == "expert" and not args.blind_lateral and not args.reader_detail:
        parser.error("expert packs need --reader-detail")
    args.func(args)


if __name__ == "__main__":
    main()

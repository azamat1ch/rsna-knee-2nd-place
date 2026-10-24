"""The OAI kneecap-cartilage grade table and its image series.

  python oai_join.py --moaks "MR Image Assessment_ASCII/Semi-Quant Scoring_ASCII" --nda NDA_INDEX --out DIR

Writes <out>/grades.csv, one row per OAI knee-visit (participant, visit, side) with the 8 MOAKS
cartilage grades PM, PL, FMA and FLA, size and full-thickness part, 0-3 (NaN when unknown), and
the primary series key of each image type; and <out>/series.csv (series key -> DICOM tarball).

- MOAKS: kMRI_SQ_MOAKS_BICL{00,01,03,05,06,08,10}.txt. Codes outside the codebook are masked;
  V01 stores cartilage codes rounded, so V01 cartilage is masked; a within-grade code (+-0.5)
  takes the grade of the same reader project's previous visit.
- One reading per knee-visit: most known fields, then reader project 22>65>30>47>46>63>61, then
  the lowest READPRJ, then the earliest file row.
- Series: NDA package submission_60350, v<visit>/series_manifest.csv. Side and type come from the
  series description <TYPE>_<RIGHT|LEFT>; a repeated acquisition keeps the highest file number.
- A row needs a known field and a series of the same participant, visit and side. 10% of
  participants (salted hash) are held out; training uses the rest.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# (grade-table name, MOAKS column), in the model's order
FIELDS = (
    ("PM_size", "MCMPM_size"),
    ("PM_ft", "MCMPM_ft"),
    ("PL_size", "MCMPL_size"),
    ("PL_ft", "MCMPL_ft"),
    ("FMA_size", "MCMFMA_size"),
    ("FMA_ft", "MCMFMA_ft"),
    ("FLA_size", "MCMFLA_size"),
    ("FLA_ft", "MCMFLA_ft"),
)
FIELD_NAMES = tuple(f for f, _ in FIELDS)
MOAKS_COLS = tuple(c for _, c in FIELDS)
ITEMS = ("MCMPM", "MCMPL", "MCMFMA", "MCMFLA")
N_THRESHOLDS = 3  # P(g>=1), P(g>=2), P(g>=3)

# OAI series type -> W slot of the competition model (slot = plane + 3 * (1 - fat_sup))
SLOT_OF_TYPE = {"SAG_IW_TSE": 0, "COR_MPR": 1, "AX_MPR": 2, "COR_IW_TSE": 4}
SERIES_COLS = {t: f"series_{t.lower()}" for t in SLOT_OF_TYPE}

VISITS = ["00", "01", "03", "05", "06", "08", "10"]
VISIT_MONTH = {"00": 0, "01": 12, "03": 24, "05": 36, "06": 48, "08": 72, "10": 96}
CART_CODES = {0.0, 1.0, 1.1, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 3.3}
WITHIN_GRADE = {0.5, -0.5}
ROUNDED_VISITS = {"01"}
PRJ_PRIORITY = ["22", "65", "30", "47", "46", "63", "61"]
SIDE_CODES = {"1: Right": "RIGHT", "2: Left": "LEFT", "1": "RIGHT", "2": "LEFT"}
HOLDOUT_SALT = "radiant-kg-v731-oai-holdout"
HOLDOUT_SHARE = 0.10


def study_key(pid, visit, side) -> str:
    return f"oai_{pid}_{visit}_{side}"


def holdout_split(pid) -> str:
    h = int.from_bytes(hashlib.sha256(f"{HOLDOUT_SALT}:{pid}".encode()).digest()[:8], "big")
    return "heldout" if h % 10000 < round(HOLDOUT_SHARE * 10000) else "train"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def parse_code(x: str) -> float:
    """'2.2: 10-75% area, ...' / '2' / '.: Missing' / '' -> float or NaN."""
    x = x.strip()
    if not x or x.startswith("."):
        return np.nan
    try:
        return float(x.split(":", 1)[0].strip())
    except ValueError:
        return np.nan


def read_readings(sq_dir: Path) -> pd.DataFrame:
    """Every MOAKS reading, masked and resolved, with the 8 fields."""
    frames = []
    for v in VISITS:
        df = pd.read_csv(sq_dir / f"kMRI_SQ_MOAKS_BICL{v}.txt", sep="|", dtype=str, encoding="latin1",
                         keep_default_na=False)
        df = df.rename(columns={"ID": "id", "SIDE": "side", "READPRJ": "readprj", "VERSION": "version"})
        side = df["side"].str.strip()
        if set(side) - set(SIDE_CODES):
            raise ValueError(f"V{v}: unknown side codes {set(side) - set(SIDE_CODES)}")
        out = pd.DataFrame({"ID": df["id"].str.strip(), "visit": f"V{v}", "side": side.map(SIDE_CODES),
                            "readprj": df["readprj"].str.strip(), "file_row": np.arange(len(df))})
        for it in ITEMS:
            out[it] = df[f"V{v}{it}"].map(parse_code)
        frames.append(out)
    raw = pd.concat(frames, ignore_index=True)
    df = raw[["ID", "visit", "side", "readprj", "file_row"]].copy()
    prj = raw.readprj.str[:2]
    vnum = raw.visit.str[1:]
    order = raw.assign(prj=prj, m=vnum.map(VISIT_MONTH)).sort_values(["ID", "side", "prj", "m"]).index
    for it in ITEMS:
        x = raw[it].astype(float).copy()
        wg = x.isin(WITHIN_GRADE)
        x[x.notna() & ~x.isin(CART_CODES) & ~wg] = np.nan
        x[vnum.isin(ROUNDED_VISITS)] = np.nan
        wg &= x.notna()
        if wg.any():
            # a within-grade code repeats the grade of the previous visit's reading(s); sibling
            # readings that disagree leave it unknown
            prev, current_visit, pending, resolved = {}, {}, {}, x.copy()
            for idx in order:
                key = (raw.at[idx, "ID"], raw.at[idx, "side"], prj.at[idx])
                visit = raw.at[idx, "visit"]
                if key in current_visit and current_visit[key] != visit:
                    values = {v for v in pending[key] if pd.notna(v)}
                    prev[key] = next(iter(values)) if len(values) == 1 else np.nan
                    pending[key] = []
                current_visit[key] = visit
                pending.setdefault(key, [])
                if x.at[idx] in WITHIN_GRADE:
                    resolved.at[idx] = prev.get(key, np.nan)
                pending[key].append(resolved.at[idx])
            x = resolved
        size = np.floor(x + 1e-6)
        df[f"{it}_size"] = size.astype("float32")
        df[f"{it}_ft"] = np.round((x - size) * 10).astype("float32")
    return df


def choose_readings(d: pd.DataFrame) -> pd.DataFrame:
    """One reading per knee-visit (most known fields, project priority, READPRJ, file row)."""
    d = d.copy()
    d["n_known"] = d[list(MOAKS_COLS)].notna().sum(axis=1)
    d["prio"] = d.readprj.str[:2].map({p: i for i, p in enumerate(PRJ_PRIORITY)}).fillna(99).astype(int)
    keys = ["ID", "visit", "side"]
    d = d.sort_values([*keys, "n_known", "prio", "readprj", "file_row"],
                      ascending=[True, True, True, False, True, True, True])
    d["n_readings"] = d.groupby(keys)["readprj"].transform("size")
    return d.groupby(keys).head(1)[[*keys, "readprj", "n_readings", *MOAKS_COLS]]


def read_series(nda_dir: Path) -> pd.DataFrame:
    """The primary series of every (participant, visit, side, type) in the NDA series manifests."""
    rows = []
    for v in VISITS:
        m = pd.read_csv(nda_dir / f"v{v}" / "series_manifest.csv", dtype=str, low_memory=False)
        desc = m.image_description.str.extract(r"^(.*)_(RIGHT|LEFT)$")
        j = pd.DataFrame({"ID": m.src_subject_id.str.strip(), "visit": f"V{v}", "side": desc[1],
                          "series": desc[0], "key": "oai_" + m.image03_id.astype("int64").astype(str),
                          "tarball": m.image_file})
        j["file_no"] = j.tarball.str.extract(r"/(\d+)\.tar\.gz$")[0].astype("int64")
        j = j.sort_values(["ID", "side", "series", "file_no"])
        rows.append(j[~j.duplicated(["ID", "side", "series"], keep="last")])
    s = pd.concat(rows, ignore_index=True)
    s = s[s.series.isin(list(SLOT_OF_TYPE))]
    if not s.key.is_unique:
        raise ValueError("an image series is listed twice")
    return s


def build(readings: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    chosen = choose_readings(readings)
    wide = series.pivot(index=["ID", "visit", "side"], columns="series", values="key")
    wide = wide.reindex(columns=list(SLOT_OF_TYPE)).rename(columns=SERIES_COLS).reset_index()
    t = chosen.merge(wide, on=["ID", "visit", "side"], how="inner", validate="one_to_one")
    t = t[t[list(MOAKS_COLS)].notna().any(axis=1)].sort_values(["ID", "visit", "side"])
    out = pd.DataFrame({"study": [study_key(*k) for k in zip(t.ID, t.visit, t.side)],
                        "participant": t.ID.values, "visit": t.visit.values, "side": t.side.values,
                        "split": [holdout_split(p) for p in t.ID], "readprj": t.readprj.values,
                        "n_readings": t.n_readings.values})
    for name, col in FIELDS:
        out[name] = t[col].astype("float64").values
    for name in FIELD_NAMES:
        out[name + "_known"] = out[name].notna().astype(int)
    for c in SERIES_COLS.values():
        out[c] = t[c].values
    validate(out)
    return out


def cumulative_targets(grades) -> np.ndarray:
    """[N, 8] grades (0..3 or NaN) -> [N, 8, 3] float32: 1[g >= t], t = 1..3; NaN where unknown."""
    g = np.asarray(grades, np.float64)
    known = np.isfinite(g)
    out = (g[:, :, None] >= np.arange(1, N_THRESHOLDS + 1)).astype(np.float32)
    out[~known] = np.nan
    return out


def validate(t: pd.DataFrame) -> None:
    if t.empty or not t.study.is_unique:
        raise ValueError("empty grade table or duplicate knee-visit")
    if not set(t.side) <= {"LEFT", "RIGHT"} or not set(t.split) <= {"train", "heldout"}:
        raise ValueError("bad side or split values")
    g = t[list(FIELD_NAMES)].to_numpy(np.float64)
    known = np.isfinite(g)
    if (~np.isin(g[known], [0, 1, 2, 3])).any() or not known.any(axis=1).all():
        raise ValueError("grades must be NaN or 0..3, with at least one known per row")


def load_table(path) -> pd.DataFrame:
    """grades.csv -> validated DataFrame (identity columns as str)."""
    t = pd.read_csv(path, dtype={"participant": str, "visit": str, "side": str, "study": str, "readprj": str,
                                 **{c: str for c in SERIES_COLS.values()}})
    for c in SERIES_COLS.values():
        t[c] = t[c].where(t[c].notna(), None)
    validate(t)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moaks", type=Path, required=True, help="folder with kMRI_SQ_MOAKS_BICL*.txt")
    ap.add_argument("--nda", type=Path, required=True, help="folder with v<visit>/series_manifest.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    series = read_series(a.nda)
    out = build(read_readings(a.moaks), series)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "grades.csv", index=False)
    used = {k for k in out[list(SERIES_COLS.values())].to_numpy().ravel() if isinstance(k, str)}
    series[series.key.isin(used)][["key", "tarball"]].to_csv(a.out / "series.csv", index=False)
    print(f"{len(out)} knee-visits, {out.participant.nunique()} participants, "
          f"{int((out.split == 'heldout').sum())} held out, {len(used)} series")


if __name__ == "__main__":
    main()

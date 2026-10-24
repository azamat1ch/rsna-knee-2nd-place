"""Run external/v731/prepare_native.py, unchanged, with your own input paths.

Its input paths are folders on Matthew's Mac and its --out flag only moves the output, so this sets the
inputs, then hands every other argument to it:

  python native.py --work work --headers work/headers --zip rsna-knee-abnormality-detection.zip \
      build --all --workers 22 --gate-tolerance 1 --out work/native

--work holds labels_D.csv, data/train_series.csv, cache384_train/, cache384_train_meta.csv, views_train.csv
and laterality_v2.csv. --headers holds series_headers.parquet and slice_geometry.parquet
(external/v8prep/headers.py). The output defaults to <work>/native.
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXTERNAL, require_decoders  # noqa: E402

sys.path.insert(0, str(EXTERNAL / "v731"))
import prepare_native as pn  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--work", required=True)
    ap.add_argument("--headers", required=True)
    ap.add_argument("--zip", required=True, help="competition archive")
    a, rest = ap.parse_known_args()
    require_decoders()
    multiprocessing.set_start_method("fork", force=True)  # its workers must inherit the paths set here
    pn.WORK, pn.HEADERS, pn.ZIP = Path(a.work), Path(a.headers), Path(a.zip)
    pn.OUT, pn.CACHE = pn.WORK / "native", pn.WORK / "cache384_train"
    fh, _cache, tmp_root, tolerance = pn.gate_series.__defaults__  # its cache default was bound at import
    pn.gate_series.__defaults__ = (fh, pn.CACHE, tmp_root, tolerance)
    sys.argv = [pn.__file__, *rest]
    pn.main()


if __name__ == "__main__":
    main()

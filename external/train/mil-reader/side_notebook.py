"""Assemble the single-file Kaggle script: embedded helper sources + fastMRI link + side_main.py."""
import json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
reader = here.parents[1] / "reader"
emb = {name: (reader / name).read_text() for name in ("canonical.py", "preprocess.py")}
out, url = pathlib.Path(sys.argv[1]), sys.argv[2]  # usage: side_notebook.py OUT.py FASTMRI_VAL_URL
out.write_text("EMBEDDED = " + json.dumps(emb) + "\nFASTMRI_VAL_URL = " + json.dumps(url) + "\n" + (here / "side_classifier.py").read_text())
print("wrote", out, out.stat().st_size)

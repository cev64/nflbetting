"""Design search on DEV seasons only (2010-2017, walk-forward from 2006).
Usage: python dev.py su|ats [--holdout]   (--holdout runs the FROZEN list in frozen.py instead; do once)
"""
from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

from data import load
from harness import DEV, walk, su_metrics, ats_metrics

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

SU_GRID = [
    ("market", {}),
    ("market", {"hl": 4}),
    ("logit", {"C": 0.01}), ("logit", {"C": 0.1}), ("logit", {"C": 1.0}),
    ("logit", {"C": 0.1, "hl": 4}),
    ("logit", {"C": 0.1, "by": "family"}), ("logit", {"C": 1.0, "by": "family"}),
    ("logit", {"C": 0.1, "by": "family", "ml": 1}),
    ("logit", {"C": 0.03, "by": "family", "ctx": 1}), ("logit", {"C": 0.1, "by": "family", "ctx": 1}),
    ("logit", {"C": 0.3, "by": "family", "ctx": 1}),
    ("logit", {"C": 0.1, "by": "family", "ctx": 1, "hl": 4}),
    ("logit", {"C": 0.1, "by": "family", "ctx": 1, "dis": 1}),
    ("logit", {"C": 0.03, "ctx": 1}), ("logit", {"C": 0.1, "ctx": 1}),
    ("lgbm", {"n": 50}), ("lgbm", {"n": 100}), ("lgbm", {"n": 200}), ("lgbm", {"n": 400}),
    ("lgbm", {"n": 100, "min_leaf": 300}), ("lgbm", {"n": 100, "hl": 4}),
    ("bma", {"tau": 1.0, "hl": 4}), ("bma", {"tau": 0.1, "hl": 4}), ("bma", {"tau": 0.02, "hl": 4}),
    ("bma", {"tau": 0.005, "hl": 4}),
    ("bma", {"tau": 0.02, "hl": 2}), ("bma", {"tau": 0.02, "hl": 8}),
    ("bma", {"tau": 0.02, "hl": 4, "cells": ["pickem"]}), ("bma", {"tau": 0.1, "hl": 4, "cells": ["pickem"]}),
    ("bma", {"tau": 0.02, "hl": 4, "cells": ["pickem", "early", "playoff"]}),
    ("router", {"k": 2, "conf": 0.8}), ("router", {"k": 2, "conf": 0.6}), ("router", {"k": 1, "conf": 0.6}),
    ("router", {"k": 3, "conf": 0.5}),
]

ATS_GRID = [
    ("logit", {"signals": None, "inj": False, "C": 1.0}),          # structure only (road/home dog prior)
    ("logit", {"signals": None, "C": 1.0}),                         # structure + injury gap
    ("logit", {"signals": None, "C": 1.0, "intercept": True}),
    ("logit", {"signals": "model", "C": 0.01}), ("logit", {"signals": "model", "C": 0.1}),
    ("logit", {"signals": "model", "C": 1.0}),
    ("logit", {"signals": "family", "C": 0.1}), ("logit", {"signals": "family", "C": 1.0}),
    ("logit", {"signals": "mean", "C": 1.0}),
    ("logit", {"signals": "family", "C": 0.1, "ctx": 1}),
    ("logit", {"signals": "family", "C": 0.1, "hl": 4}),
    ("logit", {"signals": "model", "C": 0.1, "structure": False, "inj": False}),
    ("lgbm", {"signals": "family", "n": 100}), ("lgbm", {"signals": "family", "n": 300}),
    ("vote", {}),
]


def main():
    kind = sys.argv[1]
    holdout = "--holdout" in sys.argv
    g, models = load()
    if holdout:
        from frozen import SU_FROZEN, ATS_FROZEN
        from harness import HOLD
        lo, hi = HOLD
        grid = SU_FROZEN if kind == "su" else ATS_FROZEN
    else:
        lo, hi = DEV
        grid = SU_GRID if kind == "su" else ATS_GRID
    print("models:", models)
    rows = []
    for design, cfg in grid:
        t = time.time()
        if kind == "su":
            p = walk(g, models, "su", design, cfg, lo, hi)
            r = su_metrics(g, p, lo, hi)
        else:
            p = walk(g, models, "ats", design, cfg, lo - 1, hi)
            r = ats_metrics(g, p, lo, hi)
        r = {"design": design, "cfg": str(cfg), **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
             "sec": round(time.time() - t, 1)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    print(df.to_string())
    tag = "hold" if holdout else "dev"
    df.to_csv(f"out_{kind}_{tag}.csv", index=False)


if __name__ == "__main__":
    main()

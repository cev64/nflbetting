"""Development comparison on seasons <= 2017 ONLY (walk-forward test seasons 2010-2017).

    python research/roster/dev.py          # needs work/games.parquet from run.py --features-only

Prints SU / logloss / ATS for each pre-declared candidate (feature set x model), plus the spread-only reference.
The configuration frozen in run.py was chosen from this output (see NOTES.md). Nothing here touches 2018+.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl  # noqa: E402

import model as M  # noqa: E402
from data import WORK  # noqa: E402

DEV_TEST = list(range(2010, 2018))


def main() -> None:
    g = pl.read_parquet(WORK / "games.parquet").filter(pl.col("season") <= 2017)
    rows = []
    base = M.spread_only(g, DEV_TEST)
    rows.append(("spread_only", "-", M.score(base, g, 2010, 2017), M.score(base, g, 2012, 2017)))
    for fs, feats in M.FEATURE_SETS.items():
        for kind in ("mkt", "nomkt"):
            if kind == "nomkt" and fs != "full":
                continue
            pred, _ = M.walk_forward(g, feats, kind, DEV_TEST)
            rows.append((f"{kind}", fs, M.score(pred, g, 2010, 2017), M.score(pred, g, 2012, 2017)))
        if fs not in ("agg", "inj"):
            pred = M.gbm_walk_forward(g, feats, DEV_TEST)
            rows.append(("gbm", fs, M.score(pred, g, 2010, 2017), M.score(pred, g, 2012, 2017)))
    for name, fs, a, b in rows:
        print(f"{name:12s} {fs:6s} | 2010-17 SU {a['su']:.4f} mkt {a['mkt']:.4f} ll {a['ll']:.4f} ATS {a['ats']:.4f} "
              f"| 2012-17 SU {b['su']:.4f} ll {b['ll']:.4f} ATS {b['ats']:.4f}")


if __name__ == "__main__":
    main()

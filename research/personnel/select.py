"""Feature-set selection on test seasons 2006-2011 ONLY (walk-forward, nested C).  2012+ numbers are printed for
transparency but were not used to choose (the choice rule is: lowest 2006-2011 logloss)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import polars as pl

import model as M
from common import WORK

if __name__ == "__main__":
    g = pl.read_parquet(WORK / "games.parquet")
    for mid, cands in M.CANDIDATES.items():
        for name, feats in cands.items():
            p, _ = M.walk_forward(g, feats, seasons=list(range(2006, 2026)))
            print(mid, name, "2006-11", M.quick_score(p, g, 2006, 2011), "2012-25", M.quick_score(p, g), flush=True)
    p, _ = M.walk_forward(g, ["spread_line"], seasons=list(range(2006, 2026)))
    print("market_only_logit", "2006-11", M.quick_score(p, g, 2006, 2011), "2012-25", M.quick_score(p, g))

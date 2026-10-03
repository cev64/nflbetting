"""Leakage stress tests.

1. As-of test: rebuild QB and injury features using only pbp / snap-count data dated strictly before the first
   game of a week, and check that every game of that week gets identical features to the full-data build.
2. Shuffle test: permute the personnel features across games within each test season; skill must vanish.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import polars as pl

import model as M
from common import WORK, schedules
from injuries import injury_loads
from qb import qb_game_log, team_qb_features


def asof_test(weeks=((2014, 8), (2019, 10), (2023, 15), (2025, 3))):
    s = schedules()
    full_q = team_qb_features(qb_game_log())
    full_i, _ = injury_loads()
    for season, week in weeks:
        wk = s.filter((pl.col("season") == season) & (pl.col("week") == week))
        cut = wk["date"].min()
        q = team_qb_features(qb_game_log(asof=cut)).filter(pl.col("game_id").is_in(wk["game_id"].implode()))
        f = full_q.filter(pl.col("game_id").is_in(wk["game_id"].implode()))
        j = q.join(f, on=["game_id", "team"], suffix="_full")
        cols = ["qb_epa", "qb_epa_long", "qb_cpoe", "qb_epa_delta", "qb_log_exp"]
        dq = max(float((j[c] - j[f"{c}_full"]).abs().max()) for c in cols)
        i, _ = injury_loads(asof=cut)
        i = i.filter((pl.col("season") == season) & (pl.col("week") == week))
        fi = full_i.filter((pl.col("season") == season) & (pl.col("week") == week))
        ji = i.join(fi, on=["season", "week", "team"], suffix="_full")
        icols = [c for c in i.columns if c.startswith("inj_")]
        di = max(float((ji[c] - ji[f"{c}_full"]).abs().max()) for c in icols)
        print(f"as-of {season} wk{week} (cut {cut}): {j.height} QB rows max|diff|={dq:.2e}; "
              f"{ji.height} injury rows max|diff|={di:.2e}")


def shuffle_test(seed=0):
    g = pl.read_parquet(WORK / "games.parquet")
    rng = np.random.default_rng(seed)
    for model_id, feats in M.FEATURES.items():
        pf = [f for f in feats if f != "spread_line"]
        # permute the personnel feature rows jointly within each season
        parts = []
        for (S,), sub in g.group_by("season", maintain_order=True):
            perm = rng.permutation(sub.height)
            parts.append(sub.with_columns([pl.Series(c, sub[c].to_numpy()[perm]) for c in pf]))
        gs = pl.concat(parts)
        p, _ = M.walk_forward(gs, feats, seasons=list(range(2012, 2026)))
        print(f"shuffle {model_id}: 2012-2025 {M.quick_score(p, g)}")
        if "spread_line" in feats:
            continue
        drop = [f for f in feats if not f.startswith("d_qb")]
        p, _ = M.walk_forward(g, drop, seasons=list(range(2012, 2026)))
        print(f"drop QB features {model_id}: 2012-2025 {M.quick_score(p, g)}")


if __name__ == "__main__":
    asof_test()
    shuffle_test()

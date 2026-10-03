"""Leakage checks for the roster family.

1. Truncation test: rebuild every input with all outcome-bearing data (pbp, box scores, snap counts) cut at the
   first kickoff of a target week, keeping only pre-game tables (depth charts, injury reports) for that week,
   and check that the target week's team features are identical to the full build.
2. Depth-chart timing test (2013-2024): players who left game g early injured (>=60% snaps in g-1, <30% in g,
   absent in g+1) vs players with a similar low-snap game who were back for g+1. If the weekly chart for g were
   a post-game snapshot, injured players would be missing from chart g as often as from chart g+1.
3. Shuffle test: permute the lineup features within each season; the market model should fall back to the
   spread-only result and the no-market model toward the home-win rate.

    python research/roster/stress.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
import run as R  # noqa: E402
from games import TEAM_FEATS  # noqa: E402

TARGETS = [(2011, 9), (2015, 12), (2019, 10), (2023, 15), (2025, 3)]


def _truncated_features(season: int, week: int, cut, cal, ptab):
    inp = R.build_inputs(asof=cut)
    # pre-game tables: keep only charts / reports of games up to the target week
    upto = inp["tg"].filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= week)))
    inp["dc"] = inp["dc"].join(upto.select("game_id"), on="game_id", how="semi")
    inp["inj"] = inp["inj"].join(upto.select("game_id"), on="game_id", how="semi")
    feat, *_ = R.build_features(inp, cal=cal, ptab=ptab)
    return feat


def truncation_test() -> None:
    """Cut all outcome data at a kickoff time T and compare features of every game played on T's calendar date.
    Ratings use games with date < kickoff date, so these must match the full build exactly. (Games on later
    days of the same week may legitimately use the earlier same-week games, e.g. Thursday's.)"""
    import json
    full = pl.read_parquet(D.WORK / "team_features.parquet")
    ptab = pl.read_parquet(D.WORK / "p_play_table.parquet")
    cal = {tuple(k.split("|")): tuple(v) for k, v in json.load(open(D.WORK / "calibration.json")).items()}
    s = D.schedules()
    for season, week in TARGETS:
        wk = s.filter((pl.col("season") == season) & (pl.col("week") == week))
        for which, sub in (("first slot", wk), ("Sunday", wk.filter(pl.col("weekday") == "Sunday"))):
            cut = sub["kick"].min()
            ids = wk.filter(pl.col("date") == sub.filter(pl.col("kick") == cut)["date"][0])["game_id"]
            feat = _truncated_features(season, week, cut, cal, ptab)
            a = feat.filter(pl.col("game_id").is_in(ids.implode())).sort("game_id", "team")
            b = full.filter(pl.col("game_id").is_in(ids.implode())).sort("game_id", "team")
            cols = [c for c in TEAM_FEATS if c in a.columns]
            x, y = a.select(cols).to_numpy().astype(float), b.select(cols).to_numpy().astype(float)
            diff = np.nanmax(np.abs(x - y))
            nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
            print(f"truncation {season} wk{week} ({which}, cut {cut}): {a.height} team-games, "
                  f"max |diff| over {len(cols)} features = {diff:.2e}, null mismatches {nan_mismatch}")


def depth_timing_test() -> None:
    s = D.schedules()
    tg = D.team_games(s)
    dc = D.depth_charts(tg)
    sn = D.snaps(s).with_columns(pl.max_horizontal("off_pct", "def_pct").alias("pct")).filter(
        pl.col("grp").is_in(["QB", "OL", "RB", "WR", "TE", "DL", "LB", "DB"]))
    g = tg.filter(pl.col("season").is_between(2013, 2024)).select("game_id", "team", "fr", "kick").sort("fr", "kick")
    g = g.with_columns(pl.col("game_id").shift(1).over("fr").alias("gprev"),
                       pl.col("game_id").shift(-1).over("fr").alias("gnext"))
    p = sn.join(g, on=["game_id", "team"]).select("game_id", "gprev", "gnext", "team", "gsis_id", "pct")
    p = p.join(sn.select(pl.col("game_id").alias("gprev"), "gsis_id", pl.col("pct").alias("pct_prev")),
               on=["gprev", "gsis_id"], how="left")
    p = p.join(sn.select(pl.col("game_id").alias("gnext"), "gsis_id", pl.col("pct").alias("pct_next")),
               on=["gnext", "gsis_id"], how="left")
    p = p.join(dc.select("game_id", "team", "gsis_id", pl.col("depth").alias("d_cur")),
               on=["game_id", "team", "gsis_id"], how="left")
    p = p.join(dc.select(pl.col("game_id").alias("gnext"), "team", "gsis_id", pl.col("depth").alias("d_next")),
               on=["gnext", "team", "gsis_id"], how="left").filter(pl.col("pct_prev") >= 0.6)
    groups = {
        "left game g early, missed g+1": p.filter((pl.col("pct") < 0.3) & pl.col("pct_next").is_null()),
        "low snaps in g, back in g+1  ": p.filter((pl.col("pct") < 0.3) & (pl.col("pct_next") >= 0.6)),
        "regular in g and g+1         ": p.filter((pl.col("pct") >= 0.6) & (pl.col("pct_next") >= 0.6)),
    }
    for k, x in groups.items():
        r = x.select(pl.len().alias("n"), pl.col("d_cur").is_null().mean().alias("off_chart_g"),
                     pl.col("d_next").is_null().mean().alias("off_chart_g1")).row(0, named=True)
        print(f"depth timing: {k} n={r['n']:6d}  missing from chart g {r['off_chart_g']:.3f}   "
              f"missing from chart g+1 {r['off_chart_g1']:.3f}")


def shuffle_test() -> None:
    g = pl.read_parquet(D.WORK / "games.parquet").filter(pl.col("season") <= 2017)
    rng = np.random.default_rng(11)
    dev = list(range(2010, 2018))
    for mid, (kind, fs) in R.MODELS.items():
        feats = M.FEATURE_SETS[fs]
        gs = g.with_columns(pl.int_range(pl.len()).alias("_i"))
        parts = []
        for S in sorted(gs["season"].unique().to_list()):
            x = gs.filter(pl.col("season") == S)
            perm = rng.permutation(x.height)
            parts.append(x.with_columns([pl.Series(f, x[f].to_numpy()[perm]) for f in feats]))
        sh = pl.concat(parts)
        a, _ = M.walk_forward(g, feats, kind, dev)
        b, _ = M.walk_forward(sh, feats, kind, dev)
        print(f"shuffle {mid}: real {M.score(a, g, 2010, 2017)}\n         shuffled {M.score(b, g, 2010, 2017)}")


if __name__ == "__main__":
    depth_timing_test()
    truncation_test()
    shuffle_test()

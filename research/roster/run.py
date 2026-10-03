"""Roster family (wave 2): bottom-up, player-level lineup strength for the players expected to play.

    python research/roster/run.py               # features + both models + factors (~1 min)
    python research/roster/run.py --report      # also print dev (<=2017) and holdout (2018-2025) scores

Outputs
    research/preds/wave2_roster.csv          market + lineup features (SU logistic, ATS ridge on result - spread)
    research/preds/wave2_roster_nomkt.csv    lineup/QB features only, no market input
    research/preds/wave2_roster_factors.parquet, wave2_roster_nomkt_factors.parquet
    research/roster/work/*.parquet           features, candidates (lineup detail), per-fold coefficients

Frozen configuration (chosen on dev test seasons 2010-2017 only, see dev.py / NOTES.md):
    wave2_roster        kind=mkt,   feature set "inj"
    wave2_roster_nomkt  kind=nomkt, feature set "full"
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl  # noqa: E402

import data as D  # noqa: E402
import lineup as L  # noqa: E402
import model as M  # noqa: E402
import players as P  # noqa: E402
from games import game_matrix  # noqa: E402

MODELS = {"wave2_roster": ("mkt", "inj"), "wave2_roster_nomkt": ("nomkt", "full")}
FIRST_TEST, LAST_TEST = 2006, 2026


def build_inputs(asof=None):
    """All per-player / per-team inputs. `asof` (UTC datetime) truncates every outcome-bearing table to games
    kicked off before it (used by stress.py); pre-game tables (depth charts, injury reports) are kept up to the
    week containing `asof`."""
    s = D.schedules()
    tg = D.team_games(s)
    sn = D.snaps(s)
    dc = D.depth_charts(tg)
    inj = D.injuries(tg)
    p = P.pbp_frame()
    ql, pp, kl = P.qb_log(p), P.team_passpro(p), P.kick_log(p)
    del p
    if asof is not None:
        before = tg.filter(pl.col("kick") < asof).select("game_id").unique()
        sn = sn.join(before, on="game_id", how="semi")
        ql, pp, kl = (x.join(before, on="game_id", how="semi") for x in (ql, pp, kl))
    plog = P.player_log(tg, sn, dc, pp)
    if asof is not None:
        plog = plog.join(before, on="game_id", how="semi")
    return dict(s=s, tg=tg, sn=sn, dc=dc, inj=inj, ql=ql, pp=pp, kl=kl, plog=plog)


def build_features(inp: dict, cal=None, ptab=None):
    cal = cal or P.calibrate(inp["plog"])
    feat, cand, ptab = L.build(inp["tg"], inp["dc"], inp["inj"], inp["plog"], inp["ql"], inp["kl"], cal,
                               inp["sn"]["game_id"].unique(), ptab=ptab)
    return feat, cand, ptab, cal


FACTORS = [
    # key, label, better, fmt
    ("qb_epa", "Starting QB EPA/play over replacement (recent form)", "high", "+0.000"),
    ("delta_qb", "Starting QB vs last 6 starters (EPA/play)", "high", "+0.000"),
    ("U_rec", "Receivers in the lineup: value (rec. EPA/game over replacement)", "high", "0.0"),
    ("U_rush", "Running backs in the lineup: value (rush EPA/game over replacement)", "high", "0.0"),
    ("U_olexp", "Offensive line in the lineup: experience (snap-weighted)", "high", "0.0"),
    ("U_prush", "Pass rushers in the lineup: value (sack/hit EPA/game over replacement)", "high", "0.0"),
    ("U_cover", "Coverage players in the lineup: value (INT/PD EPA/game over replacement)", "high", "0.0"),
    ("inj_off", "Offensive value lost to injuries/absences (vs full health)", "high", "+0.00"),
    ("inj_def", "Defensive value lost to injuries/absences (vs full health)", "high", "+0.00"),
    ("delta_off", "Offensive lineup vs last 6 games' lineups", "high", "+0.00"),
    ("delta_def", "Defensive lineup vs last 6 games' lineups", "high", "+0.00"),
    ("starters_out", "Depth-chart starters likely out", "low", "0"),
    ("k_val", "Kicker FG value (pts/game vs league)", "high", "+0.00"),
]


def factors_table(feat: pl.DataFrame, sched: pl.DataFrame, game_ids: pl.Series) -> pl.DataFrame:
    f = feat.with_columns(
        (pl.col("inj_rec") + pl.col("inj_rush") + pl.col("inj_ol")).alias("inj_off"),
        (pl.col("inj_prush") + pl.col("inj_cover") + pl.col("inj_rund")).alias("inj_def"),
        (pl.col("delta_rec") + pl.col("delta_rush") + pl.col("delta_ol")).alias("delta_off"),
        (pl.col("delta_prush") + pl.col("delta_cover") + pl.col("delta_rund")).alias("delta_def"),
    )
    keys = [k for k, *_ in FACTORS]
    f = f.select("game_id", "team", *[pl.col(k).cast(pl.Float64) for k in keys])
    g = sched.filter(pl.col("game_id").is_in(game_ids.implode())).select("game_id", "home_team", "away_team")
    g = g.join(f.rename({k: f"h_{k}" for k in keys} | {"team": "home_team"}), on=["game_id", "home_team"], how="left")
    g = g.join(f.rename({k: f"a_{k}" for k in keys} | {"team": "away_team"}), on=["game_id", "away_team"], how="left")
    parts = []
    for i, (k, label, better, fmt) in enumerate(FACTORS, start=1):
        parts.append(g.select("game_id", pl.lit(i).cast(pl.Int64).alias("rank"), pl.lit(k).alias("feature"),
                              pl.lit(label).alias("label"), pl.col(f"h_{k}").alias("home"),
                              pl.col(f"a_{k}").alias("away"), pl.lit(better).alias("better"), pl.lit(fmt).alias("fmt")))
    return pl.concat(parts).sort("game_id", "rank")


def main(report: bool = False) -> None:
    t0 = time.time()
    inp = build_inputs()
    feat, cand, ptab, cal = build_features(inp)
    g = game_matrix(inp["s"], feat)
    feat.write_parquet(D.WORK / "team_features.parquet")
    g.write_parquet(D.WORK / "games.parquet")
    cand.select("game_id", "team", "gsis_id", "grp", "depth", "st", "report_status", "p_play", "s_exp", "w_exp",
                *[f"v_{m}" for m in P.METRICS]).write_parquet(D.WORK / "lineup_detail.parquet")
    ptab.write_parquet(D.WORK / "p_play_table.parquet")
    with open(D.WORK / "calibration.json", "w") as fh:
        json.dump({f"{m}|{gp}": v for (m, gp), v in cal.items()}, fh, indent=1)
    print(f"features built in {time.time() - t0:.0f}s")
    tests = list(range(FIRST_TEST, LAST_TEST + 1))
    preds = {}
    for mid, (kind, fs) in MODELS.items():
        pred, recs = M.walk_forward(g, M.FEATURE_SETS[fs], kind, tests)
        pred = pred.sort("season", "week", "game_id").with_columns(pl.col("p_home", "margin", "p_home_cover").round(6))
        pred.write_csv(D.PREDS / f"{mid}.csv")
        with open(D.WORK / f"{mid}_coefs.json", "w") as fh:
            json.dump(recs, fh, indent=1)
        factors_table(feat, inp["s"], pred["game_id"]).write_parquet(D.PREDS / f"{mid}_factors.parquet")
        preds[mid] = pred
        print(mid, pred.height, "rows", f"({time.time() - t0:.0f}s)")
    if report:
        base = M.spread_only(g, tests)
        for name, pr in [("spread_only", base)] + list(preds.items()):
            for lo, hi in [(2010, 2017), (2012, 2017), (2018, 2025), (2012, 2025)]:
                print(name, M.score(pr, g, lo, hi))


if __name__ == "__main__":
    main(report="--report" in sys.argv)

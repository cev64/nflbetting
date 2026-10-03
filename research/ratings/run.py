"""Rebuild every ratings-family prediction file end to end from research/cache.

    python research/ratings/run.py            # writes research/preds/{elo,ratings_ridge,kalman,ratings_combo}[_mkt].csv
                                              # plus elo_factors.parquet and kalman_factors.parquet
Runtime: ~30 s on one CPU. Deterministic (no randomness).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FIRST_OUT, base_frame, quick, walk_forward, write_preds  # noqa: E402
from data import PREDS, load_games  # noqa: E402
from elo import run_elo  # noqa: E402
from kalman import run_kalman  # noqa: E402
from qb import compute_qb  # noqa: E402
from ridge import run_ridge  # noqa: E402

SEASON_HALF_LIFE = 6.0  # recency weighting of the walk-forward probability/blend fits

MODELS = {
    "elo": ["elo"],
    "ratings_ridge": ["r_pts", "r_epa", "qbd", "ctx"],
    "kalman": ["kal"],
    "ratings_combo": ["elo", "kal", "r_pts", "r_epa"],
}


def build_frame() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    qb = compute_qb()
    e, ef = run_elo(qb=qb)
    k, kf = run_kalman(qb=qb)
    r = run_ridge()
    df = (
        base_frame()
        .join(e.rename({"rmargin": "elo"}), on="game_id")
        .join(k.select("game_id", pl.col("rmargin").alias("kal")), on="game_id")
        .join(r.select("game_id", "r_pts", "r_epa"), on="game_id")
        .join(ef.select("game_id", "rest_pts", "travel_pts", "home_qb_adj", "away_qb_adj"), on="game_id")
        .with_columns(
            (pl.col("home_qb_adj") - pl.col("away_qb_adj")).alias("qbd"),
            (pl.col("rest_pts") + pl.col("travel_pts")).alias("ctx"),
            pl.col("r_pts").fill_null(0.0), pl.col("r_epa").fill_null(0.0),
        )
    )
    return df, ef, kf, r


def factors(model_id: str, df: pl.DataFrame, ef: pl.DataFrame, kf: pl.DataFrame, r: pl.DataFrame) -> Path:
    g = load_games().select("game_id", "season", "home_rest", "away_rest", "home_travel", "away_travel")
    f = (
        df.select("game_id", "season", "elo", "kal").filter(pl.col("season") >= FIRST_OUT)
        .join(ef, on="game_id").join(kf.select("game_id", "home_str", "away_str"), on="game_id")
        .join(r.select("game_id", "home_rpts", "away_rpts", "home_repa", "away_repa"), on="game_id")
        .join(g.drop("season"), on="game_id")
    )
    specs = [  # (feature, label, home expr, away expr)
        ("elo", "Elo rating", "home_elo", "away_elo"),
        ("qb_value", "Starting QB rating (shrunk EPA/dropback x 35, pts)", "home_qb_val", "away_qb_val"),
        ("qb_adj", "QB vs usual starter (pts)", "home_qb_adj", "away_qb_adj"),
        ("kalman_strength", "Kalman strength (pts vs avg team)", "home_str", "away_str"),
        ("massey_pts", "Opp-adjusted margin rating (pts)", "home_rpts", "away_rpts"),
        ("massey_epa", "Opp-adjusted EPA/play rating (pts)", "home_repa", "away_repa"),
        ("rest_days", "Rest days", "home_rest", "away_rest"),
        ("travel_miles", "Travel distance (miles)", "home_travel", "away_travel"),
    ]
    parts = [f.select("game_id", pl.lit(k).alias("feature"), pl.lit(lab).alias("label"),
                      pl.col(h).cast(pl.Float64).alias("home"), pl.col(a).cast(pl.Float64).alias("away"))
             for k, lab, h, a in specs]
    parts.append(f.select("game_id", pl.lit("hfa").alias("feature"), pl.lit("Home-field advantage (pts)").alias("label"),
                          pl.col("hfa").cast(pl.Float64).alias("home"), pl.lit(0.0).alias("away")))
    parts.append(f.select("game_id", pl.lit("elo_margin").alias("feature"),
                          pl.lit("Elo expected margin incl. HFA/rest/travel/QB (pts)").alias("label"),
                          pl.col("elo").alias("home"), (-pl.col("elo")).alias("away")))
    out = pl.concat(parts).sort("game_id", "feature")
    path = PREDS / f"{model_id}_factors.parquet"
    out.write_parquet(path)
    return path


def main() -> None:
    df, ef, kf, r = build_frame()
    for mid, feats in MODELS.items():
        for suffix, fs in (("", feats), ("_mkt", ["spread_line"] + feats)):
            pred = walk_forward(df, fs, season_half_life=SEASON_HALF_LIFE)
            write_preds(pred, mid + suffix)
            print(quick(pred, mid + suffix))
    for mid in ("elo", "kalman"):
        print("factors ->", factors(mid, df, ef, kf, r))


if __name__ == "__main__":
    main()

"""Game-level matrix: home/away lineup features, home-minus-away differences, market and outcomes."""
from __future__ import annotations

import polars as pl

UNITS = ["rec", "rush", "prush", "cover", "rund", "ol", "olexp"]
TEAM_FEATS = ([f"U_{u}" for u in UNITS] + [f"delta_{u}" for u in UNITS] + [f"inj_{u}" for u in UNITS]
              + ["qb_epa", "qb_epa_long", "qb_cpoe", "qb_sack_rate", "delta_qb", "k_val", "starters_out",
                 "regulars_out"])


def game_matrix(sched: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
    f = feat.select("game_id", "team", *[pl.col(c).cast(pl.Float64) for c in TEAM_FEATS])
    h = f.rename({c: f"home_{c}" for c in TEAM_FEATS} | {"team": "home_team"})
    a = f.rename({c: f"away_{c}" for c in TEAM_FEATS} | {"team": "away_team"})
    g = (sched.select("game_id", "season", "week", "game_type", "date", "home_team", "away_team", "result",
                      "spread_line", "total_line", "location", "home_rest", "away_rest", "home_qb_name", "away_qb_name")
         .join(h, on=["game_id", "home_team"], how="left").join(a, on=["game_id", "away_team"], how="left"))
    g = g.with_columns(
        [(pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"d_{c}") for c in TEAM_FEATS]
        + [(pl.col("location") == "Neutral").cast(pl.Float64).alias("neutral"),
           (pl.col("home_rest") - pl.col("away_rest")).clip(-7, 7).cast(pl.Float64).alias("rest_diff"),
           (pl.col("result") > 0).cast(pl.Float64).alias("home_win"),
           (pl.col("result") - pl.col("spread_line")).alias("ats_resid")])
    # aggregate lineup deltas (unweighted sums of the EPA-unit pieces; OL experience kept separate)
    off = ["rec", "rush", "ol"]
    de = ["prush", "cover", "rund"]
    g = g.with_columns(
        sum(pl.col(f"d_delta_{u}") for u in off).alias("d_delta_off"),
        sum(pl.col(f"d_delta_{u}") for u in de).alias("d_delta_def"),
        sum(pl.col(f"d_inj_{u}") for u in off).alias("d_inj_off"),
        sum(pl.col(f"d_inj_{u}") for u in de).alias("d_inj_def"),
        sum(pl.col(f"d_U_{u}") for u in off).alias("d_U_off"),
        sum(pl.col(f"d_U_{u}") for u in de).alias("d_U_def"),
    )
    return g

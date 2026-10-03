"""Assemble the game-level personnel feature table (one row per game, home/away + diffs)."""
from __future__ import annotations

import polars as pl

from common import GROUPS, WORK, schedules
from injuries import injury_loads
from qb import qb_game_log, team_qb_features

QB_COLS = ["qb_epa", "qb_epa_long", "qb_cpoe", "qb_log_exp", "qb_change", "qb_change_inseason", "qb_epa_delta",
           "qb_rookie", "qb_backup", "qb_start_share8"]
INJ_COLS = [f"inj_{k}_{g}" for g in GROUPS for k in ("recent", "usual")] + ["inj_starters_out", "inj_q_load",
                                                                             "inj_total"]


def build(first_season: int = 2002) -> tuple[pl.DataFrame, pl.DataFrame]:
    s = schedules().filter(pl.col("season") >= first_season)
    qbf = team_qb_features(qb_game_log())
    loads, detail = injury_loads()

    # latest available report for each team (used for upcoming games whose week has no report yet)
    played_weeks = loads.select("season", "week", "team")

    def side(prefix: str, team_col: str) -> list[pl.Expr]:
        return [pl.col(c).alias(f"{prefix}_{c}") for c in QB_COLS + INJ_COLS]

    g = s.select("game_id", "season", "week", "game_type", "date", "gametime", "home_team", "away_team",
                 "result", "spread_line", "total_line", "home_rest", "away_rest", "div_game", "location",
                 "home_qb_name", "away_qb_name")
    for prefix, tcol in (("home", "home_team"), ("away", "away_team")):
        q = qbf.rename({"team": tcol}).select("game_id", tcol, *[pl.col(c).alias(f"{prefix}_{c}") for c in QB_COLS])
        g = g.join(q, on=["game_id", tcol], how="left")
        # injury report for this game's week; if none filed yet (upcoming week), use the team's most recent
        # report from an earlier week of the same season (flag inj_stale)
        L = loads.rename({"team": tcol})
        gi = g.select("game_id", "season", "week", tcol).join(L, on=["season", "week", tcol], how="left")
        missing = gi.filter(pl.col("inj_total").is_null() & (pl.col("season") >= 2009))
        latest = (missing.select("game_id", "season", "week", tcol)
                  .join(L.rename({"week": "rep_week"}), on=["season", tcol], how="inner")
                  .filter(pl.col("rep_week") < pl.col("week"))
                  .sort("rep_week").group_by("game_id").last())
        gi = gi.join(latest.select("game_id", "rep_week", *INJ_COLS).rename({c: f"{c}_stale" for c in INJ_COLS}),
                     on="game_id", how="left")
        gi = gi.with_columns(
            [pl.coalesce(pl.col(c), pl.col(f"{c}_stale")).alias(f"{prefix}_{c}") for c in INJ_COLS]
            + [(pl.col("inj_total").is_null() & pl.col("rep_week").is_not_null()).alias(f"{prefix}_inj_stale")])
        g = g.join(gi.select("game_id", *[f"{prefix}_{c}" for c in INJ_COLS], f"{prefix}_inj_stale"),
                   on="game_id", how="left")
    g = g.with_columns(
        (pl.col("season") >= 2009).cast(pl.Int8).alias("has_inj"),
        (pl.col("location") == "Neutral").cast(pl.Int8).alias("neutral"),
        (pl.col("home_rest") - pl.col("away_rest")).clip(-7, 7).alias("rest_diff"),
        (pl.col("game_type") != "REG").cast(pl.Int8).alias("playoff"),
    )
    # games before injury data or teams without a report: no listed players -> 0 load
    g = g.with_columns([pl.col(f"{p}_{c}").fill_null(0.0) for p in ("home", "away") for c in INJ_COLS])
    for c in QB_COLS + INJ_COLS:
        g = g.with_columns((pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"d_{c}"))
    WORK.mkdir(exist_ok=True)
    g.write_parquet(WORK / "games.parquet")
    detail.write_parquet(WORK / "injury_detail.parquet")
    return g, detail


if __name__ == "__main__":
    g, _ = build()
    print(g.shape)
    print(g.filter(pl.col("season") == 2026, pl.col("week").is_in([4, 5])).select(
        "game_id", "spread_line", "home_qb_name", "d_qb_epa", "home_inj_total", "away_inj_total", "home_inj_stale"))

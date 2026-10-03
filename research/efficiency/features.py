"""Point-in-time team efficiency features from nflverse play-by-play.

Pipeline
  1. team_game_stats(): one row per (game_id, team) with that game's raw offensive / defensive / special-teams
     stats (garbage-time down-weighted). These are *game outcomes* and must never be used for that game.
  2. asof_ratings(): for every (season, week) "as-of" point and every franchise, an exponentially decayed,
     prior-season-regressed, shrunk average of the franchise's games strictly before that week; then a second
     pass that opponent-adjusts each past game using the opponent's as-of rating.
  3. qb_ratings(): decayed dropback EPA/play of the *scheduled* starting QB, from his earlier games only.
  4. build_team_game_features(): joins (2)+(3) onto the schedule -> one row per team per scheduled game.

Point-in-time rule: a game in (season S, week w) only sees games with season < S or (season == S and week < w).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

# schedules use historical codes, pbp uses current franchise codes
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# decay / regression hyper-parameters (chosen on 2003-2011 only, see NOTES.md)
HALF_LIFE = 12.0     # games
PRIOR_SEASON = 0.4   # extra multiplier per season back
SHRINK = 3.0         # pseudo-games of league-average prior
WINDOW_SEASONS = 2   # seasons back included (in addition to current)
GARBAGE_W = 0.25     # weight of plays with wp outside [0.1, 0.9]

# (metric, label, adjust_for_opponent, higher_is_better_for_offense)
METRICS = [
    ("epa", "EPA/play", True),
    ("pass_epa", "Dropback EPA/play", True),
    ("rush_epa", "Rush EPA/play", True),
    ("early_epa", "Early-down EPA/play", True),
    ("sr", "Success rate", True),
    ("explosive", "Explosive play rate", False),
    ("cpoe", "CPOE", False),
    ("sack_rate", "Sack rate", False),
    ("pressure_rate", "Sack+QB-hit rate", False),
    ("int_rate", "INT per dropback", False),
    ("to_rate", "Turnovers per play", False),
    ("to_luck_adj", "Luck-adj turnovers per game (INT + 0.5*fumbles)", False),
    ("third_conv", "3rd-down conversion", False),
    ("rz_td", "Red-zone TD rate (drives)", False),
    ("ppd", "Points per drive", True),
    ("pts", "Points per game", True),
    ("proe", "Pass rate over expected", False),
    ("plays", "Offensive plays per game", False),
]
UNPAIRED = [
    ("st_epa", "Special-teams EPA per game (net)"),
    ("pen_yds", "Penalty yards per game (committed)"),
    ("margin", "Point margin per game"),
]


def _wmean(x: pl.Expr, w: pl.Expr, mask: pl.Expr | None = None) -> pl.Expr:
    if mask is not None:
        w = pl.when(mask).then(w).otherwise(0.0)
    num = (pl.when(x.is_not_null()).then(w).otherwise(0.0) * x.fill_null(0.0)).sum()
    den = pl.when(x.is_not_null()).then(w).otherwise(0.0).sum()
    return pl.when(den > 0).then(num / den).otherwise(None)


def load_pbp() -> pl.DataFrame:
    cols = ["game_id", "season", "week", "posteam", "defteam", "play_type", "pass", "rush", "special",
            "qb_dropback", "two_point_attempt", "epa", "success", "wp", "down", "yards_gained", "cpoe",
            "sack", "qb_hit", "interception", "fumble", "fumble_lost", "penalty", "penalty_team", "penalty_yards",
            "third_down_converted", "third_down_failed", "fixed_drive", "fixed_drive_result", "yardline_100",
            "xpass", "passer_id"]
    return pl.scan_parquet(CACHE / "pbp.parquet").select(cols).collect()


def team_game_stats(pbp: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Raw per-game stats for each (game_id, team): off_* (team on offense), def_* (allowed)."""
    plays = pbp.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null()
                       & (pl.col("two_point_attempt").fill_null(0) == 0) & pl.col("posteam").is_not_null())
    gw = pl.when(pl.col("wp").is_between(0.1, 0.9) | pl.col("wp").is_null()).then(1.0).otherwise(GARBAGE_W)
    plays = plays.with_columns(
        gw=gw,
        is_db=(pl.col("pass") == 1),
        is_rush=(pl.col("rush") == 1) & (pl.col("pass") == 0),
        explosive=((pl.col("pass") == 1) & (pl.col("yards_gained") >= 20)
                   | (pl.col("rush") == 1) & (pl.col("pass") == 0) & (pl.col("yards_gained") >= 10)).cast(pl.Float64),
        pressure=pl.when(pl.col("season") >= 2006).then(
            ((pl.col("sack") == 1) | (pl.col("qb_hit") == 1)).cast(pl.Float64)).otherwise(None),
        neutral=pl.col("wp").is_between(0.2, 0.8) & (pl.col("down") <= 3),
    )
    w = pl.col("gw")
    agg = plays.group_by("game_id", "posteam", "defteam").agg(
        epa=_wmean(pl.col("epa"), w),
        pass_epa=_wmean(pl.col("epa"), w, pl.col("is_db")),
        rush_epa=_wmean(pl.col("epa"), w, pl.col("is_rush")),
        early_epa=_wmean(pl.col("epa"), w, pl.col("down").is_in([1, 2])),
        sr=_wmean(pl.col("success").cast(pl.Float64), w),
        explosive=_wmean(pl.col("explosive"), w),
        cpoe=_wmean(pl.col("cpoe"), w),
        sack_rate=_wmean(pl.col("sack").cast(pl.Float64), w, pl.col("is_db")),
        pressure_rate=_wmean(pl.col("pressure"), w, pl.col("is_db")),
        int_rate=_wmean(pl.col("interception").cast(pl.Float64), w, pl.col("is_db")),
        to_rate=_wmean((pl.col("interception") + pl.col("fumble_lost")).cast(pl.Float64), w),
        to_luck_adj=(pl.col("interception") + 0.5 * pl.col("fumble")).sum().cast(pl.Float64),
        third_conv=pl.col("third_down_converted").sum()
        / (pl.col("third_down_converted").sum() + pl.col("third_down_failed").sum()).clip(1, None),
        proe=pl.when(pl.col("neutral") & pl.col("xpass").is_not_null())
        .then(pl.col("pass") - pl.col("xpass")).otherwise(None).mean(),
        plays=pl.len().cast(pl.Float64),
    )
    # drives: points per drive & red-zone TD rate
    dr = (pbp.filter(pl.col("posteam").is_not_null() & pl.col("fixed_drive").is_not_null()
                     & pl.col("fixed_drive_result").is_not_null())
          .group_by("game_id", "posteam", "fixed_drive")
          .agg(res=pl.col("fixed_drive_result").first(), minyl=pl.col("yardline_100").min()))
    dr = dr.with_columns(pts=pl.col("res").replace_strict({"Touchdown": 6.95, "Field goal": 3.0}, default=0.0,
                                                          return_dtype=pl.Float64),
                         rz=pl.col("minyl") <= 20)
    dagg = dr.group_by("game_id", "posteam").agg(
        ppd=pl.col("pts").mean(),
        rz_td=pl.when(pl.col("rz").sum() > 0).then(
            ((pl.col("res") == "Touchdown") & pl.col("rz")).sum() / pl.col("rz").sum()).otherwise(None))
    # special teams: net EPA on kicking plays (posteam = kicking team on punts/FG, receiving team on kickoffs)
    st = pbp.filter(pl.col("play_type").is_in(["kickoff", "punt", "field_goal", "extra_point"])
                    & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
    st_pos = st.group_by("game_id", "posteam").agg(a=pl.col("epa").sum()).rename({"posteam": "team"})
    st_def = st.group_by("game_id", "defteam").agg(b=pl.col("epa").sum()).rename({"defteam": "team"})
    st_net = st_pos.join(st_def, on=["game_id", "team"], how="full", coalesce=True).select(
        "game_id", "team", st_epa=pl.col("a").fill_null(0) - pl.col("b").fill_null(0))
    pen = (pbp.filter((pl.col("penalty") == 1) & pl.col("penalty_team").is_not_null())
           .group_by("game_id", "penalty_team").agg(pen_yds=pl.col("penalty_yards").fill_null(0).sum().cast(pl.Float64))
           .rename({"penalty_team": "team"}))

    off = agg.join(dagg, on=["game_id", "posteam"], how="left")
    # points from schedule (authoritative)
    sc = sched.filter(pl.col("result").is_not_null()).select(
        "game_id", "season", "week",
        home=pl.col("home_team").replace(FRANCHISE), away=pl.col("away_team").replace(FRANCHISE),
        hs="home_score", as_="away_score")
    pts = pl.concat([sc.select("game_id", team="home", pts=pl.col("hs").cast(pl.Float64)),
                     sc.select("game_id", team="away", pts=pl.col("as_").cast(pl.Float64))])
    off = off.rename({"posteam": "team", "defteam": "opp"}).join(pts, on=["game_id", "team"], how="left")
    mcols = [m for m, _, _ in METRICS]
    o = off.select("game_id", "team", "opp", *[pl.col(m).alias(f"off_{m}") for m in mcols])
    d = off.select("game_id", team="opp", *[pl.col(m).alias(f"def_{m}") for m in mcols])
    tg = (o.join(d, on=["game_id", "team"], how="inner")
          .join(st_net, on=["game_id", "team"], how="left")
          .join(pen, on=["game_id", "team"], how="left")
          .with_columns(pl.col("pen_yds").fill_null(0.0), pl.col("st_epa").fill_null(0.0),
                        margin=pl.col("off_pts") - pl.col("def_pts")))
    gm = pl.concat([sc.select("game_id", "season", "week", team="home"), sc.select("game_id", "season", "week", team="away")])
    tg = tg.join(gm, on=["game_id", "team"], how="inner")
    return tg.sort("season", "week", "team")


def _value_cols() -> list[str]:
    return [f"{s}_{m}" for m, _, _ in METRICS for s in ("off", "def")] + [m for m, _ in UNPAIRED]


def _asof_points(sched: pl.DataFrame) -> pl.DataFrame:
    sw = sched.select("season", "week").unique()
    teams = pl.DataFrame({"team": sorted(set(FRANCHISE.get(t, t) for t in sched["home_team"].unique()))})
    return sw.join(teams, how="cross")


def _decayed(joined: pl.DataFrame, cols: list[str], prior: pl.DataFrame) -> pl.DataFrame:
    """Weighted mean of cols per (team, season, week) as-of point with shrinkage toward prior."""
    aggs = []
    for c in cols:
        wc = pl.when(pl.col(c).is_not_null()).then(pl.col("w")).otherwise(0.0)
        aggs += [(wc * pl.col(c).fill_null(0.0)).sum().alias(f"{c}__n"), wc.sum().alias(f"{c}__d")]
    r = joined.group_by("team", "season", "week").agg(*aggs, n_games=pl.col("w").count())
    r = r.join(prior, on="season", how="left")
    out = [((pl.col(f"{c}__n") + SHRINK * pl.col(f"{c}__prior").fill_null(0.0))
            / (pl.col(f"{c}__d") + SHRINK)).alias(c) for c in cols]
    return r.select("team", "season", "week", "n_games", *out)


def asof_ratings(tg: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    cols = _value_cols()
    tg = tg.sort("season", "week").with_columns(seq=pl.int_range(pl.len()).over("team"))
    A = _asof_points(sched)
    # league prior for season S = mean of season S-1 games (margin prior 0)
    prior = tg.group_by("season").agg(*[pl.col(c).mean().alias(f"{c}__prior") for c in cols]).with_columns(
        season=pl.col("season") + 1, margin__prior=pl.lit(0.0))
    g = tg.rename({"season": "g_season", "week": "g_week"})
    j = A.join(g, on="team", how="inner").filter(
        ((pl.col("g_season") < pl.col("season")) | ((pl.col("g_season") == pl.col("season")) & (pl.col("g_week") < pl.col("week"))))
        & (pl.col("g_season") >= pl.col("season") - WINDOW_SEASONS))
    j = j.with_columns(ago=(pl.col("seq").max().over("team", "season", "week") - pl.col("seq")).cast(pl.Float64))
    j = j.with_columns(w=(0.5 ** (pl.col("ago") / HALF_LIFE)) * (PRIOR_SEASON ** (pl.col("season") - pl.col("g_season"))))
    raw = _decayed(j, cols, prior)

    # opponent adjustment: game value minus (opponent's as-of rating on the other side - league mean at that point)
    adj_m = [m for m, _, a in METRICS if a]
    lg = raw.group_by("season", "week").agg(*[pl.col(f"{s}_{m}").mean().alias(f"lg_{s}_{m}") for m in adj_m for s in ("off", "def")])
    opp = raw.select("season", "week", opp="team", *[pl.col(f"{s}_{m}").alias(f"o_{s}_{m}") for m in adj_m for s in ("off", "def")],
                     o_margin=pl.col("margin"))
    j = j.join(opp, on=["season", "week", "opp"], how="left").join(lg, on=["season", "week"], how="left")
    j = j.with_columns(
        *[(pl.col(f"off_{m}") - (pl.col(f"o_def_{m}") - pl.col(f"lg_def_{m}"))).alias(f"off_{m}") for m in adj_m],
        *[(pl.col(f"def_{m}") - (pl.col(f"o_off_{m}") - pl.col(f"lg_off_{m}"))).alias(f"def_{m}") for m in adj_m],
        (pl.col("margin") + pl.col("o_margin")).alias("margin"),
    )
    adj_cols = [f"{s}_{m}" for m in adj_m for s in ("off", "def")] + ["margin"]
    adj = _decayed(j, adj_cols, prior).drop("n_games").rename({c: f"{c}_adj" for c in adj_cols})
    return raw.join(adj, on=["team", "season", "week"], how="left")


def qb_ratings(pbp: pl.DataFrame, sched: pl.DataFrame, half_life: float = 12.0, k: float = 120.0,
               prior: float = -0.10) -> pl.DataFrame:
    """Scheduled starter's decayed dropback EPA/play (shrunk toward a replacement-level prior with k plays)."""
    db = pbp.filter((pl.col("pass") == 1) & pl.col("passer_id").is_not_null() & pl.col("epa").is_not_null()
                    & pl.col("play_type").is_in(["pass", "run"]))
    qg = (db.group_by("passer_id", "game_id", "season", "week")
          .agg(n=pl.len().cast(pl.Float64), e=pl.col("epa").sum())
          .rename({"passer_id": "qb_id", "season": "g_season", "week": "g_week"})
          .sort("g_season", "g_week").with_columns(seq=pl.int_range(pl.len()).over("qb_id")))
    starts = pl.concat([sched.select("game_id", "season", "week", qb_id="home_qb_id"),
                        sched.select("game_id", "season", "week", qb_id="away_qb_id")]).drop_nulls("qb_id").unique()
    j = starts.join(qg.drop("game_id"), on="qb_id", how="inner").filter(
        ((pl.col("g_season") < pl.col("season")) | ((pl.col("g_season") == pl.col("season")) & (pl.col("g_week") < pl.col("week"))))
        & (pl.col("g_season") >= pl.col("season") - 3))
    j = j.with_columns(ago=(pl.col("seq").max().over("game_id", "qb_id") - pl.col("seq")).cast(pl.Float64))
    j = j.with_columns(w=(0.5 ** (pl.col("ago") / half_life)) * (0.8 ** (pl.col("season") - pl.col("g_season"))))
    r = j.group_by("game_id", "qb_id").agg(num=(pl.col("w") * pl.col("e")).sum(), den=(pl.col("w") * pl.col("n")).sum(),
                                          qb_plays=pl.col("n").sum())
    r = r.with_columns(qb_epa=(pl.col("num") + k * prior) / (pl.col("den") + k))
    out = starts.join(r.select("game_id", "qb_id", "qb_epa", "qb_plays"), on=["game_id", "qb_id"], how="left")
    return out.with_columns(pl.col("qb_epa").fill_null(prior), pl.col("qb_plays").fill_null(0.0)).select(
        "game_id", "qb_id", "qb_epa", "qb_plays")


def build_team_game_features(sched: pl.DataFrame | None = None, pbp: pl.DataFrame | None = None) -> pl.DataFrame:
    sched = sched if sched is not None else pl.read_parquet(CACHE / "schedules.parquet")
    pbp = pbp if pbp is not None else load_pbp()
    tg = team_game_stats(pbp, sched)
    rat = asof_ratings(tg, sched)
    qb = qb_ratings(pbp, sched)
    side = []
    for s, o in (("home", "away"), ("away", "home")):
        side.append(sched.select("game_id", "season", "week", team=f"{s}_team", opponent=f"{o}_team",
                                 is_home=pl.lit(s == "home"), qb_id=f"{s}_qb_id"))
    tgf = pl.concat(side).with_columns(team_fr=pl.col("team").replace(FRANCHISE))
    # previous game's starting QB for the QB-change flag
    tgf = tgf.sort("season", "week").with_columns(prev_qb_id=pl.col("qb_id").shift(1).over("team_fr"))
    tgf = tgf.with_columns(qb_change=(pl.col("qb_id") != pl.col("prev_qb_id")).fill_null(False))
    tgf = (tgf.join(rat.rename({"team": "team_fr"}), on=["team_fr", "season", "week"], how="left")
           .join(qb, on=["game_id", "qb_id"], how="left"))
    return tgf.sort("season", "week", "game_id", "is_home", descending=[False, False, False, True])


if __name__ == "__main__":
    import time
    t = time.time()
    f = build_team_game_features()
    print(f.shape, time.time() - t)
    print(f.filter(pl.col("season") == 2026).head(4))

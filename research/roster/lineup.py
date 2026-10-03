"""Expected lineup per team-game and its strength by unit, versus the team's recent actual lineups.

For a team-game G (kickoff K):
  candidates  = players on the team's pre-game depth chart for G (weekly chart; 2025+ latest snapshot before K)
  P(play)     = calibrated from (chart depth, injury-report status, played the team's previous game);
                the table is measured once on 2013-2017 snap participation (labels only, no game outcomes)
  share s     = expected snap share: decayed mean exposure per appearance blended with the chart depth prior
  value v     = player rating as of K (players.py), per full game above replacement
  U_exp[unit] = sum P * s * v      U_full[unit] = sum s * v (everyone healthy)
  U_act6      = mean over the team's previous 6 games of sum(actual exposure * v), with v as of K (so the
                comparison isolates who is playing, not rating drift). Pre-2013 games have no snap counts; the
                "actual" exposure there is the expected P*s of that game.
  delta       = U_exp - U_act6 ("lineup delta"), inj = U_exp - U_full ("injury cost").
QB and K are handled separately (scheduled starting QB from schedules; kicker = chart K or last team kicker).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from players import METRICS, APPLIES, player_values, qb_values, kick_values

DEPTH_SHARE = {1: 0.85, 2: 0.30, 3: 0.10}
UNITS = METRICS  # rec, rush, prush, cover, rund, ol
N_PREV = 6


def _status_class(st: pl.Expr) -> pl.Expr:
    return (pl.when(st.is_null()).then(pl.lit("none"))
            .when(st >= 0.9).then(pl.lit("out"))
            .when(st >= 0.5).then(pl.lit("q_dnp"))
            .when(st >= 0.25).then(pl.lit("q_lim"))
            .when(st >= 0.1).then(pl.lit("q_full"))
            .otherwise(pl.lit("minor")))


def prev_games(tg: pl.DataFrame, n: int, max_days: int = 400) -> pl.DataFrame:
    """For each (game_id, team): the franchise's previous n games (game_id_prev, lag 1..n)."""
    t = tg.sort("fr", "kick").with_columns(pl.int_range(pl.len()).over("fr").alias("ix"))
    rows = []
    for lag in range(1, n + 1):
        rows.append(t.select("game_id", "team", "fr", (pl.col("ix") - lag).alias("ix"), pl.lit(lag).alias("lag")))
    lk = pl.concat(rows).join(t.select("fr", "ix", pl.col("game_id").alias("gprev"), pl.col("kick").alias("kprev")),
                              on=["fr", "ix"], how="inner")
    return lk.join(tg.select("game_id", "team", "kick"), on=["game_id", "team"]).filter(
        (pl.col("kprev") < pl.col("kick")) & ((pl.col("kick") - pl.col("kprev")).dt.total_days() <= max_days)
    ).select("game_id", "team", "gprev", "lag")


def candidates(tg: pl.DataFrame, dc: pl.DataFrame, inj: pl.DataFrame, plog: pl.DataFrame,
               snap_games: pl.Series) -> pl.DataFrame:
    c = dc.filter(pl.col("grp").is_in(["OL", "RB", "WR", "TE", "DL", "LB", "DB"])).join(
        tg.select("game_id", "team", "date", "season", "kick"), on=["game_id", "team"], how="inner")
    c = c.join(inj.select("game_id", "team", "gsis_id", "p_miss", "report_status"), on=["game_id", "team", "gsis_id"],
               how="left")
    # played the franchise's previous game? (only defined when that game has snap counts)
    # previous game must be recent (same season, <= 30 days): last season's finale often rests starters
    pg = prev_games(tg, 1, max_days=30).select("game_id", "team", "gprev")
    c = c.join(pg, on=["game_id", "team"], how="left")
    played = plog.filter(pl.col("fge") > 0).select(pl.col("game_id").alias("gprev"), "gsis_id", pl.lit(1).alias("pp"))
    c = c.join(played, on=["gprev", "gsis_id"], how="left").with_columns(
        pl.when(pl.col("gprev").is_in(snap_games.implode())).then(pl.col("pp").fill_null(0).cast(pl.Utf8))
        .otherwise(pl.lit("na")).alias("prev_played"))
    return c.with_columns(_status_class(pl.col("p_miss")).alias("st"),
                          pl.col("depth").clip(1, 3).alias("dbk")).drop("pp")


def calibrate_play(c: pl.DataFrame, plog: pl.DataFrame, seasons=(2013, 2017)) -> pl.DataFrame:
    """P(plays) by (depth bucket, status class, played previous game), measured on the given seasons."""
    x = c.filter(pl.col("season").is_between(*seasons)).join(
        plog.filter(pl.col("fge") > 0).select("game_id", "gsis_id", pl.lit(1.0).alias("y")),
        on=["game_id", "gsis_id"], how="left").with_columns(pl.col("y").fill_null(0.0))
    full = x.group_by("dbk", "st", "prev_played").agg(pl.col("y").sum().alias("s"), pl.len().alias("n"))
    marg = x.group_by("dbk", "st").agg((pl.col("y").sum() / pl.len()).alias("pm"))
    tab = full.filter(pl.col("prev_played") != "na").join(marg, on=["dbk", "st"]).with_columns(((pl.col("s") + 20 * pl.col("pm")) / (pl.col("n") + 20)).alias("p_play"))
    # "na" (no snap data for the previous game) -> marginal over prev_played
    na = marg.with_columns(pl.lit("na").alias("prev_played"), pl.col("pm").alias("p_play"))
    return pl.concat([tab.select("dbk", "st", "prev_played", "p_play"), na.select("dbk", "st", "prev_played", "p_play")])


def build(tg: pl.DataFrame, dc: pl.DataFrame, inj: pl.DataFrame, plog: pl.DataFrame, ql: pl.DataFrame,
          kl: pl.DataFrame, cal: dict, snap_games: pl.Series, ptab: pl.DataFrame | None = None,
          min_season: int = 2003) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Returns (team-game features, candidate detail, P(play) table)."""
    tgx = tg.filter(pl.col("season") >= min_season)
    c = candidates(tgx, dc, inj, plog, snap_games)
    if ptab is None:
        ptab = calibrate_play(c, plog)
    c = c.join(ptab, on=["dbk", "st", "prev_played"], how="left").with_columns(pl.col("p_play").fill_null(0.5))
    # relative to a healthy player at the same chart depth (status none, played last game) so that a fully
    # healthy lineup has P = 1 and value-rich teams do not carry a spurious "expected absence" cost
    healthy = ptab.filter((pl.col("st") == "none") & (pl.col("prev_played") == "1")).select(
        "dbk", pl.col("p_play").alias("p_h"))
    c = c.join(healthy, on="dbk", how="left").with_columns(
        (pl.col("p_play") / pl.col("p_h")).clip(0.0, 1.0).alias("p_play")).drop("p_h")
    vcols = [f"v_{m}" for m in METRICS]
    keep = ["gsis_id", "date", "grp", *vcols, "u_share", "recent_apps", "exp_fge", "last_date", "tgt_rate"]
    vals = player_values(c.select("gsis_id", "date", "grp").unique(), plog, cal).select(keep)
    # ---- expected lineup
    c = c.join(vals, on=["gsis_id", "date", "grp"], how="left")
    prior = pl.col("depth").clip(1, 3).replace_strict(DEPTH_SHARE, default=0.10, return_dtype=pl.Float64)
    ra = pl.col("recent_apps").fill_null(0.0)
    c = c.with_columns(
        ((ra * pl.col("u_share").fill_null(0.0) + 1.5 * prior) / (ra + 1.5)).alias("s_exp"))
    c = c.with_columns((pl.col("p_play") * pl.col("s_exp")).alias("w_exp"),
                       (pl.col("exp_fge").fill_null(0.0) + 1).log().alias("lexp"))
    agg = []
    for m in METRICS:
        mask = pl.col("grp").is_in(APPLIES[m])
        agg += [(pl.col("w_exp") * pl.col(f"v_{m}").fill_null(0.0)).filter(mask).sum().alias(f"U_{m}"),
                (pl.col("s_exp") * pl.col(f"v_{m}").fill_null(0.0)).filter(mask).sum().alias(f"F_{m}")]
    agg += [
        (pl.col("w_exp") * pl.col("lexp")).filter(pl.col("grp") == "OL").sum().alias("U_olexp"),
        (pl.col("s_exp") * pl.col("lexp")).filter(pl.col("grp") == "OL").sum().alias("F_olexp"),
        ((pl.col("depth") == 1) & (pl.col("p_play") < 0.5)).sum().alias("starters_out"),
        ((pl.col("depth") == 1) & (pl.col("p_play") < 0.5) & (pl.col("s_exp") >= 0.6)).sum().alias("regulars_out"),
        (pl.col("w_exp").filter(pl.col("grp").is_in(["OL", "RB", "WR", "TE"])).sum()).alias("off_share"),
        (pl.col("w_exp").filter(pl.col("grp").is_in(["DL", "LB", "DB"])).sum()).alias("def_share"),
    ]
    U = c.group_by("game_id", "team").agg(agg)
    # ---- trailing lineups: the expected lineups of the previous N games (same construction, so no scale
    # mismatch), re-valued with player ratings as of this game's date
    pg = prev_games(tgx, N_PREV)
    pa = pg.join(tgx.select("game_id", "team", "date"), on=["game_id", "team"]).join(
        c.select(pl.col("game_id").alias("gprev"), "team", "gsis_id", "grp", pl.col("w_exp").alias("w")),
        on=["gprev", "team"], how="inner")
    vals2 = player_values(pa.select("gsis_id", "date", "grp").unique(), plog, cal).select(keep)
    pa = pa.join(vals2, on=["gsis_id", "date", "grp"], how="left")
    pa = pa.with_columns((pl.col("exp_fge").fill_null(0.0) + 1).log().alias("lexp"))
    agg2 = [(pl.col("w") * pl.col(f"v_{m}").fill_null(0.0)).filter(pl.col("grp").is_in(APPLIES[m])).sum().alias(f"A_{m}")
            for m in METRICS]
    agg2 += [(pl.col("w") * pl.col("lexp")).filter(pl.col("grp") == "OL").sum().alias("A_olexp")]
    per_game = pa.group_by("game_id", "team", "gprev").agg(agg2)
    A = per_game.group_by("game_id", "team").agg([pl.col(f"A_{m}").mean() for m in METRICS + ["olexp"]]
                                                + [pl.len().alias("n_prev")])
    # ---- QB: scheduled starter vs the starters of the previous 6 games (all rated as of this game)
    qb_now = tgx.filter(pl.col("qb_id").is_not_null()).select("game_id", "team", pl.col("qb_id").alias("gsis_id"),
                                                               "date", "season")
    qprev = pg.join(tg.select(pl.col("game_id").alias("gprev"), "team", pl.col("qb_id").alias("gsis_id")),
                    on=["gprev", "team"]).filter(pl.col("gsis_id").is_not_null()).join(
        tgx.select("game_id", "team", "date", "season"), on=["game_id", "team"])
    qq = pl.concat([qb_now.select("gsis_id", "date", "season"), qprev.select("gsis_id", "date", "season")]).unique()
    qv = qb_values(qq, ql)
    qcols = ["qb_epa", "qb_epa_long", "qb_cpoe", "qb_sack_rate", "qb_plays_w"]
    Q = qb_now.join(qv, on=["gsis_id", "date", "season"], how="left").select("game_id", "team", *qcols)
    Qp = qprev.join(qv, on=["gsis_id", "date", "season"], how="left").group_by("game_id", "team").agg(
        pl.col("qb_epa").mean().alias("qb_epa_prev6"))
    # ---- kicker
    kc = dc.filter((pl.col("grp") == "K")).sort("game_id", "team", "depth", "gsis_id").unique(
        ["game_id", "team"], keep="first", maintain_order=True).select(
        "game_id", "team", "gsis_id")
    kk = tgx.select("game_id", "team", "fr", "date", "kick").join(kc, on=["game_id", "team"], how="left")
    kv = kick_values(kk.filter(pl.col("gsis_id").is_not_null()).select("gsis_id", "date"), kl)
    K = kk.filter(pl.col("gsis_id").is_not_null()).with_columns(kv["k_val"]).select("game_id", "team", "k_val")
    feat = (tgx.select("game_id", "team", "season", "week", "date", "is_home")
            .join(U, on=["game_id", "team"], how="left").join(A, on=["game_id", "team"], how="left")
            .join(Q, on=["game_id", "team"], how="left").join(Qp, on=["game_id", "team"], how="left")
            .join(K, on=["game_id", "team"], how="left"))
    feat = feat.with_columns(
        [(pl.col(f"U_{m}") - pl.col(f"A_{m}")).alias(f"delta_{m}") for m in METRICS + ["olexp"]]
        + [(pl.col(f"U_{m}") - pl.col(f"F_{m}")).alias(f"inj_{m}") for m in METRICS + ["olexp"]]
        + [(pl.col("qb_epa") - pl.col("qb_epa_prev6")).alias("delta_qb"),
           pl.col("k_val").fill_null(-0.05)])
    return feat, c, ptab

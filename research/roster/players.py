"""Player-game production log and point-in-time player value ratings.

Every player-game row carries an exposure `fge` (full-game equivalents = the player's share of his unit's snaps;
2013+ from snap counts, earlier from that week's depth chart: starter 0.85, 2nd 0.30, 3rd 0.10, off-chart 0.25)
and production numerators in EPA-like units:

  rec    receiving EPA                                   (WR / TE / RB)
  rush   rushing EPA (non-QB)                            (RB)
  prush  1.7*sacks + 0.6*QB hits                         (DL / LB)
  cover  4.5*INT + 0.9*passes defended                   (DB / LB)
  rund   tackles for loss (solo tackles excluded: box scores mix in special-teams tackles)  (DL / LB / DB)
  ol     pass protection on the field: team (league sack+0.5*hit rate - own rate) * dropbacks * 1.8,
         credited to OL by exposure                      (OL)
  olx    1 per full game played (experience; no replacement level)
QB value comes from pbp (qb_epa per dropback/designed run, CPOE, sack rate), the kicker's from FG made over a
distance-based expectation.

Ratings at kickoff date D use only that player's games with date < D (as-of join, exact dates excluded):
    v = (sum_w x - repl * sum_w fge) / (sum_w fge + K),  w = 0.5 ** ((D - d) / HL)
i.e. production per full game above replacement, shrunk toward replacement (0) by exposure. Ratings follow the
player across teams and seasons (gsis id).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from data import CACHE, pg_expr

T0 = pl.date(1999, 1, 1)

# value metrics: (name, half-life days, K in full games). K and replacement levels are calibrated on
# 2013-2017 player stats only (calibrate()), no game outcomes involved.
METRICS = ["rec", "rush", "prush", "cover", "rund", "ol"]
HL = 365.0
QB_HL, QB_K, QB_PRIOR = 120.0, 100.0, -0.20   # personnel-family settings (tuned on 2006-11 there)
QB_HL_LONG, QB_PRIOR_LONG = 400.0, -0.10
K_KICK, KICK_PRIOR = 40.0, -0.03               # FG attempts; prior = made-over-expected per attempt


def _days(col: str = "date") -> pl.Expr:
    return (pl.col(col) - T0).dt.total_days().cast(pl.Float64)


# ------------------------------------------------------------------ pbp-derived logs
def pbp_frame() -> pl.DataFrame:
    return (pl.scan_parquet(CACHE / "pbp.parquet")
            .select("season", "game_id", "game_date", "posteam", "defteam", "play_type", "qb_dropback", "qb_spike",
                    "qb_kneel", "passer_id", "rusher_id", "qb_epa", "cpoe", "sack", "qb_hit",
                    "field_goal_attempt", "field_goal_result", "kick_distance", "kicker_player_id")
            .collect())


def qb_log(p: pl.DataFrame) -> pl.DataFrame:
    """Per (game, team, QB): plays, EPA sum, CPOE sum/count, sacks, dropbacks."""
    p = p.filter(pl.col("posteam").is_not_null())
    drop = p.filter((pl.col("qb_dropback") == 1) & (pl.col("qb_spike") != 1) & pl.col("passer_id").is_not_null()
                    & pl.col("qb_epa").is_not_null()).with_columns(pl.col("passer_id").alias("qb"))
    passers = drop.select("game_id", "qb").unique()
    runs = (p.filter((pl.col("play_type") == "run") & (pl.col("qb_dropback") != 1) & (pl.col("qb_kneel") != 1)
                     & pl.col("rusher_id").is_not_null() & pl.col("qb_epa").is_not_null())
            .with_columns(pl.col("rusher_id").alias("qb"), pl.lit(None, dtype=pl.Float64).alias("cpoe"),
                          pl.lit(0.0).alias("sack"), pl.lit(0.0).alias("qb_dropback"))
            .join(passers, on=["game_id", "qb"], how="semi"))
    cols = ["season", "game_id", "game_date", "posteam", "qb", "qb_epa", "cpoe", "sack", "qb_dropback"]
    allp = pl.concat([drop.select(cols).with_columns(pl.col("sack").cast(pl.Float64), pl.col("qb_dropback").cast(pl.Float64)),
                      runs.select(cols).with_columns(pl.col("sack").cast(pl.Float64), pl.col("qb_dropback").cast(pl.Float64))])
    return (allp.group_by("season", "game_id", "game_date", "posteam", "qb")
            .agg(pl.len().alias("n"), pl.col("qb_epa").sum().alias("epa"), pl.col("cpoe").sum().alias("cpoe"),
                 pl.col("cpoe").count().alias("cpoe_n"), pl.col("sack").sum().alias("sk"),
                 pl.col("qb_dropback").sum().alias("db"))
            .with_columns(pl.col("game_date").str.to_date().alias("date")).rename({"qb": "gsis_id"})
            .sort("gsis_id", "date"))


def team_passpro(p: pl.DataFrame) -> pl.DataFrame:
    """Per (game, offense): dropbacks, sacks, QB hits (pass protection on the field)."""
    d = p.filter((pl.col("qb_dropback") == 1) & pl.col("posteam").is_not_null())
    return d.group_by("season", "game_id", "posteam").agg(
        pl.len().alias("db"), pl.col("sack").sum().alias("sk"), pl.col("qb_hit").sum().alias("hit"))


def kick_log(p: pl.DataFrame) -> pl.DataFrame:
    """Per (game, kicker): FG attempts and made-minus-expected (expectation: logistic in distance fit on
    1999-2005 attempts, with a linear season trend frozen at that fit)."""
    fg = p.filter((pl.col("field_goal_attempt") == 1) & pl.col("kick_distance").is_not_null()
                  & pl.col("kicker_player_id").is_not_null()).with_columns(
        (pl.col("field_goal_result") == "made").cast(pl.Float64).alias("made"))
    from sklearn.linear_model import LogisticRegression
    tr = fg.filter(pl.col("season") <= 2005)
    X = np.c_[tr["kick_distance"].to_numpy(), tr["kick_distance"].to_numpy() ** 2]
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X / 50.0, tr["made"].to_numpy())
    Xa = np.c_[fg["kick_distance"].to_numpy(), fg["kick_distance"].to_numpy() ** 2] / 50.0
    fg = fg.with_columns(pl.Series("xp", lr.predict_proba(Xa)[:, 1]))
    # league-wide accuracy drifts up over time: express made-over-expected relative to the *previous* season's
    # league mean residual (no same-season information)
    lg = fg.group_by("season").agg((pl.col("made") - pl.col("xp")).mean().alias("lg")).sort("season")
    lg = lg.with_columns(pl.col("lg").shift(1).fill_null(0.0).alias("lg_prev")).select("season", "lg_prev")
    fg = fg.join(lg, on="season").with_columns((pl.col("made") - pl.col("xp") - pl.col("lg_prev")).alias("moe"))
    return (fg.group_by("season", "game_id", "game_date", "kicker_player_id")
            .agg(pl.len().alias("att"), pl.col("moe").sum().alias("moe"))
            .with_columns(pl.col("game_date").str.to_date().alias("date"))
            .rename({"kicker_player_id": "gsis_id"}).sort("gsis_id", "date"))


# ------------------------------------------------------------------ non-QB player-game log
def player_log(tg: pl.DataFrame, sn: pl.DataFrame, dc: pl.DataFrame, pp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team, gsis_id) the player took part in, with grp, fge and production."""
    ps = pl.read_parquet(CACHE / "player_stats_week.parquet", columns=[
        "player_id", "position", "game_id", "team", "targets", "receiving_epa", "receiving_yards", "carries",
        "rushing_epa", "def_sacks", "def_qb_hits", "def_interceptions", "def_pass_defended", "def_tackles_for_loss",
        "def_tackles_solo"]).rename({"player_id": "gsis_id"}).filter(pl.col("gsis_id").is_not_null())
    ps = ps.with_columns(pg_expr("position").alias("grp_ps")).with_columns(pl.col(pl.Float64, pl.Int32, pl.Int64).fill_null(0))
    ps = ps.with_columns(
        pl.col("receiving_epa").alias("rec"),
        pl.when(pl.col("grp_ps") == "QB").then(0.0).otherwise(pl.col("rushing_epa")).alias("rush"),
        (1.7 * pl.col("def_sacks") + 0.6 * pl.col("def_qb_hits")).alias("prush"),
        (4.5 * pl.col("def_interceptions") + 0.9 * pl.col("def_pass_defended")).alias("cover"),
        (1.0 * pl.col("def_tackles_for_loss")).alias("rund"),
        pl.col("targets").cast(pl.Float64).alias("tgt"),
    ).select("game_id", "team", "gsis_id", "grp_ps", "rec", "rush", "prush", "cover", "rund", "tgt")

    snp = sn.with_columns(
        pl.when(pl.col("grp").is_in(["DL", "LB", "DB"])).then(pl.col("def_pct"))
        .when(pl.col("grp").is_in(["QB", "OL", "RB", "WR", "TE"])).then(pl.col("off_pct"))
        .otherwise(pl.max_horizontal("off_pct", "def_pct")).alias("snap_fge")
    ).select("game_id", "team", "gsis_id", pl.col("grp").alias("grp_sn"), "snap_fge")
    snap_games = sn.select("game_id").unique()

    dcx = dc.select("game_id", "team", "gsis_id", pl.col("grp").alias("grp_dc"), "depth")
    base = (ps.select("game_id", "team", "gsis_id")
            .join(snp.select("game_id", "team", "gsis_id"), on=["game_id", "team", "gsis_id"], how="full", coalesce=True))
    # pre-snap era: offensive linemen rarely show up in box-score stats; treat depth-chart starting OL as played
    # (their exposure is still discounted by the chart estimate).  Only for games without snap data.
    ol_dc = (dcx.filter((pl.col("grp_dc") == "OL") & (pl.col("depth") == 1))
             .join(snap_games, on="game_id", how="anti").select("game_id", "team", "gsis_id"))
    base = pl.concat([base, ol_dc]).unique()
    g = (base.join(ps, on=["game_id", "team", "gsis_id"], how="left")
         .join(snp, on=["game_id", "team", "gsis_id"], how="left")
         .join(dcx, on=["game_id", "team", "gsis_id"], how="left")
         .join(tg.select("game_id", "team", "date", "season"), on=["game_id", "team"], how="inner"))
    has_snaps = pl.col("game_id").is_in(snap_games["game_id"].implode())
    dfge = pl.col("depth").replace_strict({1: 0.85, 2: 0.30, 3: 0.10}, default=0.10, return_dtype=pl.Float64)
    g = g.with_columns(
        pl.coalesce("grp_dc", "grp_sn", "grp_ps").alias("grp"),
        pl.when(has_snaps).then(pl.col("snap_fge").fill_null(0.10))
        .otherwise(pl.when(pl.col("depth").is_null()).then(0.25).otherwise(dfge)).alias("fge"),
    ).filter(pl.col("grp").is_not_null() & (pl.col("grp") != "QB") & (pl.col("grp") != "K"))
    g = g.with_columns(pl.col(["rec", "rush", "prush", "cover", "rund", "tgt"]).fill_null(0.0))
    # OL pass protection on the field
    lgr = pp.group_by("season").agg(((pl.col("sk") + 0.5 * pl.col("hit")).sum() / pl.col("db").sum()).alias("lg"))
    lgr = lgr.sort("season").with_columns(pl.col("lg").shift(1).alias("lg_prev")).with_columns(
        pl.col("lg_prev").fill_null(pl.col("lg")))
    ppx = pp.join(lgr.select("season", "lg_prev"), on="season").with_columns(
        ((pl.col("lg_prev") - (pl.col("sk") + 0.5 * pl.col("hit")) / pl.col("db")) * pl.col("db") * 1.8).alias("pp_val")
    ).select("game_id", pl.col("posteam").alias("team"), "pp_val")
    g = g.join(ppx, on=["game_id", "team"], how="left").with_columns(
        pl.when(pl.col("grp") == "OL").then(pl.col("pp_val").fill_null(0.0) * pl.col("fge") / 5.0).otherwise(0.0).alias("ol"))
    return g.select("game_id", "team", "season", "date", "gsis_id", "grp", "fge", *METRICS, "tgt").sort("gsis_id", "date")


# ------------------------------------------------------------------ calibration (stats only, 2013-2017)
APPLIES = {  # which metrics count for which position group
    "rec": ["WR", "TE", "RB"], "rush": ["RB"], "prush": ["DL", "LB"], "cover": ["DB", "LB"],
    "rund": ["DL", "LB", "DB"], "ol": ["OL"],
}


def calibrate(pl_log: pl.DataFrame, seasons=(2013, 2017)) -> dict:
    """Replacement level (per full game) and shrinkage K per (metric, grp), from player stats of the given seasons.

    replacement = exposure-weighted mean rate of players whose season exposure is < 4 full games;
    K = split-half (odd/even games) reliability within player-seasons with >= 8 full games: K = n_half (1-r)/r.
    """
    x = pl_log.filter(pl.col("season").is_between(*seasons) & (pl.col("fge") > 0.05))
    x = x.sort("gsis_id", "season", "date", "game_id", "team").with_columns(
        pl.int_range(pl.len()).over("gsis_id", "season").alias("k"))
    out = {}
    for m, grps in APPLIES.items():
        for gp in grps:
            y = x.filter(pl.col("grp") == gp)
            ps = y.group_by("gsis_id", "season").agg(pl.col("fge").sum().alias("f"), pl.col(m).sum().alias("v"))
            rep = ps.filter(pl.col("f") < 4)
            repl = float(rep["v"].sum() / max(rep["f"].sum(), 1e-9))
            half = y.with_columns((pl.col("k") % 2).alias("h")).group_by("gsis_id", "season", "h").agg(
                pl.col("fge").sum().alias("f"), pl.col(m).sum().alias("v"))
            wide = half.pivot(on="h", index=["gsis_id", "season"], values=["f", "v"]).drop_nulls().sort(
                "gsis_id", "season")
            wide = wide.filter((pl.col("f_0") >= 4) & (pl.col("f_1") >= 4))
            r0 = (wide["v_0"] / wide["f_0"]).to_numpy()
            r1 = (wide["v_1"] / wide["f_1"]).to_numpy()
            r = float(np.corrcoef(r0, r1)[0, 1]) if len(r0) > 30 else 0.1
            r = min(max(r, 0.05), 0.95)
            nh = float(np.median(np.r_[wide["f_0"].to_numpy(), wide["f_1"].to_numpy()])) if len(r0) else 4.0
            out[(m, gp)] = (repl, nh * (1 - r) / r)
    return out


# ------------------------------------------------------------------ as-of ratings
def _decayed_state(log: pl.DataFrame, cols: list[str], hl: float, key="gsis_id") -> pl.DataFrame:
    """Per player-game row: decayed cumulative sums (scaled by 2**(t/hl)); query must rescale by 2**(-D/hl)."""
    w = (2.0 ** (_days() / hl))
    log = log.group_by(key, "date").agg([pl.col(c).sum() for c in cols]).sort(key, "date")
    return log.with_columns(
        [(pl.col(c) * w).cum_sum().over(key).alias(f"S_{c}") for c in cols]
        + [pl.col("date").alias("last_date")])


def asof(queries: pl.DataFrame, state: pl.DataFrame, cols: list[str], hl: float, key="gsis_id") -> pl.DataFrame:
    """queries: key, date (+ anything). Adds decayed sums D_<c> using only state rows with date < query date."""
    st = state.select(key, "date", "last_date", *[f"S_{c}" for c in cols]).sort("date")
    q = queries.with_row_index("_qi").sort("date")
    j = q.join_asof(st, on="date", by=key, strategy="backward", allow_exact_matches=False, check_sortedness=False)
    scale = 2.0 ** (-_days() / hl)
    j = j.with_columns([(pl.col(f"S_{c}").fill_null(0.0) * scale).alias(f"D_{c}") for c in cols]).drop(
        [f"S_{c}" for c in cols])
    return j.sort("_qi").drop("_qi")


def player_values(queries: pl.DataFrame, plog: pl.DataFrame, cal: dict) -> pl.DataFrame:
    """queries: gsis_id, date, grp. Adds v_<metric> (per full game above replacement, shrunk), exp_fge
    (decayed exposure), n_games, and usual share u_share (mean exposure over last appearances, HL 60 days)."""
    cols = METRICS + ["fge", "tgt"]
    st = _decayed_state(plog.with_columns(pl.lit(1.0).alias("one")), cols + ["one"], HL)
    q = asof(queries, st, cols + ["one"], HL)
    exprs = []
    for m in METRICS:
        e = pl.lit(0.0)
        for gp in APPLIES[m]:
            repl, k = cal[(m, gp)]
            e = pl.when(pl.col("grp") == gp).then(
                (pl.col(f"D_{m}") - repl * pl.col("D_fge")) / (pl.col("D_fge") + k)).otherwise(e)
        exprs.append(e.alias(f"v_{m}"))
    q = q.with_columns(exprs).with_columns(
        (pl.col("D_tgt") / (pl.col("D_fge") + 4.0)).alias("tgt_rate"))
    # usual share: short memory mean exposure per appearance
    st2 = _decayed_state(plog.with_columns(pl.lit(1.0).alias("one")), ["fge", "one"], 60.0)
    q2 = asof(queries.select("gsis_id", "date"), st2, ["fge", "one"], 60.0)
    q = q.with_columns(
        (q2["D_fge"] / q2["D_one"]).fill_nan(None).alias("u_share"),
        q2["D_one"].alias("recent_apps"),
        pl.col("D_fge").alias("exp_fge"),
        pl.col("last_date"),
    )
    return q.drop([c for c in q.columns if c.startswith("D_")])


def qb_values(queries: pl.DataFrame, ql: pl.DataFrame) -> pl.DataFrame:
    """queries: gsis_id, date, season. QB EPA/play over replacement (form + talent), CPOE, sack rate."""
    lg = ql.group_by("season").agg((pl.col("epa").sum() / pl.col("n").sum()).alias("m")).sort("season")
    lg = lg.with_columns(pl.col("m").shift(1).fill_null(pl.col("m")).alias("ref")).select("season", "ref")
    out = queries
    for hl, prior, nm in ((QB_HL, QB_PRIOR, "qb_epa"), (QB_HL_LONG, QB_PRIOR_LONG, "qb_epa_long")):
        st = _decayed_state(ql, ["epa", "n"], hl)
        q = asof(queries.select("gsis_id", "date", "season"), st, ["epa", "n"], hl).join(lg, on="season", how="left")
        q = q.with_columns(pl.col("ref").fill_null(lg["ref"][-1]))
        v = ((pl.col("D_epa") - pl.col("ref") * pl.col("D_n") + QB_K * prior) / (pl.col("D_n") + QB_K))
        out = out.with_columns(q.select(v.alias(nm))[nm])
        if nm == "qb_epa":
            out = out.with_columns(q["D_n"].alias("qb_plays_w"))
    st = _decayed_state(ql, ["cpoe", "cpoe_n", "sk", "db"], QB_HL_LONG)
    q = asof(queries.select("gsis_id", "date"), st, ["cpoe", "cpoe_n", "sk", "db"], QB_HL_LONG)
    out = out.with_columns(
        q.select(((pl.col("D_cpoe") - 3.0 * 200) / (pl.col("D_cpoe_n") + 200)).alias("qb_cpoe"))["qb_cpoe"],
        q.select(((pl.col("D_sk") + 0.075 * 150) / (pl.col("D_db") + 150)).alias("qb_sack_rate"))["qb_sack_rate"],
    )
    return out


def kick_values(queries: pl.DataFrame, kl: pl.DataFrame) -> pl.DataFrame:
    """Kicker FG made-over-expected per attempt (shrunk), converted to points per game (x 1.7 att x 3 pts)."""
    st = _decayed_state(kl, ["moe", "att"], 730.0)
    q = asof(queries.select("gsis_id", "date"), st, ["moe", "att"], 730.0)
    v = (pl.col("D_moe") + K_KICK * KICK_PRIOR) / (pl.col("D_att") + K_KICK) * 1.7 * 3.0
    return queries.with_columns(q.select(v.alias("k_val"))["k_val"])

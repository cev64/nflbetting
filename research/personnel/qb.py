"""QB value model.

Per-QB rating from play-by-play: EPA per QB play (dropbacks incl. sacks/scrambles + designed QB runs,
using nflverse `qb_epa`) and CPOE, exponentially decayed in time (carries across seasons and teams because
it is keyed on the player's gsis id), and shrunk toward a replacement-level prior by effective sample size.

A rating queried for a game on date D only uses that QB's games with date < D (strictly earlier games).
League reference level for season S is the league QB EPA/play of season S-1 (no same-season info).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from common import CACHE, schedules

# hyper-parameters: K/prior first checked on 2001-2005 next-game QB EPA (tune()), then HL/K/prior chosen by
# walk-forward game-outcome logloss on test seasons 2006-2011 only (never 2012+). See NOTES.md.
HL_DAYS = 120.0       # half-life of a past game's weight, in days
K_EPA = 100.0         # prior strength, in QB plays
PRIOR_EPA_OFF = -0.20 # replacement level relative to last season's league average QB EPA/play
HL_LONG_DAYS = 400.0  # second, slower-moving rating ("talent" vs "form"); prior -0.10
PRIOR_LONG_OFF = -0.10
K_CPOE = 200.0
PRIOR_CPOE = -3.0


def qb_game_log(asof=None) -> pl.DataFrame:
    """One row per (game, team, QB): plays, EPA sum, CPOE sum/count. `asof` (date) keeps only games before it
    (used by the leakage test)."""
    p = (
        pl.scan_parquet(CACHE / "pbp.parquet")
        .select("season", "game_id", "game_date", "posteam", "play_type", "qb_dropback", "qb_kneel",
                "qb_spike", "passer_id", "rusher_id", "qb_epa", "cpoe")
        .filter(pl.col("qb_epa").is_not_null() & pl.col("posteam").is_not_null())
        .collect()
    )
    drop = p.filter((pl.col("qb_dropback") == 1) & (pl.col("qb_spike") != 1) & pl.col("passer_id").is_not_null())
    drop = drop.with_columns(pl.col("passer_id").alias("qb"))
    passers = drop.select("game_id", "qb").unique()
    runs = (
        p.filter((pl.col("play_type") == "run") & (pl.col("qb_dropback") != 1) & (pl.col("qb_kneel") != 1)
                 & pl.col("rusher_id").is_not_null())
        .with_columns(pl.col("rusher_id").alias("qb"))
        .join(passers, on=["game_id", "qb"], how="semi")
        .with_columns(pl.lit(None, dtype=pl.Float64).alias("cpoe"))
    )
    cols = ["season", "game_id", "game_date", "posteam", "qb", "qb_epa", "cpoe"]
    allp = pl.concat([drop.select(cols), runs.select(cols)])
    g = (
        allp.group_by("season", "game_id", "game_date", "posteam", "qb")
        .agg(pl.len().alias("n"), pl.col("qb_epa").sum().alias("epa_sum"),
             pl.col("cpoe").sum().alias("cpoe_sum"), pl.col("cpoe").count().alias("cpoe_n"))
        .with_columns(pl.col("game_date").str.to_date().alias("date"))
        .sort("qb", "date")
    )
    if asof is not None:
        g = g.filter(pl.col("date") < asof)
    return g


def league_ref(log: pl.DataFrame) -> dict[int, float]:
    """season S -> league mean QB EPA/play of season S-1 (previous season only => no leakage)."""
    s = log.group_by("season").agg((pl.col("epa_sum").sum() / pl.col("n").sum()).alias("m")).sort("season")
    m = dict(zip(s["season"].to_list(), s["m"].to_list()))
    first = min(m)
    return {S: m.get(S - 1, m[first]) for S in range(first, max(m) + 2)}


def rate(log: pl.DataFrame, queries: pl.DataFrame, hl=HL_DAYS, k=K_EPA, prior_off=PRIOR_EPA_OFF,
         kc=K_CPOE, prior_cpoe=PRIOR_CPOE) -> pl.DataFrame:
    """queries: columns qb, date, season. Returns queries + qb_epa (rel. to league ref), qb_cpoe,
    qb_eff_n (decayed plays), qb_career_n, qb_games, qb_days_since."""
    ref = league_ref(log)
    logs = {}
    for qb, sub in log.group_by("qb"):
        d = sub["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
        logs[qb[0]] = (d, sub["n"].to_numpy().astype(float), sub["epa_sum"].to_numpy(),
                       sub["cpoe_sum"].fill_null(0).to_numpy(), sub["cpoe_n"].to_numpy().astype(float))
    q = queries.with_row_index("_i")
    out = np.full((q.height, 6), np.nan)
    qd = q["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    qs = q["season"].to_numpy()
    qq = q["qb"].to_list()
    lam = np.log(2) / hl
    for i in range(q.height):
        r = ref.get(int(qs[i]), 0.0)
        L = logs.get(qq[i])
        if L is None:
            out[i] = (prior_off, prior_cpoe, 0.0, 0.0, 0.0, np.nan)
            continue
        d, n, es, cs, cn = L
        m = d < qd[i]
        if not m.any():
            out[i] = (prior_off, prior_cpoe, 0.0, 0.0, 0.0, np.nan)
            continue
        w = np.exp(-lam * (qd[i] - d[m]))
        # league reference subtracted per play using the reference of each game's season would be
        # cleaner; using the query season's reference keeps it simple and leak-free.
        ew = (w * es[m]).sum()
        nw = (w * n[m]).sum()
        val = (ew + k * (r + prior_off)) / (nw + k) - r
        cw = (w * cs[m]).sum()
        cnw = (w * cn[m]).sum()
        cval = (cw + kc * prior_cpoe) / (cnw + kc)
        out[i] = (val, cval, nw, n[m].sum(), m.sum(), qd[i] - d[m].max())
    names = ["qb_epa", "qb_cpoe", "qb_eff_n", "qb_career_n", "qb_games", "qb_days_since"]
    return queries.with_columns([pl.Series(nm, out[:, j]) for j, nm in enumerate(names)])


def tune(log: pl.DataFrame, seasons=(2001, 2005)) -> pl.DataFrame:
    """Grid over (HL, K, prior) scoring weighted MSE of predicting each QB game's EPA/play
    (games with >= 10 plays) in the given seasons. Uses no game outcomes / no test seasons."""
    tgt = log.filter(pl.col("season").is_between(*seasons) & (pl.col("n") >= 10))
    q = tgt.select("qb", "date", "season", "n", "epa_sum")
    ref = league_ref(log)
    rows = []
    for hl in (150, 300, 500, 1000):
        for k in (100, 250, 500):
            for po in (-0.05, -0.12, -0.2):
                r = rate(log, q.select("qb", "date", "season"), hl=hl, k=k, prior_off=po)
                y = (q["epa_sum"] / q["n"]).to_numpy() - np.array([ref[s] for s in q["season"].to_list()])
                e = r["qb_epa"].to_numpy() - y
                rows.append((hl, k, po, float(np.average(e ** 2, weights=q["n"].to_numpy()))))
    return pl.DataFrame(rows, schema=["hl", "k", "prior_off", "wmse"], orient="row").sort("wmse")


def team_qb_features(log: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per (game_id, team): starting QB rating and change/backup/rookie flags."""
    if log is None:
        log = qb_game_log()
    s = schedules().filter(pl.col("home_qb_id").is_not_null())
    long = pl.concat([
        s.select("game_id", "season", "week", "date", pl.col("home_team").alias("team"),
                 pl.col("home_qb_id").alias("qb"), pl.col("home_qb_name").alias("qb_name")),
        s.select("game_id", "season", "week", "date", pl.col("away_team").alias("team"),
                 pl.col("away_qb_id").alias("qb"), pl.col("away_qb_name").alias("qb_name")),
    ])
    from common import FRANCHISE
    long = long.with_columns(pl.col("team").replace(FRANCHISE).alias("franchise")).sort("franchise", "date")
    # previous game's starter for the same franchise (across seasons), and starts in last 8 team games
    long = long.with_columns(
        pl.col("qb").shift(1).over("franchise").alias("prev_qb"),
        pl.col("season").shift(1).over("franchise").alias("prev_season"),
    )
    # share of the franchise's previous 8 games started by this QB
    rows = long.select("franchise", "qb").to_dict(as_series=False)
    fr, qbs = rows["franchise"], rows["qb"]
    share = np.full(len(qbs), np.nan)
    i0 = 0
    for i in range(len(qbs)):
        if i == 0 or fr[i] != fr[i - 1]:
            i0 = i
        prev = qbs[max(i0, i - 8):i]
        share[i] = (sum(1 for x in prev if x == qbs[i]) / len(prev)) if prev else np.nan
    long = long.with_columns(pl.Series("qb_start_share8", share))
    cur = rate(log, long.select("qb", "date", "season"))
    cur_long = rate(log, long.select("qb", "date", "season"), hl=HL_LONG_DAYS, prior_off=PRIOR_LONG_OFF)
    prev = rate(log, long.select(pl.col("prev_qb").fill_null("NONE").alias("qb"), "date", "season"))
    # rookies: first season with any QB plays is this season (or no plays at all)
    first_season = log.group_by("qb").agg(pl.col("season").min().alias("first_season"))
    out = long.with_columns(
        cur["qb_epa"], cur_long["qb_epa"].alias("qb_epa_long"), cur["qb_cpoe"], cur["qb_eff_n"], cur["qb_career_n"], cur["qb_games"],
        prev["qb_epa"].alias("prev_qb_epa"),
    ).join(first_season, on="qb", how="left")
    out = out.with_columns(
        (pl.col("qb") != pl.col("prev_qb")).fill_null(False).cast(pl.Int8).alias("qb_change"),
        ((pl.col("qb") != pl.col("prev_qb")) & (pl.col("season") == pl.col("prev_season")))
        .fill_null(False).cast(pl.Int8).alias("qb_change_inseason"),
        (pl.col("qb_epa") - pl.col("prev_qb_epa")).alias("qb_epa_delta"),
        ((pl.col("first_season").is_null()) | (pl.col("first_season") >= pl.col("season")))
        .cast(pl.Int8).alias("qb_rookie"),
        (pl.col("qb_start_share8").fill_null(1.0) < 0.5).cast(pl.Int8).alias("qb_backup"),
        (pl.col("qb_career_n") + 1).log().alias("qb_log_exp"),
    )
    return out.select("game_id", "team", "qb", "qb_name", "qb_epa", "qb_epa_long", "qb_cpoe", "qb_eff_n", "qb_log_exp",
                      "qb_change", "qb_change_inseason", "qb_epa_delta", "qb_rookie", "qb_backup",
                      "qb_start_share8")


if __name__ == "__main__":
    import time
    t = time.time()
    log = qb_game_log()
    print(log.shape, time.time() - t)
    print(tune(log).head(12))

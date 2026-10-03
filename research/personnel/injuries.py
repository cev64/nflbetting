"""Injury / availability load per team-game.

For each (season, week, team) injury report (the final pre-game report, filed before kickoff), each listed
player gets a probability of missing the game from his game status x Friday practice status, and an
importance weight from snap shares in strictly earlier games:

  imp_recent : mean snap share over the franchise's previous 4 games (0 when he did not play) -> "how much of the
               lineup the team has been fielding recently is now missing" (new absences)
  imp_usual  : mean snap share over the player's last <=6 appearances (any team) within 400 days ->
               "how important is this player when healthy" (includes long-running absences)

Snap counts exist from 2013. For 2009-2012 (and any player without snap history) the importance falls back
to the previous week's depth chart: starter 0.85, 2nd string 0.30, deeper 0.10.

Load for position group G = sum over listed players in G of p_miss * importance.
"""
from __future__ import annotations

import bisect
from collections import defaultdict

import numpy as np
import polars as pl

from common import CACHE, FRANCHISE, GROUPS, norm_name, pos_group, schedules

# P(miss game | report status, practice status). Measured once from 2013-2025 snap-count participation of
# listed players (a data-quality calibration of what the labels mean, not fitted to game outcomes);
# rounded.  Keys: (status, practice-prefix).  Probable was abolished after 2015.
P_MISS = {
    ("Out", None): 1.0,
    ("Doubtful", None): 0.97,
    ("Questionable", "Did"): 0.60,
    ("Questionable", "Lim"): 0.32,
    ("Questionable", "Ful"): 0.20,
    ("Questionable", None): 0.32,
    ("Probable", None): 0.05,
    (None, "Did"): 0.28,
    (None, "Out"): 0.9,
    (None, "Lim"): 0.07,
    (None, "Ful"): 0.04,
    (None, None): 0.04,
}


def p_miss(status: str | None, prac: str | None) -> float:
    st = status if status in {"Out", "Doubtful", "Questionable", "Probable"} else None
    pr = (prac or "").strip()[:3] or None
    if pr not in {"Did", "Lim", "Ful", "Out"}:
        pr = None
    if (st, pr) in P_MISS:
        return P_MISS[(st, pr)]
    if (st, None) in P_MISS:
        return P_MISS[(st, None)]
    return P_MISS[(None, pr)] if (None, pr) in P_MISS else 0.04


def load_injuries() -> pl.DataFrame:
    inj = pl.read_parquet(CACHE / "injuries.parquet").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    # strictness: drop the handful (<0.1%) of rows whose last modification is after the game's kickoff
    # (mostly 2020 rescheduled games); date_modified is null for 2009 and 2025+ (weekly snapshots).
    s = schedules()
    kick = pl.concat([
        s.select("season", "week", pl.col(t).alias("team"), "gameday", "gametime") for t in ("home_team", "away_team")
    ]).with_columns(pl.concat_str("gameday", pl.lit(" "), pl.col("gametime").fill_null("13:00"))
                    .str.to_datetime("%Y-%m-%d %H:%M").dt.replace_time_zone("America/New_York", ambiguous="earliest")
                    .dt.convert_time_zone("UTC").alias("kick")).select("season", "week", "team", "kick")
    inj = inj.join(kick, on=["season", "week", "team"], how="left").filter(
        pl.col("date_modified").is_null() | pl.col("kick").is_null() | (pl.col("date_modified") < pl.col("kick")))
    inj = inj.with_columns(
        pl.col("full_name").map_elements(norm_name, return_dtype=pl.Utf8).alias("nm"),
        pl.col("position").map_elements(pos_group, return_dtype=pl.Utf8).alias("grp"),
        pl.struct("report_status", "practice_status").map_elements(
            lambda r: p_miss(r["report_status"], r["practice_status"]), return_dtype=pl.Float64).alias("p_miss"),
    ).filter(pl.col("grp").is_not_null())
    # one row per player-week (keep the most severe if duplicated)
    return inj.sort("p_miss", descending=True).unique(["season", "week", "team", "nm", "grp"], keep="first")


def _snap_tables(sched: pl.DataFrame, asof=None):
    sc = pl.read_parquet(CACHE / "snap_counts.parquet").join(
        sched.select("game_id", "date"), on="game_id", how="inner")
    if asof is not None:
        sc = sc.filter(pl.col("date") < asof)
    sc = sc.with_columns(
        pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8).alias("nm"),
        pl.col("position").map_elements(pos_group, return_dtype=pl.Utf8).alias("grp"),
        pl.max_horizontal(pl.col("offense_pct").fill_null(0), pl.col("defense_pct").fill_null(0)).alias("pct"),
        pl.col("team").replace(FRANCHISE).alias("fr"),
    ).filter(pl.col("grp").is_not_null())
    # franchise game list (games with snap data), sorted by date
    fr_games = defaultdict(list)
    for fr, gid, d in sc.select("fr", "game_id", "date").unique().sort("date").iter_rows():
        fr_games[fr].append((d, gid))
    pct_by = {(gid, fr, nm): p for gid, fr, nm, p in sc.select("game_id", "fr", "nm", "pct").iter_rows()}
    # player appearance history (name + group), across teams
    hist = defaultdict(list)
    for nm, grp, d, p in sc.select("nm", "grp", "date", "pct").sort("date").iter_rows():
        hist[(nm, grp)].append((d, p))
    return fr_games, pct_by, hist


def _depth_table() -> dict:
    """(season, week, gsis_id) -> importance from that week's depth chart (old format, 2001-2024)."""
    d = pl.read_parquet(CACHE / "depth_charts.parquet").filter(
        pl.col("season").is_not_null() & pl.col("formation").is_in(["Offense", "Defense"]))
    d = d.with_columns(pl.col("depth_team").cast(pl.Int32, strict=False)).filter(pl.col("depth_team").is_not_null())
    d = d.group_by("season", "week", "gsis_id").agg(pl.col("depth_team").min())
    imp = {1: 0.85, 2: 0.30}
    return {(s, w, g): imp.get(dt, 0.10) for s, w, g, dt in d.iter_rows()}


def injury_loads(asof=None) -> pl.DataFrame:
    """Per (season, week, team): load_recent_<G>, load_usual_<G>, n_out_starters, plus player detail."""
    sched = schedules()
    inj = load_injuries()
    # game date for each (season, week, team) report
    tg = pl.concat([
        sched.select("season", "week", "date", pl.col("home_team").alias("team")),
        sched.select("season", "week", "date", pl.col("away_team").alias("team")),
    ])
    inj = inj.join(tg, on=["season", "week", "team"], how="inner")
    fr_games, pct_by, hist = _snap_tables(sched, asof)
    depth = _depth_table()

    rec, usu, src = [], [], []
    for season, week, team, nm, grp, gsis, d in inj.select(
            "season", "week", "team", "nm", "grp", "gsis_id", "date").iter_rows():
        fr = FRANCHISE.get(team, team)
        # recent: franchise's previous 4 games with snap data, within 300 days
        games = fr_games.get(fr, [])
        k = bisect.bisect_left(games, (d, ""))
        prev = [g for g in games[max(0, k - 4):k] if (d - g[0]).days <= 300]
        if len(prev) >= 2:
            r = sum(pct_by.get((gid, fr, nm), 0.0) for _, gid in prev) / len(prev)
            h = hist.get((nm, grp), [])
            j = bisect.bisect_left(h, (d, -1.0))
            hp = [p for dd, p in h[max(0, j - 6):j] if (d - dd).days <= 400]
            u = float(np.mean(hp)) if hp else r
            s = "snap"
        else:
            # depth-chart fallback (previous week's chart; week 1 uses its own pre-season chart)
            r = u = depth.get((season, max(week - 1, 1), gsis), 0.05)
            s = "depth"
        rec.append(r)
        usu.append(u)
        src.append(s)
    inj = inj.with_columns(pl.Series("imp_recent", rec), pl.Series("imp_usual", usu), pl.Series("imp_src", src))
    inj = inj.with_columns(
        (pl.col("p_miss") * pl.col("imp_recent")).alias("miss_recent"),
        (pl.col("p_miss") * pl.col("imp_usual")).alias("miss_usual"),
    )
    agg = []
    for g in GROUPS:
        agg += [
            pl.col("miss_recent").filter(pl.col("grp") == g).sum().alias(f"inj_recent_{g}"),
            pl.col("miss_usual").filter(pl.col("grp") == g).sum().alias(f"inj_usual_{g}"),
        ]
    out = inj.group_by("season", "week", "team").agg(
        *agg,
        ((pl.col("p_miss") >= 0.5) & (pl.col("imp_usual") >= 0.6)).sum().alias("inj_starters_out"),
        # uncertain (Questionable) share of the load: "late news" the line may not fully price
        (pl.col("miss_usual").filter(pl.col("report_status") == "Questionable").sum()).alias("inj_q_load"),
        pl.col("miss_usual").sum().alias("inj_total"),
    )
    return out, inj


if __name__ == "__main__":
    import time
    t = time.time()
    out, det = injury_loads()
    print(time.time() - t, out.shape)
    print(det.group_by("imp_src").len())
    print(out.group_by("season").agg(pl.col("^inj_.*$").mean()).sort("season"))

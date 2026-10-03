"""Data loading for the roster (bottom-up lineup) family.

Everything here is a pure function of research/cache. Tables are keyed on the nflverse gsis player id.
Snap counts (2013+) carry only PFR ids, so a crosswalk to gsis is built from rosters_weekly
(pfr_id, then normalized name + team + season, then name + season).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
CACHE = RESEARCH / "cache"
PREDS = RESEARCH / "preds"
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)

FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: str | None) -> str | None:
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def nm_expr(col: str) -> pl.Expr:
    return pl.col(col).map_elements(norm_name, return_dtype=pl.Utf8, skip_nulls=True)


# ---------------------------------------------------------------- positions
OL = {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG", "OC"}
DL = {"DE", "DT", "NT", "DL", "EDGE", "LDE", "RDE", "LDT", "RDT", "LE", "RE", "NOSE", "UT", "END", "N", "LEO",
      "RUSH", "DPR", "NDT"}
LB = {"LB", "ILB", "OLB", "MLB", "WLB", "SLB", "LOLB", "ROLB", "LILB", "RILB", "WILL", "SAM", "MIKE", "JACK",
      "BLB", "$LB", "JLB", "RLB", "LLB", "OTTO", "WIL", "MO", "OL B", "LILBI", "0LB", "MIL"}
DB = {"CB", "S", "FS", "SS", "DB", "SAF", "LCB", "RCB", "NCB", "NB", "NKL", "NICK", "DS", "WS", "RS", "CS",
      "MCB", "NICKE", "LCR", "NDB", "S47"}
SKILL = {"WR", "TE", "RB", "FB", "HB", "SWR", "LWR", "RWR", "SE", "FL", "SLOT", "TE1", "TE2"}


def pos_group(pos: str | None) -> str | None:
    """QB, OL, RB, WR, TE, DL, LB, DB, K or None."""
    if pos is None:
        return None
    p = pos.upper().strip().split("/")[0].strip()
    if p == "QB":
        return "QB"
    if p in OL:
        return "OL"
    if p in {"RB", "HB", "FB"}:
        return "RB"
    if p in {"WR", "SWR", "LWR", "RWR", "SE", "FL", "SLOT"}:
        return "WR"
    if p in {"TE", "TE1", "TE2"}:
        return "TE"
    if p in DL:
        return "DL"
    if p in LB:
        return "LB"
    if p in DB:
        return "DB"
    if p in {"K", "PK"}:
        return "K"
    return None


def pg_expr(col: str) -> pl.Expr:
    return pl.col(col).map_elements(pos_group, return_dtype=pl.Utf8, skip_nulls=True)


# ---------------------------------------------------------------- schedule / team-games
def schedules() -> pl.DataFrame:
    s = pl.read_parquet(CACHE / "schedules.parquet")
    return s.with_columns(
        pl.col("gameday").str.to_date().alias("date"),
        pl.concat_str("gameday", pl.lit(" "), pl.col("gametime").fill_null("13:00"))
        .str.to_datetime("%Y-%m-%d %H:%M").dt.replace_time_zone("America/New_York", ambiguous="earliest")
        .dt.convert_time_zone("UTC").alias("kick"),
    )


def team_games(s: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (game_id, team) for every scheduled game (played or not)."""
    if s is None:
        s = schedules()
    cols = ["game_id", "season", "week", "game_type", "date", "kick"]
    tg = pl.concat([
        s.select(*cols, pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"),
                 pl.lit(1).alias("is_home"), pl.col("home_qb_id").alias("qb_id")),
        s.select(*cols, pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"),
                 pl.lit(0).alias("is_home"), pl.col("away_qb_id").alias("qb_id")),
    ])
    return tg.with_columns(pl.col("team").replace(FRANCHISE).alias("fr")).sort("fr", "kick", "game_id")


# ---------------------------------------------------------------- id crosswalk + snaps
def rosters() -> pl.DataFrame:
    r = pl.read_parquet(CACHE / "rosters_weekly.parquet",
                        columns=["season", "week", "team", "gsis_id", "pfr_id", "full_name", "position",
                                 "depth_chart_position", "status", "birth_date", "entry_year", "rookie_year",
                                 "draft_number", "years_exp"])
    return r.filter(pl.col("gsis_id").is_not_null())


def snaps(s: pl.DataFrame | None = None) -> pl.DataFrame:
    """Snap counts with gsis ids. Columns: game_id, season, week, team, gsis_id, grp, off_pct, def_pct, st_pct."""
    if s is None:
        s = schedules()
    sc = pl.read_parquet(CACHE / "snap_counts.parquet").with_columns(nm_expr("player").alias("nm"))
    r = rosters()
    by_pfr = r.filter(pl.col("pfr_id").is_not_null()).group_by("pfr_id").agg(pl.col("gsis_id").mode().first())
    rr = r.with_columns(nm_expr("full_name").alias("nm"))
    by_nts = rr.group_by("nm", "team", "season").agg(pl.col("gsis_id").n_unique().alias("k"),
                                                     pl.col("gsis_id").first().alias("g2")).filter(pl.col("k") == 1)
    by_ns = rr.group_by("nm", "season").agg(pl.col("gsis_id").n_unique().alias("k"),
                                            pl.col("gsis_id").first().alias("g3")).filter(pl.col("k") == 1)
    sc = (sc.join(by_pfr, left_on="pfr_player_id", right_on="pfr_id", how="left")
          .join(by_nts.drop("k"), on=["nm", "team", "season"], how="left")
          .join(by_ns.drop("k"), on=["nm", "season"], how="left"))
    sc = sc.with_columns(pl.coalesce("gsis_id", "g2", "g3").alias("gsis_id"))
    # last resort: a stable synthetic id from pfr id (still a consistent player key across seasons)
    sc = sc.with_columns(pl.coalesce("gsis_id", pl.concat_str(pl.lit("pfr:"), "pfr_player_id")).alias("gsis_id"))
    sc = sc.with_columns(pg_expr("position").alias("grp"),
                         (pl.col("offense_pct").fill_null(0)).alias("off_pct"),
                         (pl.col("defense_pct").fill_null(0)).alias("def_pct"),
                         (pl.col("st_pct").fill_null(0)).alias("st_pct"))
    return sc.select("game_id", "season", "week", "team", "gsis_id", "player", "position", "grp",
                     "off_pct", "def_pct", "st_pct").unique(["game_id", "gsis_id"], keep="first")


# ---------------------------------------------------------------- depth charts
def depth_charts(tg: pl.DataFrame) -> pl.DataFrame:
    """Pre-game depth chart per (game_id, team): gsis_id, grp, depth (1 = starter), unit (O/D/S).

    2001-2024: the weekly chart labelled with the game's (season, week).
    2025+: timestamped daily snapshots; the latest snapshot strictly before kickoff is used.
    """
    d = pl.read_parquet(CACHE / "depth_charts.parquet")
    old = d.filter(pl.col("season").is_not_null() & pl.col("formation").is_in(["Offense", "Defense", "Special Teams"]))
    old = old.with_columns(
        pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth"),
        pl.col("formation").replace({"Offense": "O", "Defense": "D", "Special Teams": "S"}).alias("unit"),
        pg_expr("depth_position").alias("grp_dc"), pg_expr("position").alias("grp_pos"),
    ).filter(pl.col("depth").is_not_null() & pl.col("gsis_id").is_not_null())
    old = old.with_columns(pl.coalesce(
        pl.when(pl.col("unit") == "S").then(pl.when(pl.col("depth_position").is_in(["K", "PK"])).then(pl.lit("K"))),
        pl.when(pl.col("unit") != "S").then(pl.coalesce("grp_dc", "grp_pos"))).alias("grp"))
    old = old.filter(pl.col("grp").is_not_null()).select(
        "season", "week", pl.col("club_code").alias("team"), "gsis_id", "grp", "depth", "unit")
    old = old.join(tg.select("game_id", "season", "week", "team"), on=["season", "week", "team"], how="inner")
    old = old.group_by("game_id", "team", "gsis_id").agg(pl.col("depth").min(), pl.col("grp").sort_by("depth").first(),
                                                         pl.col("unit").sort_by("depth").first())
    # new format
    new = d.filter(pl.col("season").is_null() & pl.col("dt").is_not_null()).with_columns(
        pl.col("dt").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC").alias("ts"))
    new = new.with_columns(
        pl.when(pl.col("pos_grp") == "Special Teams").then(pl.lit("S"))
        .when(pl.col("pos_grp").str.contains(" D$|Defense|Nickel|Dime|D$")).then(pl.lit("D"))
        .otherwise(pl.lit("O")).alias("unit"),
        pg_expr("pos_abb").alias("grp"),
    )
    new = new.with_columns(
        pl.when(pl.col("unit") == "S").then(pl.when(pl.col("grp") == "K").then(pl.lit("K"))).otherwise(pl.col("grp")).alias("grp")
    ).filter(pl.col("grp").is_not_null() & pl.col("gsis_id").is_not_null())
    snaps_ts = new.select("team", "ts").unique().sort("ts")
    tgn = tg.filter(pl.col("kick") >= pl.datetime(2025, 3, 1, time_zone="UTC")).select("game_id", "team", "kick").sort("kick")
    pick = tgn.join_asof(snaps_ts.rename({"ts": "snap_ts"}), left_on="kick", right_on="snap_ts", by="team",
                         strategy="backward", allow_exact_matches=False).filter(pl.col("snap_ts").is_not_null())
    nn = new.join(pick.select("game_id", "team", pl.col("snap_ts").alias("ts")), on=["team", "ts"], how="inner")
    nn = nn.with_columns(pl.col("pos_rank").cast(pl.Int32).alias("depth"))
    nn = nn.group_by("game_id", "team", "gsis_id").agg(pl.col("depth").min(), pl.col("grp").sort_by("depth").first(),
                                                       pl.col("unit").sort_by("depth").first())
    return pl.concat([old, nn.select(old.columns)]).unique(["game_id", "team", "gsis_id"])


# ---------------------------------------------------------------- injuries
P_MISS = {  # same label calibration as the personnel family (snap participation of listed players)
    ("Out", None): 1.0, ("Doubtful", None): 0.97, ("Questionable", "Did"): 0.60, ("Questionable", "Lim"): 0.32,
    ("Questionable", "Ful"): 0.20, ("Questionable", None): 0.32, ("Probable", None): 0.05,
    (None, "Did"): 0.28, (None, "Out"): 0.9, (None, "Lim"): 0.07, (None, "Ful"): 0.04, (None, None): 0.04,
}


def _p_miss(status, prac) -> float:
    st = status if status in {"Out", "Doubtful", "Questionable", "Probable"} else None
    pr = (prac or "").strip()[:3] or None
    if pr not in {"Did", "Lim", "Ful", "Out"}:
        pr = None
    for k in ((st, pr), (st, None), (None, pr)):
        if k in P_MISS:
            return P_MISS[k]
    return 0.04


def injuries(tg: pl.DataFrame) -> pl.DataFrame:
    """Final pre-game injury report per (game_id, team, gsis_id) with p_miss. Rows modified at/after kickoff dropped."""
    inj = pl.read_parquet(CACHE / "injuries.parquet").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)).filter(pl.col("gsis_id").is_not_null())
    inj = inj.join(tg.select("game_id", "season", "week", "team", "kick"), on=["season", "week", "team"], how="inner")
    inj = inj.filter(pl.col("date_modified").is_null() | (pl.col("date_modified") < pl.col("kick")))
    inj = inj.with_columns(pl.lit(0).alias("stale"))
    # weeks whose reports are not published yet (upcoming slate): carry each team's latest report of the same
    # season forward, flagged stale. Only applies to (season, week) with no report rows for any team.
    have = inj.select("season", "week").unique()
    last_s = inj["season"].max()
    todo = tg.filter((pl.col("season") == last_s) & (pl.col("game_type") == "REG")).join(have, on=["season", "week"], how="anti")
    if todo.height:
        lastwk = inj.filter(pl.col("season") == last_s).group_by("team").agg(pl.col("week").max().alias("wk_src"))
        src = inj.filter(pl.col("season") == last_s).join(lastwk, left_on=["team", "week"], right_on=["team", "wk_src"], how="inner").drop("game_id", "kick")
        add = todo.select("game_id", "team", "kick", "week").join(lastwk, on="team", how="inner").filter(
            pl.col("week") > pl.col("wk_src")).select("game_id", "team", "kick")
        add = add.join(src.drop("week", "season", "stale"), on="team", how="inner").with_columns(pl.lit(1).alias("stale"))
        cols = ["game_id", "team", "kick", "gsis_id", "full_name", "position", "report_status", "practice_status",
                "stale"]
        inj = pl.concat([inj.select(cols), add.select(cols)], how="vertical_relaxed")
    inj = inj.with_columns(
        pl.struct("report_status", "practice_status").map_elements(
            lambda r: _p_miss(r["report_status"], r["practice_status"]), return_dtype=pl.Float64).alias("p_miss"),
        pg_expr("position").alias("inj_grp"))
    return (inj.sort("p_miss", descending=True).unique(["game_id", "team", "gsis_id"], keep="first")
            .select("game_id", "team", "gsis_id", "full_name", "position", "inj_grp", "report_status",
                    "practice_status", "p_miss", "stale"))

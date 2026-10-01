"""Fetch NFL play-by-play from nflverse and build per-team turnover game logs.

Writes one JSON file per season to web/data/<season>.json plus an index file
(web/data/seasons.json) that the website reads to know which seasons exist.

Usage:
    python backend/fetch_data.py                   # current season only
    python backend/fetch_data.py --seasons 2024 2025
    python backend/fetch_data.py --since 2020      # 2020 through current season
    python backend/fetch_data.py --model-only      # rebuild model.json from data on disk

Only fully completed weeks are published (see last_complete_week), so running
this mid-week never adds Thursday/Sunday games before Monday night is final.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from kicks import kicks_report
from model import model_report

DATA_DIR = Path(__file__).resolve().parent.parent / "web" / "data"

# nflverse play-by-play and schedules use "LA" for the Rams; load_teams() lists
# both "LA" and "LAR", so we key everything off the schedule's abbreviations.


def turnover_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per turnover: which team gave it away and what kind it was.

    - Interceptions: the offense (posteam) gives the ball away.
    - Lost fumbles: charged to the team that fumbled (fumbled_1_team). This
      correctly handles return-team fumbles on punts/kickoffs and a defender
      fumbling back an interception return (that play then counts as two
      turnovers, one for each team).
    """
    ints = pbp.filter(pl.col("interception") == 1).select(
        "game_id",
        pl.col("posteam").alias("team"),
        pl.lit("int").alias("kind"),
    )
    fums = pbp.filter(
        (pl.col("fumble_lost") == 1) & pl.col("fumbled_1_team").is_not_null()
    ).select(
        "game_id",
        pl.col("fumbled_1_team").alias("team"),
        pl.lit("fum").alias("kind"),
    )
    return pl.concat([ints, fums])


def fumble_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Total fumbles (lost or not) by team per game, for recovery-luck stats."""
    return (
        pbp.filter((pl.col("fumble") == 1) & pl.col("fumbled_1_team").is_not_null())
        .group_by("game_id", pl.col("fumbled_1_team").alias("team"))
        .agg(pl.len().alias("fumbles"))
    )


def drive_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per team per game: how far its drives got and how they ended.

    A drive "reached" a yard line if it ran a snap from there, so only plays with
    a down count (extra points, two-point tries and kickoffs carry the spot of the
    try or kick, not the drive). This is the usual red-zone-trip definition: a
    40-yard touchdown from the opponent's 45 is not a trip inside the 40.
    """
    drives = (
        pbp.filter(pl.col("posteam").is_not_null() & pl.col("down").is_not_null())
        .group_by("game_id", "posteam", "fixed_drive")
        .agg(
            pl.col("yardline_100").min().alias("best"),
            pl.col("fixed_drive_result").first().alias("result"),
        )
    )
    td = pl.col("result") == "Touchdown"
    return drives.group_by("game_id", pl.col("posteam").alias("team")).agg(
        pl.len().alias("drives"),
        (pl.col("best") <= 40).sum().alias("t40"),
        (pl.col("best") <= 20).sum().alias("rz"),
        ((pl.col("best") <= 20) & td).sum().alias("rz_td"),
        td.sum().alias("td"),
    )


def kicking_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per team per game: field goals and extra points (what kicker props settle on)."""
    fg = pl.col("field_goal_attempt") == 1
    made = fg & (pl.col("field_goal_result") == "made")
    xp = pl.col("extra_point_attempt") == 1
    return (
        pbp.filter(pl.col("posteam").is_not_null() & (fg | xp))
        .group_by("game_id", pl.col("posteam").alias("team"))
        .agg(
            fg.sum().alias("fga"),
            made.sum().alias("fgm"),
            (made & (pl.col("kick_distance") >= 50)).sum().alias("fg50"),
            xp.sum().alias("xpa"),
            (xp & (pl.col("extra_point_result") == "good")).sum().alias("xpm"),
        )
    )


# Drive and kicking columns, stored for the team's offense and (prefixed opp_)
# for the opponent's offense, i.e. what the team's defense allowed.
DRIVE_KEYS = ["drives", "t40", "rz", "rz_td", "td", "fga", "fgm", "fg50", "xpa", "xpm"]


def build_season(season: int) -> dict:
    full_sched = nfl.load_schedules([season])
    pbp = nfl.load_pbp([season])

    cutoff = last_complete_week(full_sched, set(pbp["game_id"].unique().to_list()))
    if cutoff is None:
        raise RuntimeError(f"No completed weeks found for {season}")
    played = pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    sched = full_sched.filter(played & (pl.col("week") <= cutoff))

    # One row per team per game, from that team's point of view.
    # `line` is the team's spread in betting notation (negative = favored);
    # nflverse spread_line is from the home side with positive = home favored.
    cols = ["game_id", "season", "week", "game_type", "gameday"]
    home = sched.select(
        *cols,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opp"),
        pl.lit(True).alias("home"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
        (-pl.col("spread_line")).alias("line"),
        pl.col("total_line").alias("total"),
    )
    away = sched.select(
        *cols,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opp"),
        pl.lit(False).alias("home"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
        pl.col("spread_line").alias("line"),
        pl.col("total_line").alias("total"),
    )
    games = pl.concat([home, away])

    to = (
        turnover_plays(pbp)
        .group_by("game_id", "team")
        .agg(
            (pl.col("kind") == "int").sum().alias("int"),
            (pl.col("kind") == "fum").sum().alias("fl"),
        )
    )
    fum = fumble_counts(pbp)
    drv = drive_stats(pbp).join(kicking_stats(pbp), on=["game_id", "team"], how="full", coalesce=True)

    # Join the team's own giveaways, then the opponent's (= this team's takeaways).
    games = (
        games.join(to, on=["game_id", "team"], how="left")
        .join(fum, on=["game_id", "team"], how="left")
        .join(
            to.rename({"team": "opp", "int": "opp_int", "fl": "opp_fl"}),
            on=["game_id", "opp"],
            how="left",
        )
        .join(
            fum.rename({"team": "opp", "fumbles": "opp_fumbles"}),
            on=["game_id", "opp"],
            how="left",
        )
        .join(drv, on=["game_id", "team"], how="left")
        .join(
            drv.rename({"team": "opp", **{k: f"opp_{k}" for k in DRIVE_KEYS}}),
            on=["game_id", "opp"],
            how="left",
        )
        .with_columns(
            pl.col(["int", "fl", "fumbles", "opp_int", "opp_fl", "opp_fumbles"])
            .fill_null(0)
            .cast(pl.Int32),
            pl.col(DRIVE_KEYS + [f"opp_{k}" for k in DRIVE_KEYS]).fill_null(0).cast(pl.Int32),
        )
        .sort("week", "gameday", "team")
    )

    records = [
        {
            "game_id": r["game_id"],
            "week": r["week"],
            "type": r["game_type"],  # REG, WC, DIV, CON, SB
            "date": r["gameday"],
            "team": r["team"],
            "opp": r["opp"],
            "home": r["home"],
            "pf": r["pf"],
            "pa": r["pa"],
            "line": r["line"],
            "total": r["total"],
            # Giveaways (offense / ball security)
            "int_thrown": r["int"],
            "fum_lost": r["fl"],
            "fumbles": r["fumbles"],
            # Takeaways (defense / turnovers forced)
            "int_made": r["opp_int"],
            "fum_rec": r["opp_fl"],
            "opp_fumbles": r["opp_fumbles"],
            # Drives, red zone and kicking: offense, then (opp_) what the defense allowed
            **{k: r[k] for k in DRIVE_KEYS},
            **{f"opp_{k}": r[f"opp_{k}"] for k in DRIVE_KEYS},
        }
        for r in games.iter_rows(named=True)
    ]

    abbrs = set(games["team"].to_list())
    teams = {
        r["team_abbr"]: {
            "name": r["team_name"],
            "nick": r["team_nick"],
            "conf": r["team_conf"],
            "division": r["team_division"],
            "color": r["team_color"],
            "color2": r["team_color2"],
            "logo": r["team_logo_espn"],
        }
        for r in nfl.load_teams().iter_rows(named=True)
        if r["team_abbr"] in abbrs
    }

    return {
        "season": season,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_week": cutoff,
        "teams": teams,
        "games": records,
        "upcoming": upcoming_games(full_sched, cutoff + 1),
    }


def last_complete_week(sched: pl.DataFrame, pbp_games: set[str]) -> int | None:
    """Latest week W such that every game in weeks 1..W is final and in the pbp.

    Only whole weeks are published, so a Thursday game never gives some teams
    an extra game over the rest of the league before the week wraps on Monday.
    A game with no score that is more than a week old (e.g. the cancelled
    2022 BUF-CIN game) is treated as never happening rather than as pending.
    """
    stale = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
    cutoff = None
    for (week,), wk in sched.sort("week").group_by("week", maintain_order=True):
        for r in wk.iter_rows(named=True):
            done = r["home_score"] is not None and r["game_id"] in pbp_games
            if not done and r["gameday"] >= stale:
                return cutoff
        cutoff = week
    return cutoff


def upcoming_games(sched: pl.DataFrame, week: int) -> list[dict]:
    """The slate for the week after the published data, with betting lines.

    Games already played that week (e.g. Thursday night) are included with
    their final score; their stats appear once the whole week is published.
    nflverse spread_line is from the home team's view: positive = home favored.
    """
    return [
        {
            "game_id": r["game_id"],
            "week": r["week"],
            "type": r["game_type"],
            "date": r["gameday"],
            "time": r["gametime"],
            "home": r["home_team"],
            "away": r["away_team"],
            "spread_line": r["spread_line"],
            "total_line": r["total_line"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
        }
        for r in sched.filter(pl.col("week") == week)
        .sort("gameday", "gametime")
        .iter_rows(named=True)
    ]


def write_index() -> None:
    seasons = sorted(
        (int(p.stem) for p in DATA_DIR.glob("*.json") if p.stem.isdigit()),
        reverse=True,
    )
    index = {
        "seasons": seasons,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (DATA_DIR / "seasons.json").write_text(json.dumps(index, indent=2) + "\n")


def main() -> None:
    current = nfl.get_current_season()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seasons", type=int, nargs="+", help="Seasons to fetch")
    group.add_argument("--since", type=int, help="Fetch this season through current")
    group.add_argument("--model-only", action="store_true", help="Rebuild model.json from the data on disk")
    args = parser.parse_args()

    if args.model_only:
        write_model(current)
        return

    if args.seasons:
        seasons = args.seasons
    elif args.since:
        seasons = list(range(args.since, current + 1))
    else:
        seasons = [current]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        try:
            data = build_season(season)
        except RuntimeError as e:
            print(f"Skipping {season}: {e}")
            continue
        out = DATA_DIR / f"{season}.json"
        out.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        print(f"{season}: {len(data['games'])} team-games through week {data['last_week']} -> {out}")

    write_index()
    write_model(current)


def write_model(current: int) -> None:
    """Backtest the spread signals on every season on disk and score this week."""
    seasons = {
        int(p.stem): json.loads(p.read_text()) for p in DATA_DIR.glob("*.json") if p.stem.isdigit()
    }
    if not seasons:
        return
    current = min(current, max(seasons))
    report = model_report(seasons, current)
    report["kicks"] = kicks_report(seasons, current)
    report["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (DATA_DIR / "model.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    live = report["signals"]["power"]["live"]
    print(f"model: {len(report['spots'])} games scored for {current}; power signal live {live['w']}-{live['l']}")
    if report["kicks"]:
        bt = report["kicks"]["signals"]["stall_bend"]["backtest"]
        print(f"kicks: {len(report['kicks']['board'])} team-games on the board; stall x bend backtest 2+ FGM {bt['rate']} (n={bt['n']})")


if __name__ == "__main__":
    main()

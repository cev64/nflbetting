"""Download every nflverse dataset the models use into research/cache/*.parquet.

    python backend/fetch_cache.py            # everything (about 3 minutes, ~350 MB)
    python backend/fetch_cache.py --quick    # only refresh what changes week to week when a cache exists

The cache is not committed; it is rebuilt by the daily workflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import polars as pl

CACHE = Path(__file__).resolve().parent.parent / "research" / "cache"
FIRST = 1999


def seasons(start: int) -> list[int]:
    return list(range(start, nfl.get_current_season() + 1))


# name -> (loader, refresh-per-season?)  Season-keyed sets can refresh only the current season.
DATASETS = {
    "schedules": lambda s: nfl.load_schedules(),
    "pbp": lambda s: nfl.load_pbp(s or seasons(FIRST)),
    "player_stats_week": lambda s: nfl.load_player_stats(s or seasons(FIRST), summary_level="week"),
    "injuries": lambda s: nfl.load_injuries(s or seasons(2009)),
    "rosters_weekly": lambda s: nfl.load_rosters_weekly(s or seasons(2002)),
    "depth_charts": lambda s: nfl.load_depth_charts(s or seasons(2001)),
    "snap_counts": lambda s: nfl.load_snap_counts(s or seasons(2012)),
    "ftn": lambda s: nfl.load_ftn_charting(s or seasons(2022)),
    "teams": lambda s: nfl.load_teams(),
    "officials": lambda s: nfl.load_officials(),
}


def season_col(df: pl.DataFrame) -> str | None:
    for c in ("season", "nflverse_game_id"):
        if c in df.columns:
            return c
    return None


def fetch(quick: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cur = nfl.get_current_season()
    for name, load in DATASETS.items():
        path = CACHE / f"{name}.parquet"
        try:
            if quick and path.exists() and name not in ("schedules", "teams", "officials"):
                old = pl.read_parquet(path)
                if "season" in old.columns:
                    try:
                        new = load([cur])
                    except Exception as e:  # current season not published yet for this dataset
                        print(f"{name}: no {cur} data ({type(e).__name__}); keeping cache")
                        continue
                    new = new.select([c for c in old.columns if c in new.columns])
                    df = pl.concat([old.filter(pl.col("season") != cur), new], how="diagonal_relaxed")
                    df.write_parquet(path)
                    print(f"{name}: refreshed {cur} -> {df.shape}")
                    continue
            df = load(None)
            df.write_parquet(path)
            print(f"{name}: {df.shape}")
        except Exception as e:
            if path.exists():
                print(f"{name}: download failed ({e!r:.200}); keeping the cached copy")
            else:
                raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="refresh only the current season where a cache exists")
    fetch(ap.parse_args().quick)

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
import pyarrow.parquet as pq

CACHE = Path(__file__).resolve().parent.parent / "research" / "cache"
# nflreadpy keeps every download in RAM by default; we write our own cache, so turn that off.
nfl.config.update_config(cache_mode="off")
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


def fetch_pbp(path: Path) -> None:
    """Play-by-play one season at a time, streamed into one file (all seasons at once peak at ~6 GB of RAM)."""
    parts = CACHE / "_pbp_parts"
    parts.mkdir(exist_ok=True)
    files = []
    for s in seasons(FIRST):
        f = parts / f"{s}.parquet"
        nfl.load_pbp([s]).write_parquet(f)
        files.append(f)
    # One unified schema, then append season by season so only one season is ever in memory.
    schema = pl.concat([pl.DataFrame(schema=pl.read_parquet_schema(f)) for f in files], how="diagonal_relaxed").schema
    tmp = path.with_suffix(".tmp")
    writer = None
    for f in files:
        df = pl.read_parquet(f)
        df = df.select([pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c) for c, t in schema.items()])
        table = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
        del df, table
    writer.close()
    tmp.replace(path)
    for f in files:
        f.unlink()
    parts.rmdir()


def fetch(quick: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cur = nfl.get_current_season()
    for name, load in DATASETS.items():
        path = CACHE / f"{name}.parquet"
        try:
            # Small sets, and depth charts (whose 2025+ format has no season column), always reload in full.
            if quick and path.exists() and name not in ("schedules", "teams", "officials", "depth_charts"):
                old = pl.read_parquet(path)
                if "season" in old.columns and old["season"].null_count() == 0:
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
            if name == "pbp":
                fetch_pbp(path)
                print(f"{name}: {pl.scan_parquet(path).select(pl.len()).collect().item()} plays")
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

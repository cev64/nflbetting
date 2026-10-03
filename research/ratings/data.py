"""Shared data loading for the ratings family (Elo, ridge/Massey, Kalman).

Everything here is per-game raw information; no model state. Models consume `load_games()` in
chronological order and only ever look at a game's outcome after they have produced its prediction.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
PREDS = HERE.parent / "preds"

# Approximate home-stadium coordinates (lat, lon) per franchise abbreviation (incl. relocated ones).
TEAM_COORDS = {
    "ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62), "BUF": (42.77, -78.79),
    "CAR": (35.23, -80.85), "CHI": (41.86, -87.62), "CIN": (39.10, -84.52), "CLE": (41.51, -81.70),
    "DAL": (32.75, -97.09), "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06),
    "HOU": (29.68, -95.41), "IND": (39.76, -86.16), "JAX": (30.32, -81.64), "KC": (39.05, -94.48),
    "LA": (33.95, -118.34), "STL": (38.63, -90.19), "LAC": (33.95, -118.34), "SD": (32.78, -117.12),
    "LV": (36.09, -115.18), "OAK": (37.75, -122.20), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26),
    "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07), "NYJ": (40.81, -74.07),
    "PHI": (39.90, -75.17), "PIT": (40.45, -80.02), "SEA": (47.60, -122.33), "SF": (37.40, -121.97),
    "TB": (27.98, -82.50), "TEN": (36.17, -86.77), "WAS": (38.91, -76.86),
}
# Neutral-site stadium ids -> coordinates (international + a few US venues).
STADIUM_COORDS = {
    "LON00": (51.56, -0.28), "LON01": (51.46, -0.34), "LON02": (51.60, -0.07), "MEX00": (19.30, -99.15),
    "GER00": (48.22, 11.62), "MUN01": (48.22, 11.62), "FRA00": (50.07, 8.65), "SAO00": (-23.55, -46.47),
    "RIO00": (-22.91, -43.23), "MAD01": (40.45, -3.69), "PAR00": (48.92, 2.36), "MEL00": (-37.82, 144.98),
    "BUF01": (43.64, -79.39), "VEG00": (36.09, -115.18), "PHO00": (33.53, -112.26), "PHO99": (33.43, -111.93),
    "JAX00": (30.32, -81.64), "DAL00": (32.75, -97.09), "MIA00": (25.96, -80.24), "DET00": (42.34, -83.05),
    "ATL00": (33.76, -84.40), "ATL97": (33.76, -84.40), "NOR00": (29.95, -90.08), "IND00": (39.76, -86.16),
    "SFO01": (37.40, -121.97), "NYC01": (40.81, -74.07), "HOU00": (29.68, -95.41), "SDG00": (32.78, -117.12),
    "TAM00": (27.98, -82.50), "LAX01": (33.95, -118.34), "MIN01": (44.97, -93.26), "PIT00": (40.45, -80.02),
    "CLE00": (41.51, -81.70),
}


def haversine_miles(a, b) -> float:
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(3958.8 * 2 * np.arcsin(np.sqrt(h)))


@lru_cache(maxsize=1)
def load_games() -> pl.DataFrame:
    """All games 1999-2026 sorted chronologically, with derived context columns."""
    s = pl.read_parquet(CACHE / "schedules.parquet")
    s = s.with_columns(
        pl.col("gametime").fill_null("13:00"),
        (pl.col("game_type") != "REG").alias("playoff"),
        (pl.col("location") == "Neutral").alias("neutral"),
        pl.col("result").is_not_null().alias("played"),
    ).sort(["gameday", "gametime", "game_id"])
    hd, ad = [], []
    for row in s.select("home_team", "away_team", "neutral", "stadium_id").iter_rows():
        h, a, neu, sid = row
        venue = STADIUM_COORDS.get(sid, TEAM_COORDS[h]) if neu else TEAM_COORDS[h]
        hd.append(haversine_miles(TEAM_COORDS[h], venue))
        ad.append(haversine_miles(TEAM_COORDS[a], venue))
    s = s.with_columns(pl.Series("home_travel", hd), pl.Series("away_travel", ad))
    return s


@lru_cache(maxsize=1)
def team_game_epa() -> pl.DataFrame:
    """Per game: home/away offensive EPA/play and success rate on scrimmage plays (pass + run).

    Columns: game_id, home_plays, home_epa, home_sr, away_plays, away_epa, away_sr
    (pbp uses current franchise abbreviations, so we key on home/away rather than team code.)
    """
    p = (
        pl.scan_parquet(CACHE / "pbp.parquet")
        .select("game_id", "posteam", "home_team", "play_type", "epa", "success")
        .filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
        .with_columns((pl.col("posteam") == pl.col("home_team")).alias("is_home"))
        .group_by("game_id", "is_home")
        .agg(pl.len().alias("plays"), pl.col("epa").mean().alias("epa"), pl.col("success").mean().alias("sr"))
        .collect()
    )
    h = p.filter(pl.col("is_home")).select("game_id", pl.col("plays").alias("home_plays"),
                                           pl.col("epa").alias("home_epa"), pl.col("sr").alias("home_sr"))
    a = p.filter(~pl.col("is_home")).select("game_id", pl.col("plays").alias("away_plays"),
                                            pl.col("epa").alias("away_epa"), pl.col("sr").alias("away_sr"))
    return h.join(a, on="game_id", how="inner")


@lru_cache(maxsize=1)
def qb_game_stats() -> pl.DataFrame:
    """Per game & QB: dropbacks (att + sacks + carries) and total EPA (passing + rushing) from player_stats_week."""
    ps = pl.read_parquet(
        CACHE / "player_stats_week.parquet",
        columns=["player_id", "game_id", "team", "position", "attempts", "sacks_suffered", "carries",
                 "passing_epa", "rushing_epa"],
    )
    ps = ps.filter(pl.col("attempts").fill_null(0) > 0).with_columns(
        (pl.col("attempts").fill_null(0) + pl.col("sacks_suffered").fill_null(0) + pl.col("carries").fill_null(0)).alias("plays"),
        (pl.col("passing_epa").fill_null(0) + pl.col("rushing_epa").fill_null(0)).alias("epa"),
    )
    return ps.select("game_id", "player_id", "team", "plays", "epa")

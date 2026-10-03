"""Game-level design matrix: home-minus-away rating differences + market + context."""
from __future__ import annotations

import numpy as np
import polars as pl

from features import CACHE, METRICS, UNPAIRED

# approximate home-stadium coordinates and UTC offset (standard time) by schedule team code
LOC = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.76, -84.40, -5), "BAL": (39.28, -76.62, -5), "BUF": (42.77, -78.79, -5),
    "CAR": (35.23, -80.85, -5), "CHI": (41.86, -87.62, -6), "CIN": (39.10, -84.52, -5), "CLE": (41.51, -81.70, -5),
    "DAL": (32.75, -97.09, -6), "DEN": (39.74, -105.02, -7), "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5), "JAX": (30.32, -81.64, -5), "KC": (39.05, -94.48, -6),
    "LA": (33.95, -118.34, -8), "LAC": (33.95, -118.34, -8), "SD": (32.78, -117.12, -8), "STL": (38.63, -90.19, -6),
    "LV": (36.09, -115.18, -8), "OAK": (37.75, -122.20, -8), "MIA": (25.96, -80.24, -5), "MIN": (44.97, -93.26, -6),
    "NE": (42.09, -71.26, -5), "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5), "NYJ": (40.81, -74.07, -5),
    "PHI": (39.90, -75.17, -5), "PIT": (40.45, -80.02, -5), "SEA": (47.60, -122.33, -8), "SF": (37.40, -121.97, -8),
    "TB": (27.98, -82.50, -5), "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
}


def _hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def ml_prob(ml: pl.Expr) -> pl.Expr:
    return pl.when(ml < 0).then(-ml / (-ml + 100)).otherwise(100 / (ml + 100))


RATING_COLS = ([f"{s}_{m}" for m, _, _ in METRICS for s in ("off", "def")] + [m for m, _ in UNPAIRED]
               + [f"{s}_{m}_adj" for m, _, a in METRICS if a for s in ("off", "def")] + ["margin_adj", "qb_epa"])


def build_games(tf: pl.DataFrame, sched: pl.DataFrame | None = None) -> pl.DataFrame:
    sched = sched if sched is not None else pl.read_parquet(CACHE / "schedules.parquet")
    h = tf.filter(pl.col("is_home")).select("game_id", *[pl.col(c).alias(f"h_{c}") for c in RATING_COLS + ["n_games", "qb_change", "qb_plays"]])
    a = tf.filter(~pl.col("is_home")).select("game_id", *[pl.col(c).alias(f"a_{c}") for c in RATING_COLS + ["n_games", "qb_change", "qb_plays"]])
    g = sched.join(h, on="game_id", how="left").join(a, on="game_id", how="left")
    diffs = [(pl.col(f"h_{c}") - pl.col(f"a_{c}")).alias(f"d_{c}") for c in RATING_COLS]
    # composite nets: (home off - home def allowed) - (away off - away def allowed)
    nets = []
    for m in ["epa_adj", "pass_epa_adj", "rush_epa_adj", "sr_adj", "early_epa_adj", "ppd_adj"]:
        base = m.replace("_adj", "")
        nets.append(((pl.col(f"h_off_{base}_adj") - pl.col(f"h_def_{base}_adj"))
                     - (pl.col(f"a_off_{base}_adj") - pl.col(f"a_def_{base}_adj"))).alias(f"net_{m}"))
    hp, ap = ml_prob(pl.col("home_moneyline")), ml_prob(pl.col("away_moneyline"))
    g = g.with_columns(
        *diffs, *nets,
        ml_p_home=pl.when(pl.col("home_moneyline").is_not_null() & pl.col("away_moneyline").is_not_null())
        .then(hp / (hp + ap)).otherwise(None),
        rest_diff=(pl.col("home_rest") - pl.col("away_rest")).cast(pl.Float64),
        home_short=(pl.col("home_rest") <= 5).cast(pl.Float64),
        away_short=(pl.col("away_rest") <= 5).cast(pl.Float64),
        home_bye=(pl.col("home_rest") >= 12).cast(pl.Float64),
        away_bye=(pl.col("away_rest") >= 12).cast(pl.Float64),
        div=pl.col("div_game").cast(pl.Float64),
        neutral=(pl.col("location") == "Neutral").cast(pl.Float64),
        indoor=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64),
        turf=(~pl.col("surface").fill_null("grass").str.contains("grass")).cast(pl.Float64),
        temp_f=pl.when(pl.col("roof").is_in(["dome", "closed"])).then(70).otherwise(pl.col("temp")).cast(pl.Float64),
        wind_f=pl.when(pl.col("roof").is_in(["dome", "closed"])).then(0).otherwise(pl.col("wind")).cast(pl.Float64),
        primetime=((pl.col("gametime") >= "19:00") | ~pl.col("weekday").is_in(["Sunday"])).cast(pl.Float64),
        playoff=(pl.col("game_type") != "REG").cast(pl.Float64),
        h_qb_change=pl.col("h_qb_change").cast(pl.Float64), a_qb_change=pl.col("a_qb_change").cast(pl.Float64),
        d_qb_plays=(pl.col("h_qb_plays").clip(0, 1500) - pl.col("a_qb_plays").clip(0, 1500)),
        min_games=pl.min_horizontal("h_n_games", "a_n_games").cast(pl.Float64),
        y_win=pl.when(pl.col("result") > 0).then(1.0).when(pl.col("result") < 0).then(0.0).otherwise(None),
        resid=(pl.col("result") - pl.col("spread_line")).cast(pl.Float64),
    )
    # travel: away team distance & timezone shift (neutral-site games: unknown venue -> 0)
    hl = np.array([LOC.get(t, (np.nan, np.nan, 0)) for t in g["home_team"].to_list()])
    al = np.array([LOC.get(t, (np.nan, np.nan, 0)) for t in g["away_team"].to_list()])
    dist = _hav(al[:, 0], al[:, 1], hl[:, 0], hl[:, 1])
    tz = hl[:, 2] - al[:, 2]
    neu = g["neutral"].to_numpy() == 1
    dist[neu] = 0.0
    tz[neu] = 0.0
    g = g.with_columns(travel_mi=pl.Series(dist / 1000.0), tz_shift=pl.Series(tz))
    # west-coast team playing early (1pm ET) in the east
    g = g.with_columns(west_early=((pl.col("tz_shift") >= 2) & (pl.col("gametime") < "14:00")).cast(pl.Float64))
    return g

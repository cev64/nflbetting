"""State-space team-strength model (forward Kalman filter), on point margins.

State x = [theta_1..theta_T, hfa]. Each game observes
    y = w*margin + (1-w)*epa_margin - qb_w*qb_adj_diff - rest - travel
      = theta_home - theta_away + hfa*home + e,   e ~ N(0, R)
Process: between game dates theta += N(0, q_day * dt) per team; hfa += N(0, q_hfa * dt).
Season boundary: theta <- rho * theta (regression to the mean), P_theta <- rho^2 P_theta + Q_season.
Forward-only: the prediction for a game uses the filtered state from strictly earlier games.

Optional off/def variant (`run_kalman_od`): state [o_1..o_T, d_1..d_T, hfa], observing each team's
points: pts_home - mu = o_h - d_a + hfa/2,  pts_away - mu = o_a - d_h - hfa/2  (mu = running league mean).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from data import load_games, team_game_epa
from elo import EPA_PTS_A, EPA_PTS_B, context_pts
from qb import FRANCHISE, QB_DEFAULT, compute_qb

# Tuned by coordinate descent on 2002-2011 (logloss of a logistic on rmargin); see NOTES.md
DEFAULT = dict(
    R=13.5 ** 2,         # observation noise variance (points^2)
    q_day=0.01,          # team-strength random-walk variance per day (in-season)
    q_hfa=0.0,          # HFA random-walk variance per day
    P0=36.0,             # initial prior variance of a team strength
    hfa0=2.5, hfa_P0=1.0,
    rho=0.7,             # season-to-season shrink of strengths
    Q_season=16.0,       # added variance at season boundary
    w_score=0.8,         # weight on actual margin vs EPA-implied margin
    qb_w=0.8, bye_pts=2.0, short_pts=0.0, travel_pts=0.75,
    playoff_mult=1.0,
    **QB_DEFAULT,
)


def _setup(p, qb):
    g = load_games()
    if qb is None:
        qb = compute_qb(p)
    g = g.join(qb, on="game_id", how="left").join(team_game_epa(), on="game_id", how="left").with_columns(
        pl.col("home_team").replace(FRANCHISE).alias("fh"), pl.col("away_team").replace(FRANCHISE).alias("fa"),
        pl.col("gameday").str.to_date().alias("date"),
    )
    teams = sorted(set(g["fh"].to_list()) | set(g["fa"].to_list()))
    return g, qb, {t: i for i, t in enumerate(teams)}


def run_kalman(params: dict | None = None, qb: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    p = {**DEFAULT, **(params or {})}
    g, qb, tix = _setup(p, qb)
    T = len(tix)
    n = T + 1
    x = np.zeros(n)
    x[T] = p["hfa0"]
    P = np.eye(n) * p["P0"]
    P[T, T] = p["hfa_P0"]
    qdiag = np.r_[np.full(T, p["q_day"]), p["q_hfa"]]
    cur_season, last_date = None, None
    cols = ["game_id", "season", "date", "fh", "fa", "result", "neutral", "playoff", "home_rest", "away_rest",
            "home_travel", "away_travel", "home_qb_adj", "away_qb_adj", "home_epa", "away_epa"]
    rows, facs = [], []
    for (gid, season, date, fh, fa, res, neutral, playoff, hr, ar, ht, at, qh, qa, he, ae) in g.select(cols).iter_rows():
        if cur_season is None:
            cur_season, last_date = season, date
        if season != cur_season:
            x[:T] = p["rho"] * (x[:T] - x[:T].mean())
            P[:T, :T] = p["rho"] ** 2 * P[:T, :T] + np.eye(T) * p["Q_season"]
            P[:T, T] *= p["rho"]
            P[T, :T] *= p["rho"]
            cur_season, last_date = season, date
        dt = (date - last_date).days
        if dt > 0:
            P[np.diag_indices(n)] += qdiag * min(dt, 14)
            last_date = date
        i, j = tix[fh], tix[fa]
        H = np.zeros(n)
        H[i], H[j] = 1.0, -1.0
        H[T] = 0.0 if neutral else 1.0
        _, rest, travel = context_pts(p, neutral, 0.0, hr, ar, ht, at)
        offset = rest + travel + p["qb_w"] * ((qh or 0.0) - (qa or 0.0))
        mu = H @ x
        var = H @ P @ H
        rmargin = (mu + offset) * (p["playoff_mult"] if playoff else 1.0)
        rows.append((gid, rmargin, float(np.sqrt(var + p["R"]))))
        facs.append((gid, x[i], x[j], np.sqrt(P[i, i]), np.sqrt(P[j, j]), x[T] * H[T], rest, travel))
        if res is None:
            continue
        y = res
        if p["w_score"] < 1 and he is not None and ae is not None:
            y = p["w_score"] * res + (1 - p["w_score"]) * (EPA_PTS_A * H[T] + EPA_PTS_B * (he - ae))
        y -= offset
        PH = P @ H
        S = var + p["R"]
        K = PH / S
        x = x + K * (y - mu)
        P = P - np.outer(K, PH)
    raw = pl.DataFrame(rows, schema=["game_id", "rmargin", "rsd"], orient="row")
    fac = pl.DataFrame(facs, schema=["game_id", "home_str", "away_str", "home_str_sd", "away_str_sd", "hfa",
                                     "rest_pts", "travel_pts"], orient="row").join(qb, on="game_id", how="left")
    return raw, fac


def run_kalman_od(params: dict | None = None, qb: pl.DataFrame | None = None) -> pl.DataFrame:
    """Offense/defense variant on team points. Returns game_id, rmargin_od, rtotal_od."""
    p = {**DEFAULT, **(params or {})}
    R = p["R"] / 2.0  # each team's score noise ~ half the margin variance
    g, qb, tix = _setup(p, qb)
    T = len(tix)
    n = 2 * T + 1
    x = np.zeros(n)
    x[-1] = p["hfa0"]
    P = np.eye(n) * p["P0"] / 2
    P[-1, -1] = p["hfa_P0"]
    qdiag = np.r_[np.full(2 * T, p["q_day"] / 2), p["q_hfa"]]
    mu_lg, cur_season, last_date = 21.0, None, None
    rows = []
    cols = ["game_id", "season", "date", "fh", "fa", "home_score", "away_score", "neutral", "home_rest",
            "away_rest", "home_travel", "away_travel", "home_qb_adj", "away_qb_adj"]
    for (gid, season, date, fh, fa, hs, as_, neutral, hr, ar, ht, at, qh, qa) in g.select(cols).iter_rows():
        if cur_season is None:
            cur_season, last_date = season, date
        if season != cur_season:
            for blk in (slice(0, T), slice(T, 2 * T)):
                x[blk] = p["rho"] * (x[blk] - x[blk].mean())
            P[:-1, :-1] = p["rho"] ** 2 * P[:-1, :-1] + np.eye(2 * T) * p["Q_season"] / 2
            cur_season, last_date = season, date
        dt = (date - last_date).days
        if dt > 0:
            P[np.diag_indices(n)] += qdiag * min(dt, 14)
            last_date = date
        i, j = tix[fh], tix[fa]
        h = 0.0 if neutral else 0.5
        Hh = np.zeros(n); Hh[i] = 1; Hh[T + j] = -1; Hh[-1] = h      # home points
        Ha = np.zeros(n); Ha[j] = 1; Ha[T + i] = -1; Ha[-1] = -h     # away points
        _, rest, travel = context_pts(p, neutral, 0.0, hr, ar, ht, at)
        c_h = mu_lg + (rest + travel) / 2 + p["qb_w"] * (qh or 0.0)
        c_a = mu_lg - (rest + travel) / 2 + p["qb_w"] * (qa or 0.0)
        ph, pa = c_h + Hh @ x, c_a + Ha @ x
        rows.append((gid, ph - pa, ph + pa))
        if hs is None:
            continue
        for H, yv, c in ((Hh, hs, c_h), (Ha, as_, c_a)):
            PH = P @ H
            S = H @ PH + R
            K = PH / S
            x = x + K * (yv - c - H @ x)
            P = P - np.outer(K, PH)
        mu_lg += 0.002 * ((hs + as_) / 2 - mu_lg)
    return pl.DataFrame(rows, schema=["game_id", "rmargin_od", "rtotal_od"], orient="row")

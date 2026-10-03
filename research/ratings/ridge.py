"""Opponent-adjusted ridge (Massey-style) power ratings, refit every week.

For each (season, week) slate, fit on played games from strictly earlier weeks:
    y_g = hfa * home_g + r_home - r_away + e_g,  weights w_g = 0.5^(age_weeks/half_life) * carry^(season gap)
with an L2 penalty lam * sum(r^2). Two targets:
    pts: point margin (capped)        epa: EPA/play margin * 46 (points scale)
The predicted margin for the slate is hfa*home + r_home - r_away (hfa = 0 at neutral sites).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from data import load_games, team_game_epa
from qb import FRANCHISE

# Tuned by coordinate descent on 2002-2011 (logloss of a logistic on r_pts); see NOTES.md
DEFAULT = dict(half_life=24.0, carry=0.7, lam=3.0, cap=24.0, lookback=3, epa_scale=46.0)


def run_ridge(params: dict | None = None) -> pl.DataFrame:
    p = {**DEFAULT, **(params or {})}
    g = load_games().join(team_game_epa(), on="game_id", how="left").with_columns(
        pl.col("home_team").replace(FRANCHISE).alias("fh"), pl.col("away_team").replace(FRANCHISE).alias("fa"),
        pl.col("gameday").str.to_date().alias("date"),
    )
    teams = sorted(set(g["fh"].to_list()) | set(g["fa"].to_list()))
    tix = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    hi = np.array([tix[t] for t in g["fh"]])
    ai = np.array([tix[t] for t in g["fa"]])
    home = (~g["neutral"].to_numpy()).astype(float)
    season = g["season"].to_numpy()
    week = g["week"].to_numpy()
    days = (g["date"] - g["date"].min()).dt.total_days().to_numpy().astype(float)
    played = g["played"].to_numpy()
    res = g["result"].to_numpy().astype(float)
    y_pts = np.clip(res, -p["cap"], p["cap"])
    y_epa = p["epa_scale"] * (g["home_epa"].to_numpy() - g["away_epa"].to_numpy())
    has_epa = ~np.isnan(y_epa)

    out = {k: np.full(g.height, np.nan) for k in ("r_pts", "r_epa", "hfa_pts", "hfa_epa")}
    rating_h = {k: np.full(g.height, np.nan) for k in ("pts", "epa")}
    rating_a = {k: np.full(g.height, np.nan) for k in ("pts", "epa")}
    keys = season * 100 + week
    for key in np.unique(keys):
        cur = keys == key
        s = key // 100
        t0 = days[cur].min()
        tr = played & (keys < key) & (season >= s - p["lookback"])
        idx = np.where(tr)[0]
        if len(idx) < 20:
            continue
        age_w = (t0 - days[idx]) / 7.0
        w = 0.5 ** (age_w / p["half_life"]) * p["carry"] ** (s - season[idx])
        X = np.zeros((len(idx), T + 1))
        X[np.arange(len(idx)), hi[idx]] = 1
        X[np.arange(len(idx)), ai[idx]] = -1
        X[:, T] = home[idx]
        pen = np.full(T + 1, p["lam"])
        pen[T] = 1e-6
        for name, y, ok in (("pts", y_pts, np.ones_like(has_epa)), ("epa", y_epa, has_epa)):
            m = ok[idx]
            Xm, ym, wm = X[m], y[idx][m], w[m]
            A = Xm.T @ (Xm * wm[:, None]) + np.diag(pen)
            beta = np.linalg.solve(A, Xm.T @ (wm * ym))
            r, h = beta[:T], beta[T]
            ci = np.where(cur)[0]
            out[f"r_{name}"][ci] = r[hi[ci]] - r[ai[ci]] + h * home[ci]
            out[f"hfa_{name}"][ci] = h * home[ci]
            rating_h[name][ci] = r[hi[ci]]
            rating_a[name][ci] = r[ai[ci]]
    return pl.DataFrame({
        "game_id": g["game_id"], **out,
        "home_rpts": rating_h["pts"], "away_rpts": rating_a["pts"],
        "home_repa": rating_h["epa"], "away_repa": rating_a["epa"],
    })

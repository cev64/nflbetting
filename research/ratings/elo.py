"""FiveThirtyEight-style NFL Elo with QB adjustment, dynamic HFA, rest, travel, playoff scaling.

Ratings are in Elo points; 25 Elo points ~= 1 point of spread. All updates are sequential in
kickoff order; a game's own result is only used after its prediction is recorded.
"""
from __future__ import annotations

import math

import polars as pl

from data import load_games, team_game_epa
from qb import FRANCHISE, QB_DEFAULT, compute_qb

ELO_PER_PT = 25.0
# EPA/play differential -> points; constants fit on 1999-2005 only (result ~ a + b * epa_diff)
EPA_PTS_A, EPA_PTS_B = 3.33, 46.0

# Tuned by coordinate descent on 2002-2011 (logloss of a logistic on rmargin); see NOTES.md
DEFAULT = dict(
    k=17.0,              # base K
    regress=0.33,        # fraction regressed to mean between seasons
    hfa0=2.2,            # initial home-field advantage, points
    hfa_lr=0.0,          # learning rate of the dynamic HFA (0 = fixed)
    bye_pts=2.0,         # points for coming off a bye (rest >= 13 days) vs opponent
    short_pts=0.0,       # points penalty for short rest (<= 5 days)
    travel_pts=0.5,      # points per 1000 miles of travel difference
    playoff_mult=1.0,    # elo-diff multiplier in playoffs
    mov=True,            # margin-of-victory multiplier
    qb_w=0.8,            # weight on the QB adjustment (0 disables)
    expansion_elo=1300.0,
    w_score=1.0,         # weight of actual margin vs EPA-implied margin in the update (1 = scores only)
    **QB_DEFAULT,
)


def context_pts(p: dict, neutral: bool, hfa: float, hr, ar, ht: float, at: float) -> tuple[float, float, float]:
    """(hfa, rest, travel) adjustments in points from the home team's perspective."""
    hfa_pts = 0.0 if neutral else hfa
    hr = 7 if hr is None else hr
    ar = 7 if ar is None else ar
    rest = p["bye_pts"] * ((hr >= 13) - (ar >= 13)) - p["short_pts"] * ((hr <= 5) - (ar <= 5))
    travel = p["travel_pts"] * (at - ht) / 1000.0
    return hfa_pts, rest, travel


def run_elo(params: dict | None = None, qb: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    p = {**DEFAULT, **(params or {})}
    g = load_games()
    if qb is None:
        qb = compute_qb(p)
    g = g.join(qb, on="game_id", how="left")
    epa = {r[0]: EPA_PTS_A + EPA_PTS_B * (r[1] - r[2]) for r in
           team_game_epa().select("game_id", "home_epa", "away_epa").iter_rows()}

    elo: dict[str, float] = {}
    team_season: dict[str, int] = {}
    hfa = p["hfa0"]
    cols = ["game_id", "season", "home_team", "away_team", "result", "neutral", "playoff", "home_rest",
            "away_rest", "home_travel", "away_travel", "home_qb_adj", "away_qb_adj"]
    rows, facs = [], []
    for (gid, season, h, a, res, neutral, playoff, hr, ar, ht, at, qh, qa) in g.select(cols).iter_rows():
        fh, fa = FRANCHISE.get(h, h), FRANCHISE.get(a, a)
        for t in (fh, fa):
            if t not in elo:
                elo[t] = p["expansion_elo"] if (season > 1999 or t == "CLE") else 1500.0
                team_season[t] = season
            elif team_season[t] < season:
                elo[t] = 1505 + (1 - p["regress"]) * (elo[t] - 1505)
                team_season[t] = season
        hfa_pts, rest_pts, travel_pts = context_pts(p, neutral, hfa, hr, ar, ht, at)
        qb_pts = p["qb_w"] * ((qh or 0.0) - (qa or 0.0))
        diff = elo[fh] - elo[fa] + ELO_PER_PT * (hfa_pts + rest_pts + travel_pts + qb_pts)
        if playoff:
            diff *= p["playoff_mult"]
        rmargin = diff / ELO_PER_PT
        rows.append((gid, rmargin))
        facs.append((gid, elo[fh], elo[fa], hfa_pts, rest_pts, travel_pts))
        if res is None:
            continue
        exp_h = 1 / (1 + 10 ** (-diff / 400))
        m_obs = res
        if p["w_score"] < 1 and gid in epa:
            m_obs = p["w_score"] * res + (1 - p["w_score"]) * epa[gid]
        act = 1.0 if m_obs > 0 else (0.0 if m_obs < 0 else 0.5)
        if p["mov"]:
            wdiff = diff if m_obs > 0 else -diff
            mult = math.log(abs(m_obs) + 1) * 2.2 / (wdiff * 0.001 + 2.2)
        else:
            mult = 1.0
        shift = p["k"] * mult * (act - exp_h)
        elo[fh] += shift
        elo[fa] -= shift
        if not neutral and p["hfa_lr"] > 0:
            hfa += p["hfa_lr"] * (res - rmargin)

    raw = pl.DataFrame(rows, schema=["game_id", "rmargin"], orient="row")
    fac = pl.DataFrame(facs, schema=["game_id", "home_elo", "away_elo", "hfa", "rest_pts", "travel_pts"], orient="row")
    fac = fac.join(qb, on="game_id", how="left")
    return raw, fac

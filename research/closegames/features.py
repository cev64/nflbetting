"""Close-game features, all strictly pre-game (as of the day before kickoff).

Every rating is an exponentially-decayed, shrunk average over events dated strictly before the game day
(`as_of_decayed`). Nothing from the game itself or any later game is used. Entities:
  * kicker   (gsis id of the kicker who kicked for the team in its most recent earlier game)
  * coach    (head coach named in the schedule for this game — known before kickoff)
  * team     (franchise code; OAK/SD/STL mapped to LV/LAC/LA)

Output: one row per schedule game with `h_<x>` / `a_<x>` columns plus game-level weather/venue columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DAY = np.timedelta64(1, "D")

PBP_COLS = ["game_id", "season", "home_team", "away_team", "posteam", "defteam", "down", "ydstogo",
            "yardline_100", "wp", "qtr", "half_seconds_remaining", "play_type", "epa", "score_differential",
            "field_goal_attempt", "extra_point_attempt", "kicker_player_id", "kicker_player_name",
            "kick_distance", "field_goal_result", "home_coach", "away_coach"]


def as_of_decayed(ev_key: np.ndarray, ev_t: np.ndarray, ev_v: np.ndarray, ev_n: np.ndarray,
                  q_key: np.ndarray, q_t: np.ndarray, half_life: float, K: float, prior: np.ndarray | float):
    """Decayed shrunk mean of v/n per key, using only events with time < query time.

    value = (sum w*v + K*prior) / (sum w*n + K), w = 0.5**((t_q - t_e)/half_life). Returns (value, eff_n).
    Times are in days (float). Uses running sums scaled by 2**(t/HL) (t offset to keep it finite).
    """
    prior = np.broadcast_to(np.asarray(prior, dtype=float), q_t.shape).astype(float)
    val = prior.copy()
    effn = np.zeros(len(q_t))
    t0 = min(ev_t.min() if len(ev_t) else 0.0, q_t.min() if len(q_t) else 0.0)
    ev = pd.DataFrame({"k": ev_key, "t": ev_t - t0, "v": ev_v, "n": ev_n}).sort_values(["k", "t"], kind="mergesort")
    q = pd.DataFrame({"k": q_key, "t": q_t - t0, "i": np.arange(len(q_t))})
    groups = {k: g for k, g in ev.groupby("k", sort=False)}
    for k, qg in q.groupby("k", sort=False):
        g = groups.get(k)
        if g is None:
            continue
        t = g["t"].values
        scale = np.exp2(t / half_life)
        cv = np.concatenate([[0.0], np.cumsum(g["v"].values * scale)])
        cn = np.concatenate([[0.0], np.cumsum(g["n"].values * scale)])
        idx = np.searchsorted(t, qg["t"].values, side="left")  # strictly earlier events only
        f = np.exp2(-qg["t"].values / half_life)
        sv, sn = cv[idx] * f, cn[idx] * f
        ii = qg["i"].values
        val[ii] = (sv + K * prior[ii]) / (sn + K)
        effn[ii] = sn
    return val, effn


def _days(x) -> np.ndarray:
    return (pd.to_datetime(x).values.astype("datetime64[D]") - np.datetime64("1990-01-01")) / DAY


def load_pbp(max_season: int | None) -> pd.DataFrame:
    q = pl.scan_parquet(CACHE / "pbp.parquet").select(PBP_COLS)
    if max_season is not None:
        q = q.filter(pl.col("season") <= max_season)
    return q.collect().to_pandas()


def team_games(sched: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (game, team side) with date, franchise, coach, opponent."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = sched[["game_id", "season", "week", "gameday", f"{side}_team", f"{opp}_team", f"{side}_coach",
                   "result"]].copy()
        x.columns = ["game_id", "season", "week", "gameday", "team", "opp", "coach", "result"]
        x["side"] = side
        x["margin"] = x["result"] if side == "home" else -x["result"]
        rows.append(x)
    tg = pd.concat(rows, ignore_index=True)
    tg["team"] = tg["team"].replace(FRANCHISE)
    tg["opp"] = tg["opp"].replace(FRANCHISE)
    tg["t"] = _days(tg["gameday"])
    return tg


# ------------------------------------------------------------------------------------------- kicker
def fg_expectation(fg: pd.DataFrame) -> np.ndarray:
    """P(make | distance) for each attempt from a logistic fit on the three previous seasons only."""
    exp = np.full(len(fg), np.nan)
    seasons = sorted(fg["season"].unique())
    for S in seasons:
        tr = fg[(fg["season"] < S) & (fg["season"] >= S - 3)]
        if len(tr) < 500:  # first season of data: warm-up, use own season (only feeds 1999 ratings)
            tr = fg[fg["season"] == S]
        X = np.c_[tr["dist"].values / 10, (tr["dist"].values / 10) ** 2]
        m = LogisticRegression(C=100.0, max_iter=1000).fit(X, tr["made"].values)
        te = (fg["season"] == S).values
        Xt = np.c_[fg.loc[te, "dist"].values / 10, (fg.loc[te, "dist"].values / 10) ** 2]
        exp[te] = m.predict_proba(Xt)[:, 1]
    return exp


def kicker_features(pbp: pd.DataFrame, sched_dates: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    fg = pbp[(pbp["field_goal_attempt"] == 1) & pbp["kicker_player_id"].notna() & pbp["kick_distance"].notna()].copy()
    fg = fg.merge(sched_dates, on="game_id", how="inner")
    fg["dist"] = fg["kick_distance"].astype(float)
    fg["made"] = (fg["field_goal_result"] == "made").astype(float)
    fg["exp"] = fg_expectation(fg)
    fg["oe"] = fg["made"] - fg["exp"]
    fg["long"] = (fg["dist"] >= 45).astype(float)
    # which kicker kicks for a team in a game: most kick attempts (FG + XP) by that team in that game
    k = pbp[((pbp["field_goal_attempt"] == 1) | (pbp["extra_point_attempt"] == 1)) & pbp["kicker_player_id"].notna()]
    k = k.groupby(["game_id", "posteam", "kicker_player_id"]).size().rename("n").reset_index()
    k = k.sort_values(["game_id", "posteam", "n"], ascending=[True, True, False]).drop_duplicates(["game_id", "posteam"])
    k = k.merge(sched_dates, on="game_id").rename(columns={"posteam": "team", "kicker_player_id": "kicker"})
    # kicker for each team-game = kicker of the team's most recent earlier game (pre-game knowable)
    tgk = tg[["game_id", "team", "t"]].sort_values("t")
    k = k.sort_values("t")
    m = pd.merge_asof(tgk, k[["team", "t", "kicker"]].rename(columns={"t": "tk"}), left_on="t", right_on="tk",
                      by="team", allow_exact_matches=False, direction="backward")
    # rating per kicker: FG made over expected per attempt (all) and on 45+ yd attempts
    q_key = m["kicker"].fillna("none").values
    v_all, n_all = as_of_decayed(fg["kicker_player_id"].values, fg["t"].values, fg["oe"].values,
                                 np.ones(len(fg)), q_key, m["t"].values, half_life=730.0, K=25.0, prior=0.0)
    lg = fg[fg["long"] == 1]
    v_long, n_long = as_of_decayed(lg["kicker_player_id"].values, lg["t"].values, lg["oe"].values,
                                   np.ones(len(lg)), q_key, m["t"].values, half_life=730.0, K=15.0, prior=0.0)
    names = (pbp[pbp["kicker_player_id"].notna()].drop_duplicates("kicker_player_id", keep="last")
             .set_index("kicker_player_id")["kicker_player_name"])
    out = m[["game_id", "team"]].copy()
    out["kicker_name"] = m["kicker"].map(names).values
    out["k_fgoe"] = v_all
    out["k_fgoe_long"] = v_long
    out["k_log_att"] = np.log1p(n_all)
    return out


# --------------------------------------------------------------------------------- coach / team
def coach_fourth_down(pbp: pd.DataFrame, sched_dates: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    p = pbp[(pbp["down"] == 4) & (pbp["ydstogo"] <= 3) & (pbp["yardline_100"] <= 60) & (pbp["qtr"] <= 4)
            & pbp["wp"].between(0.1, 0.9) & (pbp["half_seconds_remaining"] > 120)
            & pbp["play_type"].isin(["pass", "run", "punt", "field_goal"])].copy()
    p["go"] = p["play_type"].isin(["pass", "run"]).astype(float)
    p["coach"] = np.where(p["posteam"] == p["home_team"], p["home_coach"], p["away_coach"])
    p = p.merge(sched_dates, on="game_id")
    lg = p.groupby("season")["go"].mean()
    prior_by_season = {S: lg.get(S - 1, lg.iloc[0]) for S in range(1999, 2040)}
    prior = tg["season"].map(prior_by_season).values
    v, n = as_of_decayed(p["coach"].values, p["t"].values, p["go"].values, np.ones(len(p)),
                         tg["coach"].fillna("none").values, tg["t"].values, half_life=730.0, K=30.0, prior=prior)
    out = tg[["game_id", "team"]].copy()
    out["coach_go4"] = v - prior  # relative to last season's league rate (league trend removed)
    return out


def team_late_efficiency(pbp: pd.DataFrame, sched_dates: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    p = p.merge(sched_dates, on="game_id")
    out = tg[["game_id", "team"]].copy()
    sets = {
        "two_min": p["half_seconds_remaining"] <= 120,
        "late_close": (p["qtr"] >= 4) & (p["score_differential"].abs() <= 8),
    }
    for name, msk in sets.items():
        x = p[msk]
        for side, col in (("off", "posteam"), ("def", "defteam")):
            v, _ = as_of_decayed(x[col].values, x["t"].values, x["epa"].values, np.ones(len(x)),
                                 tg["team"].values, tg["t"].values, half_life=365.0, K=60.0, prior=0.0)
            out[f"{name}_{side}_epa"] = v
    out["late_net_epa"] = (out["two_min_off_epa"] - out["two_min_def_epa"] + out["late_close_off_epa"]
                           - out["late_close_def_epa"]) / 2
    return out


def close_record(tg: pd.DataFrame) -> pd.DataFrame:
    """One-score-game record (|margin| <= 8), decayed; widely believed to be noise — included to test it."""
    x = tg[tg["margin"].notna() & (tg["margin"].abs() <= 8)]
    w = np.where(x["margin"] > 0, 0.5, np.where(x["margin"] < 0, -0.5, 0.0))
    v, n = as_of_decayed(x["team"].values, x["t"].values, w, np.ones(len(x)), tg["team"].values, tg["t"].values,
                         half_life=365.0, K=8.0, prior=0.0)
    out = tg[["game_id", "team"]].copy()
    out["close_wpct"] = v + 0.5
    # coach career games (experience), any team
    y = tg[tg["margin"].notna()]
    cv, cn = as_of_decayed(y["coach"].fillna("none").values, y["t"].values, np.zeros(len(y)), np.ones(len(y)),
                           tg["coach"].fillna("none").values, tg["t"].values, half_life=1e9, K=1.0, prior=0.0)
    out["coach_log_games"] = np.log1p(cn)
    return out


def build(sched: pd.DataFrame, max_season: int | None = None) -> pd.DataFrame:
    """Game-level close-game feature table for every row of `sched` (which may include unplayed games)."""
    pbp = load_pbp(max_season)
    sd = sched[["game_id", "gameday"]].copy()
    sd["t"] = _days(sd["gameday"])
    sd = sd[["game_id", "t"]]
    tg = team_games(sched)
    parts = [kicker_features(pbp, sd, tg), coach_fourth_down(pbp, sd, tg), team_late_efficiency(pbp, sd, tg),
             close_record(tg)]
    f = tg[["game_id", "team", "side"]]
    for p in parts:
        f = f.merge(p, on=["game_id", "team"], how="left")
    cols = [c for c in f.columns if c not in ("game_id", "team", "side", "kicker_name")]
    txt = f.pivot(index="game_id", columns="side", values="kicker_name")
    txt.columns = [f"{c[0]}_kicker_name" for c in txt.columns]
    h = f[f["side"] == "home"].set_index("game_id")[cols].add_prefix("h_")
    a = f[f["side"] == "away"].set_index("game_id")[cols].add_prefix("a_")
    g = sched[["game_id", "roof", "temp", "wind"]].set_index("game_id").join(h).join(a).join(txt).reset_index()
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    g["wind_out"] = np.where(g["dome"] == 1, 0.0, g["wind"].fillna(8.0))
    g["cold_out"] = np.where(g["dome"] == 1, 0.0, (g["temp"].fillna(55.0) < 40).astype(float))
    for c in cols:
        g[f"d_{c}"] = g[f"h_{c}"] - g[f"a_{c}"]
    # kicker edge matters more outdoors in wind
    g["d_k_fgoe_x_wind"] = g["d_k_fgoe"] * g["wind_out"] / 10.0
    return g.drop(columns=["roof"])

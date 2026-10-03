"""Assemble web/data/picks.json from every model's walk-forward predictions.

Reads research/preds/<model>.csv (game_id, p_home, margin[, p_home_cover]) for each model in MODELS,
the per-game factor files (research/preds/<model>_factors.parquet) and the nflverse schedule, then writes
the picks sheet the website shows: the ensemble's straight-up and against-the-spread pick for every game
of the current season, each model's view, the data behind the pick, and the 2012-2025 backtest.

Usage: python backend/build_picks.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
PREDS = RESEARCH / "preds"
CACHE = RESEARCH / "cache"
OUT = ROOT / "web" / "data" / "picks.json"

BACKTEST = (2012, 2025)
SIGMA = 13.5  # points; maps a margin to a win probability for the market line
BEST_BET_EDGE = 2.0  # ATS picks with at least this many points of edge are flagged as best bets
ERAS = [(2007, 2011), (2012, 2017), (2018, 2025)]

# Every model shown on the site. The ensemble (stacked meta-model) comes first and makes the picks.
MODELS = [
    ("ensemble", "Ensemble", "meta",
     "Stacked meta-model: learns, from earlier seasons only, how much to trust the betting market and each model below."),
    ("market", "Market", "market",
     "The betting favorite from the closing spread (home team on a pick'em). The bar every model has to clear."),
    ("elo", "Elo", "ratings",
     "FiveThirtyEight-style Elo with margin of victory, home field, rest, travel and a starting-QB adjustment."),
    ("kalman", "Kalman ratings", "ratings",
     "Team strength tracked week to week by a Kalman filter on point margins and EPA, with off-season regression."),
    ("ratings_ridge", "Power ratings", "ratings",
     "Opponent-adjusted (Massey/ridge) ratings on point margin and EPA per play, decayed toward recent games."),
    ("logit_epa", "EPA logistic", "efficiency",
     "Regularized logistic regression on ~50 opponent-adjusted play-by-play efficiency stats (no betting line)."),
    ("gbm", "Gradient boosting", "efficiency",
     "LightGBM on the betting line plus the efficiency stats; a second booster predicts the result against the spread."),
    ("xgb_margin", "XGBoost margin", "efficiency",
     "XGBoost regression of the final margin on efficiency stats (no betting line)."),
    ("personnel", "QB & injuries", "personnel",
     "The betting line adjusted by starting-QB value, QB changes and snap-weighted injury load by position group."),
    ("personnel_nomkt", "Lineup only", "personnel",
     "Who is playing, priced without the betting line: QB ratings, injuries, rest."),
    ("situational", "Situational", "situational",
     "Market structure (moneyline vs spread, implied totals) plus rest, travel, body clock, revenge and other spots that held up out of sample."),
    ("mlp", "Neural net", "situational",
     "A small neural network (5 seeds averaged) predicting how far the result lands from what the market implies."),
    ("knn", "Similar games", "situational",
     "The 300 most similar past games: how often the home side won and covered."),
]

# The data behind each pick: (group, source model file, feature, better, fmt) -- better/fmt override the file's.
FACTORS = [
    ("Market", "situational", "ml_prob", None, None),
    ("Market", "situational", "implied_pts", None, None),
    ("Team strength", "elo", "elo", "high", "0"),
    ("Team strength", "elo", "kalman_strength", "high", "+0.0"),
    ("Team strength", "logit_epa_mkt", "margin_adj", None, None),
    ("Efficiency", "logit_epa_mkt", "off_epa_adj", None, "+0.000"),
    ("Efficiency", "logit_epa_mkt", "def_epa_adj", None, "+0.000"),
    ("Efficiency", "logit_epa_mkt", "off_pass_epa_adj", None, "+0.000"),
    ("Efficiency", "logit_epa_mkt", "off_rush_epa_adj", None, "+0.000"),
    ("Efficiency", "logit_epa_mkt", "off_sr_adj", None, None),
    ("Efficiency", "logit_epa_mkt", "def_sr_adj", None, None),
    ("Efficiency", "logit_epa_mkt", "off_ppd_adj", None, None),
    ("Efficiency", "logit_epa_mkt", "def_ppd_adj", None, None),
    ("Efficiency", "logit_epa_mkt", "def_pressure_rate", None, None),
    ("Efficiency", "logit_epa_mkt", "off_to_luck_adj", None, None),
    ("Efficiency", "logit_epa_mkt", "st_epa", None, None),
    ("QB & health", "personnel", "qb_epa", None, None),
    ("QB & health", "elo", "qb_adj", "high", "+0.0"),
    ("QB & health", "personnel", "inj_starters_out", None, None),
    ("QB & health", "personnel", "key_absences", None, None),
    ("Situation", "situational", "rest", None, None),
    ("Situation", "situational", "travel_km", None, None),
    ("Situation", "situational", "tz_shift", None, None),
]


def r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


def load_schedule() -> pl.DataFrame:
    return pl.read_parquet(CACHE / "schedules.parquet")


def load_preds(sched: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out = {}
    for mid, *_ in MODELS:
        if mid == "market":
            m = sched.filter(pl.col("spread_line").is_not_null()).select("game_id", "spread_line")
            out[mid] = m.select(
                "game_id",
                p_home=pl.Series(norm.cdf(m["spread_line"].to_numpy() / SIGMA)),
                margin=pl.col("spread_line").cast(pl.Float64),
            )
            continue
        f = PREDS / f"{os.environ.get('ENSEMBLE', 'ensemble') if mid == 'ensemble' else mid}.csv"
        if not f.exists():
            continue
        d = pl.read_csv(f)
        extra = ["p_home_cover", "ats_edge", "best_bet", "p_stack"] if mid == "ensemble" else ["p_home_cover"]
        cols = ["game_id", "p_home", "margin"] + [c for c in extra if c in d.columns]
        out[mid] = d.select(cols).unique("game_id", keep="last")
    return out


def ats_prob(row_p_cover, margin, spread):
    """P(home covers). Use the model's own cover probability when it has one."""
    if row_p_cover is not None and not (isinstance(row_p_cover, float) and math.isnan(row_p_cover)):
        return float(row_p_cover)
    return float(norm.cdf((margin - spread) / SIGMA))


# ---------------------------------------------------------------- backtest

def backtest(sched: pl.DataFrame, preds: dict[str, pl.DataFrame]) -> dict:
    lo, hi = BACKTEST
    games = sched.filter(
        pl.col("result").is_not_null() & pl.col("spread_line").is_not_null() & pl.col("season").is_between(lo, hi)
    ).select("game_id", "season", "result", "spread_line")

    def stats(d: pl.DataFrame):
        g = games.join(d, on="game_id", how="inner")
        su = g.filter(pl.col("result") != 0)
        hw = (su["result"] > 0).to_numpy()
        p = su["p_home"].clip(1e-4, 1 - 1e-4).to_numpy()
        pick = p >= 0.5
        ats = g.filter((pl.col("result") - pl.col("spread_line")) != 0)
        cover = ((ats["result"] - ats["spread_line"]) > 0).to_numpy()
        if "p_home_cover" in ats.columns:
            pc = ats["p_home_cover"].fill_null(0.5).to_numpy() >= 0.5
        else:
            pc = (ats["margin"] - ats["spread_line"]).to_numpy() > 0
        return su, g, dict(
            su_acc=r((pick == hw).mean()), ats_acc=r((pc == cover).mean()),
            logloss=r(-np.mean(hw * np.log(p) + (1 - hw) * np.log(1 - p))),
            brier=r(np.mean((p - hw) ** 2)), n=int(su.height), ats_n=int(ats.height),
        )

    leaderboard, by_season, by_season_ats = [], {}, {}
    for mid, *_ in MODELS:
        if mid not in preds:
            continue
        _, _, s = stats(preds[mid])
        if s["n"] < 0.9 * games.filter(pl.col("result") != 0).height:
            continue  # model doesn't cover the window
        leaderboard.append({"id": mid, **s})
        for season in range(lo, hi + 1):
            d = preds[mid].join(games.filter(pl.col("season") == season).select("game_id"), on="game_id")
            _, _, ss = stats(d)
            by_season.setdefault(season, {"season": season})[mid] = ss["su_acc"]
            by_season_ats.setdefault(season, {"season": season})[mid] = ss["ats_acc"]

    # Ensemble detail: calibration, accuracy by confidence, ATS by edge.
    ens = preds["ensemble"]
    su, g, _ = stats(ens)
    p = su["p_home"].to_numpy()
    hw = (su["result"] > 0).to_numpy()
    conf = np.maximum(p, 1 - p)
    right = (p >= 0.5) == hw
    calibration = []
    edges = np.linspace(0, 1, 11)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.sum() >= 15:
            calibration.append({"bin": r((a + b) / 2, 2), "pred": r(p[m].mean()), "actual": r(hw[m].mean()), "n": int(m.sum())})
    by_conf = [{"min_prob": t, "acc": r(right[conf >= t].mean()), "n": int((conf >= t).sum())}
               for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85) if (conf >= t).sum() > 0]
    ats = g.filter((pl.col("result") - pl.col("spread_line")) != 0)
    pc = np.array([ats_prob(a, m, s) for a, m, s in zip(
        ats["p_home_cover"].to_list() if "p_home_cover" in ats.columns else [None] * ats.height,
        ats["margin"].to_list(), ats["spread_line"].to_list())])
    cover = ((ats["result"] - ats["spread_line"]) > 0).to_numpy()
    edge = (np.abs(ats["ats_edge"].to_numpy()) if "ats_edge" in ats.columns
            else np.abs(norm.ppf(np.clip(pc, 1e-4, 1 - 1e-4)) * SIGMA))
    ats_right = (pc >= 0.5) == cover
    ats_by_edge = [{"min_edge": t, "acc": r(ats_right[edge >= t].mean()), "n": int((edge >= t).sum())}
                   for t in (0, 0.5, 1, 1.5, 2, 3, 4) if (edge >= t).sum() >= 20]
    # Best bets: ATS edge >= BEST_BET_EDGE, by era (2018-2025 is the meta-model's untouched holdout).
    best = []
    allg = sched.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null()).select(
        "game_id", "season", "result", "spread_line").join(ens, on="game_id")
    allg = allg.filter((pl.col("result") - pl.col("spread_line")) != 0)
    if "ats_edge" in allg.columns:
        right_all = ((allg["result"] - allg["spread_line"]) > 0).to_numpy() == (allg["p_home_cover"].to_numpy() >= 0.5)
        e_all = np.abs(allg["ats_edge"].to_numpy())
        ss = allg["season"].to_numpy()
        for a, b in ERAS + [(ERAS[1][0], ERAS[-1][1])]:
            m = (ss >= a) & (ss <= b) & (e_all >= BEST_BET_EDGE)
            if m.sum():
                best.append({"window": f"{a}-{b}", "acc": r(right_all[m].mean()), "n": int(m.sum()),
                             "w": int(right_all[m].sum()), "l": int((~right_all[m]).sum()),
                             "holdout": a == 2018})
    # Season-level summary of the market favourite for the headline.
    return {
        "window": f"{lo}-{hi}",
        "leaderboard": sorted(leaderboard, key=lambda x: -(x["su_acc"] or 0)),
        "by_season": [by_season[s] for s in sorted(by_season)],
        "by_season_ats": [by_season_ats[s] for s in sorted(by_season_ats)],
        "calibration": calibration,
        "by_confidence": by_conf,
        "ats_by_edge": ats_by_edge,
        "best_bets": best,
        "best_bet_edge": BEST_BET_EDGE,
    }


# ---------------------------------------------------------------- factors

def load_factors(game_ids: list[str]) -> dict[str, list[dict]]:
    files: dict[str, pl.DataFrame] = {}
    for _, src, *_ in FACTORS:
        if src not in files:
            f = PREDS / f"{src}_factors.parquet"
            files[src] = pl.read_parquet(f).filter(pl.col("game_id").is_in(game_ids)) if f.exists() else None
    out: dict[str, list[dict]] = {gid: [] for gid in game_ids}
    for group, src, feat, better, fmt in FACTORS:
        d = files.get(src)
        if d is None:
            continue
        for row in d.filter(pl.col("feature") == feat).iter_rows(named=True):
            item = {
                "key": feat, "group": group, "label": row["label"],
                "home": r(row["home"]), "away": r(row["away"]),
                "better": better or row.get("better") or "high",
                "fmt": fmt or row.get("fmt") or "0.00",
            }
            if row.get("home_text") or row.get("away_text"):
                item["home_text"], item["away_text"] = row.get("home_text"), row.get("away_text")
            if item["home"] is None and item["away"] is None and "home_text" not in item:
                continue
            out[row["game_id"]].append(item)
    return out


# ---------------------------------------------------------------- current season

def team_form(sched: pl.DataFrame, season: int) -> dict[tuple[str, str], dict]:
    """Each team's record, ATS record and point differential entering each of its games."""
    g = sched.filter(pl.col("season") == season).sort("gameday", "gametime")
    state: dict[str, dict] = {}
    out = {}
    for row in g.iter_rows(named=True):
        for side, opp in (("home", "away"), ("away", "home")):
            t = row[f"{side}_team"]
            s = state.setdefault(t, {"w": 0, "l": 0, "t": 0, "aw": 0, "al": 0, "ap": 0, "pd": 0})
            rec = f"{s['w']}-{s['l']}" + (f"-{s['t']}" if s["t"] else "")
            out[(row["game_id"], side)] = {"record": rec, "ats": f"{s['aw']}-{s['al']}" + (f"-{s['ap']}" if s["ap"] else ""), "pd": s["pd"]}
        if row["result"] is None:
            continue
        res, line = row["result"], row["spread_line"]
        for side, sign in (("home", 1), ("away", -1)):
            s = state[row[f"{side}_team"]]
            m = sign * res
            s["w" if m > 0 else "l" if m < 0 else "t"] += 1
            s["pd"] += m
            if line is not None:
                c = sign * (res - line)
                s["aw" if c > 0 else "al" if c < 0 else "ap"] += 1
    return out


def build() -> dict:
    sched = load_schedule()
    preds = load_preds(sched)
    if "ensemble" not in preds:
        raise SystemExit("research/preds/ensemble.csv is missing; run the meta-model first")
    season = int(sched.filter(pl.col("result").is_not_null())["season"].max())
    cur = sched.filter(pl.col("season") == season).sort("week", "gameday", "gametime")
    ens = preds["ensemble"]
    # Weeks with picks: every week of the season where the ensemble has at least one game.
    have = set(ens["game_id"].to_list())
    cur = cur.filter(pl.col("game_id").is_in(have) | pl.col("spread_line").is_null())
    weeks = sorted({w for w, gid in zip(cur["week"], cur["game_id"]) if gid in have})
    pending = cur.filter(pl.col("result").is_null() & pl.col("game_id").is_in(have))
    week = int(pending["week"].min()) if pending.height else (weeks[-1] if weeks else 1)
    cur = cur.filter(pl.col("week").is_in(weeks))

    form = team_form(sched, season)
    factors = load_factors(cur["game_id"].to_list())
    by_model = {m: {row["game_id"]: row for row in d.filter(pl.col("game_id").is_in(cur["game_id"].to_list())).iter_rows(named=True)}
                for m, d in preds.items()}

    games, rec = [], {"su": {"w": 0, "l": 0}, "ats": {"w": 0, "l": 0, "p": 0}}
    for row in cur.iter_rows(named=True):
        gid = row["game_id"]
        e = by_model["ensemble"].get(gid)
        if e is None:
            continue
        home, away, line = row["home_team"], row["away_team"], row["spread_line"]
        p_home = float(e["p_home"])
        pick_home = p_home >= 0.5
        p_mkt = float(norm.cdf(line / SIGMA)) if line is not None else None
        played = row["result"] is not None
        su_correct = None
        if played and row["result"] != 0:
            su_correct = (row["result"] > 0) == pick_home
            rec["su"]["w" if su_correct else "l"] += 1
        su = {"pick": home if pick_home else away, "prob": r(p_home if pick_home else 1 - p_home, 3),
              "margin": r(e["margin"], 1),
              "market_prob": r((p_mkt if pick_home else 1 - p_mkt), 3) if p_mkt is not None else None,
              "correct": su_correct,
              "stack_prob": r(e.get("p_stack") if pick_home else 1 - e.get("p_stack"), 3) if e.get("p_stack") is not None else None}
        ats = None
        if line is not None:
            pc = ats_prob(e.get("p_home_cover"), e["margin"], line)
            cover_home = pc >= 0.5
            raw = e.get("ats_edge")
            edge = abs(float(raw)) if raw is not None else abs(float(norm.ppf(min(max(pc, 1e-4), 1 - 1e-4))) * SIGMA)
            correct = None
            if played:
                diff = row["result"] - line
                if diff != 0:
                    correct = (diff > 0) == cover_home
                    rec["ats"]["w" if correct else "l"] += 1
                else:
                    rec["ats"]["p"] += 1
            ats = {"pick": home if cover_home else away, "line": r(-line if cover_home else line, 1),
                   "prob": r(pc if cover_home else 1 - pc, 3), "edge": r(edge, 1), "correct": correct,
                   "best_bet": edge >= BEST_BET_EDGE}
        models = {m: {"p_home": r(v[gid]["p_home"], 3), "margin": r(v[gid]["margin"], 1)}
                  for m, v in by_model.items() if gid in v}
        others = [v["p_home"] >= 0.5 for m, v in models.items() if m != "ensemble"]
        agree = (sum(1 for x in others if x == pick_home) / len(others)) if others else None
        games.append({
            "game_id": gid, "season": season, "week": row["week"], "type": row["game_type"],
            "date": row["gameday"], "time": row["gametime"], "home": home, "away": away,
            "stadium": row["stadium"], "roof": row["roof"], "surface": row["surface"],
            "temp": row["temp"], "wind": row["wind"],
            "spread_line": line, "total_line": row["total_line"],
            "home_ml": row["home_moneyline"], "away_ml": row["away_moneyline"],
            "home_qb": row["home_qb_name"], "away_qb": row["away_qb_name"],
            "home_coach": row["home_coach"], "away_coach": row["away_coach"],
            "home_score": row["home_score"], "away_score": row["away_score"],
            "su": su, "ats": ats, "models": models, "agreement": r(agree, 3),
            "factors": factors.get(gid, []),
            "team_form": {"home": form.get((gid, "home")), "away": form.get((gid, "away"))},
        })

    by_week = []
    for w in weeks:
        wg = [g for g in games if g["week"] == w]
        by_week.append({
            "week": w,
            "su_w": sum(g["su"]["correct"] is True for g in wg), "su_l": sum(g["su"]["correct"] is False for g in wg),
            "ats_w": sum(bool(g["ats"]) and g["ats"]["correct"] is True for g in wg),
            "ats_l": sum(bool(g["ats"]) and g["ats"]["correct"] is False for g in wg),
            "ats_p": sum(bool(g["ats"]) and g["ats"]["correct"] is None and g["home_score"] is not None for g in wg),
        })
    rec["by_week"] = by_week

    teams = {
        t["team_abbr"]: {"name": t["team_name"], "nick": t["team_nick"], "conf": t["team_conf"],
                         "division": t["team_division"], "color": t["team_color"], "color2": t["team_color2"],
                         "logo": t["team_logo_espn"]}
        for t in pl.read_parquet(CACHE / "teams.parquet").iter_rows(named=True)
    }
    meta_path = PREDS / "ensemble_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "week": week, "weeks": weeks,
        "models": [{"id": m, "name": n, "family": f, "description": d, **({"primary": True} if m == "ensemble" else {})}
                   for m, n, f, d in MODELS if m in preds],
        "games": games, "teams": teams, "record": rec,
        "backtest": backtest(sched, preds),
        "ensemble": meta,
    }


def clean(x):
    """Replace NaN/inf (e.g. from the meta-model's tables) with null so the JSON stays valid."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def main() -> None:
    data = clean(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    lb = {x["id"]: x for x in data["backtest"]["leaderboard"]}
    e, m = lb.get("ensemble", {}), lb.get("market", {})
    print(f"picks.json: {len(data['games'])} games, season {data['season']} week {data['week']}; "
          f"backtest {data['backtest']['window']} ensemble SU {e.get('su_acc')} ATS {e.get('ats_acc')} "
          f"(market SU {m.get('su_acc')}); live {data['record']['su']} ATS {data['record']['ats']}")


if __name__ == "__main__":
    main()

"""Spread "spot" detector built on the turnover game logs.

Two signals, both computed only from games played *before* the one being rated:

1. Power edge: a turnover-adjusted point-differential rating. Turnovers are
   largely luck and swing about 4 points each, so each team's point
   differential is adjusted by BETA points per turnover of margin, blended
   with last season (regressed) and shrunk toward average. The predicted
   margin versus the spread is the edge; |edge| >= POWER_EDGE flags a lean.

2. Turnover regression: when one team's season-to-date turnover diff/game
   exceeds its opponent's by TO_GAP or more (both with MIN_GAMES played),
   lean on the team with the *worse* turnover margin, betting that the market
   overrates the lucky side.

A "strong" spot is when both signals point at the same team.

Every signal is backtested over every season on disk and the records are
published next to the picks. Parameters are round, conventional values chosen
up front, not tuned to the backtest.
"""

from __future__ import annotations

from collections import defaultdict

BETA = 4.0  # points per turnover of margin
K_PREV = 8  # last season counts as this many games...
PREV_WEIGHT = 0.5  # ...regressed halfway to average
K_SHRINK = 4  # extra shrinkage toward league average (0)
HFA = 1.5  # home-field advantage in points
POWER_EDGE = 3.0  # points of edge needed to flag a lean
TO_GAP = 1.0  # turnover diff/game gap needed for the regression fade
MIN_GAMES = 3  # games of history needed for the turnover signal
BREAK_EVEN = 0.524  # win rate needed at -110

SIGNALS = {
    "power": f"Turnover-adjusted power rating differs from the spread by {POWER_EDGE:g}+ points",
    "to_fade": f"Fade the team whose turnover diff/game is {TO_GAP:g}+ better than its opponent's",
    "strong": "Both signals agree on the same team",
}


def to_diff(g: dict) -> int:
    return g["int_made"] + g["fum_rec"] - g["int_thrown"] - g["fum_lost"]


def season_averages(season: dict | None) -> dict[str, tuple[float, float]]:
    """Per-team (point diff/game, turnover diff/game) for a whole season."""
    if not season:
        return {}
    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0])
    for g in season["games"]:
        a = agg[g["team"]]
        a[0] += 1
        a[1] += g["pf"] - g["pa"]
        a[2] += to_diff(g)
    return {t: (a[1] / a[0], a[2] / a[0]) for t, a in agg.items()}


class SeasonState:
    """Answers "how did team X look entering date D" for one season."""

    def __init__(self, season: dict, prev_season: dict | None):
        self.prev = season_averages(prev_season)
        self.by_team: dict[str, list[dict]] = defaultdict(list)
        for g in season["games"]:
            self.by_team[g["team"]].append(g)
        for games in self.by_team.values():
            games.sort(key=lambda g: g["date"])

    def entering(self, team: str, date: str) -> dict:
        prior = [g for g in self.by_team.get(team, []) if g["date"] < date]
        n = len(prior)
        pd = sum(g["pf"] - g["pa"] for g in prior)
        to = sum(to_diff(g) for g in prior)
        ppd, pto = self.prev.get(team, (0.0, 0.0))
        rating = ((pd - BETA * to) + K_PREV * PREV_WEIGHT * (ppd - BETA * pto)) / (n + K_PREV + K_SHRINK)
        return {"n": n, "rating": rating, "pd_pg": pd / n if n else None, "to_pg": to / n if n else None}


def evaluate(state: SeasonState, home: str, away: str, date: str, spread_line: float | None) -> dict:
    """Signals for one game. spread_line is nflverse's: positive = home favored."""
    h, a = state.entering(home, date), state.entering(away, date)
    predicted = h["rating"] - a["rating"] + HFA  # home margin
    out = {
        "home": home,
        "away": away,
        "predicted_home_margin": round(predicted, 1),
        "fair_spread_line": round(predicted, 1),  # same convention as spread_line
        "home_games": h["n"],
        "away_games": a["n"],
        "home_to_pg": None if h["to_pg"] is None else round(h["to_pg"], 2),
        "away_to_pg": None if a["to_pg"] is None else round(a["to_pg"], 2),
        "spread_line": spread_line,
        "edge": None,
        "signals": {},
        "lean": None,
        "strength": 0,
    }
    if spread_line is None:
        return out

    # Positive edge = value on the home side.
    edge = predicted - spread_line
    out["edge"] = round(edge, 1)
    if abs(edge) >= POWER_EDGE:
        out["signals"]["power"] = home if edge > 0 else away

    if h["n"] >= MIN_GAMES and a["n"] >= MIN_GAMES:
        gap = h["to_pg"] - a["to_pg"]
        if abs(gap) >= TO_GAP:
            out["signals"]["to_fade"] = away if gap > 0 else home

    picks = set(out["signals"].values())
    if len(picks) == 1:
        out["lean"] = picks.pop()
        out["strength"] = len(out["signals"])
        if out["strength"] == 2:
            out["signals"]["strong"] = out["lean"]
    elif len(picks) == 2:
        out["lean"] = None  # signals disagree
    return out


def home_covered(g: dict) -> str | None:
    """ATS result for the home side of a home-team game record: W / L / P."""
    if g["line"] is None:
        return None
    m = g["pf"] - g["pa"] + g["line"]
    return "W" if m > 0 else "L" if m < 0 else "P"


def summarize(recs: dict[int, dict]) -> dict:
    w = sum(r["w"] for r in recs.values())
    l = sum(r["l"] for r in recs.values())
    p = sum(r["p"] for r in recs.values())
    return {"w": w, "l": l, "p": p, "pct": round(w / (w + l), 3) if w + l else None}


def backtest(seasons: dict[int, dict]) -> tuple[dict[str, dict[int, dict]], list[dict]]:
    """Per-signal, per-season records over completed games, plus every flagged game."""
    per: dict[str, dict[int, dict]] = {k: {} for k in SIGNALS}
    history: list[dict] = []
    for yr in sorted(seasons):
        state = SeasonState(seasons[yr], seasons.get(yr - 1))
        for g in seasons[yr]["games"]:
            if not g["home"] or g["line"] is None:
                continue
            ev = evaluate(state, g["team"], g["opp"], g["date"], -g["line"])
            if not ev["signals"]:
                continue
            res = home_covered(g)
            for sig, side in ev["signals"].items():
                rec = per[sig].setdefault(yr, {"w": 0, "l": 0, "p": 0})
                if res == "P":
                    rec["p"] += 1
                elif (res == "W") == (side == g["team"]):
                    rec["w"] += 1
                else:
                    rec["l"] += 1
            history.append(
                {
                    "season": yr,
                    "week": g["week"],
                    "date": g["date"],
                    "home": g["team"],
                    "away": g["opp"],
                    "spread_line": -g["line"],
                    "edge": ev["edge"],
                    "signals": ev["signals"],
                    "home_ats": res,
                }
            )
    return per, history


def model_report(seasons: dict[int, dict], current: int) -> dict:
    """Backtest (past seasons) + live record (current season) + this week's spots."""
    per, history = backtest(seasons)
    signals = {
        sig: {
            "rule": rule,
            "by_season": {yr: summarize({yr: r}) for yr, r in sorted(per[sig].items())},
            "backtest": summarize({yr: r for yr, r in per[sig].items() if yr < current}),
            "live": summarize({yr: r for yr, r in per[sig].items() if yr == current}),
        }
        for sig, rule in SIGNALS.items()
    }

    cur = seasons.get(current)
    spots = []
    if cur:
        state = SeasonState(cur, seasons.get(current - 1))
        for u in cur.get("upcoming", []):
            ev = evaluate(state, u["home"], u["away"], u["date"], u["spread_line"])
            ev.update(
                game_id=u["game_id"],
                week=u["week"],
                date=u["date"],
                time=u["time"],
                total_line=u["total_line"],
                home_score=u.get("home_score"),
                away_score=u.get("away_score"),
            )
            spots.append(ev)
    spots.sort(key=lambda s: (-s["strength"], -abs(s["edge"] or 0)))

    past_years = sorted(yr for yr in seasons if yr < current)
    return {
        "season": current,
        "backtest_seasons": [past_years[0], past_years[-1]] if past_years else None,
        "break_even": BREAK_EVEN,
        "params": {
            "beta": BETA,
            "k_prev": K_PREV,
            "prev_weight": PREV_WEIGHT,
            "k_shrink": K_SHRINK,
            "hfa": HFA,
            "power_edge": POWER_EDGE,
            "to_gap": TO_GAP,
            "min_games": MIN_GAMES,
        },
        "signals": signals,
        "live_history": [h for h in history if h["season"] == current],
        "spots": spots,
    }

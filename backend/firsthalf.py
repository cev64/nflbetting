"""First-half under 24.5: how likely is each game to be 24 or fewer at halftime?

Projected halftime total = a + b x the market's full-game total (nflverse's
consensus line), fitted by least squares. The chance of 24 or fewer comes from
the spread of past misses (actual minus projected), not from an assumed Poisson
or normal shape. Regular season only: playoff games are left out of the fit, the
backtest, the lookback and the weekly board.

A team first-half profile was tested as a second input: each team's first-half
points scored and allowed per game entering the game (blended with last season
and the league average like the other models), combined as (A scored + B allowed
- league) + (B scored + A allowed - league). It is mostly priced into the total
already (correlation about 0.6) and did not improve the walk-forward forecasts,
so it is reported as a comparison and shown as context, not used for the odds.

Walk-forward: every season is predicted by a fit on the seasons before it only,
so the 2022-2025 backtest never sees its own games. The current season uses all
past seasons. There are no free historical first-half odds, so this can test how
well-calibrated the probabilities are, but not whether betting them was
profitable; compare the fair price with your book's actual under 24.5 price.
"""

from __future__ import annotations

from collections import defaultdict

from kicks import fair_odds
from model import K_PREV, K_SHRINK, PREV_WEIGHT

LINE = 24.5  # the under wins at 24 or fewer
MIN_TRAIN = 200  # games needed before a season can be fitted and scored
CAL_BUCKETS = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def has_h1(season: dict | None) -> bool:
    return bool(season and season["games"] and "h1_pf" in season["games"][0])


def reg(g: dict) -> bool:
    """Regular-season game with a halftime score. Playoffs are left out of this bet entirely."""
    return g["type"] == "REG" and g.get("h1_pf") is not None


def home_games(season: dict) -> list[dict]:
    """One row per regular-season game (home side) with a halftime score."""
    return sorted(
        (g for g in season["games"] if g["home"] and reg(g)),
        key=lambda g: g["date"],
    )


class H1State:
    """First-half points scored / allowed per game for any team entering a date."""

    def __init__(self, season: dict, prev: dict | None):
        games = [g for g in season["games"] if reg(g)]
        self.by_team: dict[str, list[dict]] = defaultdict(list)
        for g in games:
            self.by_team[g["team"]].append(g)
        self.games = games
        prev_games = [g for g in prev["games"] if reg(g)] if has_h1(prev) else []
        self.prev_team: dict[str, tuple[float, float]] = {}
        agg: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0])
        for g in prev_games:
            a = agg[g["team"]]
            a[0] += 1
            a[1] += g["h1_pf"]
            a[2] += g["h1_pf"] + g["h1_pa"]
        for t, (n, pf, both) in agg.items():
            self.prev_team[t] = (pf / n, (both - pf) / n)
        self.prev_lg = sum(g["h1_pf"] for g in prev_games) / len(prev_games) if prev_games else None
        self.own_lg = sum(g["h1_pf"] for g in games) / len(games) if games else 11.5
        self._lg: dict[str, float] = {}

    def league(self, date: str) -> float:
        """League first-half points per team-game entering `date` (with last season as a prior)."""
        if date not in self._lg:
            before = [g["h1_pf"] for g in self.games if g["date"] < date]
            base = self.prev_lg if self.prev_lg is not None else self.own_lg
            self._lg[date] = (sum(before) + 64 * base) / (len(before) + 64)
        return self._lg[date]

    def team(self, team: str, date: str) -> dict:
        prior = [g for g in self.by_team.get(team, []) if g["date"] < date]
        n = len(prior)
        lg = self.league(date)
        prev = self.prev_team.get(team)
        w = K_PREV * PREV_WEIGHT if prev else 0
        pf, pa = sum(g["h1_pf"] for g in prior), sum(g["h1_pa"] for g in prior)
        return {
            "n": n,
            "scored": (pf + w * (prev[0] if prev else 0) + K_SHRINK * lg) / (n + w + K_SHRINK),
            "allowed": (pa + w * (prev[1] if prev else 0) + K_SHRINK * lg) / (n + w + K_SHRINK),
            "raw_scored": pf / n if n else None,
            "raw_allowed": pa / n if n else None,
            "under": sum(g["h1_pf"] + g["h1_pa"] <= LINE for g in prior),
        }


def profile(st: H1State, home: str, away: str, date: str) -> tuple[float, dict, dict]:
    lg = st.league(date)
    h, a = st.team(home, date), st.team(away, date)
    prof = (h["scored"] + a["allowed"] - lg) + (a["scored"] + h["allowed"] - lg)
    return prof, h, a


def fit(rows: list[dict], keys: tuple[str, ...] = ("total", "profile")) -> tuple[list[float], list[float]]:
    """Least squares h1 ~ a + b*keys[0] + ...; returns coefficients and sorted residuals."""
    X = [[1.0, *(r[k] for k in keys)] for r in rows]
    y = [r["h1"] for r in rows]
    k = len(X[0])
    # normal equations, solved by Gaussian elimination
    A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    v = [sum(x[i] * t for x, t in zip(X, y)) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            f = A[j][i] / A[i][i]
            A[j] = [aj - f * ai for aj, ai in zip(A[j], A[i])]
            v[j] -= f * v[i]
    beta = [0.0] * k
    for i in reversed(range(k)):
        beta[i] = (v[i] - sum(A[i][j] * beta[j] for j in range(i + 1, k))) / A[i][i]
    resid = [t - sum(b * xi for b, xi in zip(beta, x)) for x, t in zip(X, y)]
    return beta, sorted(resid)


def predict(beta: list[float], resid: list[float], *features: float) -> dict:
    proj = beta[0] + sum(b * f for b, f in zip(beta[1:], features))
    n = len(resid)
    p = sum(r <= LINE - proj for r in resid) / n
    q = lambda f: proj + resid[min(n - 1, int(f * n))]
    return {"proj": round(proj, 1), "p": round(p, 3), "lo": round(q(0.1), 1), "hi": round(q(0.9), 1)}


def season_rows(season: dict, prev: dict | None) -> list[dict]:
    st = H1State(season, prev)
    out = []
    for g in home_games(season):
        if g.get("total") is None:
            continue
        prof, _, _ = profile(st, g["team"], g["opp"], g["date"])
        out.append({"season": season["season"], "week": g["week"], "date": g["date"], "game_id": g["game_id"],
                    "home": g["team"], "away": g["opp"], "total": g["total"], "profile": prof,
                    "h1": g["h1_pf"] + g["h1_pa"], "h1_home": g["h1_pf"], "h1_away": g["h1_pa"], "type": g["type"]})
    return out


def brier(rows: list[dict], key: str) -> float | None:
    return round(sum((r[key] - (r["h1"] <= LINE)) ** 2 for r in rows) / len(rows), 4) if rows else None


def first_half_report(seasons: dict[int, dict], current: int) -> dict | None:
    usable = {yr: s for yr, s in seasons.items() if has_h1(s)}
    if current not in usable:
        return None
    rows = {yr: season_rows(usable[yr], usable.get(yr - 1)) for yr in sorted(usable)}

    # Walk-forward predictions: season Y uses a fit on seasons before Y.
    scored: list[dict] = []
    for yr in sorted(rows):
        train = [r for y, rs in rows.items() if y < yr for r in rs]
        if len(train) < MIN_TRAIN:
            continue
        beta, resid = fit(train)
        base = sum(r["h1"] <= LINE for r in train) / len(train)
        mbeta, mresid = fit(train, ("total",))  # the model used for the odds
        for r in rows[yr]:
            pm = predict(mbeta, mresid, r["total"])
            pp = predict(beta, resid, r["total"], r["profile"])
            scored.append({**r, **pm, "p_base": base, "p_profile": pp["p"]})

    def summary(rs: list[dict]) -> dict:
        n = len(rs)
        return {
            "n": n,
            "under": sum(r["h1"] <= LINE for r in rs),
            "rate": round(sum(r["h1"] <= LINE for r in rs) / n, 3) if n else None,
            "avg_p": round(sum(r["p"] for r in rs) / n, 3) if n else None,
            "brier": brier(rs, "p"),
            "brier_profile": brier(rs, "p_profile"),
            "brier_base": brier(rs, "p_base"),
        }

    bt = [r for r in scored if r["season"] < current]
    live = [r for r in scored if r["season"] == current]

    # Each week's three highest-probability games, as a simple selection rule.
    # Ties (same total, same chance) go to the earlier kickoff, then alphabetically.
    by_week: dict[tuple, list[dict]] = defaultdict(list)
    for r in scored:
        by_week[(r["season"], r["week"])].append(r)
    for wk in by_week.values():
        wk.sort(key=lambda r: (-r["p"], r["date"], r["home"]))
        for i, r in enumerate(wk):
            r["top3"] = i < 3

    def top3(rs: list[dict]) -> list[dict]:
        return [r for r in rs if r["top3"]]

    calib = {
        phase: [
            {"lo": lo, "hi": min(hi, 1.0), **summary([r for r in rs if lo <= r["p"] < hi])}
            for lo, hi in CAL_BUCKETS
        ]
        for phase, rs in (("backtest", bt), ("live", live))
    }

    # This week's games, fitted on every completed past season.
    train = [r for y, rs in rows.items() if y < current for r in rs]
    board = []
    if len(train) >= MIN_TRAIN:
        beta, resid = fit(train, ("total",))
        pbeta, _ = fit(train)
        cur = usable[current]
        st = H1State(cur, usable.get(current - 1))
        for u in (u for u in cur.get("upcoming", []) if u["type"] == "REG"):
            prof, h, a = profile(st, u["home"], u["away"], u["date"])
            row = {
                "game_id": u["game_id"], "week": u["week"], "date": u["date"], "time": u["time"],
                "home": u["home"], "away": u["away"], "total_line": u["total_line"], "spread_line": u["spread_line"],
                "home_score": u.get("home_score"), "away_score": u.get("away_score"),
                "teams": {
                    t: {"n": s["n"], "scored": s["raw_scored"], "allowed": s["raw_allowed"], "under": s["under"]}
                    for t, s in ((u["home"], h), (u["away"], a))
                },
                "proj": None, "p": None, "lo": None, "hi": None, "fair_odds": None,
            }
            if u["total_line"] is not None:
                pr = predict(beta, resid, u["total_line"])
                row.update(pr, fair_odds=fair_odds(pr["p"]))
            board.append(row)
        board.sort(key=lambda r: -(r["p"] if r["p"] is not None else -1))
        coef = {"intercept": round(beta[0], 2), "per_total_point": round(beta[1], 3),
                "profile_model_per_profile_point": round(pbeta[2], 3)}
    else:
        coef = None

    past = sorted(yr for yr in rows if yr < current)
    return {
        "season": current,
        "line": LINE,
        "train_seasons": [past[0], past[-1]] if past else None,
        "backtest_seasons": sorted({r["season"] for r in bt}),
        "coef": coef,
        "backtest": {"all": summary(bt), "top3": summary(top3(bt)), "p70": summary([r for r in bt if r["p"] >= 0.7])},
        "live": {"all": summary(live), "top3": summary(top3(live)), "p70": summary([r for r in live if r["p"] >= 0.7])},
        "calibration": calib,
        # Every predicted game, by season, for the week-by-week lookback (best chance first within a week).
        "history": {
            yr: [
                {k: r[k] for k in ("week", "type", "date", "game_id", "home", "away", "total", "proj", "lo", "hi", "p",
                                   "h1", "h1_home", "h1_away", "top3")}
                for wk in sorted({r["week"] for r in scored if r["season"] == yr})
                for r in by_week[(yr, wk)]
            ]
            for yr in sorted({r["season"] for r in scored})
        },
        "board": board,
    }

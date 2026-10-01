"""Kicking-prop matchups: drives that get deep but end in field goals.

The idea: field-goal props (FGs made over/under 1.5, kicker points) are priced
mostly off the game total, but how a team's points arrive matters. An offense
that moves the ball but stalls in the red zone, facing a defense that gives up
yards but tightens near its goal line ("bend, don't break"), should kick more
field goals than its team total alone suggests.

For each team in each game, using only games played before kickoff:

    trips inside the 40  = offense's trips/g + defense's trips allowed/g - league
    FGA per trip         = offense's FGA/trip + defense's FGA/trip allowed - league
    projected FGA        = trips x FGA per trip
    projected FGM        = projected FGA x the team's make rate (regressed)
    projected XP         = (offense TD/g + defense TD allowed/g - league) x league XP per TD

Each per-game rate blends this season, last season (regressed halfway) and the
league average, the same way the spread model does.

Field goals are noisy: over 2021-2025 a one-FG difference in this projection
went with only about 0.2 FGs of difference in the result. So the probabilities
use a calibrated projection, league average + CAL_SLOPE x (projection - league
average), with FGs made treated as Poisson for P(2+ made) and P(kicker points
8+). CAL_SLOPE was fitted on 2021-2025, so those probabilities are in-sample;
the flags below use the raw projection and are not affected by it.

Two things are backtested on every season on disk:

- *Stall x bend*: the offense gets inside the 40 at least as often as average
  but scores TDs on fewer red-zone trips than average, and the defense allows
  at least an average number of trips but fewer red-zone TDs than average
  (both teams with MIN_GAMES played).
- *Projects 2+ FGs*: the raw projection is 2.0+ field goals made.

Each is compared with how often 2+ FGs were actually made and with a
"total-only" baseline that scales the league FG rate by the team's implied
total, roughly how a book that only knows the total would price it.
"""

from __future__ import annotations

import math
from collections import defaultdict

from model import K_PREV, K_SHRINK, MIN_GAMES, PREV_WEIGHT

MAKE_PRIOR = 20  # league-average FG attempts added to a team's make rate
LG_PRIOR = 64  # team-games of last season's league average blended into this season's
FLAG_FGM = 2.0  # raw projected FGs made that flags a game
CAL_SLOPE = 0.2  # share of the projection's distance from average that shows up in results
KPTS_LINE = 7.5  # a common kicker-points line; we report P(8+)

# Per-game counts, for the offense (key) and for what the defense allowed (opp_key).
KEYS = ["t40", "rz", "rz_td", "td", "fga", "fgm", "xpm", "pts"]

SIGNALS = {
    "stall_bend": "Offense moves the ball but stalls in the red zone, and the defense bends but doesn't break",
    "proj2": f"Matchup projects {FLAG_FGM:g}+ field goals made",
}


def counts(g: dict, side: str) -> dict[str, float]:
    """The team's offensive (side='') or defensive (side='opp_') counts for one game."""
    out = {k: g.get(side + k, 0) or 0 for k in KEYS if k != "pts"}
    out["pts"] = g["pf"] if side == "" else g["pa"]
    return out


def means(games: list[dict], side: str) -> dict[str, float]:
    if not games:
        return {}
    tot = defaultdict(float)
    for g in games:
        for k, v in counts(g, side).items():
            tot[k] += v
    return {k: v / len(games) for k, v in tot.items()}


def poisson_at_least(lam: float, k: int) -> float:
    return 1 - sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k))


def kicker_points_over(fgm: float, xpm: float, line: float) -> float:
    """P(3*FGM + XPM > line), with FGM and XPM independent Poissons."""
    pf = [math.exp(-fgm) * fgm**i / math.factorial(i) for i in range(12)]
    px = [math.exp(-xpm) * xpm**i / math.factorial(i) for i in range(14)]
    return sum(a * b for i, a in enumerate(pf) for j, b in enumerate(px) if 3 * i + j > line)


def fair_odds(p: float) -> int | None:
    """American odds with no vig for probability p."""
    if p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def has_drive_data(season: dict) -> bool:
    return bool(season and season["games"] and "t40" in season["games"][0])


class KickState:
    """Answers "what did team X's offense / defense look like entering date D"."""

    def __init__(self, season: dict, prev_season: dict | None):
        games = [g for g in season["games"] if g["type"] == "REG"]
        self.games = sorted(games, key=lambda g: g["date"])
        self.by_team: dict[str, list[dict]] = defaultdict(list)
        for g in self.games:
            self.by_team[g["team"]].append(g)
        prev = [g for g in prev_season["games"] if g["type"] == "REG"] if has_drive_data(prev_season) else []
        self.prev_lg = means(prev, "") or None
        self.prev_team = {}
        for t in {g["team"] for g in prev}:
            tg = [g for g in prev if g["team"] == t]
            self.prev_team[t] = {"": means(tg, ""), "opp_": means(tg, "opp_")}
        # No earlier season on disk: fall back to this season's full average (league mean only).
        self.own_lg = means(self.games, "")
        self._lg_cache: dict[str, dict] = {}

    def league(self, date: str) -> dict[str, float]:
        if date not in self._lg_cache:
            before = [g for g in self.games if g["date"] < date]
            base = self.prev_lg or self.own_lg
            cur = means(before, "")
            n = len(before)
            self._lg_cache[date] = {k: (cur.get(k, 0) * n + base[k] * LG_PRIOR) / (n + LG_PRIOR) for k in KEYS}
        return self._lg_cache[date]

    def rates(self, team: str, date: str, side: str) -> dict[str, float]:
        prior = [g for g in self.by_team.get(team, []) if g["date"] < date]
        n = len(prior)
        lg = self.league(date)
        prev = self.prev_team.get(team, {}).get(side)
        w = K_PREV * PREV_WEIGHT if prev else 0
        tot = means(prior, side)
        out = {
            k: (tot.get(k, 0) * n + w * (prev[k] if prev else 0) + K_SHRINK * lg[k]) / (n + w + K_SHRINK)
            for k in KEYS
        }
        out["n"] = n
        out["fga_sum"] = sum(g.get(side + "fga", 0) for g in prior)
        out["fgm_sum"] = sum(g.get(side + "fgm", 0) for g in prior)
        return out


def project(st: KickState, team: str, opp: str, date: str, implied: float | None) -> dict:
    """Kicking projection for `team`'s offense against `opp`'s defense."""
    lg = st.league(date)
    o = st.rates(team, date, "")
    d = st.rates(opp, date, "opp_")
    lg_share = lg["fga"] / lg["t40"]
    lg_make = lg["fgm"] / lg["fga"]
    lg_rz_pct = lg["rz_td"] / lg["rz"]

    trips = max(0.5, o["t40"] + d["t40"] - lg["t40"])
    share = min(0.9, max(0.05, o["fga"] / o["t40"] + d["fga"] / d["t40"] - lg_share))
    fga = trips * share
    make = (o["fgm_sum"] + MAKE_PRIOR * lg_make) / (o["fga_sum"] + MAKE_PRIOR)
    raw_fgm = fga * make
    fgm = max(0.1, lg["fgm"] + CAL_SLOPE * (raw_fgm - lg["fgm"]))
    td = max(0.2, o["td"] + d["td"] - lg["td"])
    xpm = td * lg["xpm"] / lg["td"]

    off_rz = o["rz_td"] / o["rz"]
    def_rz = d["rz_td"] / d["rz"]
    stall = o["n"] >= MIN_GAMES and o["t40"] >= lg["t40"] and off_rz < lg_rz_pct
    bend = d["n"] >= MIN_GAMES and d["t40"] >= lg["t40"] and def_rz < lg_rz_pct

    p2 = poisson_at_least(fgm, 2)
    base = None
    if implied is not None:
        base = poisson_at_least(lg["fgm"] * implied / lg["pts"], 2)

    signals = []
    if stall and bend:
        signals.append("stall_bend")
    if raw_fgm >= FLAG_FGM:
        signals.append("proj2")
    return {
        "team": team,
        "opp": opp,
        "games": o["n"],
        "opp_games": d["n"],
        "implied": None if implied is None else round(implied, 1),
        "off_t40": round(o["t40"], 2),
        "def_t40": round(d["t40"], 2),
        "off_rz_pct": round(off_rz, 3),
        "def_rz_pct": round(def_rz, 3),
        "lg_t40": round(lg["t40"], 2),
        "lg_rz_pct": round(lg_rz_pct, 3),
        "stall": stall,
        "bend": bend,
        "trips": round(trips, 2),
        "fga_per_trip": round(share, 3),
        "make": round(make, 3),
        "fga": round(fga, 2),
        "fgm_raw": round(raw_fgm, 2),
        "fgm": round(fgm, 2),
        "lg_fgm": round(lg["fgm"], 2),
        "xpm": round(xpm, 2),
        "kpts": round(3 * fgm + xpm, 1),
        "p2": round(p2, 3),
        "p2_odds": fair_odds(p2),
        "p_kpts": round(kicker_points_over(fgm, xpm, KPTS_LINE), 3),
        "p2_total_only": None if base is None else round(base, 3),
        "signals": signals,
    }


def implied_total(total: float | None, line: float | None) -> float | None:
    """Team's implied points from the game total and its own spread (negative = favored)."""
    if total is None or line is None:
        return None
    return (total - line) / 2


def empty_rec() -> dict:
    return {"n": 0, "hit": 0, "p_model": 0.0, "p_total": 0.0, "n_total": 0, "fgm": 0}


def add(rec: dict, pr: dict, fgm: int) -> None:
    rec["n"] += 1
    rec["hit"] += fgm >= 2
    rec["fgm"] += fgm
    rec["p_model"] += pr["p2"]
    if pr["p2_total_only"] is not None:
        rec["n_total"] += 1
        rec["p_total"] += pr["p2_total_only"]


def finish(rec: dict) -> dict:
    n = rec["n"]
    return {
        "n": n,
        "hit": rec["hit"],
        "rate": round(rec["hit"] / n, 3) if n else None,
        "model": round(rec["p_model"] / n, 3) if n else None,
        "total_only": round(rec["p_total"] / rec["n_total"], 3) if rec["n_total"] else None,
        "fgm_pg": round(rec["fgm"] / n, 2) if n else None,
    }


# Calibration tiers on the raw projected FGs made.
CAL_BUCKETS = [(0, 1.4), (1.4, 1.7), (1.7, 2.0), (2.0, 99)]


def kicks_report(seasons: dict[int, dict], current: int) -> dict | None:
    """Backtest + live record for the kicking signals, and this week's board."""
    usable = {yr: s for yr, s in seasons.items() if has_drive_data(s)}
    if current not in usable:
        return None

    per = {sig: defaultdict(empty_rec) for sig in ["all", *SIGNALS]}
    cal = {"backtest": [empty_rec() for _ in CAL_BUCKETS], "live": [empty_rec() for _ in CAL_BUCKETS]}
    history = []
    for yr in sorted(usable):
        st = KickState(usable[yr], usable.get(yr - 1))
        for g in st.games:
            pr = project(st, g["team"], g["opp"], g["date"], implied_total(g.get("total"), g["line"]))
            fgm = g["fgm"]
            add(per["all"][yr], pr, fgm)
            for sig in pr["signals"]:
                add(per[sig][yr], pr, fgm)
            phase = "live" if yr == current else "backtest"
            for i, (lo, hi) in enumerate(CAL_BUCKETS):
                if lo <= pr["fgm_raw"] < hi:
                    add(cal[phase][i], pr, fgm)
            if yr == current and pr["signals"]:
                history.append({**pr, "week": g["week"], "date": g["date"], "home": g["home"], "fgm_actual": fgm,
                                "fga_actual": g["fga"], "kpts_actual": 3 * fgm + g["xpm"]})

    def roll(sig: dict, keep) -> dict:
        tot = empty_rec()
        for yr, r in sig.items():
            if keep(yr):
                for k in tot:
                    tot[k] += r[k]
        return finish(tot)

    signals = {
        sig: {
            "rule": SIGNALS.get(sig, "Every team-game"),
            "by_season": {yr: finish(r) for yr, r in sorted(per[sig].items())},
            "backtest": roll(per[sig], lambda yr: yr < current),
            "live": roll(per[sig], lambda yr: yr == current),
        }
        for sig in per
    }

    cur = usable[current]
    st = KickState(cur, usable.get(current - 1))
    board = []
    for u in cur.get("upcoming", []):
        for team, opp, home in ((u["away"], u["home"], False), (u["home"], u["away"], True)):
            line = None if u["spread_line"] is None else (-u["spread_line"] if home else u["spread_line"])
            pr = project(st, team, opp, u["date"], implied_total(u["total_line"], line))
            pr.update(game_id=u["game_id"], week=u["week"], date=u["date"], time=u["time"], home=home, line=line,
                      total_line=u["total_line"])
            board.append(pr)
    board.sort(key=lambda r: -r["fgm_raw"])

    past = sorted(yr for yr in usable if yr < current)
    return {
        "season": current,
        "backtest_seasons": [past[0], past[-1]] if past else None,
        "params": {
            "min_games": MIN_GAMES,
            "flag_fgm": FLAG_FGM,
            "cal_slope": CAL_SLOPE,
            "kpts_line": KPTS_LINE,
            "make_prior": MAKE_PRIOR,
        },
        "signals": signals,
        "calibration": {
            phase: [{"lo": lo, "hi": None if hi > 10 else hi, **finish(r)} for (lo, hi), r in zip(CAL_BUCKETS, recs)]
            for phase, recs in cal.items()
        },
        "live_history": sorted(history, key=lambda h: (h["date"], h["team"]), reverse=True),
        "board": board,
    }

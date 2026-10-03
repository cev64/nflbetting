"""Starting-QB value tracker shared by all rating models.

Each QB gets a shrunk, exponentially decayed EPA/dropback rating (relative to a drifting league
average). A team's "usual QB" baseline is an EMA of the value of the QBs who started its recent
games. The QB adjustment for a game = value(scheduled starter) - team baseline, in points.
Only games strictly before the current one feed the ratings.
"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from data import load_games, qb_game_stats

FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

QB_DEFAULT = dict(
    qb_n0=250.0,         # prior dropbacks for a QB's EPA rating
    qb_prior=-0.16,      # prior EPA/dropback (relative to league) for an unknown QB
    qb_decay=0.9,        # per-game decay of a QB's evidence
    qb_season_decay=0.5, # additional decay applied at each new season
    qb_ppg=35.0,         # dropbacks per game used to convert EPA/play -> points
    qb_team_alpha=0.05,  # EMA rate of team's "usual QB" baseline value
)


class QBTracker:
    """Shrunk, exponentially decayed EPA/dropback (relative to league average) per QB."""

    def __init__(self, p):
        self.p = p
        self.se = defaultdict(float)   # decayed sum of relative EPA
        self.sn = defaultdict(float)   # decayed sum of plays
        self.season = {}
        self.lg_epa = 0.0              # running league EPA/dropback for QBs
        self.lg_n = 0.0

    def _roll(self, qb, season):
        last = self.season.get(qb)
        if last is not None and season > last:
            d = self.p["qb_season_decay"] ** (season - last)
            self.se[qb] *= d
            self.sn[qb] *= d
        self.season[qb] = season

    def rating(self, qb, season) -> float:
        if qb is None:
            return self.p["qb_prior"]
        self._roll(qb, season)
        n0 = self.p["qb_n0"]
        return (self.se[qb] + self.p["qb_prior"] * n0) / (self.sn[qb] + n0)

    def value_pts(self, qb, season) -> float:
        return self.rating(qb, season) * self.p["qb_ppg"]

    def update(self, qb, season, plays, epa):
        if qb is None or plays <= 0:
            return
        self._roll(qb, season)
        lg = self.lg_epa / self.lg_n if self.lg_n > 0 else 0.0
        d = self.p["qb_decay"]
        self.se[qb] = self.se[qb] * d + (epa - lg * plays)
        self.sn[qb] = self.sn[qb] * d + plays
        # league average: decayed so it tracks era drift (~2 seasons memory)
        self.lg_epa = self.lg_epa * 0.998 + epa
        self.lg_n = self.lg_n * 0.998 + plays



def compute_qb(params: dict | None = None) -> pl.DataFrame:
    """Per game: home/away starter value (pts vs league-average QB) and adjustment vs team baseline."""
    p = {**QB_DEFAULT, **(params or {})}
    g = load_games()
    qbs = qb_game_stats()
    qb_lookup = {(r[0], r[1]): (r[2], r[3]) for r in qbs.select("game_id", "player_id", "plays", "epa").iter_rows()}
    qbt = QBTracker(p)
    team_qb: dict[str, float] = {}
    rows = []
    for gid, season, h, a, hq, aq, played in g.select("game_id", "season", "home_team", "away_team",
                                                       "home_qb_id", "away_qb_id", "played").iter_rows():
        fh, fa = FRANCHISE.get(h, h), FRANCHISE.get(a, a)
        hv, av = qbt.value_pts(hq, season), qbt.value_pts(aq, season)
        team_qb.setdefault(fh, hv)
        team_qb.setdefault(fa, av)
        rows.append((gid, hv, av, hv - team_qb[fh], av - team_qb[fa]))
        if not played:
            continue
        for team, qb in ((fh, hq), (fa, aq)):
            st = qb_lookup.get((gid, qb))
            if st is not None:
                qbt.update(qb, season, st[0], st[1])
            team_qb[team] += p["qb_team_alpha"] * (qbt.value_pts(qb, season) - team_qb[team])
    return pl.DataFrame(rows, schema=["game_id", "home_qb_val", "away_qb_val", "home_qb_adj", "away_qb_adj"], orient="row")

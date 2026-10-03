"""Market-structure + situational feature builder (one row per game, home/away perspective).

Every team-level feature is computed from strictly earlier games: per-team "post-game state" tables are
built from played games only and attached to each game with an as-of join on kickoff date with
allow_exact_matches=False, so a game never sees its own result (or any later result).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CACHE = Path(__file__).resolve().parents[1] / "cache"

# stadium_id -> (lat, lon, standard UTC offset)
STADIUMS = {
    "ATL00": (33.757, -84.401, -5), "ATL97": (33.755, -84.401, -5), "BAL00": (39.278, -76.623, -5),
    "BOS00": (42.091, -71.264, -5), "BOS99": (42.091, -71.264, -5), "BRG00": (30.412, -91.184, -6),
    "BUF00": (42.774, -78.787, -5), "BUF01": (43.641, -79.389, -5), "CAR00": (35.226, -80.853, -5),
    "CHI98": (41.862, -87.617, -6), "CHI99": (40.099, -88.236, -6), "CIN00": (39.095, -84.516, -5),
    "CIN99": (39.097, -84.508, -5), "CLE00": (41.506, -81.700, -5), "DAL00": (32.748, -97.093, -6),
    "DAL99": (32.840, -96.911, -6), "DEN00": (39.744, -105.020, -7), "DEN99": (39.746, -105.022, -7),
    "DET00": (42.340, -83.046, -5), "DET99": (42.646, -83.255, -5), "FRA00": (50.069, 8.645, 1),
    "GER00": (48.219, 11.625, 1), "GNB00": (44.501, -88.062, -6), "HOU00": (29.685, -95.411, -6),
    "IND00": (39.760, -86.164, -5), "IND99": (39.764, -86.163, -5), "JAX00": (30.324, -81.637, -5),
    "KAN00": (39.049, -94.484, -6), "LAX01": (33.953, -118.339, -8), "LAX97": (33.864, -118.261, -8),
    "LAX99": (34.014, -118.288, -8), "LON00": (51.556, -0.280, 0), "LON01": (51.456, -0.342, 0),
    "LON02": (51.604, -0.066, 0), "MAD01": (40.453, -3.688, 1), "MEL00": (-37.820, 144.983, 10),
    "MEX00": (19.303, -99.150, -6), "MIA00": (25.958, -80.239, -5), "MIN00": (44.974, -93.258, -6),
    "MIN01": (44.974, -93.258, -6), "MIN98": (44.977, -93.225, -6), "MUN01": (48.219, 11.625, 1),
    "NAS00": (36.166, -86.771, -6), "NOR00": (29.951, -90.081, -6), "NYC00": (40.814, -74.074, -5),
    "NYC01": (40.814, -74.074, -5), "OAK00": (37.752, -122.201, -8), "PAR00": (48.924, 2.360, 1),
    "PHI00": (39.901, -75.168, -5), "PHI99": (39.906, -75.171, -5), "PHO00": (33.528, -112.263, -7),
    "PHO99": (33.426, -111.933, -7), "PIT00": (40.447, -80.016, -5), "PIT99": (40.447, -80.016, -5),
    "RIO00": (-22.912, -43.230, -3), "SAN00": (29.417, -98.479, -6), "SAO00": (-23.545, -46.474, -3),
    "SDG00": (32.783, -117.120, -8), "SEA00": (47.595, -122.332, -8), "SEA98": (47.595, -122.332, -8),
    "SEA99": (47.650, -122.302, -8), "SFO00": (37.714, -122.386, -8), "SFO01": (37.403, -121.970, -8),
    "STL00": (38.633, -90.188, -6), "TAM00": (27.976, -82.503, -5), "VEG00": (36.091, -115.184, -8),
    "WAS00": (38.908, -76.864, -5),
}
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
SPREAD_SD = 13.5  # fixed constant used only to map a spread to a win probability (no fitting)


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _ml_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    ml = np.where(np.abs(ml) < 100, np.nan, ml)  # 0 / malformed quotes -> missing
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def load_schedule() -> pd.DataFrame:
    s = pd.read_parquet(CACHE / "schedules.parquet")
    s = s[s["season"] >= 1999].copy()
    s["date"] = pd.to_datetime(s["gameday"])
    gt = s["gametime"].fillna("13:00").str.split(":", expand=True).astype(float)
    s["kick_hour"] = gt[0] + gt[1] / 60.0  # Eastern time
    s["played"] = s["result"].notna()
    s["surface"] = s["surface"].fillna("").str.strip()
    s = s.sort_values(["date", "kick_hour", "game_id"]).reset_index(drop=True)
    return s


def _team_long(s: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "game_type", "date", "kick_hour", "weekday", "played", "div_game",
            "stadium_id", "roof", "surface", "location", "total_line", "referee"]
    h = s[cols].copy()
    h["team"], h["opp"], h["is_home"] = s["home_team"], s["away_team"], 1
    h["pf"], h["pa"] = s["home_score"], s["away_score"]
    h["spread_team"] = s["spread_line"]
    h["qb"], h["coach"], h["rest"] = s["home_qb_id"], s["home_coach"], s["home_rest"]
    a = s[cols].copy()
    a["team"], a["opp"], a["is_home"] = s["away_team"], s["home_team"], 0
    a["pf"], a["pa"] = s["away_score"], s["home_score"]
    a["spread_team"] = -s["spread_line"]
    a["qb"], a["coach"], a["rest"] = s["away_qb_id"], s["away_coach"], s["away_rest"]
    tg = pd.concat([h, a], ignore_index=True)
    tg["neutral"] = (tg["location"] == "Neutral").astype(int)
    tg["away_flag"] = ((tg["is_home"] == 0) | (tg["neutral"] == 1)).astype(int)  # not playing at own stadium
    tg["margin"] = tg["pf"] - tg["pa"]
    tg["win"] = np.where(tg["margin"] > 0, 1.0, np.where(tg["margin"] < 0, 0.0, 0.5))
    tg["cover_margin"] = tg["margin"] - tg["spread_team"]
    tg["cover"] = np.where(tg["cover_margin"] > 0, 1.0, np.where(tg["cover_margin"] < 0, 0.0, 0.5))
    tg["prime"] = ((tg["kick_hour"] >= 19) | tg["weekday"].isin(["Monday", "Thursday"])).astype(int)
    tg = tg.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    return tg


def _team_home_info(s: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season): primary home stadium -> coords/tz, roof type, surface."""
    hm = s[s["location"] == "Home"]
    g = hm.groupby(["home_team", "season"]).agg(
        stadium_id=("stadium_id", lambda x: x.mode().iloc[0]),
        roof=("roof", lambda x: x.mode().iloc[0] if x.notna().any() else "outdoors"),
        surface=("surface", lambda x: x.mode().iloc[0]),
    ).reset_index().rename(columns={"home_team": "team"})
    g["lat"] = g["stadium_id"].map(lambda k: STADIUMS[k][0])
    g["lon"] = g["stadium_id"].map(lambda k: STADIUMS[k][1])
    g["tz"] = g["stadium_id"].map(lambda k: STADIUMS[k][2])
    g["dome_team"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["turf_team"] = (~g["surface"].isin(["grass", "dessograss"])).astype(int)
    return g[["team", "season", "lat", "lon", "tz", "dome_team", "turf_team"]]


def _state_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Attach as-of (strictly before kickoff) team state to every team-game row."""
    p = tg[tg["played"]].copy().sort_values(["team", "date"])
    gs = p.groupby(["team", "season"], sort=False)
    p["st_season"] = p["season"]
    p["s_gp"] = gs.cumcount() + 1
    p["s_wins"] = gs["win"].cumsum()
    p["s_margin"] = gs["margin"].cumsum()
    p["s_cover_m"] = gs["cover_margin"].cumsum()
    p["s_covers"] = gs["cover"].cumsum()
    p["l3_margin"] = gs["margin"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    p["l3_cover_m"] = gs["cover_margin"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    p["l3_covers"] = gs["cover"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    # consecutive ATS covers/non-covers streak (signed)
    def streak(x):
        out, cur = [], 0
        for v in x:
            if v == 1:
                cur = cur + 1 if cur > 0 else 1
            elif v == 0:
                cur = cur - 1 if cur < 0 else -1
            else:
                cur = 0
            out.append(cur)
        return pd.Series(out, index=x.index)
    p["ats_streak"] = gs["cover"].transform(streak)
    p["last_margin"] = p["margin"]
    p["last_cover_m"] = p["cover_margin"]
    p["last_upset_win"] = ((p["spread_team"] <= -3) & (p["margin"] > 0)).astype(float)
    p["last_upset_loss"] = ((p["spread_team"] >= 3) & (p["margin"] < 0)).astype(float)
    p["last_blowout_win"] = (p["margin"] >= 21).astype(float)
    p["last_blowout_loss"] = (p["margin"] <= -21).astype(float)
    p["last_qb"] = p["qb"]
    p["last_prime"] = p["prime"]
    p["last_date"] = p["date"]
    keep = ["team", "date", "st_season", "s_gp", "s_wins", "s_margin", "s_cover_m", "s_covers", "l3_margin",
            "l3_cover_m", "l3_covers", "ats_streak", "last_margin", "last_cover_m", "last_upset_win", "last_upset_loss",
            "last_blowout_win", "last_blowout_loss", "last_qb", "last_prime", "last_date"]
    st = p[keep].sort_values("date")
    out = pd.merge_asof(tg.sort_values("date"), st, on="date", by="team", allow_exact_matches=False)
    same = out["st_season"] == out["season"]
    in_season = ["s_gp", "s_wins", "s_margin", "s_cover_m", "s_covers", "l3_margin", "l3_cover_m", "l3_covers",
                 "ats_streak", "last_margin", "last_cover_m", "last_upset_win", "last_upset_loss", "last_blowout_win",
                 "last_blowout_loss", "last_prime"]
    for c in in_season:
        out[c] = out[c].where(same, 0.0).fillna(0.0)
    out["qb_change"] = ((out["last_qb"].notna()) & (out["qb"].notna()) & (out["last_qb"] != out["qb"])).astype(int)
    n = out["s_gp"]
    out["win_pct"] = (out["s_wins"] + 0.5 * 2) / (n + 2)           # shrunk toward .500
    out["avg_margin"] = out["s_margin"] / (n + 2)
    out["avg_cover_m"] = out["s_cover_m"] / (n + 2)
    out["ats_pct"] = (out["s_covers"] + 0.5 * 2) / (n + 2)
    out["losses"] = n - out["s_wins"]

    # previous-season final strength
    fin = p.groupby(["team", "season"]).agg(ps_margin=("margin", "mean"), ps_cover_m=("cover_margin", "mean"),
                                            ps_wins=("win", "mean")).reset_index()
    fin["season"] += 1
    out = out.merge(fin, on=["team", "season"], how="left")
    for c in ["ps_margin", "ps_cover_m"]:
        out[c] = out[c].fillna(0.0)
    out["ps_wins"] = out["ps_wins"].fillna(0.5)

    # head-to-head: most recent previous meeting (played) within ~13 months
    hh = p[["team", "opp", "date", "season", "margin", "cover_margin"]].rename(
        columns={"season": "h2h_season", "margin": "h2h_margin", "cover_margin": "h2h_cover_m"})
    hh["h2h_date"] = hh["date"]
    out = pd.merge_asof(out.sort_values("date"), hh.sort_values("date"), on="date", by=["team", "opp"],
                        allow_exact_matches=False)
    recent = (out["date"] - out["h2h_date"]).dt.days < 400
    out["rematch"] = (recent & (out["h2h_season"] == out["season"])).astype(int)
    out["h2h_margin"] = out["h2h_margin"].where(recent, 0.0).fillna(0.0)
    out["revenge"] = (out["rematch"] & (out["h2h_margin"] < 0)).astype(int)
    out["h2h_cover_m"] = out["h2h_cover_m"].where(recent, 0.0).fillna(0.0)
    return out.drop(columns=["st_season", "h2h_season", "h2h_date", "last_qb"])


def _schedule_features(tg: pd.DataFrame, home_info: pd.DataFrame) -> pd.DataFrame:
    """Features derivable from the schedule itself (known in advance): travel, streaks, lookahead, coach, QB starts."""
    tg = tg.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    g = tg.groupby(["team", "season"], sort=False)
    # consecutive games away from home (incl. this one if away), prior road streak
    def run_len(x):
        out, cur = [], 0
        for v in x:
            cur = cur + 1 if v == 1 else 0
            out.append(cur)
        return pd.Series(out, index=x.index)
    tg["away_run"] = g["away_flag"].transform(run_len)
    tg["prev_away_run"] = g["away_run"].shift(1).fillna(0)
    # lookahead / sandwich
    tg["next_opp"] = g["opp"].shift(-1)
    tg["next_div"] = g["div_game"].shift(-1).fillna(0)
    tg["next_prime"] = g["prime"].shift(-1).fillna(0)
    tg["prev_div"] = g["div_game"].shift(1).fillna(0)
    # coach tenure (games with this team) and first-season flag
    tc = tg.groupby(["team", "coach"], sort=False)
    tg["coach_games"] = tc.cumcount()
    first_season = tc["season"].transform("min")
    tg["coach_year1"] = (tg["season"] == first_season).astype(int)
    tg.loc[tg["season"] == 1999, "coach_year1"] = 0  # unknown history before 1999
    tg["coach_career"] = tg.sort_values("date").groupby("coach").cumcount().reindex(tg.index)
    # QB experience (starts in data since 1999)
    tg["qb_starts"] = tg.sort_values("date").groupby("qb").cumcount().reindex(tg.index)
    tg.loc[tg["qb"].isna(), "qb_starts"] = 0
    # travel / body clock
    tg = tg.merge(home_info, on=["team", "season"], how="left")
    ven = tg["stadium_id"].map(STADIUMS)
    tg["v_lat"] = ven.map(lambda v: v[0])
    tg["v_lon"] = ven.map(lambda v: v[1])
    tg["v_tz"] = ven.map(lambda v: v[2])
    tg["travel_km"] = _haversine(tg["lat"], tg["lon"], tg["v_lat"], tg["v_lon"])
    tg["tz_shift"] = tg["v_tz"] - tg["tz"]  # + = travelled east
    # kickoff on the team's body clock (kick_hour is ET = UTC-5)
    tg["body_kick"] = tg["kick_hour"] + (tg["tz"] - (-5))
    tg["early_body"] = ((tg["body_kick"] <= 10.5) & (tg["travel_km"] > 500)).astype(int)
    tg["late_body"] = (tg["body_kick"] >= 21.5).astype(int)
    tg["intl"] = (tg["v_tz"] >= -3).astype(int)
    tg["prev_travel_km"] = tg.groupby(["team", "season"], sort=False)["travel_km"].shift(1).fillna(0)
    return tg


def _referee_features(s: pd.DataFrame) -> pd.DataFrame:
    """Referee (crew chief) home-win residual vs market and home-cover rate, from earlier weeks only."""
    d = s[["game_id", "season", "week", "date", "referee", "result", "spread_line", "location"]].copy()
    d["mprob"] = norm.cdf(d["spread_line"] / SPREAD_SD)
    d["hw"] = np.where(d["result"] > 0, 1.0, np.where(d["result"] < 0, 0.0, 0.5))
    d["hw_res"] = (d["hw"] - d["mprob"]).where(d["result"].notna())
    cm = d["result"] - d["spread_line"]
    d["hc_res"] = np.where(cm > 0, 0.5, np.where(cm < 0, -0.5, 0.0))
    d.loc[d["result"].isna() | (d["location"] == "Neutral"), ["hw_res", "hc_res"]] = np.nan
    d["n1"] = d["hw_res"].notna().astype(float)
    d = d.sort_values(["date", "game_id"])
    out = []
    for ref, grp in d[d["referee"].notna()].groupby("referee", sort=False):
        grp = grp.copy()
        for c in ["hw_res", "hc_res", "n1"]:
            grp["c_" + c] = grp[c].fillna(0).cumsum().shift(1).fillna(0)
        out.append(grp)
    r = pd.concat(out)
    K = 60.0
    r["ref_hw_res"] = r["c_hw_res"] / (r["c_n1"] + K)
    r["ref_hc_res"] = r["c_hc_res"] / (r["c_n1"] + K)
    r["ref_games"] = r["c_n1"]
    return d[["game_id"]].merge(r[["game_id", "ref_hw_res", "ref_hc_res", "ref_games"]], on="game_id", how="left").fillna(0.0)


TEAM_FEATS = [
    "rest", "away_run", "prev_away_run", "next_div", "next_prime", "prev_div", "coach_games", "coach_year1",
    "coach_career", "qb_starts", "qb_change", "travel_km", "tz_shift", "body_kick", "early_body", "late_body", "intl",
    "prev_travel_km", "dome_team", "turf_team", "tz", "s_gp", "win_pct", "avg_margin", "avg_cover_m", "ats_pct", "losses",
    "l3_margin", "l3_cover_m", "l3_covers", "ats_streak", "last_margin", "last_cover_m", "last_upset_win",
    "last_upset_loss", "last_blowout_win", "last_blowout_loss", "last_prime", "ps_margin", "ps_cover_m", "ps_wins",
    "rematch", "revenge", "h2h_margin", "h2h_cover_m", "days_since",
]


def build(sched: pd.DataFrame | None = None) -> pd.DataFrame:
    s = load_schedule() if sched is None else sched
    tg = _team_long(s)
    home_info = _team_home_info(s)
    tg = _schedule_features(tg, home_info)
    tg = _state_features(tg)
    tg["days_since"] = ((tg["date"] - tg["last_date"]).dt.days).fillna(30).clip(upper=30)
    tg.loc[tg["s_gp"] == 0, "days_since"] = 30
    tg["rest"] = tg["rest"].fillna(7).clip(upper=21)

    H = tg[tg["is_home"] == 1].set_index("game_id")[TEAM_FEATS].add_prefix("h_")
    A = tg[tg["is_home"] == 0].set_index("game_id")[TEAM_FEATS].add_prefix("a_")
    g = s.set_index("game_id").join(H).join(A).reset_index()
    g = g.merge(_referee_features(s), on="game_id", how="left").copy()

    # ---- market structure ----
    sp = g["spread_line"]
    g["spread_prob"] = norm.cdf(sp / SPREAD_SD)
    ph, pa = _ml_to_prob(g["home_moneyline"]), _ml_to_prob(g["away_moneyline"])
    g["ml_prob"] = ph / (ph + pa)
    g["has_ml"] = g["ml_prob"].notna().astype(int)
    g["ml_prob"] = g["ml_prob"].fillna(g["spread_prob"])
    g["mkt_prob"] = g["ml_prob"].clip(0.02, 0.98)
    g["mkt_logit"] = np.log(g["mkt_prob"] / (1 - g["mkt_prob"]))
    g["ml_vs_spread"] = g["ml_prob"] - g["spread_prob"]
    hs, as_ = _ml_to_prob(g["home_spread_odds"]), _ml_to_prob(g["away_spread_odds"])
    g["spread_juice"] = (hs / (hs + as_) - 0.5)  # >0: home side priced more expensive at the spread
    g["spread_juice"] = g["spread_juice"].fillna(0.0)
    g["vig"] = (ph + pa - 1).astype(float)
    g["vig"] = g["vig"].fillna(g["vig"].median())
    tot = g["total_line"].fillna(44.0)
    g["total_line_f"] = tot
    g["home_implied"] = tot / 2 + sp / 2
    g["away_implied"] = tot / 2 - sp / 2
    asp = sp.abs()
    g["abs_spread"] = asp
    fav_sign = np.sign(sp)  # +1 home favourite
    g["key3"] = (asp == 3).astype(int)
    g["key7"] = (asp == 7).astype(int)
    g["hook_fav_35"] = fav_sign * (asp == 3.5)       # favourite laying the hook over 3 (signed: + = home fav)
    g["hook_fav_25"] = fav_sign * (asp == 2.5)
    g["hook_fav_75"] = fav_sign * (asp == 7.5)
    g["hook_fav_65"] = fav_sign * (asp == 6.5)
    g["home_dog"] = (sp < 0).astype(int)
    g["road_fav_big"] = (sp <= -7).astype(int)
    g["big_fav_sign"] = fav_sign * (asp >= 10)
    g["pickem"] = (asp <= 1).astype(int)
    g["low_total"] = (tot <= 40).astype(int)
    g["high_total"] = (tot >= 50).astype(int)
    g["dog_total_interact"] = -fav_sign * (asp >= 7) * (tot <= 41)  # big dogs in low totals (signed toward home)

    # ---- game situational ----
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["playoff"] = (g["game_type"] != "REG").astype(int)
    g["prime"] = ((g["kick_hour"] >= 19) | g["weekday"].isin(["Monday", "Thursday"])).astype(int)
    g["thursday"] = (g["weekday"] == "Thursday").astype(int)
    g["monday"] = (g["weekday"] == "Monday").astype(int)
    g["div_game"] = g["div_game"].fillna(0).astype(int)
    outdoor = ~g["roof"].isin(["dome", "closed"])
    g["outdoor"] = outdoor.astype(int)
    g["temp_f"] = g["temp"].astype(float).where(outdoor, 70.0).fillna(pd.Series(np.where(outdoor, 55.0, 70.0), index=g.index))
    g["wind_f"] = g["wind"].astype(float).where(outdoor, 0.0).fillna(pd.Series(np.where(outdoor, 8.0, 0.0), index=g.index))
    g["cold"] = (g["temp_f"] <= 35).astype(int)
    g["windy"] = (g["wind_f"] >= 15).astype(int)
    g["a_dome_in_cold"] = (g["a_dome_team"].eq(1) & (g["temp_f"] <= 40)).astype(int)
    g["h_dome_in_cold"] = (g["h_dome_team"].eq(1) & (g["temp_f"] <= 40) & (g["neutral"] == 1)).astype(int)
    g["grass"] = g["surface"].isin(["grass", "dessograss"]).astype(int)
    g["a_turf_on_grass"] = (g["a_turf_team"].eq(1) & (g["grass"] == 1)).astype(int)
    g["week_f"] = np.where(g["game_type"] == "REG", g["week"], 19)
    g["late"] = ((g["game_type"] == "REG") & (g["week"] >= 13)).astype(int)
    for side in ("h", "a"):
        # out of contention proxy: late regular season with a losing record by 3+ games
        g[f"{side}_out"] = ((g["late"] == 1) & (g[f"{side}_losses"] - (g[f"{side}_s_gp"] - g[f"{side}_losses"]) >= 3)).astype(int)
        g[f"{side}_short"] = (g[f"{side}_rest"] <= 5).astype(int)
        g[f"{side}_bye"] = (g[f"{side}_rest"] >= 12).astype(int)
        g[f"{side}_new_qb"] = ((g[f"{side}_qb_starts"] < 8)).astype(int)
    g["rest_diff"] = (g["h_rest"] - g["a_rest"]).clip(-10, 10)
    g["bye_diff"] = g["h_bye"] - g["a_bye"]
    g["short_diff"] = g["h_short"] - g["a_short"]
    g["out_diff"] = g["h_out"] - g["a_out"]
    g["travel_diff"] = (g["a_travel_km"] - g["h_travel_km"]) / 1000.0
    g["tz_diff"] = g["a_tz_shift"].abs() - g["h_tz_shift"].abs()
    g["early_body_diff"] = g["a_early_body"] - g["h_early_body"]
    # West-coast home team hosting an Eastern team at night (classic body-clock edge for the home side)
    g["west_night_home"] = ((g["h_tz"] <= -7) & (g["a_tz"] >= -5) & (g["kick_hour"] >= 19)).astype(int)
    g["road_trip_diff"] = g["a_away_run"] - g["h_away_run"]
    g["home_after_trip"] = (g["h_prev_away_run"] >= 2).astype(int)
    g["a_3rd_road"] = (g["a_away_run"] >= 3).astype(int)
    g["coach_exp_diff"] = (np.log1p(g["h_coach_career"]) - np.log1p(g["a_coach_career"]))
    g["coach_year1_diff"] = g["h_coach_year1"] - g["a_coach_year1"]
    g["qb_exp_diff"] = np.log1p(g["h_qb_starts"]) - np.log1p(g["a_qb_starts"])
    g["qb_change_diff"] = g["h_qb_change"] - g["a_qb_change"]
    g["new_qb_diff"] = g["h_new_qb"] - g["a_new_qb"]
    for f in ["win_pct", "avg_margin", "avg_cover_m", "ats_pct", "l3_margin", "l3_cover_m", "l3_covers", "ats_streak",
              "last_margin", "last_cover_m", "last_upset_win", "last_upset_loss", "last_blowout_win",
              "last_blowout_loss", "ps_margin", "ps_cover_m", "ps_wins", "revenge", "h2h_margin", "next_div",
              "next_prime", "last_prime", "prev_div", "prev_travel_km"]:
        g[f + "_diff"] = g["h_" + f] - g["a_" + f]
    g["prev_travel_diff"] = g["prev_travel_km_diff"] / 1000.0
    # lookahead: favourite with a division / primetime game next week (signed toward home)
    g["lookahead"] = (g["h_next_div"] * (sp >= 3) - g["a_next_div"] * (sp <= -3)).astype(float)
    # letdown: favourite coming off an upset win / blowout win (signed toward home)
    g["letdown"] = ((g["h_last_upset_win"] + g["h_last_blowout_win"]).clip(0, 1) * (sp >= 3)
                    - (g["a_last_upset_win"] + g["a_last_blowout_win"]).clip(0, 1) * (sp <= -3)).astype(float)
    # bounce-back: team off a blowout loss (signed toward home)
    g["bounce"] = (g["h_last_blowout_loss"] - g["a_last_blowout_loss"]).astype(float)
    g["rematch"] = g["h_rematch"]
    g["div_rematch_dog"] = (g["rematch"] * g["div_game"] * -np.sign(sp)).astype(float)  # + = home is the dog
    g["home_dog_div"] = (g["home_dog"] * g["div_game"]).astype(int)
    g["home_dog_bye"] = (g["home_dog"] * g["h_bye"]).astype(int)
    g["early_season"] = ((g["game_type"] == "REG") & (g["week"] <= 4)).astype(int)
    g["ps_x_early"] = g["ps_margin_diff"] * g["early_season"]

    # targets
    g["home_win"] = np.where(g["result"] > 0, 1.0, np.where(g["result"] < 0, 0.0, np.nan))
    resid = g["result"] - g["spread_line"]
    g["resid"] = resid
    g["home_cover"] = np.where(resid > 0, 1.0, np.where(resid < 0, 0.0, np.nan))
    return g


MARKET_FEATS = ["mkt_logit", "ml_vs_spread", "spread_juice", "vig", "has_ml", "total_line_f", "home_implied",
                "away_implied", "key3", "key7", "hook_fav_35", "hook_fav_25", "hook_fav_75", "hook_fav_65",
                "home_dog", "road_fav_big", "big_fav_sign", "pickem", "low_total", "high_total", "dog_total_interact"]
SITU_FEATS = [
    "neutral", "playoff", "prime", "thursday", "monday", "div_game", "outdoor", "temp_f", "wind_f", "cold", "windy",
    "a_dome_in_cold", "grass", "a_turf_on_grass", "week_f", "late", "rest_diff", "bye_diff", "short_diff", "out_diff",
    "travel_diff", "tz_diff", "early_body_diff", "west_night_home", "road_trip_diff", "home_after_trip", "a_3rd_road",
    "coach_exp_diff", "coach_year1_diff", "qb_exp_diff", "qb_change_diff", "new_qb_diff",
    "ref_hw_res", "ref_hc_res", "lookahead", "letdown", "bounce", "rematch", "div_rematch_dog", "home_dog_div",
    "home_dog_bye", "early_season", "ps_x_early", "intl_any",
] + [f + "_diff" for f in ["win_pct", "avg_margin", "avg_cover_m", "ats_pct", "l3_margin", "l3_cover_m", "l3_covers",
                          "ats_streak", "last_margin", "last_cover_m", "last_upset_win", "last_upset_loss",
                          "last_blowout_win", "last_blowout_loss", "ps_margin", "ps_cover_m", "ps_wins", "revenge",
                          "h2h_margin", "next_div", "next_prime", "last_prime", "prev_div"]] + ["prev_travel_diff"]


def finalize(g: pd.DataFrame) -> pd.DataFrame:
    g["intl_any"] = ((g["h_intl"] + g["a_intl"]) > 0).astype(int)
    return g


if __name__ == "__main__":
    g = finalize(build())
    print(g.shape)
    print(g[MARKET_FEATS + SITU_FEATS].describe().T.to_string())

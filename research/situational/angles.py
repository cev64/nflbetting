"""ATS angle study: does each classic situational/market angle beat the spread consistently across eras?

Every angle is a signed variable x (positive = the angle favours the HOME side covering). For each era we report
the ATS win rate of the side the angle points to (games with x != 0, pushes excluded) and a z-score.
This is descriptive. The models never use these era results directly; they re-run an equivalent stability
screen inside each walk-forward training window (see models.select_ats_features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ERAS = [(1999, 2005), (2006, 2011), (2012, 2017), (2018, 2025)]


def angle_defs(g: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    sp = g["spread_line"]
    fav = np.sign(sp)  # +1 home favourite
    d = {
        "dog": ("Always take the underdog", -fav),
        "road_dog": ("Road underdog (take away team)", -(sp > 0).astype(float)),
        "home_dog": ("Home underdog (take home)", g["home_dog"].astype(float)),
        "home_dog_div": ("Home dog in division game", g["home_dog_div"].astype(float)),
        "road_fav_big": ("Fade road favourite of 7+", g["road_fav_big"].astype(float)),
        "fade_big_fav": ("Fade favourite of 10+", -g["big_fav_sign"]),
        "fade_fav_35": ("Fade favourite laying -3.5", -g["hook_fav_35"]),
        "take_fav_25": ("Take favourite at -2.5", g["hook_fav_25"]),
        "fade_fav_75": ("Fade favourite laying -7.5", -g["hook_fav_75"]),
        "fav_at_3": ("Take favourite at exactly -3", fav * g["key3"]),
        "fav_at_7": ("Take favourite at exactly -7", fav * g["key7"]),
        "pickem_home": ("Home team in pick'em (|spread|<=1)", g["pickem"].astype(float)),
        "dog_low_total": ("Big dog (7+) in low total (<=41)", g["dog_total_interact"]),
        "dog_in_wind": ("Underdog in wind >= 15 mph", -fav * g["windy"]),
        "dog_in_cold": ("Underdog in cold (<= 35F)", -fav * g["cold"]),
        "rest_adv": ("Team with more rest (sign of rest diff)", np.sign(g["rest_diff"])),
        "bye_adv": ("Team off bye vs team not off bye", g["bye_diff"].astype(float)),
        "home_dog_bye": ("Home dog off bye", g["home_dog_bye"].astype(float)),
        "thursday_home": ("Home team on Thursday", g["thursday"].astype(float)),
        "monday_home": ("Home team on Monday night", g["monday"].astype(float)),
        "prime_dog": ("Underdog in primetime", -fav * g["prime"]),
        "west_early": ("Fade West team in early body-clock kickoff", g["early_body_diff"].astype(float)),
        "west_night_home": ("West home team vs Eastern team at night", g["west_night_home"].astype(float)),
        "third_road": ("Fade road team on 3rd+ straight road game", g["a_3rd_road"].astype(float)),
        "home_after_trip": ("Home team back from 2+ road games", g["home_after_trip"].astype(float)),
        "travel_far": ("Fade team travelling 2000km+ further", np.sign(g["travel_diff"]) * (g["travel_diff"].abs() >= 2)),
        "dome_in_cold": ("Fade dome road team in cold", g["a_dome_in_cold"].astype(float)),
        "revenge": ("Revenge: team that lost the earlier meeting this season", g["revenge_diff"].astype(float)),
        "div_rematch_dog": ("Division rematch underdog", g["div_rematch_dog"]),
        "lookahead": ("Fade favourite with division game next week", -g["lookahead"]),
        "letdown": ("Fade favourite off upset/blowout win", -g["letdown"]),
        "bounce": ("Team off a 21+ point loss", g["bounce"]),
        "fade_last_cover": ("Fade team that beat the spread most last week", -np.sign(g["last_cover_m_diff"]) * (g["last_cover_m_diff"].abs() >= 14)),
        "fade_ats_streak": ("Fade team on longer ATS win streak (diff>=3)", -np.sign(g["ats_streak_diff"]) * (g["ats_streak_diff"].abs() >= 3)),
        "fade_l3_ats": ("Fade better last-3 ATS margin (diff>=10)", -np.sign(g["l3_cover_m_diff"]) * (g["l3_cover_m_diff"].abs() >= 10)),
        "fade_season_ats": ("Fade better season ATS margin (diff>=4, wk>=6)", -np.sign(g["avg_cover_m_diff"]) * ((g["avg_cover_m_diff"].abs() >= 4) & (g["week_f"] >= 6))),
        "fade_winpct": ("Fade better record (diff>=.3, wk>=6)", -np.sign(g["win_pct_diff"]) * ((g["win_pct_diff"].abs() >= .3) & (g["week_f"] >= 6))),
        "ps_early": ("Better previous season, weeks 1-4", np.sign(g["ps_x_early"]) * (g["ps_x_early"].abs() >= 5)),
        "eliminated": ("Fade out-of-contention team (late season)", -g["out_diff"].astype(float)),
        "qb_change": ("Team with new starting QB this week", g["qb_change_diff"].astype(float)),
        "young_qb": ("Team with QB < 8 starts", g["new_qb_diff"].astype(float)),
        "coach_year1": ("Team with first-year coach", g["coach_year1_diff"].astype(float)),
        "ref_home": ("Referee with home-cover history (|res|>=.03)", np.sign(g["ref_hc_res"]) * (g["ref_hc_res"].abs() >= 0.03)),
        "ml_disc": ("Moneyline more bullish than spread (|diff|>=.02)", np.sign(g["ml_vs_spread"]) * (g["ml_vs_spread"].abs() >= 0.02) * g["has_ml"]),
        "juice": ("Side with more expensive spread juice", np.sign(g["spread_juice"]) * (g["spread_juice"].abs() >= 0.01)),
        "fade_juice": ("Fade side with more expensive juice", -np.sign(g["spread_juice"]) * (g["spread_juice"].abs() >= 0.01)),
        "late_home_dog": ("Home dog, weeks 13+", g["home_dog"] * g["late"]),
        "playoff_dog": ("Playoff underdog", -fav * g["playoff"]),
        "turf_team_on_grass": ("Fade turf road team on grass", g["a_turf_on_grass"].astype(float)),
        "neutral_dog": ("Underdog at neutral/international site", -fav * g["neutral"]),
    }
    return d


def study(g: pd.DataFrame) -> pd.DataFrame:
    g = g[g["home_cover"].notna()]
    rows = []
    for key, (label, x) in angle_defs(g).items():
        x = x.reindex(g.index).fillna(0)
        row = {"angle": key, "label": label}
        zs = []
        for lo, hi in ERAS + [(1999, 2025)]:
            m = (g["season"].between(lo, hi)) & (x != 0)
            win = np.where(x[m] > 0, g.loc[m, "home_cover"], 1 - g.loc[m, "home_cover"])
            n = len(win)
            r = win.mean() if n else np.nan
            z = (r - 0.5) / np.sqrt(0.25 / n) if n else np.nan
            tag = f"{lo % 100:02d}-{hi % 100:02d}"
            row[f"{tag}_n"], row[f"{tag}_ats"], row[f"{tag}_z"] = n, r, z
            if (lo, hi) != (1999, 2025):
                zs.append(z)
        zs = np.array(zs)
        row["eras_same_sign"] = int(max((zs > 0).sum(), (zs < 0).sum()))
        row["eras_pos"] = int((zs > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(r: pd.Series) -> str:
    """Direction-agnostic: an angle (or its reverse) 'holds' only if the same side won in all 4 eras."""
    sgn = 1 if r["99-25_z"] >= 0 else -1
    zs = [sgn * r[f"{e}_z"] for e in ("99-05", "06-11", "12-17", "18-25") if pd.notna(r[f"{e}_z"])]
    rev = " (REVERSED)" if sgn < 0 else ""
    modern = zs[-2:]
    if all(z > 0 for z in zs) and len(zs) >= 3:
        strength = "holds" if abs(r["99-25_z"]) >= 2 else "consistent but weak"
        return strength + rev
    if all(z > 0 for z in modern) and sum(z > 0 for z in zs) >= len(zs) - 1:
        return "mostly holds" + rev
    if abs(r["99-25_z"]) > 1.5 and min(modern) <= 0:
        return "faded / died" + rev
    return "no edge"


if __name__ == "__main__":
    import features as F
    g = F.finalize(F.build())
    t = study(g)
    t["verdict"] = t.apply(verdict, axis=1)
    pd.set_option("display.width", 250)
    cols = ["angle"] + [c for c in t.columns if c.endswith("_ats") or c.endswith("_n")] + ["99-25_z", "eras_pos", "verdict"]
    print(t[cols].sort_values("99-25_z", ascending=False).round(3).to_string(index=False))

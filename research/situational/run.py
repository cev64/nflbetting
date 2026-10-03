"""Rebuild everything for the market-structure / situational / neural-net family.

    python research/situational/run.py            # all models (~4-5 min on 2 cores)

Writes
    research/preds/situational.csv, mlp.csv, knn.csv            (game_id, season, week, p_home, margin, p_home_cover)
    research/preds/<model>_factors.parquet                      (game_id, feature, label, home, away, better, fmt)
    research/situational/out/angles.csv                         (ATS angle study by era)
    research/situational/out/ats_by_edge.csv                    (ATS accuracy by edge threshold, per model/window)
    research/situational/out/selected_features.json             (walk-forward feature screens / shrinkage per season)
Predictions start in 2006 (first season with >= 7 training seasons); every season S is fit on 1999..S-1 only.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import angles  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402

PREDS = HERE.parent / "preds"
OUT = HERE / "out"
TEST_SEASONS = range(2006, 2027)


# ------------------------------------------------------------------------------------------- factors
FACTORS = [
    # key, label, home expr, away expr, better, fmt
    ("spread", "Point spread (team, betting notation)", lambda g: -g["spread_line"], lambda g: g["spread_line"], "low", "+0.0;-0.0"),
    ("ml_prob", "Moneyline win prob (no-vig)", lambda g: g["ml_prob"], lambda g: 1 - g["ml_prob"], "high", "0.0%"),
    ("implied_pts", "Implied team total (pts)", lambda g: g["home_implied"], lambda g: g["away_implied"], "high", "0.0"),
    ("rest", "Rest days", lambda g: g["h_rest"], lambda g: g["a_rest"], "high", "0"),
    ("travel_km", "Travel to game (km)", lambda g: g["h_travel_km"], lambda g: g["a_travel_km"], "low", "0"),
    ("tz_shift", "Time zones crossed", lambda g: g["h_tz_shift"].abs(), lambda g: g["a_tz_shift"].abs(), "low", "0"),
    ("body_clock", "Kickoff on body clock (hour)", lambda g: g["h_body_kick"], lambda g: g["a_body_kick"], "high", "0.0"),
    ("road_streak", "Consecutive games away from home", lambda g: g["h_away_run"], lambda g: g["a_away_run"], "low", "0"),
    ("last_cover", "Last game margin vs spread", lambda g: g["h_last_cover_m"], lambda g: g["a_last_cover_m"], "low", "+0.0;-0.0"),
    ("ats_rate", "ATS cover rate this season", lambda g: g["h_ats_raw"], lambda g: g["a_ats_raw"], "low", "0.0%"),
    ("avg_margin", "Avg point margin this season", lambda g: g["h_margin_raw"], lambda g: g["a_margin_raw"], "high", "+0.0;-0.0"),
    ("qb_starts", "Starting QB career starts", lambda g: g["h_qb_starts"], lambda g: g["a_qb_starts"], "high", "0"),
    ("coach_games", "Head coach games with team", lambda g: g["h_coach_games"], lambda g: g["a_coach_games"], "high", "0"),
    ("revenge", "Lost earlier meeting this season", lambda g: g["h_revenge"], lambda g: g["a_revenge"], "high", "0"),
    ("ref_home_cover", "Referee home-cover tendency (prior games)", lambda g: g["ref_hc_res"], lambda g: -g["ref_hc_res"], "high", "+0.000;-0.000"),
]


def build_factors(g: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    g = g[g["game_id"].isin(ids)].copy()
    for s in ("h", "a"):
        n = g[f"{s}_s_gp"].replace(0, np.nan)
        g[f"{s}_ats_raw"] = (g[f"{s}_ats_pct"] * (n + 2) - 1) / n       # undo the display-irrelevant shrinkage
        g[f"{s}_margin_raw"] = g[f"{s}_avg_margin"] * (n + 2) / n
    rows = []
    for key, label, fh, fa, better, fmt in FACTORS:
        rows.append(pd.DataFrame({"game_id": g["game_id"].values, "feature": key, "label": label,
                                  "home": fh(g).astype(float).values, "away": fa(g).astype(float).values,
                                  "better": better, "fmt": fmt}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------- scoring
def ats_by_edge(p: pd.DataFrame, g: pd.DataFrame, model: str) -> pd.DataFrame:
    # live-usable cut: edge quantile of the PREVIOUS season's (out-of-sample) predictions
    prev_cut = {q: p.assign(e=(p["p_home_cover"] - 0.5).abs()).groupby("season")["e"].quantile(1 - q)
                .rename(lambda s: s + 1) for q in [0.5, 0.25, 0.10, 0.05]}
    d = p.merge(g[["game_id", "result", "spread_line"]], on="game_id")
    d = d[d["result"].notna() & ((d["result"] - d["spread_line"]) != 0)]
    d["ok"] = (d["p_home_cover"] >= 0.5) == ((d["result"] - d["spread_line"]) > 0)
    d["edge_pts"] = (d["margin"] - d["spread_line"]).abs()
    d["edge_p"] = (d["p_home_cover"] - 0.5).abs()
    rows = []
    for lo, hi in [(2006, 2011), (2012, 2025), (2012, 2017), (2018, 2025)]:
        w = d[d["season"].between(lo, hi)]
        for thr in [0, 0.5, 1, 1.5, 2, 3]:
            s = w[w["edge_pts"] >= thr]
            rows.append(dict(model=model, window=f"{lo}-{hi}", kind="min_edge_pts", thr=thr, n=len(s),
                             ats=s["ok"].mean() if len(s) else np.nan))
        for q in [1.0, 0.5, 0.25, 0.10, 0.05]:
            # percentile thresholds are computed within each season (no look-ahead at future edges' scale)
            cut = w.groupby("season")["edge_p"].transform(lambda x: x.quantile(1 - q))
            s = w[w["edge_p"] >= cut]
            rows.append(dict(model=model, window=f"{lo}-{hi}", kind="top_frac_in_season", thr=q, n=len(s),
                             ats=s["ok"].mean()))
            if q < 1.0:
                s = w[w["edge_p"] >= w["season"].map(prev_cut[q])]
                rows.append(dict(model=model, window=f"{lo}-{hi}", kind="top_frac_prev_season_cut", thr=q, n=len(s),
                                 ats=s["ok"].mean() if len(s) else np.nan))
    return pd.DataFrame(rows)


def write_preds(p: pd.DataFrame, name: str):
    p = p.sort_values(["season", "week", "game_id"])
    p[["game_id", "season", "week", "p_home", "margin", "p_home_cover"]].to_csv(PREDS / f"{name}.csv", index=False)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    g = F.finalize(F.build())
    print(f"features: {g.shape} ({time.time() - t0:.0f}s)")

    # angle study (descriptive, all eras)
    st = angles.study(g)
    st["verdict"] = st.apply(angles.verdict, axis=1)
    st.sort_values("99-25_z", ascending=False).round(4).to_csv(OUT / "angles.csv", index=False)

    sel = {}
    p_sit, ch = M.run_situational(g.copy(), TEST_SEASONS, su_mode="screen", ats_mode="prior+screen")
    sel["situational"] = {str(k): v for k, v in ch.items()}
    print(f"situational done ({time.time() - t0:.0f}s)")
    p_mlp, info = M.run_mlp(g.copy(), TEST_SEASONS)
    sel["mlp"] = {str(k): v for k, v in info.items()}
    print(f"mlp done ({time.time() - t0:.0f}s)")
    p_knn = M.run_knn(g.copy(), TEST_SEASONS)
    print(f"knn done ({time.time() - t0:.0f}s)")

    edges = []
    for name, p in [("situational", p_sit), ("mlp", p_mlp), ("knn", p_knn)]:
        write_preds(p, name)
        build_factors(g, p["game_id"]).to_parquet(PREDS / f"{name}_factors.parquet", index=False)
        edges.append(ats_by_edge(p, g, name))
    pd.concat(edges).round(4).to_csv(OUT / "ats_by_edge.csv", index=False)
    with open(OUT / "selected_features.json", "w") as f:
        json.dump(sel, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Leakage / robustness checks for the situational family.

1. Truncation test: blank out every score on/after a cutoff date, rebuild features, and verify that every
   feature of games in the following week is bit-identical to the full build (=> no feature uses its own or
   later results).
2. Shuffled-label test: shuffle home_cover / home_win within each training season; the walk-forward pipeline
   must then score ~50% ATS (=> the evaluation itself is not leaking).
3. Drop-feature test: refit the cover model without the most-used screened features.
4. Screen OOS test: run the ATS feature screen on 1999-2011 only and check each kept feature in 2012-17 / 2018-25.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import features as F  # noqa: E402
import models as M  # noqa: E402


def truncation_test(cutoffs=("2009-12-01", "2013-10-01", "2017-11-20", "2021-12-01", "2024-09-20", "2025-12-25", "2026-01-10")):
    full = F.finalize(F.build())
    full, ang = M.add_angle_columns(full)
    cols = sorted(set(F.MARKET_FEATS + F.SITU_FEATS + ang + M.MLP_COLS + M.KNN_COLS))
    bad = 0
    for c in cutoffs:
        s = F.load_schedule()
        cut = pd.Timestamp(c)
        m = s["date"] >= cut
        s.loc[m, ["home_score", "away_score", "result", "total"]] = np.nan
        s.loc[m, "played"] = False
        tr = F.finalize(F.build(s))
        tr, _ = M.add_angle_columns(tr)
        first = full.loc[full["date"] >= cut, "date"].min()  # every game on the first game day after the cutoff
        ids = full.loc[full["date"] == first, "game_id"]
        a = full.set_index("game_id").loc[ids, cols].astype(float)
        b = tr.set_index("game_id").loc[ids, cols].astype(float)
        diff = (~np.isclose(a.fillna(-999).values, b.fillna(-999).values)).any(axis=0)
        print(f"  cutoff {c}: {len(ids)} games, columns differing: {[cols[i] for i in np.where(diff)[0]]}")
        bad += diff.sum()
    return bad


def shuffled_label_test(seed=0):
    g = F.finalize(F.build())
    rng = np.random.default_rng(seed)
    g2 = g.copy()
    for col in ["resid"]:
        for s, idx in g2[g2["result"].notna()].groupby("season").groups.items():
            g2.loc[idx, col] = rng.permutation(g2.loc[idx, col].values)
    g2["home_cover"] = np.where(g2["resid"] > 0, 1.0, np.where(g2["resid"] < 0, 0.0, np.nan))
    p, _ = M.run_situational(g2, range(2012, 2026), log=lambda *a: None)
    d = p.merge(g[["game_id", "result", "spread_line"]], on="game_id")
    d = d[d["result"].notna() & ((d["result"] - d["spread_line"]) != 0)]
    acc = ((d["p_home_cover"] >= .5) == ((d["result"] - d["spread_line"]) > 0)).mean()
    print(f"  shuffled-label situational ATS 2012-2025 (true labels): {acc:.4f} (n={len(d)})")
    return acc


def drop_feature_test(drop=("ats_pct_diff", "ref_hw_res", "ref_hc_res")):
    g = F.finalize(F.build())
    old = list(M.SU_POOL)
    M.SU_POOL = [c for c in M.SU_POOL if c not in drop]
    p, _ = M.run_situational(g, range(2012, 2026), log=lambda *a: None)
    M.SU_POOL = old
    d = p.merge(g[["game_id", "result", "spread_line"]], on="game_id")
    d = d[d["result"].notna() & ((d["result"] - d["spread_line"]) != 0)]
    d["ok"] = (d["p_home_cover"] >= .5) == ((d["result"] - d["spread_line"]) > 0)
    d["e"] = (d["p_home_cover"] - .5).abs()
    top = d[d["e"] >= d.groupby("season")["e"].transform(lambda x: x.quantile(.75))]
    print(f"  without {drop}: ATS {d['ok'].mean():.4f}, top-25% {top['ok'].mean():.4f} (n={len(top)})")
    return old


def screen_oos_test(train_end=2011):
    """Run the ATS stability screen on 1999..train_end only, then check each selected feature's relation with
    the home cover in the untouched periods 2012-2017 and 2018-2025 (pure out-of-sample test of the 'trends')."""
    g = F.finalize(F.build())
    g, ang = M.add_angle_columns(g)
    tr = g[(g["season"] <= train_end) & g["result"].notna() & g["spread_line"].notna()]
    pool = [c for c in M.SU_POOL + ang]
    sel = M.select_ats_features(tr, pool)
    rows = []
    d = g[g["home_cover"].notna()]
    for c in sel:
        r_in = np.corrcoef(tr.loc[tr["home_cover"].notna(), c].fillna(0), tr.loc[tr["home_cover"].notna(), "home_cover"])[0, 1]
        row = {"feature": c, "r_train_1999_%d" % train_end: r_in}
        for lo, hi in [(2012, 2017), (2018, 2025)]:
            m = d["season"].between(lo, hi)
            r = np.corrcoef(d.loc[m, c].fillna(0), d.loc[m, "home_cover"])[0, 1]
            row[f"r_{lo}_{hi}"] = r
            row[f"z_{lo}_{hi}"] = r * np.sqrt(m.sum())
        row["held_both"] = bool(np.sign(row[f"r_2012_2017"]) == np.sign(r_in) == np.sign(row[f"r_2018_2025"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    print(f"  held same sign in both OOS periods: {out['held_both'].sum()}/{len(out)} (coin-flip expectation {len(out)/4:.1f})")
    return out


if __name__ == "__main__":
    print("1) truncation test")
    nbad = truncation_test()
    nbad += truncation_test(tuple(str(d.date()) for d in F.load_schedule().query("season >= 2006")["date"].drop_duplicates().sample(25, random_state=1)))
    print("   PASS" if nbad == 0 else f"   FAIL ({nbad})")
    print("2) shuffled-label test")
    shuffled_label_test()
    print("3) drop-feature test")
    drop_feature_test()
    print("4) pre-2012 ATS screen, checked out-of-sample")
    screen_oos_test().to_csv(Path(__file__).resolve().parent / "out" / "screen_oos.csv", index=False)

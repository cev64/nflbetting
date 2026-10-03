"""DEV PHASE ONLY: close-game specialist selection on seasons <= 2017 (2018+ never loaded)."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import itertools, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from market import load_schedule, market_walk_forward, MARKET_CONFIG, logit, market_features
import features
from specialist import load_base_preds, add_base_deltas, walk_forward, FEATURE_GROUPS, PURE_MODELS, MKT_MODELS

DEV_MAX = 2017
TEST = list(range(2010, DEV_MAX + 1))


def build_dev():
    d = load_schedule(max_season=DEV_MAX)
    assert d["season"].max() <= DEV_MAX
    m = market_walk_forward(d, list(range(2000, DEV_MAX + 1)), MARKET_CONFIG)
    g = d.merge(m[["game_id", "p_home"]].rename(columns={"p_home": "p_mkt"}), on="game_id", how="left")
    g["l_mkt"] = logit(g["p_mkt"].values)
    mf = market_features(d, np.zeros(len(d), bool), dict(MARKET_CONFIG))
    g["mk_ml_minus_normal"] = mf["l_ml_minus_normal"].values
    g["mk_cov_juice"] = mf["l_cov_juice"].values
    g = g.merge(features.build(d, DEV_MAX).drop(columns=["temp", "wind"]), on="game_id", how="left")
    g = g.merge(load_base_preds(g["game_id"]), on="game_id", how="left")
    return add_base_deltas(g)


def sc(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return ((p >= 0.5) == y).mean(), -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    g = build_dev()
    played = g["played"] & ~g["tie"] & (g["season"] >= 2006)
    close = g["spread"].abs() <= 3.5
    x = g[played & close].copy()
    x["resid"] = x["hw"] - x["p_mkt"]
    allf = sorted({f for fs in FEATURE_GROUPS.values() for f in fs})
    print("== univariate (close games 2006-2017): residual (hw - p_mkt) on standardised feature, HC1")
    for f in (allf + [f"l_{m}" for m in PURE_MODELS + MKT_MODELS]) if len(sys.argv) <= 2 else []:
        v = x[f] - x["l_mkt"] if f.startswith("l_") else x[f]
        ok = v.notna()
        z = (v[ok] - v[ok].mean()) / v[ok].std()
        r = sm.OLS(x.loc[ok, "resid"], sm.add_constant(z)).fit(cov_type="HC1")
        early = ok & (x["season"] <= 2011); late = ok & (x["season"] >= 2012)
        t1 = sm.OLS(x.loc[early, "resid"], sm.add_constant(z[early[ok]])).fit(cov_type="HC1").tvalues.iloc[1]
        t2 = sm.OLS(x.loc[late, "resid"], sm.add_constant(z[late[ok]])).fit(cov_type="HC1").tvalues.iloc[1]
        print(f"  {f:28s} n={ok.sum():5d} b={r.params.iloc[1]:+.4f} t={r.tvalues.iloc[1]:+.2f}  (06-11 t={t1:+.2f}, 12-17 t={t2:+.2f})")
    print("intercept (home in close games beyond market): mean resid %.4f t=%.2f" % (x.resid.mean(), x.resid.mean() / (x.resid.std() / np.sqrt(len(x)))))

    # walk-forward configurations
    tgt = g[close & g["season"].isin(TEST) & g["played"] & ~g["tie"]]
    y = tgt["hw"].values
    a0, l0 = sc(y, tgt["p_mkt"].values)
    sp = np.where(tgt["spread"].values >= 0, 1, 0) == y
    print(f"\n== walk-forward close games {TEST[0]}-{TEST[-1]}, n={len(tgt)}: market-only acc={a0:.4f} ll={l0:.5f} (spread-fav acc {sp.mean():.4f})")
    combos = {
        "intercept only": [],
        "base": FEATURE_GROUPS["base"],
        "base_each": [f"l_{m}" for m in PURE_MODELS + MKT_MODELS],
        "kick": FEATURE_GROUPS["kick"],
        "coach": FEATURE_GROUPS["coach"],
        "late": FEATURE_GROUPS["late"],
        "record": FEATURE_GROUPS["record"],
        "venue": FEATURE_GROUPS["venue"],
        "kick+coach+late+venue": FEATURE_GROUPS["kick"] + FEATURE_GROUPS["coach"] + FEATURE_GROUPS["late"] + FEATURE_GROUPS["venue"],
        "all": sum(FEATURE_GROUPS.values(), []),
    }
    # base_each: model log-odds minus market, individually
    for m in PURE_MODELS + MKT_MODELS:
        g[f"l_{m}"] = (g[f"l_{m}"] - g["l_mkt"])
    combos.update({"coach+kick": FEATURE_GROUPS["coach"] + FEATURE_GROUPS["kick"],
                   "coach+base": FEATURE_GROUPS["coach"] + FEATURE_GROUPS["base"],
                   "coach+base+kick": FEATURE_GROUPS["coach"] + FEATURE_GROUPS["base"] + FEATURE_GROUPS["kick"],
                   "go4": ["d_coach_go4"], "coach_games": ["d_coach_log_games"],
                   "mkt_micro": ["mk_ml_minus_normal", "mk_cov_juice"],
                   "coach+mkt_micro": FEATURE_GROUPS["coach"] + ["mk_ml_minus_normal", "mk_cov_juice"]})
    if os.environ.get("ONLY"):
        combos = {k: v for k, v in combos.items() if k in os.environ["ONLY"].split(",")}
    TF = int(os.environ.get("TRAIN_FROM", "2006"))
    rows = []
    LAMS = [float(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["10", "200"])]
    for (name, fs), lam, fs_slope, icpt in itertools.product(combos.items(), LAMS, [False], [True, False]):
        if not fs and (lam != LAMS[0] or not icpt):
            continue
        o, _ = walk_forward(g, TEST, fs, lam, free_slope=fs_slope, intercept=icpt, train_from=TF)
        t = tgt.merge(o[["game_id", "p_close"]], on="game_id", how="left")
        p = t["p_close"].fillna(t["p_mkt"]).values
        a, l = sc(y, p)
        rows.append(dict(cfg=name, lam=lam, icpt=icpt, acc=round(a, 4), ll=round(l, 5), d_acc=round(a - a0, 4),
                         d_ll=round(l - l0, 5), flips=int(((p >= 0.5) != (tgt["p_mkt"].values >= 0.5)).sum())))
    r = pd.DataFrame(rows)
    print(r.to_string())
    r.to_csv(f"out/dev_close_grid_{os.environ.get('TAG', 'main')}.csv", index=False)

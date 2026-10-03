"""Final meta-model ("ensemble"): python research/meta/run.py [--models a b ...] [--exclude x ...]
Default base models: meta/models.json (written by select_models.py, dev-only rule), else every preds/*.csv.

Builds preds/ensemble.csv and preds/ensemble_meta.json, walk-forward: every season S >= 2007 is predicted by
meta-models fit only on completed games of seasons 2006..S-1 (base-model predictions are themselves
walk-forward out-of-sample). The design was chosen on DEV seasons 2010-2017 and frozen before the 2018-2025
holdout was scored (see meta/NOTES.md, meta/frozen.py).

SU  : family stack — L2 logistic on [market log-odds (unpenalised), mean (model log-odds - market log-odds)
      per model family], alpha=0.01; then the flip guard with delta=1, i.e. the stack sets the probability but
      never picks against the market favourite (on dev, every override rule was right < 50% of the time).
ATS : L2 logistic (no intercept) on [home_dog, spread/7, playoff dog, starters-out gap/3, mean cover log-odds
      per family], alpha=0.01. Best bets = games whose |p_cover - 0.5| is in the top 10% / 20% of the previous
      season's out-of-sample distribution (a cut that is usable live).
Deterministic, single-threaded numpy/scipy; < 1 minute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from data import (PREDS, SD, NO_MARKET, available_models, family, load, logit, su_resid,  # noqa: E402
                  cover_signal)
from designs import SU_DESIGNS, ATS_DESIGNS, fam_resids, ats_matrix  # noqa: E402
from frozen import SU_PRODUCTION, ATS_PRODUCTION  # noqa: E402

HERE = Path(__file__).resolve().parent
START, FIRST_OUT = 2006, 2007
TOPS = (0.10, 0.20)


def walk_forward(g, models, kind, design, cfg, first, last, keep_info=False):
    fn = (SU_DESIGNS if kind == "su" else ATS_DESIGNS)[design]
    tgt = "home_win" if kind == "su" else "home_cover"
    out = pd.Series(np.nan, index=g.index)
    infos = {}
    for S in range(first, last + 1):
        tr = g[(g.season < S) & (g.season >= START) & g[tgt].notna()]
        te = g[g.season == S]
        if len(te) == 0 or len(tr) < 200:
            continue
        pred, info = fn(tr, models, cfg, S)
        out.loc[te.index] = pred(te)
        if keep_info:
            infos[S] = info
    return out, infos


def best_bet_tiers(g, p_cover):
    """Tier per game: 1 = top 10% edge, 2 = top 20% (by previous season's out-of-sample edge quantiles)."""
    e = (p_cover - 0.5).abs()
    tier = pd.Series(0, index=g.index)
    cuts = {}
    for S in sorted(g.season.unique()):
        prev = e[(g.season == S - 1)].dropna()
        if len(prev) < 100:
            continue
        c10, c20 = np.quantile(prev, 1 - TOPS[0]), np.quantile(prev, 1 - TOPS[1])
        cuts[int(S)] = {"top10": float(c10), "top20": float(c20)}
        cur = g.season == S
        tier[cur & (e >= c20)] = 2
        tier[cur & (e >= c10)] = 1
    return tier, cuts


def score(g, p_home, p_cover, lo, hi, tier=None):
    d = g[(g.season >= lo) & (g.season <= hi)]
    r = {"window": f"{lo}-{hi}"}
    su = d[d.home_win.notna()]
    p = np.clip(p_home.loc[su.index].to_numpy(float), 1e-4, 1 - 1e-4)
    y = su.home_win.to_numpy()
    r.update(n=int(len(su)), su_acc=float(((p >= 0.5) == (y == 1)).mean()),
             logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), brier=float(np.mean((p - y) ** 2)))
    pk = su.pickem.to_numpy() == 1
    r["pickem_su_acc"] = float(((p >= 0.5) == (y == 1))[pk].mean())
    r["pickem_n"] = int(pk.sum())
    at = d[d.home_cover.notna()]
    if p_cover is not None:
        pc = p_cover.loc[at.index].to_numpy(float)
        hit = (pc >= 0.5) == (at.home_cover.to_numpy() == 1)
        r.update(ats_n=int(len(at)), ats_acc=float(hit.mean()))
        if tier is not None:
            t = tier.loc[at.index].to_numpy()
            for k, name in [(1, "top10"), (2, "top20")]:
                m = (t >= 1) & (t <= k)
                r[f"ats_{name}_acc"] = float(hit[m].mean()) if m.any() else None
                r[f"ats_{name}_n"] = int(m.sum())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def leaderboard(g, models, windows):
    rows = []
    mkt_p = pd.Series(g.p_mkt.to_numpy(), index=g.index)
    entries = [("market", mkt_p, None)] + [
        (m, g[f"{m}__p_home"], pd.Series(1 / (1 + np.exp(-cover_signal(g, m))), index=g.index)) for m in models]
    for name, p, pc in entries:
        row = {"id": name, "family": "market" if name == "market" else family(name),
               "sees_market": name not in NO_MARKET}
        for lo, hi in windows:
            s = score(g, p, pc, lo, hi)
            for k in ["su_acc", "logloss", "brier", "pickem_su_acc", "ats_acc"]:
                if k in s:
                    row[f"{k}_{lo}_{hi}"] = s[k]
        # how often it disagrees with the market favourite, and how often it is right when it does (2012-2025)
        d = g[(g.season >= 2012) & (g.season <= 2025) & g.home_win.notna()]
        pick = p.loc[d.index] >= 0.5
        fav = d.spread_line >= 0
        dis = pick != fav
        row["dissent_rate_2012_2025"] = round(float(dis.mean()), 4)
        row["dissent_acc_2012_2025"] = round(float((pick[dis] == (d.home_win[dis] == 1)).mean()), 4) if dis.any() else None
        rows.append(row)
    return rows


def context_table(g, models):
    """Descriptive (2012-2025): accuracy of each no-market family's pick by spread bucket, and when it
    dissents from the market favourite. Explains 'when to trust which model'."""
    d = g[(g.season >= 2012) & (g.season <= 2025) & g.home_win.notna()].copy()
    names = {0: "0-2.5 (pick'em)", 1: "3-3.5", 2: "4-6.5", 3: "7-9.5", 4: "10+"}
    fams: dict[str, list[str]] = {}
    for m in models:
        if m in NO_MARKET or family(m) == "wave2":
            fams.setdefault(family(m), []).append(m)
    out = []
    for b, nm in names.items():
        x = d[d.bucket == b]
        y = x.home_win == 1
        row = {"spread_bucket": nm, "n": int(len(x)), "market_acc": round(float(((x.spread_line >= 0) == y).mean()), 4)}
        for f, ms in sorted(fams.items()):
            L = np.mean([logit(x[f"{m}__p_home"]) for m in ms], 0)
            pick = L > 0
            dis = pick != (x.spread_line >= 0).to_numpy()
            row[f"{f}_acc"] = round(float((pick == y.to_numpy()).mean()), 4)
            row[f"{f}_dissent_n"] = int(dis.sum())
            row[f"{f}_dissent_acc"] = round(float((pick == y.to_numpy())[dis].mean()), 4) if dis.any() else None
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="base models (default: every preds/*.csv except market/ensemble)")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default="ensemble")
    a = ap.parse_args()
    default = available_models()
    mj = HERE / "models.json"
    if mj.exists():
        default = [m for m in json.loads(mj.read_text())["models"] if m in set(available_models())]
    models = [m for m in (a.models or default) if m not in set(a.exclude)]
    g, models = load(models, start=START)
    last = int(g.season.max())
    su_design, su_cfg = SU_PRODUCTION
    ats_design, ats_cfg = ATS_PRODUCTION
    p_home, su_info = walk_forward(g, models, "su", su_design, su_cfg, FIRST_OUT, last, keep_info=True)
    p_cover, ats_info = walk_forward(g, models, "ats", ats_design, ats_cfg, FIRST_OUT, last, keep_info=True)
    # unguarded stack probability (what the families jointly say before the market-pick guard)
    p_stack, _ = walk_forward(g, models, "su", su_cfg["base"], su_cfg["base_cfg"], FIRST_OUT, last)
    tier, cuts = best_bet_tiers(g, p_cover)

    keep = g.season >= FIRST_OUT
    out = g.loc[keep, ["game_id", "season", "week"]].copy()
    out["p_home"] = p_home[keep].round(5)
    out["margin"] = (SD * norm.ppf(np.clip(p_home[keep], 1e-4, 1 - 1e-4))).round(3)
    out["p_home_cover"] = p_cover[keep].round(5)
    out["ats_edge"] = (SD * norm.ppf(np.clip(p_cover[keep], 1e-4, 1 - 1e-4))).round(3)
    out["best_bet"] = tier[keep].astype(int)
    out["p_stack"] = p_stack[keep].round(5)
    out = out.dropna(subset=["p_home"])
    out.to_csv(PREDS / f"{a.out}.csv", index=False)

    # ---------------- explanation for the site
    S = last
    su_coef = su_info[S]["coef"]
    ats_coef = ats_info[S]["coef"]
    recent = g[(g.season >= S - 3) & (g.season < S)]
    R = fam_resids(recent, models)
    A = ats_matrix(recent, models, ats_cfg)
    su_weights = []
    for c in R.columns:
        f = c[2:]
        su_weights.append({"family": f, "models": [m for m in models if family(m) == f],
                           "coef": su_coef.get(c), "sd_signal": round(float(R[c].std()), 4),
                           "influence": round(float(su_coef.get(c, 0) * R[c].std()), 4)})
    ats_weights = []
    for c in A.columns:
        ats_weights.append({"input": c, "coef": ats_coef.get(c), "sd_signal": round(float(A[c].std()), 4),
                            "influence": round(float(ats_coef.get(c, 0) * A[c].std()), 4)})
    windows = [(2012, 2025), (2018, 2025), (2010, 2017)]
    ens_scores = {f"{lo}-{hi}": score(g, p_home, p_cover, lo, hi, tier) for lo, hi in windows}
    stack_scores = {f"{lo}-{hi}": score(g, p_stack, None, lo, hi) for lo, hi in windows}
    by_season = []
    for s in range(FIRST_OUT, last):
        sc = score(g, p_home, p_cover, s, s, tier)
        sm = score(g, pd.Series(g.p_mkt.to_numpy(), index=g.index), None, s, s)
        by_season.append({"season": s, "su_acc": sc["su_acc"], "market_su_acc": sm["su_acc"],
                          "stack_su_acc": score(g, p_stack, None, s, s)["su_acc"],
                          "ats_acc": sc["ats_acc"], "ats_top10_acc": sc.get("ats_top10_acc"),
                          "ats_top10_n": sc.get("ats_top10_n"), "ats_top20_acc": sc.get("ats_top20_acc"),
                          "ats_top20_n": sc.get("ats_top20_n")})
    validation = {}
    for kind in ["su", "ats"]:
        for tag in ["dev", "hold"]:
            f = HERE / f"out_{kind}_{tag}.csv"
            if f.exists():
                validation[f"{kind}_{tag}"] = pd.read_csv(f).drop(columns=["sec"], errors="ignore").to_dict("records")
    meta = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": a.out,
        "base_models": [{"id": m, "family": family(m), "sees_market": m not in NO_MARKET} for m in models],
        "design": {
            "su": {"design": su_design, "cfg": su_cfg,
                   "summary": "L2 logistic stack on the market log-odds (unpenalised) plus each model family's "
                              "mean disagreement with the market (log-odds); the stack sets the probability, "
                              "but the pick never goes against the market favourite (on dev 2010-2017 every "
                              "override rule was right < 50% of the time)."},
            "ats": {"design": ats_design, "cfg": ats_cfg,
                    "summary": "L2 logistic (no intercept) on road/home-dog structure, the starters-out injury gap "
                               "and each family's mean cover signal."},
            "best_bets": {"rule": "top 10% (tier 1) / top 20% (tier 2) of |p_home_cover - 0.5| using the previous "
                                  "season's out-of-sample quantiles", "cuts_by_season": cuts},
            "validation_protocol": "designs chosen on dev 2010-2017 (each season fit on 2006..S-1), frozen, then "
                                   "run once on holdout 2018-2025.",
        },
        "latest_season": int(S), "trained_on": f"{START}-{S - 1}",
        "su_weights": su_weights, "su_intercept": su_coef.get("intercept"), "su_market_coef": su_coef.get("mkt"),
        "ats_weights": ats_weights,
        "scores": {"ensemble": ens_scores, "stack_unguarded": stack_scores},
        "by_season": by_season,
        "leaderboard": leaderboard(g, models, windows),
        "context": context_table(g, models),
        "validation": validation,
        "columns": {"p_home": "P(home wins), ensemble", "margin": "13.5 * probit(p_home), home minus away pts",
                    "p_home_cover": "P(home covers spread_line)", "ats_edge": "13.5 * probit(p_home_cover), pts",
                    "best_bet": "1 = top-10% ATS edge, 2 = top-20%, 0 = none",
                    "p_stack": "family-stack probability before the market-pick guard"},
    }
    with open(PREDS / f"{a.out}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("models:", models)
    for k, v in ens_scores.items():
        print("ensemble", v)
    for k, v in stack_scores.items():
        print("stack(unguarded)", v)
    print("SU weights (latest):", [(w["family"], w["coef"], w["influence"]) for w in su_weights])
    print("ATS weights (latest):", [(w["input"], w["coef"]) for w in ats_weights])
    print("rows:", len(out), "seasons", int(out.season.min()), "-", int(out.season.max()),
          "| 2026 rows:", int((out.season == 2026).sum()))


if __name__ == "__main__":
    main()

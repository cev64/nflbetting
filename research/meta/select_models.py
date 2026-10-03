"""Decide which extra base models (e.g. wave2_*.csv) join the ensemble, using DEV seasons only.

Rule (fixed before any wave-2 file existed): starting from the wave-1 list, a new model is added if, with the
frozen production designs, it improves dev (2010-2017) SU log-loss or ATS log-loss by >= 0.0002 without making
the other one worse by more than 0.0002. Holdout (2018-2025) numbers are printed for transparency only.
Writes meta/models.json, which run.py uses as its default base-model list.
Usage: python research/meta/select_models.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from data import available_models, load  # noqa: E402
from frozen import SU_PRODUCTION, ATS_PRODUCTION  # noqa: E402
from harness import DEV, HOLD, walk, su_metrics, ats_metrics  # noqa: E402

WAVE1 = ["elo", "elo_mkt", "gbm", "kalman", "kalman_mkt", "knn", "logit_epa", "logit_epa_mkt", "mlp", "personnel",
         "personnel_ats", "personnel_nomkt", "ratings_combo", "ratings_combo_mkt", "ratings_ridge",
         "ratings_ridge_mkt", "situational", "xgb_margin"]
TOL = 0.0002


def evaluate(g, models, lo, hi):
    su_d, su_c = SU_PRODUCTION
    at_d, at_c = ATS_PRODUCTION
    p = walk(g, models, "su", su_c["base"], su_c["base_cfg"], lo, hi)
    s = su_metrics(g, p, lo, hi)
    pa = walk(g, models, "ats", at_d, at_c, lo - 1, hi)
    a = ats_metrics(g, pa, lo, hi)
    return {"su_ll": s["ll"], "stack_su": s["su"], "ats_ll": a["ats_ll"], "ats": a["ats"], "top10": a["top10"]}


def main():
    allm = available_models()
    cands = [m for m in allm if m not in WAVE1]
    g, _ = load(WAVE1 + cands)
    chosen = [m for m in WAVE1 if m in allm]
    base = evaluate(g, chosen, *DEV)
    print("wave-1 dev:", {k: round(v, 4) for k, v in base.items()})
    for m in cands:
        r = evaluate(g, chosen + [m], *DEV)
        dsu, dats = base["su_ll"] - r["su_ll"], base["ats_ll"] - r["ats_ll"]
        ok = (dsu >= TOL and dats >= -TOL) or (dats >= TOL and dsu >= -TOL)
        h0 = evaluate(g, chosen, *HOLD)
        h1 = evaluate(g, chosen + [m], *HOLD)
        print(f"{m:28s} dev d_su_ll={dsu:+.4f} d_ats_ll={dats:+.4f} -> {'ADD' if ok else 'skip'} | dev {r} "
              f"| holdout (info only) su_ll {h0['su_ll']:.4f}->{h1['su_ll']:.4f} ats_ll {h0['ats_ll']:.4f}->"
              f"{h1['ats_ll']:.4f} top10 {h0['top10']:.3f}->{h1['top10']:.3f}")
        if ok:
            chosen.append(m)
            base = r
    Path(__file__).with_name("models.json").write_text(json.dumps({"models": chosen}, indent=1))
    print("chosen:", chosen)


if __name__ == "__main__":
    main()

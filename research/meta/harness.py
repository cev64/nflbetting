"""Walk-forward harness for meta designs.

DEV  : test seasons 2010-2017, each fit on 2006..S-1. Every design/hyperparameter decision is made here.
HOLD : test seasons 2018-2025, each fit on 2006..S-1, run once with the frozen designs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from designs import SU_DESIGNS, ATS_DESIGNS

DEV = (2010, 2017)
HOLD = (2018, 2025)


def walk(g: pd.DataFrame, models, kind: str, design: str, cfg: dict, lo: int, hi: int, start: int = 2006,
         keep_info: bool = False):
    fn = (SU_DESIGNS if kind == "su" else ATS_DESIGNS)[design]
    tgt = "home_win" if kind == "su" else "home_cover"
    out = pd.Series(np.nan, index=g.index)
    infos = {}
    for S in range(lo, hi + 1):
        tr = g[(g.season < S) & (g.season >= start) & g[tgt].notna()]
        te = g[g.season == S]
        if len(te) == 0:
            continue
        pred, info = fn(tr, models, cfg, S)
        out.loc[te.index] = pred(te)
        if keep_info:
            infos[S] = info
    return (out, infos) if keep_info else out


def su_metrics(g, p, lo, hi):
    d = g[(g.season >= lo) & (g.season <= hi) & g.home_win.notna()]
    pp = np.clip(p.loc[d.index].to_numpy(float), 1e-4, 1 - 1e-4)
    y = d.home_win.to_numpy()
    ok = (pp >= 0.5) == (y == 1)
    fav = (d.spread_line.to_numpy() >= 0) == (y == 1)
    pk = d.pickem.to_numpy() == 1
    flips = (pp >= 0.5) != (d.spread_line.to_numpy() >= 0)
    return dict(su=ok.mean(), mkt=fav.mean(), ll=-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp)),
                pk_su=ok[pk].mean(), pk_mkt=fav[pk].mean(), flips=int(flips.sum()),
                flip_acc=ok[flips].mean() if flips.any() else np.nan, n=len(d))


def ats_metrics(g, p, lo, hi, top=(0.1, 0.2, 0.3)):
    """ATS accuracy, plus 'best bets': games whose |p-0.5| exceeds the previous season's out-of-sample
    (1-q) quantile of |p-0.5| (a cut usable live)."""
    d = g[(g.season >= lo - 1) & (g.season <= hi)].copy()
    d["p"] = p.loc[d.index].to_numpy(float)
    d["e"] = (d["p"] - 0.5).abs()
    res = {}
    dd = d[(d.season >= lo) & d.home_cover.notna()]
    ok = ((dd.p >= 0.5) == (dd.home_cover == 1))
    res["ats"] = ok.mean()
    res["n"] = len(dd)
    yy = dd.home_cover.to_numpy(); pp = np.clip(dd.p.to_numpy(), 1e-4, 1 - 1e-4)
    res["ats_ll"] = -np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    for q in top:
        hits, n = 0, 0
        for S in range(lo, hi + 1):
            prev = d[d.season == S - 1]["e"].dropna()
            if len(prev) == 0:
                continue
            cut = np.quantile(prev, 1 - q)
            cur = dd[(dd.season == S) & (dd.e >= cut)]
            hits += int(((cur.p >= 0.5) == (cur.home_cover == 1)).sum()); n += len(cur)
        res[f"top{int(q*100)}"] = hits / max(n, 1)
        res[f"top{int(q*100)}_n"] = n
    return res

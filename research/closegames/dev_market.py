"""DEV PHASE ONLY: choose the market-only configuration using seasons <= 2017 (2018+ is never loaded)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from market import load_schedule, market_walk_forward, logit

DEV_MAX = 2017
TEST = list(range(2009, DEV_MAX + 1))


def score(o: pd.DataFrame, d: pd.DataFrame, label: str) -> dict:
    g = o.merge(d[["game_id", "result"]], on="game_id")
    g = g[g["result"].notna() & (g["result"] != 0)]
    y = (g["result"] > 0).values
    p = np.clip(g["p_home"].values, 1e-4, 1 - 1e-4)
    close = g["spread"].abs().values <= 3.5
    acc = ((p >= 0.5) == y)
    return dict(cfg=label, n=len(g), su=acc.mean().round(4), ll=round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
                close_su=acc[close].mean().round(4), pickem_su=acc[g["spread"].abs().values <= 1].mean().round(4))


if __name__ == "__main__":
    d = load_schedule(max_season=DEV_MAX)
    assert d["season"].max() <= DEV_MAX
    rows = []
    # references
    ref = d[d["season"].isin(TEST)][["game_id", "season", "week", "spread"]].copy()
    ref["p_home"] = np.where(d.loc[ref.index, "spread"] >= 0, 0.6, 0.4)
    rows.append(score(ref, d, "spread favourite (eval baseline)"))
    r2 = ref.copy(); r2["p_home"] = d.loc[ref.index, "p_ml"].fillna(0.5).values
    rows.append(score(r2, d, "raw no-vig ML (0.5 if missing)"))
    base = dict(bw=1.0, C=1.0, sigma_total=False)
    for hl in [None, 10, 6, 4, 2]:
        for fs in [("l_normal",), ("l_normal", "l_cov_juice"), ("l_ml_or_normal", "has_ml"), ("l_normal", "l_ml_minus_normal", "l_cov_juice")]:
            tf = 2006 if any("ml" in x for x in fs) else 1999
            cfg = dict(base, features=fs, half_life=hl)
            rows.append(score(market_walk_forward(d, TEST, cfg, train_from=tf), d, f"{'+'.join(fs)} hl={hl} from{tf}"))
    print(pd.DataFrame(rows).to_string())

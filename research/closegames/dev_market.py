"""DEV PHASE ONLY: choose the market-only configuration using seasons <= 2017 (2018+ is never loaded)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
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
    sets = {
        "normal": ("l_normal",),
        "kernel": ("l_kernel",),
        "ml_or_kernel": ("l_ml_or_kernel", "has_ml"),
        "kernel+mldiff": ("l_kernel", "l_ml_minus_kernel"),
        "kernel+mldiff+juice": ("l_kernel", "l_ml_minus_kernel", "l_cov_juice"),
        "kernel+mldiff+juiceclose": ("l_kernel", "l_ml_minus_kernel", "l_cov_juice_close"),
        "kernel+mldiff+tot": ("l_kernel", "l_ml_minus_kernel", "tot_c", "l_ml_x_tot"),
        "kernel+mldiff+neutral": ("l_kernel", "l_ml_minus_kernel", "neutral"),
        "all": ("l_kernel", "l_ml_minus_kernel", "l_cov_juice", "tot_c", "l_ml_x_tot", "neutral"),
    }
    for name, fs in sets.items():
        cfg = dict(base, features=fs)
        rows.append(score(market_walk_forward(d, TEST, cfg), d, name))
    for bw, st in itertools.product([0.5, 1.0, 1.5, 2.5], [False, True]):
        cfg = dict(base, bw=bw, sigma_total=st, features=("l_kernel", "l_ml_minus_kernel"))
        rows.append(score(market_walk_forward(d, TEST, cfg), d, f"kernel+mldiff bw={bw} sigT={st}"))
    print(pd.DataFrame(rows).to_string())

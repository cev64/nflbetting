"""Market-only features and the market-only win-probability model.

Everything here uses only pre-game market data from `schedules.parquet`: the closing spread, total,
moneylines and the juice on the spread. Outcomes enter only through walk-forward fits on earlier seasons.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _implied(odds: pd.Series) -> pd.Series:
    """American odds -> implied probability (with vig)."""
    o = odds.astype(float)
    return pd.Series(np.where(o > 0, 100.0 / (o + 100.0), -o / (-o + 100.0)), index=odds.index).where(o.notna())


def load_schedule(max_season: int | None = None) -> pd.DataFrame:
    """Schedule with a spread_line, 1999+, plus clean market columns.

    `max_season` hides every later season (used by the dev phase so that 2018+ is never loaded).
    """
    s = pl.read_parquet(CACHE / "schedules.parquet").filter(pl.col("spread_line").is_not_null())
    if max_season is not None:
        s = s.filter(pl.col("season") <= max_season)
    d = s.to_pandas()
    d["gameday"] = pd.to_datetime(d["gameday"])
    d = d.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    d["spread"] = d["spread_line"].astype(float)
    d["total"] = d["total_line"].astype(float)
    d["neutral"] = (d["location"] == "Neutral").astype(float)
    d["playoff"] = (d["game_type"] != "REG").astype(float)
    # moneyline no-vig (multiplicative); drop impossible books (negative overround)
    ih, ia = _implied(d["home_moneyline"]), _implied(d["away_moneyline"])
    d["ml_vig"] = ih + ia - 1
    ok = d["ml_vig"] > 0
    d["p_ml"] = (ih / (ih + ia)).where(ok)
    ch, ca = _implied(d["home_spread_odds"]), _implied(d["away_spread_odds"])
    d["sp_vig"] = ch + ca - 1
    d["p_cov_juice"] = (ch / (ch + ca)).where(d["sp_vig"] > 0)
    d["hw"] = np.where(d["result"].isna(), np.nan, (d["result"] > 0).astype(float))
    d["played"] = d["result"].notna()
    d["tie"] = d["result"] == 0
    return d


# ---------------------------------------------------------------- spread -> probability maps
def kernel_map(train_spread: np.ndarray, train_result: np.ndarray, query: np.ndarray, bw: float) -> np.ndarray:
    """P(home wins | spread, no tie) by Gaussian-kernel smoothing of the empirical win rate.

    Training games are symmetrised: (s, r) and (-s, -r) are both used, which assumes the spread fully prices
    home field. Ties are dropped (eval.py drops them too). Works on any spread incl. key numbers, so the map is
    nonlinear wherever the data says it is.
    """
    m = train_result != 0
    s = np.concatenate([train_spread[m], -train_spread[m]])
    w = np.concatenate([(train_result[m] > 0), (train_result[m] < 0)]).astype(float)
    q = np.asarray(query, dtype=float)
    out = np.empty(len(q))
    for i in range(0, len(q), 2000):
        k = np.exp(-0.5 * ((q[i:i + 2000, None] - s[None, :]) / bw) ** 2)
        out[i:i + 2000] = (k @ w + 0.5) / (k.sum(1) + 1.0)
    return out


def normal_map(spread: np.ndarray, sigma: float = 13.5) -> np.ndarray:
    return norm.cdf(np.asarray(spread, dtype=float) / sigma)


# ---------------------------------------------------------------- market-only model
MARKET_CONFIG = dict(
    # frozen after the dev phase (seasons <= 2017 only); see NOTES.md "Market-only model"
    bw=1.0,
    features=("l_normal", "l_cov_juice"),
    C=1.0,
    sigma_total=False,
    half_life=6.0,
    train_from=1999,
)


def market_features(d: pd.DataFrame, train_mask: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Build market-only features for all rows of `d`; kernel map fit on rows in `train_mask` only."""
    tr = d[train_mask & d["played"].values]
    f = pd.DataFrame(index=d.index)
    sp = d["spread"].values
    if cfg.get("sigma_total"):
        # total-dependent scale: express the spread in units of the typical margin SD for that total
        T = d["total"].fillna(44.0).values
        sp_eff = sp * np.sqrt(44.0 / np.clip(T, 30, 60))
        tr_T = tr["total"].fillna(44.0).values
        tr_sp = tr["spread"].values * np.sqrt(44.0 / np.clip(tr_T, 30, 60))
    else:
        sp_eff, tr_sp = sp, tr["spread"].values
    need_k = any("kernel" in x for x in cfg["features"])
    p_k = kernel_map(tr_sp, tr["result"].values, sp_eff, cfg["bw"]) if need_k else normal_map(sp_eff)
    f["p_kernel"] = p_k
    f["l_kernel"] = logit(p_k)
    f["l_normal"] = logit(normal_map(sp))
    # total-dependent scale: fewer expected points -> each point of spread is worth more probability
    T0 = d["total"].fillna(44.0).clip(30, 60).values
    f["l_normal_tot"] = logit(normal_map(sp * np.sqrt(44.0 / T0)))
    has_ml = d["p_ml"].notna().values
    f["has_ml"] = has_ml.astype(float)
    f["l_ml"] = np.where(has_ml, logit(d["p_ml"].fillna(0.5).values), 0.0)
    f["l_ml_or_kernel"] = np.where(has_ml, f["l_ml"].values, f["l_kernel"].values)
    f["l_ml_minus_kernel"] = np.where(has_ml, f["l_ml"].values - f["l_kernel"].values, 0.0)
    f["l_ml_minus_normal"] = np.where(has_ml, f["l_ml"].values - f["l_normal"].values, 0.0)
    f["l_ml_or_normal"] = np.where(has_ml, f["l_ml"].values, f["l_normal"].values)
    pc = d["p_cov_juice"].values
    f["l_cov_juice"] = np.where(np.isfinite(pc), logit(np.nan_to_num(pc, nan=0.5)), 0.0)
    # juice is worth more away from key numbers; scale by the kernel-map slope (prob per point)
    f["l_cov_juice_close"] = f["l_cov_juice"] * (np.abs(sp) <= 3.5)
    T = d["total"].fillna(44.0).values
    f["tot_c"] = (T - 44.0) / 4.5
    f["l_ml_x_tot"] = f["l_ml_or_kernel"] * f["tot_c"]
    f["neutral"] = d["neutral"].values
    return f


def fit_logit(X: np.ndarray, y: np.ndarray, C: float, w: np.ndarray | None = None) -> LogisticRegression:
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(X, y, sample_weight=w)
    return m


def market_walk_forward(d: pd.DataFrame, seasons: list[int], cfg: dict, train_from: int | None = None,
                        kernel_from: int = 1999) -> pd.DataFrame:
    """Walk-forward market-only p_home for each season in `seasons` (fit on seasons < S)."""
    out = []
    feats = list(cfg["features"])
    train_from = cfg.get("train_from", 2006) if train_from is None else train_from
    for S in seasons:
        kmask = (d["season"].values >= kernel_from) & (d["season"].values < S)
        f = market_features(d, kmask, cfg)
        tr = (d["season"].values >= train_from) & (d["season"].values < S) & d["played"].values & ~d["tie"].values
        te = d["season"].values == S
        hl = cfg.get("half_life")
        w = None if not hl else 0.5 ** ((S - 1 - d.loc[tr, "season"].values) / hl)
        m = fit_logit(f.loc[tr, feats].values, d.loc[tr, "hw"].values, cfg["C"], w)
        p = m.predict_proba(f.loc[te, feats].values)[:, 1]
        o = d.loc[te, ["game_id", "season", "week", "spread"]].copy()
        o["p_home"] = p
        o["p_kernel"] = f.loc[te, "p_kernel"].values
        out.append(o)
    return pd.concat(out, ignore_index=True)

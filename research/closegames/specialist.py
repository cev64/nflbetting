"""Close-game specialist: a logistic model fit only on games with |spread| <= CLOSE.

The market-only log-odds enters as an offset (slope fixed at 1, optionally with a free slope), so every
other input can only *move* the market's probability. Inputs: the out-of-sample predictions of the wave-1
base models (as log-odds deltas vs the market) and the close-game features from `features.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from market import logit, sigmoid

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "preds"

# wave-1 base models (fixed list so the result does not depend on whatever else lands in preds/)
PURE_MODELS = ["elo", "kalman", "ratings_ridge", "ratings_combo", "logit_epa", "xgb_margin", "personnel_nomkt"]
MKT_MODELS = ["elo_mkt", "kalman_mkt", "ratings_ridge_mkt", "ratings_combo_mkt", "logit_epa_mkt", "gbm",
              "personnel", "situational", "mlp", "knn"]

FEATURE_GROUPS = {
    "base": ["d_pure", "d_mktm"],
    "kick": ["d_k_fgoe", "d_k_fgoe_long", "d_k_fgoe_x_wind"],
    "coach": ["d_coach_go4", "d_coach_log_games"],
    "late": ["d_two_min_off_epa", "d_two_min_def_epa", "d_late_close_off_epa", "d_late_close_def_epa"],
    "record": ["d_close_wpct"],
    "venue": ["neutral", "dome", "wind_out", "cold_out"],
}


def load_base_preds(game_ids: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"game_id": game_ids.values})
    for m in PURE_MODELS + MKT_MODELS:
        f = PREDS / f"{m}.csv"
        if not f.exists():
            out[f"l_{m}"] = np.nan
            continue
        p = pd.read_csv(f, usecols=["game_id", "p_home"]).drop_duplicates("game_id", keep="last")
        p[f"l_{m}"] = logit(p["p_home"].values)
        out = out.merge(p[["game_id", f"l_{m}"]], on="game_id", how="left")
    return out


def add_base_deltas(g: pd.DataFrame) -> pd.DataFrame:
    """d_pure / d_mktm = mean over available models of (model log-odds - market log-odds); 0 if none."""
    for name, ms in (("d_pure", PURE_MODELS), ("d_mktm", MKT_MODELS)):
        L = np.column_stack([g[f"l_{m}"].values - g["l_mkt"].values for m in ms])
        n = np.isfinite(L).sum(1)
        g[name] = np.where(n > 0, np.nansum(L, 1) / np.maximum(n, 1), 0.0)
        g[f"{name}_n"] = n
    return g


def fit_offset_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray, lam: float, free_slope: bool,
                     w: np.ndarray | None = None, intercept: bool = True) -> np.ndarray:
    """L2 logistic with offset. Params: [intercept, (slope on offset), coefs]; only coefs are penalised."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    nfree = 2 if free_slope else 1

    def f(b):
        z = b[0] + offset * (1 + (b[1] if free_slope else 0.0)) + X @ b[nfree:]
        p = sigmoid(z)
        ll = np.sum(w * (np.logaddexp(0, z) - y * z))
        r = p - y
        g = np.empty_like(b)
        g[0] = np.sum(w * r) if intercept else 0.0
        if free_slope:
            g[1] = np.sum(w * r * offset)
        g[nfree:] = X.T @ (w * r) + 2 * lam * b[nfree:]
        return ll + lam * np.sum(b[nfree:] ** 2), g

    b0 = np.zeros(nfree + k)
    res = minimize(f, b0, jac=True, method="L-BFGS-B")
    return res.x


def predict_offset_logit(b: np.ndarray, X: np.ndarray, offset: np.ndarray, free_slope: bool) -> np.ndarray:
    nfree = 2 if free_slope else 1
    return sigmoid(b[0] + offset * (1 + (b[1] if free_slope else 0.0)) + X @ b[nfree:])


def walk_forward(g: pd.DataFrame, seasons: list[int], feats: list[str], lam: float, close: float = 3.5,
                 train_from: int = 2006, free_slope: bool = False, half_life: float | None = None,
                 min_train_seasons: int = 3, intercept: bool = True) -> tuple[pd.DataFrame, dict]:
    """Predict close games of each season S in `seasons` from a fit on close games of seasons [train_from, S).

    Returns per-game predictions (only close games of the test seasons; None-fit seasons are omitted) and
    the per-season coefficient dict (standardised scale).
    """
    out, coefs = [], {}
    isclose = g["spread"].abs().values <= close
    for S in seasons:
        tr = isclose & (g["season"].values >= train_from) & (g["season"].values < S) & g["played"].values \
            & ~g["tie"].values
        if g.loc[tr, "season"].nunique() < min_train_seasons:
            continue
        te = isclose & (g["season"].values == S)
        Xtr = g.loc[tr, feats].values.astype(float)
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
        Z = lambda A: np.nan_to_num((A - mu) / sd)  # noqa: E731  (missing -> mean -> no effect)
        w = None if not half_life else 0.5 ** ((S - 1 - g.loc[tr, "season"].values) / half_life)
        b = fit_offset_logit(Z(Xtr), g.loc[tr, "hw"].values, g.loc[tr, "l_mkt"].values, lam, free_slope, w, intercept)
        p = predict_offset_logit(b, Z(g.loc[te, feats].values.astype(float)), g.loc[te, "l_mkt"].values, free_slope)
        o = g.loc[te, ["game_id", "season", "week"]].copy()
        o["p_close"] = p
        out.append(o)
        nfree = 2 if free_slope else 1
        coefs[S] = dict(zip(feats, b[nfree:])) | {"_intercept": b[0]} | ({"_slope": 1 + b[1]} if free_slope else {})
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id", "p_close"])), coefs

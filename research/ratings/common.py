"""Walk-forward probability mapping, market blending, scoring and output for the ratings family.

A rating model produces a `raw` frame with one row per game (chronological) and a pre-game
`rmargin` column (expected home margin from the rating alone, in points). Everything here maps
that to p_home / margin / p_home_cover with models fit ONLY on seasons strictly before the
season being predicted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from data import PREDS, load_games

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as ev  # noqa: E402  (shared scorer)

FIRST_OUT = 2006      # first season we emit predictions for (stacker needs 2006-2011 folds)
FIRST_TRAIN = 2000    # first season usable to fit mappings (1999 = rating warm-up)
LAST = 2026


def _logit_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, l2: float = 1e-3) -> np.ndarray:
    """Weighted logistic regression with intercept (column 0 of X must be ones). Tiny L2 for stability."""
    w = np.ones(len(y)) if w is None else w

    def f(b):
        z = X @ b
        ll = np.sum(w * (np.logaddexp(0, z) - y * z)) + l2 * np.sum(b[1:] ** 2)
        g = X.T @ (w * (1 / (1 + np.exp(-z)) - y)) + 2 * l2 * np.r_[0, b[1:]]
        return ll, g

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _sig(z):
    return 1 / (1 + np.exp(-z))


def walk_forward(raw: pl.DataFrame, feats: list[str], first_train: int = FIRST_TRAIN, train_window: int | None = None) -> pl.DataFrame:
    """For each season S >= FIRST_OUT fit on played games of seasons [first_train, S):
      * p_home:       logistic  home_win ~ 1 + feats
      * margin:       OLS       result   ~ 1 + feats
      * p_home_cover: logistic  cover    ~ 1 + (margin - spread_line)   (margin = the OLS fit above, in-sample on train)
    `feats` columns must exist in raw; spread_line may be one of them (market blend).
    """
    df = raw.with_columns(
        (pl.col("result") > 0).cast(pl.Float64).alias("home_win"),
        ((pl.col("result") - pl.col("spread_line")) > 0).cast(pl.Float64).alias("cover"),
    )
    out = []
    for s in range(FIRST_OUT, LAST + 1):
        lo = first_train if train_window is None else max(first_train, s - train_window)
        tr = df.filter(pl.col("season").is_between(lo, s - 1) & pl.col("played") & pl.col("spread_line").is_not_null())
        te = df.filter((pl.col("season") == s) & pl.col("spread_line").is_not_null())
        if te.height == 0:
            continue
        Xtr = np.column_stack([np.ones(tr.height)] + [tr[c].to_numpy() for c in feats])
        Xte = np.column_stack([np.ones(te.height)] + [te[c].to_numpy() for c in feats])
        su = tr.filter(pl.col("result") != 0)
        Xsu = np.column_stack([np.ones(su.height)] + [su[c].to_numpy() for c in feats])
        b_win = _logit_fit(Xsu, su["home_win"].to_numpy())
        b_mar = _ols_fit(Xtr, tr["result"].to_numpy().astype(float))
        # ATS: edge = fitted margin - line, fit on non-push training games
        ats = tr.filter((pl.col("result") - pl.col("spread_line")) != 0)
        Xa = np.column_stack([np.ones(ats.height)] + [ats[c].to_numpy() for c in feats])
        edge_tr = Xa @ b_mar - ats["spread_line"].to_numpy()
        b_cov = _logit_fit(np.column_stack([np.ones(len(edge_tr)), edge_tr]), ats["cover"].to_numpy())
        m_te = Xte @ b_mar
        out.append(pl.DataFrame({
            "game_id": te["game_id"], "season": te["season"], "week": te["week"],
            "p_home": _sig(Xte @ b_win), "margin": m_te,
            "p_home_cover": _sig(b_cov[0] + b_cov[1] * (m_te - te["spread_line"].to_numpy())),
        }))
    return pl.concat(out)


def score(pred: pl.DataFrame, lo: int = 2012, hi: int = 2025) -> dict:
    p = pred.select([c for c in ["game_id", "p_home", "margin", "p_home_cover"] if c in pred.columns])
    return ev.score(p, lo, hi)


def quick(pred: pl.DataFrame, tag: str = "") -> str:
    r1, r0 = score(pred, 2012, 2025), score(pred, 2006, 2011)
    return (f"{tag:28s} 12-25 SU {r1['su_acc']:.4f} ATS {r1['ats_acc']:.4f} LL {r1['logloss']:.4f} | "
            f"06-11 SU {r0['su_acc']:.4f} ATS {r0['ats_acc']:.4f} LL {r0['logloss']:.4f}")


def write_preds(pred: pl.DataFrame, model_id: str) -> Path:
    path = PREDS / f"{model_id}.csv"
    pred.sort(["season", "week", "game_id"]).write_csv(path, float_precision=5)
    return path


def base_frame() -> pl.DataFrame:
    return load_games().select("game_id", "season", "week", "game_type", "played", "result", "spread_line",
                               "home_team", "away_team")

"""Walk-forward personnel models.

For every test season S (2006..2026) each model is fit on seasons [2002, S-1] only.  The L2 strength of the
logistic / ridge fits is chosen *inside* each fold by rolling-origin validation on the last 3 training seasons
(nested: never looks at season S).  Feature sets were chosen on test seasons 2006-2011 only (see NOTES.md).

Market model trick: the spread column is standardised and then multiplied by SPREAD_BOOST, which makes its L2
penalty negligible -> the market enters essentially unpenalised while personnel adjustments are shrunk.
"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TRAIN_START = 2002
TEST_SEASONS = list(range(2006, 2027))
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
ALPHA_GRID = [1, 10, 100, 1000, 10000, 100000]
SPREAD_BOOST = 50.0
RES_SD = 13.5  # sd of (result - expected margin) in the NFL; used to map margin edge -> cover probability

QB_F = ["d_qb_epa", "d_qb_epa_long", "d_qb_cpoe", "d_qb_change_inseason", "d_qb_epa_delta", "d_qb_backup", "d_qb_rookie",
        "d_qb_log_exp"]
INJ_F = [f"d_inj_usual_{g}" for g in ("QB", "OL", "SK", "DL", "LB", "DB")] + ["d_inj_starters_out", "d_inj_q_load"]
INJ_RECENT_F = [f"d_inj_recent_{g}" for g in ("QB", "OL", "SK", "DL", "LB", "DB")]
CTX_F = ["rest_diff", "neutral"]
CORE_F = ["d_qb_epa", "d_qb_epa_long", "d_qb_cpoe", "d_inj_starters_out", "d_inj_usual_QB", "d_inj_usual_SK", "rest_diff"]

CANDIDATES = {
    "personnel": {"core": ["spread_line"] + CORE_F,
                  "full": ["spread_line"] + QB_F + INJ_F + CTX_F,
                  "full_recent": ["spread_line"] + QB_F + INJ_F + INJ_RECENT_F + CTX_F},
    "personnel_nomkt": {"core": CORE_F + ["neutral"],
                        "full": QB_F + INJ_F + CTX_F,
                        "full_recent": QB_F + INJ_F + INJ_RECENT_F + CTX_F},
}
# chosen by 2006-2011 walk-forward logloss (select.py); see NOTES.md
FEATURES = {"personnel": CANDIDATES["personnel"]["core"], "personnel_nomkt": CANDIDATES["personnel_nomkt"]["core"]}


def prep(g: pl.DataFrame) -> pl.DataFrame:
    return g.with_columns(
        (pl.col("result") > 0).cast(pl.Float64).alias("y_win"),
        (pl.col("result") - pl.col("spread_line")).alias("res"),
    )


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class _Design:
    """Standardise; boost the spread column so it is effectively unpenalised."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, df):
        X = df.select(self.feats).fill_null(0.0).to_numpy().astype(float)
        self.sc = StandardScaler().fit(X)
        return self

    def __call__(self, df):
        Z = self.sc.transform(df.select(self.feats).fill_null(0.0).to_numpy().astype(float))
        if "spread_line" in self.feats:
            Z[:, self.feats.index("spread_line")] *= SPREAD_BOOST
        return Z

    def unscale(self, coef):
        c = np.array(coef, dtype=float) / self.sc.scale_
        if "spread_line" in self.feats:
            c[self.feats.index("spread_line")] *= SPREAD_BOOST
        return c


def _fit(df, feats, target, kind, reg, intercept=True):
    D = _Design(feats).fit(df)
    y = df[target].to_numpy()
    if kind == "logit":
        m = LogisticRegression(C=reg, max_iter=5000).fit(D(df), y)
        return (lambda z: m.predict_proba(D(z))[:, 1]), D.unscale(m.coef_[0]), float(m.intercept_[0])
    m = Ridge(alpha=reg, fit_intercept=intercept).fit(D(df), y)
    return (lambda z: m.predict(D(z))), D.unscale(m.coef_), float(m.intercept_)


def _choose(train, feats, target, kind, grid, intercept=True):
    seasons = sorted(train["season"].unique().to_list())
    val = seasons[-3:]
    tot = {r: 0.0 for r in grid}
    for v in val:
        tr, va = train.filter(pl.col("season") < v), train.filter(pl.col("season") == v)
        yv = va[target].to_numpy()
        for r in grid:
            f, _, _ = _fit(tr, feats, target, kind, r, intercept)
            p = f(va)
            tot[r] += _logloss(p, yv) * len(yv) if kind == "logit" else float(np.sum((p - yv) ** 2))
    return min(grid, key=lambda r: tot[r])


def walk_forward(g: pl.DataFrame, feats: list[str], seasons=None, ats: bool = True):
    """Returns (preds, coefficient table).  preds: game_id, season, week, p_home, margin[, p_home_cover]."""
    g = prep(g)
    played = g.filter(pl.col("result").is_not_null())
    mkt = "spread_line" in feats
    out, coefs = [], []
    for S in seasons or TEST_SEASONS:
        tr = played.filter(pl.col("season").is_between(TRAIN_START, S - 1))
        te = g.filter((pl.col("season") == S) & pl.col("spread_line").is_not_null())
        if te.height == 0:
            continue
        trw = tr.filter(pl.col("result") != 0)
        C = _choose(trw, feats, "y_win", "logit", C_GRID)
        fwin, cw, iw = _fit(trw, feats, "y_win", "logit", C)
        if mkt:
            # margin = market spread + ridge-shrunk personnel adjustment fitted on the ATS residual
            # (result - spread); no intercept, so ATS picks come from personnel deltas only
            pf = [f for f in feats if f != "spread_line"]
            a = _choose(tr, pf, "res", "ridge", ALPHA_GRID, intercept=False)
            fmar, cm, im = _fit(tr, pf, "res", "ridge", a, intercept=False)
            edge = fmar(te)
            margin = te["spread_line"].to_numpy() + edge
            cm = np.r_[1.0, cm]
        else:
            a = _choose(tr, feats, "result", "ridge", ALPHA_GRID)
            fmar, cm, im = _fit(tr, feats, "result", "ridge", a)
            margin = fmar(te)
        pred = te.select("game_id", "season", "week").with_columns(
            pl.Series("p_home", fwin(te)), pl.Series("margin", margin))
        if ats and mkt:
            pred = pred.with_columns(pl.Series("p_home_cover", norm.cdf(edge / RES_SD)))
        coefs.append({"season": S, "C": C, "alpha": a, "intercept_logit": iw, "intercept_margin": im,
                      **{f"logit_{f}": float(c) for f, c in zip(feats, cw)},
                      **{f"margin_{f}": float(c) for f, c in zip(feats, cm)}})
        out.append(pred)
    return pl.concat(out), pl.DataFrame(coefs)


def quick_score(pred: pl.DataFrame, g: pl.DataFrame, lo=2012, hi=2025) -> dict:
    d = pred.join(g.select("game_id", "result", "spread_line"), on="game_id").filter(
        pl.col("season").is_between(lo, hi) & pl.col("result").is_not_null())
    su = d.filter(pl.col("result") != 0)
    y = (su["result"] > 0).to_numpy().astype(float)
    p = su["p_home"].to_numpy()
    a = d.filter((pl.col("result") - pl.col("spread_line")) != 0)
    cov = (a["result"] - a["spread_line"]) > 0
    pick = (a["p_home_cover"] >= 0.5) if "p_home_cover" in a.columns else ((a["margin"] - a["spread_line"]) > 0)
    return dict(su=round(float(((p >= 0.5) == (y == 1)).mean()), 4), ll=round(_logloss(p, y), 4),
                ats=round(float((pick == cov).mean()), 4), n=su.height)


ATS_F = ["d_inj_starters_out"]
ATS_TRAIN_START = 2009  # injury reports start in 2009


def ats_walk_forward(g: pl.DataFrame, seasons=None):
    """`personnel_ats`: market spread + an unregularised, no-intercept OLS adjustment on the difference in
    regular starters listed out (fit on seasons [2009, S-1]).  NOTE: this single feature was picked after
    looking at full-sample (2009-2025) residual diagnostics, so its 2012-2025 ATS is NOT a clean out-of-sample
    number (the coefficient itself is walk-forward).  p_home = Phi(margin / 13.5)."""
    g = prep(g)
    played = g.filter(pl.col("result").is_not_null())
    out, coefs = [], []
    for S in seasons or TEST_SEASONS:
        te = g.filter((pl.col("season") == S) & pl.col("spread_line").is_not_null())
        if te.height == 0:
            continue
        tr = played.filter(pl.col("season").is_between(ATS_TRAIN_START, S - 1))
        if tr.height < 200:
            b = np.zeros(len(ATS_F))
        else:
            b = np.linalg.lstsq(tr.select(ATS_F).to_numpy(), tr["res"].to_numpy(), rcond=None)[0]
        adj = te.select(ATS_F).fill_null(0.0).to_numpy() @ b
        margin = te["spread_line"].to_numpy() + adj
        out.append(te.select("game_id", "season", "week").with_columns(
            pl.Series("p_home", norm.cdf(margin / RES_SD)), pl.Series("margin", margin),
            pl.Series("p_home_cover", norm.cdf(adj / RES_SD))))
        coefs.append({"season": S, **{f: float(c) for f, c in zip(ATS_F, b)}})
    return pl.concat(out), pl.DataFrame(coefs)

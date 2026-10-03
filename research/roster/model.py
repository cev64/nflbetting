"""Walk-forward models on the lineup features.

For test season S every model is fit on seasons FIRST_TRAIN..S-1 (completed games only). Regularisation is
chosen inside each fold by rolling-origin validation on the last 3 training seasons (nested), so no setting is
picked by looking at the test season.

  mkt   SU:  logistic(home_win ~ spread [effectively unpenalised] + lineup features [L2])
        ATS: ridge, no intercept, on (result - spread) ~ lineup features;  margin = spread + adj,
             p_home_cover = Phi(adj / 13.5)
  nomkt SU:  logistic(home_win ~ lineup/QB features + neutral + rest);  margin = ridge on result
  gbm   (dev comparison only) LightGBM with the market logit as init_score / the spread as base margin
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge

FIRST_TRAIN = 2003
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
A_GRID = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
SPREAD_BOOST = 50.0
SIGMA = 13.5

UNITS = ["rec", "rush", "prush", "cover", "rund", "ol", "olexp"]
FEATURE_SETS = {
    # pre-declared candidate sets, chosen between on dev seasons (<= 2017) only
    "agg": ["d_delta_off", "d_delta_def", "d_delta_qb", "d_inj_off", "d_inj_def", "d_inj_olexp", "d_starters_out",
            "d_k_val"],
    "inj": ["d_starters_out", "d_inj_off", "d_inj_def", "d_inj_olexp", "d_delta_qb"],
    "units": [f"d_delta_{u}" for u in UNITS] + [f"d_inj_{u}" for u in UNITS] + ["d_delta_qb", "d_starters_out",
                                                                                 "d_k_val"],
    "full": [f"d_delta_{u}" for u in UNITS] + [f"d_inj_{u}" for u in UNITS] + [f"d_U_{u}" for u in UNITS]
            + ["d_delta_qb", "d_starters_out", "d_k_val", "d_qb_epa", "d_qb_epa_long", "d_qb_cpoe",
               "d_qb_sack_rate", "rest_diff", "neutral"],
}


def _xy(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return df.select([pl.col(f).cast(pl.Float64).fill_nan(None).fill_null(0.0) for f in feats]).to_numpy()


class Std:
    def fit(self, X):
        self.m = X.mean(0)
        self.s = X.std(0)
        self.s[self.s < 1e-9] = 1.0
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


def _fit_logit(X, y, C, boost_first=False):
    X = X.copy()
    if boost_first:
        X[:, 0] *= SPREAD_BOOST
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def _pred_logit(m, X, boost_first=False):
    X = X.copy()
    if boost_first:
        X[:, 0] *= SPREAD_BOOST
    return m.predict_proba(X)[:, 1]


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_C(tr: pl.DataFrame, feats: list[str], boost_first: bool) -> float:
    seasons = sorted(tr["season"].unique().to_list())
    val_seasons = seasons[-3:]
    best, bestC = 1e9, C_GRID[len(C_GRID) // 2]
    for C in C_GRID:
        lls = []
        for vs in val_seasons:
            a, b = tr.filter(pl.col("season") < vs), tr.filter(pl.col("season") == vs)
            if a.height < 300:
                continue
            sc = Std().fit(_xy(a, feats))
            m = _fit_logit(sc(_xy(a, feats)), a["home_win"].to_numpy(), C, boost_first)
            lls.append(_ll(_pred_logit(m, sc(_xy(b, feats)), boost_first), b["home_win"].to_numpy()))
        v = float(np.mean(lls)) if lls else 1e9
        if v < best - 1e-6:
            best, bestC = v, C
    return bestC


def choose_alpha(tr: pl.DataFrame, feats: list[str], target: str, intercept: bool) -> float:
    seasons = sorted(tr["season"].unique().to_list())
    best, bestA = 1e18, A_GRID[-1]
    for A in A_GRID:
        err = []
        for vs in seasons[-3:]:
            a, b = tr.filter(pl.col("season") < vs), tr.filter(pl.col("season") == vs)
            if a.height < 300:
                continue
            sc = Std().fit(_xy(a, feats))
            m = Ridge(alpha=A, fit_intercept=intercept).fit(sc(_xy(a, feats)), a[target].to_numpy())
            err.append(np.mean((m.predict(sc(_xy(b, feats))) - b[target].to_numpy()) ** 2))
        v = float(np.mean(err)) if err else 1e18
        if v < best - 1e-9:
            best, bestA = v, A
    return bestA


def walk_forward(g: pl.DataFrame, feats: list[str], kind: str, test_seasons: list[int],
                 first_train: int = FIRST_TRAIN) -> tuple[pl.DataFrame, list[dict]]:
    """kind: 'mkt' or 'nomkt'. Returns predictions for the test seasons and per-fold coefficient records."""
    out, recs = [], []
    done = g.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null() & (pl.col("result") != 0))
    for S in test_seasons:
        tr = done.filter(pl.col("season").is_between(first_train, S - 1))
        te = g.filter((pl.col("season") == S) & pl.col("spread_line").is_not_null())
        if te.height == 0:
            continue
        if kind == "mkt":
            sf = ["spread_line"] + feats
            C = choose_C(tr, sf, True)
            sc = Std().fit(_xy(tr, sf))
            m = _fit_logit(sc(_xy(tr, sf)), tr["home_win"].to_numpy(), C, True)
            p = _pred_logit(m, sc(_xy(te, sf)), True)
            A = choose_alpha(tr, feats, "ats_resid", False)
            sc2 = Std().fit(_xy(tr, feats))
            r = Ridge(alpha=A, fit_intercept=False).fit(sc2(_xy(tr, feats)), tr["ats_resid"].to_numpy())
            adj = r.predict(sc2(_xy(te, feats)))
            margin = te["spread_line"].to_numpy() + adj
            pc = norm.cdf(adj / SIGMA)
            coef = dict(zip(sf, (m.coef_[0] * np.r_[SPREAD_BOOST, np.ones(len(feats))]).tolist()))
            recs.append({"season": S, "C": C, "alpha": A, "su": coef, "ats": dict(zip(feats, r.coef_.tolist())),
                         "sd": dict(zip(sf, sc.s.tolist()))})
        else:
            C = choose_C(tr, feats, False)
            sc = Std().fit(_xy(tr, feats))
            m = _fit_logit(sc(_xy(tr, feats)), tr["home_win"].to_numpy(), C, False)
            p = _pred_logit(m, sc(_xy(te, feats)), False)
            A = choose_alpha(tr, feats, "result", True)
            r = Ridge(alpha=A, fit_intercept=True).fit(sc(_xy(tr, feats)), tr["result"].to_numpy())
            margin = r.predict(sc(_xy(te, feats)))
            pc = norm.cdf((margin - te["spread_line"].to_numpy()) / SIGMA)
            recs.append({"season": S, "C": C, "alpha": A, "su": dict(zip(feats, m.coef_[0].tolist())),
                         "margin": dict(zip(feats, r.coef_.tolist())), "sd": dict(zip(feats, sc.s.tolist()))})
        out.append(te.select("game_id", "season", "week").with_columns(
            pl.Series("p_home", p), pl.Series("margin", margin), pl.Series("p_home_cover", pc)))
    return pl.concat(out), recs


def spread_only(g: pl.DataFrame, test_seasons: list[int], first_train: int = FIRST_TRAIN) -> pl.DataFrame:
    """Reference: walk-forward logistic on the spread alone (same folds)."""
    out = []
    done = g.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null() & (pl.col("result") != 0))
    for S in test_seasons:
        tr = done.filter(pl.col("season").is_between(first_train, S - 1))
        te = g.filter((pl.col("season") == S) & pl.col("spread_line").is_not_null())
        m = LogisticRegression(C=1e6, max_iter=2000).fit(_xy(tr, ["spread_line"]), tr["home_win"].to_numpy())
        out.append(te.select("game_id", "season", "week").with_columns(
            pl.Series("p_home", m.predict_proba(_xy(te, ["spread_line"]))[:, 1]),
            pl.Series("margin", te["spread_line"].to_numpy()), pl.Series("p_home_cover", np.full(te.height, 0.5))))
    return pl.concat(out)


def gbm_walk_forward(g: pl.DataFrame, feats: list[str], test_seasons: list[int], first_train: int = FIRST_TRAIN,
                     params: dict | None = None) -> pl.DataFrame:
    """LightGBM residual models: SU with init_score = spread logit (from a spread-only logistic fit on the same
    training seasons); ATS regression on (result - spread). Dev comparison only."""
    import lightgbm as lgb
    params = params or dict(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=100,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, random_state=7,
                            n_jobs=2, verbose=-1)
    out = []
    done = g.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null() & (pl.col("result") != 0))
    for S in test_seasons:
        tr = done.filter(pl.col("season").is_between(first_train, S - 1))
        te = g.filter((pl.col("season") == S) & pl.col("spread_line").is_not_null())
        base = LogisticRegression(C=1e6, max_iter=2000).fit(_xy(tr, ["spread_line"]), tr["home_win"].to_numpy())
        z_tr = base.decision_function(_xy(tr, ["spread_line"]))
        z_te = base.decision_function(_xy(te, ["spread_line"]))
        Xtr, Xte = _xy(tr, feats), _xy(te, feats)
        clf = lgb.LGBMClassifier(**params).fit(Xtr, tr["home_win"].to_numpy(), init_score=z_tr)
        p = 1 / (1 + np.exp(-(z_te + clf.predict(Xte, raw_score=True))))
        reg = lgb.LGBMRegressor(**params).fit(Xtr, tr["ats_resid"].to_numpy())
        adj = reg.predict(Xte)
        out.append(te.select("game_id", "season", "week").with_columns(
            pl.Series("p_home", p), pl.Series("margin", te["spread_line"].to_numpy() + adj),
            pl.Series("p_home_cover", norm.cdf(adj / SIGMA))))
    return pl.concat(out)


def score(pred: pl.DataFrame, g: pl.DataFrame, lo: int, hi: int) -> dict:
    """Same rules as research/eval.py, for any season window."""
    x = g.filter(pl.col("season").is_between(lo, hi) & pl.col("result").is_not_null()
                 & pl.col("spread_line").is_not_null()).select("game_id", "result", "spread_line").join(
        pred.select("game_id", "p_home", "margin", "p_home_cover"), on="game_id", how="inner")
    su = x.filter(pl.col("result") != 0)
    y = (su["result"] > 0).to_numpy()
    p = su["p_home"].to_numpy()
    acc = float(((p >= 0.5) == y).mean())
    mkt = float(np.where(su["spread_line"].to_numpy() >= 0, y, ~y).mean())
    at = x.filter((pl.col("result") - pl.col("spread_line")) != 0)
    cov = ((at["result"] - at["spread_line"]) > 0).to_numpy()
    ats = float(((at["p_home_cover"].to_numpy() >= 0.5) == cov).mean())
    return dict(win=f"{lo}-{hi}", n=su.height, su=round(acc, 4), mkt=round(mkt, 4), ll=round(_ll(p, y.astype(float)), 4),
                brier=round(float(np.mean((p - y) ** 2)), 4), ats=round(ats, 4), ats_n=at.height)

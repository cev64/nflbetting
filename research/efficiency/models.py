"""Walk-forward models on the game-level design matrix."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 7
SIGMA = 13.5  # sd of NFL margin around expectation

# --- feature sets -------------------------------------------------------------------------------------------
EFF_CORE = ["net_epa_adj", "net_pass_epa_adj", "net_rush_epa_adj", "net_sr_adj", "net_early_epa_adj", "net_ppd_adj",
            "d_margin_adj", "d_qb_epa"]
EFF_EXTRA = ["d_off_cpoe", "d_def_cpoe", "d_off_sack_rate", "d_def_sack_rate", "d_off_pressure_rate",
             "d_def_pressure_rate", "d_off_to_luck_adj", "d_def_to_luck_adj", "d_off_int_rate", "d_def_int_rate",
             "d_off_explosive", "d_def_explosive", "d_off_third_conv", "d_def_third_conv", "d_off_rz_td",
             "d_def_rz_td", "d_off_proe", "d_off_plays", "d_st_epa", "d_pen_yds"]
CONTEXT = ["rest_diff", "home_short", "away_short", "home_bye", "away_bye", "div", "neutral", "indoor", "turf",
           "temp_f", "wind_f", "primetime", "playoff", "h_qb_change", "a_qb_change", "d_qb_plays", "travel_mi",
           "tz_shift", "west_early", "min_games"]
MARKET = ["spread_line", "total_line"]


class Prep:
    """median-impute + standardize, fitted on the training fold only."""

    def fit(self, X: pd.DataFrame):
        self.med = X.median()
        self.sc = StandardScaler().fit(X.fillna(self.med))
        return self

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        return self.sc.transform(X.fillna(self.med))


def walk_forward(df: pd.DataFrame, fitpred, first: int = 2006, last: int = 2026, train_from: int = 2002,
                 split_week: int | None = None, train_window: int | None = None) -> pd.DataFrame:
    """For each test season S (optionally split into weeks < split_week and >= split_week), fit on all completed
    games before the chunk and predict the chunk's games that have a spread_line."""
    out = []
    for S in range(first, last + 1):
        chunks = [(0, 99)] if split_week is None else [(0, split_week - 1), (split_week, 99)]
        for lo, hi in chunks:
            lo_s = train_from if train_window is None else max(train_from, S - train_window)
            tr = df[(df.season >= lo_s) & ((df.season < S) | ((df.season == S) & (df.week < lo)))
                    & df.result.notna() & df.spread_line.notna()]
            te = df[(df.season == S) & df.week.between(lo, hi) & df.spread_line.notna()]
            if len(te) == 0:
                continue
            p = fitpred(tr, te)
            p.insert(0, "game_id", te.game_id.values)
            out.append(p)
    return pd.concat(out, ignore_index=True)


# --- model families ------------------------------------------------------------------------------------------
def logit_fp(features: list[str], C: float, resid_features: list[str] | None = None, resid_alpha: float = 2000.0,
             margin_alpha: float = 100.0):
    """L2-logistic for P(home win) on standardized features.
    margin: if resid_features is given -> spread_line + ridge prediction of (result - spread_line), i.e. a model of
    the market's error; otherwise a ridge regression of the result on `features`."""

    def f(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
        trw = tr[tr.result != 0]
        prep = Prep().fit(tr[features])
        clf = LogisticRegression(C=C, max_iter=2000).fit(prep(trw[features]), trw.y_win.astype(int))
        p = clf.predict_proba(prep(te[features]))[:, 1]
        if resid_features:
            prep2 = Prep().fit(tr[resid_features])
            # no intercept: the market is assumed unbiased on average; only feature-driven deviations move the line
            rg = Ridge(alpha=resid_alpha, fit_intercept=False).fit(prep2(tr[resid_features]), tr.resid)
            r = rg.predict(prep2(te[resid_features]))
            margin = te.spread_line.values + r
        else:
            rg = Ridge(alpha=margin_alpha).fit(prep(tr[features]), tr.result)
            margin = rg.predict(prep(te[features]))
        pc = norm.cdf((margin - te.spread_line.values) / SIGMA)
        return pd.DataFrame({"p_home": p, "margin": margin, "p_home_cover": pc})

    return f


LGB_CLS = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=80, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, n_jobs=2, random_state=SEED, verbose=-1)
LGB_REG = dict(objective="regression", learning_rate=0.02, num_leaves=7, min_child_samples=80, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, n_jobs=2, random_state=SEED, verbose=-1)


def gbm_fp(features: list[str], n_cls: int, n_reg: int, cls_params=None, reg_params=None, resid_features=None):
    """LightGBM classifier on win (market as a feature) + LightGBM regressor on the residual result - spread."""
    cp = {**LGB_CLS, **(cls_params or {})}
    rp = {**LGB_REG, **(reg_params or {})}
    rfeats = resid_features or [c for c in features if c not in MARKET] + ["spread_line"]

    def f(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
        trw = tr[tr.result != 0]
        clf = lgb.LGBMClassifier(n_estimators=n_cls, **cp).fit(trw[features], trw.y_win.astype(int))
        p = clf.predict_proba(te[features])[:, 1]
        reg = lgb.LGBMRegressor(n_estimators=n_reg, **rp).fit(tr[rfeats], tr.resid)
        r = reg.predict(te[rfeats])
        return pd.DataFrame({"p_home": p, "margin": te.spread_line.values + r, "p_home_cover": norm.cdf(r / SIGMA)})

    return f


def xgb_margin_fp(features: list[str], n: int = 300, params: dict | None = None):
    """XGBoost regressor on the home margin from efficiency features only (no market) -> diverse signal.
    p_home = Phi(margin / s), s = residual sd on the training fold."""
    import xgboost as xgb
    pr = dict(learning_rate=0.03, max_depth=3, min_child_weight=50, subsample=0.8, colsample_bytree=0.6,
              reg_lambda=10.0, n_jobs=2, random_state=SEED, **(params or {}))

    def f(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
        m = xgb.XGBRegressor(n_estimators=n, **pr).fit(tr[features], tr.result)
        s = float(np.std(tr.result - m.predict(tr[features])))
        mt = m.predict(te[features])
        return pd.DataFrame({"p_home": norm.cdf(mt / s), "margin": mt,
                             "p_home_cover": norm.cdf((mt - te.spread_line.values) / s)})

    return f

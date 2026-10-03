"""Walk-forward models on market + situational features.

situational : regularised logistic regression (SU) + separate logistic regression for the cover, whose
              features are chosen by an era-stability screen run inside each training window only.
mlp         : small sklearn MLPs (seed ensemble) for P(home win) and for the residual result - spread_line.
knn         : similar-games model (k nearest neighbours in a standardised situational space).

For test season S every model is fit on seasons 1999..S-1 only (all hyper-parameters are either fixed a priori
or tuned by rolling-origin validation inside 1999..S-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

import angles
import features as F

FIRST_TRAIN = 1999
RESID_SD = 13.5  # sd of (result - spread_line); used only to map an expected residual to a cover probability


# ----------------------------------------------------------------------------------------------- helpers
def add_angle_columns(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = []
    for k, (_, x) in angles.angle_defs(g).items():
        g["ang_" + k] = x.fillna(0).astype(float)
        cols.append("ang_" + k)
    return g, cols


def design(g: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return g[cols].astype(float).fillna(0.0).to_numpy()


class Std:
    """Standardiser that can up-weight the market column so L2 shrinkage barely touches it."""

    def __init__(self, cols, boost=None):
        self.cols, self.boost = cols, boost or {}

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.w = np.array([self.boost.get(c, 1.0) for c in self.cols])
        return self

    def transform(self, X):
        return self.sc.transform(X) * self.w


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ----------------------------------------------------------------------- ATS feature stability screen
def select_ats_features(tr: pd.DataFrame, pool: list[str], n_blocks: int = 4, z_min: float = 1.5) -> list[str]:
    """Keep a candidate only if its univariate relation with home_cover has the same sign in every block of
    the training window AND the pooled |z| >= z_min. Uses training seasons only."""
    d = tr[tr["home_cover"].notna()]
    seasons = np.sort(d["season"].unique())
    blocks = np.array_split(seasons, n_blocks)
    y = d["home_cover"].to_numpy() - 0.5
    keep = []
    for c in pool:
        x = d[c].astype(float).fillna(0).to_numpy()
        if x.std() == 0:
            continue
        xs = (x - x.mean()) / x.std()
        signs = []
        for b in blocks:
            m = d["season"].isin(b).to_numpy()
            if xs[m].std() == 0:
                signs.append(0)
                continue
            signs.append(np.sign(np.corrcoef(xs[m], y[m])[0, 1]))
        r = np.corrcoef(xs, y)[0, 1]
        z = r * np.sqrt(len(xs))
        if abs(z) >= z_min and (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            keep.append(c)
    return keep


def fit_plogit(X, y, off, lam):
    """L2-penalised logistic regression with an (unpenalised) intercept and an unpenalised slope on `off`
    (the market logit).  Returns params [b0, a, beta...]."""
    n, k = X.shape

    def f(w):
        z = w[0] + w[1] * off + X @ w[2:]
        ll = np.sum(np.logaddexp(0, z) - y * z)
        p = 1 / (1 + np.exp(-z))
        r = p - y
        grad = np.concatenate([[r.sum(), r @ off], X.T @ r + lam * w[2:]])
        return ll + 0.5 * lam * w[2:] @ w[2:], grad

    w0 = np.zeros(k + 2)
    w0[1] = 1.0 if np.any(off) else 0.0
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


def pred_plogit(w, X, off):
    return 1 / (1 + np.exp(-(w[0] + w[1] * off + X @ w[2:])))


def tune_lam(tr: pd.DataFrame, cols: list[str], target: str, lams, use_off=True, n_val: int = 4) -> float:
    """Rolling-origin inside the training window: for each of its last n_val seasons, fit on the seasons before
    it and score log-loss on it.  Returns the penalty with the best mean log-loss."""
    seasons = np.sort(tr["season"].unique())
    best, best_ll = lams[-1], np.inf
    folds = []
    for v in seasons[-n_val:]:
        a = tr[(tr["season"] < v) & tr[target].notna()]
        b = tr[(tr["season"] == v) & tr[target].notna()]
        sc = StandardScaler().fit(design(a, cols))
        folds.append((sc.transform(design(a, cols)), a[target].to_numpy(), a["mkt_logit"].to_numpy() * use_off,
                      sc.transform(design(b, cols)), b[target].to_numpy(), b["mkt_logit"].to_numpy() * use_off))
    for lam in lams:
        lls = [logloss(yb, pred_plogit(fit_plogit(Xa, ya, oa, lam), Xb, ob)) for Xa, ya, oa, Xb, yb, ob in folds]
        if np.mean(lls) < best_ll:
            best, best_ll = lam, np.mean(lls)
    return best


def screen_su_features(tr: pd.DataFrame, pool: list[str], n_blocks: int = 4, z_min: float = 2.0) -> list[str]:
    """Same stability screen as for ATS, but on the SU residual (home_win - market probability)."""
    d = tr[tr["home_win"].notna()].copy()
    d["su_res"] = d["home_win"] - d["mkt_prob"]
    seasons = np.sort(d["season"].unique())
    blocks = np.array_split(seasons, n_blocks)
    y = d["su_res"].to_numpy()
    keep = []
    for c in pool:
        x = d[c].astype(float).fillna(0).to_numpy()
        if x.std() == 0:
            continue
        signs = []
        for b in blocks:
            m = d["season"].isin(b).to_numpy()
            signs.append(np.sign(np.corrcoef(x[m], y[m])[0, 1]) if x[m].std() > 0 else 0)
        z = np.corrcoef(x, y)[0, 1] * np.sqrt(len(x))
        if abs(z) >= z_min and (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            keep.append(c)
    return keep


# ------------------------------------------------------------------------------------- the models
SU_POOL = [c for c in F.MARKET_FEATS if c != "mkt_logit"] + F.SITU_FEATS
LAMS = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


def run_situational(g: pd.DataFrame, test_seasons, log=print):
    g, ang_cols = add_angle_columns(g)
    ats_pool = SU_POOL + ang_cols
    out, chosen = [], {}
    for S in test_seasons:
        tr = g[(g["season"] < S) & (g["season"] >= FIRST_TRAIN) & g["result"].notna() & g["spread_line"].notna()]
        te = g[(g["season"] == S) & g["spread_line"].notna()]
        if te.empty:
            continue
        # --- SU: unpenalised market logit offset + stability-screened situational features (L2, tuned)
        trs = tr[tr["home_win"].notna()]
        su_sel = screen_su_features(trs, SU_POOL) or ["neutral"]
        lam_su = tune_lam(trs, su_sel, "home_win", LAMS)
        sc = StandardScaler().fit(design(trs, su_sel))
        w = fit_plogit(sc.transform(design(trs, su_sel)), trs["home_win"].to_numpy(), trs["mkt_logit"].to_numpy(), lam_su)
        p_home = pred_plogit(w, sc.transform(design(te, su_sel)), te["mkt_logit"].to_numpy())
        # --- ATS: stability-screened features only, no market offset (cover is already market-relative)
        sel = select_ats_features(tr, ats_pool)
        trc = tr[tr["home_cover"].notna()]
        if sel:
            lam_c = tune_lam(trc, sel, "home_cover", LAMS, use_off=False)
            scc = StandardScaler().fit(design(trc, sel))
            wc = fit_plogit(scc.transform(design(trc, sel)), trc["home_cover"].to_numpy(), np.zeros(len(trc)), lam_c)
            p_cover = pred_plogit(wc, scc.transform(design(te, sel)), np.zeros(len(te)))
            coefs = dict(zip(sel, wc[2:]))
        else:
            lam_c, p_cover, coefs = None, np.full(len(te), 0.5), {}
        resid_hat = norm.ppf(np.clip(p_cover, 0.01, 0.99)) * RESID_SD
        chosen[S] = dict(su=dict(zip(su_sel, w[2:])), su_mkt_slope=w[1], lam_su=lam_su, ats=coefs, lam_ats=lam_c)
        log(f"  situational {S}: su={su_sel} lam={lam_su} | ats({len(sel)}) lam={lam_c}")
        out.append(pd.DataFrame({"game_id": te["game_id"].values, "season": S, "week": te["week"].values,
                                 "p_home": p_home, "margin": te["spread_line"].values + resid_hat,
                                 "p_home_cover": p_cover}))
    return pd.concat(out, ignore_index=True), chosen


MLP_COLS = ["mkt_logit", "spread_line"] + [c for c in F.MARKET_FEATS if c != "mkt_logit"] + F.SITU_FEATS
MLP_SEEDS = (0, 1, 2, 3, 4)
MLP_CFG = dict(hidden_layer_sizes=(16, 8), alpha=1.0, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
               validation_fraction=0.15, n_iter_no_change=15)


def run_mlp(g: pd.DataFrame, test_seasons, cfg=None, log=print):
    cfg = cfg or MLP_CFG
    out = []
    for S in test_seasons:
        tr = g[(g["season"] < S) & (g["season"] >= FIRST_TRAIN) & g["result"].notna() & g["spread_line"].notna()]
        te = g[(g["season"] == S) & g["spread_line"].notna()]
        if te.empty:
            continue
        trs = tr[tr["home_win"].notna()]
        sc = StandardScaler().fit(design(tr, MLP_COLS))
        Xs, Xr, Xt = sc.transform(design(trs, MLP_COLS)), sc.transform(design(tr, MLP_COLS)), sc.transform(design(te, MLP_COLS))
        yr = (tr["resid"].clip(-35, 35) / RESID_SD).to_numpy()
        ph, rh = [], []
        for seed in MLP_SEEDS:
            c = MLPClassifier(random_state=seed, **cfg).fit(Xs, trs["home_win"].astype(int))
            ph.append(c.predict_proba(Xt)[:, 1])
            r = MLPRegressor(random_state=seed, **cfg).fit(Xr, yr)
            rh.append(r.predict(Xt) * RESID_SD)
        p_home, resid_hat = np.mean(ph, axis=0), np.mean(rh, axis=0)
        p_cover = norm.cdf(resid_hat / RESID_SD)
        log(f"  mlp {S}: mean|resid_hat|={np.abs(resid_hat).mean():.2f}")
        out.append(pd.DataFrame({"game_id": te["game_id"].values, "season": S, "week": te["week"].values,
                                 "p_home": p_home, "margin": te["spread_line"].values + resid_hat,
                                 "p_home_cover": p_cover}))
    return pd.concat(out, ignore_index=True)


KNN_COLS = ["spread_line", "total_line_f", "rest_diff", "travel_diff", "tz_diff", "div_game", "prime",
            "avg_margin_diff", "avg_cover_m_diff", "l3_cover_m_diff", "last_cover_m_diff", "ps_margin_diff",
            "week_f", "qb_exp_diff", "wind_f", "temp_f"]
KNN_W = {"spread_line": 3.0, "total_line_f": 1.5}  # spread dominates similarity
KNN_K = 300


def run_knn(g: pd.DataFrame, test_seasons, log=print):
    out = []
    for S in test_seasons:
        tr = g[(g["season"] < S) & (g["season"] >= FIRST_TRAIN) & g["result"].notna() & g["spread_line"].notna()]
        te = g[(g["season"] == S) & g["spread_line"].notna()]
        if te.empty:
            continue
        sc = Std(KNN_COLS, KNN_W).fit(design(tr, KNN_COLS))
        nn = NearestNeighbors(n_neighbors=KNN_K).fit(sc.transform(design(tr, KNN_COLS)))
        dist, idx = nn.kneighbors(sc.transform(design(te, KNN_COLS)))
        res = tr["resid"].clip(-30, 30).to_numpy()[idx]
        hw = tr["home_win"].fillna(0.5).to_numpy()[idx]
        hc = tr["home_cover"].fillna(0.5).to_numpy()[idx]
        w = 1.0 / (1.0 + dist)
        resid_hat = (res * w).sum(1) / w.sum(1)
        # shrink: neighbour sets are noisy, pull 60% back toward market
        resid_hat *= 0.4
        p_cover = 0.5 + 0.4 * ((hc * w).sum(1) / w.sum(1) - 0.5)
        mkt = te["mkt_prob"].to_numpy()
        p_nb = (hw * w).sum(1) / w.sum(1)
        p_home = np.clip(0.5 * mkt + 0.5 * p_nb + 0.0, 0.01, 0.99)
        out.append(pd.DataFrame({"game_id": te["game_id"].values, "season": S, "week": te["week"].values,
                                 "p_home": p_home, "margin": te["spread_line"].values + resid_hat,
                                 "p_home_cover": p_cover}))
    return pd.concat(out, ignore_index=True)

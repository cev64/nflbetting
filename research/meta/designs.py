"""Meta-learner designs. Each SU design: fit(train, cfg) -> predictor(test) -> p_home.
Each ATS design: same, returning p_home_cover. `train` only ever contains seasons < test season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data import NO_MARKET, family, logit, su_resid, cover_signal

CTX = ["pickem", "early", "late", "playoff"]
MKT_SCALE = 10.0  # market column multiplied so the L2 penalty barely touches it


def sample_weight(seasons: np.ndarray, test_season: int, half_life: float | None) -> np.ndarray:
    if not half_life:
        return np.ones(len(seasons))
    return 0.5 ** ((test_season - 1 - seasons) / half_life)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- SU feature builders
def fam_resids(g: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    fams: dict[str, list[np.ndarray]] = {}
    for m in models:
        fams.setdefault(family(m), []).append(su_resid(g, m))
    return pd.DataFrame({f"r_{k}": np.mean(v, 0) for k, v in sorted(fams.items())}, index=g.index)


def su_matrix(g, models, cfg):
    cols = {"mkt": g["l_mkt"].to_numpy() / MKT_SCALE}
    if cfg.get("ml"):
        cols["ml_gap"] = g["ml_gap"].to_numpy()
    if cfg.get("by") == "family":
        R = fam_resids(g, models)
    else:
        R = pd.DataFrame({f"r_{m}": su_resid(g, m) for m in models}, index=g.index)
    for c in R.columns:
        cols[c] = R[c].to_numpy()
    if cfg.get("ctx"):
        for k in CTX:
            z = g[k].to_numpy()
            cols[f"{k}"] = z
            cols[f"mkt_x_{k}"] = cols["mkt"] * z
            for c in R.columns:
                cols[f"{c}_x_{k}"] = R[c].to_numpy() * z
    if cfg.get("dis"):
        cols["nm_sd"] = g["nm_sd"].to_numpy()
        cols["fam_dis_x_resid"] = g["fam_dissent"].to_numpy() * g["nm_resid_mean"].to_numpy()
    return pd.DataFrame(cols, index=g.index)


class Const:
    def __init__(self, p): self.p = p
    def __call__(self, g): return self.p(g)


# ---------------------------------------------------------------- SU designs
def su_market(tr, models, cfg, S):
    w = sample_weight(tr.season.values, S, cfg.get("hl"))
    X = tr[["l_mkt"]].to_numpy()
    clf = LogisticRegression(C=1e4, max_iter=2000).fit(X, tr.home_win, sample_weight=w)
    return lambda te: clf.predict_proba(te[["l_mkt"]].to_numpy())[:, 1], {"coef": clf.coef_[0].tolist()}


def su_logit(tr, models, cfg, S):
    """L2 logistic stack. cfg: C, by ('model'|'family'), ctx (interactions), dis, ml, hl."""
    w = sample_weight(tr.season.values, S, cfg.get("hl"))
    X = su_matrix(tr, models, cfg)
    clf = LogisticRegression(C=cfg.get("C", 0.1), max_iter=5000).fit(X.to_numpy(), tr.home_win, sample_weight=w)
    info = {"features": list(X.columns), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    return lambda te: clf.predict_proba(su_matrix(te, models, cfg).to_numpy())[:, 1], info


def su_lgbm(tr, models, cfg, S):
    """LightGBM boosting from a fitted market offset (init_score): trees only learn corrections."""
    import lightgbm as lgb
    w = sample_weight(tr.season.values, S, cfg.get("hl"))
    base = LogisticRegression(C=1e4, max_iter=2000).fit(tr[["l_mkt"]].to_numpy(), tr.home_win, sample_weight=w)
    def off(d): return base.decision_function(d[["l_mkt"]].to_numpy())
    def feats(d):
        F = fam_resids(d, models)
        for c in ["abs_spread", "l_mkt", "week", "playoff", "last_reg_week", "ml_gap", "total_line",
                  "fam_dissent", "nm_sd", "nm_resid_mean", "dissent_frac", "rest_diff"]:
            F[c] = d[c].to_numpy()
        return F
    params = dict(objective="binary", learning_rate=cfg.get("lr", 0.02), num_leaves=cfg.get("leaves", 4),
                  max_depth=cfg.get("depth", 2), min_data_in_leaf=cfg.get("min_leaf", 150),
                  lambda_l2=cfg.get("l2", 10.0), feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  seed=7, deterministic=True, force_row_wise=True, num_threads=2, verbose=-1)
    ds = lgb.Dataset(feats(tr), tr.home_win.to_numpy(), weight=w, init_score=off(tr))
    bst = lgb.train(params, ds, num_boost_round=cfg.get("n", 100))
    imp = dict(zip(bst.feature_name(), bst.feature_importance("gain").round(2).tolist()))
    return lambda te: sigmoid(off(te) + bst.predict(feats(te), raw_score=True)), {"gain": imp}


def su_bma(tr, models, cfg, S):
    """Bayesian-model-averaging-style linear pool: weight_m ∝ exp(tau * decayed log-likelihood_m),
    computed separately per context cell (cfg['cells']: list of context column names, or [])."""
    hl, tau = cfg.get("hl", 4), cfg.get("tau", 0.05)
    cells = cfg.get("cells", [])
    cand = ["__market__"] + models
    base = LogisticRegression(C=1e4, max_iter=2000).fit(tr[["l_mkt"]].to_numpy(), tr.home_win)
    def probs(d):
        P = {"__market__": base.predict_proba(d[["l_mkt"]].to_numpy())[:, 1]}
        for m in models:
            P[m] = d[f"{m}__p_home"].to_numpy(float)
        return np.column_stack([np.clip(P[m], 1e-4, 1 - 1e-4) for m in cand])
    def key(d):
        return d[cells].astype(int).astype(str).agg("".join, axis=1) if cells else pd.Series("all", index=d.index)
    w = sample_weight(tr.season.values, S, hl)
    P, y, k = probs(tr), tr.home_win.to_numpy(), key(tr)
    ll = y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P)
    W = {}
    for kk in k.unique():
        msk = (k == kk).to_numpy()
        s = (ll[msk] * w[msk, None]).sum(0)
        s = tau * (s - s.max())
        ww = np.exp(s); W[kk] = ww / ww.sum()
    glob = np.exp(tau * ((ll * w[:, None]).sum(0) - (ll * w[:, None]).sum(0).max())); glob /= glob.sum()
    def pred(te):
        Pt, kt = probs(te), key(te)
        Wt = np.vstack([W.get(x, glob) for x in kt])
        return (Pt * Wt).sum(1)
    return pred, {"weights": {kk: dict(zip(cand, np.round(v, 4).tolist())) for kk, v in W.items()}}


def su_router(tr, models, cfg, S):
    """Disagreement router: pick the market favorite unless >= k diverse no-market families dissent AND the
    dissenting side has been right > 50% in that context in training (posterior P(acc>0.5) >= conf)."""
    from scipy.stats import beta
    k_req, conf = cfg.get("k", 2), cfg.get("conf", 0.8)
    base = LogisticRegression(C=1e4, max_iter=2000).fit(tr[["l_mkt"]].to_numpy(), tr.home_win)
    def cell(d): return np.where(d["pickem"] == 1, "pk", np.where(d["bucket"] == 1, "b3", "big"))
    fav_home_tr = tr.spread_line.values >= 0
    dis_right = ((tr.home_win.values == 1) != fav_home_tr)
    c_tr = cell(tr)
    rules = {}
    for c in ["pk", "b3", "big"]:
        for k in range(k_req, int(tr.n_fams.max()) + 1):
            msk = (c_tr == c) & (tr.fam_dissent.values >= k)
            n, wins = int(msk.sum()), int(dis_right[msk].sum())
            pr = 1 - beta.cdf(0.5, wins + 1, n - wins + 1) if n else 0.0
            rules[(c, k)] = (n, wins / max(n, 1), pr)
    def pred(te):
        p = base.predict_proba(te[["l_mkt"]].to_numpy())[:, 1]
        c_te, fd = cell(te), te.fam_dissent.values
        out = p.copy()
        for i in range(len(te)):
            best = None
            for k in range(int(fd[i]), k_req - 1, -1):
                r = rules.get((c_te[i], k))
                if r and r[2] >= conf:
                    best = r; break
            if best is not None:
                fav_home = te.spread_line.values[i] >= 0
                acc = best[1]
                out[i] = 1 - acc if fav_home else acc
        return out
    return pred, {"rules": {f"{c}|>={k}": [n, round(a, 3), round(pr, 3)] for (c, k), (n, a, pr) in rules.items()}}


# ---------------------------------------------------------------- ATS
def ats_matrix(g, models, cfg):
    cols = {}
    if cfg.get("structure", True):
        cols["home_dog"] = g["home_dog"].to_numpy()
        cols["spread"] = g["spread_line"].to_numpy() / 7.0
        cols["playoff_dog_home"] = g["playoff"].to_numpy() * (g["home_dog"].to_numpy() - g["road_dog"].to_numpy())
    if cfg.get("inj", True):
        cols["inj_gap"] = g["d_inj_starters_out"].to_numpy() / 3.0
    if cfg.get("signals", "model") == "model":
        for m in models:
            cols[f"c_{m}"] = cover_signal(g, m)
    elif cfg["signals"] == "family":
        fams: dict[str, list] = {}
        for m in models:
            fams.setdefault(family(m), []).append(cover_signal(g, m))
        for k, v in sorted(fams.items()):
            cols[f"c_{k}"] = np.mean(v, 0)
    elif cfg["signals"] == "mean":
        cols["c_mean"] = np.mean([cover_signal(g, m) for m in models], 0)
    if cfg.get("ctx"):
        base = [c for c in cols if c.startswith("c_")] + (["inj_gap"] if "inj_gap" in cols else [])
        for k in ["early", "late", "pickem"]:
            z = g[k].to_numpy()
            for c in base:
                cols[f"{c}_x_{k}"] = cols[c] * z
    return pd.DataFrame(cols, index=g.index)


def ats_logit(tr, models, cfg, S):
    tr = tr[tr.home_cover.notna()]
    w = sample_weight(tr.season.values, S, cfg.get("hl"))
    X = ats_matrix(tr, models, cfg)
    if X.shape[1] == 0:
        return lambda te: np.full(len(te), 0.5), {}
    clf = LogisticRegression(C=cfg.get("C", 0.05), max_iter=5000, fit_intercept=cfg.get("intercept", False)
                             ).fit(X.to_numpy(), tr.home_cover, sample_weight=w)
    info = {"features": list(X.columns), "coef": clf.coef_[0].tolist()}
    return lambda te: clf.predict_proba(ats_matrix(te, models, cfg).to_numpy())[:, 1], info


def ats_lgbm(tr, models, cfg, S):
    import lightgbm as lgb
    tr = tr[tr.home_cover.notna()]
    w = sample_weight(tr.season.values, S, cfg.get("hl"))
    def feats(d):
        F = ats_matrix(d, models, {**cfg, "ctx": False})
        for c in ["abs_spread", "week", "playoff", "total_line", "ml_gap", "rest_diff"]:
            F[c] = d[c].to_numpy()
        return F
    params = dict(objective="binary", learning_rate=0.02, num_leaves=4, max_depth=2,
                  min_data_in_leaf=cfg.get("min_leaf", 200), lambda_l2=10.0, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, seed=7, deterministic=True, force_row_wise=True,
                  num_threads=2, verbose=-1)
    bst = lgb.train(params, lgb.Dataset(feats(tr), tr.home_cover.to_numpy(), weight=w),
                    num_boost_round=cfg.get("n", 100))
    return lambda te: bst.predict(feats(te)), {}


def ats_vote(tr, models, cfg, S):
    """Equal-weight average of cover log-odds (no fitting) + nothing else."""
    def pred(te):
        return sigmoid(np.mean([cover_signal(te, m) for m in models], 0))
    return pred, {}


SU_DESIGNS = {"market": su_market, "logit": su_logit, "lgbm": su_lgbm, "bma": su_bma, "router": su_router}
ATS_DESIGNS = {"logit": ats_logit, "lgbm": ats_lgbm, "vote": ats_vote}

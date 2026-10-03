"""Meta-model data: one row per game with every base model's out-of-sample prediction + pre-game context.

Everything here is known before kickoff: base-model predictions are themselves walk-forward out-of-sample,
context comes from the pre-game line / schedule, and the injury gap is personnel's pre-game report feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

HERE = Path(__file__).resolve().parent.parent
PREDS = HERE / "preds"
CACHE = HERE / "cache"
SD = 13.5

# Model families: used for "diverse family" dissent counting and for the site's explanation.
FAMILIES = {
    "elo": "ratings", "kalman": "ratings", "ratings_ridge": "ratings", "ratings_combo": "ratings",
    "elo_mkt": "ratings_mkt", "kalman_mkt": "ratings_mkt", "ratings_ridge_mkt": "ratings_mkt",
    "ratings_combo_mkt": "ratings_mkt",
    "logit_epa": "efficiency", "xgb_margin": "efficiency",
    "logit_epa_mkt": "efficiency_mkt", "gbm": "efficiency_mkt",
    "personnel_nomkt": "personnel", "personnel": "personnel_mkt", "personnel_ats": "personnel_mkt",
    "situational": "situational", "mlp": "situational", "knn": "situational",
}
# Models that do not see the market line: the only ones that can genuinely "disagree" with it.
NO_MARKET = ["elo", "kalman", "ratings_ridge", "ratings_combo", "logit_epa", "xgb_margin", "personnel_nomkt"]
SKIP = {"market_baseline"}


def family(m: str) -> str:
    if m in FAMILIES:
        return FAMILIES[m]
    if m.startswith("wave2_"):
        return "wave2"
    return "other"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def available_models() -> list[str]:
    out = []
    for f in sorted(PREDS.glob("*.csv")):
        m = f.stem
        if m in SKIP or m.startswith("ensemble") or m.startswith("meta_"):
            continue
        out.append(m)
    return out


def _ml_prob(home_ml, away_ml):
    def imp(ml):
        ml = np.asarray(ml, dtype=float)
        ml = np.where(np.abs(ml) < 100, np.nan, ml)
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ph, pa = imp(home_ml), imp(away_ml)
        out = ph / (ph + pa)
    return np.where((out > 0.01) & (out < 0.99), out, np.nan)


def load(models: list[str] | None = None, start: int = 2006) -> tuple[pd.DataFrame, list[str]]:
    models = models or available_models()
    s = pl.read_parquet(CACHE / "schedules.parquet").filter(
        (pl.col("season") >= start) & pl.col("spread_line").is_not_null())
    g = s.select("game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "result",
                 "spread_line", "total_line", "home_moneyline", "away_moneyline", "div_game",
                 "home_rest", "away_rest", "location").to_pandas()
    used = []
    for m in models:
        f = PREDS / f"{m}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).drop_duplicates("game_id", keep="last")
        d = d[["game_id", "p_home", "margin"] + (["p_home_cover"] if "p_home_cover" in d.columns else [])]
        d = d.rename(columns={c: f"{m}__{c}" for c in d.columns if c != "game_id"})
        g = g.merge(d, on="game_id", how="left")
        used.append(m)
    # A base model with no prediction for a game is treated as agreeing with the market (zero residual).
    p_mkt0 = norm.cdf(g["spread_line"].to_numpy(float) / SD)
    for m in used:
        miss = g[f"{m}__p_home"].isna()
        g[f"{m}__has"] = (~miss).astype(float)
        g.loc[miss, f"{m}__p_home"] = p_mkt0[miss.to_numpy()]
        g.loc[g[f"{m}__margin"].isna(), f"{m}__margin"] = g["spread_line"]
        if f"{m}__p_home_cover" in g.columns:
            g[f"{m}__p_home_cover"] = g[f"{m}__p_home_cover"].fillna(0.5)
    # personnel's pre-game injury gap (report filed before kickoff; 0 before 2009)
    pg = pl.read_parquet(HERE / "personnel" / "work" / "games.parquet").select(
        "game_id", "d_inj_starters_out", "d_qb_epa").to_pandas()
    g = g.merge(pg, on="game_id", how="left")
    g["d_inj_starters_out"] = g["d_inj_starters_out"].fillna(0.0)

    # ---- market + context ----
    sp = g["spread_line"].to_numpy(float)
    g["p_mkt"] = norm.cdf(sp / SD)
    g["l_mkt"] = logit(g["p_mkt"])
    pml = _ml_prob(g["home_moneyline"], g["away_moneyline"])
    g["p_ml"] = np.where(np.isfinite(pml), pml, g["p_mkt"])
    g["ml_gap"] = logit(g["p_ml"]) - g["l_mkt"]
    g["abs_spread"] = np.abs(sp)
    g["bucket"] = pd.cut(g["abs_spread"], [-0.1, 2.5, 3.5, 6.5, 9.5, 99], labels=[0, 1, 2, 3, 4]).astype(int)
    g["pickem"] = (g["abs_spread"] <= 2.5).astype(float)
    g["playoff"] = (g["game_type"] != "REG").astype(float)
    last_reg = g[g.game_type == "REG"].groupby("season")["week"].transform("max")
    g["last_reg_week"] = 0.0
    g.loc[last_reg.index, "last_reg_week"] = (g.loc[last_reg.index, "week"] == last_reg).astype(float)
    g["early"] = ((g["week"] <= 4) & (g["game_type"] == "REG")).astype(float)
    g["late"] = ((g["week"] >= 13) & (g["game_type"] == "REG")).astype(float)
    g["neutral"] = (g["location"] == "Neutral").astype(float)
    g["home_dog"] = (sp < 0).astype(float)
    g["road_dog"] = (sp > 0).astype(float)
    g["rest_diff"] = (g["home_rest"] - g["away_rest"]).clip(-7, 7).fillna(0)

    # ---- agreement / disagreement among models ----
    fav_home = sp >= 0
    L = np.column_stack([logit(g[f"{m}__p_home"]) for m in used])
    g["n_models"] = len(used)
    g["dissent_frac"] = ((L > 0) != fav_home[:, None]).mean(1)
    nm = [m for m in used if m in NO_MARKET or family(m) == "wave2"]
    if nm:
        Ln = np.column_stack([logit(g[f"{m}__p_home"]) for m in nm])
        g["nm_dissent_frac"] = ((Ln > 0) != fav_home[:, None]).mean(1)
        fams = sorted({family(m) for m in nm})
        fam_votes = []
        for fm in fams:
            cols = [i for i, m in enumerate(nm) if family(m) == fm]
            fam_votes.append(Ln[:, cols].mean(1))
        F = np.column_stack(fam_votes)
        g["fam_dissent"] = ((F > 0) != fav_home[:, None]).sum(1)
        g["n_fams"] = len(fams)
        g["nm_resid_mean"] = (Ln - g["l_mkt"].to_numpy()[:, None]).mean(1)
        g["nm_sd"] = Ln.std(1)
    else:
        for c in ["nm_dissent_frac", "fam_dissent", "n_fams", "nm_resid_mean", "nm_sd"]:
            g[c] = 0.0
    g["all_sd"] = L.std(1)

    # ---- targets ----
    g["home_win"] = np.where(g["result"].notna() & (g["result"] != 0), (g["result"] > 0).astype(float), np.nan)
    ats = g["result"] - g["spread_line"]
    g["home_cover"] = np.where(g["result"].notna() & (ats != 0), (ats > 0).astype(float), np.nan)
    g = g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    return g, used


def su_resid(g: pd.DataFrame, m: str) -> np.ndarray:
    """Model's home-win log-odds relative to the market's."""
    return logit(g[f"{m}__p_home"]) - g["l_mkt"].to_numpy()


def cover_signal(g: pd.DataFrame, m: str) -> np.ndarray:
    """Model's cover signal as a log-odds: p_home_cover if given, else (margin - spread)/SD mapped through a normal."""
    c = f"{m}__p_home_cover"
    if c in g.columns and g[c].notna().any():
        p = g[c].fillna(0.5).to_numpy(float)
        if np.nanstd(p) > 1e-6:
            return logit(p)
    e = (g[f"{m}__margin"] - g["spread_line"]).to_numpy(float)
    return logit(norm.cdf(e / SD))

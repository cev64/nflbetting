"""Close-games / market-microstructure family. Single entry point; deterministic; rebuilds everything.

    python research/closegames/run.py                  # ~30 s: preds, factors, ceiling, dev/holdout tables
    python research/closegames/run.py --check-leakage  # + truncation test (features & market as of 2017)

Writes
    research/preds/wave2_market.csv          best market-only model (spread + spread juice, walk-forward logistic)
    research/preds/wave2_close.csv           close games (|spread| <= 3.5) from the close-game specialist,
                                             everything else = wave2_market
    research/preds/wave2_close_factors.parquet
    research/closegames/out/{ceiling_by_season.csv, eval_dev_holdout.csv, specialist_coefs.csv}

All settings below were FROZEN after the dev phase (dev_market.py / dev_close.py, which load seasons <= 2017
only). 2018-2025 is a clean holdout scored once with these settings.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features  # noqa: E402
from market import MARKET_CONFIG, fit_logit, load_schedule, logit, market_features, market_walk_forward  # noqa: E402
from specialist import add_base_deltas, load_base_preds, walk_forward  # noqa: E402

PREDS = HERE.parent / "preds"
OUT = HERE / "out"

# ------------------------------------------------------------------ frozen settings (chosen on <= 2017 only)
COVER_CONFIG = dict(features=("sp10",), half_life=6.0, train_from=1999, C=1.0)
SPECIALIST_CONFIG = dict(features=["d_coach_go4", "d_coach_log_games"], lam=50.0, intercept=False,
                         free_slope=False, close=3.5, train_from=2006, min_train_seasons=3, half_life=None)
FIRST_OUT = 2006          # first season written to the preds files (stacker wants 2006+)
DEV = (2010, 2017)        # dev test seasons used for selection (close-game specialist: 2010+, market: 2009+)
HOLDOUT = (2018, 2025)
SIGMA = 13.5


# ------------------------------------------------------------------ models
def cover_walk_forward(d: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """P(home covers) from a recency-weighted logistic on the spread (captures the dog/home-fav tilt)."""
    d = d.assign(sp10=d["spread"] / 10.0)
    cov = np.where(d["result"].isna() | (d["result"] == d["spread"]), np.nan, (d["result"] > d["spread"]).astype(float))
    out = []
    feats = list(COVER_CONFIG["features"])
    for S in seasons:
        tr = (d["season"].values >= COVER_CONFIG["train_from"]) & (d["season"].values < S) & np.isfinite(cov)
        te = d["season"].values == S
        w = 0.5 ** ((S - 1 - d.loc[tr, "season"].values) / COVER_CONFIG["half_life"])
        m = fit_logit(d.loc[tr, feats].values, cov[tr], COVER_CONFIG["C"], w)
        out.append(pd.DataFrame({"game_id": d.loc[te, "game_id"].values,
                                 "p_home_cover": m.predict_proba(d.loc[te, feats].values)[:, 1]}))
    return pd.concat(out, ignore_index=True)


def build_all(max_season: int | None = None) -> tuple[pd.DataFrame, dict]:
    d = load_schedule(max_season=max_season)
    last = int(d["season"].max())
    seasons = list(range(2000, last + 1))
    mk = market_walk_forward(d, seasons, MARKET_CONFIG)[["game_id", "p_home"]].rename(columns={"p_home": "p_mkt"})
    cv = cover_walk_forward(d, seasons)
    g = d.merge(mk, on="game_id", how="left").merge(cv, on="game_id", how="left")
    g["l_mkt"] = logit(g["p_mkt"].values)
    mf = market_features(d, np.zeros(len(d), bool), dict(MARKET_CONFIG))
    g["p_ml_novig"] = d["p_ml"].values
    g["l_ml_minus_normal"] = mf["l_ml_minus_normal"].values
    feat = features.build(d, max_season)
    g = g.merge(feat.drop(columns=["temp", "wind"]), on="game_id", how="left")
    g = add_base_deltas(g.merge(load_base_preds(g["game_id"]), on="game_id", how="left"))
    sc = SPECIALIST_CONFIG
    sp, coefs = walk_forward(g, list(range(2007, last + 1)), sc["features"], sc["lam"], close=sc["close"],
                             train_from=sc["train_from"], free_slope=sc["free_slope"], half_life=sc["half_life"],
                             min_train_seasons=sc["min_train_seasons"], intercept=sc["intercept"])
    g = g.merge(sp[["game_id", "p_close"]], on="game_id", how="left")
    g["is_close"] = g["spread"].abs() <= sc["close"]
    g["p_final"] = np.where(g["p_close"].notna(), g["p_close"], g["p_mkt"])
    return g, coefs


def write_preds(g: pd.DataFrame) -> None:
    o = g[g["season"] >= FIRST_OUT].copy()
    o["margin"] = o["spread"] + SIGMA * norm.ppf(np.clip(o["p_home_cover"].values, 1e-4, 1 - 1e-4))
    cols = ["game_id", "season", "week", "p_home", "margin", "p_home_cover"]
    m = o.assign(p_home=o["p_mkt"])[cols]
    c = o.assign(p_home=o["p_final"])[cols]
    m.to_csv(PREDS / "wave2_market.csv", index=False, float_format="%.6f")
    c.to_csv(PREDS / "wave2_close.csv", index=False, float_format="%.6f")
    print(f"wrote wave2_market.csv / wave2_close.csv: {len(o)} rows, seasons {o.season.min()}-{o.season.max()}, "
          f"{int(o['result'].isna().sum())} unplayed; specialist used on {int(o['p_close'].notna().sum())} close games")


def write_factors(g: pd.DataFrame) -> None:
    o = g[g["season"] >= FIRST_OUT].copy()
    rows = []

    def add(feature, label, h, a, better, fmt, ht=None, at=None):
        rows.append(pd.DataFrame({"game_id": o["game_id"].values, "feature": feature, "label": label,
                                  "home": np.asarray(h, dtype=float), "away": np.asarray(a, dtype=float),
                                  "home_text": ht if ht is not None else None,
                                  "away_text": at if at is not None else None, "better": better, "fmt": fmt}))

    sp = o["spread"].values
    add("spread", "Point spread (team, betting notation)", -sp, sp, "low", "+0.0;-0.0")
    add("win_prob", "Model win probability (close games: specialist)", o["p_final"], 1 - o["p_final"], "high", "0.0%")
    add("market_prob", "Market-only win probability (spread + juice)", o["p_mkt"], 1 - o["p_mkt"], "high", "0.0%")
    add("ml_prob", "Moneyline win probability (no-vig)", o["p_ml_novig"], 1 - o["p_ml_novig"], "high", "0.0%")
    pc = o["p_cov_juice"].values
    add("spread_juice", "No-vig cover probability from spread juice", pc, 1 - pc, "high", "0.0%")
    add("close_game", "Close game (|spread| <= 3.5): specialist active", o["is_close"].astype(float),
        o["is_close"].astype(float), "high", "0")
    add("coach_go4", "Coach 4th-down go rate vs league (4th & <=3, decayed)", o["h_coach_go4"], o["a_coach_go4"],
        "high", "+0.0%", o["home_coach"].values, o["away_coach"].values)
    add("coach_games", "Coach career games as head coach", np.expm1(o["h_coach_log_games"]),
        np.expm1(o["a_coach_log_games"]), "high", "0")
    add("kicker_fgoe", "Kicker FG% over expected (all, shrunk)", o["h_k_fgoe"], o["a_k_fgoe"], "high", "+0.0%",
        o["h_kicker_name"].values, o["a_kicker_name"].values)
    add("kicker_fgoe_long", "Kicker FG% over expected, 45+ yds (shrunk)", o["h_k_fgoe_long"], o["a_k_fgoe_long"],
        "high", "+0.0%")
    add("two_min_off", "Offense EPA/play, last 2 min of halves", o["h_two_min_off_epa"], o["a_two_min_off_epa"],
        "high", "+0.00")
    add("two_min_def", "Defense EPA/play allowed, last 2 min of halves", o["h_two_min_def_epa"],
        o["a_two_min_def_epa"], "low", "+0.00")
    add("late_close_net", "Net EPA/play in 4th qtr within 8 pts (off - def)",
        o["h_late_close_off_epa"] - o["h_late_close_def_epa"], o["a_late_close_off_epa"] - o["a_late_close_def_epa"],
        "high", "+0.00")
    add("one_score_record", "One-score game win% (decayed; mostly luck)", o["h_close_wpct"], o["a_close_wpct"],
        "high", "0.0%")
    f = pd.concat(rows, ignore_index=True)
    f.to_parquet(PREDS / "wave2_close_factors.parquet", index=False)
    print(f"wrote wave2_close_factors.parquet: {len(f)} rows ({f.feature.nunique()} factors/game)")


# ------------------------------------------------------------------ evaluation
def _metrics(x: pd.DataFrame, pcol: str) -> dict:
    su = x[x["result"].notna() & (x["result"] != 0)]
    y = (su["result"] > 0).values
    p = np.clip(su[pcol].values, 1e-4, 1 - 1e-4)
    acc = (p >= 0.5) == y
    close = su["spread"].abs().values <= 3.5
    pk = su["spread"].abs().values <= 1.5
    fav = np.where(su["spread"].values >= 0, y, ~y)
    ats = x[x["result"].notna() & (x["result"] != x["spread"])]
    cov = (ats["result"] > ats["spread"]).values
    return dict(n=len(su), su=acc.mean(), mkt_fav_su=fav.mean(),
                logloss=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                ats=((ats["p_home_cover"].values >= 0.5) == cov).mean(),
                n_close=int(close.sum()), close_su=acc[close].mean(), close_fav_su=fav[close].mean(),
                close_ll=-np.mean(y[close] * np.log(p[close]) + (1 - y[close]) * np.log(1 - p[close])),
                n_pk=int(pk.sum()), pickem_su=acc[pk].mean(), pickem_fav_su=fav[pk].mean())


def evaluate(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (lo, hi) in {"dev 2010-2017": DEV, "eval 2012-2017 (dev)": (2012, 2017),
                           "HOLDOUT 2018-2025": HOLDOUT, "2012-2025": (2012, 2025)}.items():
        x = g[g["season"].between(lo, hi)]
        for model, col in (("wave2_market", "p_mkt"), ("wave2_close", "p_final")):
            rows.append(dict(window=name, model=model) | _metrics(x, col))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "eval_dev_holdout.csv", index=False)
    return r


def poisson_binomial_tail(q: np.ndarray, k: int) -> float:
    """P(sum of Bernoulli(q_i) >= k), exact DP."""
    dist = np.zeros(len(q) + 1)
    dist[0] = 1.0
    for i, qi in enumerate(q):
        dist[1:i + 2] = dist[1:i + 2] * (1 - qi) + dist[0:i + 1] * qi
        dist[0] *= (1 - qi)
    return float(dist[k:].sum())


def ceiling(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Expected SU accuracy of a perfectly calibrated market-level model, and its chance of 68%."""
    x = g[(g["season"].between(2006, 2025)) & g["result"].notna() & (g["result"] != 0)].copy()
    x["q"] = np.maximum(x["p_mkt"], 1 - x["p_mkt"])
    x["fav_correct"] = np.where(x["spread"] >= 0, x["result"] > 0, x["result"] < 0)
    x["mkt_correct"] = (x["p_mkt"] >= 0.5) == (x["result"] > 0)
    rows = []
    for S, s in x.groupby("season"):
        q = s["q"].values
        n = len(q)
        k68 = int(np.ceil(0.68 * n - 1e-9))
        rows.append(dict(season=S, n=n, expected_acc=q.mean(), sd_acc=np.sqrt((q * (1 - q)).sum()) / n,
                         actual_mkt_model=s["mkt_correct"].mean(), actual_spread_fav=s["fav_correct"].mean(),
                         p_reach_68=poisson_binomial_tail(q, k68),
                         close_share=(s["spread"].abs() <= 3.5).mean(),
                         close_expected=s.loc[s["spread"].abs() <= 3.5, "q"].mean()))
    by = pd.DataFrame(rows)
    by.to_csv(OUT / "ceiling_by_season.csv", index=False)
    pooled = {}
    for lo, hi in ((2012, 2025), (2018, 2025), (2012, 2017)):
        s = x[x["season"].between(lo, hi)]
        q = s["q"].values
        n = len(q)
        pooled[f"{lo}-{hi}"] = dict(n=n, expected_acc=q.mean(), sd=np.sqrt((q * (1 - q)).sum()) / n,
                                    actual=s["mkt_correct"].mean(),
                                    p_reach_68=poisson_binomial_tail(q, int(np.ceil(0.68 * n))),
                                    exp_seasons_68=by[by.season.between(lo, hi)]["p_reach_68"].sum(),
                                    seasons_market_68=int((by[by.season.between(lo, hi)]["actual_spread_fav"] >= 0.68).sum()))
    # How much extra information (beyond the spread) would a model need for an expected 68%?
    # Normal margin model: result = spread + delta + eps. The market misses delta ~ N(0, tau^2); a model that
    # knows delta has residual SD sqrt(sig^2 - tau^2). sig is calibrated so that tau = 0 reproduces the
    # calibrated market's expected accuracy (the NFL margin is peakier than a normal with the raw residual SD).
    s = x[x["season"].between(2012, 2025)]
    mu = s["spread"].values
    zs, ws = np.polynomial.hermite_e.hermegauss(60)
    ws = ws / ws.sum()

    def acc_with(tau, sig):
        se = np.sqrt(sig ** 2 - tau ** 2)
        m = mu[:, None] + tau * zs[None, :]
        return float((norm.cdf(np.abs(m) / se) @ ws).mean())

    target0 = float(s["q"].mean())
    sigs = np.linspace(8, 16, 321)
    a0 = np.array([acc_with(0.0, sg) for sg in sigs])
    sig = float(np.interp(-target0, -a0, sigs))
    taus = np.linspace(0, 7.9, 159)
    accs = np.array([acc_with(t, sig) for t in taus])
    tau68 = float(np.interp(0.68, accs, taus))
    pooled["info_needed"] = dict(raw_resid_sd=float(np.std(s["result"] - s["spread"])), sigma_eff=sig,
                                 acc_tau0=acc_with(0.0, sig), tau_for_68=tau68,
                                 share_of_margin_var=(tau68 / sig) ** 2)
    return by, pooled


def microstructure(g: pd.DataFrame) -> pd.DataFrame:
    """Which side do moneyline / spread juice favour in close games, and does it beat the spread favourite?"""
    x = g[g["result"].notna() & (g["result"] != 0) & g["season"].between(2006, 2025)].copy()
    y = x["result"] > 0
    x["fav"] = np.where(x["spread"] >= 0, y, ~y)
    x["ml"] = np.where(x["p_ml_novig"].fillna(0.5) >= 0.5, y, ~y)
    x["juice"] = np.where(x["p_cov_juice"].fillna(0.5) >= 0.5, y, ~y)
    x["model"] = (x["p_mkt"] >= 0.5) == y
    x["disagree"] = (x["p_ml_novig"] >= 0.5) != (x["spread"] >= 0)
    x["period"] = np.where(x["season"] <= 2017, "2006-2017 (dev)", "2018-2025 (holdout)")
    x["bucket"] = pd.cut(x["spread"].abs(), [-0.1, 1.0, 2.5, 3.5, 6.5, 99], labels=["0-1", "1.5-2.5", "3-3.5", "4-6.5", "7+"])
    r = x[x["p_ml_novig"].notna()].groupby(["period", "bucket"], observed=True).agg(
        n=("fav", "size"), spread_fav=("fav", "mean"), ml_fav=("ml", "mean"), juice_side=("juice", "mean"),
        market_model=("model", "mean"), ml_disagrees=("disagree", "sum")).reset_index()
    r.to_csv(OUT / "microstructure.csv", index=False)
    return r


def data_quality(g: pd.DataFrame) -> dict:
    x = g[g["season"] >= 2006]
    return dict(games_with_spread=len(x), spread_zero=int((x["spread"] == 0).sum()),
                missing_ml=int(x["home_moneyline"].isna().sum()),
                missing_ml_by_season=x[x["home_moneyline"].isna()].groupby("season").size().to_dict(),
                negative_ml_overround=int((x["ml_vig"] <= 0).sum()),
                ml_sign_conflicts_2_5plus=int((((x["spread"] >= 2.5) & (x["p_ml"] < 0.5))
                                               | ((x["spread"] <= -2.5) & (x["p_ml"] > 0.5))).sum()),
                ml_side_differs_from_spread_side=int(((x["p_ml"] >= 0.5) != (x["spread"] >= 0))[x["p_ml"].notna()].sum()))


def check_leakage() -> None:
    """Truncation test: everything for 2017 games must be identical when built from data <= 2017 only."""
    full, _ = build_all(None)
    trunc, _ = build_all(2017)
    cols = ["p_mkt", "p_home_cover", "h_k_fgoe", "a_k_fgoe", "h_k_fgoe_long", "h_coach_go4", "a_coach_go4",
            "h_coach_log_games", "h_two_min_off_epa", "a_late_close_def_epa", "h_close_wpct", "p_final"]
    a = full[full["season"] == 2017].set_index("game_id")[cols]
    b = trunc[trunc["season"] == 2017].set_index("game_id").loc[a.index, cols]
    diff = (a - b).abs().max()
    print("truncation test (2017 games, full build vs build with seasons <= 2017):")
    print(diff.to_string())
    assert (diff.fillna(0) < 1e-9).all(), "LEAKAGE: 2017 features depend on later data"
    print("PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-leakage", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    g, coefs = build_all(None)
    write_preds(g)
    write_factors(g)
    pd.DataFrame(coefs).T.rename_axis("season").to_csv(OUT / "specialist_coefs.csv")
    pd.set_option("display.width", 200)
    print("\n== dev / holdout (own scorer; eval.py output in NOTES.md)")
    print(evaluate(g).round(4).to_string(index=False))
    by, pooled = ceiling(g)
    print("\n== ceiling: perfectly calibrated market-level model")
    print(by.round(4).to_string(index=False))
    for k, v in pooled.items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    print("\n== market microstructure (games with a moneyline)")
    print(microstructure(g).round(4).to_string(index=False))
    print("data quality:", data_quality(g))
    print("\nspecialist coefficients (standardised, per test season):")
    print(pd.DataFrame(coefs).T.round(4).tail(6).to_string())
    if args.check_leakage:
        check_leakage()


if __name__ == "__main__":
    main()

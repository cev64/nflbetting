"""Walk-forward stacking meta-model over every base model's out-of-sample predictions.

For each test season S, a regularized logistic regression is fit on seasons < S using the base
models' (already out-of-sample) predictions as inputs, then applied to season S. Two stacks:
  * SU:  P(home wins)   from logit(p_home) of each model (+ market logit)
  * ATS: P(home covers) from each model's (margin - spread_line) and p_home_cover logits
Usage: python research/stack/stack.py [--models a b c] [--C 0.1] [--start 2006] [--out ensemble]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent.parent
PREDS = HERE / "preds"
SCHED = pl.read_parquet(HERE / "cache" / "schedules.parquet")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_models(names: list[str] | None) -> dict[str, pl.DataFrame]:
    out = {}
    for f in sorted(PREDS.glob("*.csv")):
        m = f.stem
        if m.startswith("ensemble") or m == "market_baseline":
            continue
        if names and m not in names:
            continue
        d = pl.read_csv(f)
        cols = ["game_id", "p_home", "margin"] + (["p_home_cover"] if "p_home_cover" in d.columns else [])
        out[m] = d.select(cols).unique("game_id", keep="last")
    return out


def build(models: dict[str, pl.DataFrame]) -> pl.DataFrame:
    g = SCHED.filter(pl.col("spread_line").is_not_null()).select(
        "game_id", "season", "week", "result", "spread_line")
    g = g.with_columns(p_mkt=pl.Series(norm.cdf(g["spread_line"].to_numpy() / 13.5)))
    for m, d in models.items():
        ren = {c: f"{m}__{c}" for c in d.columns if c != "game_id"}
        g = g.join(d.rename(ren), on="game_id", how="left")
    return g


def features(g: pl.DataFrame, names: list[str], kind: str) -> np.ndarray:
    cols = []
    if kind == "su":
        cols.append(logit(g["p_mkt"].to_numpy()))
        for m in names:
            cols.append(logit(g[f"{m}__p_home"].to_numpy()))
    else:
        for m in names:
            if f"{m}__p_home_cover" in g.columns:
                cols.append(logit(g[f"{m}__p_home_cover"].fill_null(0.5).to_numpy()))
            else:
                cols.append((g[f"{m}__margin"] - g["spread_line"]).to_numpy() / 7.0)
    return np.column_stack(cols)


def run(names=None, C=0.1, start=2006, out="ensemble", verbose=True):
    models = load_models(names)
    names = list(models)
    g = build(models)
    # Keep games where every chosen base model has a prediction.
    g = g.drop_nulls([f"{m}__p_home" for m in names])
    rows = []
    for S in range(start + 1, int(g["season"].max()) + 1):
        tr = g.filter((pl.col("season") < S) & (pl.col("season") >= start) & pl.col("result").is_not_null())
        te = g.filter(pl.col("season") == S)
        if tr.height < 200 or te.height == 0:
            continue
        su_tr = tr.filter(pl.col("result") != 0)
        clf = LogisticRegression(C=C, max_iter=2000).fit(features(su_tr, names, "su"), su_tr["result"] > 0)
        p_home = clf.predict_proba(features(te, names, "su"))[:, 1]
        ats_tr = tr.filter((pl.col("result") - pl.col("spread_line")) != 0)
        clf2 = LogisticRegression(C=C, max_iter=2000).fit(
            features(ats_tr, names, "ats"), (ats_tr["result"] - ats_tr["spread_line"]) > 0)
        p_cover = clf2.predict_proba(features(te, names, "ats"))[:, 1]
        margin = norm.ppf(np.clip(p_home, 1e-4, 1 - 1e-4)) * 13.5
        rows.append(te.select("game_id", "season", "week").with_columns(
            p_home=pl.Series(p_home), margin=pl.Series(margin), p_home_cover=pl.Series(p_cover)))
        if verbose and S == int(g["season"].max()):
            print("SU coefs", dict(zip(["market"] + names, np.round(clf.coef_[0], 3))))
            print("ATS coefs", dict(zip(names, np.round(clf2.coef_[0], 3))))
    res = pl.concat(rows)
    res.write_csv(PREDS / f"{out}.csv")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--start", type=int, default=2006)
    ap.add_argument("--out", default="ensemble")
    a = ap.parse_args()
    run(a.models, a.C, a.start, a.out)

"""Shared scorer for every model. Usage: python eval.py preds/<model>.csv [more.csv ...]

A prediction file is a CSV with one row per game:
  game_id      nflverse game_id (e.g. 2024_01_BAL_KC)
  p_home       model probability the HOME team wins (0..1)
  margin       model expected home margin (home pts - away pts); used for ATS
  p_home_cover (optional) model probability home covers spread_line; if absent, ATS pick = sign(margin - spread_line)

Scoring rules (identical for every model):
  * Straight-up: pick home if p_home > 0.5 (exactly 0.5 -> pick home). Ties (games ending tied) are excluded.
  * ATS: home covers if result - spread_line > 0; pushes excluded. spread_line is nflverse's (home-favored positive).
  * Primary window: seasons 2012-2025, regular season + playoffs, every game with a spread_line.
  * Market baseline: pick the spread favorite (home on pick'em).
"""
from __future__ import annotations
import sys
import numpy as np
import polars as pl
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHED = pl.read_parquet(HERE / "cache" / "schedules.parquet")
WINDOWS = [(2012, 2025), (2018, 2025), (2022, 2025)]


def games() -> pl.DataFrame:
    return SCHED.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null()).select(
        "game_id", "season", "week", "game_type", "result", "spread_line")


def score(pred: pl.DataFrame, lo: int, hi: int) -> dict:
    g = games().filter(pl.col("season").is_between(lo, hi)).join(pred, on="game_id", how="left")
    missing = g["p_home"].null_count()
    g = g.drop_nulls("p_home")
    su = g.filter(pl.col("result") != 0)
    pick_home = su["p_home"] >= 0.5
    hw = su["result"] > 0
    acc = float((pick_home == hw).mean())
    fav = np.where(su["spread_line"] >= 0, hw, ~hw).mean()
    p = su["p_home"].clip(1e-4, 1 - 1e-4).to_numpy()
    y = hw.to_numpy().astype(float)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    ats = g.filter((pl.col("result") - pl.col("spread_line")) != 0)
    cover = (ats["result"] - ats["spread_line"]) > 0
    if "p_home_cover" in ats.columns and ats["p_home_cover"].null_count() < ats.height:
        pick_cover = ats["p_home_cover"].fill_null(0.5) >= 0.5
    else:
        pick_cover = (ats["margin"] - ats["spread_line"]) > 0
    ats_acc = float((pick_cover == cover).mean())
    return dict(window=f"{lo}-{hi}", n=su.height, missing=missing, su_acc=round(acc, 4), market_su=round(float(fav), 4),
                logloss=round(ll, 4), brier=round(brier, 4), ats_n=ats.height, ats_acc=round(ats_acc, 4))


def by_season(pred: pl.DataFrame) -> list[tuple[int, float, float, float]]:
    out = []
    for s in range(2012, 2026):
        r = score(pred, s, s)
        out.append((s, r["su_acc"], r["market_su"], r["ats_acc"]))
    return out


def load(path: str) -> pl.DataFrame:
    p = pl.read_csv(path)
    cols = ["game_id", "p_home", "margin"] + (["p_home_cover"] if "p_home_cover" in p.columns else [])
    return p.select(cols).unique("game_id", keep="last")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        pred = load(f)
        print(f"== {f}")
        for lo, hi in WINDOWS:
            print("  ", score(pred, lo, hi))
        print("   season  SU    mkt   ATS")
        for s, a, m, t in by_season(pred):
            print(f"   {s}  {a:.3f} {m:.3f} {t:.3f}")

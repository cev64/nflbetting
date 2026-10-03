"""Does the closing spread under/over-react to personnel?  Descriptive tests on the ATS residual
res = result - spread_line (home perspective).  Run: python research/personnel/analysis.py

(a) full-sample OLS (2009-2025, HC1 SEs) - descriptive, in-sample
(b) situational cover rates (team perspective, both sides stacked)
(c) walk-forward residual coefficients of the personnel model (stability over time)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import polars as pl
import statsmodels.api as sm

from common import WORK

FEATS = ["d_qb_epa", "d_qb_epa_long", "d_qb_cpoe", "d_qb_change_inseason", "d_qb_epa_delta", "d_qb_backup",
         "d_qb_rookie", "d_inj_starters_out", "d_inj_q_load", "d_inj_usual_QB", "d_inj_usual_OL", "d_inj_usual_SK",
         "d_inj_usual_DL", "d_inj_usual_LB", "d_inj_usual_DB", "d_inj_recent_SK", "d_inj_recent_DB", "rest_diff"]


def ols_table(g: pl.DataFrame, lo: int, hi: int) -> None:
    d = g.filter(pl.col("season").is_between(lo, hi)).to_pandas()
    print(f"\n(a) ATS residual on each feature, {lo}-{hi}, n={len(d)}  [univariate | multivariate], pts per 1 SD")
    Xm = sm.add_constant(d[FEATS])
    mm = sm.OLS(d["res"], Xm).fit(cov_type="HC1")
    for f in FEATS:
        m = sm.OLS(d["res"], sm.add_constant(d[[f]])).fit(cov_type="HC1")
        sd = d[f].std()
        print(f"   {f:22s} uni {m.params[f] * sd:+.2f} (t {m.tvalues[f]:+.2f}) | multi {mm.params[f] * sd:+.2f} "
              f"(t {mm.tvalues[f]:+.2f})")


def situational(g: pl.DataFrame) -> None:
    rows = []
    for side, sgn in (("home", 1), ("away", -1)):
        opp = "away" if side == "home" else "home"
        rows.append(g.select(
            "season", (pl.col("res") * sgn).alias("tres"),
            (pl.col(f"{side}_inj_starters_out") - pl.col(f"{opp}_inj_starters_out")).alias("so_diff"),
            pl.col(f"{side}_qb_change_inseason").alias("qbchg"),
            (pl.col(f"{side}_qb_epa_delta")).alias("qbdelta"),
            pl.col(f"{side}_inj_q_load").alias("qload"),
            pl.col(f"{side}_qb_backup").alias("backup"),
            (pl.col(f"{side}_qb_epa") - pl.col(f"{opp}_qb_epa")).alias("qbdiff")))
    t = pl.concat(rows).filter(pl.col("tres") != 0)

    def rep(name, cond):
        x = t.filter(cond)
        c = float((x["tres"] > 0).mean())
        se = np.sqrt(c * (1 - c) / max(x.height, 1))
        print(f"   {name:58s} n={x.height:5d} cover={c:.3f} (+/-{1.96 * se:.3f}) mean res={x['tres'].mean():+.2f}")

    print("\n(b) team-perspective cover rates, 2009-2025 (each game counted from both sides)")
    rep("team has >=3 more starters out than opponent", pl.col("so_diff") >= 3)
    rep("team has >=2 more starters out than opponent", pl.col("so_diff") >= 2)
    rep("team has >=2 fewer starters out", pl.col("so_diff") <= -2)
    rep("new starting QB this week (in-season change)", pl.col("qbchg") == 1)
    rep("new QB, downgrade > 0.10 EPA/play vs previous starter", (pl.col("qbchg") == 1) & (pl.col("qbdelta") < -0.10))
    rep("new QB, upgrade > 0.05 EPA/play (e.g. starter returns)", (pl.col("qbchg") == 1) & (pl.col("qbdelta") > 0.05))
    rep("QB started <50% of last 8 games", pl.col("backup") == 1)
    rep("team QB rating > opponent's by >0.15 EPA/play", pl.col("qbdiff") > 0.15)
    rep("heavy Questionable load (>=1.0 expected starter-equivalents)", pl.col("qload") >= 1.0)


def coef_stability() -> None:
    p = WORK / "personnel_coefs.parquet"
    if not p.exists():
        return
    c = pl.read_parquet(p)
    cols = [x for x in c.columns if x.startswith("margin_") and x != "margin_spread_line"]
    print("\n(c) walk-forward residual (ATS) ridge coefficients of `personnel` by test season (pts per unit)")
    print(c.select("season", "alpha", *cols).filter(pl.col("season") >= 2012)
          .with_columns([pl.col(x).round(2) for x in cols]))


def main(g: pl.DataFrame | None = None) -> None:
    if g is None:
        g = pl.read_parquet(WORK / "games.parquet")
    g = g.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null()).with_columns(
        (pl.col("result") - pl.col("spread_line")).alias("res"))
    ols_table(g, 2009, 2025)
    ols_table(g, 2016, 2025)
    situational(g.filter(pl.col("season").is_between(2009, 2025)))
    coef_stability()


if __name__ == "__main__":
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(30)
    main()

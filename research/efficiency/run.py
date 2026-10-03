"""Entry point: rebuild the team efficiency feature table and every prediction file of the `efficiency` family.

    python research/efficiency/run.py                 # features + all models + factors (~2 min)
    python research/efficiency/run.py --check-leakage # also verify features are point-in-time (truncation test)

Outputs
  research/features/team_game_features.parquet   point-in-time team ratings (one row per team per game)
  research/preds/logit_epa.csv                    L2-logistic on efficiency diffs, NO market input
  research/preds/logit_epa_mkt.csv                L2-logistic on spread/total + core efficiency diffs
  research/preds/gbm.csv                          LightGBM win classifier (market + all features) + residual regressor
  research/preds/xgb_margin.csv                   XGBoost margin regressor on efficiency features only (no market)
  research/preds/<model>_factors.parquet          home/away input values behind each pick (logit_epa_mkt, logit_epa)
  research/efficiency/importances.csv             coefficients / gains from the final (2026) fits
Walk-forward: for test season S (2006..2026) every model is fit on completed games of seasons 2002..S-1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features as F  # noqa: E402
import models as M  # noqa: E402
from games import build_games  # noqa: E402

ROOT = HERE.parent
PREDS = ROOT / "preds"
FEAT_OUT = ROOT / "features" / "team_game_features.parquet"

ALL = M.EFF_CORE + M.EFF_EXTRA + M.CONTEXT
# fixed configs, chosen by walk-forward log-loss on test seasons 2006-2011 only (see NOTES.md)
CONFIGS = {
    "logit_epa": M.logit_fp(ALL, C=0.003, margin_alpha=300.0),
    "logit_epa_mkt": M.logit_fp(M.EFF_CORE + M.MARKET, C=0.1, resid_features=M.EFF_CORE + ["spread_line"],
                                resid_alpha=10000.0),
    "gbm": M.gbm_fp(ALL + M.MARKET, n_cls=150, n_reg=100),
    "xgb_margin": M.xgb_margin_fp(ALL, n=150),
}

# factors shown on the site: (feature column in team table, label, better)
FACTORS = [
    ("off_epa_adj", "Offense EPA/play (opp-adj, decayed)", "high", "0.000"),
    ("def_epa_adj", "Defense EPA/play allowed (opp-adj)", "low", "0.000"),
    ("off_pass_epa_adj", "Dropback EPA/play (opp-adj)", "high", "0.000"),
    ("def_pass_epa_adj", "Dropback EPA/play allowed (opp-adj)", "low", "0.000"),
    ("off_rush_epa_adj", "Rush EPA/play (opp-adj)", "high", "0.000"),
    ("off_sr_adj", "Offense success rate (opp-adj)", "high", "0.0%"),
    ("def_sr_adj", "Defense success rate allowed (opp-adj)", "low", "0.0%"),
    ("off_ppd_adj", "Points per drive (opp-adj)", "high", "0.00"),
    ("def_ppd_adj", "Points per drive allowed (opp-adj)", "low", "0.00"),
    ("margin_adj", "Point margin per game (SRS-style)", "high", "+0.0"),
    ("qb_epa", "Starting QB EPA/dropback (decayed)", "high", "0.000"),
    ("off_to_luck_adj", "Giveaways/game (INT + 0.5*fumbles)", "low", "0.00"),
    ("def_pressure_rate", "Defensive pressure rate (sack+hit)", "high", "0.0%"),
    ("st_epa", "Special-teams EPA/game", "high", "+0.00"),
]


def write_factors(tf: pl.DataFrame, game_ids: list[str], model_id: str) -> None:
    rows = []
    sub = tf.filter(pl.col("game_id").is_in(game_ids))
    h = sub.filter(pl.col("is_home")).to_pandas().set_index("game_id")
    a = sub.filter(~pl.col("is_home")).to_pandas().set_index("game_id")
    gids = [g for g in game_ids if g in h.index and g in a.index]
    for i, (col, label, better, fmt) in enumerate(FACTORS):
        rows.append(pd.DataFrame({"game_id": gids, "rank": i + 1, "feature": col, "label": label,
                                  "home": h.loc[gids, col].astype(float).values,
                                  "away": a.loc[gids, col].astype(float).values, "better": better, "fmt": fmt}))
    pl.from_pandas(pd.concat(rows, ignore_index=True)).sort("game_id", "rank").write_parquet(
        PREDS / f"{model_id}_factors.parquet")


def importances(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficients / gains of each model refit on all completed games (= the 2026 production fit)."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    tr = df[df.result.notna() & df.spread_line.notna() & (df.season >= 2002) & (df.result != 0)]
    out = []
    for name, feats, C in [("logit_epa", ALL, 0.003), ("logit_epa_mkt", M.EFF_CORE + M.MARKET, 0.1)]:
        prep = M.Prep().fit(tr[feats])
        clf = LogisticRegression(C=C, max_iter=2000).fit(prep(tr[feats]), tr.y_win.astype(int))
        out += [(name, f, float(c)) for f, c in zip(feats, clf.coef_[0])]
    feats = ALL + M.MARKET
    g = lgb.LGBMClassifier(n_estimators=150, importance_type="gain", **M.LGB_CLS).fit(tr[feats], tr.y_win.astype(int))
    tot = g.feature_importances_.sum()
    out += [("gbm", f, float(v / tot)) for f, v in zip(feats, g.feature_importances_)]
    imp = pd.DataFrame(out, columns=["model", "feature", "value"])
    imp["abs"] = imp.value.abs()
    return imp.sort_values(["model", "abs"], ascending=[True, False]).drop(columns="abs")


def check_leakage(sched: pl.DataFrame, pbp: pl.DataFrame, tf_full: pl.DataFrame, cut=(2019, 10)) -> None:
    """Recompute features with every game at/after `cut` removed (scores nulled, pbp dropped) and confirm the
    features of week-`cut` games are bit-identical: proves nothing from the game itself or later leaks in."""
    S, W = cut
    later = (pl.col("season") > S) | ((pl.col("season") == S) & (pl.col("week") >= W))
    sched_t = sched.with_columns([pl.when(later).then(None).otherwise(pl.col(c)).alias(c)
                                  for c in ["result", "home_score", "away_score"]])
    tf_t = F.build_team_game_features(sched_t, pbp.filter(~later))
    num = [c for c, t in tf_full.schema.items() if t.is_numeric() and c not in ("season", "week")]
    a = tf_full.filter((pl.col("season") == S) & (pl.col("week") == W)).sort("game_id", "team").select(num).to_numpy()
    b = tf_t.filter((pl.col("season") == S) & (pl.col("week") == W)).sort("game_id", "team").select(num).to_numpy()
    diff = np.nanmax(np.abs(a - b))
    print(f"leakage check (features for {S} wk{W} with all games >= wk{W} removed): max abs diff = {diff:.2e}")
    assert diff < 1e-9, "feature table is not point-in-time!"


def main(check: bool = False) -> None:
    t0 = time.time()
    sched = pl.read_parquet(F.CACHE / "schedules.parquet")
    pbp = F.load_pbp()
    tf = F.build_team_game_features(sched, pbp)
    FEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tf.drop("team_fr").rename({"n_games": "n_prior_games"}).write_parquet(FEAT_OUT)
    print(f"features: {tf.shape} -> {FEAT_OUT} ({time.time() - t0:.0f}s)")
    if check:
        check_leakage(sched, pbp, tf)

    df = build_games(tf, sched).to_pandas()
    meta = df.set_index("game_id")[["season", "week"]]
    for name, fp in CONFIGS.items():
        t = time.time()
        p = M.walk_forward(df, fp, first=2006, last=2026, train_from=2002)
        p = p.join(meta, on="game_id")[["game_id", "season", "week", "p_home", "margin", "p_home_cover"]]
        p.to_csv(PREDS / f"{name}.csv", index=False)
        print(f"{name}: {len(p)} rows ({time.time() - t:.0f}s)")
        if name in ("logit_epa_mkt", "logit_epa"):
            write_factors(tf, p.game_id.tolist(), name)
    importances(df).to_csv(HERE / "importances.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(check="--check-leakage" in sys.argv)

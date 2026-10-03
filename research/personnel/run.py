"""Personnel model family: rebuilds features, walk-forward predictions and factor tables end to end.

    python research/personnel/run.py              # features + both models + factors (~2 min)
    python research/personnel/run.py --analysis   # also print the market-mispricing analysis

Outputs
    research/preds/personnel.csv, personnel_nomkt.csv, personnel_ats.csv           (game_id, season, week, p_home, margin[, p_home_cover])
    research/preds/<model_id>_factors.parquet for each of the three
    research/personnel/work/*.parquet                            (intermediate tables, coefficients)
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl  # noqa: E402

import features  # noqa: E402
import model as M  # noqa: E402
from common import PREDS, WORK  # noqa: E402

FACTORS = [
    # key, label, better, fmt
    ("qb_epa", "Starting QB EPA/play vs league (recent form, 120-day half-life)", "high", "+0.000"),
    ("qb_epa_long", "Starting QB EPA/play vs league (long-term, 400-day half-life)", "high", "+0.000"),
    ("qb_cpoe", "Starting QB completion % over expected (decayed)", "high", "+0.0"),
    ("qb_change_inseason", "New starting QB vs last game (1 = yes)", "low", "0"),
    ("qb_backup", "Starting QB started <50% of team's last 8 games (1 = yes)", "low", "0"),
    ("qb_rookie", "Rookie starting QB (1 = yes)", "low", "0"),
    ("inj_starters_out", "Regular starters Out/Doubtful/likely out (injury report)", "low", "0.0"),
    ("inj_usual_QB", "Injury load: QB (snap-share-weighted expected absences)", "low", "0.00"),
    ("inj_usual_OL", "Injury load: offensive line", "low", "0.00"),
    ("inj_usual_SK", "Injury load: RB/WR/TE", "low", "0.00"),
    ("inj_usual_DL", "Injury load: defensive line", "low", "0.00"),
    ("inj_usual_LB", "Injury load: linebackers", "low", "0.00"),
    ("inj_usual_DB", "Injury load: defensive backs", "low", "0.00"),
    ("inj_q_load", "Injury load from Questionable players (game-time uncertainty)", "low", "0.00"),
    ("rest", "Rest days", "high", "0"),
]


def key_absences(detail: pl.DataFrame) -> pl.DataFrame:
    """Per (season, week, team): text list of the most important likely-absent players."""
    d = detail.filter((pl.col("p_miss") >= 0.5) & (pl.col("imp_usual") >= 0.4)).sort("imp_usual", descending=True)
    return d.group_by("season", "week", "team").agg(
        pl.concat_str(pl.col("full_name"), pl.lit(" ("), pl.col("position"), pl.lit(", "),
                      pl.col("report_status").fill_null("DNP"), pl.lit(")")).head(4).str.join(", ").alias("txt"),
        pl.len().alias("n"))


def factors_table(g: pl.DataFrame, detail: pl.DataFrame, game_ids: pl.Series) -> pl.DataFrame:
    g = g.filter(pl.col("game_id").is_in(game_ids.implode()))
    g = g.with_columns(pl.col("home_rest").cast(pl.Float64).alias("home_rest"),
                       pl.col("away_rest").cast(pl.Float64).alias("away_rest"))
    parts = []
    for key, label, better, fmt in FACTORS:
        hcol, acol = (f"home_{key}", f"away_{key}")
        txt_h = pl.col("home_qb_name") if key.startswith("qb_epa") else pl.lit(None, dtype=pl.Utf8)
        txt_a = pl.col("away_qb_name") if key.startswith("qb_epa") else pl.lit(None, dtype=pl.Utf8)
        parts.append(g.select(
            "game_id", pl.lit(key).alias("feature"), pl.lit(label).alias("label"),
            pl.col(hcol).cast(pl.Float64).alias("home"), pl.col(acol).cast(pl.Float64).alias("away"),
            txt_h.alias("home_text"), txt_a.alias("away_text"),
            pl.lit(better).alias("better"), pl.lit(fmt).alias("fmt")))
    ka = key_absences(detail)
    kh = g.select("game_id", "season", "week", "home_team", "away_team").join(
        ka.rename({"team": "home_team", "txt": "home_text", "n": "home"}), on=["season", "week", "home_team"], how="left"
    ).join(ka.rename({"team": "away_team", "txt": "away_text", "n": "away"}), on=["season", "week", "away_team"],
           how="left")
    parts.append(kh.select(
        "game_id", pl.lit("key_absences").alias("feature"),
        pl.lit("Key players likely out (count; names)").alias("label"),
        pl.col("home").fill_null(0).cast(pl.Float64), pl.col("away").fill_null(0).cast(pl.Float64),
        "home_text", "away_text", pl.lit("low").alias("better"), pl.lit("0").alias("fmt")))
    order = {k: i for i, (k, *_) in enumerate(FACTORS)} | {"key_absences": len(FACTORS)}
    return pl.concat(parts).with_columns(pl.col("feature").replace_strict(order).alias("rank")).sort(
        "game_id", "rank").drop("rank")


def main(analysis: bool = False) -> None:
    g, detail = features.build()
    PREDS.mkdir(exist_ok=True)
    for model_id, feats in M.FEATURES.items():
        pred, coefs = M.walk_forward(g, feats)
        pred = pred.sort("season", "week", "game_id")
        pred.write_csv(PREDS / f"{model_id}.csv")
        coefs.write_parquet(WORK / f"{model_id}_coefs.parquet")
        factors_table(g, detail, pred["game_id"]).write_parquet(PREDS / f"{model_id}_factors.parquet")
        print(model_id, pred.height, "rows;", "2012-2025", M.quick_score(pred, g), "| 2006-2011",
              M.quick_score(pred, g, 2006, 2011))
    pred, coefs = M.ats_walk_forward(g)
    pred = pred.sort("season", "week", "game_id")
    pred.write_csv(PREDS / "personnel_ats.csv")
    coefs.write_parquet(WORK / "personnel_ats_coefs.parquet")
    factors_table(g, detail, pred["game_id"]).write_parquet(PREDS / "personnel_ats_factors.parquet")
    print("personnel_ats", pred.height, "rows;", "2012-2025", M.quick_score(pred, g))
    if analysis:
        import analysis as A
        A.main(g)


if __name__ == "__main__":
    main(analysis="--analysis" in sys.argv)

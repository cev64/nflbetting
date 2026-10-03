# efficiency — play-by-play efficiency features + ML

**Bottom line (honest):** play-by-play efficiency stats do **not** add measurable information beyond the closing
`spread_line`. The best model here (`logit_epa_mkt`, market + efficiency) is the market: 66.5% SU vs 66.4% for
the favorite, with no ATS edge. Without the market, efficiency features alone reach ~64.7-65.1% SU. Nothing
came close to the 68% SU / 52.4% ATS targets, so no result needed the >67.5% leakage stress test. The
feature table was still checked for point-in-time correctness (truncation test below).

## Results, walk-forward 2012-2025 (`python research/eval.py preds/<m>.csv`)

| model | inputs | SU | ATS | logloss | brier |
|---|---|---|---|---|---|
| market favorite (reference) | spread | 0.6643 | 0.5091 | 0.6108 | 0.2115 |
| `logit_epa_mkt` | spread, total + 8 core efficiency diffs (L2 logistic); ATS = spread + no-intercept ridge on residual | **0.6648** | 0.4928 | **0.6097** | **0.2112** |
| `gbm` | LightGBM win classifier on market + 48 features; LightGBM regressor on result − spread | 0.6572 | 0.4995 | 0.6168 | 0.2140 |
| `logit_epa` | 48 efficiency/context features, **no market** (L2 logistic; ridge margin) | 0.6473 | 0.4984 | 0.6260 | 0.2183 |
| `xgb_margin` | XGBoost margin regressor, efficiency/context, **no market** | 0.6507 | 0.4979 | 0.6269 | 0.2186 |

2006-2011 (also out-of-sample; this is the window used to choose configs): logit_epa 0.653 SU / 0.533 ATS,
logit_epa_mkt 0.662 / 0.498, gbm 0.652 / 0.509, xgb_margin 0.634 / 0.520.

For stacking: `logit_epa` and `xgb_margin` are the market-free efficiency signals (well calibrated, see
below); `logit_epa_mkt` is essentially a recalibrated market line.

Calibration 2012-2025 (pred vs actual home-win rate by bin) is good for all four; e.g. logit_epa_mkt
0.29/0.26, 0.41/0.43, 0.59/0.57, 0.72/0.73, 0.85/0.86; logit_epa 0.29/0.29, 0.43/0.40, 0.58/0.55, 0.72/0.72, 0.85/0.84.

### Full eval.py output
```
== preds/logit_epa.csv
   {'window': '2012-2025', 'n': 3816, 'missing': 0, 'su_acc': 0.6473, 'market_su': 0.6643, 'logloss': 0.626, 'brier': 0.2183, 'ats_n': 3736, 'ats_acc': 0.4984}
   {'window': '2018-2025', 'n': 2219, 'missing': 0, 'su_acc': 0.6462, 'market_su': 0.662, 'logloss': 0.6303, 'brier': 0.2202, 'ats_n': 2175, 'ats_acc': 0.4887}
   {'window': '2022-2025', 'n': 1136, 'missing': 0, 'su_acc': 0.654, 'market_su': 0.6752, 'logloss': 0.6279, 'brier': 0.2193, 'ats_n': 1110, 'ats_acc': 0.4784}
   season  SU    mkt   ATS
   2012  0.628 0.647 0.496
   2013  0.658 0.707 0.531
   2014  0.692 0.673 0.513
   2015  0.614 0.629 0.525
   2016  0.623 0.653 0.492
   2017  0.678 0.697 0.513
   2018  0.645 0.657 0.492
   2019  0.609 0.643 0.482
   2020  0.668 0.672 0.513
   2021  0.630 0.623 0.509
   2022  0.652 0.663 0.471
   2023  0.642 0.674 0.435
   2024  0.691 0.705 0.516
   2025  0.630 0.658 0.489
== preds/logit_epa_mkt.csv
   {'window': '2012-2025', 'n': 3816, 'missing': 0, 'su_acc': 0.6648, 'market_su': 0.6643, 'logloss': 0.6097, 'brier': 0.2112, 'ats_n': 3736, 'ats_acc': 0.4928}
   {'window': '2018-2025', 'n': 2219, 'missing': 0, 'su_acc': 0.662, 'market_su': 0.662, 'logloss': 0.6091, 'brier': 0.2109, 'ats_n': 2175, 'ats_acc': 0.4897}
   {'window': '2022-2025', 'n': 1136, 'missing': 0, 'su_acc': 0.6761, 'market_su': 0.6752, 'logloss': 0.6069, 'brier': 0.21, 'ats_n': 1110, 'ats_acc': 0.5018}
   season  SU    mkt   ATS
   2012  0.658 0.647 0.511
   2013  0.707 0.707 0.496
   2014  0.673 0.673 0.494
   2015  0.633 0.629 0.486
   2016  0.649 0.653 0.504
   2017  0.693 0.697 0.490
   2018  0.664 0.657 0.484
   2019  0.639 0.643 0.467
   2020  0.664 0.672 0.446
   2021  0.623 0.623 0.509
   2022  0.667 0.663 0.442
   2023  0.670 0.674 0.524
   2024  0.705 0.705 0.530
   2025  0.662 0.658 0.511
== preds/gbm.csv
   {'window': '2012-2025', 'n': 3816, 'missing': 0, 'su_acc': 0.6572, 'market_su': 0.6643, 'logloss': 0.6168, 'brier': 0.214, 'ats_n': 3736, 'ats_acc': 0.4995}
   {'window': '2018-2025', 'n': 2219, 'missing': 0, 'su_acc': 0.6571, 'market_su': 0.662, 'logloss': 0.6163, 'brier': 0.2137, 'ats_n': 2175, 'ats_acc': 0.4887}
   {'window': '2022-2025', 'n': 1136, 'missing': 0, 'su_acc': 0.6681, 'market_su': 0.6752, 'logloss': 0.6151, 'brier': 0.2132, 'ats_n': 1110, 'ats_acc': 0.5}
   season  SU    mkt   ATS
   2012  0.654 0.647 0.527
   2013  0.662 0.707 0.496
   2014  0.699 0.673 0.494
   2015  0.625 0.629 0.498
   2016  0.615 0.653 0.546
   2017  0.689 0.697 0.525
   2018  0.664 0.657 0.469
   2019  0.635 0.643 0.432
   2020  0.668 0.672 0.498
   2021  0.616 0.623 0.505
   2022  0.670 0.663 0.474
   2023  0.660 0.674 0.476
   2024  0.698 0.705 0.523
   2025  0.644 0.658 0.525
== preds/xgb_margin.csv
   {'window': '2012-2025', 'n': 3816, 'missing': 0, 'su_acc': 0.6507, 'market_su': 0.6643, 'logloss': 0.6269, 'brier': 0.2186, 'ats_n': 3736, 'ats_acc': 0.4979}
   {'window': '2018-2025', 'n': 2219, 'missing': 0, 'su_acc': 0.6494, 'market_su': 0.662, 'logloss': 0.6318, 'brier': 0.2208, 'ats_n': 2175, 'ats_acc': 0.4929}
   {'window': '2022-2025', 'n': 1136, 'missing': 0, 'su_acc': 0.6452, 'market_su': 0.6752, 'logloss': 0.6321, 'brier': 0.221, 'ats_n': 1110, 'ats_acc': 0.4847}
   season  SU    mkt   ATS
   2012  0.635 0.647 0.496
   2013  0.650 0.707 0.523
   2014  0.688 0.673 0.506
   2015  0.633 0.629 0.444
   2016  0.645 0.653 0.530
   2017  0.663 0.697 0.529
   2018  0.660 0.657 0.539
   2019  0.650 0.643 0.471
   2020  0.672 0.672 0.506
   2021  0.634 0.623 0.491
   2022  0.624 0.663 0.496
   2023  0.618 0.674 0.450
   2024  0.705 0.705 0.530
   2025  0.634 0.658 0.461
```

## Pipeline (`python research/efficiency/run.py [--check-leakage]`, ~65 s)
1. `features.py` — pbp (only 32 columns via `pl.scan_parquet`) → per team-game raw stats → point-in-time decayed,
   regressed, opponent-adjusted ratings → `research/features/team_game_features.parquet` (columns documented below).
2. `games.py` — game-level matrix: home−away differences of every rating, composite nets
   `net_X = (home off_X − home def_X) − (away off_X − away def_X)`, market (spread_line, total_line; a vig-free
   moneyline prob is computed but unused since it matched the spread exactly in SU), rest diff / short week (≤5 days) / off a bye (≥12), div_game, neutral site, indoor, turf,
   temp, wind, primetime, playoff, QB-change flags, QB experience, travel miles & timezone shift of the away
   team, west-coast team at 1pm ET, min games in rating window.
3. `models.py` — walk-forward: for each test season S = 2006..2026 every model is refit on all completed games of
   seasons 2002..S-1 (2026 predictions use 2002-2025 fits + 2026 features from weeks already played).
4. `run.py` — writes `preds/{logit_epa,logit_epa_mkt,gbm,xgb_margin}.csv` (every game 2006-2026 with a
   spread_line, incl. the unplayed 2026 week 4-5 games), `preds/{logit_epa_mkt,logit_epa}_factors.parquet`
   (14 home/away factor rows per game: game_id, rank, feature, label, home, away, better, fmt) and
   `importances.csv`.

## Leakage controls
- Ratings for a game in (S, w) use only games with season < S or (season = S and week < w); QB rating likewise.
- `run.py --check-leakage` recomputes the whole feature table after deleting every game from 2019 week 10 on
  (pbp rows dropped, schedule scores nulled) and asserts week-10 features are identical (max abs diff 4e-16).
- All scalers/imputers/models are fit on the training fold only. No preds file scored > 67.5% SU or > 54% ATS.

## How configs were chosen (and honesty notes)
- Decay hyper-parameters (half-life 12 games, prior-season multiplier 0.4, 3 pseudo-games of shrinkage) chosen on
  2003-2011 by correlation of rating diffs with results (27-config grid; the surface was very flat, 0.37-0.39).
- Model configs chosen by walk-forward log-loss on test seasons 2006-2011: logistic C ∈ {0.001..1}, feature set
  (core vs all), LightGBM trees ∈ {75,150,300,600}, XGB trees ∈ {150,300,600}, ridge alphas.
- Caveat: while exploring I printed 2012-2025 numbers next to 2006-2011 for ~40 configs. Choices were made on the
  2006-11 column, but all configs landed within ±1 pt SU of each other, so selection bias is small either way.
  Variants tried that did not help (out-of-sample, both windows): mid-season refit (split at week 10), 8-season
  rolling window, two-stage "efficiency margin vs spread" blend, moneyline vs spread (identical SU), recency /
  previous-cover overreaction features (|r| < 0.02 with result−spread), GBM with 300-600 trees (overfits).
- The residual-vs-market regression is fit **without an intercept** (decided before scoring it): with an intercept
  the ATS pick mostly tracks the historical home-cover bias of the training years rather than the features.
- Residual (result − spread) correlations: the most stable were d_qb_epa (0.032 in 2002-11 / 0.036 in 2012-25),
  d_margin_adj (0.033 / 0.030), net_sr_adj (0.041 / 0.018); far too small to beat −110 juice.

## Feature importance (final fits on 2002-2025, i.e. the 2026 production models)
logit_epa (standardized coefficients, top 12): d_qb_epa +0.198, d_margin_adj +0.125, net_sr_adj +0.100,
net_ppd_adj +0.097, d_st_epa +0.086, h_qb_change −0.076, d_def_explosive −0.074, d_off_sack_rate −0.056,
d_qb_plays +0.054, a_qb_change +0.053, away_bye −0.044, net_early_epa_adj +0.044.

logit_epa_mkt (standardized coefficients): spread_line +0.796, d_margin_adj +0.151, net_epa_adj −0.151 (collinear
with the other nets), d_qb_epa +0.089, net_pass_epa_adj −0.070, net_sr_adj +0.051, net_early_epa_adj +0.049,
total_line −0.036, net_ppd_adj −0.026, net_rush_epa_adj −0.005.

gbm (share of gain): spread_line 0.62, net_ppd_adj 0.058, d_margin_adj 0.052, d_qb_epa 0.046, net_epa_adj 0.041,
sack/pressure rates ~0.012 each, turnover luck-adj ~0.011 each. Full lists: `importances.csv`.

Takeaways: the starting-QB EPA rating, SRS-style adjusted margin, success rate and points per drive carry the
signal; once the spread is known they are redundant (the spread already prices them).


## Shared feature table: `research/features/team_game_features.parquet`
One row per team per scheduled game (1999-2026, incl. unplayed 2026 games). Key: `season, week, game_id, team`.
All values are **pre-game**: built only from games with `season < S` or `season == S and week < w`.
`team` uses the schedule's code (OAK/SD/STL historically); ratings are keyed internally by franchise (LV/LAC/LA).

Rating construction (per franchise, per as-of week): weighted mean over the franchise's games from the current
and previous 2 seasons, weight = 0.5^(games_ago/12) * 0.4^(seasons_back), shrunk toward the previous season's
league mean with 3 pseudo-games. Plays with win prob outside [0.1, 0.9] get weight 0.25 (garbage time);
kneels/spikes/2-pt tries excluded. Hyper-parameters chosen on 2003-2011 only.

| column | meaning |
|---|---|
| opponent, is_home | opponent code, home flag |
| qb_id, prev_qb_id, qb_change | scheduled starter (gsis id), previous game's starter, flag if different |
| n_prior_games | games in the rating window |
| off_X / def_X | team offense / defense-allowed decayed average of X |
| X = epa | EPA/play (pass+run plays) |
| pass_epa | dropback EPA/play (incl. sacks & scrambles) |
| rush_epa | designed-run EPA/play |
| early_epa | 1st/2nd-down EPA/play |
| sr | success rate |
| explosive | rate of 20+ yd dropbacks / 10+ yd runs |
| cpoe | completion % over expected (2006+; earlier = 0) |
| sack_rate | sacks per dropback |
| pressure_rate | (sack or QB hit) per dropback (2006+) |
| int_rate | INT per dropback |
| to_rate | (INT + fumbles lost) per play |
| to_luck_adj | INT + 0.5 * fumbles (recovery luck removed), per game |
| third_conv | 3rd-down conversion rate |
| rz_td | TD rate on drives reaching the opp 20 |
| ppd | points per drive (TD = 6.95, FG = 3) |
| pts | points per game (schedule scores) |
| proe | pass rate over expected in neutral situations (2006+) |
| plays | offensive plays per game (pace) |
| st_epa | net special-teams EPA per game (kickoffs, punts, FG, XP) |
| pen_yds | penalty yards committed per game |
| margin | point margin per game |
| *_adj | opponent-adjusted versions (each past game's value minus the opponent's as-of rating on the other side of the ball, relative to league mean; margin_adj adds opponent's margin rating — SRS-like) for epa, pass_epa, rush_epa, early_epa, sr, ppd, pts, margin |
| qb_epa | scheduled starter's own decayed dropback EPA/play (last 3 seasons, half-life 12 games, 0.8/season, shrunk toward -0.10 with 120 pseudo-plays) |
| qb_plays | starter's dropbacks in that window (experience) |

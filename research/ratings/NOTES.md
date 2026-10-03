# Ratings family: Elo, ridge/Massey, Kalman (+ market blends)

Entry point: `python research/ratings/run.py` takes about 20 s, is deterministic and has no randomness. It writes:

| file | what |
|---|---|
| `preds/elo.csv` | FiveThirtyEight-style Elo with QB adjustment, rest, travel and HFA |
| `preds/ratings_ridge.csv` | Weekly-refit, opponent-adjusted ridge (Massey) ratings on point margin and EPA/play margin, plus QB adjustment and rest/travel context |
| `preds/kalman.csv` | Forward Kalman filter of team strengths with an HFA state and season-boundary variance inflation |
| `preds/ratings_combo.csv` | Walk-forward logistic regression on [elo, kalman, ridge_pts, ridge_epa] |
| `preds/<id>_mkt.csv` | Same as `<id>`, with `spread_line` added as a feature in every walk-forward fit |
| `preds/elo_factors.parquet`, `preds/kalman_factors.parquet` | Per-game home/away factor values for the site (same factor set) |

Every file has rows for **2006 through the 2026 live slate**: every game with a `spread_line`, including the
unplayed 2026 week 4 and week 5 games. The stacker needs predictions from 2006 onward. Columns are
`game_id, season, week, p_home, margin, p_home_cover`.

Code: `data.py` (loading, travel distances, pbp EPA per game, QB per-game stats), `qb.py` (QB tracker),
`elo.py`, `ridge.py`, `kalman.py`, `common.py` (walk-forward mapping, market blend, scoring), `run.py`.

## Leakage and walk-forward discipline
* **Ratings.** Every rating model is sequential. A game's prediction is recorded before any state update that
  uses its result.
  * Elo and Kalman process games in kickoff order. A team's state only changes when that team plays.
  * The ridge model is refit for each (season, week) slate, using only games from strictly earlier weeks.
* **QB values.** These use only the scheduled starter IDs (`home_qb_id` / `away_qb_id`) and stats from strictly
  earlier games.
* **EPA.** Per-game EPA from pbp is used only after the game, to update ratings. The scale that converts
  EPA/play to points (46.0, intercept 3.33) was fit on 1999-2005 only.
* **Mapping to outputs.** `common.walk_forward` builds `p_home`, `margin` and `p_home_cover`. For test season S it
  fits on seasons 2000..S-1 (1999 is the warm-up season), with recency weights of 0.5^((S-1-season)/6):
  * `p_home`: logistic regression of home win on the features
  * `margin`: OLS of the result on the features
  * `p_home_cover`: logistic regression of cover on (fitted margin - spread_line)
* **Hyperparameter tuning.** Rating hyperparameters were tuned **only on 2002-2011**, by coordinate descent. The
  objective was the logloss of a two-parameter logistic regression of home win on the rating margin.
  * The 2006-2011 predictions are therefore *in-sample for the hyperparameters*, though out-of-sample for all
    fitted coefficients. Keep this in mind when the stacker trains on them.
  * The 2012-2025 numbers are clean.
* **Honest disclosure.** After seeing 2012-2025 results I compared three calibration recency settings (none,
  half-life 6, half-life 3) and a few HFA learning rates. Everything moved by ±0.3 pt SU, which is noise.
  * I kept half-life 6, a choice made on prior grounds: market efficiency drifts over time.
  * I did **not** adopt any setting because it scored better on 2012-2025.
  * A rolling-origin tuner, which picks the best config on trailing seasons before each test season, gave the
    same results as fixed pre-2012 tuning, so I dropped it for simplicity.
* **Stress tests.** None of the models scores above the 67.5% SU or 54% ATS flags.

## Final eval.py numbers (`python research/eval.py preds/<id>.csv`)

| model | window | n | SU | market SU | logloss | brier | ATS (n) |
|---|---|---|---|---|---|---|---|
| elo | 2006-2011 | 1601 | 0.6396 | 0.6627 | 0.6217 | 0.2169 | 0.5208 (1559) |
| elo | **2012-2025** | 3816 | **0.6536** | 0.6643 | 0.6239 | 0.2176 | 0.5021 (3736) |
| elo | 2018-2025 | 2219 | 0.6453 | 0.6620 | 0.6305 | 0.2206 | 0.4979 (2175) |
| elo | 2022-2025 | 1136 | 0.6496 | 0.6752 | 0.6261 | 0.2190 | 0.4964 (1110) |
| elo_mkt | 2006-2011 | 1601 | 0.6583 | 0.6627 | 0.6108 | 0.2119 | 0.5247 (1559) |
| elo_mkt | **2012-2025** | 3816 | **0.6654** | 0.6643 | 0.6087 | 0.2108 | 0.5094 (3736) |
| elo_mkt | 2018-2025 | 2219 | 0.6643 | 0.6620 | 0.6099 | 0.2112 | 0.5039 (2175) |
| elo_mkt | 2022-2025 | 1136 | 0.6761 | 0.6752 | 0.6064 | 0.2099 | 0.4973 (1110) |
| ratings_ridge | 2006-2011 | 1601 | 0.6490 | 0.6627 | 0.6208 | 0.2166 | 0.5292 (1559) |
| ratings_ridge | **2012-2025** | 3816 | **0.6441** | 0.6643 | 0.6237 | 0.2175 | 0.5072 (3736) |
| ratings_ridge | 2018-2025 | 2219 | 0.6390 | 0.6620 | 0.6295 | 0.2199 | 0.5067 (2175) |
| ratings_ridge | 2022-2025 | 1136 | 0.6444 | 0.6752 | 0.6296 | 0.2202 | 0.4946 (1110) |
| ratings_ridge_mkt | 2006-2011 | 1601 | 0.6621 | 0.6627 | 0.6112 | 0.2121 | 0.5202 (1559) |
| ratings_ridge_mkt | **2012-2025** | 3816 | **0.6659** | 0.6643 | 0.6087 | 0.2108 | 0.5096 (3736) |
| ratings_ridge_mkt | 2018-2025 | 2219 | 0.6634 | 0.6620 | 0.6095 | 0.2111 | 0.5085 (2175) |
| ratings_ridge_mkt | 2022-2025 | 1136 | 0.6769 | 0.6752 | 0.6065 | 0.2098 | 0.4955 (1110) |
| kalman | 2006-2011 | 1601 | 0.6465 | 0.6627 | 0.6171 | 0.2149 | 0.5343 (1559) |
| kalman | **2012-2025** | 3816 | **0.6499** | 0.6643 | 0.6250 | 0.2180 | 0.5005 (3736) |
| kalman | 2018-2025 | 2219 | 0.6386 | 0.6620 | 0.6331 | 0.2214 | 0.4984 (2175) |
| kalman | 2022-2025 | 1136 | 0.6452 | 0.6752 | 0.6304 | 0.2206 | 0.4955 (1110) |
| kalman_mkt | 2006-2011 | 1601 | 0.6596 | 0.6627 | 0.6101 | 0.2116 | 0.5318 (1559) |
| kalman_mkt | **2012-2025** | 3816 | **0.6651** | 0.6643 | 0.6090 | 0.2109 | 0.5169 (3736) |
| kalman_mkt | 2018-2025 | 2219 | 0.6625 | 0.6620 | 0.6101 | 0.2113 | 0.5103 (2175) |
| kalman_mkt | 2022-2025 | 1136 | 0.6752 | 0.6752 | 0.6061 | 0.2097 | 0.5126 (1110) |
| ratings_combo | 2006-2011 | 1601 | 0.6490 | 0.6627 | 0.6181 | 0.2154 | 0.5305 (1559) |
| ratings_combo | **2012-2025** | 3816 | **0.6517** | 0.6643 | 0.6234 | 0.2172 | 0.5102 (3736) |
| ratings_combo | 2018-2025 | 2219 | 0.6399 | 0.6620 | 0.6307 | 0.2204 | 0.5053 (2175) |
| ratings_combo | 2022-2025 | 1136 | 0.6435 | 0.6752 | 0.6278 | 0.2195 | 0.4973 (1110) |
| ratings_combo_mkt | 2006-2011 | 1601 | 0.6521 | 0.6627 | 0.6101 | 0.2116 | 0.5356 (1559) |
| ratings_combo_mkt | **2012-2025** | 3816 | **0.6664** | 0.6643 | 0.6090 | 0.2109 | 0.5037 (3736) |
| ratings_combo_mkt | 2018-2025 | 2219 | 0.6652 | 0.6620 | 0.6103 | 0.2114 | 0.4947 (2175) |
| ratings_combo_mkt | 2022-2025 | 1136 | 0.6761 | 0.6752 | 0.6059 | 0.2097 | 0.4973 (1110) |
| market_baseline | 2012-2025 | 3816 | 0.6643 | 0.6643 | 0.6108 | 0.2115 | 0.5091 (3736) |

Per season, each cell is SU / ATS:

| season | market SU | elo | elo_mkt | ratings_ridge | ratings_ridge_mkt | kalman | kalman_mkt | ratings_combo | ratings_combo_mkt |
|---|---|---|---|---|---|---|---|---|---|
| 2006 | 0.603 | 0.596 / 0.529 | 0.599 / 0.544 | 0.599 / 0.533 | 0.614 / 0.517 | 0.603 / 0.533 | 0.603 / 0.548 | 0.603 / 0.517 | 0.599 / 0.544 |
| 2007 | 0.685 | 0.640 / 0.573 | 0.678 / 0.553 | 0.659 / 0.523 | 0.670 / 0.546 | 0.655 / 0.530 | 0.678 / 0.569 | 0.655 / 0.553 | 0.663 / 0.557 |
| 2008 | 0.677 | 0.639 / 0.536 | 0.673 / 0.513 | 0.628 / 0.521 | 0.680 / 0.513 | 0.628 / 0.533 | 0.677 / 0.487 | 0.628 / 0.548 | 0.654 / 0.513 |
| 2009 | 0.689 | 0.704 / 0.490 | 0.689 / 0.517 | 0.689 / 0.560 | 0.693 / 0.513 | 0.678 / 0.525 | 0.682 / 0.506 | 0.689 / 0.502 | 0.674 / 0.483 |
| 2010 | 0.659 | 0.599 / 0.466 | 0.648 / 0.523 | 0.629 / 0.492 | 0.648 / 0.534 | 0.633 / 0.538 | 0.652 / 0.534 | 0.644 / 0.519 | 0.648 / 0.550 |
| 2011 | 0.663 | 0.659 / 0.531 | 0.663 / 0.496 | 0.689 / 0.547 | 0.667 / 0.496 | 0.682 / 0.547 | 0.667 / 0.547 | 0.674 / 0.543 | 0.674 / 0.566 |
| 2012 | 0.647 | 0.639 / 0.469 | 0.647 / 0.481 | 0.650 / 0.523 | 0.647 / 0.527 | 0.647 / 0.511 | 0.650 / 0.519 | 0.650 / 0.523 | 0.658 / 0.504 |
| 2013 | 0.707 | 0.699 / 0.515 | 0.707 / 0.504 | 0.658 / 0.496 | 0.703 / 0.488 | 0.680 / 0.508 | 0.703 / 0.531 | 0.677 / 0.508 | 0.711 / 0.511 |
| 2014 | 0.673 | 0.669 / 0.487 | 0.677 / 0.525 | 0.680 / 0.521 | 0.680 / 0.502 | 0.707 / 0.483 | 0.680 / 0.506 | 0.699 / 0.494 | 0.677 / 0.494 |
| 2015 | 0.629 | 0.663 / 0.599 | 0.629 / 0.595 | 0.611 / 0.529 | 0.633 / 0.525 | 0.629 / 0.541 | 0.633 / 0.552 | 0.652 / 0.560 | 0.625 / 0.556 |
| 2016 | 0.653 | 0.664 / 0.508 | 0.649 / 0.500 | 0.657 / 0.511 | 0.657 / 0.496 | 0.653 / 0.481 | 0.653 / 0.523 | 0.660 / 0.511 | 0.645 / 0.530 |
| 2017 | 0.697 | 0.655 / 0.471 | 0.693 / 0.498 | 0.652 / 0.467 | 0.697 / 0.529 | 0.678 / 0.498 | 0.693 / 0.525 | 0.670 / 0.506 | 0.693 / 0.502 |
| 2018 | 0.657 | 0.653 / 0.515 | 0.660 / 0.531 | 0.657 / 0.508 | 0.660 / 0.523 | 0.653 / 0.512 | 0.657 / 0.519 | 0.653 / 0.535 | 0.664 / 0.531 |
| 2019 | 0.643 | 0.639 / 0.467 | 0.658 / 0.490 | 0.617 / 0.552 | 0.639 / 0.549 | 0.628 / 0.479 | 0.647 / 0.498 | 0.628 / 0.479 | 0.647 / 0.444 |
| 2020 | 0.672 | 0.664 / 0.498 | 0.668 / 0.498 | 0.653 / 0.502 | 0.675 / 0.502 | 0.645 / 0.509 | 0.672 / 0.498 | 0.638 / 0.513 | 0.683 / 0.476 |
| 2021 | 0.623 | 0.609 / 0.516 | 0.623 / 0.523 | 0.609 / 0.516 | 0.623 / 0.516 | 0.602 / 0.505 | 0.623 / 0.516 | 0.627 / 0.527 | 0.623 / 0.516 |
| 2022 | 0.663 | 0.642 / 0.522 | 0.667 / 0.507 | 0.638 / 0.507 | 0.663 / 0.507 | 0.642 / 0.507 | 0.663 / 0.507 | 0.621 / 0.515 | 0.667 / 0.507 |
| 2023 | 0.674 | 0.635 / 0.491 | 0.674 / 0.502 | 0.632 / 0.498 | 0.677 / 0.502 | 0.625 / 0.502 | 0.674 / 0.509 | 0.628 / 0.502 | 0.674 / 0.502 |
| 2024 | 0.705 | 0.663 / 0.480 | 0.705 / 0.473 | 0.677 / 0.480 | 0.709 / 0.480 | 0.667 / 0.480 | 0.705 / 0.512 | 0.674 / 0.480 | 0.705 / 0.480 |
| 2025 | 0.658 | 0.658 / 0.493 | 0.658 / 0.507 | 0.630 / 0.493 | 0.658 / 0.493 | 0.648 / 0.493 | 0.658 / 0.521 | 0.651 / 0.493 | 0.658 / 0.500 |

How the pure ratings relate to the market over 2012-2025: the correlation of logit(p_home) with the market
probability is 0.88 for Elo and 0.90 for Kalman, ridge and combo. Each pure rating disagrees with the market's
SU pick in 13-15% of games. That makes them **diverse inputs for the stacker**. The `_mkt` variants are about
0.99 correlated with the market and differ from its pick in under 1% of games.

## Bottom line (honest)
* **Market target.** No rating model in this family reaches the 68% target, and none clearly beats the closing
  line.
  * The pure ratings score 64.4-65.4% SU over 2012-2025, against 66.4% for the market.
  * The market blends score 66.5-66.6% SU, which is +0.1 to +0.2 pt (4-8 games over 14 seasons), with logloss
    of 0.6087 against 0.6108 for the market favorite's mapping.
  * That logloss gain is mostly better calibration of the spread → probability mapping, not information.
* **ATS.** All models are at 50-52%, below the 52.4% break-even. No ATS edge.
* **Rating information beyond the spread.** A diagnostic in-sample logistic regression of home win on
  [spread, rating - spread] gives these t-statistics for the rating term:

  | rating | 2002-2011 | 2012-2025 | 2018-2025 |
  |---|---|---|---|
  | elo | 2.0 | 1.4 | -0.6 |
  | kalman | 2.5 | -0.1 | -2.0 |

  Ratings carried a little information beyond the closing line in the 2000s. Since about 2012, and especially
  since 2018, they carry none: the market fully prices team strength, QB changes, rest and travel. The
  walk-forward blends learn this and shrink the rating weight toward zero.

## What helped and what didn't (all judged on 2002-2011 tuning logloss, LL)
* **Elo, untuned 538 defaults to tuned:** LL went from 0.6253 to 0.6214, and walk-forward SU from 64.4% to
  65.4%.
  * **Helped:**
    * QB adjustment: removing it costs about 0.003 LL.
    * MOV multiplier: removing it costs about 0.009 LL.
    * Bye bonus of 2 pts.
    * Travel at 0.5 pt per 1000 mi, larger than 538's 0.16.
    * K=17 and 1/3 regression to the mean.
    * Playoff multiplier of 1.0: 538's 1.2 hurt.
  * **QB parameters** (prior and memory):
    * Strong prior toward a below-average QB: -0.16 EPA/dropback, worth 250 dropbacks.
    * Short memory: evidence decays by 0.9 per game and 0.5 per new season.
    * Slow team baseline: EMA alpha 0.05.
  * **Did not help:**
    * Blending the EPA-implied margin into the Elo update: the w_score sweep was flat or worse for Elo.
    * Dynamic HFA (hfa_lr > 0) on 2002-2011, so it was left off. It looks mildly positive on later seasons,
      which is consistent with HFA declining, but it was not adopted.
* **Kalman.** This is the best pure model on the tuning window (LL 0.6185) and is as good as Elo out of sample.
  * Tuned values:
    * Process variance: q_day = 0.01 pts²/day.
    * Season boundary: shrink 0.7 toward the mean and add 16 pts² of variance.
    * Observation: blend 80% actual margin with 20% EPA-implied margin.
    * HFA is a state with no drift (q_hfa = 0 was chosen).
  * The offense/defense version on team points (`kalman.run_kalman_od`) gives a margin prediction identical to
    the net version (LL 0.62066 vs 0.62066), so it is not shipped separately. It also yields a total if anyone
    wants one.
* **Ridge/Massey.** This is the weakest pure model.
  * Tuned values: decay half-life 24 weeks, prior-season carry 0.7, three seasons of lookback, λ=3, margin
    capped at 24.
  * The EPA/play-margin rating is *less* predictive than the points rating on its own (LL 0.6315 vs 0.6268).
    Combined with points, QB and context features it reaches 0.624 walk-forward.
* **Tuning ratings for complementarity with the market didn't help out of sample.** I re-tuned Elo and Kalman to
  minimize the logloss of [spread, rating] on 2002-2011. That pushed QB weight to 1.2, bye to 3 pts and travel to
  1.0, and improved in-sample LL from 0.6106 to 0.6097. On 2012-2025 it gave no gain: elo_mt_mkt 0.6087 and
  kal_mt_mkt 0.6090 LL, at 66.5% and 66.4% SU. Not adopted.
* **Rolling-origin hyperparameter selection didn't help.** I ran 60 random Kalman configs and, for each test
  season, picked the best on the trailing 10 or 6 seasons. Result: 65.2% / 64.9% SU and LL 0.6245, the same as
  fixed tuning.
* **Training window and recency weighting of the probability mapping:** all within noise.

## Factors (`elo_factors.parquet`, `kalman_factors.parquet`)
The table is long-format: `game_id, feature, label, home, away`, for 2006-2026.

| feature | label | notes |
|---|---|---|
| `elo` | Elo rating | Pre-game, about 1505 average |
| `elo_margin` | Elo expected margin incl. HFA/rest/travel/QB (pts) | Home value is the margin; away value is its negative |
| `qb_value` | Starting QB rating (shrunk EPA/dropback × 35, pts) | Shrunk toward a backup-level prior, so typical values run about -3 |
| `qb_adj` | QB vs usual starter (pts) | Starter value minus the team's recent-starter baseline. Strongly negative when a backup starts |
| `kalman_strength` | Kalman strength (pts vs avg team) | |
| `massey_pts` | Opponent-adjusted margin rating (pts) | |
| `massey_epa` | Opponent-adjusted EPA/play rating (pts) | |
| `rest_days` | Rest days | |
| `travel_miles` | Travel distance (miles) | |
| `hfa` | Home-field advantage (pts) | Home only; 0 at neutral sites |

**Importance** (rough share of the Elo margin's variance, 2012-2025):
1. Elo rating difference, which dominates.
2. QB adjustment: about 1.3 pts mean absolute value per team.
3. HFA: 2.2 pts.
4. Bye/rest: ±2 pts when one team is off a bye.
5. Travel: ≤ 1.5 pts except for international games.

## Recommendation for the stack
* Use the **pure ratings** (`elo`, `kalman`, `ratings_ridge`, or just `ratings_combo`) as stacker inputs. They
  are well calibrated (LL about 0.623), are fit walk-forward from 2006, and are only about 0.88-0.90 correlated
  with the market, so they are the most diverse signal this family can offer.
* For a stand-alone pick from this family, use `elo_mkt` (66.5% SU, LL 0.6087). Expect market-level accuracy.
* The `_mkt` variants are near-duplicates of the market and add little to a stack that already has the spread.

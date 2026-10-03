# closegames: market microstructure, a close-game specialist, and the SU ceiling (wave 2)

**Short version.**
* **Market-only model.** The best one I could build is a recency-weighted logistic on the spread plus the
  spread juice. It *is* the market: 66.43% SU over 2012-2025, logloss 0.6087. That ties the best wave-1 logloss.
* **Moneyline, totals and key numbers add nothing for SU** beyond the closing spread.
* **Close-game specialist.** It was chosen on seasons ≤ 2017 and then frozen. It looked marginally positive in dev
  (close games 57.5% vs 57.2%) and **lost on the clean 2018-2025 holdout** (55.9% vs 56.4% for the market).
* **The 68% target is mostly out of reach for a market-level model.**
  * A perfectly calibrated market-level model has an expected accuracy of 66.7% over 2012-2025.
  * Its chance of reaching 68% over those 14 seasons is about **4%**.
  * Getting an *expected* 68% would need private information worth about 3 points of margin SD that the
    closing line does not have. That is about 7% of the game-outcome variance.

## Files
```
python research/closegames/run.py [--check-leakage]   # ~12 s (~25 s with the leakage test); deterministic
```

| file | what |
|---|---|
| `preds/wave2_market.csv` | Best market-only model. `p_home` = walk-forward logistic on [logit Φ(spread/13.5), logit(no-vig spread-juice cover prob)], recency half-life 6 seasons, fit on 1999..S-1. `p_home_cover` = walk-forward logistic of cover on spread (dog/home-fav tilt). `margin` = spread + 13.5·Φ⁻¹(p_home_cover) |
| `preds/wave2_close.csv` | Same as wave2_market except for close games (\|spread\| ≤ 3.5). There `p_home` comes from the close-game specialist. `margin`/`p_home_cover` are unchanged, because the specialist is SU-only |
| `preds/wave2_close_factors.parquet` | 14 factors per game: `game_id, feature, label, home, away, home_text, away_text, better, fmt`. Kicker names and coach names are in `*_text` |
| `closegames/market.py` | Market features: no-vig moneyline, no-vig spread-juice, kernel (key-number) and normal spread maps, total-scaled map. Also the market walk-forward |
| `closegames/features.py` | Pre-game close-game features: kicker FG over expected (all and 45+), coach 4th-down go rate, coach experience, 2-minute and late-and-close EPA, one-score record, venue |
| `closegames/specialist.py` | Offset-logistic specialist (market log-odds as offset, L2 on everything else). It also loads wave-1 base preds |
| `closegames/dev_market.py`, `dev_close.py` | **Dev-phase** selection scripts. They load seasons ≤ 2017 only (asserted). Grids are in `out/dev_*_grid*.csv` |
| `closegames/out/` | `eval_output.txt` (eval.py), `eval_dev_holdout.csv`, `ceiling_by_season.csv`, `microstructure.csv`, `specialist_coefs.csv` |

The preds files cover **2006-2026**: 5,510 rows, every game with a spread_line, including the 31 unplayed 2026
week 4-5 games. Fallbacks:
* The specialist needs 3 training seasons from 2006, so 2006-2008 close games use the market model.
* A missing moneyline or juice falls back to the spread (the juice term is 0).
* Missing base-model preds would contribute 0. They are not used by the frozen specialist anyway.

## Validation protocol (what was honest and what wasn't)
* **Dev = seasons ≤ 2017.**
  * `dev_market.py` and `dev_close.py` call `load_schedule(max_season=2017)` and `features.build(..., 2017)`.
    Neither 2018+ schedules nor 2018+ pbp were loaded during selection.
  * The wave-1 base-pred CSVs do contain 2018+ rows, but they are joined onto the ≤ 2017 schedule, so those rows
    were never touched.
* **Dev test seasons:** walk-forward 2009-2017 for the market model, 2010-2017 for the specialist (it needs 3
  training seasons from 2006).
* **Freeze.** The settings were frozen in `run.py` (`MARKET_CONFIG`, `COVER_CONFIG`, `SPECIALIST_CONFIG`). Then
  2018-2025 was scored **once**. Nothing was changed after seeing the holdout.
* **Sweep size (disclosed).** On the dev window I evaluated:
  * about 48 market configs
  * about 230 specialist configs: 10 feature groups plus combinations, × L2 strength 1-5000, × free slope or not,
    × intercept or not, × training start 2001 or 2006
  * The chosen specialist is the best dev logloss out of ~230. Its dev gain is therefore optimistic by
    construction, and the holdout confirms that.
* **Prior knowledge (also disclosed).** Before starting I had seen the brief's 2012-2025 close-game numbers and
  the wave-1 NOTES, which include 2012-2025 results. I did not use them to pick settings.
* **Leakage.**
  * Every feature is an as-of decayed average over events dated strictly before the game day.
  * FG make expectation is fit on the 3 previous seasons. The 4th-down baseline is the previous season's league
    rate. The kicker is the team's kicker in its most recent earlier game.
  * `run.py --check-leakage` rebuilds everything from data ≤ 2017. All 2017 features and predictions are
    bit-identical to the full build (max |diff| = 0). **PASS.**
  * Nothing scored above 67.5% SU or 54% ATS, so no stress flag applies.

## 1. Market microstructure (dev findings, holdout confirmed)

**Data quality** (2006-2026, 5,510 games with a spread):
* **Pick'em.** `spread_line == 0` occurs only **5 times** since 2006. eval.py's home-on-pick'em rule is
  therefore irrelevant. True pick'em games are listed at ±1.
* **Missing moneylines and spread odds.** 136 games are missing them: 2006 (47), 2008 (73), 2009 (14), plus one
  each in 2007 and 2017. Every model falls back to the spread for these.
* **Bad moneyline.** One book has a negative overround: 2020_01_LV_CAR, +134/-124. Its moneyline is dropped.
* **No spread sign errors.** No game has a moneyline favourite opposite to a spread of 2.5 or more. 43 games have
  the moneyline side differing from the spread side, all at |spread| ≤ 1.
* **Spread juice is informative about the price but not the outcome.** Juice varies a lot: only 4-6% of
  pre-2022 games are at -110/-110, and 16-37% of games after 2022.
  * The raw no-vig juice-implied cover probability picks covers **49.8%** (dev 2009-17).
  * As a feature it improves win-prob logloss by only about 0.0001.

**Which side the moneyline favours in close games.** Win rate of each side among games with a moneyline:

| \|spread\| | period | n | spread fav | ML fav | juice side | ML ≠ spread side |
|---|---|---|---|---|---|---|
| 0-1 | 2006-17 dev | 326 | 54.9% | 54.3% | 55.5% | 32 |
| 0-1 | 2018-25 holdout | 144 | 44.4% | 43.8% | 47.2% | 11 |
| 1.5-2.5 | 2006-17 | 296 | 52.0% | 52.0% | – | 0 |
| 1.5-2.5 | 2018-25 | 363 | 55.7% | 55.7% | – | 0 |
| 3-3.5 | 2006-17 | 766 | 59.0% | 59.0% | – | 0 |
| 3-3.5 | 2018-25 | 532 | 60.0% | 60.0% | – | 0 |

* **Pick'em games are coin flips.** When the moneyline disagrees with a ±1 spread, it is a ~1-cent disagreement
  and it is right about half the time.
* **The 2018-25 ±1 favourite won only 44%** (n=144). That is a 1.3-SE fluke in the "wrong" direction, not
  something a model could have known.
* **Moneyline vs spread as a probability source** (dev 2009-17 logloss): raw no-vig moneyline 0.6079, while a
  recalibrated spread scores 0.6071-0.6073. Adding the moneyline's deviation from the spread does not help
  (0.6074-0.6076). The nflverse moneyline is a slightly *worse* probability source than the closing spread.

**Spread → probability map** (dev logloss, 2009-17 walk-forward, n=2,398):

| map | logloss |
|---|---|
| logistic on logit Φ(spread/13.5), fit from 1999 | 0.60726 |
| + spread-juice logit | 0.60717 |
| + recency weighting (half-life 6) **[chosen]** | **0.60695** |
| half-life 2 / 4 | 0.60687 / 0.60690 (flat; 6 kept on prior grounds, as in ratings/) |
| nonparametric kernel map (key-number aware), bandwidth 0.5-2.5 | 0.6079-0.6083 |
| + total-scaled spread (spread·√(44/total), "low totals → spreads matter more") | 0.6079-0.6098 (worse) |
| + total main effect / spread×total interaction | 0.6074 / 0.6080 |
| raw no-vig moneyline | 0.60792 |

* **Key numbers matter for covers, not for winning.** P(home win | spread) is smooth in the spread, and the
  kernel map only adds noise.
* **Margin variance does not depend on the total.** The SD of result − spread is 13.2-13.7 in every total bin
  from <38 to >50 (1999-2017). In the logistic, the spread×total interaction has t = 0.25. The "low total →
  spread matters more" idea is not supported.
* **The market has sharpened over time.** The logistic slope on the spread rose from 0.130 (1999-05) to 0.141
  (2006-11) to 0.157 (2012-17). That is why recency weighting helps a little.

## 2. Close-game specialist (|spread| ≤ 3.5, about 47% of games)
**Design.** An offset logistic regression is fit only on close games of seasons 2006..S-1, scored on season S.
* The market model's log-odds is a fixed offset with slope 1.
* All other inputs are standardised and L2-penalised, so they can only move the market probability.

**Univariate dev diagnostics** (close games 2006-2017, n=1,447). Each row regresses the residual (home_win − p_mkt)
on the standardised feature (HC1 t-stats):

| input | t (2006-17) | t 06-11 / 12-17 |
|---|---|---|
| wave-1 pure models (elo, kalman, ridge, combo, logit_epa, personnel_nomkt), each minus the market | +1.0 to +1.9 | mixed, all < 2 |
| wave-1 market-inclusive models (minus market) | −0.3 to +1.4 | sign flips (logit_epa_mkt +1.1 / −1.9) |
| d_coach_go4 (4th-down aggressiveness, coach) | +1.57 | +0.5 / +1.9 |
| d_coach_log_games (coach experience) | −2.20 | −1.0 / −2.0 |
| kicker FG over expected (all / 45+ / × wind) | +0.4 / +0.1 / +0.5 | sign flips |
| 2-minute off/def EPA | 0.0 | – |
| late-and-close off/def EPA | +0.6 / +0.9 | sign flips |
| one-score record | −0.5 | noise, as expected |
| dome / wind / cold / neutral | +1.5 / −1.4 / −0.2 / +0.6 | unstable |
| home intercept in close games (beyond market) | −0.9 | – |

**What a wave-1 model does in close games** (dev 2006-2017, close games). The market favourite wins 56.7%.
* Pure models disagree with the market in about 25% of close games. They pick only 52.7-55.8% correctly.
* Market-inclusive models score 55.6-56.8%.
* So on its own, no base model knows more than the line in close games.

**Walk-forward dev grid** (close games 2010-2017, n=1,014; market 57.20% / ll 0.67860). Full grid:
`out/dev_close_grid_*.csv`.
* Every group except "coach" *hurt* logloss. That includes base-model deltas (individually or averaged), kicker,
  late-game EPA, venue, one-score record and market microstructure.
* "All features" hurt logloss by +0.003 to +0.016 depending on the penalty.
* Best dev config: coach group (go-rate and experience), λ=50, no intercept. Close-game SU went 57.50% vs 57.20%
  (27 flips), and logloss 0.67668 vs 0.67860. **This was frozen.**
* Coefficients in recent folds (per SD): go-rate about +0.05, experience about −0.02 to −0.07 (log-odds). These
  are tiny, and the experience effect shrinks every year.

**Holdout result: the specialist does not survive.**

| window | model | SU | logloss | ATS | close-game SU (n) | close ll | pick'em (\|s\|≤1.5) SU |
|---|---|---|---|---|---|---|---|
| dev 2010-17 | market favourite | 66.59% | – | – | 57.20% (1014) | – | 51.9% |
| dev 2010-17 | wave2_market | 66.59% | 0.6102 | 50.8% | 57.20% | 0.6786 | 51.9% |
| dev 2010-17 | wave2_close | **66.73%** | 0.6093 | 50.8% | **57.50%** | 0.6767 | 52.3% |
| **holdout 2018-25** | market favourite | 66.20% | – | – | 56.35% (1040) | – | 48.7% |
| **holdout 2018-25** | wave2_market | 66.20% | **0.6087** | **51.4%** | 56.35% | 0.6822 | 48.7% |
| **holdout 2018-25** | wave2_close | 65.98% | 0.6095 | 51.4% | **55.87%** | 0.6841 | 47.0% |
| 2012-25 | wave2_market | 66.43% | 0.6087 | 51.0% | 56.88% (1809) | 0.6802 | 50.1% |
| 2012-25 | wave2_close | 66.35% | 0.6087 | 51.0% | 56.72% | 0.6801 | 49.2% |

The specialist's dev gain (+0.3 pt in close games) became −0.5 pt on the holdout. That is consistent with zero
signal plus selection from ~230 configs. **Recommendation:** treat wave2_close as a negative result. Give it at
most a tiny stacking weight. Use `wave2_market` as the market input.

## 3. eval.py output (`out/eval_output.txt`)
```
== preds/wave2_market.csv
   {'window': '2012-2025', 'n': 3816, 'missing': 0, 'su_acc': 0.6643, 'market_su': 0.6643, 'logloss': 0.6087, 'brier': 0.2107, 'ats_n': 3736, 'ats_acc': 0.5099}
   {'window': '2018-2025', 'n': 2219, 'missing': 0, 'su_acc': 0.662, 'market_su': 0.662, 'logloss': 0.6087, 'brier': 0.2107, 'ats_n': 2175, 'ats_acc': 0.5136}
   {'window': '2022-2025', 'n': 1136, 'missing': 0, 'su_acc': 0.6752, 'market_su': 0.6752, 'logloss': 0.606, 'brier': 0.2097, 'ats_n': 1110, 'ats_acc': 0.5072}
   season  SU    mkt   ATS
   2012  0.647 0.647 0.519
   2013  0.711 0.707 0.481
   2014  0.673 0.673 0.521
   2015  0.629 0.629 0.541
   2016  0.653 0.653 0.485
   2017  0.693 0.697 0.483
   2018  0.657 0.657 0.512
   2019  0.643 0.643 0.533
   2020  0.672 0.672 0.506
   2021  0.623 0.623 0.530
   2022  0.663 0.663 0.518
   2023  0.674 0.674 0.517
   2024  0.705 0.705 0.480
   2025  0.658 0.658 0.514
== preds/wave2_close.csv
   {'window': '2012-2025', 'n': 3816, 'missing': 0, 'su_acc': 0.6635, 'market_su': 0.6643, 'logloss': 0.6087, 'brier': 0.2107, 'ats_n': 3736, 'ats_acc': 0.5099}
   {'window': '2018-2025', 'n': 2219, 'missing': 0, 'su_acc': 0.6598, 'market_su': 0.662, 'logloss': 0.6095, 'brier': 0.2111, 'ats_n': 2175, 'ats_acc': 0.5136}
   {'window': '2022-2025', 'n': 1136, 'missing': 0, 'su_acc': 0.6734, 'market_su': 0.6752, 'logloss': 0.6068, 'brier': 0.2101, 'ats_n': 1110, 'ats_acc': 0.5072}
   season  SU    mkt   ATS
   2012  0.647 0.647 0.519
   2013  0.707 0.707 0.481
   2014  0.677 0.673 0.521
   2015  0.633 0.629 0.541
   2016  0.653 0.653 0.485
   2017  0.697 0.697 0.483
   2018  0.645 0.657 0.512
   2019  0.639 0.643 0.533
   2020  0.672 0.672 0.506
   2021  0.627 0.623 0.530
   2022  0.663 0.663 0.518
   2023  0.663 0.674 0.517
   2024  0.709 0.705 0.480
   2025  0.658 0.658 0.514
```
* **ATS.** The market-only cover model (a spread-only logistic that learns the dog tilt) is 51.0% over 2012-25
  and 51.4% on the holdout. That is below the 52.4% break-even.
* **Juice.** In dev, the juice-implied cover probability picked 49.8% of covers. It carries no ATS information.

## 4. The SU ceiling: what a perfectly calibrated market-level model can expect (`out/ceiling_by_season.csv`)
**Method.**
* Each game's chance of being picked correctly is q = max(p, 1−p), where p is the walk-forward market-only
  probability. That probability is calibrated: 2012-25 expected accuracy is 66.7% and the actual is 66.4%.
* Season accuracy is then a Poisson-binomial. P(≥ 68%) is computed exactly by dynamic programming.
* This is the distribution of results for a model exactly as good as the market. It uses no outcomes. Ties are
  excluded, as in eval.py.

| season | n | E[acc] | SD | market fav actual | P(≥68%) | share \|s\|≤3.5 | E[acc] close games |
|---|---|---|---|---|---|---|---|
| 2006 | 267 | 67.1% | 2.8 | 60.3% | 0.38 | 42% | 58.3% |
| 2007 | 267 | 67.2% | 2.8 | 68.5% | 0.39 | 43% | 57.9% |
| 2008 | 266 | 66.5% | 2.8 | 67.7% | 0.32 | 42% | 57.7% |
| 2009 | 267 | 69.3% | 2.7 | 68.9% | 0.69 | 34% | 57.8% |
| 2010 | 267 | 65.4% | 2.9 | 65.9% | 0.18 | 47% | 58.2% |
| 2011 | 267 | 66.9% | 2.8 | 66.3% | 0.35 | 45% | 57.8% |
| 2012 | 266 | 65.8% | 2.9 | 64.7% | 0.23 | 49% | 58.4% |
| 2013 | 266 | 66.4% | 2.8 | 70.7% | 0.31 | 47% | 58.0% |
| 2014 | 266 | 67.1% | 2.8 | 67.3% | 0.39 | 43% | 58.6% |
| 2015 | 267 | 65.8% | 2.9 | 62.9% | 0.23 | 51% | 58.9% |
| 2016 | 265 | 64.9% | 2.9 | 65.3% | 0.13 | 53% | 58.5% |
| 2017 | 267 | 67.0% | 2.8 | 69.7% | 0.36 | 46% | 58.3% |
| 2018 | 265 | 67.4% | 2.8 | 65.7% | 0.40 | 46% | 58.5% |
| 2019 | 266 | 68.0% | 2.8 | 64.3% | 0.52 | 41% | 58.6% |
| 2020 | 268 | 67.5% | 2.8 | 67.2% | 0.42 | 43% | 58.4% |
| 2021 | 284 | 69.0% | 2.7 | 62.3% | 0.63 | 41% | 59.1% |
| 2022 | 282 | 65.8% | 2.8 | 66.3% | 0.23 | 51% | 58.0% |
| 2023 | 285 | 65.7% | 2.8 | 67.4% | 0.22 | 54% | 58.9% |
| 2024 | 285 | 65.8% | 2.8 | 70.5% | 0.22 | 48% | 58.8% |
| 2025 | 284 | 67.3% | 2.7 | 65.9% | 0.38 | 50% | 59.3% |

| window | n | E[acc] | SD | actual market | P(window ≥ 68%) | expected # seasons ≥ 68% | actual # seasons ≥ 68% |
|---|---|---|---|---|---|---|---|
| 2012-2025 | 3816 | **66.7%** | 0.75 pt | 66.4% | **3.9%** | 4.7 of 14 | 3 (2013, 2017, 2024) |
| 2018-2025 | 2219 | 67.1% | 0.98 pt | 66.2% | 16.9% | 3.0 of 8 | 1 |
| 2012-2017 | 1597 | 66.2% | 1.16 pt | 66.8% | 5.9% | 1.7 of 6 | 2 |

**What this means for the owner.**
* **Single seasons swing a lot.** A model exactly as good as the market has an SD of about **±2.8 points** per
  season.
  * It hits 68%+ in roughly **1 season in 3** (P = 0.13-0.69, depending on how lopsided that year's spreads
    are).
  * A 70% season (2013, 2024) is ordinary luck, and so is a 62% season (2021). One season says almost nothing
    about model quality.
* **A 14-season average of 68% is a different matter.** A market-level model has about a **4% chance** of
  getting there. A 68% headline over 2012-2025 would therefore be strong evidence of real edge, or of leakage.
* **Close games cap everything.** The market's own expected accuracy in |spread| ≤ 3.5 games is only 58-59%, and
  those games are 41-54% of each season.
  * Reaching 68% overall needs close games at about 61-62%, while the market *expects* 58.5%.
  * That means beating the closing line's own probability by 3 points, on the games where it is least certain.
* **Information needed** (normal-margin approximation, σ calibrated so that a zero-information model reproduces
  the market's 66.7%; σ_eff = 11.6 pts, raw residual SD 12.9):
  * An *expected* 68% needs a model that knows, before kickoff, a component of the margin with SD of about
    **3.0 points** that the closing line does not price. That is about **6.6%** of the margin variance.
  * For comparison, the wave-1 pure models carry roughly 0-0.5 point of independent signal (their t-stats
    beyond the spread are ≤ 2, and about 0 since 2018).
  * Public box-score information is essentially fully priced. A 3-point information edge over the close would
    be a very large betting edge: it would also give roughly 55%+ ATS.

## 5. Factors (`wave2_close_factors.parquet`)
Rows by importance for this family. The first five carry essentially all of the prediction; the rest are
context and showed no out-of-sample value.
1. `spread`
2. `win_prob` (final)
3. `market_prob`
4. `ml_prob` (no-vig moneyline)
5. `spread_juice` (no-vig cover probability from juice)
6. `close_game` flag (specialist active)
7. `coach_go4` (coach 4th-and-≤3 go rate vs league; coach names in `*_text`). This is a specialist input,
   about +0.05 log-odds per SD.
8. `coach_games` (head-coach career games). Specialist input, about −0.02 to −0.07 log-odds per SD.
9. `kicker_fgoe` (kicker names in `*_text`) and `kicker_fgoe_long`
10. `two_min_off`, `two_min_def`, `late_close_net`
11. `one_score_record`. Shown because fans ask about it. It is luck, with no predictive value (t = −0.5).

## Recommendations for the stack and the owner
* **`wave2_market`** is a clean, calibrated market input (LL 0.6087 on the holdout), with moneyline and juice
  fallbacks handled. Use it as the stacker's market column instead of Φ(spread/13.5).
* **Don't expect close-game specialisation to unlock SU.** The closing line is efficient where it is 50/50.
  Every close-game input tested here is noise out of sample: kicking, coaching, 2-minute, venue, weather, and
  wave-1 model disagreement.
* **Set the target accordingly.** 66-67% SU is the honest ceiling for public-data models. 68% over 14 seasons
  is a 4% luck event for a market-level model. Reporting single-season 68-70% results as "beating the target"
  would be reporting noise.

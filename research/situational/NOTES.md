# Situational / market-structure / neural-net family (`research/situational/`)

**The short version.** None of these three models beats the market's straight-up (SU) pick by a meaningful amount.
None is reliably profitable against the spread (ATS) either. All the strict checks passed: features are fixed
as of kickoff, training is walk-forward only, and shuffled labels score about 50%. The SU target of 68% is out of
reach for market-plus-situational information. Over 2012-2025 these models land at 66.3-66.5%, against 66.4% for
the market. Overall ATS is 50.4-51.2%, below the 52.4% break-even.

Two kinds of ATS signal are real. Underdogs (especially road underdogs) cover a little more often than favourites,
and a few weak angles keep the same sign in every era. But each is worth about 51-52%, under the cost of the vig.
Most popular situational "trends" do not hold up out of sample.

## Files / how to rebuild
```
python research/situational/run.py      # ~2 min (OMP threads capped at 2). Rebuilds features, all 3 models, factors, tables
python research/eval.py research/preds/{situational,mlp,knn}.csv
python research/situational/stress.py   # leakage/robustness checks (~3 min)
python research/situational/angles.py   # prints the ATS angle-by-era study
```
* `features.py`: one row per game. Team state is attached with an as-of join on kickoff date using
  `allow_exact_matches=False`, built from played games only, so no game ever sees its own result or a later one.
* `angles.py`: definitions of 52 classic ATS angles, plus a study of each one by era.
* `models.py`: the `situational`, `mlp` and `knn` models, all walk-forward.
* `run.py`: the single entry point. `stress.py`: the leakage tests.
* `out/`: `angles.csv`, `ats_by_edge.csv`, `screen_oos.csv`, `selected_features.json` (what each season's fit
  selected), `eval_output.txt`.
* Predictions go to `research/preds/{situational,mlp,knn}.csv` with columns
  `game_id, season, week, p_home, margin, p_home_cover`. Every game from 2006-2026 that has a spread is included:
  5,510 rows, of which 79 are 2026 games. That covers weeks 1-5, including the unplayed week 4-5 games that
  already have lines.
* Factor files go to `research/preds/<model>_factors.parquet` with columns
  `game_id, feature, label, home, away, better, fmt`. There are 15 situational factors per game: the spread, the
  no-vig moneyline probability, implied team totals, rest, travel distance, time zones crossed, kickoff on the
  body clock, road streak, last game's margin against the spread, season ATS rate, season average margin, QB
  starts, coach tenure, revenge flag, and referee home-cover tendency.

## Data and features (all computed before kickoff)
* **Market:** the logit of the no-vig moneyline probability (spread-implied before 2006, when moneylines start);
  the gap between the moneyline and the spread-implied probability; the imbalance in spread juice; the vig;
  the total; implied team totals; key numbers 3 and 7; hook flags at 2.5/3.5/6.5/7.5 (signed by favourite);
  home dog; road favourite of 7+; favourite of 10+; pick'em; low and high totals.
* **Situational:** rest difference; bye and short-week flags; travel distance (haversine, from each team's
  stadium coordinates that season to the venue); time zones crossed; kickoff on each team's body clock and an
  early body-clock flag; a West-coast home team hosting an Eastern team at night; consecutive games away from
  home and home-after-road-trip; primetime, Thursday and Monday games; division games, rematches and revenge;
  head-to-head margin; look-ahead (a favourite with a division game next week); let-down (a favourite coming off
  an upset or blowout win); bounce-back after a 21+ point loss; out of contention (week 13+ with 3+ more losses
  than wins).
* **Weather and venue:** temperature, wind, cold and windy flags (outdoor games only); a dome team playing
  outdoors in the cold; a turf team playing on grass.
* **People:** coach tenure with the team and in his career, and first-year coach; QB career starts, a change of
  starting QB, and a QB with fewer than 8 starts.
* **Referee:** the crew chief's home-win residual against the market and home-cover residual, from that
  referee's earlier games only, shrunk with K=60. The referee column comes from schedules, which covers 1999+.
  `officials.parquet` covers only 2015+.
* **Form:** season ATS rate, last-3 ATS margin, ATS streak, last game's margin against the spread, season and
  last-3 point margin, win%, and the previous season's margin and ATS. These come from earlier games only and
  are shrunk.
* The EPA file from the efficiency agent (`features/team_game_features.parquet`) was tested but is **not used**.
  As an ATS input next to the spread structure it gave 51.4% over 2012-25, no better than the spread structure
  alone. Leaving it out keeps this family independent of the efficiency family.

## Models (each season S from 2006-2026 is fit on 1999 to S-1)
1. **`situational`**
   * *SU:* L2 logistic regression. The market logit enters as an **unpenalised** input. Situational features
     enter penalised, and only if they pass an *era-stability screen* inside the training window. The screen
     splits the training seasons into 4 blocks and keeps a feature only if its correlation with
     (home_win - market prob) has the same sign in all 4 blocks and the pooled |z| is at least 2. The penalty is
     chosen by rolling-origin validation on the last 4 training seasons.
   * *ATS:* a separate L2 logistic regression on home_cover. It always includes a spread-structure prior
     (spread, home dog, road dog, total) plus market, situational and angle features that pass the same screen
     on home_cover (|z| at least 1.5). Penalty chosen the same way.
   * The margin is spread + Φ⁻¹(p_cover)·13.5.
2. **`mlp`**
   * A two-output sklearn `MLPRegressor` with layers (16, 8), alpha 10 and early stopping, averaged over 5 seeds,
     on standardized market and situational inputs (≈110 features). Both targets are measured relative to the
     market: (home_win - market prob)/0.47 and (result - spread)/13.5.
   * How much to trust each output (a 0-1 shrink factor) is learned on an inner time holdout: fit on seasons
     before S-3, then score S-3 to S-1. For SU, p_home = market prob + k_win·output.
   * For ATS: residual = the spread-structure prior + k_cov·output. Both k values are mostly 0-0.3, which means
     the network rarely beats the market (see `selected_features.json`).
3. **`knn`**
   * Similar-games model: the 300 nearest past games in a weighted, standardized space (spread, total, rest,
     travel, time zones, division, primetime, form, previous season, week, QB experience, weather).
   * p_cover is the neighbours' cover rate, shrunk 60% toward 0.5. p_home is 0.5·market + 0.5·the neighbours'
     win rate.

## Final eval.py output (headline window 2012-2025)
| model | SU 12-25 | logloss | Brier | ATS 12-25 | SU 18-25 | ATS 18-25 | SU 22-25 | ATS 22-25 | SU 06-11* | ATS 06-11* |
|---|---|---|---|---|---|---|---|---|---|---|
| market baseline | 0.6643 | 0.6108 | 0.2115 | 0.5091 | 0.662 | 0.511 | 0.675 | 0.496 | – | – |
| situational | **0.6646** | 0.6104 | 0.2115 | 0.5104 | 0.6625 | 0.5200 | 0.6752 | 0.5243 | 0.658 | 0.516 |
| mlp | **0.6646** | **0.6098** | 0.2112 | 0.5043 | 0.6616 | 0.5067 | 0.6725 | 0.4973 | 0.660 | 0.518 |
| knn | 0.6630 | 0.6115 | 0.2118 | 0.5115 | 0.6616 | 0.5195 | 0.6743 | 0.5081 | 0.662 | 0.511 |

\*2006-2011 is scored with the same rules in my own scratch code, because eval.py only scores 2012+.

Per season (SU / ATS):
```
season  situational   mlp          knn          market SU
2012    .647 / .527   .647 / .508  .643 / .519  .647
2013    .711 / .450   .718 / .446  .699 / .419  .707
2014    .677 / .494   .665 / .533  .673 / .487  .673
2015    .629 / .521   .625 / .533  .625 / .541  .629
2016    .638 / .504   .649 / .488  .649 / .508  .653
2017    .704 / .486   .708 / .498  .700 / .529  .697
2018    .657 / .519   .660 / .504  .657 / .515  .657
2019    .643 / .510   .643 / .517  .647 / .521  .643
2020    .675 / .498   .675 / .539  .675 / .550  .672
2021    .623 / .534   .623 / .505  .616 / .537  .623
2022    .667 / .551   .656 / .522  .663 / .504  .663
2023    .677 / .524   .670 / .509  .667 / .513  .674
2024    .698 / .498   .709 / .473  .709 / .480  .705
2025    .658 / .525   .655 / .486  .658 / .535  .658
```

## ATS by edge (`out/ats_by_edge.csv`)
"Top x%" ranks games by |p_cover - 0.5|.
* **In-season** uses a cut-off computed from that season's own edge distribution, so it is a subset report
  only and could not be used live.
* **Previous-season cut** uses last season's out-of-sample quantile, so it could be used live.

| model | subset | 2006-11 | 2012-17 | 2018-25 | **2012-25** |
|---|---|---|---|---|---|
| situational | all | .516 (1559) | .497 (1561) | .520 (2175) | **.510 (3736)** |
| situational | top 50% in-season | .534 (782) | .495 (782) | .534 (1090) | **.518 (1872)** |
| situational | top 25% in-season | .548 (392) | .506 (393) | .546 (548) | **.529 (941)** |
| situational | top 10% in-season | .604 (159) | .459 (159) | .554 (222) | **.514 (381)** |
| situational | top 5% in-season | .617 (81) | .407 (81) | .540 (113) | **.484 (194)** |
| situational | top 25%, prev-season cut (live) | .538 (379) | .475 (238) | .536 (552) | **.518 (790)** |
| situational | top 10%, prev-season cut (live) | .606 (213) | .424 (106) | .528 (265) | **.499 (371)** |
| situational | edge ≥ 1 pt | .538 (773) | .488 (482) | .546 (496) | **.517 (978)** |
| situational | edge ≥ 2 pts | .531 (335) | .452 (155) | .667 (45) | **.500 (200)** |
| mlp | all | .518 | .501 | .507 | **.504 (3736)** |
| mlp | top 25% in-season | .524 (393) | .485 (394) | .509 (548) | **.499 (942)** |
| mlp | top 10% in-season | .572 (159) | .478 (159) | .563 (222) | **.528 (381)** |
| mlp | top 10%, prev-season cut | .512 (256) | .517 (178) | .508 (234) | **.512 (412)** |
| knn | all | .511 | .500 | .519 | **.511 (3736)** |
| knn | top 25% in-season | .528 (392) | .486 (393) | .507 (548) | **.498 (941)** |
| knn | top 10% in-season | .566 (159) | .447 (159) | .549 (222) | **.507 (381)** |
| knn | top 10%, prev-season cut | .577 (137) | .462 (143) | .546 (229) | **.513 (372)** |

**How to read this.** Every model shows the same pattern: high-edge subsets look great in 2006-11 and 2018-25 and
lose clearly in 2012-17. A real edge doesn't flip sign from one six-year block to the next. The best result,
situational top 25% (in-season cut), is 52.9% on 941 games over 2012-25 and above 50% in 10 of 14 seasons, but
it drops to 51.8% with a cut-off that could actually be used live. **There is no clearly profitable
high-confidence subset.** Several subsets go above 54%: situational 2018-25 top 10% at 55.4%, and situational
2006-11 top 5-10% at about 60%. Each comes from one era of 80-220 games, and the next era reverses it, so I read
them as noise.

## Which ATS angles survived honest testing (`out/angles.csv`, `out/screen_oos.csv`)
Each angle is scored on the side it favours, pushes excluded, in four eras: 1999-05, 2006-11, 2012-17 and
2018-25. 52 angles were tested. If the eras were independent, about 6 would show the same sign in all 4 eras by
chance (2/16 × 52).

**Consistent in every era (or its reverse is). These are the only ones I would call real, and none clears
52.4% by a safe margin:**
* **Road underdog: take the away team when the home team is favoured.** .510/.524/.506/.527 by era,
  51.7% on 4,604 games, z = 2.3. This is the familiar favourite/home bias. It is built into every model through
  the spread-structure prior.
* **Always take the underdog.** 51.3% on 7,050 games, z = 2.1. Same effect.
* **Playoff underdog.** .520/.569/.532/.560, 54.6%, but only n = 302 (z = 1.6). Plausible, small sample.
* **Revenge: the team that lost the earlier meeting this season.** .534/.514/.540/.508, 52.3% on 1,479,
  z = 1.8. Weak.
* **Reversed, consistent but weak:** the favourite at -2.5 loses ATS (47.6% for the favourite). A home team back
  from 2+ road games covers 48.8%.

**Holds in 3 of 4 eras, including both recent ones. Weak, and below break-even:**
* Fade a favourite that has a division game next week (look-ahead): 52.1%, n = 2,018.
* Fade the team that beat the spread most last week (market over-reaction): 51.5%, n = 2,997.
* Fade the team with the better season ATS margin: 51.0%.
* A team off a 21+ point loss: 51.4%.
* Reversed: the favourite at exactly -3 covers only 47.6%, so the dog at +3 covers 52.4%.

**Faded or died. Strong early, gone recently:**
* Fade a road favourite of 7+: 56.0%/60.3% in the first two eras, then 47.2%, then 56.8%. Unstable.
* A dome team on the road in the cold: 59-69% before 2012, then 48-51%.
* A West-coast home team hosting an Eastern team at night: 57-77% on small samples, then 50%/49%.
* The West-coast team in an early body-clock kickoff: 54-56% before 2012, then 47-49%. The market has priced it.

**No edge (about 50%, or the sign flips between eras):**
* Home dogs, including in division games and off a bye.
* Rest advantage, bye advantage, Thursday or Monday home teams.
* Travel distance, a 3rd straight road game, time zones.
* Let-down, division-rematch dog, big dog with a low total, dogs in wind or cold.
* First-year coach, new or young QB, out-of-contention teams.
* Referee home tendency, moneyline-vs-spread discrepancy, spread-juice direction.
* Favourites of 10+, -3.5, -7 and -7.5.
* Fading teams with better records or ATS streaks, and the previous season's strength in weeks 1-4.

**The walk-forward screen, tested directly out of sample** (`stress.py` step 4, `out/screen_oos.csv`). I ran
the ATS stability screen on 1999-2011 only. It kept 15 features, each with the same sign in 4 blocks and
|z| ≥ 1.5. Only **3 of the 15** had the same sign in both 2012-17 and 2018-25, which is what chance predicts
(15/4 ≈ 3.8):
* The three that held: temperature (colder favours the home team, z ≈ -1.2 in each period), key-3 and
  late-season. All are tiny.
* Travel distance and time-zone difference **reversed sign** after 2011: home teams now cover *less* when the
  visitor travels far. This fits the market now pricing travel, plus lower home-field advantage.
* The referee home-win residual looked strong in 2009-16 (within-season r = -0.05). It is zero in 2017-25.
* This is why the situational ATS model ends up at about 51%: most of what it learns is the dog/home prior.

## SU findings
* The market logit carries almost everything. Its slope in the SU model is 0.94-1.05 in every season.
* The SU stability screen kept few features. The most frequent across the 15 fits for 2012-2026: QB change this
  week (9 of 15, the home team starting a new QB is slightly worse than the market thinks), time-zone difference
  (7), key-3 spreads (6), travel difference (4), and the 2.5-point hook (3).
* For 2019, 2024 and 2026 nothing passed, and the model is just the market recalibrated.
* SU accuracy gains over the market are 0.0-0.3 points, which is noise, and logloss improves by about 0.001.

## Leakage / stress tests (`stress.py`)
1. **Truncation test: PASS.**
   * For 32 cutoff dates (7 fixed and 25 random from 2006-2026), I blanked every score on or after the cutoff,
     rebuilt the features, and compared every model input for the games on the first game day after it.
   * Every market, situational, angle, MLP and kNN feature was bit-identical to the full build.
2. **Shuffled labels:** I shuffled the residual within each season and refit the situational model walk-forward.
   It scored **49.8%** ATS on the real labels.
3. **Dropping the screened features** that looked strongest (season ATS, referee home-win and home-cover
   residuals): ATS goes from 51.0% to 51.0%, and top 25% goes from 52.9% to 51.5%.
4. **Screen out of sample:** covered above (3 of 15 survive, the chance rate).

## Honesty about tuning and sweeps
* **Chosen a priori (never tuned):** screen settings (4 blocks, z thresholds 2.0 for SU and 1.5 for ATS), kNN
  settings (k = 300, weights, 60% shrink), the 13.5 SD for spread and residual, the shrink constant K = 60 in the
  referee and form features, and the λ grid.
* **λ** is chosen per season by rolling-origin validation on training seasons only.
* **MLP architecture:**
  * I swept 12 configurations on raw targets (2 feature sets × 2 sizes × 3 alpha values), then 4 on the
    market-relative design. Every one was scored on 2006-2011 test seasons only, each fit on earlier seasons.
  * I chose (16, 8) with alpha 10, the best 2006-11 logloss. 2012-2025 was never used for this choice.
* **Situational variants:** I compared three on 2006-11 only.
  * SU: screened features beat market-only, so I used screening.
  * ATS: spread-prior-only and prior+screen tied on cover logloss (0.69276 vs 0.69285). I kept prior+screen
    because it is the model the brief asks for.
* **Decisions made after seeing 2012+ results.** Please discount these:
  1. Adding the spread-structure prior to the MLP's ATS output. Without it, the network leaned toward
     favourites and scored 48.3% ATS over 2012-25. With it, 50.4%. The prior learns the dog bias from training
     data only, but adding it was prompted by the 2012+ result.
  2. Making the kNN margin consistent with its cover probability. This changes `margin` only. Picks and
     p_cover are unchanged.
* **The angle study is descriptive and uses every era,** so its verdicts are not out-of-sample statements
  except where marked. The genuinely out-of-sample evidence is the walk-forward screen (step 4) and the
  per-era breakdown.

## Value for the stack
* All three models are about as well calibrated as the market and differ from it mainly in coin-flip games.
* `mlp` has the best logloss of the three (0.6098, against 0.6108 for the market).
* The ATS outputs carry a consistent "lean dog / road dog" tilt plus small situational adjustments. As base
  learners they are low-variance, near-market inputs, so expect small stacking weights.
* Next steps, if someone continues:
  * Line movement (open vs close) is not in the cache and is the most promising missing ATS input.
  * Injury reports could be added as situational inputs.

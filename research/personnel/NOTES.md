# Personnel family: QB value, injuries, availability

Entry point: `python research/personnel/run.py` (about 25 s; add `--analysis` to also print the
market-mispricing diagnostics). It rebuilds all features from `research/cache` and writes:

| file | what |
|---|---|
| `preds/personnel.csv` | market spread plus personnel adjustments (logistic for `p_home`; `margin` = spread + ridge ATS adjustment; `p_home_cover`) |
| `preds/personnel_nomkt.csv` | personnel only, no market input (QB ratings, injuries, rest, neutral site). Included for stacking diversity |
| `preds/personnel_ats.csv` | market spread plus a single "starters-out differential" ATS adjustment (**post-hoc feature choice**, see below) |
| `preds/<model_id>_factors.parquet` | per game: `game_id, feature, label, home, away, home_text, away_text, better, fmt` (16 rows per game; QB names in `*_text` of the QB rows, key absent players in the `key_absences` row) |

Predictions are walk-forward from **2006 through 2026**, as the protocol addendum asks. They cover every game
with a `spread_line` (5,510 rows), including the unplayed 2026 week-4 and week-5 games.
Training always uses seasons 2002 to S-1 for test season S. Supporting modules: `qb.py`, `injuries.py`,
`features.py`, `model.py`, plus `select.py` (feature-set selection), `stress.py` (leakage tests) and
`analysis.py` (market diagnostics). Intermediate tables and per-season coefficients are written to `personnel/work/`.

## 1. QB value model (`qb.py`)
* Per QB-game from pbp: QB plays = dropbacks (passes, sacks, scrambles; spikes excluded) + designed runs by that
  game's passers. Each play is scored with nflverse `qb_epa`. CPOE is averaged over pass attempts.
* Rating at date D uses only that QB's games with date < D. Each past game is weighted by
  `0.5^(days_ago / HL)`, and the result is shrunk toward replacement level:
  `(Σw·EPA + K·(ref + prior)) / (Σw·plays + K) − ref`. Here `ref` is the league QB EPA/play of the *previous*
  season, so the rating is relative to league average with no same-season information.
  Ratings carry across seasons and teams because they are keyed on the gsis id, which matches
  schedules `home_qb_id`/`away_qb_id`.
* Two time scales: `qb_epa` (HL 120 d, K 100, prior −0.20 = "form") and `qb_epa_long` (HL 400 d, prior −0.10 =
  "talent"). CPOE uses K 200 and prior −3.
* Flags: `qb_change_inseason` (starter ≠ the team's previous-game starter), `qb_epa_delta` (rating minus the
  previous starter's rating at the same date), `qb_backup` (started <50% of the franchise's last 8 games),
  `qb_rookie` (no QB plays before this season), `qb_log_exp`.
* Starting QB: schedules give the actual starter for past games and the scheduled starter for upcoming games
  (the protocol allows this).

## 2. Injury and availability load (`injuries.py`)
* Source: the final pre-game injury report per (season, week, team). I checked `date_modified` against kickoff:
  the median report is about 2 days before kickoff, and only 0.1% of rows were modified after kickoff (mostly
  2020 rescheduled games). Those rows are dropped.
* P(miss) by game status × Friday practice status. These are measured once from 2013-2025 snap-count
  participation (a calibration of what the labels mean, not fitted to outcomes):
  Out 1.0, Doubtful 0.97, Questionable+DNP 0.60, Q+Limited 0.32, Q+Full 0.20, Probable 0.05,
  no status+DNP 0.28.
* Importance, all from games strictly before the game:
  * `imp_usual`: the player's mean snap share over his last ≤6 appearances (any team, matched on name +
    position group, within 400 days).
  * `imp_recent`: mean snap share over the franchise's last 4 games (0 if absent).
  * Snap counts only exist from 2013. For 2009-2012 (and 2013 week 1) importance falls back to the
    previous week's depth chart (starter 0.85, 2nd string 0.30, deeper 0.10).
* Loads: Σ P(miss)·importance by group (QB, OL, SK = RB/WR/TE, DL, LB, DB), plus `inj_starters_out`
  (count of players with P(miss) ≥ 0.5 and imp_usual ≥ 0.6), `inj_q_load` (load from Questionable players) and
  `inj_total`. Before 2009 there are no reports, so loads are 0.
* Upcoming games whose week has no report yet (2026 week 5) use the team's latest report of the same season,
  week 4, flagged `inj_stale` in `work/games.parquet`. Week-4 games use the week-4 report.

## 3. Models (`model.py`)
* **personnel**: logistic regression on `[spread_line, d_qb_epa, d_qb_epa_long, d_qb_cpoe, d_inj_starters_out,
  d_inj_usual_QB, d_inj_usual_SK, rest_diff]` (d_ = home − away). The spread is standardised and multiplied
  by 50, so it is effectively unpenalised; the personnel terms are L2-shrunk. `margin` = spread + ridge fit of
  the ATS residual (result − spread) on the personnel terms, with no intercept, so ATS picks come only from
  personnel. `p_home_cover = Φ(edge/13.5)`.
* **personnel_nomkt**: the same personnel features plus a neutral-site flag, with no market input.
  `p_home` comes from the logistic and `margin` from ridge regression on the result.
* **personnel_ats**: margin = spread + b·d_inj_starters_out. `b` is a no-intercept OLS fit on 2009 to S-1 and
  is about −0.25 to −0.33 pts per starter in recent folds. `p_home = Φ(margin/13.5)`.
* Regularisation (logistic C, ridge α) is chosen inside every fold by rolling-origin validation on the last 3
  training seasons (nested). LightGBM was tried (regression and classification, 16-config grid). With the
  market it never beat the logistic on logloss (2006-11: ≥0.6136 vs 0.6125; 2012-25: ≥0.6108 vs 0.6082).
  Without the market it was equal at best. It was not shipped.

## 4. Final eval.py output (`work/eval_output.txt`)

| model | window | SU acc | market SU | logloss | brier | ATS acc (n) |
|---|---|---|---|---|---|---|
| personnel | 2012-2025 | **0.6640** | 0.6643 | **0.6082** | 0.2106 | 0.5091 (3736) |
| personnel | 2018-2025 | 0.6625 | 0.6620 | 0.6079 | 0.2105 | 0.5030 |
| personnel | 2022-2025 | 0.6761 | 0.6752 | 0.6055 | 0.2095 | 0.4883 |
| personnel_nomkt | 2012-2025 | 0.6308 | 0.6643 | 0.6364 | 0.2231 | 0.5139 |
| personnel_nomkt | 2018-2025 | 0.6246 | 0.6620 | 0.6363 | 0.2232 | 0.5131 |
| personnel_nomkt | 2022-2025 | 0.6259 | 0.6752 | 0.6321 | 0.2214 | 0.5108 |
| personnel_ats | 2012-2025 | 0.6640 | 0.6643 | 0.6103 | 0.2113 | 0.5046 |
| personnel_ats | 2018-2025 | 0.6616 | 0.6620 | 0.6097 | 0.2111 | 0.5039 |
| personnel_ats | 2022-2025 | 0.6743 | 0.6752 | 0.6082 | 0.2104 | 0.5090 |
| (market_baseline.csv) | 2012-2025 | 0.6643 | | 0.6108 | 0.2115 | 0.5091 |
| (spread-only walk-forward logistic) | 2012-2025 | 0.6640 | | 0.6086 | | |

Pre-2012 (2006-2011, n=1601): personnel SU 0.6633 / ll 0.6125; nomkt 0.6352 / 0.6417.

Per season (SU, market SU, ATS):

| season | personnel | nomkt | personnel_ats | market |
|---|---|---|---|---|
| 2012 | .650 / .550 | .654 / .511 | .647 / .488 | .647 |
| 2013 | .703 / .496 | .662 / .500 | .707 / .527 | .707 |
| 2014 | .673 / .513 | .658 / .521 | .673 / .517 | .673 |
| 2015 | .629 / .502 | .584 / .533 | .629 / .479 | .629 |
| 2016 | .649 / .534 | .641 / .500 | .649 / .519 | .653 |
| 2017 | .693 / .510 | .637 / .525 | .700 / .502 | .697 |
| 2018 | .653 / .504 | .657 / .562 | .657 / .488 | .657 |
| 2019 | .647 / .514 | .583 / .475 | .643 / .482 | .643 |
| 2020 | .672 / .520 | .627 / .539 | .672 / .535 | .672 |
| 2021 | .623 / .534 | .627 / .487 | .623 / .487 | .623 |
| 2022 | .667 / .464 | .631 / .525 | .660 / .456 | .663 |
| 2023 | .674 / .505 | .614 / .509 | .674 / .524 | .674 |
| 2024 | .705 / .523 | .653 / .516 | .705 / .534 | .705 |
| 2025 | .658 / .461 | .606 / .493 | .658 / .521 | .658 |

**These miss the 68% SU target.** With the market included, personnel information moves SU by about 0 and
logloss by about 0.0004 versus a spread-only logistic. The market already prices QBs and injuries almost fully.
The value of this family for the stack is (a) `personnel_nomkt`, a market-independent 63% model built only from
who is playing, and (b) the ATS injury signal below.

### Feature importance (final fold, trained 2002-2025; coefficient × feature SD)

| feature | nomkt logit/SD | nomkt margin pts/SD | personnel logit/SD |
|---|---|---|---|
| spread_line | – | – | +0.819 |
| d_qb_epa (form) | +0.364 | +2.80 | +0.031 |
| d_qb_epa_long (talent) | +0.246 | +2.11 | +0.022 |
| d_inj_starters_out | −0.091 | −0.73 | −0.029 |
| rest_diff | +0.059 | +0.45 | +0.011 |
| d_inj_usual_SK | −0.057 | −0.36 | −0.028 |
| neutral | −0.051 | −0.34 | – |
| d_inj_usual_QB | −0.033 | −0.30 | −0.011 |
| d_qb_cpoe | +0.024 | −0.02 | +0.007 |

Drop test: removing the QB features from nomkt drops 2012-25 SU from 0.631 to 0.571, so the QB ratings carry
the model.

## 5. Does the market misprice personnel? (`analysis.py`)
ATS residual = result − spread. The numbers below are descriptive, full-sample 2009-2025, HC1 SEs.

* **Cumulative injuries: the market under-reacts.** The starters-out differential has a univariate effect of
  −0.55 pts per SD (t −2.8), and the effect is monotone in the threshold. The team with ≥3 more starters out
  covers 44.5% ± 4.2% (n=535). By sub-period, the fade-the-injured-team win rate at ≥3 is
  2009-12 57.6%, 2013-16 57.1%, 2017-20 53.7%, 2021-25 54.4%. At ≥4 it is 57.7% (n=189). It is stronger when the
  injured team is the underdog (62%, n=197) and later in the season (wk ≥10: 59%).
  Walk-forward (`personnel_ats`, 2012-25):
  * |diff| ≥ 2: 52.2% (n=1184)
  * |diff| ≥ 3: 54.7% (n=435)
  * |diff| ≥ 4: 56.7% (n=157)
  * all games: 50.5%, because ties default to picking home
  
  **Caveat:** I chose this feature after seeing the full-sample residual table, so the 2012-25 ATS for it is
  not a clean out-of-sample number. The coefficient is walk-forward, and the ≥3 result is about z≈2 before
  correcting for selection. The feature itself was in the pre-registered "core" set, and the effect is
  monotone and the same sign in every sub-period, which argues against pure noise.
* **QB quality: slight under-reaction, not robust.** Better-rated QBs (long-term EPA) cover slightly more:
  +0.59 pts per SD univariate (t 3.1). But the team whose QB rating exceeds the opponent's by >0.15 EPA/play
  covers only 48.8%, and walk-forward residual models on QB ratings got about 50.9% ATS. No usable edge.
* **QB changes / late news: priced correctly.** In-season QB change: 50.4% cover. Downgrades of >0.10
  EPA/play: 51.3%. Starter returning: 49.4%. A starting QB listed Questionable who played: 48.4%. A listed QB
  who was replaced, by status (mean residual, all within 1 SE of 0):
  * Questionable: −1.05 ± 1.1
  * Doubtful: +0.1 ± 2.1
  * Out: −0.96 ± 0.9
  
  Heavy Questionable load ("game-time uncertainty") covers exactly 50.0%. The closing line absorbs late-week
  QB and game-time-decision news. There is no evidence of under-adjustment there. The only late-news proxy
  with a consistent sign is `d_inj_q_load` (−0.27 pts/SD, t −1.4), which is not significant.
* Positional loads: RB/WR/TE (SK) absences show the clearest under-reaction (t −2.2). OL and DL absences show
  none (OL coefficient ≈ 0).

## 6. Leakage checks (`stress.py`)
* **As-of rebuild**: QB and injury features recomputed with pbp and snap data truncated before the first game
  of the week (2014 wk8, 2019 wk10, 2023 wk15, 2025 wk3) match the full build exactly (max |diff| 0 / 1e-15).
* **Shuffle**: permuting personnel features within each season drops nomkt to 55.5% SU / ll 0.687, about
  home-team rate. The market model stays at market level (66.4%, ll 0.6092 vs 0.6082 unshuffled).
* Injury rows modified after kickoff are dropped. Snap-share importance uses `bisect_left` on dates, so
  same-day games are excluded.
* No result exceeded 67.5% SU or 54% ATS overall. The 54.7% ATS subset (|diff| ≥ 3) is a conditional,
  post-hoc slice and is reported as such.

## 7. Honest accounting of sweeps / selection
1. QB HL/K/prior grids (63 configs) scored on next-game QB EPA, 2001-2005 (no outcomes, no test seasons).
2. QB HL/K/prior grid (63 configs) scored on walk-forward game-outcome logloss of **2006-2011 test seasons**.
   This makes the 2006-2011 predictions slightly in-sample for these 3 numbers; 2012+ is clean. Full
   disclosure: an earlier setting (HL 300) that I had already scored on 2012-25 gave nomkt 63.9% SU
   (ll 0.6371). The rule-chosen HL 120 gives 63.1% (ll 0.6364). I kept the rule-chosen value.
3. Feature sets (`select.py`: core / full / full_recent, × 2 models) were chosen by **2006-2011** logloss. The
   2012-25 numbers were printed alongside. Per rule, `core` won for both models. full/full_recent scored
   2012-25 SU 0.6654/0.6648 (market) and 0.6352/0.6373 (nomkt), with logloss within 0.0003 of core.
4. Exploratory scripts (scratch, not shipped) printed 2012-25 results for: ~50 logistic feature × C
   combinations, 22 ATS residual configs (ridge/LGB), and a 32-config LightGBM grid. Nothing from them was
   adopted, except the `personnel_ats` feature choice (point 5).
5. `personnel_ats` uses one feature picked from full-sample (2009-2025) residual diagnostics. Its 2012-25 ATS
   is optimistic by construction.

## 8. Live picks / production notes
* Upcoming games use the scheduled QB from schedules and the latest injury report available. For 2026 week 5,
  that is the week-4 report, flagged stale. Re-running after the new week's report is in the cache
  refreshes them.
* Deterministic (no randomness in the shipped models). Thread env vars are forced to 1.
* Limitations:
  * Name matching between injury reports and snap counts (normalized name + position group) misses
    a few players.
  * Depth-chart importance for 2009-12 is coarser than snap shares.
  * The ≥2025 depth-chart format (timestamped) is not used. It isn't needed, because snap counts cover 2013+.

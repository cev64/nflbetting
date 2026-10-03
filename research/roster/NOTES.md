# Roster family (wave 2): bottom-up, player-level lineup strength

**Bottom line (honest):** valuing the players actually expected to play, unit by unit, does **not** add
information beyond the closing line.
* The market model `wave2_roster` matches the spread-only logistic on SU: 66.3% vs 66.2% over the 2018-2025
  holdout, a difference of 2 games out of 2,219. Its log loss is slightly *worse* (0.6095 vs 0.6084).
* Its ATS pick was 51.5% on dev (2012-17) but **47.6% on the 2018-2025 holdout**. The injury under-reaction
  that wave 1 found (and that looked real on ≤2017 data) did not persist after 2017 in this construction.
* The no-market model `wave2_roster_nomkt` is a reasonable market-free signal for the stack: 63.4% SU and
  0.6291 log loss on the holdout, versus personnel_nomkt's 62.5% / 0.6363 over the same window. Its ATS of
  51.2% is not usable at −110.
* Nothing reached the 68% SU target. Nothing scored above 67.5% SU or 54% ATS overall, so the extra leakage
  flags did not apply. The leakage tests were run anyway (§5).

## Files
```
python research/roster/run.py            # rebuild features + both models + factors (~25 s, deterministic)
python research/roster/run.py --report   # also print dev / holdout windows vs a spread-only reference
python research/roster/dev.py            # the dev (<=2017) comparison used to freeze the configuration
python research/roster/stress.py         # depth-chart timing test, truncation test, shuffle test (~2 min)
```
| file | content |
|---|---|
| `data.py` | Schedules and kickoffs (UTC). Snap counts mapped to gsis (pfr_id → name+team+season → name+season; 99.2% mapped). Pre-game depth charts (weekly 2001-24; 2025+ = latest timestamped snapshot strictly before kickoff). Final pre-game injury reports (rows modified at or after kickoff dropped; unpublished upcoming weeks carry the team's latest same-season report, flagged `stale`). |
| `players.py` | Player-game production log and point-in-time player values (as-of joins, strictly earlier dates). |
| `lineup.py` | Expected lineup per team-game, P(play), expected snap share, unit strengths, trailing lineups, injury cost, QB and kicker. |
| `games.py` | Game matrix (home−away differences). `model.py`: walk-forward models + scorer. `run.py`: entry point. |
| `work/` | `team_features.parquet`, `games.parquet`, `lineup_detail.parquet` (every candidate player with P(play), share and values), `p_play_table.parquet`, `calibration.json`, `*_coefs.json` (per-fold C/alpha/coefficients), `dev_output.txt`, `report_output.txt`, `eval_output.txt`, `stress_output.txt` |

Outputs:
* `preds/wave2_roster.csv` and `preds/wave2_roster_nomkt.csv`: 5,510 rows each. Every game from 2006 to 2026 with
  a `spread_line`, including the unplayed 2026 week-4 and week-5 games. Columns: `game_id, season, week,
  p_home, margin, p_home_cover`.
* `preds/wave2_roster_factors.parquet` (and an identical `_nomkt_factors`): 13 factors per game with schema
  `game_id, rank, feature, label, home, away, better, fmt`.

## 1. Player values (`players.py`)
Every player-game has an exposure `fge`, the "full-game equivalents" he played:
* 2013+: his share of his unit's snaps.
* Before 2013 (no snap counts): estimated from that week's depth chart: starter 0.85, second string 0.30, third
  0.10, off the chart but in the box score 0.25.

Production numerators are in EPA-like units:

| metric | definition | applies to |
|---|---|---|
| rec | receiving EPA | WR, TE, RB |
| rush | rushing EPA | RB |
| prush | 1.7·sacks + 0.6·QB hits | DL, LB |
| cover | 4.5·INT + 0.9·passes defended | DB, LB |
| rund | tackles for loss (solo tackles left out because box scores mix in special-teams tackles) | DL, LB, DB |
| ol | team pass protection while on the field: (league sack+½hit rate − own rate)·dropbacks·1.8, credited by exposure | OL |
| olexp | log(1 + decayed career exposure), i.e. experience | OL |

Value at kickoff date D uses only that player's games with date < D:
`v = (Σw·x − repl·Σw·fge) / (Σw·fge + K)`, with `w = 0.5^((D−d)/365 days)`. That is production per full game
above replacement, shrunk toward replacement (0) by exposure. It follows the gsis id across teams and seasons.
Replacement level and K for each (metric, position) come from **2013-2017 player stats only**:
* replacement = exposure-weighted rate of player-seasons with fewer than 4 full games;
* K = split-half (odd/even games) reliability, `K = n_half·(1−r)/r`. Examples: WR rec K≈10 games, RB rush ≈11,
  DL pass rush ≈4.6, DB coverage ≈21, RB receiving ≈96 (essentially noise).

No game outcomes enter this calibration.

* **QB:** EPA per dropback or designed run (nflverse `qb_epa`), relative to the previous season's league
  mean. Two versions: form (half-life 120 days, K 100 plays, prior −0.20) and talent (400 days, prior −0.10).
  These are the personnel family's settings. Also CPOE and sack rate (shrunk). The starter is the scheduled
  starter from schedules.
* **Kicker:** field goals made over expectation per attempt. The expectation is a logistic in distance fit on
  1999-2005, adjusted by the previous season's league residual. Shrunk with K=40 attempts, then converted to
  points per game (×1.7 attempts ×3).

## 2. Expected lineup and lineup delta (`lineup.py`)
For team-game G:
* **Candidates:** every player on the team's pre-game depth chart for G.
* **P(play):** a calibration table by (chart depth, injury-report class, played the team's previous game within
  30 days). It is measured on 2013-2017 snap participation, so it calibrates labels and uses no game outcomes.
  The result is divided by the healthy-starter rate, so a fully healthy lineup has P=1.
  * Example for depth-1 players: Out ≈0; Questionable+DNP ≈0.25-0.42; Questionable+Limited ≈0.59-0.84.
  * Not listed but missed last game: 0.48. Not listed and played last game: 0.98.
  * The "missed last game" bucket captures IR and inactive players, who do not appear on the report.
* **Expected share:** `s = (n·u + 1.5·depth_prior) / (n + 1.5)`. Here u is the mean exposure per appearance
  (half-life 60 days) and n is the decayed number of recent appearances.
* **Unit strengths:** `U[unit] = Σ P·s·v`. `F[unit]` is the same sum with everyone healthy.
  `inj[unit] = U − F` is the value lost to absences.
* **Trailing lineups:** `A[unit]` is the mean over the team's previous 6 games of the same construction
  (that game's P·s), **re-valued with ratings as of G**. This isolates who is playing from rating drift.
  `delta[unit] = U − A` is the "lineup delta".
* **QB:** `delta_qb` = scheduled starter's rating minus the mean rating of the last 6 starters, all rated as of G.
* **Counts:** `starters_out` = depth-1 players with P < 0.5.

Bugs found and fixed during development (before any holdout look):
* An unsigned-int subtraction in `d_starters_out` flipped its sign.
* "Played previous game" originally reached back to the previous season's finale, where good teams rest their
  starters. This created a team-quality artifact in the injury cost.
* Several tie-breaks were nondeterministic (depth-chart position ties, the PFR→gsis mode, odd/even split).
  `run.py` now reproduces byte-identical CSVs.

## 3. Models (`model.py`)
Walk-forward. For test season S, each model fits on 2003..S-1. Regularisation (logistic C, ridge α) is
chosen *inside each fold* by rolling-origin validation on the last 3 training seasons.
* **`wave2_roster`** (`kind=mkt`, feature set `inj`):
  * SU: logistic regression on `spread_line` (standardised ×50, so effectively unpenalised) plus
    `d_starters_out, d_inj_off, d_inj_def, d_inj_olexp, d_delta_qb` (L2).
  * ATS: a ridge fit, with no intercept, of (result − spread) on the same lineup features.
    `margin = spread + adj`, `p_home_cover = Φ(adj/13.5)`.
* **`wave2_roster_nomkt`** (`kind=nomkt`, feature set `full`):
  * SU: logistic regression on all unit deltas (7), injury costs (7), unit levels (7), `delta_qb`,
    `starters_out`, `k_val`, QB form/talent/CPOE/sack rate, rest difference and neutral site.
  * `margin` = ridge regression on the result; `p_home_cover = Φ((margin − spread)/13.5)`.

## 4. Honest validation: dev (≤2017) then a single holdout run (2018-2025)
**Development used only seasons ≤ 2017.** Test seasons were 2010-2017, walk-forward. Four pre-declared feature
sets (`agg`, `inj`, `units`, `full`) were compared with the market model, plus a LightGBM residual variant
(SU with the spread logit as init_score; ATS regression on the residual) and the no-market logistic. That is
8 configurations in total (`work/dev_output.txt`):

| dev config (test 2010-17) | SU | log loss | ATS | 2012-17 SU | 2012-17 ll | 2012-17 ATS |
|---|---|---|---|---|---|---|
| spread-only logistic (reference) | .6654 | .6103 | .4940 | .6669 | .6089 | .4933 |
| mkt `agg` | .6649 | .6115 | .5089 | .6669 | .6090 | .5112 |
| **mkt `inj` (frozen → wave2_roster)** | **.6664** | **.6109** | **.5099** | .6681 | .6084 | .5151 |
| mkt `units` | .6635 | .6115 | .4925 | .6669 | .6086 | .5016 |
| mkt `full` | .6635 | .6122 | .5084 | .6669 | .6092 | .5189 |
| gbm `units` | .6574 | .6178 | .4844 | .6600 | .6146 | .4907 |
| gbm `full` | .6565 | .6168 | .4911 | .6606 | .6132 | .5010 |
| **nomkt `full` (frozen → wave2_roster_nomkt)** | .6410 | .6342 | .5123 | .6475 | .6316 | .5151 |

Choice rule: the lowest 2010-17 log loss among market models (`inj`). LightGBM was worse on dev and was not
shipped. Selection disclosure:
* The `inj` set was added after seeing the first three sets' dev numbers.
* The bug fixes in §2 were made while looking at 2009-2017 univariate correlations.
* No 2018+ number was computed before the configuration was frozen. The holdout was then run once.
* After that one run I fixed a calibration tie-break nondeterminism. That fix moved the scores in the 4th
  decimal and changed no setting.

**Holdout (2018-2025, clean) and full windows, from `python research/eval.py`:**

| model | window | SU | market SU | log loss | Brier | ATS (n) |
|---|---|---|---|---|---|---|
| wave2_roster | dev 2012-2017 | 0.6681 | 0.6675 | 0.6084 | 0.2106 | 0.5151 (1561) |
| wave2_roster | **holdout 2018-2025** | **0.6629** | 0.6620 | **0.6095** | 0.2112 | **0.4759** (2175) |
| wave2_roster | 2012-2025 | 0.6651 | 0.6643 | 0.6090 | 0.2109 | 0.4922 (3736) |
| wave2_roster | 2022-2025 | 0.6752 | 0.6752 | 0.6050 | 0.2092 | 0.4712 (1110) |
| wave2_roster_nomkt | dev 2012-2017 | 0.6475 | 0.6675 | 0.6316 | 0.2204 | 0.5151 |
| wave2_roster_nomkt | **holdout 2018-2025** | **0.6341** | 0.6620 | **0.6291** | 0.2198 | **0.5122** (2175) |
| wave2_roster_nomkt | 2012-2025 | 0.6397 | 0.6643 | 0.6301 | 0.2201 | 0.5134 (3736) |
| wave2_roster_nomkt | 2022-2025 | 0.6488 | 0.6752 | 0.6224 | 0.2167 | 0.5099 (1110) |
| spread-only logistic (same folds) | holdout 2018-2025 | 0.6620 | | 0.6084 | 0.2106 | – |
| spread-only logistic | dev 2012-2017 | 0.6669 | | 0.6089 | 0.2109 | – |

2006-2011 (also walk-forward, not used for selection): wave2_roster SU 0.6577 (market 0.6627), log loss 0.6162,
ATS 0.4798. wave2_roster_nomkt SU 0.6340, log loss 0.6478, ATS 0.4978. The 2006-08 seasons have no injury
reports, so the lineup there is depth-chart-only.

Per season (SU / ATS):
```
season  wave2_roster   nomkt          market SU
2012    .650 / .504    .658 / .481    .647
2013    .707 / .527    .620 / .519    .707
2014    .677 / .475    .658 / .525    .673
2015    .629 / .510    .611 / .514    .629
2016    .645 / .534    .675 / .523    .653
2017    .700 / .540    .663 / .529    .697
2018    .657 / .481    .649 / .562    .657
2019    .650 / .475    .560 / .482    .643
2020    .672 / .468    .642 / .535    .672
2021    .623 / .498    .623 / .480    .623
2022    .663 / .427    .649 / .547    .663
2023    .674 / .491    .635 / .480    .674
2024    .705 / .502    .670 / .505    .705
2025    .658 / .465    .641 / .507    .658
```

### Does lineup information add anything beyond the closing line?
**SU: no.** In both dev and holdout, the market model's SU is within ±0.1 pt of a spread-only logistic.
Holdout log loss is 0.0011 worse.

**ATS: no, and the dev signal reversed in the holdout.** Correlations with (result − spread), descriptive only
(the holdout half was computed *after* freezing):

| feature (home − away) | 2009-2017 | 2018-2025 |
|---|---|---|
| starters likely out | −0.043 | −0.009 |
| injury cost, offence | −0.031 | +0.031 |
| injury cost, defence | −0.017 | +0.024 |
| injury cost, pass rush | −0.030 | +0.021 |
| lineup delta, defence | +0.003 | +0.037 |

On dev, fading the team with ≥3 more starters out won 56.8% (n=266). On the holdout it won 50.5% (n=392).
The model's top-25% ATS edges went from 53.7% on dev to 48.9% on the holdout.

I read this as the closing line having caught up with cumulative injuries after about 2017, and the
value-weighted injury costs being noise of either sign. The residual correlations are |r| < 0.04
(roughly 1-2 SE). Personnel's sub-period table (≥3 starters: 53.7% in 2017-20, 54.4% in 2021-25) used a
different, snap-share-based definition. With my depth-chart definition the effect is about zero after 2017.

**What survives:** `wave2_roster_nomkt` is a market-free model built only from who is playing. It is
well-shaped for stacking: 63.4% SU / 0.629 log loss on the holdout, slightly better than `personnel_nomkt` over
the same window (62.5% / 0.636).

### Feature importance (final fold, trained 2003-2025, standardised coefficients)
`wave2_roster` (C=0.03, α=3000):

| feature | SU logit/SD | ATS adjustment, pts/SD |
|---|---|---|
| spread_line | +0.876 | – |
| d_delta_qb | +0.053 | +0.23 |
| d_starters_out | −0.036 | −0.24 |
| d_inj_def | +0.031 | +0.00 |
| d_inj_off | +0.013 | −0.07 |
| d_inj_olexp | −0.013 | −0.11 |

`wave2_roster_nomkt` (C=0.01). The largest SU terms (logit/SD), with the margin ridge in pts/SD:

| feature | SU logit/SD | margin pts/SD |
|---|---|---|
| d_qb_epa | +0.341 | +2.5 |
| d_U_prush | +0.179 | +1.2 |
| d_qb_epa_long | +0.154 | +1.4 |
| d_U_cover | +0.154 | +1.2 |
| d_U_rush | +0.095 | +1.1 |
| d_U_ol | +0.063 | +0.3 |
| rest_diff | +0.062 | +0.4 |
| d_qb_cpoe | +0.061 | +0.4 |
| d_delta_prush | −0.058 | −0.3 |
| d_U_olexp | +0.049 | +0.3 |
| d_starters_out | −0.047 | −0.5 |
| d_inj_cover | +0.046 | +0.2 |

The unit *levels* (pass rush, coverage, run game, OL) carry the no-market model after the QB. The *deltas*
(week-to-week lineup changes) carry almost nothing.

## 5. Leakage checks (`stress.py`, `work/stress_output.txt`)
* **Depth-chart timing (2013-2024).** Take players who left game g early (≥60% of snaps in g−1, <30% in g)
  and missed g+1.
  * They are missing from the week-g chart only 4.1% of the time, against 2.3% for a matched low-snap group
    that played g+1 and 1.4% for regulars.
  * They are missing from the week-g+1 chart 17.0% of the time.
  * So the weekly chart labelled g is a pre-game snapshot. Injuries suffered in g do not appear on it.
  * 2025+ charts are timestamped, and the latest snapshot strictly before kickoff is used.
* **Truncation test.** Every outcome-bearing table (pbp, box scores, snap counts) was cut at a kickoff T, with
  depth charts and injury reports kept only up to T's week. The features of every game on T's date were then
  rebuilt.
  * Targets: 2011 wk 9, 2015 wk 12, 2019 wk 10, 2023 wk 15, 2025 wk 3, each tested at the Thursday slot and the
    Sunday slot.
  * All match the full build to ≤1e-15 with no null mismatches.
  * Games later in the same week (Monday) may legitimately use that week's earlier games: ratings use
    date < kickoff date.
* **Shuffle test** (dev, 2010-17). Permuting the lineup features within each season:
  * The market model falls back to market level (log loss 0.6114 vs 0.6109 real; ATS 49.8%).
  * The no-market model falls to 56.3% SU / log loss 0.687, about the home-win rate.
* **Labels and reports.** Injury rows modified at or after kickoff are dropped. The starting QB comes from
  schedules (the scheduled starter for upcoming games). Calibration constants (replacement/K, P(play) table)
  use 2013-2017 stat or participation labels only, never game results.

## 6. Live picks / limitations
* **Upcoming weeks.** 2026 week 5 has no injury report yet, so it uses each team's latest 2026 report (week 4),
  flagged `stale`.
  * The cached week-4 report is mid-week: almost no game statuses are set (7 players with P(miss) ≥ 0.5
    league-wide). Week-4/5 lineup features therefore understate absences until the cache refreshes.
  * Depth charts for upcoming games are the latest daily snapshot.
* **Pre-2013 exposure.** Before 2013, exposure comes from the depth chart, so player values are coarser.
  2006-08 have no injury reports, and P(play) there depends only on chart depth.
* **OL.** Offensive-line value is a team-level pass-protection proxy (no individual pressure data in the
  cache) plus experience.
* **FTN charting (2022+)** was not used: it is too short a history for walk-forward training under the
  ≤2017 dev rule.

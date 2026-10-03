# meta: the ensemble that powers the site

`python research/meta/run.py` takes about 4 s and is deterministic. It writes `preds/ensemble.csv` and `preds/ensemble_meta.json`.
- **`ensemble.csv` columns:** `game_id, season, week, p_home, margin, p_home_cover, ats_edge, best_bet, p_stack`.
  It covers 2007-2026, 5,243 rows, including the 79 2026 games that have a line.
- **`ensemble_meta.json` contents:** base models, the latest weights, the leaderboard, a per-context table, the by-season record, and the dev/holdout validation tables.

## Files
- `data.py`: builds one row per game. Each row holds every base model's prediction plus pre-game context:
  spread bucket, week flags, playoff flag, moneyline gap, the starters-out injury gap, and how many model families dissent.
  A model that is missing a game is treated as agreeing with the market.
- `designs.py`: the meta-learners.
  - SU: market recalibration, a penalised logistic stack (per model or per family, optional context interactions), LightGBM boosting from a market offset, BMA with time-decayed and optional per-context weights, a disagreement router, and a flip guard.
  - ATS: a logistic stack, LightGBM, and an equal-weight vote.
- `harness.py`, `dev.py`: the walk-forward search.
  - `dev.py su|ats` scores the grid on DEV.
  - `dev.py su|ats --holdout` runs `frozen.py` once on the holdout.
  - Tables are saved in `out_{su,ats}_{dev,hold}.csv`.
- `frozen.py`: the designs frozen after the dev search, with the production choice marked.
- `select_models.py`: a dev-only rule for adding new base models (wave2_*). It writes `models.json`, which run.py reads.

## Protocol
- DEV is test seasons 2010-2017. Each season S is fit on 2006..S-1. Every design and hyperparameter choice was made here, with dev log-loss as the criterion.
- The best configuration of each design type, and the production pick, were then frozen in `frozen.py`. They were run exactly once on HOLDOUT 2018-2025, also fit on 2006..S-1.
- Caveat: some base models tuned hyperparameters on 2002-2011. Their 2006-2011 predictions are therefore slightly in-sample. That inflates how much the meta-learner trusts them in early training, and it affects dev.
- Caveat: personnel's starters-out gap was chosen by its author after looking at 2009-2025 residuals. Any holdout gain from it is not clean.

## Results: dev 2010-2017 vs holdout 2018-2025
In these tables, "market" means the market favourite: 66.59% SU on dev, 66.20% on holdout.

**SU:**

| design | dev SU | dev LL | holdout SU | holdout LL | holdout flips (acc) |
|---|---|---|---|---|---|
| market recalibrated, hl 4 | .6654 | .6104 | .6620 | .6086 | 0 |
| flat per-model logit, α .03 | .6626 | .6096 | .6643 | .6091 | 9 (78%) |
| flat family logit, α .01 | .6621 | .6094 | .6652 | .6090 | 15 (73%) |
| contextual family logit, α .03 | .6631 | .6099 | .6656 | .6091 | 12 (83%) |
| contextual per-model logit | .6640 | .6101 | .6656 | .6097 | 14 (79%) |
| LightGBM on market offset, 100 trees | .6603 | .6105 | .6643 | .6085 | 41 (56%) |
| BMA, τ 1, hl 4 | .6645 | .6094 | .6634 | .6092 | 7 (71%) |
| contextual BMA, cells pickem/early/playoff | .6645 | .6099 | .6611 | .6124 | 46 (48%) |
| disagreement router, k 3 | .6649 | .6106 | .6620 | .6081 | 74 (50%) |
| guard over BMA, δ .02 | .6664 | .6093 | .6616 | .6092 | 1 (0%) |
| **PRODUCTION: guard over family logit, δ 1** | **.6659** | **.6092** | **.6620** | **.6091** | 0 |

On dev, every way of overruling the market favourite was right only 31-47% of the time. The router never found a context where dissent beat 50%. So the dev-optimal rule is "the stack sets the probability; the pick stays with the market favourite".

On holdout, the unguarded stacks' handful of flips went 73-83%, but on 9-15 games. Pooled over 2010-2025, the family stack's flips were 23 of 47, which is 49%. That is a coin flip, not a skill that was hidden by the guard.

**ATS** (top-N% uses the previous season's edge quantile, a cut usable live):

| design | dev ATS | dev top10 (n) | dev top20 | holdout ATS | holdout top10 (n) | holdout top20 |
|---|---|---|---|---|---|---|
| structure (home/road dog) + injury gap | .520 | .497 (187) | .523 | .519 | .566 (226) | .541 |
| per-model logit, α .03 | .520 | .577 (189) | .560 | .511 | .519 (212) | .522 |
| **PRODUCTION: family logit, α .01** | .520 | **.592 (206)** | .559 | **.509** | **.511 (219)** | .502 |
| family logit + context | .527 | .571 (217) | .539 | .512 | .554 (224) | .509 |
| mean cover signal | .521 | .591 (181) | .557 | .519 | .515 (231) | .526 |
| LightGBM | .525 | .511 (227) | .514 | .527 | .512 (211) | .504 |
| equal-weight vote | .507 | .563 (190) | .565 | .501 | .550 (202) | .500 |

The dev ATS "best bets" edge, about 59% on the top 10%, **did not survive the holdout**: 51.1%.
- On dev, the signal came from the no-market models (ratings, efficiency, personnel_nomkt) disagreeing with the line. On holdout it is gone. This matches the ratings family's finding that their information beyond the closing line vanished after about 2012-2018.
- Structure plus the injury gap looks best on holdout (56.6% top 10%). It was worst on dev, and its feature was chosen post-hoc. It is **not** a clean win and was not adopted.

Ablations, run for information only:
- The production model with no injury gap scores holdout top-10 .519.
- With no `personnel_ats` as well, it scores .525.

## eval.py, 2012-2025 (for comparability with wave 1)
```
{'window': '2012-2025', 'n': 3816, 'su_acc': 0.6643, 'market_su': 0.6643, 'logloss': 0.6085, 'brier': 0.2107, 'ats_n': 3736, 'ats_acc': 0.5139}
{'window': '2018-2025', 'n': 2219, 'su_acc': 0.662,  'market_su': 0.662,  'logloss': 0.6091, 'brier': 0.211,  'ats_n': 2175, 'ats_acc': 0.509}
{'window': '2022-2025', 'n': 1136, 'su_acc': 0.6752, 'market_su': 0.6752, 'logloss': 0.6065, 'brier': 0.2099, 'ats_n': 1110, 'ats_acc': 0.5126}
```
- The 2012-2025 log-loss is 0.6085, against 0.6108 for the market. The best single model, `personnel`, scores 0.6082, so the ensemble is not better than it on log-loss.
- The SU pick is identical to the market favourite.
- Best bets over 2012-2025: top 10% scores 54.2% (n=380) and top 20% scores 52.9% (n=730). Most of that comes from 2012-2017, which is part of dev. The clean holdout figures are 51.1% and 50.2%.

## Latest weights (fit on 2006-2025, used for 2026)
- **SU family coefficients**, on each family's mean log-odds disagreement with the market:
  - personnel (no-market QB/injury model): +0.148, the only material weight
  - ratings_mkt: +0.036
  - ratings: +0.027
  - situational: +0.018
  - personnel_mkt: +0.015
  - efficiency: ~0
  - efficiency_mkt: ~0
  - The market slope is about 1 after recalibration.
- **ATS coefficients:**
  - starters-out gap: −0.109 per 3 starters
  - playoff dog: +0.095
  - efficiency cover signal: +0.10
  - personnel cover signal: +0.106
  - ratings_mkt cover signal: +0.073
  - home dog: −0.044
  - spread: −0.030
  - Remaining family cover signals: ≤ 0.04
- `ensemble_meta.json → context` has the descriptive per-spread-bucket table for 2012-2025. In pick'em games the market is 52.7% right. When a no-market family dissents there, the family is right 47-48% of the time.

## Assessment vs the 68% target
The target is not reachable with these inputs, and no meta-design changes that.
- Market-level SU (about 66.4% over 2012-2025) is the honest ceiling of this ensemble.
- The only place to gain is the roughly 23% of games with spreads of 2.5 or less. The market is 52-54% there, and no base model or combination beats it out of sample.
- Reaching 68% overall would need about 60% in pick'em games with everything else held equal.

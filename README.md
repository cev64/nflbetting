# NFL Picks

A weekly NFL picks sheet. Every game gets a **straight-up winner** and an **against-the-spread** pick from a stacked
ensemble of 18 models. The site shows the data behind every pick and how each model compares. It is built on free
[nflverse](https://github.com/nflverse) data and refreshed every morning.

- **Website** (`web/`): static, no build step. It reads one file, `web/data/picks.json`.
  - **Picks:** the week's games with logos, the winner and spread picks, win and cover probabilities, how many models
    agree, and best bets. Played games show ✓/✗.
  - **Data behind the pick:** for each game, the ensemble against the market, every model's probability, and about
    30 inputs grouped as Market, Team strength, Efficiency, QB & health, Lineup and Situation (QB names and injured
    starters included). It also shows form, the line and the weather.
  - **Models:** the leaderboard, accuracy by season, accuracy by confidence, spread record by edge, calibration,
    what the ensemble listens to, how often models that argue with the market are right, and the best-bet record.
- **Models** (`research/<family>/run.py`): each family writes walk-forward predictions to `research/preds/*.csv`.
- **Pipeline** (`backend/pipeline.py`): fetch data → run every family → meta-model → `backend/build_picks.py` → `picks.json`.
- **Automation** (`.github/workflows/update-data.yml`): runs the pipeline every morning at 8am ET (plus Tuesday 10am
  ET for Monday night's play-by-play), commits `picks.json` and deploys to GitHub Pages. A full run takes about 5
  minutes with a 5 GB memory peak.

## How good is it? (honest numbers)

Every number below is **walk-forward and out of sample**: each season is predicted by models fit only on earlier
seasons, using only information available before kickoff. `research/eval.py` scores every model the same way.

| 2012–2025 (3,816 games) | Straight-up | Against the spread | Log loss |
|---|---|---|---|
| **Ensemble** (the picks) | **66.4%** | **51.4%** | **0.609** |
| Betting favorite (market) | 66.4% | 50.9% | 0.611 |
| Best single model with the line (`wave2_roster`, `situational`) | 66.5% | 49–51% | 0.609–0.610 |
| Best model without the line (`elo`) | 65.4% | 50.2% | 0.624 |

- **By season** the ensemble ranged from 62.3% (2021) to 70.7% (2013). It was 70.5% in 2024 and 67.5% over 2022–2025.
- **Confidence tracks accuracy:** picks at 60%+ win probability hit 71%, at 70%+ 80%, and at 80%+ 87%.
- **Best bets** are spread picks where the ensemble's edge over the line is at least 2 points. They have covered more
  than the 52.4% break-even in every era:

  | Seasons | Best-bet record | Cover rate |
  |---|---|---|
  | 2007–2011 | 134-119 | 53.0% |
  | 2012–2017 | 44-26 | 62.9% |
  | 2018–2025 (holdout) | 58-45 | 56.3% |

  That is only about 13 games a season, so treat best bets as a lean, not a lock.

### Why not 68% straight-up?

The target was 68% straight-up. Eight research agents built about 20 models and an ensemble to chase it, and **none
reached it honestly.** The closing betting line already contains almost everything public data can say:

- **What was tried.** Elo with QB adjustments, Kalman and ridge power ratings, about 50 opponent-adjusted
  play-by-play efficiency stats with gradient boosting, QB value models, snap-weighted injury reports, player-by-player
  lineup ratings, rest, travel, body clock, weather, referees, coaching, a neural net, a similar-games model and a
  close-game specialist. **Every model that sees the line lands within ±0.2 points of the betting favorite; models that don't are 1–3 points worse.**
- **Overruling the favorite loses.** When a model that doesn't see the line picks against the favorite, it is right
  only 42–48% of the time, even in pick'em games. The ensemble learned this on 2010–2017. It was frozen before 2018–2025
  was scored and confirmed on those seasons, so its winner pick stays with the favorite. The models set the probability
  and drive the spread picks.
- **The ceiling.** A model that knows exactly what the market knows should expect about **66.7%** over 2012–2025. A
  68% *season* happens by luck about a third of the time; the favorite alone did it in 2013, 2017 and 2024. A 68%
  *average* over 14 seasons has a 4% chance by luck. It would take a model that knows about 3 points per game the
  closing line doesn't price, which would also be worth about 57% against the spread. A backtest showing 68% over many
  seasons almost always means the model peeked at the future (leakage).

The research notes, with every model, test and dead end, are in `research/*/NOTES.md`. The ceiling analysis is in
`research/closegames/NOTES.md`.

## The models

| Family | Models (`research/preds/<id>.csv`) | Idea |
|---|---|---|
| Ratings | `elo`, `kalman`, `ratings_ridge`, `ratings_combo` (+ `_mkt` blends) | Team strength from results: 538-style Elo with QB/rest/travel adjustments, a week-to-week Kalman filter, opponent-adjusted margin and EPA ratings |
| Efficiency | `logit_epa`, `logit_epa_mkt`, `gbm`, `xgb_margin` | ~50 point-in-time, opponent-adjusted, recency-weighted play-by-play stats (EPA/play, success rate, pressure, turnover luck, special teams…) → logistic regression, LightGBM, XGBoost |
| Personnel | `personnel`, `personnel_nomkt`, `personnel_ats` | Starting-QB value (EPA + CPOE, shrunk, followed across teams), QB changes, snap-weighted injury load by position group from pre-game injury reports |
| Situational | `situational`, `mlp`, `knn` | Market structure (moneyline vs spread, implied totals), rest, travel, body clock, revenge, referee and more; a 5-seed neural net; a similar-games model |
| Roster | `wave2_roster`, `wave2_roster_nomkt` | Bottom-up: every expected starter rated from his own production, summed by unit, with injury cost and lineup changes |
| Close games | `wave2_market`, `wave2_close` | The sharpest market-only probability (spread + juice, recency-weighted) and a close-game specialist |
| **Meta** | **`ensemble`** | The models "talking to each other": an L2 logistic stack on the market plus each family's disagreement with it, fit walk-forward. A separate ATS stack uses each family's cover lean, the injury gap and underdog structure |

The meta-model picks its base models with a rule fixed in advance on development seasons only
(`research/meta/select_models.py`). The 18 first-round models qualify. The second-round models (`wave2_*`) are shown on
the site for comparison but didn't improve the development seasons, so they aren't in the stack.

## Run it locally

```bash
pip install -r backend/requirements.txt

python backend/pipeline.py              # download data (~3 min, ~350 MB in research/cache/), run all models, build picks.json
python backend/pipeline.py --quick      # refresh only the current season of the big datasets
python backend/pipeline.py --no-fetch   # rerun the models on the cached data
python research/eval.py research/preds/ensemble.csv research/preds/elo.csv   # score any model

python -m http.server 8000 -d web       # then open http://localhost:8000
```

The site has to be served over HTTP; opening `index.html` directly blocks the JSON fetch.

To add a model, write `research/<family>/run.py` that emits `research/preds/<id>.csv` with
`game_id, season, week, p_home, margin[, p_home_cover]` for every game since 2006, plus the upcoming slate. Follow the
rules in `research/PROTOCOL.md` (no leakage, walk-forward, score with `eval.py`). Add the family to
`backend/pipeline.py` and the model to `MODELS` in `backend/build_picks.py`.

## Hosting

1. **Settings → Pages → Build and deployment → Source**: choose **GitHub Actions**.
2. **Actions → Update NFL picks → Run workflow** on the default branch. The workflow also redeploys on any push that
   changes `web/`. Cron runs in UTC, so it schedules both 12:00 and 13:00 UTC and keeps whichever is 8am in New York.
3. Open `https://<user>.github.io/<repo>/`.

## Data

- **Source:** nflverse via `nflreadpy`. That covers schedules and closing lines (`spread_line`: positive means the
  home team is favored), moneylines, starting QBs, weather, play-by-play from 1999, player stats, pre-game injury
  reports, depth charts, snap counts and FTN charting.
- **Lines:** nflverse doesn't say which sportsbook its lines come from, so treat them as a market consensus. For
  upcoming games the line is whatever nflverse had at the last refresh, so check your own book's number.
- **Logos:** `web/logos/<TEAM>.png`. The Rams (`LA`) fall back to the ESPN logo.
- **Contract:** `research/SITE_SCHEMA.md` describes `picks.json`.

Probabilities are model estimates, not guarantees. For entertainment.

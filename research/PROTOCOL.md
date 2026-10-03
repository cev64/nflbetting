# Model research protocol (read fully before starting)

Goal: pick the winner of every NFL game (straight-up, "SU") and pick against the spread ("ATS").
Target: **68% SU accuracy, walk-forward out-of-sample, 2012-2025**. The market favorite gets 66.4%
over that window (`preds/market_baseline.csv`). ATS break-even at -110 is 52.4%.

## Data (already downloaded, read-only, shared — do NOT re-download)
`research/cache/*.parquet` (nflverse via nflreadpy):
- `schedules.parquet` 1999-2026: scores, `result` (home-away), `spread_line` (home favored positive, ~closing),
  `total_line`, moneylines (2006+ partially), rest days, div_game, roof, surface, temp, wind, starting QB ids/names,
  coaches, referee, stadium. 2026 weeks 1-3 are final; week 4 is in progress; weeks 5+ have no lines yet.
- `pbp.parquet` 1999-2026 play-by-play (372 cols: epa, wpa, success, cpoe, air_yards, xpass, etc). 300MB —
  load only the columns you need (`pl.scan_parquet(...).select([...])`).
- `team_stats_week`, `player_stats_week` (1999+), `injuries` (2009+), `rosters_weekly` (2002+),
  `depth_charts` (2001+), `snap_counts` (2012+), `officials`, `ftn` charting (2022+), `teams`.
- Python env has polars, pandas, numpy, scipy, scikit-learn, lightgbm, xgboost, statsmodels. 4 CPUs, 15 GB RAM
  shared by ~5 agents — keep jobs lean (n_jobs<=2).

## Hard rules (no exceptions)
1. **No leakage.** A prediction for a game may only use information available before kickoff:
   results/stats from strictly earlier games, the pre-game spread/total/moneyline, the scheduled starting QB,
   rest, weather, injury reports filed before the game. Never use the game's own stats, final score, or any
   season-aggregate that includes the game or later games. Same-week earlier games (e.g. Thursday before Sunday)
   may be used only if you are careful; simplest is to use only games from earlier weeks.
2. **Walk-forward training.** To predict season S, fit on seasons < S only (refitting each season is fine;
   refitting weekly within S using only earlier weeks is also fine). Hyperparameters you tune must be tuned on
   seasons before the test season (e.g., nested: tune on 2006-2011 or rolling), not picked by looking at 2012-2025 results.
   If you do sweep, say so honestly and report it — a sweep over many configs that is then scored on the same
   seasons is in-sample selection and inflates accuracy.
3. Score only with `python research/eval.py preds/<model>.csv` (same rules for everyone).
4. Anything that scores above ~67.5% SU or ~54% ATS must be stress-tested for leakage before you report it
   (shuffle check, drop-the-feature check, check the feature timestamps). Report honest numbers even if they miss the target.

## Output contract
Write `research/preds/<model_id>.csv` with columns:
`game_id, season, week, p_home, margin[, p_home_cover]`
- one row per game for **every game 2012-2026 that has a spread_line**, including unplayed 2026 games with a line
  (the current week's slate) — those are the live picks.
- `p_home` = probability home wins; `margin` = expected home minus away points; `p_home_cover` optional
  (probability home covers `spread_line`; ATS pick falls back to sign(margin - spread_line)).
Put your code in `research/<your_dir>/` with a single entry point `run.py` that rebuilds your preds CSV(s) end to
end from `research/cache` in a reasonable time (ideally < 5 minutes). Your code will later be moved into the
production backend, so keep it clean and deterministic (fixed seeds).

Also write `research/<your_dir>/NOTES.md`: what you tried, what worked/didn't, final numbers from eval.py
(all three windows + per-season), and the list of features with their importance — the website will show
"the data behind each pick", so for your final model(s) also emit
`research/preds/<model_id>_factors.parquet`: per game_id, the key input values for home and away
(columns like `game_id, feature, label, home, away`) for the most important ~8-15 features.

A stacking meta-model will combine everyone's predictions afterward, so diverse, well-calibrated
probabilities are valuable even if a single model is slightly weaker than the market alone.

# `web/data/picks.json` — contract between the model pipeline and the website

```jsonc
{
  "updated": "2026-10-03T12:00:00+00:00",
  "season": 2026,
  "week": 4,                       // the week shown by default (current/upcoming slate)
  "weeks": [1,2,3,4],              // weeks of the current season that have picks (past weeks graded)
  "models": [                      // every model, the ensemble first
    {"id": "ensemble", "name": "Ensemble (stacked)", "family": "meta",
     "description": "Logistic stack of all models below, fit walk-forward", "primary": true},
    {"id": "elo", "name": "Elo (QB-adjusted)", "family": "ratings", "description": "..."}
    // ... gbm, kalman, personnel, market, etc.
  ],
  "games": [
    {
      "game_id": "2026_04_SEA_ARI", "season": 2026, "week": 4, "type": "REG",
      "date": "2026-10-01", "time": "20:15", "home": "ARI", "away": "SEA",
      "stadium": "State Farm Stadium", "roof": "dome", "temp": null, "wind": null,
      "spread_line": -1.5,          // nflverse: positive = home favored
      "total_line": 44.5, "home_ml": 105, "away_ml": -125,
      "home_qb": "Kyler Murray", "away_qb": "Sam Darnold",
      "home_score": null, "away_score": null,   // filled once played

      "su":  {"pick": "SEA", "prob": 0.58, "margin": -2.1,      // ensemble: winner, P(pick wins), home-minus-away pts
              "market_prob": 0.55, "correct": null},           // correct: true/false/null (unplayed or tie)
      "ats": {"pick": "ARI", "line": 1.5,                      // pick's spread in betting notation (+1.5)
              "prob": 0.53, "edge": 0.6, "correct": null},     // edge = model margin vs spread, points; null result = unplayed/push
      "models": {"elo": {"p_home": 0.44, "margin": -1.8}, "gbm": {"p_home": 0.40, "margin": -2.9}},
      "agreement": 0.83,            // share of models agreeing with the SU pick
      "factors": [                  // data behind the pick, most important first
        {"key": "off_epa", "label": "Offense EPA/play (decayed)", "home": 0.042, "away": 0.081,
         "better": "high", "fmt": "0.000"}
        // ... 8-15 rows; better = "high" | "low" says which side is favorable
      ],
      "team_form": {"home": {"record": "1-2", "ats": "1-2", "pd": -14}, "away": {"record": "2-1", "ats": "2-1", "pd": 11}}
    }
  ],
  "teams": {"ARI": {"name": "Arizona Cardinals", "color": "#97233F", "color2": "#000000"}},

  "record": {                       // live record of the ensemble this season
    "su": {"w": 30, "l": 15}, "ats": {"w": 24, "l": 20, "p": 1},
    "by_week": [{"week": 1, "su_w": 11, "su_l": 5, "ats_w": 9, "ats_l": 7, "ats_p": 0}]
  },
  "backtest": {                     // walk-forward history, every model incl. "market" (always pick the favorite)
    "window": "2012-2025",
    "leaderboard": [
      {"id": "ensemble", "su_acc": 0.68, "ats_acc": 0.53, "logloss": 0.60, "brier": 0.208, "n": 3816, "ats_n": 3736}
    ],
    "by_season": [
      {"season": 2012, "market": 0.647, "ensemble": 0.66, "elo": 0.64}   // SU acc per model
    ],
    "by_season_ats": [{"season": 2012, "ensemble": 0.53}],
    "calibration": [{"bin": 0.55, "pred": 0.552, "actual": 0.56, "n": 410}],
    "by_confidence": [{"min_prob": 0.5, "acc": 0.68, "n": 3816}, {"min_prob": 0.6, "acc": 0.74, "n": 2500}],
    "ats_by_edge": [{"min_edge": 0, "acc": 0.53, "n": 3736}, {"min_edge": 3, "acc": 0.56, "n": 600}]
  }
}
```
Logos: `web/logos/<TEAM>.png` (LA = Rams missing → fall back to `teams[...].logo` or ESPN URL
`https://a.espncdn.com/i/teamlogos/nfl/500/lar.png`).

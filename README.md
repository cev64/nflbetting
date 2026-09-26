# NFL Turnover Tracker

Turnover metrics for NFL betting, built on free [nflverse](https://github.com/nflverse) data.

- **Backend** (`backend/`): a Python script that pulls play-by-play and schedules with
  [`nflreadpy`](https://github.com/nflverse/nflreadpy) and writes per-team game logs to `web/data/*.json`.
- **Website** (`web/`): a static site with no build step that reads those JSON files and shows the tables and comparisons.
- **Automation** (`.github/workflows/update-data.yml`): refreshes the current season every Tuesday and Friday,
  commits the new data, and (optionally) publishes the site to GitHub Pages.

## What it shows

**League table**: every team's takeaways (INT made + fumbles recovered), giveaways (INT thrown + fumbles lost),
and turnover differential, per game or as totals. The table is sortable and shaded against the league average.
It also has two fumble-luck columns:

- *Fum kept %*: the share of a team's own fumbles it recovered.
- *Opp fum rec %*: the share of opponent fumbles it recovered.

Fumble recoveries are close to a coin flip, so a team well above or below 50% is a regression candidate.

**Upcoming games**: the next week's slate with the spread, the total, and each team's turnover diff per game.
Click a game to open the matchup.

**Matchup**: two teams side by side with league ranks, plus:
- Team A's offense (giveaways) vs Team B's defense (takeaways), and the reverse.
- A simple projected turnover margin.
- A cumulative turnover-differential chart.
- Both teams' game logs.

**Filters**: season, regular season/playoffs, full season or last 3/5/8 games, and home/away splits.

### How turnovers are counted

From nflverse play-by-play:
- An interception is charged to the offense.
- A lost fumble is charged to the team that fumbled. That covers punt and kick returners, and a defender who
  fumbles an interception return back to the offense.

The 2025 totals match nflverse's official team stats (380 INTs, 248 of 249 lost fumbles).

## Run it locally

```bash
pip install -r backend/requirements.txt

python backend/fetch_data.py               # current season
python backend/fetch_data.py --since 2020  # backfill 2020 through the current season
python backend/fetch_data.py --seasons 2024 2025

python -m http.server 8000 -d web          # then open http://localhost:8000
```

The site has to be served over HTTP, because opening `index.html` directly blocks the JSON fetch.

## Hosting / auto-updates

1. Merge to the default branch. Scheduled workflows only run from the default branch.
2. Go to **Settings → Pages → Source** and choose **GitHub Actions**.
3. The workflow then updates the data twice a week and redeploys the site. You can also run it
   on demand from the **Actions** tab ("Update NFL data" → *Run workflow*).

## Data format

`web/data/<season>.json` has one record per team per game:

| field | meaning |
|---|---|
| `week`, `type`, `date`, `team`, `opp`, `home`, `pf`, `pa` | game info from the team's point of view (`type` is REG/WC/DIV/CON/SB) |
| `int_thrown`, `fum_lost`, `fumbles` | giveaways, plus total fumbles (lost or not) |
| `int_made`, `fum_rec`, `opp_fumbles` | takeaways, plus total opponent fumbles |

`upcoming` lists the next week's unplayed games with `spread_line` (positive = home favored) and `total_line`.

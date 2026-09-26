# NFL Turnover Tracker

Turnover metrics for NFL betting, built on free [nflverse](https://github.com/nflverse) data.

- **Backend** (`backend/`): a Python script that pulls play-by-play and schedules with
  [`nflreadpy`](https://github.com/nflverse/nflreadpy) and writes per-team game logs to `web/data/*.json`.
- **Website** (`web/`): a static site with no build step that reads those JSON files and shows the tables and comparisons.
- **Automation** (`.github/workflows/update-data.yml`): refreshes the current season every Tuesday morning,
  commits the new data, and (optionally) publishes the site to GitHub Pages.

## What it shows

**League table**: every team's takeaways (INT made + fumbles recovered), giveaways (INT thrown + fumbles lost),
and turnover differential, per game or as totals. The table is sortable and shaded against the league average.
It also has two fumble-luck columns:

- *Fum kept %*: the share of a team's own fumbles it recovered.
- *Opp fum rec %*: the share of opponent fumbles it recovered.

Fumble recoveries are close to a coin flip, so a team well above or below 50% is a regression candidate.

**vs Spread**: turnover margin against covering the spread (closing lines from nflverse).
- Cover rate when a team won, tied, or lost the turnover battle.
- Cover rate by the game's final turnover margin (−3 or worse through +3 or better).
- Cover rate by the *entering-the-game* turnover edge: the team's season-to-date TO diff/game minus its opponent's.
  This is the one you can actually bet on.
- Per-team ATS record, cover %, average cover margin, and ATS record split by won/even/lost turnover battle.
- Switch between the selected season and all loaded seasons pooled together.

ATS records and each game's line and cover margin also appear in the league table, the matchup view, and the game logs.

**Upcoming games**: the next week's slate with the spread, the total, and each team's turnover diff per game.
Click a game to open the matchup.

**Matchup**: two teams side by side with league ranks, plus:
- Team A's offense (giveaways) vs Team B's defense (takeaways), and the reverse.
- A simple projected turnover margin.
- A cumulative turnover-differential chart.
- Both teams' game logs.

**Filters**: season, regular season/playoffs, full season or last 3/5/8 games, and home/away splits.

### Weekly data cutoff

Data only goes through the last *fully completed* week. A Thursday night game isn't added until that whole week
is final on Monday night, so every team through week N has the same games (bye weeks aside). The site's header
shows "through week N", and the upcoming-games strip shows the next week's slate. Thursday games already played
show their final score there.

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

The site lives in `web/`, so GitHub Pages must deploy it with the workflow. "Deploy from a branch" won't work:
that only serves the repo root or `/docs`, and at the root you just get this README.

1. **Settings → Pages → Build and deployment → Source**: choose **GitHub Actions**.
2. **Actions → Update NFL data → Run workflow** (on the default branch) to fetch the data and deploy.
   It also redeploys on any push that changes `web/`, and runs on its own every Tuesday.
3. Open `https://<user>.github.io/<repo>/`.

Scheduled runs and Pages deployments only happen on the repo's default branch.

## Logos

`web/logos/<TEAM>.png` holds the primary logos from the Uniform Lab pack, resized to 128px. The Rams (`LA`)
weren't in the pack, so the site falls back to nflverse's logo URL for any team without a local file. To add
one, drop in `web/logos/LA.png`.

## Data format

`web/data/<season>.json` has one record per team per game:

| field | meaning |
|---|---|
| `week`, `type`, `date`, `team`, `opp`, `home`, `pf`, `pa` | game info from the team's point of view (`type` is REG/WC/DIV/CON/SB) |
| `line` | the team's closing spread (negative = favored); it covered if `pf - pa + line > 0` |
| `int_thrown`, `fum_lost`, `fumbles` | giveaways, plus total fumbles (lost or not) |
| `int_made`, `fum_rec`, `opp_fumbles` | takeaways, plus total opponent fumbles |

`upcoming` lists the games in the week after the published data, with `spread_line` (positive = home favored),
`total_line`, and the score if it has already been played.

# NFL Turnover Tracker

Turnover, spread and field-goal prop metrics for NFL betting, built on free [nflverse](https://github.com/nflverse) data.

- **Backend** (`backend/`): a Python script that pulls play-by-play and schedules with
  [`nflreadpy`](https://github.com/nflverse/nflreadpy) and writes per-team game logs to `web/data/*.json`.
- **Website** (`web/`): a static site with no build step that reads those JSON files and shows the tables and comparisons.
- **Automation** (`.github/workflows/update-data.yml`): refreshes the current season every morning at 8am ET
  (for the latest lines), commits the new data, and (optionally) publishes the site to GitHub Pages.

## What it shows

**League table**: every team's takeaways (INT made + fumbles recovered), giveaways (INT thrown + fumbles lost),
and turnover differential, per game or as totals. The table is sortable and shaded against the league average.
It also has two fumble-luck columns:

- *Fum kept %*: the share of a team's own fumbles it recovered.
- *Opp fum rec %*: the share of opponent fumbles it recovered.

Fumble recoveries are close to a coin flip, so a team well above or below 50% is a regression candidate.

**vs Spread**: turnover margin against covering the spread (nflverse spread lines).
- Cover rate when a team won, tied, or lost the turnover battle.
- Cover rate by the game's final turnover margin (−3 or worse through +3 or better).
- Cover rate by the *entering-the-game* turnover edge: the team's season-to-date TO diff/game minus its opponent's.
  This is the one you can actually bet on.
- Per-team ATS record, cover %, average cover margin, and ATS record split by won/even/lost turnover battle.
- Switch between the selected season and all loaded seasons pooled together.

ATS records and each game's line and cover margin also appear in the league table, the matchup view, and the game logs.

**Spots**: automatic spread signals for the upcoming week's games (`backend/model.py`).
- *Power edge*: each team's point differential minus 4 points per turnover of margin, blended with last season,
  pulled toward average, plus 1.5 points for home field. It gives a model "fair line"; a gap of 3+ points from the
  market flags a lean.
- *Turnover fade*: when one team's turnover diff/game is 1.0+ better than its opponent's (both with 3+ games),
  it leans on the worse-turnover team, betting on regression.
- *Both agree*: both signals point at the same team.

Every signal is backtested on all seasons on disk (2021–2025) and tracked live in the current season. The site
shows those records beside the picks. **So far none of them beats the 52.4% needed at −110.** They're all near
50%, because the spread already prices in turnover luck. Treat them as a screen, not as bets.

**1H Under** (the default tab): one chart ranking the week's games by the chance they're **24 or fewer at
halftime**, for first-half under 24.5 bets (`backend/firsthalf.py`). Each row shows that chance, the fair (no-vig)
price, and dashed break-even lines for −150, −200 and −250; bet only when your book's under 24.5 price beats fair.
- A week strip above the chart looks back at every week since 2022: three dots per week show whether that
  week's top 3 went under (green) or over (red). Click a week to see its chart with each game's actual halftime
  points; pick a season from the menu. Ties in the top 3 go to the earlier kickoff.
- Halftime scores come from play-by-play (the running score at the end of the first half, so defensive and return
  scores, PATs and two-point tries are included). Final scores from the same field match nflverse's schedule.
- Projection = a + b × the full-game total; the chance of 24 or fewer comes from how far real halftime totals
  landed from past projections (not an assumed Poisson/normal shape). Each season is predicted by a fit on earlier
  seasons only.
- Each team's first-half points scored and allowed were tested as a second input and **didn't improve the forecast**:
  the market's total already reflects them. They're shown in each game's tooltip as context.
- 2022–2025: 63% of games were 24 or fewer at half (720 of 1,139); the week's top 3 by the model, 71% (179 of 252).
  The probabilities are well calibrated (the "Show how this was tested" panel).
- **Not tested: profit.** nflverse has no first-half odds, so this can't show whether betting the under made money
  at the prices actually offered. An odds feed (e.g. The Odds API's `totals_h1` / `alternate_totals_h1`) would allow that.

**Kicks**: field-goal props, built on drive and red-zone data (`backend/kicks.py`). The idea is that FG props are
priced mostly off the game total, so an offense that **moves the ball but stalls in the red zone**, facing a defense
that **bends but doesn't break**, should kick more field goals than its total suggests.
- Two scatter charts (offenses and defenses): trips inside the opponent's 40 per game against red-zone TD %, with the
  "drives but kicks" corner shaded. Team tables rank every offense by a *stall score* and every defense by a
  *bend score* (trips above average minus red-zone TD % above average, in standard deviations). Both follow the filters.
- A weekly board with one row per kicker: projected trips inside the 40 × projected field-goal tries per trip × the
  team's make rate, P(2+ FGs made) with fair no-vig odds for over 1.5, and projected kicker points with P(8+).
- A backtest on every season on disk, compared with how often kickers actually made 2+ field goals and with a
  "total-only" price that just scales the league FG rate by the team's implied points:

  | 2021–2025, regular season | team-games | made 2+ FGs |
  |---|---|---|
  | Every kicker | 2,718 | 51.1% |
  | Stall offense × bend defense | 115 | 50.4% |
  | Projection 2.0+ FGs made | 412 | 57.8% (total-only price: 50.6%) |

  The plain stall × bend matchup found nothing. The trips × tries-per-trip projection did better, but over 1.5 is
  often juiced (54.5% break-even at −120, 58.3% at −140), so it only pays where your book's price is short.
  Field goals are noisy: a one-FG gap in the raw projection showed up as only about 0.2 FGs in results, so the
  displayed probabilities shrink the projection toward league average by that fitted slope (`CAL_SLOPE`, fitted
  in-sample on 2021–2025). The flag uses the raw projection and isn't affected by that fit.

### Where the betting lines come from

Spreads and totals come from nflverse's schedule data (`nflreadpy.load_schedules()`, from the `games` dataset
maintained in [nflverse/nfldata](https://github.com/nflverse/nfldata)). Its docs don't say which sportsbook the
lines come from, so treat them as a market consensus. For upcoming games, the line is whatever nflverse had
at the last refresh (every morning at 8am ET). The site is only as fresh as nflverse's file, which is not a live odds
feed, so check your own book's current number against the model's fair line.

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
python backend/fetch_data.py --since 2021  # backfill 2021 through the current season
python backend/fetch_data.py --seasons 2024 2025
python backend/fetch_data.py --model-only  # rebuild model.json (spots + kicks) from the data on disk

python -m http.server 8000 -d web          # then open http://localhost:8000
```

The site has to be served over HTTP, because opening `index.html` directly blocks the JSON fetch.

## Hosting / auto-updates

The site lives in `web/`, so GitHub Pages must deploy it with the workflow. "Deploy from a branch" won't work:
that only serves the repo root or `/docs`, and at the root you just get this README.

1. **Settings → Pages → Build and deployment → Source**: choose **GitHub Actions**.
2. **Actions → Update NFL data → Run workflow** (on the default branch) to fetch the data and deploy.
   It also redeploys on any push that changes `web/`, and runs on its own every morning at 8am ET
   (plus Tuesday at 14:00 UTC, 10am EDT / 9am EST, in case Monday night's play-by-play was late). GitHub cron runs in UTC with no daylight
   saving, so the workflow schedules both 12:00 and 13:00 UTC and skips whichever one isn't 8am in New York.
   Scheduled runs can start several minutes late when GitHub is busy.
3. Open `https://<user>.github.io/<repo>/`.

Scheduled runs and Pages deployments only happen on the repo's default branch.

## Design

The site follows a "fluid glass" style: a bright white surface with navy ink, Inter for text and Barlow Condensed
for headings, and glass only on floating layers (the sticky header, the filter bar once it pins, and tooltips).
Each tab leads with one or two simple charts (team-logo scatters, a market-vs-model dumbbell for spreads, a dot plot
of projected field goals per kicker, mirrored bars for matchups); the full tables, game logs and method notes sit
behind "Show …" toggles. Tabs and toggles use a pill that glides to the selection. When a table re-sorts or the filters change, rows slide to
their new place, washing green if they moved up and red if they moved down. All motion is off under
`prefers-reduced-motion`.

## Logos

`web/logos/<TEAM>.png` holds the primary logos from the Uniform Lab pack, resized to 128px. The Rams (`LA`)
weren't in the pack, so the site falls back to nflverse's logo URL for any team without a local file. To add
one, drop in `web/logos/LA.png`.

## Data format

`web/data/<season>.json` has one record per team per game:

| field | meaning |
|---|---|
| `week`, `type`, `date`, `team`, `opp`, `home`, `pf`, `pa` | game info from the team's point of view (`type` is REG/WC/DIV/CON/SB) |
| `line` | the team's spread from nflverse (negative = favored); it covered if `pf - pa + line > 0` |
| `total` | the game's over/under from nflverse |
| `h1_pf`, `h1_pa` | first-half points for and against (null if the game has no play-by-play) |
| `int_thrown`, `fum_lost`, `fumbles` | giveaways, plus total fumbles (lost or not) |
| `int_made`, `fum_rec`, `opp_fumbles` | takeaways, plus total opponent fumbles |
| `drives`, `t40`, `rz`, `rz_td`, `td` | offense: drives, trips inside the 40 and the 20 (a snap from there), red-zone TDs, TD drives |
| `fga`, `fgm`, `fg50`, `xpa`, `xpm` | offense: field goals tried / made / made from 50+, extra points tried / made |
| `opp_` + any of the ten above | the same, for the opponent's offense (what this team's defense allowed) |

`model.json` holds the spread signals: backtest and live records per signal, graded flags for the
current season, and `spots` for the upcoming week. Its `first_half` key holds the under 24.5 backtest,
calibration, every predicted game by season (`history`) and the weekly `board`; its `kicks` key holds the kicking backtest, calibration
tiers, this season's graded flags, and the weekly `board`.

`upcoming` lists the games in the week after the published data, with `spread_line` (positive = home favored),
`total_line`, and the score if it has already been played.

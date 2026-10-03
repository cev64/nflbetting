import nflreadpy as nfl, polars as pl, sys
out = "cache"
s = nfl.load_schedules()
s.write_parquet(f"{out}/schedules.parquet"); print("sched", s.shape, s["season"].min(), s["season"].max())
seasons = list(range(1999, 2027))
for name, fn in [("pbp", lambda: nfl.load_pbp(seasons)),
                 ("team_stats_week", lambda: nfl.load_team_stats(seasons, summary_level="week")),
                 ("player_stats_week", lambda: nfl.load_player_stats(seasons, summary_level="week")),
                 ("injuries", lambda: nfl.load_injuries(list(range(2009, 2027)))),
                 ("rosters_weekly", lambda: nfl.load_rosters_weekly(list(range(2002, 2027)))),
                 ("depth_charts", lambda: nfl.load_depth_charts(list(range(2001, 2027)))),
                 ("teams", lambda: nfl.load_teams()),
                 ("officials", lambda: nfl.load_officials()),
                 ("snap_counts", lambda: nfl.load_snap_counts(list(range(2012, 2027)))),
                 ("nextgen_passing", lambda: nfl.load_nextgen_stats(stat_type="passing")),
                 ("ftn", lambda: nfl.load_ftn_charting(list(range(2022, 2027)))),
                 ]:
    try:
        d = fn(); d.write_parquet(f"{out}/{name}.parquet"); print(name, d.shape)
    except Exception as e:
        print("FAIL", name, repr(e)[:300])

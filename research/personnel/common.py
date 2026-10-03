"""Shared helpers for the personnel model family."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
CACHE = RESEARCH / "cache"
PREDS = RESEARCH / "preds"
WORK = HERE / "work"  # intermediate parquet outputs

FIRST_TEST_SEASON = 2006

# franchise relocations -> keep team codes as they appear in each table (all tables use the
# same nflverse codes per season: OAK/LV, SD/LAC, STL/LA), so no remapping is needed for joins
# within a season.  For cross-season continuity map to the current code.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: str | None) -> str | None:
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name_expr(col: str) -> pl.Expr:
    return pl.col(col).map_elements(norm_name, return_dtype=pl.Utf8)


def pos_group(pos: str | None) -> str | None:
    """Map a raw position string to QB, OL, SK (RB/WR/TE skill), DL, LB, DB, or ST/None."""
    if pos is None:
        return None
    p = pos.upper().split("/")[0].strip()
    if p == "QB":
        return "QB"
    if p in {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}:
        return "OL"
    if p in {"WR", "TE", "RB", "FB", "HB"}:
        return "SK"
    if p in {"DE", "DT", "NT", "DL", "EDGE"}:
        return "DL"
    if p in {"LB", "ILB", "OLB", "MLB"}:
        return "LB"
    if p in {"CB", "S", "FS", "SS", "DB", "SAF"}:
        return "DB"
    return None


GROUPS = ["QB", "OL", "SK", "DL", "LB", "DB"]


def schedules() -> pl.DataFrame:
    s = pl.read_parquet(CACHE / "schedules.parquet")
    return s.with_columns(pl.col("gameday").str.to_date().alias("date"))

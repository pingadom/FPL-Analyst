"""Recover club names for the two seasons whose archive has none.

The 2016/17 and 2017/18 snapshots carry no `teams.csv` and no `raw.json`, so
`build_season` falls back to `Team 1` ... `Team 20`. Two things break as a result,
and neither is obvious from the outside:

* every club's rating history restarts in 2018/19, because
  `add_causal_team_strength` scopes its `team_key` per season for placeholder
  names — there is nothing to carry a rating across the 2017 to 2018 boundary; and
* nothing can be joined to those seasons by club, which silently excludes the
  betting market from precisely the two seasons used to *select* weights.

FPL's per-season `team` id is not stable and not reliably alphabetical — Leeds and
Leicester swap in two seasons, and 2025/26 puts Burnley at id 3 — so ordering
cannot be trusted. `team_code` is stable: Arsenal is 3 and Chelsea is 8 in every
season. Codes seen in later named seasons therefore resolve directly.

Clubs that never appear in a named season (Hull, Middlesbrough, Stoke, Sunderland,
Swansea) have no anchor, so their codes are filled by alphabetical position among
whatever names remain. That is a heuristic, which is why `verify_mapping` exists:
each reconstructed club's season goal totals are checked against the same club's
totals in an independent source. A wrong assignment shows up immediately as a
goal-total mismatch, so the guess is never taken on trust.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

import historical_odds as odds


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "fpl-data"
PLACEHOLDER_SEASONS = ("2016-17", "2017-18")

# The reconstruction, frozen after verification. Deriving this at build time would
# put a network fetch inside `build_season`; a table that has been checked against
# all 760 matches is both offline and reviewable. Re-derive with
# `python analysis/team_identity.py`, which reprints it and re-runs the check.
PLACEHOLDER_TEAM_NAMES: dict[str, dict[int, str]] = {
    "2016-17": {
        1: "Arsenal",
        2: "Bournemouth",
        3: "Burnley",
        4: "Chelsea",
        5: "Crystal Palace",
        6: "Everton",
        7: "Hull",
        8: "Leicester",
        9: "Liverpool",
        10: "Man City",
        11: "Man Utd",
        12: "Middlesbrough",
        13: "Southampton",
        14: "Stoke",
        15: "Sunderland",
        16: "Swansea",
        17: "Spurs",
        18: "Watford",
        19: "West Brom",
        20: "West Ham",
    },
    "2017-18": {
        1: "Arsenal",
        2: "Bournemouth",
        3: "Brighton",
        4: "Burnley",
        5: "Chelsea",
        6: "Crystal Palace",
        7: "Everton",
        8: "Huddersfield",
        9: "Leicester",
        10: "Liverpool",
        11: "Man City",
        12: "Man Utd",
        13: "Newcastle",
        14: "Southampton",
        15: "Stoke",
        16: "Swansea",
        17: "Spurs",
        18: "Watford",
        19: "West Brom",
        20: "West Ham",
    },
}


def _players(season: str) -> pd.DataFrame:
    return pd.read_csv(
        CACHE / season / "players_raw.csv", encoding="latin-1", low_memory=False
    )


def code_to_name(named_seasons: list[str]) -> dict[int, str]:
    """Stable FPL club code to the FPL spelling, from seasons that have names."""
    mapping: dict[int, str] = {}
    for season in named_seasons:
        players = _players(season)
        if "team_code" not in players or "team" not in players:
            continue
        teams_path = CACHE / season / "teams.csv"
        if not teams_path.exists():
            continue
        teams = pd.read_csv(teams_path, encoding="latin-1", low_memory=False)
        names = dict(zip(teams["id"].astype(int), teams["name"].astype(str)))
        for team_id, team_code in set(
            zip(players["team"].astype(int), players["team_code"].astype(int))
        ):
            if team_id in names:
                mapping.setdefault(int(team_code), names[team_id])
    return mapping


def season_club_names(season: str) -> list[str]:
    """The twenty clubs that actually played, from the market fixture list."""
    market = odds.load_market_fixtures([season])
    if market.empty:
        return []
    return sorted({*market["home_key"], *market["away_key"]})


def build_mapping(season: str, anchors: dict[int, str]) -> dict[int, str]:
    """Return team id to club name for one placeholder season."""
    players = _players(season)
    pairs = sorted(
        {
            (int(team_id), int(team_code))
            for team_id, team_code in zip(
                players["team"].astype(int), players["team_code"].astype(int)
            )
        }
    )
    resolved: dict[int, str] = {}
    unresolved: list[int] = []
    for team_id, team_code in pairs:
        name = anchors.get(team_code)
        if name:
            resolved[team_id] = name
        else:
            unresolved.append(team_id)

    # Whatever is left is filled by alphabetical position among the club keys the
    # market lists but the anchors did not claim. Verified afterwards, not trusted.
    claimed = {odds.normalise_team(name) for name in resolved.values()}
    remaining = [key for key in season_club_names(season) if key not in claimed]
    for team_id, key in zip(sorted(unresolved), remaining):
        resolved[team_id] = key
    return resolved


def archive_fixtures(season: str) -> pd.DataFrame:
    """Rebuild match-level results from the per-player archive.

    Every appearance row carries its fixture id, whether the player was at home,
    and both teams' scores, so one fixture's identity and result can be recovered
    from any row belonging to it.
    """
    gw = pd.read_csv(
        CACHE / season / "merged_gw.csv", encoding="latin-1", low_memory=False
    )
    players = _players(season)
    team_of = dict(
        zip(players["id"].astype(int), players["team"].astype(int))
    )
    gw = gw[gw["fixture"].notna() & gw["opponent_team"].notna()].copy()
    gw["own_team"] = gw["element"].astype(int).map(team_of)
    gw = gw[gw["own_team"].notna()]
    was_home = gw["was_home"].astype(str).str.lower().isin(("true", "1"))
    gw["home_id"] = np.where(was_home, gw["own_team"], gw["opponent_team"])
    gw["away_id"] = np.where(was_home, gw["opponent_team"], gw["own_team"])
    # Decide each fixture by majority rather than by whichever row comes first.
    # `players_raw` is an end-of-season snapshot, so it lists a January signing
    # under the club they joined, not the one they played these matches for. One
    # such row taken as representative misattributes the whole fixture — in
    # 2017/18 that produced a Burnley-versus-Burnley match. Thirty-odd appearances
    # per fixture outvote the handful who moved.
    gw = gw[gw["team_h_score"].notna()]
    fixtures = (
        gw.groupby("fixture")[
            ["home_id", "away_id", "team_h_score", "team_a_score"]
        ]
        .agg(lambda column: column.mode().iloc[0])
        .reset_index()
    )
    return fixtures.astype(
        {
            "home_id": int,
            "away_id": int,
            "team_h_score": int,
            "team_a_score": int,
        }
    )


def source_fixtures(season: str) -> set[tuple[str, str, int, int]]:
    """The same matches from the independent source, keyed by club name."""
    payload = odds._download(season)
    results: set[tuple[str, str, int, int]] = set()
    for row in csv.DictReader(io.StringIO(payload)):
        if not row.get("HomeTeam") or not row.get("FTHG"):
            continue
        results.add(
            (
                odds.normalise_team(row["HomeTeam"]),
                odds.normalise_team(row["AwayTeam"]),
                int(row["FTHG"]),
                int(row["FTAG"]),
            )
        )
    return results


def verify_mapping(season: str, mapping: dict[int, str]) -> dict:
    """Check the mapping against an independent record, match by match.

    Season goal totals are far too coarse to verify this: twenty clubs land
    between roughly 26 and 84 goals, so totals collide constantly and a club sits
    as near another's total as its own. Comparing whole fixtures instead uses the
    opponent and the ground as well as the score, which is a fingerprint rather
    than a summary — a single swapped pair misplaces every one of that pair's
    seventy-odd matches, so the test cannot pass unless the mapping is right.
    """
    fixtures = archive_fixtures(season)
    expected = source_fixtures(season)
    agreed = 0
    misses: list[tuple] = []
    for row in fixtures.itertuples(index=False):
        key = (
            odds.normalise_team(mapping.get(row.home_id, "")),
            odds.normalise_team(mapping.get(row.away_id, "")),
            int(row.team_h_score),
            int(row.team_a_score),
        )
        if key in expected:
            agreed += 1
        else:
            misses.append(key)
    return {
        "season": season,
        "matches": len(fixtures),
        "agreed": agreed,
        "misses": misses[:10],
    }


if __name__ == "__main__":
    named = [
        season
        for season in odds.SEASON_TAGS
        if season not in PLACEHOLDER_SEASONS
    ]
    anchors = code_to_name(named)
    print(f"stable club codes resolved from named seasons: {len(anchors)}")
    for season in PLACEHOLDER_SEASONS:
        mapping = build_mapping(season, anchors)
        report = verify_mapping(season, mapping)
        print()
        print(
            f"=== {season}: {report['agreed']}/{report['matches']} matches verified ==="
        )
        for team_id, name in sorted(mapping.items()):
            print(f"  {team_id:>2}  {name}")
        for miss in report["misses"]:
            print(f"  unmatched: {miss}")

"""Understat xG and xA for the seasons whose FPL archive has none.

FPL only began publishing expected goals and assists partway through 2022/23.
For 2016/17 to 2021/22, six of the ten replayed seasons and both training
seasons, every goal and assist rate in the prepared frame ran on actual goals
alone. That is the noisiest signal in the game, and the live model never sees
it that way, because the live feed always carries xG.

Understat publishes per-match xG and xA back to 2014/15. The vaastav archive
this project already replays keeps every player's full Understat history in
its 2021/22 and 2022/23 folders, with a map from Understat IDs to FPL IDs. So
the backfill reaches every player still in the league in 2021 or later.
Players who retired before then are missing, and `coverage()` reports how much
that costs rather than hiding it.

Understat and Opta (FPL's provider) are different xG models. They agree
closely on scale but are not identical, which is why the feed is used only
where FPL has nothing, never mixed into a season that has official values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "external" / "vaastav_sparse" / "data"
ARCHIVE = ROOT / "work" / "fpl-data"
FOLDERS = ("2021-22", "2022-23")
# Seasons whose FPL rows carry no expected goals at all, plus the partial one.
BACKFILL_SEASONS = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23")


def understat_matches() -> pd.DataFrame:
    """Every Understat player-match in the two folders, deduplicated."""
    frames = []
    for folder in FOLDERS:
        for path in (SOURCE / folder / "understat").glob("*.csv"):
            understat_id = path.stem.rsplit("_", 1)[-1]
            if not understat_id.isdigit():
                continue
            frame = pd.read_csv(path)
            frame["understat_id"] = int(understat_id)
            frames.append(frame)
    matches = pd.concat(frames, ignore_index=True)
    matches["when"] = pd.to_datetime(matches["date"])
    return matches.drop_duplicates(["understat_id", "id"])


def understat_to_code() -> dict[int, int]:
    """Understat ID to the stable FPL player code, via each folder's ID map."""
    mapping: dict[int, int] = {}
    for folder in FOLDERS:
        ids = pd.read_csv(SOURCE / folder / "id_dict.csv", skipinitialspace=True)
        ids.columns = [column.strip() for column in ids.columns]
        players = pd.read_csv(ARCHIVE / folder / "players_raw.csv", encoding="latin-1", low_memory=False)
        code_by_element = dict(zip(players["id"], players["code"]))
        for row in ids.itertuples(index=False):
            code = code_by_element.get(int(row.FPL_ID))
            if code is not None:
                mapping.setdefault(int(row.Understat_ID), int(code))
    return mapping


def fixture_dates(season: str) -> pd.DataFrame:
    """(player code, match date) to Gameweek, from the archive's own kickoff times."""
    merged = pd.read_csv(ARCHIVE / season / "merged_gw.csv", encoding="latin-1", low_memory=False)
    players = pd.read_csv(ARCHIVE / season / "players_raw.csv", encoding="latin-1", low_memory=False)
    code_by_element = dict(zip(players["id"], players["code"]))
    merged["player_code"] = merged["element"].map(code_by_element)
    merged["when"] = pd.to_datetime(merged["kickoff_time"], utc=True).dt.tz_localize(None)
    return merged[["player_code", "GW", "when"]].dropna().drop_duplicates()


# Understat stamps kick-offs several hours ahead of UTC, so an 8pm UK kick-off
# lands on the next calendar date. Exact date equality silently drops every
# late game; nearest kick-off within a day cannot pick a wrong match, because
# no club plays twice inside 24 hours.
MATCH_TOLERANCE = pd.Timedelta("1D")


def nearest_join(left: pd.DataFrame, right: pd.DataFrame, by: str) -> pd.DataFrame:
    """Attach each `left` row to the `right` row for the same `by` nearest in time."""
    left = left.sort_values("when").copy()
    right = right.sort_values("when").copy()
    left[by] = left[by].astype("int64") if by == "player_code" else left[by]
    return pd.merge_asof(
        left, right, on="when", by=by, direction="nearest", tolerance=MATCH_TOLERANCE
    )


def backfill_table() -> pd.DataFrame:
    """Per (season, Gameweek, player code): Understat xG, xA and key passes."""
    matches = understat_matches()
    code_of = understat_to_code()
    matches["player_code"] = matches["understat_id"].map(code_of)
    matches = matches.dropna(subset=["player_code"])
    matches["player_code"] = matches["player_code"].astype(int)
    tables = []
    for season in BACKFILL_SEASONS:
        dates = fixture_dates(season)
        dates["player_code"] = dates["player_code"].astype(int)
        columns = ["player_code", "when", "xG", "xA", "key_passes", "npxG", "xGChain", "time"]
        joined = nearest_join(dates, matches[columns], "player_code").dropna(subset=["xG"])
        per_gw = joined.groupby(["player_code", "GW"], as_index=False)[
            ["xG", "xA", "key_passes", "npxG", "xGChain", "time"]
        ].sum()
        per_gw["season"] = season
        tables.append(per_gw)
    table = pd.concat(tables, ignore_index=True)
    return table.rename(
        columns={
            "xG": "understat_xg",
            "xA": "understat_xa",
            "key_passes": "understat_key_passes",
            "npxG": "understat_npxg",
            "xGChain": "understat_xgchain",
            "time": "understat_minutes",
        }
    )


LEAGUE_CACHE = ROOT / "work" / "external" / "understat_league"
LEAGUE_SOURCE = "https://understat.com/getLeagueData/EPL/{year}"


def _league_payload(season: str) -> dict:
    """One Understat league-season, cached. The endpoint serves gzip JSON."""
    import gzip
    import json
    import urllib.request

    year = season[:4]
    target = LEAGUE_CACHE / f"EPL_{year}.json"
    if not target.exists() or target.stat().st_size == 0:
        LEAGUE_CACHE.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            LEAGUE_SOURCE.format(year=year),
            headers={
                "User-Agent": "Mozilla/5.0 (FPL-Lens research)",
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Encoding": "gzip",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        target.write_bytes(body)
    return json.loads(target.read_text(encoding="utf-8"))


def team_match_xg(season: str) -> pd.DataFrame:
    """Every club-match of a season: club key, date, xG for and against.

    Unlike the player files this is complete: 760 club-matches in every season,
    so it can replace a team total that partial player coverage never could.
    """
    from european_fixtures import club_key

    rows = []
    for team in _league_payload(season)["teams"].values():
        key = club_key(team["title"])
        key = TEAM_ALIASES.get(key, key)
        for match in team["history"]:
            rows.append(
                {
                    "club": key,
                    "when": pd.Timestamp(match["date"]),
                    "understat_team_xg": float(match["xG"]),
                    "understat_team_xga": float(match["xGA"]),
                }
            )
    return pd.DataFrame(rows)


# Club keys where Understat's full name and FPL's short name still differ after
# the shared normaliser.
TEAM_ALIASES = {"sheffieldunited": "sheffieldutd"}


def fill_team_xg(team_fixtures: pd.DataFrame, season: str, team_names: dict) -> pd.DataFrame:
    """Replace absent FPL team xG (a zero) with Understat's, club-match by club-match.

    FPL's figure is the sum of its players' xG and a real club-match essentially
    never sums to exactly zero, so zero means the feed was absent. A season
    whose fixtures do not all match is refused rather than half-filled: a team
    rating built from xG for some clubs and goals for others would be worse than
    either alone.
    """
    from european_fixtures import club_key

    if season not in BACKFILL_SEASONS:
        return team_fixtures
    absent = (team_fixtures["team_xg"] <= 0) | (team_fixtures["team_xga"] <= 0)
    if not absent.any():
        return team_fixtures
    source = team_match_xg(season)
    frame = team_fixtures.copy()
    frame["club"] = frame["team_id"].map(team_names).map(
        lambda name: TEAM_ALIASES.get(club_key(name), club_key(name))
    )
    frame["when"] = pd.to_datetime(frame["kickoff_time"], utc=True).dt.tz_localize(None)
    frame["_row"] = range(len(frame))
    merged = nearest_join(frame, source, "club").sort_values("_row").reset_index(drop=True)
    unmatched = merged["understat_team_xg"].isna()
    # A postponed fixture leaves an empty row at its original date (Arsenal v
    # Man City, 11 March 2020: 59 player rows, 0 minutes) and a real one later
    # under the same fixture id. That empty slot has nothing to fill; any other
    # miss still refuses the season.
    matched_fixtures = set(
        zip(merged.loc[~unmatched, "club"], merged.loc[~unmatched, "fixture"])
    )
    postponed_slot = pd.Series(
        [
            (club, fixture) in matched_fixtures
            for club, fixture in zip(merged["club"], merged["fixture"])
        ]
    ) & unmatched
    missing = (unmatched & ~postponed_slot)[absent.to_numpy()]
    if missing.any():
        sample = merged.loc[absent.to_numpy()][missing.to_numpy()][["club", "when"]].head(5)
        raise RuntimeError(
            f"{int(missing.sum())} of {int(absent.sum())} {season} club-matches have no "
            f"Understat match; refusing a half-filled season. First misses:\n{sample}"
        )
    fill = absent.to_numpy() & ~unmatched.to_numpy()
    team_fixtures = team_fixtures.astype({"team_xg": float, "team_xga": float})
    team_fixtures.loc[fill, "team_xg"] = merged.loc[fill, "understat_team_xg"].to_numpy()
    team_fixtures.loc[fill, "team_xga"] = merged.loc[fill, "understat_team_xga"].to_numpy()
    return team_fixtures


def attach_player_backfill(data: pd.DataFrame) -> pd.DataFrame:
    """Rate-formula xG and xA: FPL's where it published any, Understat's elsewhere.

    Written to separate columns (`rate_xg`, `rate_xa`) that only the per-player
    rate formulas read. `expected_goals` itself is left untouched so nothing
    that sums players into a club total can pick up partial coverage.
    """
    table = backfill_table()
    frame = data.copy()
    frame["_code"] = pd.to_numeric(frame["player_code"], errors="coerce")
    frame["_gw"] = frame["GW"].astype(int)
    merged = frame.merge(
        table[["season", "GW", "player_code", "understat_xg", "understat_xa"]].rename(
            columns={"GW": "_gw", "player_code": "_code"}
        ),
        on=["season", "_gw", "_code"],
        how="left",
    )
    feed_present = (
        merged.groupby(["season", "_gw"])["expected_goals"].transform("sum") > 0
    )
    merged["rate_xg"] = merged["expected_goals"].where(
        feed_present, merged["understat_xg"].fillna(0.0)
    )
    merged["rate_xa"] = merged["expected_assists"].where(
        feed_present, merged["understat_xa"].fillna(0.0)
    )
    return merged.drop(columns=["_code", "_gw", "understat_xg", "understat_xa"])


def coverage(data: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Share of appearances in each season that the backfill reaches."""
    appeared = data[data["minutes"] > 0][["season", "GW", "player_code", "minutes"]].copy()
    appeared["player_code"] = pd.to_numeric(appeared["player_code"], errors="coerce")
    merged = appeared.merge(
        table[["season", "GW", "player_code"]].assign(found=1),
        on=["season", "GW", "player_code"],
        how="left",
    )
    merged["found"] = merged["found"].fillna(0)
    merged["minutes_found"] = merged["minutes"] * merged["found"]
    grouped = merged.groupby("season")
    return pd.DataFrame(
        {
            "appearances": grouped["found"].size(),
            "share": grouped["found"].mean().round(3),
            "minutes_share": (grouped["minutes_found"].sum() / grouped["minutes"].sum()).round(3),
        }
    )


if __name__ == "__main__":
    import calibrate_model as lens

    table = backfill_table()
    print(f"{len(table):,} player-Gameweeks with Understat values")
    data, _ = lens.load_or_build_prepared_history()
    print(coverage(data, table).to_string())

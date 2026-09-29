"""Deadline-safe anytime-scorer prices from Betfair's historical exchange data.

The forecast's goal route is built from each player's own history: goals, xG
where FPL published it, threat and bonus. A betting market prices the same
event with information the archive never recorded — who is on penalties this
month, whether a striker is back in training, a manager's hints at a press
conference. `historical_odds.py` already brings the match market into the team
ratings; this module is the player-level equivalent.

Source
------
Betfair's Historic Data site sells the Exchange stream for every market since
2016. Its BASIC tier is free but still has to be "purchased" (at zero cost) and
downloaded from a Betfair account, so the files cannot be fetched here: download
Soccer, market type TO_SCORE, English Premier League, and put the files (the
.tar archives, or the .bz2 files inside them) anywhere under `work/betfair/`.

BASIC files hold one JSON message per line. Market definitions carry the event
name ("Arsenal v Leicester"), the kickoff (`marketTime`), the runners' names and
whether the market has gone in-play; runner changes carry the last traded price
(`ltp`) with a publish time (`pt`, milliseconds since the epoch), at one-minute
resolution.

Timestamp discipline
--------------------
As with the match odds, the whole value of this depends on using only what a
manager could have seen. The price used for a player-fixture is the last trade
**before that Gameweek's deadline**, never the kickoff price: late team news
moves scorer markets far more than it moves match odds, and a price taken after
the deadline would carry it. Updates after a market goes in-play are discarded
outright. A deadline here is the Gameweek's first kickoff minus 90 minutes,
which is the earliest FPL has used, so the cut is never late.

A trade older than `MAX_PRICE_AGE` at the deadline is treated as no price: thin
player markets can go days without a match, and a week-old price predates the
news this exists to capture.
"""

from __future__ import annotations

import bz2
import io
import json
import tarfile
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from historical_odds import normalise_team

ROOT = Path(__file__).resolve().parents[1]
BETFAIR_ROOT = ROOT / "work" / "betfair"
ARCHIVE = ROOT / "work" / "fpl-data"
UPDATES_CACHE = BETFAIR_ROOT / "to-score-updates.pkl"

MARKET_TYPES = ("TO_SCORE",)
DEADLINE_LEAD = timedelta(minutes=90)
MAX_PRICE_AGE = timedelta(days=4)

# Exchange spellings that do not normalise onto FPL's, mapped to the key
# `historical_odds.normalise_team` gives the FPL name.
BETFAIR_TEAM_ALIASES = {
    "manchestercity": "mancity",
    "manchesterunited": "manutd",
    "tottenhamhotspur": "spurs",
    "nottinghamforest": "nottmforest",
    "sheffieldunited": "sheffutd",
    "westbromwichalbion": "westbrom",
    "wolverhamptonwanderers": "wolves",
    "brightonhovealbion": "brighton",
    "newcastleunited": "newcastle",
    "westhamunited": "westham",
    "leicestercity": "leicester",
    "leedsunited": "leeds",
    "cardiffcity": "cardiff",
    "swanseacity": "swansea",
    "stokecity": "stoke",
    "hullcity": "hull",
    "norwichcity": "norwich",
    "huddersfieldtown": "huddersfield",
    "ipswichtown": "ipswich",
    "lutontown": "luton",
    "afcbournemouth": "bournemouth",
}


def team_key(name: object) -> str:
    key = normalise_team(name)
    return normalise_team(BETFAIR_TEAM_ALIASES.get(key, key))


def person_key(name: object) -> str:
    """Lower-case ASCII letters only: 'Martin Ødegaard' -> 'martinodegaard'."""
    text = unicodedata.normalize("NFKD", str(name).replace("_", " "))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("æ", "ae")
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


# ---------------------------------------------------------------------------
# Reading the stream files
# ---------------------------------------------------------------------------


def _open_text(name: str, payload: bytes) -> Iterator[str]:
    raw = bz2.decompress(payload) if name.endswith(".bz2") else payload
    yield from io.StringIO(raw.decode("utf-8", errors="replace"))


def iter_stream_files(root: Path = BETFAIR_ROOT) -> Iterator[tuple[str, Iterator[str]]]:
    """Every market file under `root`, whether loose or inside .tar archives."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".pkl":
            continue
        if path.suffix == ".tar":
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.isfile():
                        handle = archive.extractfile(member)
                        if handle is not None:
                            yield member.name, _open_text(member.name, handle.read())
        else:
            yield path.name, _open_text(path.name, path.read_bytes())


def parse_market(lines: Iterable[str], market_types: tuple[str, ...] = MARKET_TYPES) -> list[dict]:
    """Pre-off last-traded prices from one market file, one row per change."""
    definition: dict = {}
    names: dict[int, str] = {}
    rows: list[dict] = []
    in_play = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        published = message.get("pt")
        for change in message.get("mc", []):
            market_definition = change.get("marketDefinition")
            if market_definition:
                definition = market_definition
                for runner in market_definition.get("runners", []):
                    if "name" in runner:
                        names[int(runner["id"])] = str(runner["name"])
                if market_definition.get("inPlay") or market_definition.get("status") == "CLOSED":
                    in_play = True
            if in_play or definition.get("marketType") not in market_types:
                continue
            for runner_change in change.get("rc", []):
                if "ltp" not in runner_change or published is None:
                    continue
                runner_id = int(runner_change["id"])
                rows.append(
                    {
                        "market_id": str(change.get("id")),
                        "event_name": str(definition.get("eventName", "")),
                        "market_time": definition.get("marketTime"),
                        "runner_id": runner_id,
                        "runner_name": names.get(runner_id, ""),
                        "published": int(published),
                        "ltp": float(runner_change["ltp"]),
                    }
                )
    # Names can arrive in a later definition than the first price.
    for row in rows:
        if not row["runner_name"]:
            row["runner_name"] = names.get(row["runner_id"], "")
    return rows


def load_updates(root: Path = BETFAIR_ROOT, refresh: bool = False) -> pd.DataFrame:
    if UPDATES_CACHE.exists() and not refresh:
        return pd.read_pickle(UPDATES_CACHE)
    rows: list[dict] = []
    for _, lines in iter_stream_files(root):
        rows.extend(parse_market(lines))
    frame = pd.DataFrame.from_records(
        rows,
        columns=["market_id", "event_name", "market_time", "runner_id", "runner_name", "published", "ltp"],
    )
    if not frame.empty:
        frame["published"] = pd.to_datetime(frame["published"], unit="ms", utc=True)
        frame["market_time"] = pd.to_datetime(frame["market_time"], utc=True, errors="coerce")
        teams = frame["event_name"].str.split(r"\s+v\s+", n=1, expand=True, regex=True)
        frame["home_key"] = teams[0].map(team_key)
        frame["away_key"] = teams[1].map(team_key) if 1 in teams else ""
        frame["runner_key"] = frame["runner_name"].map(person_key)
        UPDATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(UPDATES_CACHE)
    return frame


# ---------------------------------------------------------------------------
# Joining onto player-fixtures
# ---------------------------------------------------------------------------


def _read_archive(path: Path) -> pd.DataFrame:
    """The archive is UTF-8 in later seasons and Latin-1 in earlier ones."""
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def _team_names(season: str) -> dict[int, str]:
    """Team id to club name; 2016/17 and 2017/18 come from `team_identity`."""
    path = ARCHIVE / season / "teams.csv"
    if path.exists():
        teams = _read_archive(path)
        return dict(zip(teams["id"].astype(int), teams["name"].astype(str)))
    import team_identity

    named = [
        folder.name for folder in sorted(ARCHIVE.iterdir())
        if (folder / "teams.csv").exists()
    ]
    return team_identity.build_mapping(season, team_identity.code_to_name(named))


def archive_player_fixtures(season: str) -> pd.DataFrame:
    """One row per player-fixture from the archive, with both clubs and the deadline."""
    folder = ARCHIVE / season
    merged = _read_archive(folder / "merged_gw.csv")
    team_names = _team_names(season)
    players = _read_archive(folder / "players_raw.csv")
    full = players.set_index("id")
    merged["kickoff"] = pd.to_datetime(merged["kickoff_time"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["kickoff"])
    merged["GW"] = merged["GW"].astype(int)
    # A player's own club is the side of the fixture that is not his opponent,
    # which survives mid-season transfers where players_raw's club does not.
    sides = merged.groupby("fixture")["opponent_team"].agg(lambda values: sorted(set(values.astype(int))))
    def own_team(row) -> int | None:
        pair = sides.get(row["fixture"], [])
        others = [team for team in pair if team != int(row["opponent_team"])]
        return others[0] if others else None
    merged["team_id"] = merged.apply(own_team, axis=1)
    merged = merged.dropna(subset=["team_id"])
    merged["team_id"] = merged["team_id"].astype(int)
    home = np.where(merged["was_home"].astype(bool), merged["team_id"], merged["opponent_team"].astype(int))
    away = np.where(merged["was_home"].astype(bool), merged["opponent_team"].astype(int), merged["team_id"])
    merged["home_key"] = [team_key(team_names.get(int(team), "")) for team in home]
    merged["away_key"] = [team_key(team_names.get(int(team), "")) for team in away]
    element = merged["element"].astype(int)
    first = element.map(full["first_name"]) if "first_name" in full else pd.Series("", index=merged.index)
    second = element.map(full["second_name"]) if "second_name" in full else pd.Series("", index=merged.index)
    web = element.map(full["web_name"]) if "web_name" in full else merged["name"]
    merged["full_key"] = (first.fillna("") + " " + second.fillna("")).map(person_key)
    merged["surname_key"] = second.fillna("").map(lambda name: person_key(str(name).split()[-1] if str(name).split() else ""))
    merged["first_key"] = first.fillna("").map(lambda name: person_key(str(name).split()[0] if str(name).split() else ""))
    merged["web_key"] = web.fillna("").map(person_key)
    first_kickoff = merged.groupby("GW")["kickoff"].transform("min")
    merged["deadline"] = first_kickoff - DEADLINE_LEAD
    merged["season"] = season
    return merged[
        ["season", "GW", "element", "fixture", "kickoff", "deadline", "home_key", "away_key",
         "full_key", "surname_key", "first_key", "web_key", "goals_scored", "minutes"]
    ]


def match_runner(runner_key: str, candidates: pd.DataFrame) -> int | None:
    """The element a runner name refers to among one fixture's players, or None.

    Tried in order, each only if it is unambiguous: full name, FPL web name,
    surname; then the runner's first name plus the surname or web name anywhere
    in it (FPL keeps middle and family names the exchange drops: 'Mohamed Salah
    Hamed Ghaly'); then a runner name ending in the surname or web name, which
    covers short first names ('Andy Robertson').
    """
    for column in ("full_key", "web_key", "surname_key"):
        hit = candidates.loc[candidates[column] == runner_key, "element"].unique()
        if len(hit) == 1:
            return int(hit[0])

    def contains_name(row) -> bool:
        return any(
            len(row[column]) >= 3 and row[column] in runner_key
            for column in ("surname_key", "web_key")
        )

    def ends_with_name(row) -> bool:
        return any(
            len(row[column]) >= 3 and runner_key.endswith(row[column])
            for column in ("surname_key", "web_key")
        )

    rules = (
        lambda row: bool(row["first_key"]) and runner_key.startswith(row["first_key"]) and contains_name(row),
        ends_with_name,
    )
    for rule in rules:
        hit = candidates[candidates.apply(rule, axis=1)]["element"].unique()
        if len(hit) == 1:
            return int(hit[0])
    return None


def deadline_prices(updates: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Last pre-deadline price per matched player-fixture."""
    if updates.empty or fixtures.empty:
        return pd.DataFrame(columns=["season", "GW", "element", "fixture", "market_score_probability", "market_price_age_hours"])
    records: list[dict] = []
    fixture_groups = {
        (row.home_key, row.away_key, row.kickoff.date()): group
        for row, group in (
            (group.iloc[0], group)
            for _, group in fixtures.groupby(["fixture"], sort=False)
        )
    }
    for (market_id, home, away), market in updates.groupby(["market_id", "home_key", "away_key"], sort=False):
        kickoff = market["market_time"].iloc[0]
        players = None
        for shift in (0, -1, 1):
            if pd.isna(kickoff):
                break
            players = fixture_groups.get((home, away, (kickoff + timedelta(days=shift)).date()))
            if players is not None:
                break
        if players is None:
            continue
        deadline = players["deadline"].iloc[0]
        usable = market[(market["published"] <= deadline) & (market["published"] >= deadline - MAX_PRICE_AGE)]
        if usable.empty:
            continue
        last = usable.sort_values("published").groupby("runner_id").tail(1)
        for row in last.itertuples():
            element = match_runner(row.runner_key, players)
            if element is None or row.ltp <= 1.0:
                continue
            records.append(
                {
                    "season": players["season"].iloc[0],
                    "GW": int(players["GW"].iloc[0]),
                    "element": element,
                    "fixture": int(players["fixture"].iloc[0]),
                    "market_score_probability": 1.0 / row.ltp,
                    "market_price_age_hours": (deadline - row.published).total_seconds() / 3600,
                }
            )
    return pd.DataFrame.from_records(records)


def build(seasons: list[str], refresh: bool = False) -> pd.DataFrame:
    updates = load_updates(refresh=refresh)
    frames = [deadline_prices(updates, archive_player_fixtures(season)) for season in seasons]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage_report(prices: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Share of each season's scoring appearances that carry a pre-deadline price."""
    rows = []
    for season in seasons:
        fixtures = archive_player_fixtures(season)
        played = fixtures[fixtures["minutes"] > 0]
        scorers = played[played["goals_scored"] > 0]
        priced = prices[prices["season"] == season] if not prices.empty else prices
        key = set(zip(priced.get("element", []), priced.get("fixture", [])))
        rows.append(
            {
                "season": season,
                "priced_rows": len(priced),
                "appearances_priced": float(np.mean([(e, f) in key for e, f in zip(played["element"], played["fixture"])])) if len(played) else 0.0,
                "scorers_priced": float(np.mean([(e, f) in key for e, f in zip(scorers["element"], scorers["fixture"])])) if len(scorers) else 0.0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    seasons = sys.argv[1:] or ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
                               "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    prices = build(seasons, refresh=True)
    print(coverage_report(prices, seasons).round(3).to_string(index=False))

"""Deadline-safe historical betting market for the Premier League.

The live model already blends a Matchbook exchange price into its team ratings,
but the backtest has never had a market view: exchange data does not exist far
enough back. That asymmetry meant any odds-derived feature could be shipped and
never validated.

football-data.co.uk closes that gap. It publishes free, unauthenticated CSVs of
closing prices for every Premier League match, and Pinnacle — the sharpest of the
listed books — covers all ten seasons the model replays, 380 matches each.

Timestamp discipline
--------------------
These are *closing* prices, taken at kick-off. A Gameweek deadline falls before
kick-off, so a closing line contains team news the manager did not have: late
injuries, rotation, and in the extreme a confirmed lineup. Using it raw would
leak, in the same way the archive's final blank/double assignments would.

Two things keep that honest. The market's weight is capped well below the live
path's, and it is applied to *team* scoring rates only, never to a player's own
availability, which is where post-deadline news does its real damage. A shifted
team total is a much weaker leak than a shifted lineup, and the cap is a tunable
so the sensitivity can be measured rather than assumed.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from live_external_signals import implied_goal_rates as external_implied_goal_rates


ROOT = Path(__file__).resolve().parents[1]
ODDS_CACHE = ROOT / "work" / "fpl-data" / "odds"
ODDS_BASE = "https://www.football-data.co.uk/mmz4281"

# Season label to the site's four-digit tag.
SEASON_TAGS = {
    "2016-17": "1617",
    "2017-18": "1718",
    "2018-19": "1819",
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

# Eighteen of twenty club names already agree once punctuation and case are
# stripped. These are the rest, mapped onto the FPL spelling.
TEAM_ALIASES = {
    "manunited": "manutd",
    "tottenham": "spurs",
    "nottmforest": "nottmforest",
    "sheffieldunited": "sheffutd",
    "sheffieldweds": "sheffieldwed",
    "westbrom": "westbrom",
    "wolverhampton": "wolves",
    "huddersfield": "huddersfield",
    "cardiff": "cardiff",
    "leeds": "leeds",
}

# Pinnacle first: it is the sharpest book on the sheet and the only one complete
# across every replayed season. The others stand in where it is missing.
PRICE_COLUMNS = (
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
)
OVER_COLUMNS = (("Avg>2.5",), ("B365>2.5",), ("Max>2.5",))


def normalise_team(name: object) -> str:
    key = "".join(
        character for character in str(name).lower() if character.isalnum()
    )
    return TEAM_ALIASES.get(key, key)


def _download(season: str) -> str:
    tag = SEASON_TAGS.get(season)
    if tag is None:
        return ""
    ODDS_CACHE.mkdir(parents=True, exist_ok=True)
    target = ODDS_CACHE / f"{tag}.csv"
    if target.exists() and target.stat().st_size > 0:
        return target.read_text(encoding="utf-8-sig", errors="replace")
    request = urllib.request.Request(
        f"{ODDS_BASE}/{tag}/E0.csv", headers={"User-Agent": "FPL-Lens/1.0"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read().decode("utf-8-sig", errors="replace")
    target.write_text(payload, encoding="utf-8")
    return payload


def _first_available(row: dict, groups: tuple) -> tuple | None:
    for names in groups:
        values = []
        for name in names:
            raw = row.get(name, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                values = []
                break
            if value <= 1.0:
                values = []
                break
            values.append(value)
        if len(values) == len(names):
            return tuple(values)
    return None


def load_market_fixtures(seasons: list[str]) -> pd.DataFrame:
    """One row per match, with de-vigged probabilities and implied team goals."""
    records: list[dict] = []
    for season in seasons:
        payload = _download(season)
        if not payload:
            continue
        for row in csv.DictReader(io.StringIO(payload)):
            if not row.get("HomeTeam") or not row.get("AwayTeam"):
                continue
            prices = _first_available(row, PRICE_COLUMNS)
            if prices is None:
                continue
            # Strip the bookmaker's margin: raw reciprocals sum above one.
            raw = np.array([1.0 / price for price in prices], dtype=float)
            probabilities = raw / raw.sum()
            over = _first_available(row, OVER_COLUMNS)
            over_probability = None
            if over is not None:
                # The over/under pair is priced separately, so de-vig it alone.
                under = _first_available(
                    row, (("Avg<2.5",), ("B365<2.5",), ("Max<2.5",))
                )
                if under is not None:
                    total = 1.0 / over[0] + 1.0 / under[0]
                    over_probability = (1.0 / over[0]) / total
            home_goals, away_goals = external_implied_goal_rates(
                float(probabilities[0]),
                float(probabilities[1]),
                float(probabilities[2]),
                over_probability,
            )
            records.append(
                {
                    "season": season,
                    "home_key": normalise_team(row["HomeTeam"]),
                    "away_key": normalise_team(row["AwayTeam"]),
                    "market_home_goals": float(home_goals),
                    "market_away_goals": float(away_goals),
                    "market_home_probability": float(probabilities[0]),
                    "market_draw_probability": float(probabilities[1]),
                    "market_away_probability": float(probabilities[2]),
                    "market_has_total": over_probability is not None,
                }
            )
    return pd.DataFrame.from_records(records)


def attach_market_rates(data: pd.DataFrame) -> pd.DataFrame:
    """Join the market onto each club-fixture as expected goals for and against.

    Matching is by season and the two club names, which identifies a fixture
    uniquely: a pair meets twice a season, once at each ground.
    """
    seasons = list(dict.fromkeys(data["season"].astype(str)))
    market = load_market_fixtures(seasons)
    if market.empty:
        data["market_expected_goals_for"] = np.nan
        data["market_expected_goals_against"] = np.nan
        return data

    team_key = data["team_name"].map(normalise_team)
    opponent_key = data["opponent_name"].map(normalise_team)
    was_home = data["was_home"].fillna(False).astype(bool)
    frame = pd.DataFrame(
        {
            "season": data["season"].astype(str),
            "home_key": np.where(was_home, team_key, opponent_key),
            "away_key": np.where(was_home, opponent_key, team_key),
            "was_home": was_home,
        }
    )
    merged = frame.merge(
        market, on=["season", "home_key", "away_key"], how="left"
    )
    data["market_expected_goals_for"] = np.where(
        merged["was_home"],
        merged["market_home_goals"],
        merged["market_away_goals"],
    )
    data["market_expected_goals_against"] = np.where(
        merged["was_home"],
        merged["market_away_goals"],
        merged["market_home_goals"],
    )
    return data


def coverage_report(data: pd.DataFrame) -> pd.DataFrame:
    """How much of each season the market actually reaches.

    A silent join failure would look exactly like a market with no opinion, so
    this is worth printing rather than trusting.
    """
    scored = data[data["fixture_count"] > 0]
    return (
        scored.assign(matched=scored["market_expected_goals_for"].notna())
        .groupby("season")["matched"]
        .agg(rows="size", matched="sum")
        .assign(share=lambda frame: (100 * frame["matched"] / frame["rows"]).round(1))
    )


if __name__ == "__main__":
    import calibrate_model as lens

    history, _ = lens.load_or_build_prepared_history()
    history = attach_market_rates(history)
    print(coverage_report(history).to_string())
    matched = history[history["market_expected_goals_for"].notna()]
    matched = matched[matched["fixture_count"] > 0]
    print()
    print(
        "market xG for: mean %.3f | model %.3f | realised team goals %.3f"
        % (
            matched["market_expected_goals_for"].mean(),
            matched["team_expected_goals_for"].mean(),
            matched["team_goals"].mean() / matched["team_games"].clip(lower=1).mean(),
        )
    )
    for column in ("market_expected_goals_for", "team_expected_goals_for"):
        print(
            "corr(%s, team goals scored) = %.4f"
            % (
                column,
                float(
                    np.corrcoef(
                        matched[column],
                        matched["team_goals"] / matched["team_games"].clip(lower=1),
                    )[0, 1]
                ),
            )
        )

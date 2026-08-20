"""Deadline-safe Opta priors and Matchbook exchange fixture signals.

The public Opta articles provide a slow season-strength prior. Matchbook's
public exchange endpoint provides a fast, timestamped fixture probability. The
two are deliberately kept separate: Opta informs team carry-over; Matchbook
informs the match. Neither source is allowed to enter historical rows after the
corresponding FPL deadline.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEAM_PRIORS_PATH = ROOT / "analysis" / "data" / "current-team-priors.json"
ELITE_CONSENSUS_PATH = ROOT / "analysis" / "data" / "elite-consensus-2026-27.json"
OPTA_FIXTURES_PATH = ROOT / "analysis" / "data" / "current-opta-fixtures.json"
MATCHBOOK_EVENTS = "https://api.matchbook.com/edge/rest/events"

TEAM_ALIASES = {
    "afc bournemouth": "bournemouth",
    "brighton": "brighton and hove albion",
    "brighton hove albion": "brighton and hove albion",
    "brighton & hove albion": "brighton and hove albion",
    "leeds": "leeds united",
    "man city": "manchester city",
    "man utd": "manchester united",
    "newcastle": "newcastle united",
    "nottm forest": "nottingham forest",
    "spurs": "tottenham hotspur",
}


def normalize_team(name: str) -> str:
    value = " ".join(
        str(name).lower().replace(".", "").replace("'", "").replace("-", " ").split()
    )
    return TEAM_ALIASES.get(value, value)


def load_team_priors() -> dict:
    return json.loads(TEAM_PRIORS_PATH.read_text(encoding="utf-8"))


def load_elite_consensus() -> dict:
    return json.loads(ELITE_CONSENSUS_PATH.read_text(encoding="utf-8"))


def load_opta_fixture_predictions() -> dict:
    payload = json.loads(OPTA_FIXTURES_PATH.read_text(encoding="utf-8"))
    payload["lookup"] = {
        (normalize_team(row["homeTeam"]), normalize_team(row["awayTeam"])): row
        for row in payload.get("fixtures", [])
    }
    return payload


def midpoint_probability(runner: dict) -> tuple[float | None, float, float | None]:
    """Return spread-aware midpoint probability, visible liquidity and spread."""
    prices = runner.get("prices", [])
    backs = [
        row for row in prices
        if row.get("side") == "back" and float(row.get("decimal-odds", 0)) > 1
    ]
    lays = [
        row for row in prices
        if row.get("side") == "lay" and float(row.get("decimal-odds", 0)) > 1
    ]
    best_back = max(backs, key=lambda row: float(row["decimal-odds"]), default=None)
    best_lay = min(lays, key=lambda row: float(row["decimal-odds"]), default=None)
    if best_back is None and best_lay is None:
        return None, 0.0, None
    back_odds = float(best_back["decimal-odds"]) if best_back else None
    lay_odds = float(best_lay["decimal-odds"]) if best_lay else None
    implied = [1 / value for value in (back_odds, lay_odds) if value]
    probability = float(np.mean(implied))
    liquidity = sum(
        float(row.get("available-amount", 0)) for row in (best_back, best_lay) if row
    )
    spread = (
        (lay_odds - back_odds) / math.sqrt(lay_odds * back_odds)
        if back_odds and lay_odds
        else None
    )
    return probability, liquidity, spread


def no_vig(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("Cannot normalise empty market probabilities")
    return [value / total for value in values]


def poisson_outcomes(home_goals: float, away_goals: float, maximum: int = 10) -> tuple[float, float, float, float]:
    home_mass = np.asarray(
        [math.exp(-home_goals) * home_goals**score / math.factorial(score) for score in range(maximum + 1)]
    )
    away_mass = np.asarray(
        [math.exp(-away_goals) * away_goals**score / math.factorial(score) for score in range(maximum + 1)]
    )
    matrix = np.outer(home_mass, away_mass)
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    total_mass = np.convolve(home_mass, away_mass)
    over25 = float(total_mass[3:].sum())
    normalised = no_vig([home, draw, away])
    return normalised[0], normalised[1], normalised[2], over25


def implied_goal_rates(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    over25_probability: float | None,
) -> tuple[float, float]:
    """Fit independent Poisson rates to exchange 1X2 and optional O/U 2.5."""
    target = np.asarray([home_probability, draw_probability, away_probability], float)
    best = (1.4, 1.2)
    best_loss = math.inf
    for home_goals in np.arange(0.25, 3.76, 0.04):
        for away_goals in np.arange(0.25, 3.76, 0.04):
            model = poisson_outcomes(float(home_goals), float(away_goals))
            loss = float(np.square(np.asarray(model[:3]) - target).sum())
            if over25_probability is not None:
                loss += 0.65 * (model[3] - over25_probability) ** 2
            if loss < best_loss:
                best_loss = loss
                best = (float(home_goals), float(away_goals))
    return best


def parse_matchbook_event(event: dict) -> dict | None:
    match_market = next(
        (market for market in event.get("markets", []) if market.get("name") == "Match Odds"),
        None,
    )
    if not match_market or len(match_market.get("runners", [])) != 3:
        return None
    runner_values: dict[str, tuple[float, float, float | None]] = {}
    for runner in match_market["runners"]:
        runner_values[normalize_team(runner.get("name", ""))] = midpoint_probability(runner)
    draw_key = normalize_team("Draw")
    participants = [key for key in runner_values if key != draw_key]
    event_sides = [normalize_team(value) for value in str(event.get("name", "")).split(" vs ")]
    if len(participants) != 2 or len(event_sides) != 2 or draw_key not in runner_values:
        return None
    home_key, away_key = event_sides
    if home_key not in runner_values or away_key not in runner_values:
        return None
    raw = [runner_values[key][0] for key in (home_key, draw_key, away_key)]
    if any(value is None for value in raw):
        return None
    home_probability, draw_probability, away_probability = no_vig(
        [float(value) for value in raw]
    )
    total_market = next(
        (
            market for market in event.get("markets", [])
            if market.get("name") == "Total" and float(market.get("handicap", -1)) == 2.5
        ),
        None,
    )
    over25_probability: float | None = None
    total_liquidity = 0.0
    if total_market:
        totals = {
            str(runner.get("name", "")).upper(): midpoint_probability(runner)
            for runner in total_market.get("runners", [])
        }
        over = next((value for key, value in totals.items() if key.startswith("OVER")), None)
        under = next((value for key, value in totals.items() if key.startswith("UNDER")), None)
        if over and under and over[0] is not None and under[0] is not None:
            over25_probability = no_vig([float(over[0]), float(under[0])])[0]
            total_liquidity = over[1] + under[1]
    home_goals, away_goals = implied_goal_rates(
        home_probability, draw_probability, away_probability, over25_probability
    )
    spreads = [runner_values[key][2] for key in (home_key, draw_key, away_key)]
    maximum_spread = max((value for value in spreads if value is not None), default=0.25)
    visible_liquidity = sum(runner_values[key][1] for key in (home_key, draw_key, away_key))
    market_volume = float(match_market.get("volume", event.get("volume", 0)) or 0)
    volume_quality = min(1.0, math.log1p(market_volume) / math.log1p(25_000))
    spread_quality = float(np.clip(1 - maximum_spread / 0.18, 0.15, 1.0))
    quality = float(np.clip(0.65 * volume_quality + 0.35 * spread_quality, 0.2, 1.0))
    return {
        "eventId": int(event["id"]),
        "fixture": str(event.get("name", "")),
        "kickoff": str(event.get("start", "")),
        "homeTeam": home_key,
        "awayTeam": away_key,
        "homeProbability": home_probability,
        "drawProbability": draw_probability,
        "awayProbability": away_probability,
        "over25Probability": over25_probability,
        "homeExpectedGoals": home_goals,
        "awayExpectedGoals": away_goals,
        "matchVolume": market_volume,
        "visibleLiquidity": visible_liquidity + total_liquidity,
        "maximumSpread": maximum_spread,
        "quality": quality,
    }


def fetch_matchbook_signals(
    expected_fixtures: list[tuple[str, str]],
    timeout: int = 30,
) -> dict:
    query = urllib.parse.urlencode(
        {"exchange-type": "back-lay", "sport-ids": 15, "offset": 0, "per-page": 500}
    )
    request = urllib.request.Request(
        f"{MATCHBOOK_EVENTS}?{query}", headers={"User-Agent": "FPL-Lens/8.0"}
    )
    captured = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as error:
        return {
            "status": "unavailable",
            "capturedAt": captured,
            "reason": f"{type(error).__name__}: {error}",
            "fixtures": [],
        }
    expected = {
        (normalize_team(home), normalize_team(away)) for home, away in expected_fixtures
    }
    fixtures = []
    for event in payload.get("events", []):
        event_sides = tuple(
            normalize_team(value) for value in str(event.get("name", "")).split(" vs ")
        )
        if event_sides not in expected:
            continue
        parsed = parse_matchbook_event(event)
        if parsed and (parsed["homeTeam"], parsed["awayTeam"]) in expected:
            fixtures.append(parsed)
    return {
        "status": "available" if fixtures else "unavailable",
        "capturedAt": captured,
        "source": MATCHBOOK_EVENTS,
        "expectedFixtureCount": len(expected),
        "fixtureCount": len(fixtures),
        "fixtures": fixtures,
    }


def fixture_lookup(payload: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["homeTeam"]), str(row["awayTeam"])): row
        for row in payload.get("fixtures", [])
    }

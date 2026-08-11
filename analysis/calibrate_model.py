"""Calibrate the FPL Lens ranking model on leak-free historical gameweeks.

The script downloads public FPL snapshots, constructs only pre-deadline features,
replays every gameweek from 2018-19 onward, evaluates hundreds of candidate
weight sets, and writes a compact JSON artifact consumed by the website.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "fpl-data"
OUTPUT = ROOT / "app" / "data" / "model-results.json"
TRAINING_SEASONS = ["2016-17", "2017-18"]
EVALUATION_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]
SEASONS = TRAINING_SEASONS + EVALUATION_SEASONS
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
REEP_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"
CURRENT_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
CURRENT_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
TRIALS = 2400
RECURSIVE_FINALISTS = 240
CHIP_POLICY_TRIALS = 144
POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
XI_QUOTAS = {1: 1, 2: 3, 3: 5, 4: 2}
SQUAD_QUOTAS = {1: 2, 2: 5, 3: 5, 4: 3}
AFCON_WINDOWS = {
    "2021-22": (20, 24),
    "2023-24": (20, 24),
    "2025-26": (16, 22),
}
AFCON_NATIONS = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini",
    "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
    "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
    "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
    "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal", "Sierra Leone",
    "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia",
    "Uganda", "Zambia", "Zimbabwe",
}


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        target.write_bytes(response.read())
    return target


def get_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def percentile(series: pd.Series) -> pd.Series:
    if series.notna().sum() < 2:
        return pd.Series(0.5, index=series.index)
    return series.rank(method="average", pct=True).fillna(0.5)


def parse_dob(value: object) -> date | None:
    if value is None or pd.isna(value) or str(value) in {"None", "", "nan"}:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def load_age_register() -> dict[int, str]:
    path = download(REEP_URL, CACHE / "reep-people.csv")
    people = pd.read_csv(
        path,
        usecols=["date_of_birth", "key_opta_numeric"],
        dtype=str,
        low_memory=False,
    ).dropna()
    people["key_opta_numeric"] = pd.to_numeric(
        people["key_opta_numeric"], errors="coerce"
    )
    people = people.dropna(subset=["key_opta_numeric"])
    return dict(
        zip(
            people["key_opta_numeric"].astype(int),
            people["date_of_birth"].astype(str),
        )
    )


def load_nationality_register() -> dict[int, str]:
    path = download(REEP_URL, CACHE / "reep-people.csv")
    people = pd.read_csv(
        path,
        usecols=["nationality", "key_opta_numeric"],
        dtype=str,
        low_memory=False,
    ).dropna()
    people["key_opta_numeric"] = pd.to_numeric(
        people["key_opta_numeric"], errors="coerce"
    )
    people = people.dropna(subset=["key_opta_numeric"])
    return dict(
        zip(
            people["key_opta_numeric"].astype(int),
            people["nationality"].astype(str),
        )
    )


def season_files(season: str) -> tuple[Path, Path, Path | None]:
    folder = CACHE / season
    gw = download(f"{BASE}/{season}/gws/merged_gw.csv", folder / "merged_gw.csv")
    players = download(f"{BASE}/{season}/players_raw.csv", folder / "players_raw.csv")
    try:
        teams = download(f"{BASE}/{season}/teams.csv", folder / "teams.csv")
    except HTTPError as error:
        if error.code != 404:
            raise
        try:
            teams = download(f"{BASE}/{season}/raw.json", folder / "raw.json")
        except HTTPError as raw_error:
            if raw_error.code != 404:
                raise
            teams = None
    return gw, players, teams


def build_season(
    season: str, ages: dict[int, str], nationalities: dict[int, str]
) -> tuple[pd.DataFrame, dict]:
    gw_path, players_path, teams_path = season_files(season)
    gw = pd.read_csv(gw_path, encoding="latin-1", low_memory=False)
    players = pd.read_csv(players_path, encoding="latin-1", low_memory=False)
    if teams_path is None:
        teams = pd.DataFrame(
            {
                "id": sorted(players["team"].dropna().astype(int).unique()),
                "name": [
                    f"Team {team_id}"
                    for team_id in sorted(players["team"].dropna().astype(int).unique())
                ],
            }
        )
    elif teams_path.suffix == ".json":
        teams = pd.DataFrame(json.loads(teams_path.read_text(encoding="utf-8"))["teams"])
    else:
        teams = pd.read_csv(teams_path, encoding="latin-1", low_memory=False)

    team_names = dict(zip(teams["id"].astype(int), teams["name"].astype(str)))
    meta_cols = ["id", "code", "element_type", "team", "web_name"]
    if "birth_date" in players.columns:
        meta_cols.append("birth_date")
    meta = players[meta_cols].copy().rename(
        columns={
            "id": "element",
            "code": "player_code",
            "element_type": "position_id",
            "team": "team_id",
            "web_name": "display_name",
        }
    )
    if "birth_date" not in meta.columns:
        meta["birth_date"] = None
    meta["birth_date"] = meta.apply(
        lambda row: row["birth_date"]
        if parse_dob(row["birth_date"])
        else ages.get(int(row["player_code"]))
        if pd.notna(row["player_code"])
        else None,
        axis=1,
    )
    meta["nationality"] = meta["player_code"].map(nationalities).fillna("")

    wanted = [
        "element",
        "GW",
        "total_points",
        "minutes",
        "value",
        "selected",
        "opponent_team",
        "was_home",
        "fixture",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "transfers_balance",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "yellow_cards",
        "red_cards",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "defensive_contribution",
    ]
    raw = gw[[column for column in wanted if column in gw.columns]].copy()
    raw = raw.merge(meta, on="element", how="left")
    raw["team_name"] = raw["team_id"].map(team_names)
    raw["opponent_name"] = raw["opponent_team"].map(team_names)
    if "selected" not in raw:
        raw["selected"] = 0
    raw["selected"] = pd.to_numeric(raw["selected"], errors="coerce").fillna(0)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").fillna(45)
    for column in [
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "transfers_balance",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "yellow_cards",
        "red_cards",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "defensive_contribution",
    ]:
        if column not in raw:
            raw[column] = 0
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)
    raw["GW"] = pd.to_numeric(raw["GW"], errors="coerce")
    raw = raw.dropna(subset=["GW", "element", "position_id"]).copy()
    raw["GW"] = raw["GW"].astype(int)

    weekly = (
        raw.groupby(["element", "GW"], as_index=False)
        .agg(
            points=("total_points", "sum"),
            minutes=("minutes", "sum"),
            price=("value", "mean"),
            selected=("selected", "max"),
            player_code=("player_code", "first"),
            position_id=("position_id", "first"),
            team_id=("team_id", "first"),
            team_name=("team_name", "first"),
            opponent_team=("opponent_team", "first"),
            opponent_name=("opponent_name", "first"),
            was_home=("was_home", "max"),
            fixture_count=("fixture", "nunique"),
            ict=("ict_index", "sum"),
            influence=("influence", "sum"),
            creativity=("creativity", "sum"),
            threat=("threat", "sum"),
            transfers_balance=("transfers_balance", "max"),
            display_name=("display_name", "first"),
            birth_date=("birth_date", "first"),
            nationality=("nationality", "first"),
            goals=("goals_scored", "sum"),
            assists=("assists", "sum"),
            clean_sheets=("clean_sheets", "sum"),
            goals_conceded=("goals_conceded", "sum"),
            saves=("saves", "sum"),
            bonus=("bonus", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            expected_goals=("expected_goals", "sum"),
            expected_assists=("expected_assists", "sum"),
            expected_goals_conceded=("expected_goals_conceded", "sum"),
            defensive_actions=("defensive_contribution", "sum"),
        )
        .sort_values(["element", "GW"])
    )
    weekly["season"] = season
    season_start = date(int(season[:4]), 8, 1)
    weekly["age"] = weekly["birth_date"].map(
        lambda value: (
            (season_start - parse_dob(value)).days / 365.2425
            if parse_dob(value)
            else np.nan
        )
    )

    by_player = weekly.groupby("element", sort=False)
    weekly["observations"] = by_player.cumcount()
    weekly["long_raw"] = by_player["points"].transform(
        lambda values: values.expanding().mean().shift(1)
    )
    weekly["recent_raw"] = by_player["points"].transform(
        lambda values: values.rolling(4, min_periods=2).mean().shift(1)
    )
    weekly["past_minutes"] = by_player["minutes"].transform(
        lambda values: values.expanding().mean().shift(1)
    )
    weekly["minutes_security_raw"] = by_player["minutes"].transform(
        lambda values: values.clip(upper=90).div(90).rolling(6, min_periods=2).mean().shift(1)
    )
    weekly["minutes_security_raw"] = weekly["minutes_security_raw"].fillna(
        (weekly["past_minutes"] / 90).clip(0, 1)
    )
    weekly["underlying_game"] = (
        weekly["ict"] / weekly["minutes"].clip(lower=45) * 90
    ).clip(0, 35)
    weekly["long_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.expanding().mean().shift(1)
    )
    weekly["recent_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.rolling(4, min_periods=2).mean().shift(1)
    )
    weekly["recent_underlying_raw"] = weekly["recent_underlying_raw"].fillna(
        weekly["long_underlying_raw"]
    )
    weekly["recent_raw"] = weekly["recent_raw"].fillna(weekly["long_raw"])
    weekly["long_value_raw"] = weekly["long_raw"] / (weekly["price"] / 10).clip(3.5)
    weekly["recent_value_raw"] = weekly["recent_raw"] / (weekly["price"] / 10).clip(3.5)
    weekly["age_raw"] = np.exp(-((weekly["age"].fillna(27.5) - 27.5) / 7.5) ** 2)
    transfer_momentum = np.sign(weekly["transfers_balance"]) * np.log1p(
        weekly["transfers_balance"].abs()
    )
    weekly["crowd_raw"] = (
        np.log1p(weekly["selected"].clip(lower=0)) + 0.12 * transfer_momentum
    )

    allowed = (
        weekly.groupby(["opponent_team", "position_id", "GW"], as_index=False)["points"]
        .mean()
        .sort_values(["opponent_team", "position_id", "GW"])
    )
    allowed["fixture_raw"] = allowed.groupby(
        ["opponent_team", "position_id"], sort=False
    )["points"].transform(lambda values: values.expanding().mean().shift(1))
    weekly = weekly.merge(
        allowed[["opponent_team", "position_id", "GW", "fixture_raw"]],
        on=["opponent_team", "position_id", "GW"],
        how="left",
    )
    weekly["fixture_raw"] = weekly["fixture_raw"].fillna(
        weekly.groupby(["GW", "position_id"])["fixture_raw"].transform("median")
    )
    weekly["fixture_raw"] = weekly["fixture_raw"].fillna(2.5) + weekly[
        "was_home"
    ].fillna(False).astype(float) * 0.18

    # The next six opponents are known at the deadline. Their strength is always
    # estimated at the current GW, never with results that happened later.
    schedule = raw[["team_id", "GW", "opponent_team", "was_home"]].drop_duplicates()
    schedule_map: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for row in schedule.itertuples(index=False):
        schedule_map.setdefault((int(row.team_id), int(row.GW)), []).append(
            (int(row.opponent_team), bool(row.was_home))
        )
    fixture_lookup = {
        (int(row.GW), int(row.position_id), int(row.opponent_team)): float(row.fixture_raw)
        for row in weekly[["GW", "position_id", "opponent_team", "fixture_raw"]]
        .dropna()
        .drop_duplicates(["GW", "position_id", "opponent_team"])
        .itertuples(index=False)
    }
    fixture_median = {
        (int(gw_number), int(position)): float(value)
        for (gw_number, position), value in weekly.groupby(["GW", "position_id"])[
            "fixture_raw"
        ].median().items()
    }
    horizon_weights = (1.0, 0.86, 0.74, 0.64, 0.55, 0.47)

    def fixture_horizon(row: pd.Series) -> float:
        values: list[tuple[float, float]] = []
        base_gw = int(row["GW"])
        position = int(row["position_id"])
        team = int(row["team_id"])
        fallback = fixture_median.get((base_gw, position), 2.5)
        for offset, horizon_weight in enumerate(horizon_weights):
            for opponent, home in schedule_map.get((team, base_gw + offset), []):
                strength = fixture_lookup.get(
                    (base_gw, position, opponent), fallback
                ) + (0.18 if home else 0.0)
                values.append((strength, horizon_weight))
        if not values:
            return fallback
        return sum(value * weight for value, weight in values) / sum(
            weight for _, weight in values
        )

    weekly["fixture_horizon_raw"] = weekly.apply(fixture_horizon, axis=1)
    weekly["horizon_weighted_games"] = weekly.apply(
        lambda row: sum(
            horizon_weight * len(
                schedule_map.get((int(row["team_id"]), int(row["GW"]) + offset), [])
            )
            for offset, horizon_weight in enumerate(horizon_weights)
        ),
        axis=1,
    ).clip(lower=1.0)

    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_raw", "fixture_now"),
        ("crowd_raw", "crowd"),
        ("minutes_security_raw", "minutes_security"),
        ("long_underlying_raw", "long_underlying"),
        ("recent_underlying_raw", "recent_underlying"),
    ]:
        weekly[rank_name] = weekly.groupby(["GW", "position_id"])[raw_name].transform(
            percentile
        )

    eligible = weekly[weekly["price"] >= 35].copy()
    age_coverage = float(weekly["age"].notna().mean())
    summary = {
        "season": season,
        "rows": int(len(weekly)),
        "eligibleRows": int(len(eligible)),
        "ageCoverage": round(age_coverage * 100, 1),
        "gameweeks": int(eligible["GW"].nunique()),
    }
    return eligible, summary


def prepare_causal_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Carry player priors across seasons and build component expected points."""
    data = pd.concat(frames, ignore_index=True)
    season_order = {season: index for index, season in enumerate(SEASONS)}
    data["season_order"] = data["season"].map(season_order).astype(int)
    fallback_key = data["season"].astype(str) + ":" + data["element"].astype(str)
    numeric_code = pd.to_numeric(data["player_code"], errors="coerce")
    data["player_key"] = numeric_code.astype("Int64").astype(str).where(
        numeric_code.notna(), fallback_key
    )
    data.sort_values(
        ["player_key", "season_order", "GW"], inplace=True, kind="stable"
    )
    by_player = data.groupby("player_key", sort=False)
    data["observations"] = by_player.cumcount()

    points_prior = data["position_id"].map({1: 3.2, 2: 2.6, 3: 2.8, 4: 2.6})
    minutes_prior = data["position_id"].map({1: 0.66, 2: 0.58, 3: 0.57, 4: 0.55})
    underlying_prior = data["position_id"].map({1: 2.5, 2: 4.0, 3: 6.0, 4: 6.5})

    data["long_raw"] = by_player["points"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(points_prior)
    data["recent_raw"] = by_player["points"].transform(
        lambda values: values.rolling(4, min_periods=1).mean().shift(1)
    ).fillna(data["long_raw"])
    data["past_minutes"] = by_player["minutes"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(minutes_prior * 90)
    per_fixture_minutes = (
        data["minutes"] / data["fixture_count"].clip(lower=1)
    ).clip(upper=90)
    data["per_fixture_minutes"] = per_fixture_minutes
    data["minutes_security_raw"] = data.groupby("player_key", sort=False)[
        "per_fixture_minutes"
    ].transform(
        lambda values: values.div(90).rolling(6, min_periods=1).mean().shift(1)
    ).fillna(minutes_prior).clip(0.05, 1.0)
    data["expected_minutes"] = (data["minutes_security_raw"] * 90).clip(5, 90)

    data["underlying_game"] = (
        data["ict"] / data["minutes"].clip(lower=45) * 90
    ).clip(0, 35)
    data["long_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(underlying_prior)
    data["recent_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.rolling(6, min_periods=1).mean().shift(1)
    ).fillna(data["long_underlying_raw"])

    minute_denominator = data["minutes"].clip(lower=45)
    data["goal_signal_game"] = np.where(
        data["expected_goals"] > 0,
        0.72 * data["expected_goals"] + 0.28 * data["goals"],
        data["goals"],
    ) / minute_denominator * 90
    data["assist_signal_game"] = np.where(
        data["expected_assists"] > 0,
        0.72 * data["expected_assists"] + 0.28 * data["assists"],
        data["assists"],
    ) / minute_denominator * 90
    data["clean_sheet_game"] = (
        data["clean_sheets"] / data["fixture_count"].clip(lower=1)
    ).clip(0, 1)
    for source, target, prior in [
        ("goal_signal_game", "goal_rate", {1: 0.01, 2: 0.04, 3: 0.20, 4: 0.28}),
        ("assist_signal_game", "assist_rate", {1: 0.01, 2: 0.08, 3: 0.18, 4: 0.13}),
        ("clean_sheet_game", "clean_sheet_rate", {1: 0.28, 2: 0.28, 3: 0.22, 4: 0.0}),
        ("saves", "save_rate", {1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}),
        ("bonus", "bonus_rate", {1: 0.18, 2: 0.22, 3: 0.28, 4: 0.28}),
        ("yellow_cards", "yellow_rate", {1: 0.05, 2: 0.12, 3: 0.10, 4: 0.08}),
        ("red_cards", "red_rate", {1: 0.005, 2: 0.008, 3: 0.006, 4: 0.005}),
        ("goals_conceded", "conceded_rate", {1: 1.35, 2: 1.35, 3: 0.0, 4: 0.0}),
        ("defensive_actions", "defensive_rate", {1: 0.0, 2: 4.5, 3: 4.0, 4: 2.2}),
    ]:
        rolling = data.groupby("player_key", sort=False)[source].transform(
            lambda values: values.rolling(12, min_periods=1).mean().shift(1)
        )
        data[target] = rolling.fillna(data["position_id"].map(prior)).clip(lower=0)

    fixture_multiplier = 0.72 + 0.56 * data["fixture_now"].fillna(0.5)
    minutes_factor = data["expected_minutes"] / 90
    p_play = (data["expected_minutes"] / 35).clip(0, 1)
    p_sixty = ((data["expected_minutes"] - 25) / 45).clip(0, 1)
    goal_points = data["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4})
    clean_sheet_points = data["position_id"].map({1: 4, 2: 4, 3: 1, 4: 0})
    appearance_points = p_play + p_sixty
    attacking_points = (
        data["goal_rate"] * goal_points + data["assist_rate"] * 3
    ) * minutes_factor * fixture_multiplier
    clean_sheet_points_ev = (
        data["clean_sheet_rate"]
        * clean_sheet_points
        * p_sixty
        * fixture_multiplier
    )
    save_points = (
        data["save_rate"] / 3 * minutes_factor
        * np.where(data["position_id"] == 1, 1.0, 0.0)
    )
    bonus_points = data["bonus_rate"] * minutes_factor * fixture_multiplier
    discipline_points = -(
        data["yellow_rate"] + 3 * data["red_rate"]
    ) * minutes_factor
    conceded_points = -(
        data["conceded_rate"] / 2 * minutes_factor
        * data["position_id"].isin([1, 2]).astype(float)
    )
    defensive_threshold = np.where(data["position_id"] == 2, 10.0, 12.0)
    defensive_points = (
        2 * (data["defensive_rate"] / defensive_threshold).clip(0, 1)
        * minutes_factor
        * data["position_id"].isin([2, 3, 4]).astype(float)
        * (data["season_order"] >= season_order.get("2025-26", 99)).astype(float)
    )
    data["component_xpts"] = (
        appearance_points
        + attacking_points
        + clean_sheet_points_ev
        + save_points
        + bonus_points
        + discipline_points
        + conceded_points
        + defensive_points
    ).clip(0.2, 13.0) * data["fixture_count"].clip(lower=1)
    horizon_multiplier = 0.74 + 0.52 * data["fixture"].fillna(0.5)
    single_fixture_base = data["component_xpts"] / data["fixture_count"].clip(lower=1)
    data["component_horizon"] = (
        single_fixture_base
        * data["horizon_weighted_games"].clip(lower=1)
        * horizon_multiplier
    ).clip(0.5, 50)
    data["prediction_uncertainty"] = (
        1.35
        + 2.0 * (1 - data["minutes_security_raw"])
        + 1.8 / np.sqrt(data["observations"] + 1)
    ).clip(1.2, 5.0)

    lagged_transfer_balance = by_player["transfers_balance"].shift(1).fillna(0)
    transfer_momentum = np.sign(lagged_transfer_balance) * np.log1p(
        lagged_transfer_balance.abs()
    )
    data["crowd_raw"] = (
        np.log1p(data["selected"].clip(lower=0)) + 0.08 * transfer_momentum
    )
    data["long_value_raw"] = data["long_raw"] / (data["price"] / 10).clip(3.5)
    data["recent_value_raw"] = data["component_xpts"] / (
        data["price"] / 10
    ).clip(3.5)
    data["age_raw"] = np.exp(-((data["age"].fillna(27.5) - 27.5) / 7.5) ** 2)

    rank_groups = ["season", "GW", "position_id"]
    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("component_xpts", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_raw", "fixture_now"),
        ("crowd_raw", "crowd"),
        ("minutes_security_raw", "minutes_security"),
        ("long_underlying_raw", "long_underlying"),
        ("recent_underlying_raw", "recent_underlying"),
    ]:
        data[rank_name] = data.groupby(rank_groups)[raw_name].transform(percentile)

    data.sort_values(
        ["season_order", "GW", "position_id", "element"], inplace=True, kind="stable"
    )
    data.reset_index(drop=True, inplace=True)
    return data


@dataclass(frozen=True)
class Candidate:
    performance: float
    value: float
    age: float
    fixture: float
    crowd: float
    minutes: float
    underlying: float
    recent_share: float

    @property
    def coefficients(self) -> np.ndarray:
        return np.array(
            [
                self.performance * self.recent_share,
                self.performance * (1 - self.recent_share),
                self.value * self.recent_share,
                self.value * (1 - self.recent_share),
                self.age,
                self.fixture,
                self.crowd,
                self.minutes,
                self.underlying * self.recent_share,
                self.underlying * (1 - self.recent_share),
            ],
            dtype=float,
        )

    def as_dict(self) -> dict:
        raw = {
            "performance": self.performance,
            "value": self.value,
            "age": self.age,
            "fixture": self.fixture,
            "crowd": self.crowd,
            "minutes": self.minutes,
            "underlying": self.underlying,
        }
        rounded = {key: round(value * 100) for key, value in raw.items()}
        rounded[max(raw, key=raw.get)] += 100 - sum(rounded.values())
        rounded["recent"] = round(self.recent_share * 100)
        rounded["history"] = 100 - rounded["recent"]
        return rounded


def candidate_pool() -> tuple[list[Candidate], int]:
    rng = np.random.default_rng(20260811)
    raw_weights = rng.dirichlet(
        [4.2, 1.4, 0.25, 2.1, 0.40, 2.8, 2.2], size=TRIALS - 5
    )
    recent = rng.beta(5.0, 1.8, size=TRIALS - 5) * 0.55 + 0.40
    candidates = [
        Candidate(*weights, float(recency))
        for weights, recency in zip(raw_weights, recent)
    ]
    candidates.extend(
        [
            # Official-winner principles: form + medium-term fixtures, reliable
            # minutes, underlying data, restrained ownership and almost no age prior.
            Candidate(0.34, 0.08, 0.00, 0.18, 0.04, 0.21, 0.15, 0.78),
            Candidate(0.42, 0.07, 0.00, 0.20, 0.02, 0.19, 0.10, 0.82),
            Candidate(0.31, 0.09, 0.00, 0.17, 0.03, 0.23, 0.17, 0.72),
            Candidate(0.46, 0.06, 0.00, 0.14, 0.02, 0.20, 0.12, 0.76),
            # Lens 1.0: retained as a proper recursive baseline.
            Candidate(0.36, 0.09, 0.01, 0.04, 0.50, 0.00, 0.00, 0.59),
        ]
    )
    return candidates, len(candidates) - 1


def feature_matrix(data: pd.DataFrame) -> np.ndarray:
    return data[
        [
            "recent",
            "long",
            "recent_value",
            "long_value",
            "age_score",
            "fixture",
            "crowd",
            "minutes_security",
            "recent_underlying",
            "long_underlying",
        ]
    ].to_numpy(dtype=float)


def candidate_forecasts(
    data: pd.DataFrame, candidate: Candidate
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_score = feature_matrix(data) @ candidate.coefficients
    calibration = 0.72 + 0.56 * model_score
    current = data["component_xpts"].to_numpy(float) * calibration
    horizon = data["component_horizon"].to_numpy(float) * calibration
    return current, horizon, model_score


def snapshot_replay(
    data: pd.DataFrame, candidates: list[Candidate]
) -> tuple[np.ndarray, list[str]]:
    """Fast predictive screen before the fully stateful season replay."""
    features = feature_matrix(data)
    actual = data["points"].to_numpy(dtype=float)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    coefficients = np.vstack([candidate.coefficients for candidate in candidates])
    results = np.zeros((len(candidates), len(seasons)), dtype=float)
    for season_id, season in enumerate(seasons):
        mask = data["season"].to_numpy() == season
        season_features = features[mask]
        season_actual = actual[mask]
        centered_features = season_features - season_features.mean(axis=0)
        centered_actual = season_actual - season_actual.mean()
        covariance = centered_features.T @ centered_actual
        feature_covariance = centered_features.T @ centered_features
        numerator = coefficients @ covariance
        denominator = np.sqrt(
            np.einsum(
                "ij,jk,ik->i", coefficients, feature_covariance, coefficients
            )
            * float(centered_actual @ centered_actual)
        )
        results[:, season_id] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    return results, seasons


@dataclass(frozen=True)
class SimulationStrategy:
    name: str
    transfer_hurdle: float
    bank_limit: int
    force_weekly_review: bool
    safe_captain: bool
    max_hits: int = 3
    hit_immediate_hurdle: float = 2.5


EXPERT_STRATEGY = SimulationStrategy(
    "Patient six-GW transfers + safe captain", 8.00, 5, False, True, 0, 99.0
)
WEEKLY_CHASE_STRATEGY = SimulationStrategy(
    "Six-GW planner + adaptive banking", 5.00, 5, False, False, 0, 99.0
)


@dataclass(frozen=True)
class ChipPolicy:
    wildcard_gap: float
    free_hit_gap: float
    bench_score: float
    triple_score: float
    afcon_bonus: float

    def as_dict(self) -> dict:
        return {
            "wildcardGap": round(self.wildcard_gap, 3),
            "freeHitGap": round(self.free_hit_gap, 3),
            "benchScore": round(self.bench_score, 3),
            "tripleScore": round(self.triple_score, 3),
            "afconBonus": round(self.afcon_bonus, 3),
        }


def chip_policy_pool() -> list[ChipPolicy]:
    rng = np.random.default_rng(20260812)
    policies = [
        ChipPolicy(
            wildcard_gap=float(rng.uniform(2.0, 7.0)),
            free_hit_gap=float(rng.uniform(0.80, 2.80)),
            bench_score=float(rng.uniform(3.45, 4.35)),
            triple_score=float(rng.uniform(1.55, 2.05)),
            afcon_bonus=float(rng.uniform(0.20, 0.80)),
        )
        for _ in range(CHIP_POLICY_TRIALS - 4)
    ]
    policies.extend(
        [
            ChipPolicy(3.0, 1.10, 3.70, 1.70, 0.40),
            ChipPolicy(4.5, 1.60, 3.95, 1.82, 0.55),
            ChipPolicy(2.2, 0.90, 3.55, 1.60, 0.30),
            ChipPolicy(6.0, 2.20, 4.20, 1.95, 0.70),
        ]
    )
    return policies


def chip_windows(season: str, first_gw: int, last_gw: int) -> list[dict]:
    windows = [
        {"chip": "Wildcard", "start": first_gw + 4, "end": min(19, last_gw)},
        {"chip": "Wildcard", "start": max(23, first_gw), "end": last_gw},
    ]
    if season == "2025-26":
        for chip in ("Free Hit", "Bench Boost", "Triple Captain"):
            windows.extend(
                [
                    {"chip": chip, "start": first_gw, "end": min(19, last_gw)},
                    {"chip": chip, "start": max(20, first_gw), "end": last_gw},
                ]
            )
    else:
        for chip in ("Free Hit", "Bench Boost", "Triple Captain"):
            windows.append({"chip": chip, "start": first_gw, "end": last_gw})
    return [window for window in windows if window["start"] <= window["end"]]


def initial_squad(
    frame: pd.DataFrame,
    scores: np.ndarray,
    budget_limit: int = 1000,
    excluded_elements: set[int] | None = None,
) -> list[int]:
    """Fast legal £100m squad build used at the start of each recursive season."""
    if excluded_elements:
        frame = frame[~frame["element"].isin(excluded_elements)]
    best: list[int] = []
    best_score = -math.inf
    frame_indices = frame.index.to_numpy(int)
    prices = frame["price"].to_numpy(int)
    player_positions = frame["position_id"].to_numpy(int)
    player_clubs = frame["team_id"].to_numpy(int)
    price_penalties = np.linspace(0.0, 0.032, 17)
    for penalty in price_penalties:
        adjusted = scores[frame_indices] - penalty * (prices - 35)
        order = np.argsort(adjusted)[::-1]
        chosen: list[int] = []
        positions = {position: 0 for position in SQUAD_QUOTAS}
        clubs: dict[int, int] = {}
        for local_index in order:
            position = int(player_positions[int(local_index)])
            club = int(player_clubs[int(local_index)])
            if positions.get(position, 0) >= SQUAD_QUOTAS.get(position, 0):
                continue
            if clubs.get(club, 0) >= 3:
                continue
            chosen.append(int(frame_indices[int(local_index)]))
            positions[position] += 1
            clubs[club] = clubs.get(club, 0) + 1
            if len(chosen) == 15:
                break
        if len(chosen) != 15:
            continue
        cost = int(data_price_sum(frame, chosen))
        if cost > budget_limit:
            continue
        score = float(scores[chosen].sum())
        if score > best_score:
            best = chosen
            best_score = score
    if len(best) != 15:
        fallback = frame.copy()
        fallback["model_score"] = scores[fallback.index.to_numpy(int)]
        best, _ = pick_squad(fallback)
    return best


def data_price_sum(frame: pd.DataFrame, indices: list[int]) -> int:
    return int(frame.loc[indices, "price"].sum())


def precompute_fresh_squads(
    data: pd.DataFrame, scores: np.ndarray
) -> dict[tuple[str, int], list[int]]:
    fresh: dict[tuple[str, int], list[int]] = {}
    for (season, gw), frame in data.groupby(["season", "GW"], sort=False):
        afcon_window = AFCON_WINDOWS.get(str(season))
        afcon_risk = bool(
            afcon_window and afcon_window[0] - 1 <= int(gw) <= afcon_window[1]
        )
        excluded = set(
            frame.loc[
                afcon_risk & frame["nationality"].isin(AFCON_NATIONS), "element"
            ].astype(int)
        )
        fresh[(str(season), int(gw))] = initial_squad(
            frame, scores, excluded_elements=excluded
        )
    return fresh


def choose_xi(
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    scores: np.ndarray,
    excluded_elements: set[int] | None = None,
) -> tuple[list[int], list[int]]:
    excluded_elements = excluded_elements or set()

    def selection_score(element: int) -> float:
        if element in excluded_elements:
            return -1.5
        return scores[row_by_element[element]] if element in row_by_element else -1.0

    best_xi: list[int] = []
    best_score = -math.inf
    for defenders in (3, 4, 5):
        for forwards in (1, 2, 3):
            midfielders = 10 - defenders - forwards
            if not 2 <= midfielders <= 5:
                continue
            formation = {1: 1, 2: defenders, 3: midfielders, 4: forwards}
            chosen: list[int] = []
            for position, count in formation.items():
                position_pool = [
                    element
                    for element, state in squad.items()
                    if int(state["position"]) == position
                ]
                position_pool.sort(key=selection_score, reverse=True)
                chosen.extend(position_pool[:count])
            total = sum(
                selection_score(element)
                for element in chosen
            )
            if len(chosen) == 11 and total > best_score:
                best_xi = chosen
                best_score = total
    bench = [element for element in squad if element not in set(best_xi)]
    bench.sort(
        key=lambda element: (
            int(squad[element]["position"]) != 1,
            selection_score(element),
        ),
        reverse=True,
    )
    # FPL puts the reserve goalkeeper in a separate slot; outfield order is by score.
    bench_gk = [element for element in bench if int(squad[element]["position"]) == 1]
    bench_outfield = [
        element for element in bench if int(squad[element]["position"]) != 1
    ]
    bench_outfield.sort(
        key=selection_score,
        reverse=True,
    )
    return best_xi, bench_gk + bench_outfield


def legal_xi(elements: list[int], squad: dict[int, dict]) -> bool:
    counts = {
        position: sum(int(squad[element]["position"]) == position for element in elements)
        for position in SQUAD_QUOTAS
    }
    return (
        len(elements) == 11
        and counts[1] == 1
        and counts[2] >= 3
        and counts[3] >= 2
        and counts[4] >= 1
    )


def realised_week_breakdown(
    xi: list[int],
    bench: list[int],
    captain: int,
    vice: int,
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    actual: np.ndarray,
    minutes: np.ndarray,
) -> dict[str, float]:
    def played(element: int) -> bool:
        return element in row_by_element and minutes[row_by_element[element]] > 0

    final_xi = list(xi)
    absent = [element for element in final_xi if not played(element)]
    for substitute in bench:
        if not absent or not played(substitute):
            continue
        for missing in list(absent):
            trial = [substitute if element == missing else element for element in final_xi]
            if legal_xi(trial, squad):
                final_xi = trial
                absent.remove(missing)
                break
    xi_points = sum(
        actual[row_by_element[element]]
        for element in final_xi
        if played(element)
    )
    captain_bonus = 0.0
    if played(captain):
        captain_bonus = float(actual[row_by_element[captain]])
    elif played(vice):
        captain_bonus = float(actual[row_by_element[vice]])
    normal = float(xi_points + captain_bonus)
    all_squad_points = sum(
        actual[row_by_element[element]]
        for element in squad
        if played(element)
    )
    return {
        "normal": normal,
        "bench_boost": float(all_squad_points + captain_bonus),
        "triple_captain": float(normal + captain_bonus),
        "captain_bonus": captain_bonus,
    }


def realised_week_points(
    xi: list[int],
    bench: list[int],
    captain: int,
    vice: int,
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    actual: np.ndarray,
    minutes: np.ndarray,
) -> float:
    return realised_week_breakdown(
        xi, bench, captain, vice, squad, row_by_element, actual, minutes
    )["normal"]


def selling_price(purchase_price: int, current_price: int) -> int:
    if current_price <= purchase_price:
        return current_price
    return purchase_price + (current_price - purchase_price) // 2


def simulate_candidate(
    data: pd.DataFrame,
    scores: np.ndarray,
    strategy: SimulationStrategy,
    chip_policy: ChipPolicy | None = None,
    fresh_squads: dict[tuple[str, int], list[int]] | None = None,
    plan_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Carry one legal squad through each season and make deadline-only transfers."""
    if plan_scores is None:
        plan_scores = scores
    actual = data["points"].to_numpy(float)
    played_minutes = data["minutes"].to_numpy(float)
    element_values = data["element"].to_numpy(int)
    position_values = data["position_id"].to_numpy(int)
    team_values = data["team_id"].to_numpy(int)
    price_values = data["price"].to_numpy(int)
    fixture_counts = data["fixture_count"].to_numpy(int)
    nationality_values = data["nationality"].fillna("").to_numpy(str)
    safe_captain_score = (
        0.42 * data["recent"].to_numpy(float)
        + 0.18 * data["long"].to_numpy(float)
        + 0.14 * data["fixture_now"].to_numpy(float)
        + 0.20 * data["minutes_security"].to_numpy(float)
        + 0.06 * data["crowd"].to_numpy(float)
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    totals = np.zeros(len(seasons), dtype=float)
    season_stats: list[dict] = []

    for season_id, season in enumerate(seasons):
        season_data = data[data["season"] == season]
        weeks = sorted(int(value) for value in season_data["GW"].unique())
        squad: dict[int, dict] = {}
        bank = 0
        free_transfers = 1
        transfers = 0
        hits = 0
        hit_cost = 0
        rolled = 0
        weekly_changes: list[int] = []
        weekly_totals: list[float] = []
        chips = (
            [dict(window, used=False) for window in chip_windows(season, weeks[0], weeks[-1])]
            if chip_policy
            else []
        )
        chip_log: list[dict] = []

        for week_number, gw in enumerate(weeks):
            frame = season_data[season_data["GW"] == gw]
            frame_indices = frame.index.to_numpy(int)
            row_by_element = dict(
                zip(element_values[frame_indices].tolist(), frame_indices.tolist())
            )
            afcon_window = AFCON_WINDOWS.get(season)
            afcon_active = bool(
                afcon_window and afcon_window[0] <= gw <= afcon_window[1]
            )
            afcon_risk_active = bool(
                afcon_window and afcon_window[0] - 1 <= gw <= afcon_window[1]
            )
            excluded_elements = {
                int(element_values[index])
                for index in frame_indices
                if afcon_active and nationality_values[index] in AFCON_NATIONS
            }
            afcon_risk_elements = {
                int(element_values[index])
                for index in frame_indices
                if afcon_risk_active
                and nationality_values[index] in AFCON_NATIONS
            }
            incoming_by_position: dict[int, np.ndarray] = {}
            for position in SQUAD_QUOTAS:
                position_indices = frame_indices[
                    position_values[frame_indices] == position
                ]
                incoming_by_position[position] = position_indices[
                    np.argsort(plan_scores[position_indices])[::-1]
                ][:40]
            if week_number == 0:
                initial_indices = initial_squad(
                    frame, plan_scores, excluded_elements=excluded_elements
                )
                for index in initial_indices:
                    squad[int(element_values[index])] = {
                        "position": int(position_values[index]),
                        "team": int(team_values[index]),
                        "purchase": int(price_values[index]),
                        "last_price": int(price_values[index]),
                        "nationality": str(nationality_values[index]),
                    }
                bank = 1000 - sum(state["purchase"] for state in squad.values())
                weekly_changes.append(15)
                squad_before_transfers = {
                    element: state.copy() for element, state in squad.items()
                }
                bank_before_transfers = bank
                free_transfers_before = free_transfers
                transfers_before = transfers
            else:
                bank_limit = 5 if season in {"2024-25", "2025-26"} else 2
                if season == "2025-26" and gw == 16:
                    free_transfers = 5
                for element, state in squad.items():
                    if element in row_by_element:
                        current_index = row_by_element[element]
                        state["team"] = int(team_values[current_index])
                        state["last_price"] = int(price_values[current_index])
                        state["nationality"] = str(
                            nationality_values[current_index]
                        )
                if afcon_active:
                    excluded_elements.update(
                        element
                        for element, state in squad.items()
                        if str(state.get("nationality", "")) in AFCON_NATIONS
                    )
                if afcon_risk_active:
                    afcon_risk_elements.update(
                        element
                        for element, state in squad.items()
                        if str(state.get("nationality", "")) in AFCON_NATIONS
                    )

                squad_before_transfers = {
                    element: state.copy() for element, state in squad.items()
                }
                bank_before_transfers = bank
                free_transfers_before = free_transfers
                transfers_before = transfers
                changes_this_week = 0
                hits_before = hits
                hit_cost_before = hit_cost
                hit_points_this_week = 0
                for move_number in range(free_transfers + 1):
                    is_hit = move_number >= free_transfers
                    if is_hit and hits >= strategy.max_hits:
                        break
                    team_counts: dict[int, int] = {}
                    for state in squad.values():
                        team_counts[int(state["team"])] = (
                            team_counts.get(int(state["team"]), 0) + 1
                        )
                    best_move: tuple[float, int, int, int, int] | None = None
                    for outgoing, state in squad.items():
                        out_index = row_by_element.get(outgoing)
                        out_score = (
                            plan_scores[out_index]
                            if out_index is not None and outgoing not in excluded_elements
                            else -0.30
                        )
                        current_price = (
                            int(price_values[out_index])
                            if out_index is not None
                            else int(state["last_price"])
                        )
                        sale = selling_price(int(state["purchase"]), current_price)
                        position = int(state["position"])
                        for incoming_index in incoming_by_position[position]:
                            incoming_index = int(incoming_index)
                            incoming_element = int(element_values[incoming_index])
                            if (
                                incoming_element in squad
                                or incoming_element in excluded_elements
                            ):
                                continue
                            incoming_team = int(team_values[incoming_index])
                            incoming_price = int(price_values[incoming_index])
                            if incoming_price > bank + sale:
                                continue
                            if (
                                incoming_team != int(state["team"])
                                and team_counts.get(incoming_team, 0) >= 3
                            ):
                                continue
                            gain = float(plan_scores[int(incoming_index)] - out_score)
                            if best_move is None or gain > best_move[0]:
                                best_move = (
                                    gain,
                                    outgoing,
                                    incoming_element,
                                    int(incoming_index),
                                    sale,
                                )
                    if best_move is None:
                        break
                    gain, outgoing, incoming, incoming_index, sale = best_move
                    move_hurdle = strategy.transfer_hurdle + (
                        4.0 if is_hit else 0.0
                    )
                    if gain <= move_hurdle:
                        break
                    out_index = row_by_element.get(outgoing)
                    immediate_out = (
                        scores[out_index]
                        if out_index is not None and outgoing not in excluded_elements
                        else -0.30
                    )
                    immediate_gain = float(scores[incoming_index] - immediate_out)
                    if is_hit and immediate_gain <= strategy.hit_immediate_hurdle:
                        break
                    bank += sale - int(price_values[incoming_index])
                    del squad[outgoing]
                    squad[incoming] = {
                        "position": int(position_values[incoming_index]),
                        "team": int(team_values[incoming_index]),
                        "purchase": int(price_values[incoming_index]),
                        "last_price": int(price_values[incoming_index]),
                        "nationality": str(nationality_values[incoming_index]),
                    }
                    changes_this_week += 1
                    transfers += 1
                    if is_hit:
                        hits += 1
                        hit_cost += 4
                        hit_points_this_week += 4
                if changes_this_week == 0:
                    rolled += 1
                weekly_changes.append(changes_this_week)
                free_transfers = min(
                    bank_limit,
                    max(0, free_transfers - changes_this_week) + 1,
                )

            xi, bench = choose_xi(
                squad, row_by_element, scores, excluded_elements
            )
            captain_metric = safe_captain_score if strategy.safe_captain else scores
            captain_order = sorted(
                xi,
                key=lambda element: captain_metric[row_by_element[element]]
                if element in row_by_element and element not in excluded_elements
                else -1.0,
                reverse=True,
            )
            captain, vice = captain_order[:2]
            base_breakdown = realised_week_breakdown(
                xi,
                bench,
                captain,
                vice,
                squad,
                row_by_element,
                actual,
                played_minutes,
            )
            week_points = base_breakdown["normal"] - (
                hit_points_this_week if week_number > 0 else 0
            )

            if chip_policy:
                fresh_indices = (
                    fresh_squads.get((season, gw), [])
                    if fresh_squads is not None
                    else initial_squad(
                        frame, scores, excluded_elements=excluded_elements
                    )
                )
                fresh_state = {
                    int(element_values[index]): {
                        "position": int(position_values[index]),
                        "team": int(team_values[index]),
                        "purchase": int(price_values[index]),
                        "last_price": int(price_values[index]),
                        "nationality": str(nationality_values[index]),
                    }
                    for index in fresh_indices
                }
                fresh_xi, fresh_bench = choose_xi(
                    fresh_state, row_by_element, scores, excluded_elements
                )
                fresh_captain_order = sorted(
                    fresh_xi,
                    key=lambda element: captain_metric[row_by_element[element]]
                    if element in row_by_element and element not in excluded_elements
                    else -1.0,
                    reverse=True,
                )
                fresh_captain, fresh_vice = fresh_captain_order[:2]

                def predicted_lineup_value(
                    active_xi: list[int], active_captain: int
                ) -> float:
                    return float(
                        sum(
                            scores[row_by_element[element]]
                            for element in active_xi
                            if element in row_by_element
                            and element not in excluded_elements
                        )
                        + (
                            scores[row_by_element[active_captain]]
                            if active_captain in row_by_element
                            and active_captain not in excluded_elements
                            else 0
                        )
                    )

                current_lineup_value = predicted_lineup_value(xi, captain)
                fresh_lineup_value = predicted_lineup_value(
                    fresh_xi, fresh_captain
                )
                current_squad_value = sum(
                    scores[row_by_element[element]]
                    for element in squad
                    if element in row_by_element
                    and element not in excluded_elements
                )
                fresh_squad_value = sum(
                    scores[row_by_element[element]]
                    for element in fresh_state
                    if element in row_by_element
                    and element not in excluded_elements
                )
                afcon_count = sum(element in afcon_risk_elements for element in squad)
                blank_count = sum(
                    element not in row_by_element or element in excluded_elements
                    for element in squad
                )
                double_count = sum(
                    fixture_counts[row_by_element[element]] > 1
                    for element in fresh_xi
                    if element in row_by_element
                )
                bench_double_count = sum(
                    fixture_counts[row_by_element[element]] > 1
                    for element in bench
                    if element in row_by_element
                    and element not in excluded_elements
                )
                bench_metric = sum(
                    max(0.0, scores[row_by_element[element]])
                    + 0.15 * max(0, fixture_counts[row_by_element[element]] - 1)
                    for element in bench
                    if element in row_by_element
                    and element not in excluded_elements
                )
                captain_index = row_by_element.get(captain)
                triple_metric = (
                    float(captain_metric[captain_index])
                    * max(1, int(fixture_counts[captain_index]))
                    if captain_index is not None and captain not in excluded_elements
                    else 0.0
                )
                metrics = {
                    "Wildcard": fresh_squad_value
                    - current_squad_value
                    + chip_policy.afcon_bonus * afcon_count,
                    "Free Hit": fresh_lineup_value
                    - current_lineup_value
                    + 0.22 * max(0, blank_count - 1)
                    + 0.12 * double_count,
                    "Bench Boost": bench_metric,
                    "Triple Captain": triple_metric,
                }
                thresholds = {
                    "Wildcard": chip_policy.wildcard_gap,
                    "Free Hit": chip_policy.free_hit_gap,
                    "Bench Boost": chip_policy.bench_score,
                    "Triple Captain": chip_policy.triple_score,
                }
                available = [
                    window
                    for window in chips
                    if not window["used"]
                    and int(window["start"]) <= gw <= int(window["end"])
                ]
                def has_structural_signal(chip_name: str) -> bool:
                    if chip_name == "Free Hit":
                        return blank_count >= 3 or double_count >= 5
                    if chip_name == "Bench Boost":
                        return bench_double_count >= 1
                    if chip_name == "Triple Captain":
                        return bool(
                            captain_index is not None
                            and fixture_counts[captain_index] > 1
                        )
                    return True

                choices = [
                    window
                    for window in available
                    if metrics[str(window["chip"])]
                    >= thresholds[str(window["chip"])]
                    and has_structural_signal(str(window["chip"]))
                ]
                chosen_window = max(
                    choices,
                    key=lambda window: metrics[str(window["chip"])]
                    / max(0.01, thresholds[str(window["chip"])]),
                    default=None,
                )
                if chosen_window is not None:
                    chip_name = str(chosen_window["chip"])
                    no_chip_points = week_points
                    if chip_name == "Wildcard":
                        if week_number > 0:
                            hits = hits_before
                            hit_cost = hit_cost_before
                            hit_points_this_week = 0
                            free_transfers = free_transfers_before
                        squad = fresh_state
                        bank = 1000 - sum(
                            int(price_values[index]) for index in fresh_indices
                        )
                        xi, bench = fresh_xi, fresh_bench
                        captain, vice = fresh_captain, fresh_vice
                        base_breakdown = realised_week_breakdown(
                            xi,
                            bench,
                            captain,
                            vice,
                            squad,
                            row_by_element,
                            actual,
                            played_minutes,
                        )
                        week_points = base_breakdown["normal"]
                    elif chip_name == "Free Hit":
                        fresh_breakdown = realised_week_breakdown(
                            fresh_xi,
                            fresh_bench,
                            fresh_captain,
                            fresh_vice,
                            fresh_state,
                            row_by_element,
                            actual,
                            played_minutes,
                        )
                        week_points = fresh_breakdown["normal"]
                        squad = squad_before_transfers
                        bank = bank_before_transfers
                        transfers = transfers_before
                        if week_number > 0:
                            hits = hits_before
                            hit_cost = hit_cost_before
                            hit_points_this_week = 0
                            if weekly_changes[-1] > 0:
                                rolled += 1
                            weekly_changes[-1] = 0
                            free_transfers = min(
                                bank_limit, free_transfers_before + 1
                            )
                    elif chip_name == "Bench Boost":
                        week_points = base_breakdown["bench_boost"]
                    elif chip_name == "Triple Captain":
                        week_points = base_breakdown["triple_captain"]
                    chosen_window["used"] = True
                    chip_log.append(
                        {
                            "chip": chip_name,
                            "gw": gw,
                            "gain": round(float(week_points - no_chip_points)),
                            "signal": round(float(metrics[chip_name]), 3),
                            "reason": "signal cleared threshold",
                        }
                    )

            totals[season_id] += week_points
            weekly_totals.append(float(week_points))

        season_stats.append(
            {
                "season": season,
                "transfers": transfers,
                "hits": hits,
                "hitCost": hit_cost,
                "rolled": rolled,
                "weeksChanged": sum(change > 0 for change in weekly_changes[1:]),
                "gameweeks": len(weeks),
                "weeklyPoints": [round(value, 1) for value in weekly_totals],
                "chips": chip_log,
                "chipPoints": int(sum(item["gain"] for item in chip_log)),
            }
        )
    return totals, season_stats


def recursive_replay(
    data: pd.DataFrame,
    candidates: list[Candidate],
    strategy: SimulationStrategy,
) -> np.ndarray:
    features = feature_matrix(data)
    results = np.zeros((len(candidates), len(SEASONS)), dtype=float)
    for trial_index, candidate in enumerate(candidates):
        model_score = features @ candidate.coefficients
        calibration = 0.72 + 0.56 * model_score
        trial_scores = data["component_xpts"].to_numpy(float) * calibration
        trial_plan_scores = data["component_horizon"].to_numpy(float) * calibration
        results[trial_index], _ = simulate_candidate(
            data, trial_scores, strategy, plan_scores=trial_plan_scores
        )
        if (trial_index + 1) % 40 == 0 or trial_index + 1 == len(candidates):
            print(
                f"Recursive replay {trial_index + 1}/{len(candidates)} "
                f"({strategy.name})"
            )
    return results


def replay_chip_policies(
    data: pd.DataFrame,
    scores: np.ndarray,
    plan_scores: np.ndarray,
    policies: list[ChipPolicy],
) -> tuple[np.ndarray, list[list[dict]], dict[tuple[str, int], list[int]]]:
    fresh_squads = precompute_fresh_squads(data, plan_scores)
    results = np.zeros((len(policies), len(SEASONS)), dtype=float)
    stats: list[list[dict]] = []
    for policy_index, policy in enumerate(policies):
        totals, season_stats = simulate_candidate(
            data,
            scores,
            WEEKLY_CHASE_STRATEGY,
            chip_policy=policy,
            fresh_squads=fresh_squads,
            plan_scores=plan_scores,
        )
        results[policy_index] = totals
        stats.append(season_stats)
        if (policy_index + 1) % 24 == 0 or policy_index + 1 == len(policies):
            print(f"Chip-policy replay {policy_index + 1}/{len(policies)}")
    return results, stats, fresh_squads


def add_rank_target_estimates(
    data: pd.DataFrame, backtest: list[dict]
) -> dict:
    """Attach a transparent top-500k pace proxy and bootstrap confidence.

    Exact historic rank cut-offs are not exposed by the FPL API. The proxy is
    anchored to a published 2,150-point top-500k benchmark. The 2019/20 player
    environment is the normalisation reference for season-to-season scaling.
    """
    evaluation = data[data["season"].isin(EVALUATION_SEASONS)]
    environment: dict[str, float] = {}
    for season, season_frame in evaluation.groupby("season", sort=False):
        weekly_elite = season_frame.groupby("GW")["points"].apply(
            lambda values: float(values.nlargest(min(60, len(values))).mean())
        )
        environment[str(season)] = float(weekly_elite.mean())
    reference = environment.get("2019-20") or float(np.mean(list(environment.values())))
    rng = np.random.default_rng(20260811)
    hit_count = 0
    probabilities: list[float] = []
    margins: list[int] = []
    for item in backtest:
        season_key = str(item["season"]).replace("/", "-")
        target = int(round(2150 * environment[season_key] / reference / 5) * 5)
        weekly = np.asarray(item.pop("weeklyPoints"), dtype=float)
        samples = rng.choice(weekly, size=(4000, len(weekly)), replace=True).sum(axis=1)
        probability = float(np.mean(samples >= target))
        margin = int(item["points"] - target)
        item["top500Target"] = target
        item["targetMargin"] = margin
        item["targetHit"] = margin >= 0
        item["targetProbability"] = round(probability * 100)
        item["estimatedBand"] = (
            "Top 100k pace"
            if margin >= 100
            else "Top 500k pace"
            if margin >= 0
            else "500k-1m pace"
            if margin >= -90
            else "Outside 1m pace"
        )
        hit_count += int(margin >= 0)
        probabilities.append(probability)
        margins.append(margin)
    return {
        "target": "Top 500k",
        "hits": hit_count,
        "seasons": len(backtest),
        "hitRate": round(100 * hit_count / max(1, len(backtest))),
        "averageProbability": round(100 * float(np.mean(probabilities))),
        "averageMargin": round(float(np.mean(margins))),
        "worstMargin": min(margins) if margins else 0,
        "method": (
            "Estimated pace, not an official rank reconstruction: a published "
            "2,150-point benchmark, scaled to each season's top-player scoring environment. "
            "Probability bootstraps that season's weekly scores 4,000 times."
        ),
    }


def pick_squad(players: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Budgeted positional knapsack followed by a three-per-club repair."""
    budget_limit = 1000
    position_options: dict[int, dict[int, tuple[float, tuple[int, ...]]]] = {}
    for position, quota in SQUAD_QUOTAS.items():
        pool = players[players["position_id"] == position].nlargest(70, "model_score")
        states: list[dict[int, tuple[float, tuple[int, ...]]]] = [
            {0: (0.0, tuple())}
        ] + [{} for _ in range(quota)]
        for index, row in pool.iterrows():
            cost = int(row["price"])
            score = float(row["model_score"])
            for count in range(quota, 0, -1):
                for spent, (total, chosen) in list(states[count - 1].items()):
                    new_spent = spent + cost
                    if new_spent > budget_limit:
                        continue
                    new_total = total + score
                    current = states[count].get(new_spent)
                    if current is None or new_total > current[0]:
                        states[count][new_spent] = (new_total, chosen + (int(index),))
        position_options[position] = states[quota]

    combined: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, tuple())}
    for position in SQUAD_QUOTAS:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for spent, (score, chosen) in combined.items():
            for pos_spent, (pos_score, pos_chosen) in position_options[position].items():
                total_spent = spent + pos_spent
                if total_spent > budget_limit:
                    continue
                total_score = score + pos_score
                current = next_states.get(total_spent)
                if current is None or total_score > current[0]:
                    next_states[total_spent] = (total_score, chosen + pos_chosen)
        combined = next_states
    _, (_, chosen_tuple) = max(combined.items(), key=lambda item: item[1][0])
    chosen = list(chosen_tuple)

    for _ in range(10):
        selected = players.loc[chosen]
        counts = selected["team_id"].value_counts()
        excess_teams = counts[counts > 3]
        if excess_teams.empty:
            break
        team_id = int(excess_teams.index[0])
        removable = selected[selected["team_id"] == team_id].sort_values("model_score")
        replaced = False
        spent = int(selected["price"].sum())
        for remove_index, remove in removable.iterrows():
            alternatives = players[
                (players["position_id"] == remove["position_id"])
                & (~players.index.isin(chosen))
                & (players["team_id"] != team_id)
                & (players["price"] <= budget_limit - spent + remove["price"])
            ].sort_values("model_score", ascending=False)
            for add_index, add in alternatives.iterrows():
                add_team_count = int((selected["team_id"] == add["team_id"]).sum())
                if add_team_count < 3:
                    chosen[chosen.index(int(remove_index))] = int(add_index)
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            break

    squad = players.loc[chosen]
    best_xi: list[int] = []
    best_score = -1.0
    for defenders in (3, 4, 5):
        for forwards in (1, 2, 3):
            midfielders = 10 - defenders - forwards
            if not 2 <= midfielders <= 5:
                continue
            xi = []
            for position, count in {1: 1, 2: defenders, 3: midfielders, 4: forwards}.items():
                xi.extend(
                    squad[squad["position_id"] == position]
                    .nlargest(count, "model_score")
                    .index.astype(int)
                    .tolist()
                )
            score = float(players.loc[xi, "model_score"].sum())
            if score > best_score:
                best_score = score
                best_xi = xi
    return chosen, best_xi


def current_recommendation(
    historical: pd.DataFrame, best: Candidate
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], dict]:
    bootstrap = get_json(CURRENT_BOOTSTRAP)
    fixtures = get_json(CURRENT_FIXTURES)
    assert isinstance(bootstrap, dict) and isinstance(fixtures, list)
    teams = pd.DataFrame(bootstrap["teams"])
    current = pd.DataFrame(bootstrap["elements"])
    events = pd.DataFrame(bootstrap["events"])
    next_event = events.loc[~events["finished"].astype(bool)].iloc[0]
    gw_number = int(next_event["id"])
    deadline = str(next_event["deadline_time"])
    team_name = dict(zip(teams["id"], teams["short_name"]))

    prior = historical[historical["season"] == "2025-26"].copy()
    raw_prior_path = CACHE / "2025-26" / "merged_gw.csv"
    raw_prior = pd.read_csv(raw_prior_path, encoding="latin-1", low_memory=False)
    prior_summary = (
        raw_prior.sort_values("GW")
        .groupby("element", as_index=False)
        .agg(
            previous_points=("total_points", "sum"),
            previous_minutes=("minutes", "sum"),
            previous_name=("name", "last"),
        )
    )
    players_2526 = pd.read_csv(
        CACHE / "2025-26" / "players_raw.csv", encoding="latin-1", low_memory=False
    )[["id", "code"]]
    prior_summary = prior_summary.merge(players_2526, left_on="element", right_on="id", how="left")
    prior_summary = prior_summary.rename(columns={"code": "player_code"})
    tails = (
        prior.sort_values("GW")
        .groupby("player_code", as_index=False)
        .agg(
            recent_raw=("recent_raw", "last"),
            long_raw=("long_raw", "last"),
            recent_underlying_raw=("recent_underlying_raw", "last"),
            long_underlying_raw=("long_underlying_raw", "last"),
            minutes_security_raw=("minutes_security_raw", "last"),
        )
    )
    prior_summary = prior_summary.merge(tails, on="player_code", how="left")

    first_fixtures = pd.DataFrame(fixtures)
    first_fixtures = first_fixtures[first_fixtures["event"] == gw_number]
    fixture_map: dict[int, dict] = {}
    for _, fixture in first_fixtures.iterrows():
        fixture_map[int(fixture["team_h"])] = {
            "opponent": int(fixture["team_a"]),
            "home": True,
            "kickoff": fixture["kickoff_time"],
        }
        fixture_map[int(fixture["team_a"])] = {
            "opponent": int(fixture["team_h"]),
            "home": False,
            "kickoff": fixture["kickoff_time"],
        }

    current = current.merge(
        prior_summary[
            [
                "player_code",
                "previous_points",
                "previous_minutes",
                "recent_raw",
                "long_raw",
                "recent_underlying_raw",
                "long_underlying_raw",
                "minutes_security_raw",
            ]
        ],
        left_on="code",
        right_on="player_code",
        how="left",
    )
    current["position_id"] = current["element_type"].astype(int)
    current["team_id"] = current["team"].astype(int)
    current["team_name"] = current["team_id"].map(team_name)
    current["display_name"] = current["web_name"].astype(str)
    current["display_name"] = current["display_name"].str.replace(
        f"Gu{chr(0xFFFD)}hi", "Guehi", regex=False
    )
    current["price"] = current["now_cost"].astype(int)
    current["ownership"] = pd.to_numeric(current["selected_by_percent"], errors="coerce").fillna(0)
    current["ep_next_num"] = pd.to_numeric(current["ep_next"], errors="coerce").fillna(0)
    fallback = current["ep_next_num"].where(current["ep_next_num"] > 0, 2.0)
    current["long_raw"] = current["long_raw"].fillna(fallback)
    current["recent_raw"] = current["recent_raw"].fillna(current["long_raw"])
    current["minutes_security_raw"] = current["minutes_security_raw"].fillna(
        (current["previous_minutes"].fillna(0) / (38 * 90)).clip(0, 1)
    )
    current["minutes_security_raw"] = current["minutes_security_raw"].where(
        current["minutes_security_raw"] > 0,
        (current["ep_next_num"] / 4.5).clip(0.25, 0.85),
    )
    current_ict = pd.to_numeric(current["ict_index"], errors="coerce").fillna(0)
    current_appearances = (
        pd.to_numeric(current["minutes"], errors="coerce").fillna(0) / 90
    ).clip(lower=1)
    underlying_fallback = (current_ict / current_appearances).clip(0, 35)
    current["long_underlying_raw"] = current["long_underlying_raw"].fillna(
        underlying_fallback
    )
    current["recent_underlying_raw"] = current["recent_underlying_raw"].fillna(
        current["long_underlying_raw"]
    )
    current["long_value_raw"] = current["long_raw"] / (current["price"] / 10).clip(3.5)
    current["recent_value_raw"] = current["recent_raw"] / (current["price"] / 10).clip(3.5)
    season_start = date(2026, 8, 1)
    current["age"] = current["birth_date"].map(
        lambda value: (
            (season_start - parse_dob(value)).days / 365.2425
            if parse_dob(value)
            else 27.5
        )
    )
    current["age_raw"] = np.exp(-((current["age"] - 27.5) / 7.5) ** 2)
    current["crowd_raw"] = np.log1p(current["ownership"])
    strengths = teams.set_index("id")
    horizon_fixtures = pd.DataFrame(fixtures)
    horizon_fixtures = horizon_fixtures[
        horizon_fixtures["event"].between(gw_number, gw_number + 5)
    ]

    def fixture_strength(row: pd.Series) -> float:
        fixture = fixture_map.get(int(row["team_id"]))
        if not fixture:
            return 0.5
        opponent = strengths.loc[int(fixture["opponent"])]
        opponent_strength = float(
            opponent["strength_overall_away"]
            if fixture["home"]
            else opponent["strength_overall_home"]
        )
        home_bonus = 30 if fixture["home"] else 0
        return 1400 - opponent_strength + home_bonus

    current["fixture_raw"] = current.apply(fixture_strength, axis=1)
    horizon_map: dict[int, list[tuple[float, float]]] = {}
    horizon_weight = {
        gw_number + offset: weight
        for offset, weight in enumerate((1.0, 0.86, 0.74, 0.64, 0.55, 0.47))
    }
    for _, fixture in horizon_fixtures.iterrows():
        event = int(fixture["event"])
        home = int(fixture["team_h"])
        away = int(fixture["team_a"])
        away_strength = float(strengths.loc[away, "strength_overall_away"])
        home_strength = float(strengths.loc[home, "strength_overall_home"])
        horizon_map.setdefault(home, []).append(
            (1400 - away_strength + 30, horizon_weight[event])
        )
        horizon_map.setdefault(away, []).append(
            (1400 - home_strength, horizon_weight[event])
        )

    def horizon_strength(team_id: int) -> float:
        values = horizon_map.get(int(team_id), [])
        if not values:
            return 0.5
        return sum(value * weight for value, weight in values) / sum(
            weight for _, weight in values
        )

    current["fixture_horizon_raw"] = current["team_id"].map(horizon_strength)
    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_raw", "fixture_now"),
        ("crowd_raw", "crowd"),
        ("minutes_security_raw", "minutes_security"),
        ("long_underlying_raw", "long_underlying"),
        ("recent_underlying_raw", "recent_underlying"),
    ]:
        current[rank_name] = current.groupby("position_id")[raw_name].transform(percentile)
    matrix = feature_matrix(current)
    current["model_score"] = matrix @ best.coefficients
    current["availability"] = pd.to_numeric(
        current["chance_of_playing_next_round"], errors="coerce"
    ).fillna(100)
    current_minutes = pd.to_numeric(current["minutes"], errors="coerce").fillna(0)
    previous_minutes = current_minutes.where(
        current_minutes > 0, current["previous_minutes"].fillna(0)
    )
    nineties = (previous_minutes / 90).clip(lower=0)
    rate_denominator = nineties + 5.0

    def numeric_current(column: str) -> pd.Series:
        if column not in current:
            return pd.Series(0.0, index=current.index)
        return pd.to_numeric(current[column], errors="coerce").fillna(0.0)

    expected_minutes = (
        90
        * (
            0.72 * current["minutes_security_raw"].clip(0, 1)
            + 0.28 * (current["ep_next_num"] / 5.0).clip(0.25, 1)
        )
        * (current["availability"] / 100).clip(0, 1)
    ).clip(15, 90)
    appearance_share = expected_minutes / 90
    goals = numeric_current("goals_scored")
    assists = numeric_current("assists")
    expected_goals = numeric_current("expected_goals")
    expected_assists = numeric_current("expected_assists")
    clean_sheets = numeric_current("clean_sheets")
    saves = numeric_current("saves")
    bonus = numeric_current("bonus")
    yellow_cards = numeric_current("yellow_cards")
    red_cards = numeric_current("red_cards")
    goals_conceded = numeric_current("goals_conceded")
    defensive_points = numeric_current("defensive_contribution")
    defensive_threshold = current["position_id"].map({1: 10, 2: 10, 3: 12, 4: 12}).astype(float)
    goal_points = current["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4}).astype(float)
    clean_points = current["position_id"].map({1: 4, 2: 4, 3: 1, 4: 0}).astype(float)
    component_projection = (
        1.0
        + appearance_share
        + ((goals + expected_goals) / (2 * rate_denominator))
        * appearance_share
        * goal_points
        + ((assists + expected_assists) / (2 * rate_denominator))
        * appearance_share
        * 3
        + ((clean_sheets + 1.5) / rate_denominator).clip(0, 0.75)
        * clean_points
        * appearance_share
        + (saves / rate_denominator / 3) * appearance_share
        + (bonus / rate_denominator) * appearance_share
        + 2
        * (defensive_points / rate_denominator / defensive_threshold).clip(0, 1)
        * appearance_share
        - (yellow_cards + 3 * red_cards) / rate_denominator * appearance_share
        - np.where(
            current["position_id"].isin([1, 2]),
            goals_conceded / rate_denominator / 2 * appearance_share,
            0,
        )
    )
    own_projection = component_projection * (
        0.82 + current["fixture_now"] * 0.28
    ) * (0.74 + current["model_score"] * 0.42)
    current["raw_projection"] = (
        0.52 * current["ep_next_num"].where(current["ep_next_num"] > 0, own_projection)
        + 0.48 * own_projection
    ).clip(0.5, 12.5)
    weighted_games = current["team_id"].map(
        lambda team_id: sum(weight for _, weight in horizon_map.get(int(team_id), []))
    ).clip(lower=1.0)
    current["horizon_projection"] = (
        current["raw_projection"]
        * weighted_games
        * (0.82 + current["fixture"] * 0.30)
    )
    current["expected_minutes"] = expected_minutes
    current["uncertainty"] = (
        1.0
        - (nineties / (nineties + 8)).clip(0, 0.88)
        + (1.0 - current["availability"] / 100).clip(0, 1)
    ).clip(0.08, 1.5)
    current["model_score"] = (
        0.48 * current["model_score"].rank(pct=True)
        + 0.34 * current["horizon_projection"].rank(pct=True)
        + 0.18 * current["raw_projection"].rank(pct=True)
    )
    pool = current[
        (current["status"].isin(["a", "d"]))
        & (current["availability"] >= 75)
        & (current["price"] >= 35)
        & (
            (current["previous_minutes"].fillna(0) >= 180)
            | (current["ownership"] >= 0.5)
            | (current["ep_next_num"] >= 2.5)
        )
    ].copy()
    pool.reset_index(drop=True, inplace=True)
    chosen, xi = pick_squad(pool)
    chosen_set = set(chosen)
    xi_set = set(xi)
    pool["captain_score"] = (
        0.56 * pool["raw_projection"].rank(pct=True)
        + 0.18 * pool["fixture_now"]
        + 0.20 * pool["minutes_security"]
        + 0.06 * pool["crowd"]
    )
    selected = pool.loc[chosen].sort_values(
        ["position_id", "model_score"], ascending=[True, False]
    )
    captain_order = pool.loc[xi].sort_values("captain_score", ascending=False).index.tolist()
    captain = captain_order[0]
    vice = captain_order[1]

    def player_payload(index: int, row: pd.Series) -> dict:
        fixture = fixture_map.get(int(row["team_id"]), {})
        opponent_id = fixture.get("opponent")
        return {
            "id": int(row["id"]),
            "name": str(row["display_name"]),
            "team": str(row["team_name"]),
            "position": POSITION_LABELS[int(row["position_id"])],
            "price": round(float(row["price"]) / 10, 1),
            "ownership": round(float(row["ownership"]), 1),
            "projected": round(float(row["raw_projection"]), 1),
            "sixWeekProjected": round(float(row["horizon_projection"]), 1),
            "expectedMinutes": round(float(row["expected_minutes"])),
            "uncertainty": round(float(row["uncertainty"]), 2),
            "captainRating": round(float(row["captain_score"]) * 100),
            "score": round(float(row["model_score"]) * 100),
            "features": {
                "recent": round(float(row["recent"]), 4),
                "history": round(float(row["long"]), 4),
                "recentValue": round(float(row["recent_value"]), 4),
                "historyValue": round(float(row["long_value"]), 4),
                "age": round(float(row["age_score"]), 4),
                "fixture": round(float(row["fixture"]), 4),
                "crowd": round(float(row["crowd"]), 4),
                "minutes": round(float(row["minutes_security"]), 4),
                "underlying": round(
                    float(
                        best.recent_share * row["recent_underlying"]
                        + (1 - best.recent_share) * row["long_underlying"]
                    ),
                    4,
                ),
            },
            "opponent": team_name.get(opponent_id, "TBD"),
            "venue": "H" if fixture.get("home") else "A",
            "starter": index in xi_set,
            "captain": index == captain,
            "vice": index == vice,
            "trend": "up"
            if float(row["recent_raw"]) > float(row["long_raw"]) + 0.35
            else "down"
            if float(row["recent_raw"]) + 0.35 < float(row["long_raw"])
            else "flat",
        }

    squad = [player_payload(int(index), row) for index, row in selected.iterrows()]
    squad.sort(key=lambda item: (not item["starter"], ["GK", "DEF", "MID", "FWD"].index(item["position"]), -item["score"]))

    top_players = pool.nlargest(12, "model_score")
    watchlist = [player_payload(int(index), row) for index, row in top_players.iterrows()]
    all_players = [
        player_payload(int(index), row)
        for index, row in pool.sort_values("model_score", ascending=False).iterrows()
    ]
    matchups: list[dict] = []
    for _, fixture in first_fixtures.iterrows():
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        match_pool = pool[pool["team_id"].isin([home, away])]
        if match_pool.empty:
            continue
        model_pick = match_pool.nlargest(1, "model_score").iloc[0]
        popular_pick = match_pool.nlargest(1, "ownership").iloc[0]
        matchups.append(
            {
                "fixture": f"{team_name[home]}  v  {team_name[away]}",
                "modelPick": str(model_pick["display_name"]),
                "popularPick": str(popular_pick["display_name"]),
                "edge": round(
                    float(model_pick["model_score"] - popular_pick["model_score"]) * 100
                ),
            }
        )
    matchups.sort(key=lambda item: item["edge"], reverse=True)

    headline = {
        "gameweek": gw_number,
        "season": "2026/27",
        "deadline": deadline,
        "budget": round(float(selected["price"].sum()) / 10, 1),
        "projected": round(float(pool.loc[xi, "raw_projection"].sum()) + float(pool.loc[captain, "raw_projection"]), 1),
        "formation": f"{sum(pool.loc[xi, 'position_id'] == 2)}-{sum(pool.loc[xi, 'position_id'] == 3)}-{sum(pool.loc[xi, 'position_id'] == 4)}",
        "captain": str(pool.loc[captain, "display_name"]),
        "vice": str(pool.loc[vice, "display_name"]),
    }
    current_meta = {
        "playersScored": int(len(pool)),
        "fixturesScored": int(len(first_fixtures)),
        "sourceUpdated": datetime.now().astimezone().isoformat(timespec="minutes"),
    }
    return headline, squad, watchlist, matchups[:6], all_players, current_meta


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    ages = load_age_register()
    nationalities = load_nationality_register()
    season_frames: list[pd.DataFrame] = []
    data_summary: list[dict] = []
    for season in SEASONS:
        frame, summary = build_season(season, ages, nationalities)
        season_frames.append(frame)
        data_summary.append(summary)
        print(f"Prepared {season}: {summary['eligibleRows']:,} eligible player-weeks")

    data = prepare_causal_history(season_frames)
    candidates, baseline_index = candidate_pool()
    snapshot_scores, seasons = snapshot_replay(data, candidates)
    gameweeks = np.array(
        [data.loc[data["season"] == season, "GW"].nunique() for season in seasons],
        dtype=float,
    )
    snapshot_per_gameweek = snapshot_scores / gameweeks
    snapshot_stability = snapshot_per_gameweek.mean(axis=1) - snapshot_per_gameweek.std(axis=1) * 0.18

    # Select the recursive search space using training-only seasons. Reported
    # 2018/19 onward results cannot influence which candidates are evaluated.
    shortlist_indices: list[int] = []
    priority = [baseline_index] + list(range(len(candidates) - 5, len(candidates)))
    training_count = len(TRAINING_SEASONS)
    for season_id in range(training_count):
        priority.extend(
            np.argsort(snapshot_per_gameweek[:, season_id])[-8:][::-1].astype(int).tolist()
        )
    training_snapshot_stability = (
        snapshot_per_gameweek[:, :training_count].mean(axis=1)
        - snapshot_per_gameweek[:, :training_count].std(axis=1) * 0.25
    )
    priority.extend(np.argsort(training_snapshot_stability)[::-1].astype(int).tolist())
    for index in priority:
        if index not in shortlist_indices:
            shortlist_indices.append(index)
        if len(shortlist_indices) == RECURSIVE_FINALISTS:
            break
    recursive_candidates = [candidates[index] for index in shortlist_indices]
    recursive_scores = recursive_replay(
        data, recursive_candidates, WEEKLY_CHASE_STRATEGY
    )
    per_gameweek = recursive_scores / gameweeks
    stability = per_gameweek.mean(axis=1) - per_gameweek.std(axis=1) * 0.18
    best_local_index = int(np.argmax(stability))
    best_index = shortlist_indices[best_local_index]
    best = candidates[best_index]
    baseline_local_index = shortlist_indices.index(baseline_index)

    walk_forward: list[dict] = []
    best_scores, best_plan_scores, _ = candidate_forecasts(data, best)
    chip_policies = chip_policy_pool()
    chip_scores, chip_policy_stats, best_fresh_squads = replay_chip_policies(
        data, best_scores, best_plan_scores, chip_policies
    )
    no_chip_best = recursive_scores[best_local_index]
    chip_gains = chip_scores - no_chip_best
    chip_stability = chip_gains.mean(axis=1) - chip_gains.std(axis=1) * 0.18
    best_chip_policy_index = int(np.argmax(chip_stability))
    best_chip_policy = chip_policies[best_chip_policy_index]

    def blend_candidates(indices: np.ndarray) -> Candidate:
        values = np.array(
            [
                [
                    recursive_candidates[int(index)].performance,
                    recursive_candidates[int(index)].value,
                    recursive_candidates[int(index)].age,
                    recursive_candidates[int(index)].fixture,
                    recursive_candidates[int(index)].crowd,
                    recursive_candidates[int(index)].minutes,
                    recursive_candidates[int(index)].underlying,
                    recursive_candidates[int(index)].recent_share,
                ]
                for index in indices
            ],
            dtype=float,
        )
        return Candidate(*values.mean(axis=0).tolist())

    def blend_chip_policies(indices: np.ndarray) -> ChipPolicy:
        values = np.array(
            [
                [
                    chip_policies[int(index)].wildcard_gap,
                    chip_policies[int(index)].free_hit_gap,
                    chip_policies[int(index)].bench_score,
                    chip_policies[int(index)].triple_score,
                    chip_policies[int(index)].afcon_bonus,
                ]
                for index in indices
            ],
            dtype=float,
        )
        return ChipPolicy(*values.mean(axis=0).tolist())

    for season_id, season in enumerate(seasons):
        if season_id == 0:
            trial_candidate = candidates[-5]
            mode = "fixed preseason seed"
            trial_policy = chip_policies[-4]
            chip_mode = "fixed preseason seed"
        else:
            prior = per_gameweek[:, :season_id]
            train_score = prior.mean(axis=1) - prior.std(axis=1) * 0.25
            ensemble_indices = np.argsort(train_score)[-12:]
            trial_candidate = blend_candidates(ensemble_indices)
            mode = f"12-model ensemble trained on {season_id} prior season{'s' if season_id != 1 else ''}"
            prior_chip_gain = chip_gains[:, :season_id]
            chip_train_score = (
                prior_chip_gain.mean(axis=1)
                - prior_chip_gain.std(axis=1) * 0.25
            )
            chip_ensemble_indices = np.argsort(chip_train_score)[-12:]
            trial_policy = blend_chip_policies(chip_ensemble_indices)
            chip_mode = (
                f"12-policy ensemble trained on {season_id} prior "
                f"season{'s' if season_id != 1 else ''}"
            )
        trial_scores, trial_plan_scores, _ = candidate_forecasts(data, trial_candidate)
        no_chip_totals, no_chip_stats = simulate_candidate(
            data,
            trial_scores,
            WEEKLY_CHASE_STRATEGY,
            plan_scores=trial_plan_scores,
        )
        trial_fresh_squads = precompute_fresh_squads(data, trial_plan_scores)
        trial_totals, trial_stats = simulate_candidate(
            data,
            trial_scores,
            WEEKLY_CHASE_STRATEGY,
            chip_policy=trial_policy,
            fresh_squads=trial_fresh_squads,
            plan_scores=trial_plan_scores,
        )
        points = round(float(trial_totals[season_id]))
        baseline = round(float(no_chip_totals[season_id]))
        season_transfer_stats = trial_stats[season_id]
        if season not in EVALUATION_SEASONS:
            continue
        walk_forward.append(
            {
                "season": season.replace("-", "/"),
                "points": points,
                "baseline": baseline,
                "uplift": round((points / baseline - 1) * 100, 1) if baseline else 0,
                "mode": mode,
                "chipMode": chip_mode,
                "weights": trial_candidate.as_dict(),
                "transfers": season_transfer_stats["transfers"],
                "weeksChanged": season_transfer_stats["weeksChanged"],
                "rolled": season_transfer_stats["rolled"],
                "hits": season_transfer_stats["hits"],
                "hitCost": season_transfer_stats["hitCost"],
                "gameweeks": season_transfer_stats["gameweeks"],
                "weeklyPoints": season_transfer_stats["weeklyPoints"],
                "chipPoints": points - baseline,
                "chips": season_transfer_stats["chips"],
                "legacyBaseline": round(
                    float(recursive_scores[baseline_local_index, season_id])
                ),
            }
        )

    top_indices = np.argsort(stability)[-5:][::-1]
    leaderboard = [
        {
            "rank": rank + 1,
            "trial": shortlist_indices[int(index)] + 1,
            "pointsPerGameweek": round(float(per_gameweek[index].mean()), 2),
            "stability": round(float(stability[index]), 3),
            "weights": recursive_candidates[int(index)].as_dict(),
        }
        for rank, index in enumerate(top_indices)
    ]
    curve_indices = np.linspace(0, len(recursive_candidates) - 1, 16).astype(int)
    sorted_scores = np.sort(stability)
    calibration_curve = [
        {
            "percentile": round(int(index) / (len(candidates) - 1) * 100),
            "score": round(float(sorted_scores[index]), 2),
        }
        for index in curve_indices
    ]

    best_totals, best_stats = simulate_candidate(
        data, best_scores, WEEKLY_CHASE_STRATEGY, plan_scores=best_plan_scores
    )
    weekly_safe_captain = SimulationStrategy(
        "Six-GW planner + safe captain", 5.00, 5, False, True, 0, 99.0
    )
    patient_model_captain = SimulationStrategy(
        "Patient six-GW transfers + model captain", 8.00, 5, False, False, 0, 99.0
    )
    permissive_hit_strategy = SimulationStrategy(
        "Six-GW planner + three paid hits", 5.00, 5, False, False, 3, 2.5
    )
    safe_captain_totals, _ = simulate_candidate(
        data, best_scores, weekly_safe_captain, plan_scores=best_plan_scores
    )
    patient_totals, patient_stats = simulate_candidate(
        data, best_scores, patient_model_captain, plan_scores=best_plan_scores
    )
    permissive_hit_totals, _ = simulate_candidate(
        data, best_scores, permissive_hit_strategy, plan_scores=best_plan_scores
    )
    baseline_totals = recursive_scores[baseline_local_index]

    def advice_test(label: str, improved: np.ndarray, comparison: np.ndarray, detail: str) -> dict:
        delta = float(np.mean(improved - comparison))
        return {
            "label": label,
            "delta": round(delta, 1),
            "result": "helped" if delta > 0.05 else "hurt" if delta < -0.05 else "neutral",
            "detail": detail,
        }

    expert_tests = [
        advice_test(
            "Paid-hit restraint",
            best_totals,
            permissive_hit_totals,
            "The calibration allows a paid-hit alternative, but keeps it only if the historical replay beats patient free transfers.",
        ),
        advice_test(
            "Patience over churn",
            patient_totals,
            best_totals,
            "Bank a transfer unless the best same-position upgrade clears the model hurdle.",
        ),
        advice_test(
            "Safety-first captaincy",
            safe_captain_totals,
            best_totals,
            "Blend proven output, the immediate fixture and 60-minute security; ownership is only a tiebreaker.",
        ),
        advice_test(
            "Expert data layer",
            best_totals,
            baseline_totals,
            "Add component expected points, six-GW fixtures, minutes security and underlying involvement to the original Lens feature set.",
        ),
    ]

    best_chip_totals = chip_scores[best_chip_policy_index]
    best_chip_stats = chip_policy_stats[best_chip_policy_index]
    chip_gains_by_type: dict[str, list[int]] = {
        "Wildcard": [],
        "Free Hit": [],
        "Bench Boost": [],
        "Triple Captain": [],
    }
    for season_stat in best_chip_stats:
        for chip in season_stat["chips"]:
            chip_gains_by_type[str(chip["chip"])].append(int(chip["gain"]))
    chip_breakdown = [
        {
            "chip": chip,
            "uses": len(gains),
            "averageGain": round(float(np.mean(gains)), 1) if gains else 0.0,
            "totalGain": int(sum(gains)),
        }
        for chip, gains in chip_gains_by_type.items()
    ]

    rank_target = add_rank_target_estimates(data, walk_forward)
    headline, squad, watchlist, matchups, all_players, current_meta = current_recommendation(data, best)
    result = {
        "product": "FPL Lens",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
        "model": {
            "version": "Lens 4.0",
            "trials": len(candidates),
            "recursiveTrials": len(recursive_candidates),
            "seasons": len(EVALUATION_SEASONS),
            "trainingSeasons": len(TRAINING_SEASONS),
            "playerWeeks": int(len(data)),
            "bestTrial": best_index + 1,
            "weights": best.as_dict(),
            "method": "Leak-free six-GW walk-forward replay: one legal 15-player squad, banked transfers and chip inventory carry into every deadline using only information then available.",
            "objective": "Maximise legal autosubbed XI, captain and chip points while penalising season-to-season volatility.",
            "strategy": WEEKLY_CHASE_STRATEGY.name,
        },
        "headline": headline,
        "currentMeta": current_meta,
        "squad": squad,
        "watchlist": watchlist,
        "fixtureMatchups": matchups,
        "currentPlayers": all_players,
        "backtest": walk_forward,
        "rankTarget": rank_target,
        "chipStrategy": {
            "policyTrials": len(chip_policies),
            "policy": best_chip_policy.as_dict(),
            "averageGain": round(float(np.mean(best_chip_totals - best_totals)), 1),
            "walkForwardAverageGain": round(
                float(np.mean([item["chipPoints"] for item in walk_forward])), 1
            ),
            "breakdown": chip_breakdown,
            "seasonPlans": [
                {
                    "season": stat["season"].replace("-", "/"),
                    "chipPoints": stat["chipPoints"],
                    "chips": stat["chips"],
                }
                for stat in best_chip_stats
            ],
            "current": {
                "chip": "Hold",
                "gameweek": headline["gameweek"],
                "reason": "Single-Gameweek slate and a freshly optimised squad. Preserve the first-half chips for a larger fixture or availability edge.",
                "nextReview": "Re-score after every deadline; blank clashes, double fixtures, injuries and rotation are explicit triggers.",
            },
            "rules": "Two of each chip: one set through GW19 and one from GW20, with one chip permitted per Gameweek.",
        },
        "expertTests": expert_tests,
        "simulationSummary": {
            "averageTransfers": round(float(np.mean([item["transfers"] for item in best_stats])), 1),
            "averageWeeksChanged": round(float(np.mean([item["weeksChanged"] for item in best_stats])), 1),
            "averageRolled": round(float(np.mean([item["rolled"] for item in best_stats])), 1),
            "averageHits": round(float(np.mean([item["hits"] for item in best_stats])), 1),
            "patientAverageTransfers": round(float(np.mean([item["transfers"] for item in patient_stats])), 1),
        },
        "leaderboard": leaderboard,
        "calibrationCurve": calibration_curve,
        "dataSummary": [
            item for item in data_summary if item["season"] in EVALUATION_SEASONS
        ],
        "sources": [
            {
                "label": "Historical FPL dataset",
                "url": "https://github.com/vaastav/Fantasy-Premier-League",
            },
            {
                "label": "Official FPL API",
                "url": "https://fantasy.premierleague.com/api/bootstrap-static/",
            },
            {
                "label": "Reep identity register",
                "url": "https://github.com/withqwerty/reep",
            },
            {
                "label": "FPL champion: squad + transfers",
                "url": "https://www.premierleague.com/en/news/4671982/fpl-champion-how-to-build-the-perfect-squad-and-make-the-best-transfers",
            },
            {
                "label": "FPL champion: captaincy + chips",
                "url": "https://www.premierleague.com/en/news/4672128/fpl-champion-how-to-pick-your-captain-and-maximise-your-chips",
            },
            {
                "label": "FPL champion: 4–6 GW planning",
                "url": "https://www.premierleague.com/en/news/4025381",
            },
            {
                "label": "Official 2026/27 FPL changes",
                "url": "https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627",
            },
            {
                "label": "FPL Review projection methodology",
                "url": "https://docs.fplreview.com/getting-started/about-fplreview/",
            },
            {
                "label": "Published top-500k points benchmark",
                "url": "https://www.fantasyfootballscout.co.uk/2019/11/28/quantifying-the-impact-of-fpl-decisions-with-chip-season-arriving-earlier/",
            },
            {
                "label": "Official FPL chip rules",
                "url": "https://fantasy.premierleague.com/help/",
            },
            {
                "label": "FPL chip strategy basics",
                "url": "https://www.premierleague.com/en/news/2174900/fpl-basics-chips",
            },
        ],
        "notes": [
            "Every historical GW is recursive: the prior squad, bank and season-correct free-transfer cap carry forward; transfers use contemporaneous prices and FPL selling-price rules.",
            "The replay selects a legal formation, orders the bench, applies autosubs and hands the armband to the vice-captain when required.",
            "Chip decisions are causal threshold rules, not hindsight-selected best weeks: AFCON disruption informs Wildcards, blank/double clashes inform Free Hits, and doubles raise Bench Boost and Triple Captain signals.",
            "The transfer planner looks six Gameweeks ahead and can bank up to the cap that applied in that season; paid-hit variants were tested and rejected when they reduced replay points.",
            "The top-500k result is an estimated scoring-pace test because official historic rank cut-offs are not exposed by the FPL API.",
            "Age is an availability/consistency prior, not a claim that younger or older players are inherently better.",
            "Current projections are decision support, not guarantees; late team news should override the model.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Best trial {best_index + 1}/{len(candidates)} from "
        f"{len(recursive_candidates)} recursive finalists; wrote {OUTPUT.relative_to(ROOT)}"
    )


def refresh_current_artifact() -> None:
    """Refresh live recommendations without rerunning the expensive calibration."""
    if not OUTPUT.exists():
        raise FileNotFoundError("Run the full calibration before --refresh-current")
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    weights = result["model"]["weights"]
    best = Candidate(
        weights["performance"] / 100,
        weights["value"] / 100,
        weights["age"] / 100,
        weights["fixture"] / 100,
        weights["crowd"] / 100,
        weights["minutes"] / 100,
        weights["underlying"] / 100,
        weights["recent"] / 100,
    )
    frame, _ = build_season(
        "2025-26", load_age_register(), load_nationality_register()
    )
    historical = prepare_causal_history([frame])
    headline, squad, watchlist, matchups, all_players, current_meta = (
        current_recommendation(historical, best)
    )
    result.update(
        {
            "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
            "headline": headline,
            "squad": squad,
            "watchlist": watchlist,
            "fixtureMatchups": matchups,
            "currentPlayers": all_players,
            "currentMeta": current_meta,
        }
    )
    result["rankTarget"]["method"] = (
        "Estimated pace, not an official rank reconstruction: a published "
        "2,150-point benchmark, scaled to each season's top-player scoring environment. "
        "Probability bootstraps that season's weekly scores 4,000 times."
    )
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Refreshed current recommendations in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--refresh-current" in sys.argv:
        refresh_current_artifact()
    else:
        main()

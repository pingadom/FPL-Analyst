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


def sigmoid(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    clipped = np.clip(values, -18, 18)
    return 1 / (1 + np.exp(-clipped))


def poisson_tail(
    mean: pd.Series | np.ndarray | float, threshold: pd.Series | np.ndarray | float
) -> np.ndarray:
    """Return P(X >= threshold) for a Poisson variable without SciPy."""
    lam = np.asarray(mean, dtype=float)
    cut = np.asarray(threshold, dtype=int)
    result = np.zeros(np.broadcast(lam, cut).shape, dtype=float)
    lam, cut = np.broadcast_arrays(lam, cut)
    for target in np.unique(cut):
        mask = cut == target
        target_int = max(1, int(target))
        local = lam[mask]
        probability = np.exp(-local)
        cumulative = probability.copy()
        for count in range(1, target_int):
            probability = probability * local / count
            cumulative += probability
        result[mask] = 1 - cumulative
    return np.clip(result, 0, 1)


def normal_cdf(values: pd.Series | np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1 + erf(array / math.sqrt(2)))


def monotone_probability_map(
    successes: np.ndarray, counts: np.ndarray, global_rate: float
) -> np.ndarray:
    """Beta-smoothed isotonic bin estimates without an extra dependency."""
    shrinkage = 32.0
    values = (successes + shrinkage * global_rate) / (counts + shrinkage)
    weights = counts + shrinkage
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left[3] + right[3]
            blocks.append(
                [
                    left[0],
                    right[1],
                    (left[2] * left[3] + right[2] * right[3]) / merged_weight,
                    merged_weight,
                ]
            )
    mapped = np.zeros(10, dtype=float)
    for start, end, value, _ in blocks:
        mapped[int(start) : int(end) + 1] = value
    return np.clip(mapped, 0.005, 0.995)


def causal_calibrate_distributions(data: pd.DataFrame) -> pd.DataFrame:
    """Calibrate event probabilities and 80% bands using prior GWs only."""
    data = data.reset_index(drop=True)
    event_specs = {
        "blank_probability": ("raw_blank_probability", data["points"] <= 2, 0.72),
        "return5_probability": ("raw_return5_probability", data["points"] >= 5, 0.12),
        "haul8_probability": ("raw_haul8_probability", data["points"] >= 8, 0.045),
    }
    calibrated = {
        output: data[raw].to_numpy(float).copy()
        for output, (raw, _, _) in event_specs.items()
    }
    half_width = np.zeros(len(data), dtype=float)
    state: dict[int, dict[str, object]] = {}
    ordered_groups = data.groupby(["season_order", "GW"], sort=True).groups
    for _, group_index in ordered_groups.items():
        group_positions = data.loc[group_index, "position_id"].astype(int)
        for position in sorted(group_positions.unique()):
            indices = np.asarray(
                group_positions[group_positions == position].index, dtype=int
            )
            position_state = state.setdefault(
                position,
                {
                    "events": {
                        output: {
                            "successes": np.zeros(10, dtype=float),
                            "counts": np.zeros(10, dtype=float),
                            "total_successes": 0.0,
                            "total_count": 0.0,
                        }
                        for output in event_specs
                    },
                    "ratio_hist": np.zeros(81, dtype=float),
                },
            )
            events = position_state["events"]
            assert isinstance(events, dict)
            for output, (raw_column, _, prior_rate) in event_specs.items():
                event_state = events[output]
                assert isinstance(event_state, dict)
                total_count = float(event_state["total_count"])
                global_rate = (
                    float(event_state["total_successes"]) + 60 * prior_rate
                ) / (total_count + 60)
                mapping = monotone_probability_map(
                    np.asarray(event_state["successes"], dtype=float),
                    np.asarray(event_state["counts"], dtype=float),
                    global_rate,
                )
                raw_values = data.loc[indices, raw_column].to_numpy(float)
                bins = np.minimum((raw_values * 10).astype(int), 9)
                calibrated[output][indices] = mapping[bins]
            ratio_hist = np.asarray(position_state["ratio_hist"], dtype=float)
            if ratio_hist.sum() >= 250:
                quantile_bin = int(
                    np.searchsorted(np.cumsum(ratio_hist), 0.80 * ratio_hist.sum())
                )
                ratio_80 = max(0.25, min(2.50, quantile_bin / 20))
            else:
                ratio_80 = 0.70
            half_width[indices] = (
                data.loc[indices, "prediction_uncertainty"].to_numpy(float)
                * ratio_80
            )

        # Only update after scoring the whole deadline, preventing same-GW leakage.
        for position in sorted(group_positions.unique()):
            indices = np.asarray(
                group_positions[group_positions == position].index, dtype=int
            )
            position_state = state[position]
            events = position_state["events"]
            assert isinstance(events, dict)
            for output, (raw_column, target, _) in event_specs.items():
                raw_values = data.loc[indices, raw_column].to_numpy(float)
                bins = np.minimum((raw_values * 10).astype(int), 9)
                outcomes = target.loc[indices].to_numpy(float)
                event_state = events[output]
                assert isinstance(event_state, dict)
                np.add.at(event_state["counts"], bins, 1)
                np.add.at(event_state["successes"], bins, outcomes)
                event_state["total_count"] = float(event_state["total_count"]) + len(indices)
                event_state["total_successes"] = float(
                    event_state["total_successes"]
                ) + float(outcomes.sum())
            ratios = (
                np.abs(
                    data.loc[indices, "points"].to_numpy(float)
                    - data.loc[indices, "component_xpts"].to_numpy(float)
                )
                / data.loc[indices, "prediction_uncertainty"].to_numpy(float).clip(0.1)
            ).clip(0, 4)
            ratio_bins = np.minimum((ratios * 20).astype(int), 80)
            np.add.at(position_state["ratio_hist"], ratio_bins, 1)

    for output, values in calibrated.items():
        data[output] = values
    data["prediction_half_width_80"] = half_width.clip(0.4, 9.0)
    data["prediction_p10"] = (
        data["component_xpts"] - data["prediction_half_width_80"]
    ).clip(lower=0)
    data["prediction_p90"] = (
        data["component_xpts"] + data["prediction_half_width_80"]
    ).clip(upper=25)
    return data


def calibrate_live_distributions(
    current: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    """Apply terminal historical calibration maps to the next deadline."""
    event_specs = {
        "blank_probability": ("raw_blank_probability", historical["points"] <= 2, 0.72),
        "return5_probability": ("raw_return5_probability", historical["points"] >= 5, 0.12),
        "haul8_probability": ("raw_haul8_probability", historical["points"] >= 8, 0.045),
    }
    for position in sorted(current["position_id"].astype(int).unique()):
        current_mask = current["position_id"].astype(int) == position
        history_mask = historical["position_id"].astype(int) == position
        for output, (raw_column, target, prior_rate) in event_specs.items():
            raw_history = historical.loc[history_mask, raw_column].to_numpy(float)
            bins = np.minimum((raw_history * 10).astype(int), 9)
            outcomes = target.loc[history_mask].to_numpy(float)
            counts = np.bincount(bins, minlength=10).astype(float)
            successes = np.bincount(bins, weights=outcomes, minlength=10).astype(float)
            global_rate = (outcomes.sum() + 60 * prior_rate) / (len(outcomes) + 60)
            mapping = monotone_probability_map(successes, counts, global_rate)
            live_raw = current.loc[current_mask, raw_column].to_numpy(float)
            live_bins = np.minimum((live_raw * 10).astype(int), 9)
            current.loc[current_mask, output] = mapping[live_bins]
        ratios = (
            np.abs(
                historical.loc[history_mask, "points"].to_numpy(float)
                - historical.loc[history_mask, "component_xpts"].to_numpy(float)
            )
            / historical.loc[
                history_mask, "prediction_uncertainty"
            ].to_numpy(float).clip(0.1)
        ).clip(0, 4)
        ratio_80 = float(np.quantile(ratios, 0.80)) if len(ratios) else 0.70
        current.loc[current_mask, "prediction_half_width_80"] = (
            current.loc[current_mask, "projection_std"] * np.clip(ratio_80, 0.25, 2.50)
        )
    current["prediction_p10"] = (
        current["raw_projection"] - current["prediction_half_width_80"]
    ).clip(lower=0)
    current["prediction_p50"] = current["raw_projection"]
    current["prediction_p90"] = (
        current["raw_projection"] + current["prediction_half_width_80"]
    ).clip(upper=25)
    return current


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
        "kickoff_time",
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
        "bps",
        "starts",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "key_passes",
        "big_chances_created",
        "open_play_crosses",
        "penalties_missed",
        "penalties_saved",
        "own_goals",
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
        "bps",
        "starts",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "key_passes",
        "big_chances_created",
        "open_play_crosses",
        "penalties_missed",
        "penalties_saved",
        "own_goals",
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

    starts_available = "starts" in gw.columns
    defensive_exact_available = (
        "defensive_contribution" in gw.columns
        or {
            "clearances_blocks_interceptions",
            "recoveries",
            "tackles",
        }.issubset(gw.columns)
    )
    raw["start_observed"] = (
        raw["starts"].clip(0, 1)
        if starts_available
        else (raw["minutes"] >= 45).astype(float)
    )
    raw["appearance_observed"] = (raw["minutes"] > 0).astype(float)
    raw["sixty_observed"] = (raw["minutes"] >= 60).astype(float)
    raw["bench_appearance_observed"] = (
        (raw["appearance_observed"] > 0) & (raw["start_observed"] <= 0)
    ).astype(float)
    raw["start_minutes_total"] = raw["minutes"] * raw["start_observed"]
    raw["bench_minutes_total"] = raw["minutes"] * raw["bench_appearance_observed"]
    reconstructed_defence = (
        raw["clearances_blocks_interceptions"] + raw["tackles"]
    ) + np.where(raw["position_id"].isin([3, 4]), raw["recoveries"], 0)
    raw["defensive_actions_observed"] = np.where(
        "defensive_contribution" in gw.columns,
        raw["defensive_contribution"],
        reconstructed_defence,
    )
    raw["defensive_exact"] = float(defensive_exact_available)
    # Missing middle-season event feeds are assigned a transparent post-match
    # proxy for counterfactual scoring only. This proxy never enters a deadline
    # feature as if it had been observed.
    defensive_proxy = (
        raw["position_id"].map({1: 0.0, 2: 4.2, 3: 3.8, 4: 2.1}).fillna(2.5)
        + 0.075 * raw["influence"].clip(lower=0)
        + 0.055 * raw["bps"].clip(lower=0)
    ) * (raw["minutes"] / 90).clip(0, 1)
    raw["defensive_actions_counterfactual"] = np.where(
        raw["defensive_exact"] > 0,
        raw["defensive_actions_observed"],
        defensive_proxy,
    )
    defensive_threshold = np.where(raw["position_id"] == 2, 10, 12)
    defensive_probability = poisson_tail(
        raw["defensive_actions_counterfactual"], defensive_threshold
    )
    raw["current_rule_dc_points"] = np.where(
        raw["position_id"].isin([2, 3, 4]),
        np.where(
            raw["defensive_exact"] > 0,
            2 * (raw["defensive_actions_observed"] >= defensive_threshold),
            2 * defensive_probability,
        ),
        0,
    )
    # The 2026/27 BPS revision is smaller than the defensive-points change.
    # Apply a conservative role adjustment without claiming unavailable Opta
    # sub-components were reconstructed exactly.
    raw["current_rule_bps_adjustment"] = np.select(
        [
            raw["position_id"] == 1,
            (raw["position_id"] == 2) & (raw["defensive_actions_counterfactual"] >= 10),
            raw["position_id"].isin([3, 4]),
        ],
        [0.07, -0.06, 0.03],
        default=0.0,
    ) * raw["appearance_observed"]
    dc_already_scored = season == "2025-26"
    raw["points_current_rules"] = (
        raw["total_points"]
        + (0 if dc_already_scored else raw["current_rule_dc_points"])
        + raw["current_rule_bps_adjustment"]
    )

    # One row per club-fixture prevents team goals/xG from being counted once
    # for every player. These realised values are shifted before being used as
    # features, so a deadline can only see earlier matches.
    team_fixtures = (
        raw.dropna(subset=["team_id", "opponent_team", "fixture"])
        .groupby(
            ["team_id", "GW", "fixture", "opponent_team", "was_home"],
            as_index=False,
        )
        .agg(
            kickoff_time=("kickoff_time", "first"),
            team_goals=("goals_scored", "sum"),
            team_xg=("expected_goals", "sum"),
            team_goals_against=("goals_conceded", "max"),
            team_xga=("expected_goals_conceded", "max"),
            team_clean_sheet=("clean_sheets", "max"),
        )
    )
    team_fixtures["kickoff_time"] = pd.to_datetime(
        team_fixtures["kickoff_time"], errors="coerce", utc=True
    )
    team_fixtures.sort_values(["team_id", "kickoff_time", "GW"], inplace=True)
    team_fixtures["team_rest_days"] = (
        team_fixtures.groupby("team_id")["kickoff_time"].diff().dt.total_seconds()
        / 86400
    ).clip(2, 14).fillna(7)
    team_fixtures["team_result_points"] = np.select(
        [
            team_fixtures["team_goals"] > team_fixtures["team_goals_against"],
            team_fixtures["team_goals"] == team_fixtures["team_goals_against"],
        ],
        [3.0, 1.0],
        default=0.0,
    )
    team_weeks = (
        team_fixtures.groupby(["team_id", "GW"], as_index=False)
        .agg(
            team_games=("fixture", "nunique"),
            team_goals=("team_goals", "sum"),
            team_xg=("team_xg", "sum"),
            team_goals_against=("team_goals_against", "sum"),
            team_xga=("team_xga", "sum"),
            team_clean_sheets=("team_clean_sheet", "sum"),
            team_result_points=("team_result_points", "sum"),
            team_rest_days=("team_rest_days", "min"),
        )
    )

    weekly = (
        raw.groupby(["element", "GW"], as_index=False)
        .agg(
            points=("total_points", "sum"),
            points_current_rules=("points_current_rules", "sum"),
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
            bps=("bps", "sum"),
            starts_observed=("start_observed", "sum"),
            appearances_observed=("appearance_observed", "sum"),
            sixty_observed=("sixty_observed", "sum"),
            bench_appearances_observed=("bench_appearance_observed", "sum"),
            start_minutes_total=("start_minutes_total", "sum"),
            bench_minutes_total=("bench_minutes_total", "sum"),
            defensive_exact=("defensive_exact", "max"),
            defensive_actions_observed=("defensive_actions_observed", "sum"),
            defensive_actions_counterfactual=("defensive_actions_counterfactual", "sum"),
            current_rule_dc_points=("current_rule_dc_points", "sum"),
            key_passes=("key_passes", "sum"),
            big_chances_created=("big_chances_created", "sum"),
            open_play_crosses=("open_play_crosses", "sum"),
            penalties_missed=("penalties_missed", "sum"),
            penalties_saved=("penalties_saved", "sum"),
            own_goals=("own_goals", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            expected_goals=("expected_goals", "sum"),
            expected_assists=("expected_assists", "sum"),
            expected_goals_conceded=("expected_goals_conceded", "sum"),
            defensive_actions=("defensive_actions_counterfactual", "sum"),
        )
        .sort_values(["element", "GW"])
    )
    weekly = weekly.merge(team_weeks, on=["team_id", "GW"], how="left")
    for column in [
        "team_games",
        "team_goals",
        "team_xg",
        "team_goals_against",
        "team_xga",
        "team_clean_sheets",
        "team_result_points",
        "team_rest_days",
    ]:
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce").fillna(0)
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
    weekly["transfer_pressure_raw"] = (
        weekly["transfers_balance"]
        / weekly["selected"].abs().clip(lower=2500)
    ).clip(-1.5, 1.5)
    weekly["transfer_pressure_rank"] = weekly.groupby("GW")[
        "transfer_pressure_raw"
    ].transform(percentile)
    weekly["price_rise_probability"] = sigmoid(
        11 * (weekly["transfer_pressure_rank"] - 0.72)
    )
    weekly["price_fall_probability"] = sigmoid(
        11 * (0.28 - weekly["transfer_pressure_rank"])
    )
    weekly["next_price_change"] = by_player["price"].shift(-1) - weekly["price"]

    allowed = (
        weekly.groupby(["opponent_team", "position_id", "GW"], as_index=False)
        .agg(
            points_allowed=("points", "mean"),
            goals_allowed=("goals", "mean"),
            assists_allowed=("assists", "mean"),
            xg_allowed=("expected_goals", "mean"),
        )
        .sort_values(["opponent_team", "position_id", "GW"])
    )
    allowed["fixture_raw"] = allowed.groupby(
        ["opponent_team", "position_id"], sort=False
    )["points_allowed"].transform(lambda values: values.expanding().mean().shift(1))
    for source, target in [
        ("goals_allowed", "opponent_goal_vulnerability"),
        ("assists_allowed", "opponent_assist_vulnerability"),
        ("xg_allowed", "opponent_xg_vulnerability"),
    ]:
        allowed[target] = allowed.groupby(
            ["opponent_team", "position_id"], sort=False
        )[source].transform(lambda values: values.rolling(10, min_periods=2).mean().shift(1))
    weekly = weekly.merge(
        allowed[
            [
                "opponent_team",
                "position_id",
                "GW",
                "fixture_raw",
                "opponent_goal_vulnerability",
                "opponent_assist_vulnerability",
                "opponent_xg_vulnerability",
            ]
        ],
        on=["opponent_team", "position_id", "GW"],
        how="left",
    )
    weekly["fixture_raw"] = weekly["fixture_raw"].fillna(
        weekly.groupby(["GW", "position_id"])["fixture_raw"].transform("median")
    )
    weekly["fixture_raw"] = weekly["fixture_raw"].fillna(2.5) + weekly[
        "was_home"
    ].fillna(False).astype(float) * 0.18
    for column, fallback in [
        ("opponent_goal_vulnerability", 0.10),
        ("opponent_assist_vulnerability", 0.10),
        ("opponent_xg_vulnerability", 0.10),
    ]:
        weekly[column] = weekly[column].fillna(
            weekly.groupby(["GW", "position_id"])[column].transform("median")
        ).fillna(fallback)

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


def add_causal_team_strength(data: pd.DataFrame) -> pd.DataFrame:
    """Add deadline-safe, time-decayed attack/defence and Poisson match rates."""
    team_columns = [
        "season",
        "season_order",
        "GW",
        "team_id",
        "team_name",
        "team_games",
        "team_goals",
        "team_xg",
        "team_goals_against",
        "team_xga",
        "team_clean_sheets",
        "team_result_points",
    ]
    team = data[team_columns].drop_duplicates(
        ["season", "GW", "team_id"], keep="first"
    ).copy()
    team.sort_values(["season_order", "GW", "team_id"], inplace=True)
    games = team["team_games"].clip(lower=1)
    goals_for = team["team_goals"] / games
    goals_against = team["team_goals_against"] / games
    xg_for = team["team_xg"] / games
    xg_against = team["team_xga"] / games
    team["attack_observation"] = np.where(
        team["team_xg"] > 0,
        0.72 * xg_for + 0.28 * goals_for,
        goals_for,
    )
    team["defence_observation"] = np.where(
        team["team_xga"] > 0,
        0.72 * xg_against + 0.28 * goals_against,
        goals_against,
    )
    team["form_observation"] = team["team_result_points"] / games
    team["clean_observation"] = team["team_clean_sheets"] / games

    league_week = (
        team.groupby(["season", "season_order", "GW"], as_index=False)
        .agg(league_goals=("team_goals", "sum"), league_games=("team_games", "sum"))
        .sort_values(["season_order", "GW"])
    )
    league_week["league_observation"] = (
        league_week["league_goals"] / league_week["league_games"].clip(lower=1)
    )
    league_week["league_goal_rate"] = league_week.groupby(
        "season", sort=False
    )["league_observation"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(1.40).clip(0.9, 2.0)
    team = team.merge(
        league_week[["season", "GW", "league_goal_rate"]],
        on=["season", "GW"],
        how="left",
    )
    normalized_name = (
        team["team_name"].fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    )
    team["team_key"] = np.where(
        team["team_name"].fillna("").str.startswith("Team "),
        team["season"].astype(str) + ":" + normalized_name,
        normalized_name,
    )
    by_team = team.groupby("team_key", sort=False)
    team["prior_team_games"] = by_team["team_games"].transform(
        lambda values: values.cumsum().shift(1)
    ).fillna(0)
    team_confidence = (
        team["prior_team_games"] / (team["prior_team_games"] + 8)
    ).clip(0, 0.94)

    def dynamic_rating(column: str, prior: pd.Series | float) -> pd.Series:
        rolling = team.groupby("team_key", sort=False)[column].transform(
            lambda values: values.ewm(alpha=0.22, adjust=False).mean().shift(1)
        )
        if isinstance(prior, pd.Series):
            fallback = prior
        else:
            fallback = pd.Series(float(prior), index=team.index)
        rolling = rolling.fillna(fallback)
        return team_confidence * rolling + (1 - team_confidence) * fallback

    team["team_attack_rating"] = dynamic_rating(
        "attack_observation", team["league_goal_rate"]
    ).clip(0.45, 2.70)
    team["team_defence_rating"] = dynamic_rating(
        "defence_observation", team["league_goal_rate"]
    ).clip(0.45, 2.70)
    team["team_form_rating"] = dynamic_rating("form_observation", 1.35).clip(0, 3)
    team["team_clean_rating"] = dynamic_rating("clean_observation", 0.28).clip(0, 0.75)
    fast_attack = team.groupby("team_key", sort=False)["attack_observation"].transform(
        lambda values: values.ewm(alpha=0.48, adjust=False).mean().shift(1)
    ).fillna(team["league_goal_rate"])
    fast_defence = team.groupby("team_key", sort=False)["defence_observation"].transform(
        lambda values: values.ewm(alpha=0.48, adjust=False).mean().shift(1)
    ).fillna(team["league_goal_rate"])
    team["team_regime_shift"] = (
        (
            (fast_attack - team["team_attack_rating"]).abs()
            + (fast_defence - team["team_defence_rating"]).abs()
        )
        / (2 * team["league_goal_rate"].clip(lower=0.8))
    ).clip(0, 0.75)
    regime_weight = (0.25 + team["team_regime_shift"]).clip(0.25, 0.75)
    team["team_attack_rating"] = (
        (1 - regime_weight) * team["team_attack_rating"] + regime_weight * fast_attack
    ).clip(0.45, 2.70)
    team["team_defence_rating"] = (
        (1 - regime_weight) * team["team_defence_rating"] + regime_weight * fast_defence
    ).clip(0.45, 2.70)
    team["team_rating_confidence"] = (
        team_confidence * (1 - 0.40 * team["team_regime_shift"])
    ).clip(0, 0.94)

    rating_columns = [
        "season",
        "GW",
        "team_id",
        "league_goal_rate",
        "team_attack_rating",
        "team_defence_rating",
        "team_form_rating",
        "team_clean_rating",
        "team_rating_confidence",
        "team_regime_shift",
    ]
    data = data.merge(team[rating_columns], on=["season", "GW", "team_id"], how="left")
    opponent = team[rating_columns].rename(
        columns={
            "team_id": "opponent_team",
            "team_attack_rating": "opponent_attack_rating",
            "team_defence_rating": "opponent_defence_rating",
            "team_form_rating": "opponent_form_rating",
            "team_clean_rating": "opponent_clean_rating",
            "team_rating_confidence": "opponent_rating_confidence",
            "team_regime_shift": "opponent_regime_shift",
            "league_goal_rate": "opponent_league_goal_rate",
        }
    )
    data = data.merge(
        opponent,
        on=["season", "GW", "opponent_team"],
        how="left",
    )
    league_rate = data["league_goal_rate"].fillna(1.40).clip(0.9, 2.0)
    for column in [
        "team_attack_rating",
        "team_defence_rating",
        "opponent_attack_rating",
        "opponent_defence_rating",
    ]:
        data[column] = data[column].fillna(league_rate)
    data["team_form_rating"] = data["team_form_rating"].fillna(1.35)
    data["opponent_form_rating"] = data["opponent_form_rating"].fillna(1.35)
    data["team_clean_rating"] = data["team_clean_rating"].fillna(0.28)
    data["team_rating_confidence"] = data["team_rating_confidence"].fillna(0)
    data["team_regime_shift"] = data["team_regime_shift"].fillna(0)

    home_ga_factor = np.where(data["was_home"].fillna(False), 0.88, 1.12)
    home_gf_factor = np.where(data["was_home"].fillna(False), 1.12, 0.88)
    data["team_expected_goals_against"] = (
        league_rate
        * (data["team_defence_rating"] / league_rate).pow(0.70)
        * (data["opponent_attack_rating"] / league_rate).pow(0.70)
        * home_ga_factor
    ).clip(0.30, 3.40)
    data["team_expected_goals_for"] = (
        league_rate
        * (data["team_attack_rating"] / league_rate).pow(0.70)
        * (data["opponent_defence_rating"] / league_rate).pow(0.70)
        * home_gf_factor
    ).clip(0.30, 3.40)
    data["team_clean_probability"] = np.exp(
        -data["team_expected_goals_against"]
    ).clip(0.03, 0.74)
    attack_index = data["team_attack_rating"] / league_rate
    defence_index = league_rate / data["team_defence_rating"].clip(lower=0.35)
    form_index = data["team_form_rating"] / 1.35
    fixture_defence_index = league_rate / data["team_expected_goals_against"]
    data["team_context_raw"] = (
        0.28 * attack_index
        + 0.32 * defence_index
        + 0.12 * form_index
        + 0.28 * fixture_defence_index
    ).clip(0.35, 2.75)
    data["team_defence_raw"] = fixture_defence_index.clip(0.30, 3.0)
    data["team_attack_raw"] = (
        data["team_expected_goals_for"] / league_rate
    ).clip(0.30, 3.0)
    return data


def prepare_causal_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Carry player priors across seasons and build component expected points."""
    data = pd.concat(frames, ignore_index=True)
    season_order = {season: index for index, season in enumerate(SEASONS)}
    data["season_order"] = data["season"].map(season_order).astype(int)
    data = add_causal_team_strength(data)
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

    def rolling_total(column: str, window: int = 10) -> pd.Series:
        return data.groupby("player_key", sort=False)[column].transform(
            lambda values: values.rolling(window, min_periods=1).sum().shift(1)
        )

    prior_start = data["position_id"].map({1: 0.68, 2: 0.58, 3: 0.56, 4: 0.54})
    prior_sub = data["position_id"].map({1: 0.05, 2: 0.30, 3: 0.42, 4: 0.43})
    prior_sixty_start = data["position_id"].map({1: 0.95, 2: 0.82, 3: 0.76, 4: 0.72})
    prior_strength = 4.0
    prior_games = rolling_total("fixture_count").fillna(0)
    prior_starts = rolling_total("starts_observed").fillna(0)
    prior_appearances = rolling_total("appearances_observed").fillna(0)
    prior_sixties = rolling_total("sixty_observed").fillna(0)
    prior_bench_appearances = rolling_total("bench_appearances_observed").fillna(0)
    prior_nonstarts = (prior_games - prior_starts).clip(lower=0)
    data["start_probability"] = (
        (prior_starts + prior_strength * prior_start)
        / (prior_games + prior_strength)
    ).clip(0.03, 0.98)
    data["sub_probability_given_bench"] = (
        (prior_bench_appearances + 3.0 * prior_sub)
        / (prior_nonstarts + 3.0)
    ).clip(0.02, 0.88)
    data["sixty_probability_given_start"] = (
        (prior_sixties + 4.0 * prior_sixty_start)
        / (prior_starts + 4.0)
    ).clip(0.25, 0.99)
    start_minutes_sum = rolling_total("start_minutes_total").fillna(0)
    bench_minutes_sum = rolling_total("bench_minutes_total").fillna(0)
    start_minutes_prior = data["position_id"].map({1: 88.0, 2: 80.0, 3: 76.0, 4: 73.0})
    bench_minutes_prior = data["position_id"].map({1: 5.0, 2: 16.0, 3: 20.0, 4: 22.0})
    data["minutes_if_start"] = (
        start_minutes_sum + 3.0 * start_minutes_prior
    ) / (prior_starts + 3.0)
    data["minutes_if_bench"] = (
        bench_minutes_sum + 3.0 * bench_minutes_prior
    ) / (prior_bench_appearances + 3.0)
    start_rate_observed = data["starts_observed"] / data["fixture_count"].clip(lower=1)
    data["rotation_volatility"] = data.groupby("player_key", sort=False)[
        "starts_observed"
    ].transform(
        lambda values: values.rolling(8, min_periods=2).std().shift(1)
    ).fillna(0.35).clip(0, 1)
    rest_penalty = (
        (4.5 - data["team_rest_days"].fillna(7)).clip(lower=0) / 5.0
        * (0.30 + 0.70 * data["rotation_volatility"])
    ).clip(0, 0.28)
    data["start_probability"] *= 1 - rest_penalty
    data["play_probability"] = (
        data["start_probability"]
        + (1 - data["start_probability"]) * data["sub_probability_given_bench"]
    ).clip(0.05, 0.995)
    data["sixty_probability"] = (
        data["start_probability"] * data["sixty_probability_given_start"]
    ).clip(0.02, 0.98)
    data["expected_minutes"] = (
        data["start_probability"] * data["minutes_if_start"]
        + (1 - data["start_probability"])
        * data["sub_probability_given_bench"]
        * data["minutes_if_bench"]
    ).clip(3, 90)
    second_moment = (
        data["start_probability"]
        * (data["minutes_if_start"].pow(2) + 12**2)
        + (1 - data["start_probability"])
        * data["sub_probability_given_bench"]
        * (data["minutes_if_bench"].pow(2) + 10**2)
    )
    data["minutes_std"] = np.sqrt(
        (second_moment - data["expected_minutes"].pow(2)).clip(lower=16)
    ).clip(4, 42)
    data["minutes_security_raw"] = (
        0.65 * data["sixty_probability"] + 0.35 * data["play_probability"]
    ).clip(0.05, 1.0)
    data["minutes_model_confidence"] = (
        prior_games / (prior_games + 8)
    ).clip(0, 0.95)
    data["start_observed_rate"] = start_rate_observed.clip(0, 1)
    data["sixty_observed_rate"] = (
        data["sixty_observed"] / data["fixture_count"].clip(lower=1)
    ).clip(0, 1)

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
    data["defensive_actions_game"] = (
        data["defensive_actions"] / data["fixture_count"].clip(lower=1)
    ).clip(0, 35)
    data["bps_game"] = (
        data["bps"] / data["fixture_count"].clip(lower=1)
    ).clip(-10, 80)
    for source, target, prior in [
        ("goal_signal_game", "goal_rate", {1: 0.01, 2: 0.04, 3: 0.20, 4: 0.28}),
        ("assist_signal_game", "assist_rate", {1: 0.01, 2: 0.08, 3: 0.18, 4: 0.13}),
        ("clean_sheet_game", "clean_sheet_rate", {1: 0.28, 2: 0.28, 3: 0.22, 4: 0.0}),
        ("saves", "save_rate", {1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}),
        ("bonus", "bonus_rate", {1: 0.18, 2: 0.22, 3: 0.28, 4: 0.28}),
        ("yellow_cards", "yellow_rate", {1: 0.05, 2: 0.12, 3: 0.10, 4: 0.08}),
        ("red_cards", "red_rate", {1: 0.005, 2: 0.008, 3: 0.006, 4: 0.005}),
        ("goals_conceded", "conceded_rate", {1: 1.35, 2: 1.35, 3: 0.0, 4: 0.0}),
        ("defensive_actions_game", "defensive_rate", {1: 0.0, 2: 6.8, 3: 6.0, 4: 3.0}),
        ("bps_game", "bps_rate", {1: 15.0, 2: 14.0, 3: 12.0, 4: 11.0}),
        ("penalties_saved", "penalty_save_rate", {1: 0.025, 2: 0.0, 3: 0.0, 4: 0.0}),
        ("penalties_missed", "penalty_miss_rate", {1: 0.0, 2: 0.002, 3: 0.01, 4: 0.015}),
        ("own_goals", "own_goal_rate", {1: 0.002, 2: 0.008, 3: 0.003, 4: 0.002}),
    ]:
        rolling = data.groupby("player_key", sort=False)[source].transform(
            lambda values: values.rolling(12, min_periods=1).mean().shift(1)
        )
        data[target] = rolling.fillna(data["position_id"].map(prior)).clip(lower=0)

    fixture_multiplier = 0.72 + 0.56 * data["fixture_now"].fillna(0.5)
    minutes_factor = data["expected_minutes"] / 90
    p_play = data["play_probability"]
    p_sixty = data["sixty_probability"]
    goal_points = data["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4})
    clean_sheet_points = data["position_id"].map({1: 4, 2: 4, 3: 1, 4: 0})
    appearance_points = p_play + p_sixty
    vulnerability_group = ["season", "GW", "position_id"]
    goal_vulnerability = (
        data["opponent_goal_vulnerability"]
        / data.groupby(vulnerability_group)["opponent_goal_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).clip(0.68, 1.42)
    assist_vulnerability = (
        data["opponent_assist_vulnerability"]
        / data.groupby(vulnerability_group)["opponent_assist_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).clip(0.72, 1.35)
    attacking_points = (
        data["goal_rate"] * goal_points * goal_vulnerability
        + data["assist_rate"] * 3 * assist_vulnerability
    ) * minutes_factor * fixture_multiplier
    blended_clean_probability = (
        0.82 * data["team_clean_probability"]
        + 0.18 * data["clean_sheet_rate"]
    ).clip(0.03, 0.78)
    clean_sheet_points_ev = (
        blended_clean_probability * clean_sheet_points * p_sixty
    )
    save_points = (
        data["save_rate"] / 3 * minutes_factor
        * np.where(data["position_id"] == 1, 1.0, 0.0)
    )
    defender_clean_bonus = (
        0.12
        * data["team_clean_probability"]
        * p_sixty
        * data["position_id"].isin([1, 2]).astype(float)
    )
    bps_rule_multiplier = np.select(
        [
            data["position_id"] == 1,
            (data["position_id"] == 2) & (data["defensive_rate"] >= 9),
            data["position_id"].isin([3, 4]),
        ],
        [1.06, 0.94, 1.03],
        default=1.0,
    )
    bonus_points = (
        data["bonus_rate"] * minutes_factor * fixture_multiplier * bps_rule_multiplier
        + defender_clean_bonus
    )
    discipline_points = -(
        data["yellow_rate"] + 3 * data["red_rate"]
    ) * minutes_factor
    rare_event_points = (
        5 * data["penalty_save_rate"]
        - 2 * data["penalty_miss_rate"]
        - 2 * data["own_goal_rate"]
    ) * minutes_factor
    conceded_points = -(
        data["team_expected_goals_against"] / 2
        * minutes_factor
        * data["position_id"].isin([1, 2]).astype(float)
    )
    defensive_threshold = np.where(data["position_id"] == 2, 10.0, 12.0)
    data["defensive_return_probability"] = poisson_tail(
        data["defensive_rate"] * minutes_factor,
        defensive_threshold,
    ) * data["position_id"].isin([2, 3, 4]).astype(float)
    defensive_points_current_rules = 2 * data["defensive_return_probability"]
    defensive_points_season_rules = defensive_points_current_rules * (
        data["season_order"] >= season_order.get("2025-26", 99)
    ).astype(float)
    structural_without_dc = (
        appearance_points
        + attacking_points
        + clean_sheet_points_ev
        + save_points
        + bonus_points
        + discipline_points
        + rare_event_points
        + conceded_points
    )
    data["component_xpts_structural"] = (
        structural_without_dc + defensive_points_season_rules
    ).clip(0.2, 13.0) * data["fixture_count"].clip(lower=1)
    data["component_xpts_current_rules"] = (
        structural_without_dc + defensive_points_current_rules
    ).clip(0.2, 13.5) * data["fixture_count"].clip(lower=1)

    data["empirical_xpts"] = (
        (0.62 * data["recent_raw"] + 0.38 * data["long_raw"])
        * (0.82 + 0.36 * data["fixture_now"].fillna(0.5))
        * (0.72 + 0.28 * data["play_probability"])
    ).clip(0.2, 13.5)
    position_base = data["position_id"].map({1: 3.2, 2: 2.8, 3: 3.0, 4: 2.8})
    data["market_role_xpts"] = (
        position_base
        * (0.64 + 0.46 * data["minutes_security_raw"])
        * (0.78 + 0.34 * data["fixture_now"].fillna(0.5))
        * (0.82 + 0.28 * data["team_context_raw"].clip(0.4, 1.8))
        * (
            0.94
            + 0.12
            * data.groupby(["season", "GW", "position_id"])["crowd_raw"]
            .transform(percentile)
        )
    ).clip(0.2, 13.5)
    ensemble_models = [
        "component_xpts_structural",
        "empirical_xpts",
        "market_role_xpts",
    ]
    error_keys = ["season_order", "GW", "position_id"]
    error_table = data[error_keys].drop_duplicates().sort_values(error_keys).copy()
    for model_name in ensemble_models:
        data[f"{model_name}_absolute_error"] = (
            data[model_name] - data["points"]
        ).abs()
        weekly_error = (
            data.groupby(error_keys, as_index=False)[f"{model_name}_absolute_error"]
            .mean()
            .sort_values(error_keys)
        )
        weekly_error[f"{model_name}_mae"] = weekly_error.groupby(
            "position_id", sort=False
        )[f"{model_name}_absolute_error"].transform(
            lambda values: values.expanding().mean().shift(1)
        )
        error_table = error_table.merge(
            weekly_error[error_keys + [f"{model_name}_mae"]],
            on=error_keys,
            how="left",
        )
    data = data.merge(error_table, on=error_keys, how="left")
    data = data.copy()
    default_mae = {
        "component_xpts_structural_mae": 2.85,
        "empirical_xpts_mae": 3.05,
        "market_role_xpts_mae": 3.25,
    }
    inverse_errors = []
    for column, fallback in default_mae.items():
        data[column] = data[column].fillna(fallback).clip(1.4, 6.0)
        inverse_errors.append(1 / data[column].pow(2))
    inverse_total = sum(inverse_errors)
    data["ensemble_structural_weight"] = inverse_errors[0] / inverse_total
    data["ensemble_empirical_weight"] = inverse_errors[1] / inverse_total
    data["ensemble_market_weight"] = inverse_errors[2] / inverse_total
    data["component_xpts"] = (
        data["ensemble_structural_weight"] * data["component_xpts_structural"]
        + data["ensemble_empirical_weight"] * data["empirical_xpts"]
        + data["ensemble_market_weight"] * data["market_role_xpts"]
    ).clip(0.2, 13.5)
    current_rule_uplift = (
        data["component_xpts_current_rules"] - data["component_xpts_structural"]
    )
    data["ensemble_xpts_current_rules"] = (
        data["component_xpts"] + current_rule_uplift
    ).clip(0.2, 14.0)
    model_stack = data[ensemble_models].to_numpy(float)
    data["ensemble_disagreement"] = np.std(model_stack, axis=1)
    horizon_multiplier = 0.74 + 0.52 * data["fixture"].fillna(0.5)
    single_fixture_base = data["component_xpts"] / data["fixture_count"].clip(lower=1)
    data["component_horizon"] = (
        single_fixture_base
        * data["horizon_weighted_games"].clip(lower=1)
        * horizon_multiplier
    ).clip(0.5, 50)
    current_rule_single_fixture = (
        data["ensemble_xpts_current_rules"] / data["fixture_count"].clip(lower=1)
    )
    data["component_horizon_current_rules"] = (
        current_rule_single_fixture
        * data["horizon_weighted_games"].clip(lower=1)
        * horizon_multiplier
    ).clip(0.5, 52)
    data["prediction_uncertainty"] = np.sqrt(
        1.1**2
        + 0.020 * data["minutes_std"].pow(2)
        + 0.85 * data["ensemble_disagreement"].pow(2)
        + 2.2 / np.sqrt(data["observations"] + 1)
    ).clip(1.2, 5.5)
    data["raw_blank_probability"] = normal_cdf(
        (2.5 - data["component_xpts"]) / data["prediction_uncertainty"]
    )
    data["raw_return5_probability"] = 1 - normal_cdf(
        (4.5 - data["component_xpts"]) / data["prediction_uncertainty"]
    )
    data["raw_haul8_probability"] = 1 - normal_cdf(
        (7.5 - data["component_xpts"]) / data["prediction_uncertainty"]
    )
    data["blank_probability"] = data["raw_blank_probability"]
    data["return5_probability"] = data["raw_return5_probability"]
    data["haul8_probability"] = data["raw_haul8_probability"]
    data = causal_calibrate_distributions(data)

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
        ("team_context_raw", "team_context"),
        ("team_defence_raw", "team_defence"),
        ("team_attack_raw", "team_attack"),
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
    team: float
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
                self.team,
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
            "team": self.team,
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
        [4.0, 1.2, 0.20, 1.7, 2.3, 0.35, 2.6, 2.0], size=TRIALS - 5
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
            Candidate(0.30, 0.06, 0.00, 0.13, 0.17, 0.03, 0.18, 0.13, 0.78),
            Candidate(0.36, 0.05, 0.00, 0.13, 0.18, 0.02, 0.17, 0.09, 0.82),
            Candidate(0.28, 0.07, 0.00, 0.12, 0.19, 0.03, 0.18, 0.13, 0.72),
            Candidate(0.38, 0.04, 0.00, 0.10, 0.18, 0.02, 0.17, 0.11, 0.76),
            # Lens 1.0: retained as a proper recursive baseline.
            Candidate(0.36, 0.09, 0.01, 0.04, 0.00, 0.50, 0.00, 0.00, 0.59),
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
            "team_context",
            "crowd",
            "minutes_security",
            "recent_underlying",
            "long_underlying",
        ]
    ].to_numpy(dtype=float)


def candidate_forecasts(
    data: pd.DataFrame,
    candidate: Candidate,
    current_rules: bool = False,
    robust_planning: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_score = feature_matrix(data) @ candidate.coefficients
    calibration = 0.72 + 0.56 * model_score
    current_column = "ensemble_xpts_current_rules" if current_rules else "component_xpts"
    horizon_column = (
        "component_horizon_current_rules" if current_rules else "component_horizon"
    )
    current = data[current_column].to_numpy(float) * calibration
    horizon = data[horizon_column].to_numpy(float) * calibration
    horizon_risk = (
        data["prediction_uncertainty"].to_numpy(float)
        * np.sqrt(data["horizon_weighted_games"].to_numpy(float).clip(1, None))
    )
    price_option = (
        data["price_rise_probability"].to_numpy(float)
        - data["price_fall_probability"].to_numpy(float)
    )
    upside = data["haul8_probability"].to_numpy(float)
    robust_plan = horizon - 0.10 * horizon_risk + 0.32 * price_option + 0.20 * upside
    return current, robust_plan if robust_planning else horizon, model_score


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
    actual_column: str = "points",
) -> tuple[np.ndarray, list[dict]]:
    """Carry one legal squad through each season and make deadline-only transfers."""
    if plan_scores is None:
        plan_scores = scores
    actual = data[actual_column].to_numpy(float)
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
        season_teams = set(season_data["team_id"].astype(int).unique())
        schedule_counts: dict[int, dict[int, int]] = {}
        for schedule_gw, schedule_frame in season_data.groupby("GW", sort=False):
            observed = (
                schedule_frame.groupby("team_id")["fixture_count"].max().astype(int).to_dict()
            )
            schedule_counts[int(schedule_gw)] = {
                team_id: int(observed.get(team_id, 0)) for team_id in season_teams
            }
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

                def schedule_opportunity(chip_name: str, target_gw: int) -> float:
                    counts = schedule_counts.get(target_gw, {})
                    blanks = sum(value == 0 for value in counts.values())
                    doubles = sum(value > 1 for value in counts.values())
                    largest_double = max(counts.values(), default=1) - 1
                    if chip_name == "Free Hit":
                        return 0.36 * blanks + 0.22 * doubles
                    if chip_name == "Bench Boost":
                        return 0.22 * doubles + 0.18 * largest_double
                    if chip_name == "Triple Captain":
                        return 0.28 * doubles + 0.22 * largest_double
                    return 0.0

                effective_thresholds: dict[int, float] = {}
                option_values: dict[int, float] = {}
                for window in available:
                    chip_name = str(window["chip"])
                    base_threshold = thresholds[chip_name]
                    remaining = max(0, int(window["end"]) - gw)
                    current_schedule_signal = schedule_opportunity(chip_name, gw)
                    future_signals = [
                        schedule_opportunity(chip_name, future_gw)
                        * (0.96 ** max(0, future_gw - gw))
                        for future_gw in weeks
                        if gw < future_gw <= int(window["end"])
                    ]
                    continuation_value = max(future_signals, default=0.0)
                    option_cost = 0.30 * max(
                        0.0, continuation_value - current_schedule_signal
                    )
                    expiry_relief = base_threshold * 0.22 * math.exp(-remaining / 2.3)
                    key = id(window)
                    option_values[key] = continuation_value
                    effective_thresholds[key] = max(
                        0.60 * base_threshold,
                        base_threshold + option_cost - expiry_relief,
                    )

                choices = [
                    window
                    for window in available
                    if metrics[str(window["chip"])]
                    >= effective_thresholds[id(window)]
                    and has_structural_signal(str(window["chip"]))
                ]
                chosen_window = max(
                    choices,
                    key=lambda window: metrics[str(window["chip"])]
                    / max(0.01, effective_thresholds[id(window)]),
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
                            "threshold": round(float(effective_thresholds[id(chosen_window)]), 3),
                            "continuationValue": round(float(option_values[id(chosen_window)]), 3),
                            "reason": "signal beat option-value adjusted threshold",
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


def build_calibration_diagnostics(data: pd.DataFrame, backtest: list[dict]) -> dict:
    evaluation = data[data["season"].isin(EVALUATION_SEASONS)].copy()
    actual_return = (evaluation["points"] >= 5).astype(float)
    return_probability = evaluation["return5_probability"].clip(0, 1)
    return_brier = float(np.mean((return_probability - actual_return) ** 2))
    minute_brier = float(
        np.mean(
            (
                evaluation["sixty_probability"].clip(0, 1)
                - evaluation["sixty_observed_rate"].clip(0, 1)
            )
            ** 2
        )
    )
    clean_rows = evaluation[evaluation["position_id"].isin([1, 2])]
    clean_actual = (
        clean_rows["clean_sheets"] / clean_rows["fixture_count"].clip(lower=1)
    ).clip(0, 1)
    clean_brier = float(
        np.mean((clean_rows["team_clean_probability"].clip(0, 1) - clean_actual) ** 2)
    )
    interval_coverage = float(
        (
            (evaluation["points"] >= evaluation["prediction_p10"])
            & (evaluation["points"] <= evaluation["prediction_p90"])
        ).mean()
    )
    calibration_bins: list[dict] = []
    for lower in np.arange(0, 1, 0.1):
        upper = lower + 0.1
        mask = (return_probability >= lower) & (
            return_probability < upper if upper < 1 else return_probability <= upper
        )
        if not mask.any():
            continue
        calibration_bins.append(
            {
                "forecast": round(100 * float(return_probability[mask].mean())),
                "observed": round(100 * float(actual_return[mask].mean())),
                "players": int(mask.sum()),
            }
        )
    position_errors = []
    for position_id, frame in evaluation.groupby("position_id"):
        position_errors.append(
            {
                "position": POSITION_LABELS[int(position_id)],
                "mae": round(float((frame["component_xpts"] - frame["points"]).abs().mean()), 2),
                "returnBrier": round(
                    float(
                        np.mean(
                            (
                                frame["return5_probability"]
                                - (frame["points"] >= 5).astype(float)
                            )
                            ** 2
                        )
                    ),
                    3,
                ),
                "rows": int(len(frame)),
            }
        )
    weakest = sorted(backtest, key=lambda item: item["targetMargin"])[:3]
    return {
        "returnBrier": round(return_brier, 3),
        "minutes60Brier": round(minute_brier, 3),
        "cleanSheetBrier": round(clean_brier, 3),
        "p10P90Coverage": round(100 * interval_coverage),
        "mae": round(float((evaluation["component_xpts"] - evaluation["points"]).abs().mean()), 2),
        "defensiveEventCoverage": round(
            100
            * float(
                evaluation.loc[
                    evaluation["position_id"].isin([2, 3, 4]), "defensive_exact"
                ].mean()
            )
        ),
        "returnCalibration": calibration_bins,
        "positionErrors": position_errors,
        "weakSeasons": [
            {
                "season": item["season"],
                "margin": item["targetMargin"],
                "points": item["points"],
                "diagnosis": (
                    "high-variance season: minutes and captain outcomes dominated"
                    if item["targetMargin"] <= -180
                    else "below target: transfer timing and player-return calibration"
                ),
            }
            for item in weakest
        ],
        "method": "Causal probability calibration on evaluation player-weeks; no future match enters a forecast bin.",
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
    historical: pd.DataFrame, best: Candidate, robust_planning: bool
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
    team_full_name = dict(zip(teams["id"], teams["name"]))

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
            start_probability=("start_probability", "last"),
            sub_probability_given_bench=("sub_probability_given_bench", "last"),
            sixty_probability_given_start=("sixty_probability_given_start", "last"),
            minutes_if_start=("minutes_if_start", "last"),
            minutes_if_bench=("minutes_if_bench", "last"),
            minutes_std_prior=("minutes_std", "last"),
            rotation_volatility=("rotation_volatility", "last"),
            defensive_rate_prior=("defensive_rate", "last"),
            bps_rate_prior=("bps_rate", "last"),
            goal_rate_prior=("goal_rate", "last"),
            assist_rate_prior=("assist_rate", "last"),
            bonus_rate_prior=("bonus_rate", "last"),
            save_rate_prior=("save_rate", "last"),
            clean_sheet_rate_prior=("clean_sheet_rate", "last"),
            ensemble_structural_weight=("ensemble_structural_weight", "last"),
            ensemble_empirical_weight=("ensemble_empirical_weight", "last"),
            ensemble_market_weight=("ensemble_market_weight", "last"),
        )
    )
    prior_summary = prior_summary.merge(tails, on="player_code", how="left")

    first_fixtures = pd.DataFrame(fixtures)
    first_fixtures = first_fixtures[first_fixtures["event"] == gw_number]
    fixture_map: dict[int, dict] = {}
    for _, fixture in first_fixtures.iterrows():
        fixture_map[int(fixture["team_h"])] = {
            "fixture_id": int(fixture["id"]),
            "opponent": int(fixture["team_a"]),
            "home": True,
            "kickoff": fixture["kickoff_time"],
        }
        fixture_map[int(fixture["team_a"])] = {
            "fixture_id": int(fixture["id"]),
            "opponent": int(fixture["team_h"]),
            "home": False,
            "kickoff": fixture["kickoff_time"],
        }

    played_history = historical[
        (historical["minutes"] > 0) & historical["player_code"].notna()
    ].copy()
    general_history = (
        played_history.groupby("player_code", as_index=False)
        .agg(
            history_matches=("points", "size"),
            history_points=("points", "sum"),
            history_minutes=("minutes", "sum"),
            history_average=("points", "mean"),
            history_volatility=("points", "std"),
            history_returns=("points", lambda values: float((values >= 5).mean())),
        )
    )
    general_history["history_per90"] = (
        90 * general_history["history_points"]
        / general_history["history_minutes"].clip(lower=1)
    )
    opponent_history = (
        played_history.dropna(subset=["opponent_name"])
        .groupby(["player_code", "opponent_name"], as_index=False)
        .agg(
            opponent_matches=("points", "size"),
            opponent_points=("points", "sum"),
            opponent_minutes=("minutes", "sum"),
            opponent_average=("points", "mean"),
            opponent_returns=("points", lambda values: float((values >= 5).mean())),
        )
    )
    opponent_history["opponent_per90"] = (
        90 * opponent_history["opponent_points"]
        / opponent_history["opponent_minutes"].clip(lower=1)
    )
    opponent_profiles = (
        historical.sort_values(["season_order", "GW"])
        .dropna(subset=["opponent_name"])
        .groupby(["opponent_name", "position_id"], as_index=False)
        .tail(1)[
            [
                "opponent_name",
                "position_id",
                "opponent_goal_vulnerability",
                "opponent_assist_vulnerability",
                "opponent_xg_vulnerability",
            ]
        ]
    )
    opponent_profiles = opponent_profiles.rename(
        columns={"opponent_name": "opponent_full_name"}
    )
    team_profiles = historical[
        [
            "season_order",
            "GW",
            "team_name",
            "team_attack_rating",
            "team_defence_rating",
            "team_form_rating",
            "team_clean_rating",
            "team_rating_confidence",
            "team_regime_shift",
        ]
    ].drop_duplicates(["season_order", "GW", "team_name"]).copy()
    team_profiles["team_key"] = (
        team_profiles["team_name"].fillna("").str.lower().str.replace(
            r"[^a-z0-9]", "", regex=True
        )
    )
    team_profiles.sort_values(["season_order", "GW"], inplace=True)
    team_profiles = team_profiles.groupby("team_key", as_index=False).tail(1)

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
                "start_probability",
                "sub_probability_given_bench",
                "sixty_probability_given_start",
                "minutes_if_start",
                "minutes_if_bench",
                "minutes_std_prior",
                "rotation_volatility",
                "defensive_rate_prior",
                "bps_rate_prior",
                "goal_rate_prior",
                "assist_rate_prior",
                "bonus_rate_prior",
                "save_rate_prior",
                "clean_sheet_rate_prior",
                "ensemble_structural_weight",
                "ensemble_empirical_weight",
                "ensemble_market_weight",
            ]
        ],
        left_on="code",
        right_on="player_code",
        how="left",
    )
    current["position_id"] = current["element_type"].astype(int)
    current["team_id"] = current["team"].astype(int)
    current["team_name"] = current["team_id"].map(team_name)
    current["team_full_name"] = current["team_id"].map(team_full_name)
    current["team_key"] = (
        current["team_full_name"].fillna("").str.lower().str.replace(
            r"[^a-z0-9]", "", regex=True
        )
    )
    current = current.merge(
        team_profiles[
            [
                "team_key",
                "team_attack_rating",
                "team_defence_rating",
                "team_form_rating",
                "team_clean_rating",
                "team_rating_confidence",
                "team_regime_shift",
            ]
        ],
        on="team_key",
        how="left",
    )
    current["opponent_full_name"] = current["team_id"].map(
        lambda team_id: team_full_name.get(
            fixture_map.get(int(team_id), {}).get("opponent"), "TBD"
        )
    )
    current = current.merge(general_history, on="player_code", how="left")
    current = current.merge(
        opponent_history,
        left_on=["player_code", "opponent_full_name"],
        right_on=["player_code", "opponent_name"],
        how="left",
    )
    current = current.merge(
        opponent_profiles,
        on=["opponent_full_name", "position_id"],
        how="left",
    )
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
    league_goal_rate = 1.40
    current["team_attack_rating"] = current["team_attack_rating"].fillna(
        league_goal_rate
    )
    current["team_defence_rating"] = current["team_defence_rating"].fillna(
        league_goal_rate
    )
    current["team_form_rating"] = current["team_form_rating"].fillna(1.35)
    current["team_clean_rating"] = current["team_clean_rating"].fillna(0.28)
    current["team_rating_confidence"] = current["team_rating_confidence"].fillna(0)
    current["team_regime_shift"] = current["team_regime_shift"].fillna(0)
    team_snapshot = current[
        [
            "team_id",
            "team_attack_rating",
            "team_defence_rating",
            "team_form_rating",
            "team_clean_rating",
            "team_rating_confidence",
        ]
    ].drop_duplicates("team_id").set_index("team_id")

    def match_rates(team_id: int, opponent_id: int, home: bool) -> tuple[float, float, float]:
        team_row = team_snapshot.loc[int(team_id)]
        opponent_row = team_snapshot.loc[int(opponent_id)]
        expected_against = float(
            league_goal_rate
            * (float(team_row["team_defence_rating"]) / league_goal_rate) ** 0.70
            * (float(opponent_row["team_attack_rating"]) / league_goal_rate) ** 0.70
            * (0.88 if home else 1.12)
        )
        expected_for = float(
            league_goal_rate
            * (float(team_row["team_attack_rating"]) / league_goal_rate) ** 0.70
            * (float(opponent_row["team_defence_rating"]) / league_goal_rate) ** 0.70
            * (1.12 if home else 0.88)
        )
        expected_against = float(np.clip(expected_against, 0.30, 3.40))
        expected_for = float(np.clip(expected_for, 0.30, 3.40))
        return expected_for, expected_against, float(np.exp(-expected_against))

    immediate_rates: dict[int, tuple[float, float, float]] = {}
    for team_id, fixture in fixture_map.items():
        immediate_rates[int(team_id)] = match_rates(
            int(team_id), int(fixture["opponent"]), bool(fixture["home"])
        )
    current["team_expected_goals_for"] = current["team_id"].map(
        lambda team_id: immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))[0]
    )
    current["team_expected_goals_against"] = current["team_id"].map(
        lambda team_id: immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))[1]
    )
    current["team_clean_probability"] = current["team_id"].map(
        lambda team_id: immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))[2]
    )

    horizon_rates: dict[int, tuple[float, float, float]] = {}
    for team_id, values in horizon_map.items():
        weighted_for = 0.0
        weighted_against = 0.0
        weighted_clean = 0.0
        total_weight = 0.0
        team_fixtures = horizon_fixtures[
            (horizon_fixtures["team_h"] == int(team_id))
            | (horizon_fixtures["team_a"] == int(team_id))
        ]
        for _, fixture in team_fixtures.iterrows():
            home = int(fixture["team_h"]) == int(team_id)
            opponent_id = int(fixture["team_a"] if home else fixture["team_h"])
            weight = horizon_weight[int(fixture["event"])]
            expected_for, expected_against, clean_probability = match_rates(
                int(team_id), opponent_id, home
            )
            weighted_for += weight * expected_for
            weighted_against += weight * expected_against
            weighted_clean += weight * clean_probability
            total_weight += weight
        if total_weight > 0:
            horizon_rates[int(team_id)] = (
                weighted_for / total_weight,
                weighted_against / total_weight,
                weighted_clean / total_weight,
            )
    current["team_horizon_expected_goals_for"] = current["team_id"].map(
        lambda team_id: horizon_rates.get(
            int(team_id), immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))
        )[0]
    )
    current["team_horizon_expected_goals_against"] = current["team_id"].map(
        lambda team_id: horizon_rates.get(
            int(team_id), immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))
        )[1]
    )
    current["team_horizon_clean_probability"] = current["team_id"].map(
        lambda team_id: horizon_rates.get(
            int(team_id), immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))
        )[2]
    )
    current["team_context_raw"] = (
        0.28 * current["team_attack_rating"] / league_goal_rate
        + 0.32 * league_goal_rate / current["team_defence_rating"].clip(lower=0.35)
        + 0.12 * current["team_form_rating"] / 1.35
        + 0.28 * league_goal_rate / current["team_expected_goals_against"].clip(lower=0.30)
    ).clip(0.35, 2.75)
    current["team_defence_raw"] = (
        league_goal_rate / current["team_expected_goals_against"].clip(lower=0.30)
    ).clip(0.30, 3.0)
    current["team_attack_raw"] = (
        current["team_expected_goals_for"] / league_goal_rate
    ).clip(0.30, 3.0)
    team_match_context = (
        current[
            [
                "team_name",
                "team_expected_goals_for",
                "team_expected_goals_against",
            ]
        ]
        .drop_duplicates("team_name")
        .copy()
    )
    team_match_context["team_attack_rank"] = team_match_context[
        "team_expected_goals_for"
    ].rank(method="min", ascending=False)
    team_match_context["team_defence_rank"] = team_match_context[
        "team_expected_goals_against"
    ].rank(method="min", ascending=True)
    team_match_context["team_strength_rank"] = (
        team_match_context["team_expected_goals_for"]
        / team_match_context["team_expected_goals_against"].clip(lower=0.25)
    ).rank(method="min", ascending=False)
    current = current.merge(
        team_match_context[
            ["team_name", "team_attack_rank", "team_defence_rank", "team_strength_rank"]
        ],
        on="team_name",
        how="left",
    )
    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_raw", "fixture_now"),
        ("team_context_raw", "team_context"),
        ("team_defence_raw", "team_defence"),
        ("team_attack_raw", "team_attack"),
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
    current["sample_nineties"] = nineties
    rate_denominator = nineties + 5.0

    def numeric_current(column: str) -> pd.Series:
        if column not in current:
            return pd.Series(0.0, index=current.index)
        return pd.to_numeric(current[column], errors="coerce").fillna(0.0)

    current = current.copy()

    prior_start = current["start_probability"].fillna(
        current["position_id"].map({1: 0.68, 2: 0.58, 3: 0.56, 4: 0.54})
    )
    prior_sub = current["sub_probability_given_bench"].fillna(
        current["position_id"].map({1: 0.05, 2: 0.30, 3: 0.42, 4: 0.43})
    )
    prior_sixty_start = current["sixty_probability_given_start"].fillna(
        current["position_id"].map({1: 0.95, 2: 0.82, 3: 0.76, 4: 0.72})
    )
    completed_rounds = max(0, gw_number - 1)
    season_starts = numeric_current("starts").clip(0, completed_rounds)
    current["start_probability"] = (
        6 * prior_start + season_starts
    ) / (6 + completed_rounds)
    current["start_probability"] *= (
        current["availability"] / 100
    ).clip(0, 1)
    current["sub_probability_given_bench"] = prior_sub
    current["play_probability"] = (
        current["start_probability"]
        + (1 - current["start_probability"]) * prior_sub
    ).clip(0.02, 0.995)
    current["sixty_probability"] = (
        current["start_probability"] * prior_sixty_start
    ).clip(0.01, 0.99)
    minutes_if_start = current["minutes_if_start"].fillna(
        current["position_id"].map({1: 88.0, 2: 80.0, 3: 76.0, 4: 73.0})
    )
    minutes_if_bench = current["minutes_if_bench"].fillna(
        current["position_id"].map({1: 5.0, 2: 16.0, 3: 20.0, 4: 22.0})
    )
    current["minutes_if_start_forecast"] = minutes_if_start
    current["minutes_if_bench_forecast"] = minutes_if_bench
    expected_minutes = (
        current["start_probability"] * minutes_if_start
        + (1 - current["start_probability"])
        * current["sub_probability_given_bench"]
        * minutes_if_bench
    ).clip(1, 90)
    second_moment = (
        current["start_probability"] * (minutes_if_start.pow(2) + 12**2)
        + (1 - current["start_probability"])
        * current["sub_probability_given_bench"]
        * (minutes_if_bench.pow(2) + 10**2)
    )
    current["minutes_std"] = np.sqrt(
        (second_moment - expected_minutes.pow(2)).clip(lower=16)
    ).clip(4, 42)
    current["minutes_security_raw"] = (
        0.65 * current["sixty_probability"] + 0.35 * current["play_probability"]
    )
    appearance_share = expected_minutes / 90
    sixty_probability = current["sixty_probability"]
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
    team_attack_multiplier = (
        current["team_expected_goals_for"] / league_goal_rate
    ).pow(0.45).clip(0.70, 1.38)
    goal_vulnerability = (
        current["opponent_goal_vulnerability"]
        / current.groupby("position_id")["opponent_goal_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).fillna(1).clip(0.68, 1.42)
    assist_vulnerability = (
        current["opponent_assist_vulnerability"]
        / current.groupby("position_id")["opponent_assist_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).fillna(1).clip(0.72, 1.35)
    goal_rate_live = (
        0.72 * expected_goals
        + 0.28 * goals
        + 5 * current["goal_rate_prior"].fillna(
            current["position_id"].map({1: 0.01, 2: 0.04, 3: 0.20, 4: 0.28})
        )
    ) / rate_denominator
    assist_rate_live = (
        0.72 * expected_assists
        + 0.28 * assists
        + 5 * current["assist_rate_prior"].fillna(
            current["position_id"].map({1: 0.01, 2: 0.08, 3: 0.18, 4: 0.13})
        )
    ) / rate_denominator
    penalties_order = numeric_current("penalties_order")
    free_kick_order = numeric_current("direct_freekicks_order")
    corner_order = numeric_current("corners_and_indirect_freekicks_order")
    penalty_role_probability = np.select(
        [penalties_order == 1, penalties_order == 2], [0.86, 0.12], default=0.0
    )
    set_piece_goal_rate = (
        0.075 * current["team_expected_goals_for"] * penalty_role_probability
        + 0.018 * (free_kick_order == 1).astype(float)
    )
    set_piece_assist_rate = (
        0.025 * (corner_order == 1).astype(float)
        + 0.010 * (corner_order == 2).astype(float)
    )
    appearance_component = 1.0 + appearance_share
    goal_component = (
        (goal_rate_live + set_piece_goal_rate)
        * appearance_share
        * goal_points
        * team_attack_multiplier
        * goal_vulnerability
    )
    assist_component = (
        (assist_rate_live + set_piece_assist_rate)
        * appearance_share
        * 3
        * team_attack_multiplier
        * assist_vulnerability
    )
    personal_clean_probability = (
        (
            clean_sheets
            + 5
            * current["clean_sheet_rate_prior"].fillna(
                current["position_id"].map({1: 0.28, 2: 0.28, 3: 0.22, 4: 0.0})
            )
        )
        / rate_denominator
    ).clip(0, 0.75)
    blended_clean_probability = (
        0.82 * current["team_clean_probability"]
        + 0.18 * personal_clean_probability
    ).clip(0.03, 0.78)
    clean_component = (
        blended_clean_probability * clean_points * sixty_probability
    )
    save_rate_live = (
        saves
        + 5
        * current["save_rate_prior"].fillna(
            current["position_id"].map({1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0})
        )
    ) / rate_denominator
    save_component = (save_rate_live / 3) * appearance_share
    bps_rule_multiplier = np.select(
        [
            current["position_id"] == 1,
            (current["position_id"] == 2)
            & (current["defensive_rate_prior"].fillna(0) >= 9),
            current["position_id"].isin([3, 4]),
        ],
        [1.06, 0.94, 1.03],
        default=1.0,
    )
    bonus_rate_live = (
        bonus
        + 5
        * current["bonus_rate_prior"].fillna(
            current["position_id"].map({1: 0.18, 2: 0.22, 3: 0.28, 4: 0.28})
        )
    ) / rate_denominator
    bonus_component = (
        bonus_rate_live * appearance_share * bps_rule_multiplier
        + 0.12
        * current["team_clean_probability"]
        * sixty_probability
        * current["position_id"].isin([1, 2]).astype(float)
    )
    defensive_rate_live = (
        defensive_points
        + 5
        * current["defensive_rate_prior"].fillna(
            current["position_id"].map({1: 0.0, 2: 6.8, 3: 6.0, 4: 3.0})
        )
    ) / rate_denominator
    current["defensive_return_probability"] = poisson_tail(
        defensive_rate_live * appearance_share,
        defensive_threshold,
    ) * current["position_id"].isin([2, 3, 4]).astype(float)
    defensive_component = 2 * current["defensive_return_probability"]
    discipline_component = -(
        (yellow_cards + 3 * red_cards) / rate_denominator * appearance_share
    )
    conceded_component = -pd.Series(
        np.where(
            current["position_id"].isin([1, 2]),
            current["team_expected_goals_against"] / 2 * appearance_share,
            0,
        ),
        index=current.index,
    )
    component_projection = (
        appearance_component
        + goal_component
        + assist_component
        + clean_component
        + save_component
        + bonus_component
        + defensive_component
        + discipline_component
        + conceded_component
    )
    own_projection = component_projection * (
        0.82 + current["fixture_now"] * 0.28
    ) * (0.74 + current["model_score"] * 0.42)
    empirical_projection = (
        (0.62 * current["recent_raw"] + 0.38 * current["long_raw"])
        * (0.82 + 0.36 * current["fixture_now"])
        * (0.72 + 0.28 * current["play_probability"])
    ).clip(0.3, 13.5)
    position_base = current["position_id"].map({1: 3.2, 2: 2.8, 3: 3.0, 4: 2.8})
    market_projection = (
        position_base
        * (0.64 + 0.46 * current["minutes_security_raw"])
        * (0.78 + 0.34 * current["fixture_now"])
        * (0.82 + 0.28 * current["team_context_raw"].clip(0.4, 1.8))
        * (0.94 + 0.12 * current["crowd_raw"].rank(pct=True))
    ).clip(0.3, 13.5)
    structural_weight = current["ensemble_structural_weight"].fillna(0.40)
    empirical_weight = current["ensemble_empirical_weight"].fillna(0.34)
    market_weight = current["ensemble_market_weight"].fillna(0.26)
    weight_total = structural_weight + empirical_weight + market_weight
    structural_weight /= weight_total
    empirical_weight /= weight_total
    market_weight /= weight_total
    public_weight = pd.Series(
        np.where(current["ep_next_num"] > 0, 0.18, 0.0), index=current.index
    )
    internal_weight = 1 - public_weight
    current["raw_projection"] = (
        internal_weight
        * (
            structural_weight * own_projection
            + empirical_weight * empirical_projection
            + market_weight * market_projection
        )
        + public_weight * current["ep_next_num"]
    ).clip(0.4, 13.8)
    ensemble_stack = np.vstack(
        [
            own_projection.to_numpy(float),
            empirical_projection.to_numpy(float),
            market_projection.to_numpy(float),
            current["ep_next_num"].where(
                current["ep_next_num"] > 0, own_projection
            ).to_numpy(float),
        ]
    ).T
    current["ensemble_disagreement"] = np.std(ensemble_stack, axis=1)
    current["ensemble_structural_weight_live"] = structural_weight * internal_weight
    current["ensemble_empirical_weight_live"] = empirical_weight * internal_weight
    current["ensemble_market_weight_live"] = market_weight * internal_weight
    current["ensemble_public_weight_live"] = public_weight
    component_scale = current["raw_projection"] / component_projection.clip(lower=0.25)
    current["component_appearance"] = appearance_component * component_scale
    current["component_goals"] = goal_component * component_scale
    current["component_assists"] = assist_component * component_scale
    current["component_clean"] = clean_component * component_scale
    current["component_defence"] = (
        save_component + defensive_component
    ) * component_scale
    current["component_bonus"] = bonus_component * component_scale
    current["component_adjustment"] = current["raw_projection"] - (
        current["component_appearance"]
        + current["component_goals"]
        + current["component_assists"]
        + current["component_clean"]
        + current["component_defence"]
        + current["component_bonus"]
    )
    weighted_games = current["team_id"].map(
        lambda team_id: sum(weight for _, weight in horizon_map.get(int(team_id), []))
    ).clip(lower=1.0)
    horizon_attack_ratio = (
        (current["team_horizon_expected_goals_for"] + 0.40)
        / (current["team_expected_goals_for"] + 0.40)
    ).clip(0.70, 1.40)
    horizon_clean_ratio = (
        (current["team_horizon_clean_probability"] + 0.08)
        / (current["team_clean_probability"] + 0.08)
    ).clip(0.65, 1.50)
    team_horizon_multiplier = pd.Series(
        np.select(
            [
                current["position_id"].isin([1, 2]),
                current["position_id"] == 3,
            ],
            [
                0.65 * horizon_clean_ratio + 0.35 * horizon_attack_ratio,
                0.18 * horizon_clean_ratio + 0.82 * horizon_attack_ratio,
            ],
            default=horizon_attack_ratio,
        ),
        index=current.index,
    ).clip(0.72, 1.35)
    current["horizon_projection"] = (
        current["raw_projection"]
        * weighted_games
        * team_horizon_multiplier
    )
    current["expected_minutes"] = expected_minutes
    official_disagreement = (
        (own_projection - current["ep_next_num"]).abs()
        / current["raw_projection"].clip(lower=1)
    ).clip(0, 1)
    current["projection_std"] = np.sqrt(
        1.05**2
        + 0.020 * current["minutes_std"].pow(2)
        + 0.90 * current["ensemble_disagreement"].pow(2)
        + 1.8 / np.sqrt(nineties + 1)
    ).clip(1.15, 5.8)
    current["uncertainty"] = (
        current["projection_std"]
        / (current["raw_projection"] + current["projection_std"]).clip(lower=1)
        + 0.08 * official_disagreement
        + 0.10 * (1 - current["availability"] / 100).clip(0, 1)
    ).clip(0.05, 1.0)
    current["confidence"] = (100 * (1 - current["uncertainty"])).clip(0, 95)
    current["raw_blank_probability"] = normal_cdf(
        (2.5 - current["raw_projection"]) / current["projection_std"]
    )
    current["raw_return5_probability"] = 1 - normal_cdf(
        (4.5 - current["raw_projection"]) / current["projection_std"]
    )
    current["raw_haul8_probability"] = 1 - normal_cdf(
        (7.5 - current["raw_projection"]) / current["projection_std"]
    )
    current["blank_probability"] = current["raw_blank_probability"]
    current["return5_probability"] = current["raw_return5_probability"]
    current["haul8_probability"] = current["raw_haul8_probability"]
    current = calibrate_live_distributions(current, historical)
    transfer_pressure = (
        numeric_current("transfers_in_event")
        - numeric_current("transfers_out_event")
    ) / numeric_current("selected").clip(lower=2500)
    transfer_pressure_rank = transfer_pressure.rank(pct=True)
    current["price_rise_probability"] = sigmoid(
        11 * (transfer_pressure_rank - 0.72)
    )
    current["price_fall_probability"] = sigmoid(
        11 * (0.28 - transfer_pressure_rank)
    )
    current["risk_adjusted_projection"] = (
        current["raw_projection"] - 0.10 * current["projection_std"]
    ).clip(lower=0.2)
    robust_horizon = (
        current["horizon_projection"]
        - 0.10 * current["projection_std"] * np.sqrt(weighted_games)
        + 0.32
        * (current["price_rise_probability"] - current["price_fall_probability"])
        + 0.20 * current["haul8_probability"]
    ).clip(lower=0.2)
    current["risk_adjusted_horizon"] = (
        robust_horizon if robust_planning else current["horizon_projection"]
    )
    current["value_projection"] = (
        current["risk_adjusted_horizon"] / (current["price"] / 10).clip(lower=3.5)
    )
    current["model_score"] = (
        0.42 * current["model_score"].rank(pct=True)
        + 0.38 * current["risk_adjusted_horizon"].rank(pct=True)
        + 0.20 * current["risk_adjusted_projection"].rank(pct=True)
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
    pool["fixture_id"] = pool["team_id"].map(
        lambda team_id: fixture_map.get(int(team_id), {}).get("fixture_id", -1)
    )
    pool["position_rank"] = pool.groupby("position_id")[
        "risk_adjusted_horizon"
    ].rank(method="min", ascending=False)
    pool["position_count"] = pool.groupby("position_id")["id"].transform("size")
    pool["projection_percentile"] = pool.groupby("position_id")[
        "risk_adjusted_horizon"
    ].rank(pct=True)
    pool["balanced_utility"] = pool["model_score"]
    pool["protect_utility"] = (
        0.40 * pool["prediction_p10"].rank(pct=True)
        + 0.30 * pool["risk_adjusted_horizon"].rank(pct=True)
        + 0.20 * pool["sixty_probability"].rank(pct=True)
        + 0.10 * pool["confidence"].rank(pct=True)
    )
    pool["chase_utility"] = (
        0.38 * pool["prediction_p90"].rank(pct=True)
        + 0.30 * pool["haul8_probability"].rank(pct=True)
        + 0.22 * pool["horizon_projection"].rank(pct=True)
        + 0.10 * pool["team_attack"].rank(pct=True)
    )
    chosen, xi = pick_squad(pool)
    strategy_profiles: list[dict] = []
    for profile_name, utility_column in [
        ("Protect", "protect_utility"),
        ("Balanced", "balanced_utility"),
        ("Chase", "chase_utility"),
    ]:
        profile_pool = pool.copy()
        profile_pool["model_score"] = profile_pool[utility_column]
        profile_chosen, profile_xi = pick_squad(profile_pool)
        strategy_profiles.append(
            {
                "name": profile_name,
                "squadIds": profile_pool.loc[profile_chosen, "id"].astype(int).tolist(),
                "expectedXI": round(float(profile_pool.loc[profile_xi, "raw_projection"].sum()), 1),
                "downsideXI": round(float(profile_pool.loc[profile_xi, "prediction_p10"].sum()), 1),
                "upsideXI": round(float(profile_pool.loc[profile_xi, "prediction_p90"].sum()), 1),
                "spend": round(float(profile_pool.loc[profile_chosen, "price"].sum()) / 10, 1),
            }
        )
    xi_set = set(xi)
    pool["captain_score"] = (
        0.56 * pool["risk_adjusted_projection"].rank(pct=True)
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
    scenario_rng = np.random.default_rng(20260813)
    scenario_count = 5000
    xi_frame = pool.loc[xi]
    independent = scenario_rng.normal(size=(scenario_count, len(xi)))
    scenario_values = np.zeros_like(independent)
    for column_index, (_, player_row) in enumerate(xi_frame.iterrows()):
        same_team_columns = [
            local_index
            for local_index, (_, peer_row) in enumerate(xi_frame.iterrows())
            if int(peer_row["team_id"]) == int(player_row["team_id"])
        ]
        if column_index == same_team_columns[0]:
            common_shock = scenario_rng.normal(size=scenario_count)
            for same_team_index in same_team_columns:
                peer = xi_frame.iloc[same_team_index]
                scenario_values[:, same_team_index] = np.clip(
                    float(peer["raw_projection"])
                    + float(peer["projection_std"])
                    * (
                        math.sqrt(0.78) * independent[:, same_team_index]
                        + math.sqrt(0.22) * common_shock
                    ),
                    0,
                    25,
                )
    captain_column = xi.index(int(captain))
    scenario_totals = scenario_values.sum(axis=1) + scenario_values[:, captain_column]
    scenario_summary = {
        "simulations": scenario_count,
        "p10": round(float(np.quantile(scenario_totals, 0.10)), 1),
        "median": round(float(np.quantile(scenario_totals, 0.50)), 1),
        "p90": round(float(np.quantile(scenario_totals, 0.90)), 1),
        "probability70": round(100 * float((scenario_totals >= 70).mean())),
        "probability80": round(100 * float((scenario_totals >= 80).mean())),
        "correlation": "Team clean-sheet outcomes share a 22% scenario shock.",
    }

    def player_payload(index: int, row: pd.Series) -> dict:
        fixture = fixture_map.get(int(row["team_id"]), {})
        opponent_id = fixture.get("opponent")
        fixture_peers = pool[
            (pool["fixture_id"] == int(row["fixture_id"]))
            & (pool["id"] != int(row["id"]))
        ]
        popular_rival = (
            fixture_peers.nlargest(1, "ownership").iloc[0]
            if not fixture_peers.empty
            else row
        )
        fixture_rank = int(
            1 + (fixture_peers["model_score"] > float(row["model_score"])).sum()
        )

        def clean_number(name: str, default: float = 0.0) -> float:
            value = row.get(name, default)
            return default if pd.isna(value) else float(value)

        set_pieces: list[str] = []
        for label, column in [
            ("Penalties", "penalties_order"),
            ("Direct free-kicks", "direct_freekicks_order"),
            ("Corners", "corners_and_indirect_freekicks_order"),
        ]:
            order = clean_number(column, 0)
            if 0 < order <= 2:
                set_pieces.append(f"{label} #{int(order)}")
        risk_flags: list[str] = []
        if float(row["expected_minutes"]) < 60:
            risk_flags.append("Minutes risk")
        if float(row["availability"]) < 100:
            risk_flags.append("Fitness flag")
        if float(row["sample_nineties"]) < 8:
            risk_flags.append("Small sample")
        if float(row["uncertainty"]) >= 0.42:
            risk_flags.append("Wide projection")
        if not risk_flags:
            risk_flags.append("No major flag")
        confidence = round(float(row["confidence"]))
        projection_percentile = float(row["projection_percentile"])
        verdict = (
            "Priority"
            if projection_percentile >= 0.90 and confidence >= 65
            else "Strong"
            if projection_percentile >= 0.75 and confidence >= 55
            else "Watch"
            if projection_percentile >= 0.45
            else "Fade"
        )
        if int(row["position_id"]) == 2:
            attacking_route = clean_number("goal_rate_prior") + clean_number(
                "assist_rate_prior"
            )
            defensive_route = clean_number("defensive_rate_prior")
            archetype = (
                "Set-piece centre-back"
                if attacking_route >= 0.18 and defensive_route >= 7.5
                else "Attacking full-back"
                if attacking_route >= 0.22
                else "Defensive centre-back"
                if defensive_route >= 8.5
                else "Balanced defender"
            )
        elif int(row["position_id"]) == 1:
            archetype = "Shot-stopping goalkeeper"
        elif int(row["position_id"]) == 3 and clean_number("defensive_rate_prior") >= 8:
            archetype = "Defensive-contribution midfielder"
        else:
            archetype = "Attacking role"
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
            "confidence": confidence,
            "valueProjected": round(float(row["value_projection"]), 2),
            "verdict": verdict,
            "setPieces": set_pieces,
            "riskFlags": risk_flags,
            "archetype": archetype,
            "minutesModel": {
                "startProbability": round(100 * float(row["start_probability"])),
                "playProbability": round(100 * float(row["play_probability"])),
                "sixtyProbability": round(100 * float(row["sixty_probability"])),
                "minutesIfStart": round(clean_number("minutes_if_start_forecast", 75)),
                "minutesIfBench": round(clean_number("minutes_if_bench_forecast", 18)),
                "minutesStd": round(float(row["minutes_std"]), 1),
                "rotationVolatility": round(100 * clean_number("rotation_volatility", 0.35)),
            },
            "distribution": {
                "p10": round(float(row["prediction_p10"]), 1),
                "median": round(float(row["prediction_p50"]), 1),
                "p90": round(float(row["prediction_p90"]), 1),
                "blankProbability": round(100 * float(row["blank_probability"])),
                "return5Probability": round(100 * float(row["return5_probability"])),
                "haul8Probability": round(100 * float(row["haul8_probability"])),
                "standardDeviation": round(float(row["projection_std"]), 2),
            },
            "defenderModel": {
                "actionRate": round(clean_number("defensive_rate_prior"), 1),
                "contributionProbability": round(
                    100 * float(row["defensive_return_probability"])
                ),
                "bpsRate": round(clean_number("bps_rate_prior"), 1),
                "goalRoute": round(clean_number("goal_rate_prior"), 3),
                "assistRoute": round(clean_number("assist_rate_prior"), 3),
            },
            "ensemble": {
                "structural": round(100 * float(row["ensemble_structural_weight_live"])),
                "empirical": round(100 * float(row["ensemble_empirical_weight_live"])),
                "marketRole": round(100 * float(row["ensemble_market_weight_live"])),
                "official": round(100 * float(row["ensemble_public_weight_live"])),
                "disagreement": round(float(row["ensemble_disagreement"]), 2),
            },
            "marketForecast": {
                "priceRiseProbability": round(100 * float(row["price_rise_probability"])),
                "priceFallProbability": round(100 * float(row["price_fall_probability"])),
            },
            "teamContext": {
                "expectedGoalsFor": round(float(row["team_expected_goals_for"]), 2),
                "expectedGoalsAgainst": round(float(row["team_expected_goals_against"]), 2),
                "cleanSheetProbability": round(100 * float(row["team_clean_probability"])),
                "horizonExpectedGoalsAgainst": round(
                    float(row["team_horizon_expected_goals_against"]), 2
                ),
                "horizonCleanSheetProbability": round(
                    100 * float(row["team_horizon_clean_probability"])
                ),
                "attackRank": round(float(row["team_attack_rank"])),
                "defenceRank": round(float(row["team_defence_rank"])),
                "strengthRank": round(float(row["team_strength_rank"])),
                "ratingConfidence": round(100 * float(row["team_rating_confidence"])),
                "regimeShift": round(100 * float(row["team_regime_shift"])),
            },
            "components": {
                "appearance": round(float(row["component_appearance"]), 2),
                "goals": round(float(row["component_goals"]), 2),
                "assists": round(float(row["component_assists"]), 2),
                "cleanSheet": round(float(row["component_clean"]), 2),
                "defence": round(float(row["component_defence"]), 2),
                "bonus": round(float(row["component_bonus"]), 2),
                "adjustment": round(float(row["component_adjustment"]), 2),
            },
            "history": {
                "matches": round(clean_number("history_matches")),
                "average": round(clean_number("history_average"), 2),
                "per90": round(clean_number("history_per90"), 2),
                "returnRate": round(100 * clean_number("history_returns")),
                "volatility": round(clean_number("history_volatility"), 2),
            },
            "opponentHistory": {
                "matches": round(clean_number("opponent_matches")),
                "average": round(clean_number("opponent_average"), 2),
                "per90": round(clean_number("opponent_per90"), 2),
                "returnRate": round(100 * clean_number("opponent_returns")),
            },
            "comparison": {
                "fixtureRank": fixture_rank,
                "fixturePlayers": int(len(fixture_peers) + 1),
                "positionRank": int(row["position_rank"]),
                "positionPlayers": int(row["position_count"]),
                "projectionRank": round(100 * projection_percentile),
                "popularRival": str(popular_rival["display_name"]),
                "popularRivalOwnership": round(float(popular_rival["ownership"]), 1),
                "popularRivalProjection": round(float(popular_rival["raw_projection"]), 1),
                "edgeVsPopular": round(
                    float(row["raw_projection"] - popular_rival["raw_projection"]), 1
                ),
            },
            "captainRating": round(float(row["captain_score"]) * 100),
            "score": round(float(row["model_score"]) * 100),
            "strategyScores": {
                "protect": round(float(row["protect_utility"]), 4),
                "balanced": round(float(row["balanced_utility"]), 4),
                "chase": round(float(row["chase_utility"]), 4),
            },
            "features": {
                "recent": round(float(row["recent"]), 4),
                "history": round(float(row["long"]), 4),
                "recentValue": round(float(row["recent_value"]), 4),
                "historyValue": round(float(row["long_value"]), 4),
                "age": round(float(row["age_score"]), 4),
                "fixture": round(float(row["fixture"]), 4),
                "team": round(float(row["team_context"]), 4),
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
                "modelProjection": round(float(model_pick["raw_projection"]), 1),
                "popularProjection": round(float(popular_pick["raw_projection"]), 1),
                "popularOwnership": round(float(popular_pick["ownership"]), 1),
                "modelConfidence": round(float(model_pick["confidence"])),
                "edge": round(
                    float(
                        model_pick["raw_projection"]
                        - popular_pick["raw_projection"]
                    ),
                    1,
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
        "scenario": scenario_summary,
    }
    current_meta = {
        "playersScored": int(len(pool)),
        "fixturesScored": int(len(first_fixtures)),
        "historicalSeasons": int(historical["season"].nunique()),
        "componentModel": "Probabilistic minutes + causal position ensemble + team Poisson + defender DC/BPS",
        "strategyProfiles": strategy_profiles,
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
    best_scores, robust_plan_scores, _ = candidate_forecasts(data, best)
    _, central_plan_scores, _ = candidate_forecasts(
        data, best, robust_planning=False
    )
    chip_policies = chip_policy_pool()
    gate_policy = chip_policies[-4]
    robust_gate_fresh = precompute_fresh_squads(data, robust_plan_scores)
    robust_probe_totals, _ = simulate_candidate(
        data,
        best_scores,
        WEEKLY_CHASE_STRATEGY,
        chip_policy=gate_policy,
        fresh_squads=robust_gate_fresh,
        plan_scores=robust_plan_scores,
    )
    central_gate_fresh = precompute_fresh_squads(data, central_plan_scores)
    central_probe_totals, _ = simulate_candidate(
        data,
        best_scores,
        WEEKLY_CHASE_STRATEGY,
        chip_policy=gate_policy,
        fresh_squads=central_gate_fresh,
        plan_scores=central_plan_scores,
    )
    robust_planning_enabled = bool(
        np.mean(robust_probe_totals[:training_count])
        >= np.mean(central_probe_totals[:training_count])
    )
    best_plan_scores = (
        robust_plan_scores if robust_planning_enabled else central_plan_scores
    )
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
                    recursive_candidates[int(index)].team,
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
        trial_scores, trial_plan_scores, _ = candidate_forecasts(
            data,
            trial_candidate,
            robust_planning=robust_planning_enabled,
        )
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
        current_rule_scores, current_rule_plan_scores, _ = candidate_forecasts(
            data,
            trial_candidate,
            current_rules=True,
            robust_planning=robust_planning_enabled,
        )
        current_rule_fresh_squads = precompute_fresh_squads(
            data, current_rule_plan_scores
        )
        current_rule_totals, _ = simulate_candidate(
            data,
            current_rule_scores,
            WEEKLY_CHASE_STRATEGY,
            chip_policy=trial_policy,
            fresh_squads=current_rule_fresh_squads,
            plan_scores=current_rule_plan_scores,
            actual_column="points_current_rules",
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
                "currentRulePoints": round(float(current_rule_totals[season_id])),
                "currentRuleDelta": round(float(current_rule_totals[season_id] - points)),
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
    non_team_scale = 1 / max(1 - best.team, 0.01)
    no_team_candidate = Candidate(
        best.performance * non_team_scale,
        best.value * non_team_scale,
        best.age * non_team_scale,
        best.fixture * non_team_scale,
        0.0,
        best.crowd * non_team_scale,
        best.minutes * non_team_scale,
        best.underlying * non_team_scale,
        best.recent_share,
    )
    no_team_scores, no_team_plan_scores, _ = candidate_forecasts(
        data, no_team_candidate
    )
    no_team_totals, _ = simulate_candidate(
        data,
        no_team_scores,
        WEEKLY_CHASE_STRATEGY,
        plan_scores=no_team_plan_scores,
    )
    best_model_score = feature_matrix(data) @ best.coefficients
    best_calibration = 0.72 + 0.56 * best_model_score
    structural_only_scores = (
        data["component_xpts_structural"].to_numpy(float) * best_calibration
    )
    structural_only_plan = (
        data["component_horizon"].to_numpy(float)
        * (
            data["component_xpts_structural"]
            / data["component_xpts"].clip(lower=0.2)
        ).to_numpy(float)
        * best_calibration
    )
    structural_only_totals, _ = simulate_candidate(
        data,
        structural_only_scores,
        WEEKLY_CHASE_STRATEGY,
        plan_scores=structural_only_plan,
    )
    rejected_plan_totals = (
        central_probe_totals if robust_planning_enabled else robust_probe_totals
    )
    selected_plan_totals = (
        robust_probe_totals if robust_planning_enabled else central_probe_totals
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
        advice_test(
            "Team-strength signal",
            best_totals,
            no_team_totals,
            "Keep all structural clean-sheet logic fixed, then test whether the separately learned causal team attack/defence feature improves recursive squad decisions.",
        ),
        advice_test(
            "Causal position ensemble",
            best_totals,
            structural_only_totals,
            "Compare the dynamically error-weighted structural, empirical and market-role blend with the same structural model on its own.",
        ),
        advice_test(
            "Planning-objective gate",
            selected_plan_totals,
            rejected_plan_totals,
            "Use the two pre-2018 training seasons and the same fixed preseason chip policy to choose between central six-GW expected points and a downside/price/upside objective, then freeze that choice before evaluation.",
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
    calibration_diagnostics = build_calibration_diagnostics(data, walk_forward)
    current_rules_replay = {
        "averagePoints": round(
            float(np.mean([item["currentRulePoints"] for item in walk_forward])), 1
        ),
        "averageScoringDelta": round(
            float(np.mean([item["currentRuleDelta"] for item in walk_forward])), 1
        ),
        "seasons": [
            {
                "season": item["season"],
                "points": item["currentRulePoints"],
                "deltaVsHistoricalRules": item["currentRuleDelta"],
            }
            for item in walk_forward
        ],
        "eventCoverage": calibration_diagnostics["defensiveEventCoverage"],
        "method": "Counterfactual 2026/27 scoring replay. Exact CBIT/CBIRT where public event data exists; coverage-labelled post-match proxy elsewhere. It is not used as a historical rank estimate.",
    }
    headline, squad, watchlist, matchups, all_players, current_meta = current_recommendation(
        data, best, robust_planning_enabled
    )
    result = {
        "product": "FPL Lens",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
        "model": {
            "version": "Lens 6.0",
            "trials": len(candidates),
            "recursiveTrials": len(recursive_candidates),
            "seasons": len(EVALUATION_SEASONS),
            "trainingSeasons": len(TRAINING_SEASONS),
            "playerWeeks": int(len(data)),
            "bestTrial": best_index + 1,
            "weights": best.as_dict(),
            "method": "Leak-free six-GW walk-forward replay with probabilistic minutes, position-specific causal ensembles, team Poisson rates, defender contribution thresholds, price option value and stochastic downside/upside forecasts.",
            "objective": (
                "Maximise legal autosubbed XI, captain and chip points; a training-only gate selected the downside/price/upside six-GW objective."
                if robust_planning_enabled
                else "Maximise legal autosubbed XI, captain and chip points; a training-only gate rejected the risk overlay and retained central six-GW expected points."
            ),
            "robustPlanningEnabled": robust_planning_enabled,
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
        "currentRulesReplay": current_rules_replay,
        "calibrationDiagnostics": calibration_diagnostics,
        "probabilisticEngine": {
            "playerIntervals": "10th, median and 90th percentile forecasts for every current player",
            "squadScenarios": headline["scenario"]["simulations"],
            "riskProfiles": current_meta["strategyProfiles"],
            "minutesModel": "Start, bench appearance, 60-minute and conditional-minutes distributions",
            "ensemble": "Causally error-weighted structural, empirical and market-role models plus the official current projection when available",
            "defenderModel": "Poisson CBIT/CBIRT threshold probability, clean-sheet correlation, set-piece route and 2026/27 BPS adjustment",
            "planningObjective": {
                "selected": "downside/price/upside" if robust_planning_enabled else "central expected points",
                "gate": "Frozen using only the two pre-2018 training seasons",
            },
        },
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
                "label": "Dixon-Coles dynamic score model",
                "url": "https://www.research.lancs.ac.uk/portal/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-football-betting-market%28d16276a2-d6e0-483b-a708-1d29663f1992%29.html",
            },
            {
                "label": "Bayesian hierarchical football model",
                "url": "https://discovery.ucl.ac.uk/id/eprint/16040/",
            },
            {
                "label": "Goal-chance team-strength evidence",
                "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0104647",
            },
            {
                "label": "OpenFPL forecasting + optimisation",
                "url": "https://arxiv.org/abs/2508.09992",
            },
            {
                "label": "Official defender contribution analysis",
                "url": "https://www.premierleague.com/en/news/4361968/which-defenders-will-get-the-most-defensive-contribution-points-in-fpl",
            },
            {
                "label": "Official 2026/27 defensive contributions",
                "url": "https://www.premierleague.com/en/news/4361991/whats-happening-with-defensive-contribution-points-in-202627-fantasy",
            },
            {
                "label": "Official 2026/27 BPS changes",
                "url": "https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system",
            },
            {
                "label": "Robust FPL integer optimisation",
                "url": "https://arxiv.org/abs/2505.02170",
            },
            {
                "label": "Probability calibration survey",
                "url": "https://link.springer.com/article/10.1007/s10994-023-06336-7",
            },
            {
                "label": "Event-sequence xG research",
                "url": "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0312278",
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
            "Minutes are a probability tree: start chance, bench appearance, conditional minutes and 60-minute chance are estimated separately, with rest and rotation volatility applied before the deadline.",
            "Three position-specific causal forecasts are blended by their prior out-of-sample error; current official xPts joins only as a fourth live ensemble member and never enters historical rows retroactively.",
            "The replay selects a legal formation, orders the bench, applies autosubs and hands the armband to the vice-captain when required.",
            "Chip decisions compare the current signal with the discounted option value of known future blank/double structures and receive expiry relief near the end of each chip window.",
            "The transfer planner looks six Gameweeks ahead and can bank up to the cap that applied in that season; paid-hit variants were tested and rejected when they reduced replay points.",
            "The top-500k result is an estimated scoring-pace test because official historic rank cut-offs are not exposed by the FPL API.",
            "Player analysis decomposes expected points by scoring route, labels sample size, estimates minutes and uncertainty, and treats opponent history as descriptive rather than predictive on its own.",
            "Team attack and defence are shifted, exponentially weighted and shrunk toward the league mean; current xG/xGA is blended with goals where available, so promoted and low-sample teams are not overconfidently rated.",
            "Defender and goalkeeper clean-sheet points are driven primarily by the opponent-adjusted team Poisson rate, then combined with expected minutes, attacking involvement, defensive contributions and bonus routes.",
            "Current-rule counterfactual points use exact defensive event counts where public data exists and a labelled post-match proxy elsewhere; they are kept separate from historical rank estimates.",
            "Live squads are evaluated in 5,000 correlated scenarios and exposed as Protect, Balanced and Chase profiles; the deterministic legal squad constraints remain binding in every profile.",
            "Price-rise and fall probabilities use transfer pressure as an option-value tiebreaker, not as a substitute for expected points.",
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
        weights.get("team", 0) / 100,
        weights["crowd"] / 100,
        weights["minutes"] / 100,
        weights["underlying"] / 100,
        weights["recent"] / 100,
    )
    ages = load_age_register()
    nationalities = load_nationality_register()
    historical = prepare_causal_history(
        [build_season(season, ages, nationalities)[0] for season in SEASONS]
    )
    headline, squad, watchlist, matchups, all_players, current_meta = (
        current_recommendation(
            historical,
            best,
            bool(result["model"].get("robustPlanningEnabled", False)),
        )
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

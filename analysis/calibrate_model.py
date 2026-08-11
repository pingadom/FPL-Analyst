"""Calibrate the FPL Lens ranking model on leak-free historical gameweeks.

The script downloads public FPL snapshots, constructs only pre-deadline features,
replays every gameweek from 2018-19 onward, evaluates hundreds of candidate
weight sets, and writes a compact JSON artifact consumed by the website.
"""

from __future__ import annotations

import json
import math
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
SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
REEP_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"
CURRENT_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
CURRENT_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
TRIALS = 2400
RECURSIVE_FINALISTS = 240
POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
XI_QUOTAS = {1: 1, 2: 3, 3: 5, 4: 2}
SQUAD_QUOTAS = {1: 2, 2: 5, 3: 5, 4: 3}


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


def season_files(season: str) -> tuple[Path, Path, Path]:
    folder = CACHE / season
    gw = download(f"{BASE}/{season}/gws/merged_gw.csv", folder / "merged_gw.csv")
    players = download(f"{BASE}/{season}/players_raw.csv", folder / "players_raw.csv")
    try:
        teams = download(f"{BASE}/{season}/teams.csv", folder / "teams.csv")
    except HTTPError as error:
        if error.code != 404:
            raise
        teams = download(f"{BASE}/{season}/raw.json", folder / "raw.json")
    return gw, players, teams


def build_season(season: str, ages: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    gw_path, players_path, teams_path = season_files(season)
    gw = pd.read_csv(gw_path, encoding="latin-1", low_memory=False)
    players = pd.read_csv(players_path, encoding="latin-1", low_memory=False)
    if teams_path.suffix == ".json":
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

    wanted = [
        "element",
        "GW",
        "total_points",
        "minutes",
        "value",
        "selected",
        "opponent_team",
        "was_home",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "transfers_balance",
    ]
    raw = gw[[column for column in wanted if column in gw.columns]].copy()
    raw = raw.merge(meta, on="element", how="left")
    raw["team_name"] = raw["team_id"].map(team_names)
    raw["opponent_name"] = raw["opponent_team"].map(team_names)
    raw["selected"] = pd.to_numeric(raw.get("selected", 0), errors="coerce").fillna(0)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").fillna(45)
    for column in [
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "transfers_balance",
    ]:
        raw[column] = pd.to_numeric(raw.get(column, 0), errors="coerce").fillna(0)
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
            ict=("ict_index", "sum"),
            influence=("influence", "sum"),
            creativity=("creativity", "sum"),
            threat=("threat", "sum"),
            transfers_balance=("transfers_balance", "max"),
            display_name=("display_name", "first"),
            birth_date=("birth_date", "first"),
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

    # The next four opponents are known at the deadline. Their strength is always
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
    horizon_weights = (1.0, 0.82, 0.67, 0.55)

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

    eligible = weekly[
        (weekly["observations"] >= 3)
        & (weekly["past_minutes"].fillna(0) >= 20)
        & (weekly["price"] >= 35)
    ].copy()
    age_coverage = float(weekly["age"].notna().mean())
    summary = {
        "season": season,
        "rows": int(len(weekly)),
        "eligibleRows": int(len(eligible)),
        "ageCoverage": round(age_coverage * 100, 1),
        "gameweeks": int(eligible["GW"].nunique()),
    }
    return eligible, summary


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
        [3.5, 1.7, 0.45, 1.6, 1.0, 2.0, 1.8], size=TRIALS - 5
    )
    recent = rng.beta(4.0, 2.5, size=TRIALS - 5) * 0.75 + 0.15
    candidates = [
        Candidate(*weights, float(recency))
        for weights, recency in zip(raw_weights, recent)
    ]
    candidates.extend(
        [
            # Official-winner principles: form + medium-term fixtures, reliable
            # minutes, underlying data, restrained ownership and almost no age prior.
            Candidate(0.32, 0.10, 0.01, 0.15, 0.08, 0.18, 0.16, 0.65),
            Candidate(0.48, 0.08, 0.01, 0.20, 0.04, 0.12, 0.07, 0.68),
            Candidate(0.34, 0.10, 0.00, 0.15, 0.05, 0.20, 0.16, 0.58),
            Candidate(0.52, 0.09, 0.00, 0.10, 0.03, 0.14, 0.12, 0.52),
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


EXPERT_STRATEGY = SimulationStrategy(
    "Patient transfers + safe captain", 0.12, 2, False, True
)
WEEKLY_CHASE_STRATEGY = SimulationStrategy(
    "Forced weekly transfer + model captain", 0.0, 1, True, False
)


def initial_squad(frame: pd.DataFrame, scores: np.ndarray) -> list[int]:
    """Fast legal £100m squad build used at the start of each recursive season."""
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
        if cost > 1000:
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


def choose_xi(
    squad: dict[int, dict], row_by_element: dict[int, int], scores: np.ndarray
) -> tuple[list[int], list[int]]:
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
                position_pool.sort(
                    key=lambda element: scores[row_by_element[element]]
                    if element in row_by_element
                    else -1.0,
                    reverse=True,
                )
                chosen.extend(position_pool[:count])
            total = sum(
                scores[row_by_element[element]]
                if element in row_by_element
                else -1.0
                for element in chosen
            )
            if len(chosen) == 11 and total > best_score:
                best_xi = chosen
                best_score = total
    bench = [element for element in squad if element not in set(best_xi)]
    bench.sort(
        key=lambda element: (
            int(squad[element]["position"]) != 1,
            scores[row_by_element[element]] if element in row_by_element else -1.0,
        ),
        reverse=True,
    )
    # FPL puts the reserve goalkeeper in a separate slot; outfield order is by score.
    bench_gk = [element for element in bench if int(squad[element]["position"]) == 1]
    bench_outfield = [
        element for element in bench if int(squad[element]["position"]) != 1
    ]
    bench_outfield.sort(
        key=lambda element: scores[row_by_element[element]]
        if element in row_by_element
        else -1.0,
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
    points = sum(
        actual[row_by_element[element]]
        for element in final_xi
        if played(element)
    )
    if played(captain):
        points += actual[row_by_element[captain]]
    elif played(vice):
        points += actual[row_by_element[vice]]
    return float(points)


def selling_price(purchase_price: int, current_price: int) -> int:
    if current_price <= purchase_price:
        return current_price
    return purchase_price + (current_price - purchase_price) // 2


def simulate_candidate(
    data: pd.DataFrame,
    scores: np.ndarray,
    strategy: SimulationStrategy,
) -> tuple[np.ndarray, list[dict]]:
    """Carry one legal squad through each season and make deadline-only transfers."""
    actual = data["points"].to_numpy(float)
    played_minutes = data["minutes"].to_numpy(float)
    element_values = data["element"].to_numpy(int)
    position_values = data["position_id"].to_numpy(int)
    team_values = data["team_id"].to_numpy(int)
    price_values = data["price"].to_numpy(int)
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
        rolled = 0
        weekly_changes: list[int] = []

        for week_number, gw in enumerate(weeks):
            frame = season_data[season_data["GW"] == gw]
            frame_indices = frame.index.to_numpy(int)
            row_by_element = dict(
                zip(element_values[frame_indices].tolist(), frame_indices.tolist())
            )
            incoming_by_position: dict[int, np.ndarray] = {}
            for position in SQUAD_QUOTAS:
                position_indices = frame_indices[
                    position_values[frame_indices] == position
                ]
                incoming_by_position[position] = position_indices[
                    np.argsort(scores[position_indices])[::-1]
                ][:40]
            if week_number == 0:
                initial_indices = initial_squad(frame, scores)
                for index in initial_indices:
                    squad[int(element_values[index])] = {
                        "position": int(position_values[index]),
                        "team": int(team_values[index]),
                        "purchase": int(price_values[index]),
                        "last_price": int(price_values[index]),
                    }
                bank = 1000 - sum(state["purchase"] for state in squad.values())
                weekly_changes.append(15)
            else:
                for element, state in squad.items():
                    if element in row_by_element:
                        current_index = row_by_element[element]
                        state["team"] = int(team_values[current_index])
                        state["last_price"] = int(price_values[current_index])

                changes_this_week = 0
                for _ in range(free_transfers):
                    team_counts: dict[int, int] = {}
                    for state in squad.values():
                        team_counts[int(state["team"])] = (
                            team_counts.get(int(state["team"]), 0) + 1
                        )
                    best_move: tuple[float, int, int, int, int] | None = None
                    for outgoing, state in squad.items():
                        out_index = row_by_element.get(outgoing)
                        out_score = scores[out_index] if out_index is not None else -0.12
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
                            if incoming_element in squad:
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
                            gain = float(scores[int(incoming_index)] - out_score)
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
                    if gain <= strategy.transfer_hurdle and not (
                        strategy.force_weekly_review and gain > 0
                    ):
                        break
                    bank += sale - int(price_values[incoming_index])
                    del squad[outgoing]
                    squad[incoming] = {
                        "position": int(position_values[incoming_index]),
                        "team": int(team_values[incoming_index]),
                        "purchase": int(price_values[incoming_index]),
                        "last_price": int(price_values[incoming_index]),
                    }
                    changes_this_week += 1
                    transfers += 1
                if changes_this_week == 0:
                    rolled += 1
                weekly_changes.append(changes_this_week)
                free_transfers = min(
                    strategy.bank_limit,
                    max(0, free_transfers - changes_this_week) + 1,
                )

            xi, bench = choose_xi(squad, row_by_element, scores)
            captain_metric = safe_captain_score if strategy.safe_captain else scores
            captain_order = sorted(
                xi,
                key=lambda element: captain_metric[row_by_element[element]]
                if element in row_by_element
                else -1.0,
                reverse=True,
            )
            captain, vice = captain_order[:2]
            totals[season_id] += realised_week_points(
                xi,
                bench,
                captain,
                vice,
                squad,
                row_by_element,
                actual,
                played_minutes,
            )

        season_stats.append(
            {
                "season": season,
                "transfers": transfers,
                "rolled": rolled,
                "weeksChanged": sum(change > 0 for change in weekly_changes[1:]),
                "gameweeks": len(weeks),
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
        trial_scores = features @ candidate.coefficients
        results[trial_index], _ = simulate_candidate(data, trial_scores, strategy)
        if (trial_index + 1) % 40 == 0 or trial_index + 1 == len(candidates):
            print(
                f"Recursive replay {trial_index + 1}/{len(candidates)} "
                f"({strategy.name})"
            )
    return results


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
        horizon_fixtures["event"].between(gw_number, gw_number + 3)
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
    horizon_weight = {gw_number + offset: weight for offset, weight in enumerate((1.0, 0.82, 0.67, 0.55))}
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
    current["raw_projection"] = (
        best.recent_share * current["recent_raw"]
        + (1 - best.recent_share) * current["long_raw"]
    ) * (0.80 + current["fixture_now"] * 0.30) * (
        0.72 + current["minutes_security"] * 0.34
    )
    current["raw_projection"] = current["raw_projection"].clip(1.0, 10.5)
    current["availability"] = pd.to_numeric(
        current["chance_of_playing_next_round"], errors="coerce"
    ).fillna(100)
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
    season_frames: list[pd.DataFrame] = []
    data_summary: list[dict] = []
    for season in SEASONS:
        frame, summary = build_season(season, ages)
        season_frames.append(frame)
        data_summary.append(summary)
        print(f"Prepared {season}: {summary['eligibleRows']:,} eligible player-weeks")

    data = pd.concat(season_frames, ignore_index=True)
    candidates, baseline_index = candidate_pool()
    snapshot_scores, seasons = snapshot_replay(data, candidates)
    gameweeks = np.array(
        [data.loc[data["season"] == season, "GW"].nunique() for season in seasons],
        dtype=float,
    )
    snapshot_per_gameweek = snapshot_scores / gameweeks
    snapshot_stability = snapshot_per_gameweek.mean(axis=1) - snapshot_per_gameweek.std(axis=1) * 0.18

    # Recursively replay the most promising and most season-diverse candidates.
    shortlist_indices: list[int] = []
    priority = [baseline_index] + list(range(len(candidates) - 5, len(candidates)))
    for season_id in range(len(seasons)):
        priority.extend(
            np.argsort(snapshot_per_gameweek[:, season_id])[-8:][::-1].astype(int).tolist()
        )
    priority.extend(np.argsort(snapshot_stability)[::-1].astype(int).tolist())
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
    all_features = feature_matrix(data)

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

    for season_id, season in enumerate(seasons):
        if season_id == 0:
            trial_candidate = best
            mode = "calibration seed"
        else:
            prior = per_gameweek[:, :season_id]
            train_score = prior.mean(axis=1) - prior.std(axis=1) * 0.25
            ensemble_indices = np.argsort(train_score)[-12:]
            trial_candidate = blend_candidates(ensemble_indices)
            mode = f"12-model ensemble trained on {season_id} prior season{'s' if season_id != 1 else ''}"
        trial_vector = all_features @ trial_candidate.coefficients
        trial_totals, trial_stats = simulate_candidate(
            data, trial_vector, WEEKLY_CHASE_STRATEGY
        )
        points = round(float(trial_totals[season_id]))
        baseline = round(float(recursive_scores[baseline_local_index, season_id]))
        season_transfer_stats = trial_stats[season_id]
        walk_forward.append(
            {
                "season": season.replace("-", "/"),
                "points": points,
                "baseline": baseline,
                "uplift": round((points / baseline - 1) * 100, 1) if baseline else 0,
                "mode": mode,
                "weights": trial_candidate.as_dict(),
                "transfers": season_transfer_stats["transfers"],
                "weeksChanged": season_transfer_stats["weeksChanged"],
                "rolled": season_transfer_stats["rolled"],
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

    best_vector = all_features @ best.coefficients
    best_totals, best_stats = simulate_candidate(
        data, best_vector, WEEKLY_CHASE_STRATEGY
    )
    weekly_safe_captain = SimulationStrategy(
        "Forced weekly transfer + safe captain", 0.0, 1, True, True
    )
    patient_model_captain = SimulationStrategy(
        "Patient transfers + model captain", 0.12, 2, False, False
    )
    safe_captain_totals, _ = simulate_candidate(
        data, best_vector, weekly_safe_captain
    )
    patient_totals, patient_stats = simulate_candidate(
        data, best_vector, patient_model_captain
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
            "Add four-GW fixtures, minutes security and ICT involvement to the original Lens feature set.",
        ),
    ]

    headline, squad, watchlist, matchups, all_players, current_meta = current_recommendation(data, best)
    result = {
        "product": "FPL Lens",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
        "model": {
            "version": "Lens 2.0",
            "trials": len(candidates),
            "recursiveTrials": len(recursive_candidates),
            "seasons": len(seasons),
            "playerWeeks": int(len(data)),
            "bestTrial": best_index + 1,
            "weights": best.as_dict(),
            "method": "Stateful walk-forward replay: one legal 15-player squad is carried into the next deadline, re-ranked, transferred and selected using only data available then.",
            "objective": "Maximise autosubbed XI + captain points, with legal budget/club limits and a season-volatility penalty.",
            "strategy": WEEKLY_CHASE_STRATEGY.name,
        },
        "headline": headline,
        "currentMeta": current_meta,
        "squad": squad,
        "watchlist": watchlist,
        "fixtureMatchups": matchups,
        "currentPlayers": all_players,
        "backtest": walk_forward,
        "expertTests": expert_tests,
        "simulationSummary": {
            "averageTransfers": round(float(np.mean([item["transfers"] for item in best_stats])), 1),
            "averageWeeksChanged": round(float(np.mean([item["weeksChanged"] for item in best_stats])), 1),
            "averageRolled": round(float(np.mean([item["rolled"] for item in best_stats])), 1),
            "patientAverageTransfers": round(float(np.mean([item["transfers"] for item in patient_stats])), 1),
        },
        "leaderboard": leaderboard,
        "calibrationCurve": calibration_curve,
        "dataSummary": data_summary,
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
        ],
        "notes": [
            "Every historical GW is recursive: the prior squad, bank and up to two free transfers carry forward; transfers use contemporaneous prices and FPL selling-price rules.",
            "The replay selects a legal formation, orders the bench, applies autosubs and hands the armband to the vice-captain when required.",
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


if __name__ == "__main__":
    main()

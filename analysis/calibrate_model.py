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
TRIALS = 640
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
    ]
    raw = gw[[column for column in wanted if column in gw.columns]].copy()
    raw = raw.merge(meta, on="element", how="left")
    raw["team_name"] = raw["team_id"].map(team_names)
    raw["opponent_name"] = raw["opponent_team"].map(team_names)
    raw["selected"] = pd.to_numeric(raw.get("selected", 0), errors="coerce").fillna(0)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").fillna(45)
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
    weekly["recent_raw"] = weekly["recent_raw"].fillna(weekly["long_raw"])
    weekly["long_value_raw"] = weekly["long_raw"] / (weekly["price"] / 10).clip(3.5)
    weekly["recent_value_raw"] = weekly["recent_raw"] / (weekly["price"] / 10).clip(3.5)
    weekly["age_raw"] = np.exp(-((weekly["age"].fillna(27.5) - 27.5) / 7.5) ** 2)
    weekly["crowd_raw"] = np.log1p(weekly["selected"].clip(lower=0))

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

    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_raw", "fixture"),
        ("crowd_raw", "crowd"),
    ]:
        weekly[rank_name] = weekly.groupby("GW")[raw_name].transform(percentile)

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
            ],
            dtype=float,
        )

    def as_dict(self) -> dict:
        return {
            "performance": round(self.performance * 100),
            "value": round(self.value * 100),
            "age": round(self.age * 100),
            "fixture": round(self.fixture * 100),
            "crowd": round(self.crowd * 100),
            "recent": round(self.recent_share * 100),
            "history": round((1 - self.recent_share) * 100),
        }


def candidate_pool() -> list[Candidate]:
    rng = np.random.default_rng(20260811)
    raw_weights = rng.dirichlet([3.2, 2.2, 0.8, 1.7, 0.7], size=TRIALS - 4)
    recent = rng.beta(4.0, 2.5, size=TRIALS - 4) * 0.75 + 0.15
    candidates = [
        Candidate(*weights, float(recency))
        for weights, recency in zip(raw_weights, recent)
    ]
    candidates.extend(
        [
            Candidate(0.48, 0.20, 0.06, 0.18, 0.08, 0.68),
            Candidate(0.58, 0.20, 0.02, 0.15, 0.05, 0.50),
            Candidate(0.35, 0.35, 0.05, 0.20, 0.05, 0.75),
            Candidate(0.20, 0.15, 0.05, 0.25, 0.35, 0.65),
        ]
    )
    return candidates


def replay(data: pd.DataFrame, candidates: list[Candidate]) -> tuple[np.ndarray, list[str]]:
    features = data[
        ["recent", "long", "recent_value", "long_value", "age_score", "fixture", "crowd"]
    ].to_numpy(dtype=float)
    actual = data["points"].to_numpy(dtype=float)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    season_index = {season: index for index, season in enumerate(seasons)}
    groups: list[tuple[np.ndarray, int, int]] = []
    captain_groups: dict[tuple[str, int], list[np.ndarray]] = {}
    for (season, gw, position), frame in data.groupby(
        ["season", "GW", "position_id"], sort=False
    ):
        quota = XI_QUOTAS.get(int(position), 0)
        if quota <= 0 or len(frame) < quota:
            continue
        indices = frame.index.to_numpy(dtype=int)
        groups.append((indices, quota, season_index[season]))
        captain_groups.setdefault((season, int(gw)), []).append(indices)

    results = np.zeros((len(candidates), len(seasons)), dtype=float)
    for trial_index, candidate in enumerate(candidates):
        scores = features @ candidate.coefficients
        picked_by_week: dict[tuple[str, int], list[int]] = {}
        for indices, quota, season_id in groups:
            local = scores[indices]
            chosen = indices[np.argpartition(local, -quota)[-quota:]]
            results[trial_index, season_id] += actual[chosen].sum()
            row = data.iloc[int(chosen[0])]
            key = (str(row["season"]), int(row["GW"]))
            picked_by_week.setdefault(key, []).extend(chosen.tolist())
        for (season, _gw), chosen in picked_by_week.items():
            captain = max(chosen, key=lambda index: scores[index])
            results[trial_index, season_index[season]] += actual[captain]
    return results, seasons


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
            ["player_code", "previous_points", "previous_minutes", "recent_raw", "long_raw"]
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
    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_raw", "fixture"),
        ("crowd_raw", "crowd"),
    ]:
        current[rank_name] = current.groupby("position_id")[raw_name].transform(percentile)
    matrix = current[
        ["recent", "long", "recent_value", "long_value", "age_score", "fixture", "crowd"]
    ].to_numpy(float)
    current["model_score"] = matrix @ best.coefficients
    current["raw_projection"] = (
        best.recent_share * current["recent_raw"]
        + (1 - best.recent_share) * current["long_raw"]
    ) * (0.84 + current["fixture"] * 0.32)
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
    selected = pool.loc[chosen].sort_values(
        ["position_id", "model_score"], ascending=[True, False]
    )
    captain_order = pool.loc[xi].sort_values("raw_projection", ascending=False).index.tolist()
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
            "score": round(float(row["model_score"]) * 100),
            "features": {
                "recent": round(float(row["recent"]), 4),
                "history": round(float(row["long"]), 4),
                "recentValue": round(float(row["recent_value"]), 4),
                "historyValue": round(float(row["long_value"]), 4),
                "age": round(float(row["age_score"]), 4),
                "fixture": round(float(row["fixture"]), 4),
                "crowd": round(float(row["crowd"]), 4),
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
    candidates = candidate_pool()
    scores, seasons = replay(data, candidates)
    gameweeks = np.array(
        [data.loc[data["season"] == season, "GW"].nunique() for season in seasons],
        dtype=float,
    )
    per_gameweek = scores / gameweeks
    stability = per_gameweek.mean(axis=1) - per_gameweek.std(axis=1) * 0.18
    best_index = int(np.argmax(stability))
    best = candidates[best_index]
    baseline_index = len(candidates) - 3

    walk_forward: list[dict] = []
    for season_id, season in enumerate(seasons):
        if season_id == 0:
            trial_index = best_index
            mode = "calibration seed"
        else:
            train_score = per_gameweek[:, :season_id].mean(axis=1)
            trial_index = int(np.argmax(train_score))
            mode = f"trained on {season_id} prior season{'s' if season_id != 1 else ''}"
        points = round(float(scores[trial_index, season_id]))
        baseline = round(float(scores[baseline_index, season_id]))
        walk_forward.append(
            {
                "season": season.replace("-", "/"),
                "points": points,
                "baseline": baseline,
                "uplift": round((points / baseline - 1) * 100, 1) if baseline else 0,
                "mode": mode,
                "weights": candidates[trial_index].as_dict(),
            }
        )

    top_indices = np.argsort(stability)[-5:][::-1]
    leaderboard = [
        {
            "rank": rank + 1,
            "trial": int(index) + 1,
            "pointsPerGameweek": round(float(per_gameweek[index].mean()), 2),
            "stability": round(float(stability[index]), 3),
            "weights": candidates[int(index)].as_dict(),
        }
        for rank, index in enumerate(top_indices)
    ]
    curve_indices = np.linspace(0, len(candidates) - 1, 16).astype(int)
    sorted_scores = np.sort(stability)
    calibration_curve = [
        {
            "percentile": round(int(index) / (len(candidates) - 1) * 100),
            "score": round(float(sorted_scores[index]), 2),
        }
        for index in curve_indices
    ]

    headline, squad, watchlist, matchups, all_players, current_meta = current_recommendation(data, best)
    result = {
        "product": "FPL Lens",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
        "model": {
            "version": "Lens 1.0",
            "trials": len(candidates),
            "seasons": len(seasons),
            "playerWeeks": int(len(data)),
            "bestTrial": best_index + 1,
            "weights": best.as_dict(),
            "method": "Walk-forward replay; every feature is shifted one gameweek to prevent future leakage.",
            "objective": "Maximise XI + captain FPL points while penalising season-to-season volatility.",
        },
        "headline": headline,
        "currentMeta": current_meta,
        "squad": squad,
        "watchlist": watchlist,
        "fixtureMatchups": matchups,
        "currentPlayers": all_players,
        "backtest": walk_forward,
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
        ],
        "notes": [
            "Backtests use a fixed legal 3-5-2 XI plus the model's top-ranked captain; budget optimisation is applied to the current 15-player squad.",
            "Age is an availability/consistency prior, not a claim that younger or older players are inherently better.",
            "Current projections are decision support, not guarantees; late team news should override the model.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Best trial {best_index + 1}/{len(candidates)}; wrote {OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()

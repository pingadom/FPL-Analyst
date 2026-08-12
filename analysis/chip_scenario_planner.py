"""Paired Monte Carlo chip diagnostics with conservative option-value gates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from prospective_common import APP_DATA, ROOT, SHADOW_ROOT, atomic_json, normal_scenarios, optimise_squad


DRAWS = 5_000
SEED = 271828


def summary(values: np.ndarray) -> dict:
    return {
        "meanGain": round(float(values.mean()), 1),
        "p10Gain": round(float(np.quantile(values, 0.10)), 1),
        "medianGain": round(float(np.median(values)), 1),
        "p90Gain": round(float(np.quantile(values, 0.90)), 1),
        "probabilityPositive": round(float((values > 0).mean() * 100), 1),
    }


def best_lineup(players: list[dict], squad: list[int], score_key: str) -> tuple[list[int], int, int]:
    options = [
        {"GK": 1, "DEF": defenders, "MID": 10 - defenders - forwards, "FWD": forwards}
        for defenders in (3, 4, 5)
        for forwards in (1, 2, 3)
        if 2 <= 10 - defenders - forwards <= 5
    ]
    xi = max(
        (
            [
                index
                for position, count in formation.items()
                for index in sorted(
                    [candidate for candidate in squad if players[candidate]["position"] == position],
                    key=lambda candidate: float(players[candidate][score_key]),
                    reverse=True,
                )[:count]
            ]
            for formation in options
        ),
        key=lambda selection: sum(float(players[index][score_key]) for index in selection)
        if len(selection) == 11 else -1e9,
    )
    captain = max(xi, key=lambda index: float(players[index][score_key]))
    vice = max((index for index in xi if index != captain), key=lambda index: float(players[index][score_key]))
    return xi, captain, vice


def scenario_score(draws: np.ndarray, xi: list[int], captain: int) -> np.ndarray:
    return draws[:, xi].sum(axis=1) + draws[:, captain]


def fixture_structure(snapshot: dict, gameweek: int) -> tuple[dict[int, Counter], dict[int, str]]:
    teams = {int(row["id"]): row["short_name"] for row in snapshot["official"]["teams"]}
    counts: dict[int, Counter] = defaultdict(Counter)
    for fixture in snapshot["official"]["fixtures"]:
        event = fixture.get("event")
        if event is None or not gameweek <= int(event) <= gameweek + 7:
            continue
        counts[int(event)][int(fixture["team_h"])] += 1
        counts[int(event)][int(fixture["team_a"])] += 1
    return counts, teams


def main() -> None:
    status = json.loads((APP_DATA / "deadline-status.json").read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / status["snapshotPath"]).read_text(encoding="utf-8"))
    player_rows = json.loads((APP_DATA / "current-players.json").read_text(encoding="utf-8"))
    model_results = json.loads((APP_DATA / "model-results.json").read_text(encoding="utf-8"))
    frontier_path = APP_DATA / "frontier-scores.json"
    frontier = {}
    if frontier_path.exists():
        frontier = {
            int(row["id"]): row for row in json.loads(frontier_path.read_text(encoding="utf-8"))["players"]
        }
    intelligence = {int(row["id"]): row for row in snapshot["deadlineIntelligence"]}
    players: list[dict] = []
    for row in player_rows:
        intel = intelligence.get(int(row["id"]), {})
        old_minutes = max(float(row["expectedMinutes"]), 8)
        new_minutes = float(intel.get("expectedMinutes", old_minutes))
        minute_ratio = np.clip(new_minutes / old_minutes, 0.20, 1.25)
        adjusted = float(row["projected"]) * (0.25 + 0.75 * minute_ratio)
        horizon_adjusted = float(row["sixWeekProjected"]) * (0.75 + 0.25 * minute_ratio)
        challenger = frontier.get(int(row["id"]), {})
        players.append(
            {
                **row,
                "structuralScore": float(adjusted),
                "horizonScore": float(horizon_adjusted),
                "frontierScore": float(challenger.get("blend25", adjusted)) * (0.25 + 0.75 * minute_ratio),
                "deadlineMinutes": new_minutes,
            }
        )
    id_to_index = {int(row["id"]): index for index, row in enumerate(players)}
    current = [id_to_index[int(row["id"])] for row in model_results["squad"] if int(row["id"]) in id_to_index]
    if len(current) != 15:
        raise RuntimeError("The frozen current squad could not be reconstructed.")
    current_xi, current_captain, current_vice = best_lineup(players, current, "structuralScore")
    fresh, fresh_xi, fresh_captain, fresh_vice = optimise_squad(players, "structuralScore")
    wildcard, wildcard_xi, wildcard_captain, wildcard_vice = optimise_squad(players, "horizonScore")

    mean = np.asarray([row["structuralScore"] for row in players])
    std = np.asarray([
        max(0.6, (float(row["distribution"]["p90"]) - float(row["distribution"]["p10"])) / 2.5632)
        for row in players
    ])
    draws = normal_scenarios(mean, std * np.sqrt(1 - 0.22**2), DRAWS, SEED)
    rng = np.random.default_rng(SEED + 1)
    for team in sorted({row["team"] for row in players}):
        indices = [index for index, row in enumerate(players) if row["team"] == team]
        shock = rng.normal(0, 1, size=(DRAWS, 1))
        draws[:, indices] = np.clip(draws[:, indices] + shock * std[indices] * 0.22, 0, None)
    baseline = scenario_score(draws, current_xi, current_captain)
    free_hit_gain = scenario_score(draws, fresh_xi, fresh_captain) - baseline
    bench_gain = draws[:, [index for index in current if index not in current_xi]].sum(axis=1)
    triple_gain = draws[:, current_captain]

    horizon_mean = np.asarray([row["horizonScore"] for row in players])
    horizon_std = std * np.sqrt(4.5)
    horizon_draws = normal_scenarios(horizon_mean, horizon_std, DRAWS, SEED + 2)
    wildcard_gain = (
        scenario_score(horizon_draws, wildcard_xi, wildcard_captain)
        - scenario_score(horizon_draws, *best_lineup(players, current, "horizonScore")[:2])
    )

    gameweek = int(status["gameweek"])
    counts, team_names = fixture_structure(snapshot, gameweek)
    name_to_team = {value: key for key, value in team_names.items()}
    current_counts = counts.get(gameweek, Counter())
    blank_players = sum(current_counts.get(name_to_team.get(players[index]["team"], -1), 0) == 0 for index in current)
    double_players = sum(current_counts.get(name_to_team.get(players[index]["team"], -1), 0) >= 2 for index in current)
    bench = [index for index in current if index not in current_xi]
    bench_doubles = sum(current_counts.get(name_to_team.get(players[index]["team"], -1), 0) >= 2 for index in bench)
    captain_double = current_counts.get(name_to_team.get(players[current_captain]["team"], -1), 0) >= 2
    future_events = []
    for event in range(gameweek + 1, gameweek + 8):
        event_counts = counts.get(event, Counter())
        future_events.append(
            {
                "gameweek": event,
                "blankTeams": sorted(team_names[team] for team in team_names if event_counts.get(team, 0) == 0),
                "doubleTeams": sorted(team_names[team] for team, count in event_counts.items() if count >= 2),
            }
        )
    wait_value = max(
        [0.0]
        + [2.5 * len(row["doubleTeams"]) + 0.7 * len(row["blankTeams"]) for row in future_events]
    )
    distributions = {
        "Hold": summary(np.zeros(DRAWS)),
        "Free Hit": summary(free_hit_gain),
        "Bench Boost": summary(bench_gain),
        "Triple Captain": summary(triple_gain),
        "Wildcard": summary(wildcard_gain),
    }
    gates = {
        "Free Hit": blank_players >= 3 and distributions["Free Hit"]["meanGain"] >= 10 and distributions["Free Hit"]["p10Gain"] >= 0 and distributions["Free Hit"]["probabilityPositive"] >= 75,
        "Bench Boost": bench_doubles >= 1 and distributions["Bench Boost"]["meanGain"] >= 10 and distributions["Bench Boost"]["p10Gain"] >= 4,
        "Triple Captain": captain_double and distributions["Triple Captain"]["meanGain"] >= 12 and distributions["Triple Captain"]["p10Gain"] >= 4,
        "Wildcard": gameweek > 1 and distributions["Wildcard"]["meanGain"] >= 25 and distributions["Wildcard"]["p10Gain"] >= 5 and distributions["Wildcard"]["probabilityPositive"] >= 75,
    }
    eligible = [name for name, passed in gates.items() if passed]
    if eligible:
        recommendation = max(
            eligible,
            key=lambda name: distributions[name]["meanGain"] - wait_value,
        )
        if distributions[recommendation]["meanGain"] <= wait_value:
            recommendation = "Hold"
    else:
        recommendation = "Hold"

    def saved_manager_plan(squad: list[int]) -> dict:
        """Re-evaluate chips against a manager's own recursive squad state."""
        xi, captain, _ = best_lineup(players, squad, "structuralScore")
        base = scenario_score(draws, xi, captain)
        free_hit = scenario_score(draws, fresh_xi, fresh_captain) - base
        bench_indices = [index for index in squad if index not in xi]
        bench_boost = draws[:, bench_indices].sum(axis=1)
        triple_captain = draws[:, captain]
        horizon_xi, horizon_captain, _ = best_lineup(players, squad, "horizonScore")
        wildcard_delta = (
            scenario_score(horizon_draws, wildcard_xi, wildcard_captain)
            - scenario_score(horizon_draws, horizon_xi, horizon_captain)
        )
        local_distributions = {
            "Hold": summary(np.zeros(DRAWS)),
            "Free Hit": summary(free_hit),
            "Bench Boost": summary(bench_boost),
            "Triple Captain": summary(triple_captain),
            "Wildcard": summary(wildcard_delta),
        }
        local_blank = sum(current_counts.get(name_to_team.get(players[index]["team"], -1), 0) == 0 for index in squad)
        local_double = sum(current_counts.get(name_to_team.get(players[index]["team"], -1), 0) >= 2 for index in squad)
        local_bench_double = sum(current_counts.get(name_to_team.get(players[index]["team"], -1), 0) >= 2 for index in bench_indices)
        local_captain_double = current_counts.get(name_to_team.get(players[captain]["team"], -1), 0) >= 2
        local_gates = {
            "Free Hit": local_blank >= 3 and local_distributions["Free Hit"]["meanGain"] >= 10 and local_distributions["Free Hit"]["p10Gain"] >= 0 and local_distributions["Free Hit"]["probabilityPositive"] >= 75,
            "Bench Boost": local_bench_double >= 1 and local_distributions["Bench Boost"]["meanGain"] >= 10 and local_distributions["Bench Boost"]["p10Gain"] >= 4,
            "Triple Captain": local_captain_double and local_distributions["Triple Captain"]["meanGain"] >= 12 and local_distributions["Triple Captain"]["p10Gain"] >= 4,
            "Wildcard": gameweek > 1 and local_distributions["Wildcard"]["meanGain"] >= 25 and local_distributions["Wildcard"]["p10Gain"] >= 5 and local_distributions["Wildcard"]["probabilityPositive"] >= 75,
        }
        local_eligible = [name for name, passed in local_gates.items() if passed]
        local_recommendation = "Hold"
        if local_eligible:
            candidate = max(local_eligible, key=lambda name: local_distributions[name]["meanGain"] - wait_value)
            if local_distributions[candidate]["meanGain"] > wait_value:
                local_recommendation = candidate
        return {
            "recommendation": local_recommendation,
            "currentStructure": {
                "blankPlayers": local_blank,
                "doublePlayers": local_double,
                "benchDoublePlayers": local_bench_double,
                "captainHasDouble": local_captain_double,
            },
            "scenarios": [
                {
                    "chip": name,
                    **values,
                    "gatePassed": True if name == "Hold" else local_gates[name],
                    "netOfWaitValue": round(values["meanGain"] - (0 if name == "Hold" else wait_value), 1),
                }
                for name, values in local_distributions.items()
            ],
        }

    manager_plans = {}
    season_state_root = SHADOW_ROOT / status["season"]
    for manager_id in ("structural-control", "structural-scenarios", "frontier-challenger"):
        state_path = season_state_root / manager_id / "state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        squad = [id_to_index[player_id] for player_id in state["squadIds"] if player_id in id_to_index]
        if len(squad) == 15:
            manager_plans[manager_id] = saved_manager_plan(squad)
    output = {
        "schemaVersion": 1,
        "season": status["season"],
        "gameweek": gameweek,
        "snapshotHash": status["snapshotHash"],
        "snapshotStatus": status["status"],
        "simulationCount": DRAWS,
        "recommendation": recommendation,
        "managerPlans": manager_plans,
        "optionValueOfWaiting": round(wait_value, 1),
        "currentStructure": {
            "blankPlayers": blank_players,
            "doublePlayers": double_players,
            "benchDoublePlayers": bench_doubles,
            "captainHasDouble": captain_double,
        },
        "scenarios": [
            {
                "chip": name,
                **values,
                "gatePassed": True if name == "Hold" else gates[name],
                "netOfWaitValue": round(values["meanGain"] - (0 if name == "Hold" else wait_value), 1),
            }
            for name, values in distributions.items()
        ],
        "futureScheduleSignals": future_events,
        "squads": {
            "current": [int(players[index]["id"]) for index in current],
            "freeHit": [int(players[index]["id"]) for index in fresh],
            "wildcard": [int(players[index]["id"]) for index in wildcard],
        },
        "captains": {
            "current": players[current_captain]["name"],
            "freeHit": players[fresh_captain]["name"],
            "wildcard": players[wildcard_captain]["name"],
        },
        "method": "Paired 5,000-draw Monte Carlo with common player outcomes, correlated team shocks, explicit downside gates, schedule structure, and the estimated option value of saving each chip.",
        "warning": "This is a frozen policy diagnostic, not a promise of rank. A chip is recommended only when both structural and downside gates pass.",
    }
    atomic_json(APP_DATA / "chip-scenarios.json", output)
    print(json.dumps({"recommendation": recommendation, "scenarios": output["scenarios"]}, indent=2))


if __name__ == "__main__":
    main()

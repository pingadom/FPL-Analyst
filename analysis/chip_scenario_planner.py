"""Manager-specific, paired Monte Carlo chip decisions with option value.

Triple Captain and Bench Boost are evaluated as marginal scoring changes on the
current squad.  Free Hit receives its own one-week squad and never changes the
recursive squad state.  Wildcard is valued over the planning horizon and uses
the manager's actual selling budget.  Every chip is tracked by half-season set,
with one-chip-per-Gameweek and expiry collision constraints.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from prospective_common import (
    APP_DATA,
    ROOT,
    SHADOW_ROOT,
    atomic_json,
    available_squad_budget,
    chip_inventory_key,
    chip_set_for_gameweek,
    normal_scenarios,
    optimise_squad,
    selling_price,
    used_chip_keys,
)


DRAWS = 5_000
SEED = 271828
CHIPS = ("Wildcard", "Free Hit", "Bench Boost", "Triple Captain")


def summary(values: np.ndarray) -> dict:
    return {
        "meanGain": round(float(values.mean()), 1),
        "p10Gain": round(float(np.quantile(values, 0.10)), 1),
        "medianGain": round(float(np.median(values)), 1),
        "p90Gain": round(float(np.quantile(values, 0.90)), 1),
        "probabilityPositive": round(float((values > 0).mean() * 100), 1),
    }


def best_lineup(
    players: list[dict], squad: list[int], score_key: str
) -> tuple[list[int], int, int]:
    options = [
        {
            "GK": 1,
            "DEF": defenders,
            "MID": 10 - defenders - forwards,
            "FWD": forwards,
        }
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
                    [
                        candidate
                        for candidate in squad
                        if players[candidate]["position"] == position
                    ],
                    key=lambda candidate: float(players[candidate][score_key]),
                    reverse=True,
                )[:count]
            ]
            for formation in options
        ),
        key=lambda selection: sum(
            float(players[index][score_key]) for index in selection
        )
        if len(selection) == 11
        else -1e9,
    )
    captain = max(xi, key=lambda index: float(players[index][score_key]))
    vice = max(
        (index for index in xi if index != captain),
        key=lambda index: float(players[index][score_key]),
    )
    return xi, captain, vice


def ordered_bench(
    players: list[dict], squad: list[int], xi: list[int], score_key: str
) -> list[int]:
    bench = [index for index in squad if index not in xi]
    goalkeeper = next(index for index in bench if players[index]["position"] == "GK")
    outfield = sorted(
        [index for index in bench if index != goalkeeper],
        key=lambda index: float(players[index][score_key]),
        reverse=True,
    )
    return [goalkeeper, *outfield]


def formation_is_legal(counts: Counter) -> bool:
    return (
        counts["GK"] == 1
        and 3 <= counts["DEF"] <= 5
        and 2 <= counts["MID"] <= 5
        and 1 <= counts["FWD"] <= 3
    )


def captain_extra(
    draws: np.ndarray,
    appearances: np.ndarray,
    captain: int,
    vice: int,
) -> np.ndarray:
    return np.where(
        appearances[:, captain],
        draws[:, captain],
        np.where(appearances[:, vice], draws[:, vice], 0.0),
    )


def normal_team_score(
    players: list[dict],
    draws: np.ndarray,
    appearances: np.ndarray,
    squad: list[int],
    xi: list[int],
    captain: int,
    vice: int,
    score_key: str,
) -> np.ndarray:
    """Score an XI with captain fallback and FPL-compatible automatic subs."""
    result = np.zeros(draws.shape[0], dtype=float)
    bench = ordered_bench(players, squad, xi, score_key)
    starting_goalkeeper = next(
        index for index in xi if players[index]["position"] == "GK"
    )
    bench_goalkeeper = bench[0]
    outfield_bench = bench[1:]
    for scenario in range(draws.shape[0]):
        active = [index for index in xi if appearances[scenario, index]]
        points = float(draws[scenario, active].sum())
        if (
            not appearances[scenario, starting_goalkeeper]
            and appearances[scenario, bench_goalkeeper]
        ):
            points += float(draws[scenario, bench_goalkeeper])
        absent = [
            index
            for index in xi
            if players[index]["position"] != "GK"
            and not appearances[scenario, index]
        ]
        nominal = Counter(players[index]["position"] for index in xi)
        for substitute in outfield_bench:
            if not absent or not appearances[scenario, substitute]:
                continue
            for outgoing in list(absent):
                candidate = nominal.copy()
                candidate[players[outgoing]["position"]] -= 1
                candidate[players[substitute]["position"]] += 1
                if formation_is_legal(candidate):
                    nominal = candidate
                    absent.remove(outgoing)
                    points += float(draws[scenario, substitute])
                    break
        result[scenario] = points
    return result + captain_extra(draws, appearances, captain, vice)


def appearance_scenarios(
    players: list[dict], draws_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-part point draws preserve explicit no-show and autosub risk."""
    means = np.asarray([float(row["structuralScore"]) for row in players])
    stds = np.asarray(
        [
            max(
                0.6,
                float(row.get("distribution", {}).get("standardDeviation", 0.0))
                or (
                    float(row["distribution"]["p90"])
                    - float(row["distribution"]["p10"])
                )
                / 2.5632,
            )
            for row in players
        ]
    )
    play_probability = []
    for row in players:
        original = max(float(row.get("expectedMinutes", 0.0)), 8.0)
        minute_ratio = np.clip(float(row["deadlineMinutes"]) / original, 0.0, 1.25)
        stated = float(row.get("minutesModel", {}).get("playProbability", 0.0)) / 100
        if stated <= 0:
            stated = np.clip(float(row["deadlineMinutes"]) / 75, 0.0, 0.98)
        play_probability.append(float(np.clip(stated * minute_ratio, 0.0, 0.995)))
    probability = np.asarray(play_probability)
    rng = np.random.default_rng(seed)
    appearances = rng.random((draws_count, len(players))) < probability
    draws = np.zeros((draws_count, len(players)), dtype=float)
    for index, (mean, std, play) in enumerate(zip(means, stds, probability)):
        if mean <= 0 or play <= 0:
            continue
        conditional_mean = mean / play
        conditional_variance = max(
            0.05,
            (std * std - play * (1 - play) * conditional_mean * conditional_mean)
            / play,
        )
        shape = max(0.25, conditional_mean * conditional_mean / conditional_variance)
        scale = max(0.01, conditional_variance / conditional_mean)
        draws[:, index] = (
            rng.gamma(shape, scale, size=draws_count) * appearances[:, index]
        )
    # A multiplicative club shock adds realistic within-team dependence while
    # retaining exact zeroes for non-appearances.
    for team in sorted({str(row["team"]) for row in players}):
        indices = [index for index, row in enumerate(players) if str(row["team"]) == team]
        shock = np.exp(rng.normal(-0.5 * 0.14**2, 0.14, size=(draws_count, 1)))
        draws[:, indices] *= shock
    return draws, appearances


def scenario_score(draws: np.ndarray, xi: list[int], captain: int) -> np.ndarray:
    return draws[:, xi].sum(axis=1) + draws[:, captain]


def fixture_structure(
    snapshot: dict, gameweek: int
) -> tuple[dict[int, Counter], dict[int, str]]:
    teams = {int(row["id"]): row["short_name"] for row in snapshot["official"]["teams"]}
    counts: dict[int, Counter] = defaultdict(Counter)
    for fixture in snapshot["official"]["fixtures"]:
        event = fixture.get("event")
        if event is None or not gameweek <= int(event) <= gameweek + 7:
            continue
        counts[int(event)][int(fixture["team_h"])] += 1
        counts[int(event)][int(fixture["team_a"])] += 1
    return counts, teams


def reservation_calibration() -> dict[str, dict[str, float]]:
    path = ROOT / "analysis" / "data" / "sequential_chip_value_validation.json"
    if not path.exists():
        return {
            "Triple Captain": {"q25": 8.6, "q50": 9.3, "q65": 9.8},
            "Bench Boost": {"q25": 3.4, "q50": 4.9, "q65": 5.8},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("historicalReservationCalibration", {})


def free_hit_live_calibration() -> dict | None:
    path = ROOT / "analysis" / "data" / "freehit_live_calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def calibrated_free_hit_value(calibration: dict | None, features: dict) -> float | None:
    if not calibration:
        return None
    names = calibration.get("features", [])
    try:
        values = np.asarray([float(features[name]) for name in names], dtype=float)
        centre = np.asarray(calibration["featureMean"], dtype=float)
        scale = np.asarray(calibration["featureScale"], dtype=float)
        coefficients = np.asarray(calibration["coefficients"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if not (len(values) == len(centre) == len(scale) == len(coefficients)):
        return None
    standardized = (values - centre) / np.where(scale > 0, scale, 1.0)
    return float(calibration["intercept"] + standardized @ coefficients)


def best_permanent_transfer_value(
    players: list[dict], squad: list[int], state: dict | None
) -> float:
    """Approximate the horizon value FH displaces before any transfer is made."""
    if state is None:
        return 0.0
    active = list(squad)
    bank = float(state.get("bank", 0.0))
    purchase = {
        int(key): float(value)
        for key, value in state.get("purchasePrices", {}).items()
    }
    available = min(int(state.get("freeTransfers", 1)), 2)
    total = 0.0
    for _ in range(available):
        clubs = Counter(str(players[index]["team"]) for index in active)
        best: tuple[float, int, int, float] | None = None
        for outgoing in active:
            outgoing_id = int(players[outgoing]["id"])
            sale = selling_price(
                float(players[outgoing]["price"]),
                purchase.get(outgoing_id, float(players[outgoing]["price"])),
            )
            for incoming, candidate in enumerate(players):
                if incoming in active:
                    continue
                if candidate["position"] != players[outgoing]["position"]:
                    continue
                if float(candidate["price"]) > bank + sale + 1e-6:
                    continue
                if (
                    str(candidate["team"]) != str(players[outgoing]["team"])
                    and clubs[str(candidate["team"])] >= 3
                ):
                    continue
                gain = float(candidate["horizonScore"]) - float(
                    players[outgoing]["horizonScore"]
                )
                proposal = (gain, outgoing, incoming, sale)
                if best is None or proposal[0] > best[0]:
                    best = proposal
        if best is None or best[0] < 2.2:
            break
        gain, outgoing, incoming, sale = best
        bank = round(bank + sale - float(players[incoming]["price"]), 1)
        active[active.index(outgoing)] = incoming
        purchase[int(players[incoming]["id"])] = float(players[incoming]["price"])
        total += gain
    return float(total)


def empirical_reservation(
    calibration: dict[str, dict[str, float]],
    chip: str,
    slack_fraction: float,
) -> float:
    values = calibration.get(chip, {})
    pairs = sorted(
        (int(key[1:]) / 100, float(value))
        for key, value in values.items()
        if key.startswith("q")
    )
    if not pairs:
        return 0.0
    target_quantile = 0.65 * max(0.0, slack_fraction)
    quantiles = np.asarray([0.0, *[pair[0] for pair in pairs]], dtype=float)
    scores = np.asarray([0.70 * pairs[0][1], *[pair[1] for pair in pairs]], dtype=float)
    return float(np.interp(target_quantile, quantiles, scores))


def chip_reservations(
    gameweek: int,
    remaining_chips: list[str],
    calibration: dict[str, dict[str, float]],
) -> tuple[dict[str, float], bool, int]:
    start, end = (1, 19) if chip_set_for_gameweek(gameweek) == 1 else (20, 38)
    weeks_left = end - gameweek + 1
    forced_by_collision = weeks_left <= len(remaining_chips)
    slack = max(0, weeks_left - len(remaining_chips))
    slack_fraction = slack / max(1, end - start + 1)
    reservations = {
        "Triple Captain": empirical_reservation(
            calibration, "Triple Captain", slack_fraction
        ),
        "Bench Boost": empirical_reservation(
            calibration, "Bench Boost", slack_fraction
        ),
        "Free Hit": 3.0 + 7.0 * slack_fraction,
        "Wildcard": 8.0 + 17.0 * slack_fraction,
    }
    if forced_by_collision:
        reservations = {chip: 0.0 for chip in reservations}
    return reservations, forced_by_collision, weeks_left


def manager_inventory(state: dict | None, gameweek: int) -> list[str]:
    used = used_chip_keys(state)
    return [
        chip
        for chip in CHIPS
        if chip_inventory_key(chip, gameweek) not in used
    ]


def currently_available_chips(
    state: dict | None, gameweek: int, remaining: list[str]
) -> list[str]:
    available = list(remaining)
    used = used_chip_keys(state)
    if gameweek == 1:
        # Both transfer chips are unavailable in the opening Gameweek because
        # the initial squad already has unlimited transfers.
        available = [
            chip for chip in available if chip not in {"Free Hit", "Wildcard"}
        ]
    if (
        gameweek == 20
        and chip_inventory_key("Free Hit", 19) in used
        and "Free Hit" in available
    ):
        available.remove("Free Hit")
    return available


def build_alternative_squads(
    players: list[dict], budget: float
) -> tuple[list[int], list[int], int, list[int], list[int], int]:
    minimum_spend = max(0.0, budget - 0.5)
    free_hit, free_hit_xi, free_hit_captain, _ = optimise_squad(
        players,
        "structuralScore",
        bench_weight=0.0,
        captain_weight=1.0,
        bench_premium_limit=0.8,
        minimum_spend=minimum_spend,
        budget_limit=budget,
    )
    wildcard, wildcard_xi, wildcard_captain, _ = optimise_squad(
        players,
        "horizonScore",
        minimum_spend=minimum_spend,
        budget_limit=budget,
    )
    return (
        free_hit,
        free_hit_xi,
        free_hit_captain,
        wildcard,
        wildcard_xi,
        wildcard_captain,
    )


def evaluate_squad(
    players: list[dict],
    squad: list[int],
    state: dict | None,
    gameweek: int,
    draws: np.ndarray,
    appearances: np.ndarray,
    horizon_draws: np.ndarray,
    current_counts: Counter,
    name_to_team: dict[str, int],
    calibration: dict[str, dict[str, float]],
    free_hit_model: dict | None,
) -> dict:
    xi, captain, vice = best_lineup(players, squad, "structuralScore")
    budget = available_squad_budget(players, state)
    (
        free_hit,
        free_hit_xi,
        free_hit_captain,
        wildcard,
        wildcard_xi,
        wildcard_captain,
    ) = build_alternative_squads(players, budget)
    _, free_hit_vice = (
        free_hit_captain,
        max(
            (index for index in free_hit_xi if index != free_hit_captain),
            key=lambda index: float(players[index]["structuralScore"]),
        ),
    )
    base = normal_team_score(
        players,
        draws,
        appearances,
        squad,
        xi,
        captain,
        vice,
        "structuralScore",
    )
    free_hit_points = normal_team_score(
        players,
        draws,
        appearances,
        free_hit,
        free_hit_xi,
        free_hit_captain,
        free_hit_vice,
        "structuralScore",
    )
    bench_boost_points = draws[:, squad].sum(axis=1) + captain_extra(
        draws, appearances, captain, vice
    )
    triple_gain = captain_extra(draws, appearances, captain, vice)
    horizon_xi, horizon_captain, _ = best_lineup(players, squad, "horizonScore")
    wildcard_gain = scenario_score(
        horizon_draws, wildcard_xi, wildcard_captain
    ) - scenario_score(horizon_draws, horizon_xi, horizon_captain)
    distributions = {
        "Hold": summary(np.zeros(DRAWS)),
        "Free Hit": summary(free_hit_points - base),
        "Bench Boost": summary(bench_boost_points - base),
        "Triple Captain": summary(triple_gain),
        "Wildcard": summary(wildcard_gain),
    }

    bench = ordered_bench(players, squad, xi, "structuralScore")
    fixture_count = lambda index: current_counts.get(
        name_to_team.get(str(players[index]["team"]), -1), 0
    )
    blank_players = sum(fixture_count(index) == 0 for index in squad)
    double_players = sum(fixture_count(index) >= 2 for index in squad)
    bench_doubles = sum(fixture_count(index) >= 2 for index in bench)
    captain_double = fixture_count(captain) >= 2
    bench_minutes = sum(float(players[index]["deadlineMinutes"]) for index in bench)
    bench_play = [
        float(players[index].get("minutesModel", {}).get("playProbability", 0.0))
        / 100
        for index in bench
    ]
    captain_row = players[captain]
    captain_return = (
        float(captain_row.get("distribution", {}).get("return5Probability", 0.0))
        / 100
    )
    premium_single = (
        not captain_double
        and float(captain_row["price"]) >= 11.0
        and float(captain_row["deadlineMinutes"]) >= 65
        and captain_return >= 0.35
    )
    free_hit_double_count = sum(fixture_count(index) >= 2 for index in free_hit_xi)
    current_lineup_value = sum(
        float(players[index]["structuralScore"]) for index in xi
    ) + float(players[captain]["structuralScore"])
    free_hit_lineup_value = sum(
        float(players[index]["structuralScore"]) for index in free_hit_xi
    ) + float(players[free_hit_captain]["structuralScore"])
    free_hit_immediate_signal = (
        free_hit_lineup_value
        - current_lineup_value
        + 0.22 * max(0, blank_players - 1)
        + 0.12 * free_hit_double_count
    )
    permanent_transfer_value = best_permanent_transfer_value(
        players, squad, state
    )
    half_start = 1 if chip_set_for_gameweek(gameweek) == 1 else 20
    free_hit_features = {
        "predictedFreeHitImmediateGain": free_hit_immediate_signal,
        "predictedFreeHitGain": free_hit_immediate_signal
        - permanent_transfer_value,
        "permanentTransferValueForegone": permanent_transfer_value,
        "freeHitBlankCount": blank_players,
        "freeHitDoubleCount": free_hit_double_count,
        "freeHitLineupOverlap": len(set(free_hit_xi).intersection(xi)),
        "currentLineupValue": current_lineup_value,
        "freeHitLineupValue": free_hit_lineup_value,
        "currentLineupExpectedMinutes": sum(
            float(players[index]["deadlineMinutes"]) for index in xi
        ),
        "freeHitLineupExpectedMinutes": sum(
            float(players[index]["deadlineMinutes"]) for index in free_hit_xi
        ),
        "currentLineupUncertainty": sum(
            float(players[index].get("uncertainty", 0.5)) for index in xi
        ),
        "freeHitLineupUncertainty": sum(
            float(players[index].get("uncertainty", 0.5))
            for index in free_hit_xi
        ),
        "windowProgress": (gameweek - half_start) / 18,
    }
    learned_free_hit_immediate = calibrated_free_hit_value(
        free_hit_model, free_hit_features
    )
    learned_free_hit_net = (
        learned_free_hit_immediate - permanent_transfer_value
        if learned_free_hit_immediate is not None
        else None
    )

    remaining_chips = manager_inventory(state, gameweek)
    available_chips = currently_available_chips(state, gameweek, remaining_chips)
    reservations, forced, weeks_left = chip_reservations(
        gameweek, remaining_chips, calibration
    )
    risk_adjusted = {
        chip: distributions[chip]["meanGain"]
        - 0.30
        * (distributions[chip]["meanGain"] - distributions[chip]["p10Gain"])
        for chip in CHIPS
    }
    reliable_bench = (
        bench_minutes >= 250
        and float(np.mean(bench_play)) >= 0.72
        and min(bench_play, default=0.0) >= 0.45
    )
    wildcard_review = (
        gameweek > 1
        and distributions["Wildcard"]["meanGain"]
        >= reservations["Wildcard"]
        and distributions["Wildcard"]["p10Gain"] >= 0
        and distributions["Wildcard"]["probabilityPositive"] >= 70
    )
    gates = {
        "Free Hit": (
            gameweek > 1
            and learned_free_hit_net is not None
            and learned_free_hit_net
            >= float((free_hit_model or {}).get("activationThreshold", 3.0))
            and distributions["Free Hit"]["meanGain"]
            >= reservations["Free Hit"]
            and distributions["Free Hit"]["p10Gain"] >= -2
            and distributions["Free Hit"]["probabilityPositive"] >= 65
            and (
                blank_players >= 3
                or distributions["Free Hit"]["meanGain"]
                >= reservations["Free Hit"] + 3
            )
        ),
        "Bench Boost": (
            distributions["Bench Boost"]["meanGain"]
            >= reservations["Bench Boost"]
            and distributions["Bench Boost"]["p10Gain"] >= 0
            and (
                bench_doubles >= 1
                or reliable_bench
                or distributions["Bench Boost"]["meanGain"]
                >= reservations["Bench Boost"] + 3
            )
        ),
        "Triple Captain": (
            distributions["Triple Captain"]["meanGain"]
            >= reservations["Triple Captain"]
            and distributions["Triple Captain"]["p10Gain"] >= 2
            and (
                captain_double
                or premium_single
                or distributions["Triple Captain"]["meanGain"]
                >= reservations["Triple Captain"] + 2
            )
        ),
        # Historical automatic WC policies all lost out of sample. Preserve the
        # signal for a human review, but never auto-activate unless expiry makes
        # holding strictly dominated in the collision override below.
        "Wildcard": False,
    }
    for chip in CHIPS:
        if chip not in available_chips:
            gates[chip] = False
        elif forced and ((chip != "Free Hit" or gameweek > 1) and (chip != "Wildcard" or gameweek > 1)):
            gates[chip] = risk_adjusted[chip] > 0
    eligible = [chip for chip in CHIPS if gates[chip]]
    recommendation = "Hold"
    if eligible:
        recommendation = max(
            eligible,
            key=lambda chip: risk_adjusted[chip] - reservations[chip],
        )
        if (
            not forced
            and risk_adjusted[recommendation] <= reservations[recommendation]
        ):
            recommendation = "Hold"

    return {
        "recommendation": recommendation,
        "chipSet": chip_set_for_gameweek(gameweek),
        "chipsRemaining": remaining_chips,
        "weeksUntilSetExpiry": weeks_left,
        "forcedByExpiryCollision": forced,
        "currentStructure": {
            "blankPlayers": blank_players,
            "doublePlayers": double_players,
            "benchDoublePlayers": bench_doubles,
            "captainHasDouble": captain_double,
            "captainIsStrongPremiumSingle": premium_single,
            "benchExpectedMinutes": round(bench_minutes, 1),
            "benchMeanPlayProbability": round(float(np.mean(bench_play)) * 100, 1),
            "benchMinimumPlayProbability": round(min(bench_play, default=0.0) * 100, 1),
        },
        "scenarios": [
            {
                "chip": name,
                **values,
                "gatePassed": True if name == "Hold" else gates[name],
                "available": True if name == "Hold" else name in available_chips,
                "reservationValue": round(
                    0.0 if name == "Hold" else reservations[name], 1
                ),
                "riskAdjustedGain": round(
                    0.0 if name == "Hold" else risk_adjusted[name], 1
                ),
                "netOfReservation": round(
                    0.0
                    if name == "Hold"
                    else risk_adjusted[name] - reservations[name],
                    1,
                ),
                "learnedExpectedGain": (
                    round(learned_free_hit_net, 1)
                    if name == "Free Hit" and learned_free_hit_net is not None
                    else None
                ),
                "permanentTransferValueForegone": (
                    round(permanent_transfer_value, 1)
                    if name == "Free Hit"
                    else None
                ),
                "manualReviewTriggered": (
                    wildcard_review if name == "Wildcard" else False
                ),
            }
            for name, values in distributions.items()
        ],
        "squads": {
            "current": [int(players[index]["id"]) for index in squad],
            "freeHit": [int(players[index]["id"]) for index in free_hit],
            "wildcard": [int(players[index]["id"]) for index in wildcard],
        },
        "captains": {
            "current": players[captain]["name"],
            "freeHit": players[free_hit_captain]["name"],
            "wildcard": players[wildcard_captain]["name"],
        },
        "availableBudget": budget,
    }


def main() -> None:
    status = json.loads((APP_DATA / "deadline-status.json").read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / status["snapshotPath"]).read_text(encoding="utf-8"))
    player_rows = json.loads((APP_DATA / "current-players.json").read_text(encoding="utf-8"))
    model_results = json.loads((APP_DATA / "model-results.json").read_text(encoding="utf-8"))
    frontier_path = APP_DATA / "frontier-scores.json"
    frontier = {}
    if frontier_path.exists():
        frontier = {
            int(row["id"]): row
            for row in json.loads(frontier_path.read_text(encoding="utf-8"))["players"]
        }
    intelligence = {int(row["id"]): row for row in snapshot["deadlineIntelligence"]}
    players: list[dict] = []
    for row in player_rows:
        intel = intelligence.get(int(row["id"]), {})
        old_minutes = max(float(row["expectedMinutes"]), 8)
        new_minutes = float(intel.get("expectedMinutes", old_minutes))
        minute_ratio = np.clip(new_minutes / old_minutes, 0.20, 1.25)
        adjusted = float(row["projected"]) * (0.25 + 0.75 * minute_ratio)
        horizon_adjusted = float(row["sixWeekProjected"]) * (
            0.75 + 0.25 * minute_ratio
        )
        challenger = frontier.get(int(row["id"]), {})
        players.append(
            {
                **row,
                "structuralScore": float(adjusted),
                "horizonScore": float(horizon_adjusted),
                "frontierScore": float(challenger.get("blend25", adjusted))
                * (0.25 + 0.75 * minute_ratio),
                "deadlineMinutes": new_minutes,
            }
        )
    id_to_index = {int(row["id"]): index for index, row in enumerate(players)}
    current = [
        id_to_index[int(row["id"])]
        for row in model_results["squad"]
        if int(row["id"]) in id_to_index
    ]
    if len(current) != 15:
        raise RuntimeError("The frozen current squad could not be reconstructed.")

    draws, appearances = appearance_scenarios(players, DRAWS, SEED)
    horizon_mean = np.asarray([row["horizonScore"] for row in players])
    horizon_std = np.asarray(
        [
            max(
                0.6,
                (
                    float(row["distribution"]["p90"])
                    - float(row["distribution"]["p10"])
                )
                / 2.5632,
            )
            * np.sqrt(4.5)
            for row in players
        ]
    )
    horizon_draws = normal_scenarios(
        horizon_mean, horizon_std, DRAWS, SEED + 2
    )
    gameweek = int(status["gameweek"])
    counts, team_names = fixture_structure(snapshot, gameweek)
    name_to_team = {value: key for key, value in team_names.items()}
    current_counts = counts.get(gameweek, Counter())
    calibration = reservation_calibration()
    free_hit_model = free_hit_live_calibration()
    plan = evaluate_squad(
        players,
        current,
        None,
        gameweek,
        draws,
        appearances,
        horizon_draws,
        current_counts,
        name_to_team,
        calibration,
        free_hit_model,
    )

    future_events = []
    for event in range(gameweek + 1, min(38, gameweek + 7) + 1):
        event_counts = counts.get(event, Counter())
        future_events.append(
            {
                "gameweek": event,
                "blankTeams": sorted(
                    team_names[team]
                    for team in team_names
                    if event_counts.get(team, 0) == 0
                ),
                "doubleTeams": sorted(
                    team_names[team]
                    for team, count in event_counts.items()
                    if count >= 2
                ),
            }
        )

    manager_plans = {}
    season_state_root = SHADOW_ROOT / status["season"]
    for manager_id in (
        "structural-control",
        "structural-scenarios",
        "frontier-challenger",
        "captain-route-consensus",
    ):
        state_path = season_state_root / manager_id / "state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        squad = [
            id_to_index[player_id]
            for player_id in state["squadIds"]
            if player_id in id_to_index
        ]
        if len(squad) == 15:
            manager_plans[manager_id] = evaluate_squad(
                players,
                squad,
                state,
                gameweek,
                draws,
                appearances,
                horizon_draws,
                current_counts,
                name_to_team,
                calibration,
                free_hit_model,
            )

    output = {
        "schemaVersion": 2,
        "season": status["season"],
        "gameweek": gameweek,
        "snapshotHash": status["snapshotHash"],
        "snapshotStatus": status["status"],
        "simulationCount": DRAWS,
        **plan,
        "managerPlans": manager_plans,
        "futureScheduleSignals": future_events,
        "reservationCalibration": calibration,
        "freeHitCalibration": {
            key: free_hit_model[key]
            for key in (
                "status",
                "trainedThrough",
                "trainingRows",
                "activationThreshold",
                "causalValidation",
            )
        }
        if free_hit_model
        else None,
        "method": (
            "Paired 5,000-draw two-part Monte Carlo with explicit no-shows, "
            "autosubs, captain-to-vice fallback, club shocks, manager-specific "
            "selling budgets and chip-specific reservation values. TC/BB are "
            "marginal gains; FH is temporary; WC is valued over the horizon."
        ),
        "warning": (
            "Historical TC/BB reservation values are causal but modestly stable. "
            "Wildcard and Free Hit retain conservative gates until their paired "
            "recursive challengers beat the no-chip control out of sample."
        ),
    }
    atomic_json(APP_DATA / "chip-scenarios.json", output)
    print(
        json.dumps(
            {
                "recommendation": plan["recommendation"],
                "chipSet": plan["chipSet"],
                "chipsRemaining": plan["chipsRemaining"],
                "scenarios": plan["scenarios"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

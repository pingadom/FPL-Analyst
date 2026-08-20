"""Run four frozen prospective managers from the same deadline snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from breakthrough_engine import (
    DEFAULT_FIELDABILITY_POLICY,
    fieldability_audit,
    hard_unavailable,
    play_probability,
    premium_access_diagnostic,
)

from prospective_common import (
    APP_DATA,
    ROOT,
    SHADOW_ROOT,
    available_squad_budget,
    atomic_json,
    chip_inventory_key,
    optimise_squad,
    payload_hash,
    selling_price,
    used_chip_keys,
)


MANAGERS = (
    {
        "id": "structural-control",
        "name": "Structural control",
        "planScore": "horizonScore",
        "lineupScore": "immediateForCaptain",
        "captainScore": "immediateForCaptain",
        "chips": False,
        "transferHurdle": 2.8,
        "description": "Frozen Lens structural projections; chips disabled.",
    },
    {
        "id": "structural-scenarios",
        "name": "Structural + scenarios",
        "planScore": "riskScore",
        "lineupScore": "immediateForCaptain",
        "captainScore": "immediateForCaptain",
        "chips": True,
        "transferHurdle": 2.2,
        "description": "Same projections with downside-aware chip gates and modest upside utility.",
    },
    {
        "id": "frontier-challenger",
        "name": "Hybrid challenger",
        "planScore": "listwiseHorizonScore",
        "lineupScore": "frontierImmediateScore",
        "captainScore": "listwiseCaptainScore",
        "chips": True,
        "transferHurdle": 2.2,
        "description": "Frontier next-GW, listwise six-week and captain reranks at frozen blends.",
    },
    {
        "id": "captain-route-consensus",
        "name": "Scoring-route captain",
        "planScore": "listwiseHorizonScore",
        "lineupScore": "frontierImmediateScore",
        "captainScore": "routeCaptainScore",
        "chips": True,
        "transferHurdle": 2.2,
        "description": "Same squad model as the hybrid challenger; captaincy adds a five-seed scoring-route consensus and conservative defender tie-break.",
    },
    {
        "id": "breakthrough-decision",
        "name": "Breakthrough decision",
        "planScore": "listwiseHorizonScore",
        "lineupScore": "frontierImmediateScore",
        "captainScore": "routeCaptainScore",
        "chips": True,
        "transferHurdle": 2.2,
        "fieldability": True,
        "description": "Fieldability-constrained squad, XI and captain with conservative emergency repairs and the leak-free route captain.",
    },
    {
        "id": "forecast-breakthrough-v2",
        "name": "Forecast breakthrough v2",
        "planScore": "listwiseHorizonScore",
        "lineupScore": "frontierImmediateScore",
        "captainScore": "forecastV2CaptainScore",
        "chips": True,
        "transferHurdle": 2.2,
        "description": "Frozen large-search winner: route captain plus a 20% exact-capture dynamic match boundary, causal minutes downside protection and separately gated chip policy; falls back safely when market data is unavailable.",
    },
)


def legal(players: list[dict], squad: list[int], budget: float) -> bool:
    if len(squad) != 15 or len(set(squad)) != 15:
        return False
    quota = Counter(players[index]["position"] for index in squad)
    clubs = Counter(players[index]["team"] for index in squad)
    return (
        quota == Counter({"GK": 2, "DEF": 5, "MID": 5, "FWD": 3})
        and max(clubs.values()) <= 3
        and sum(float(players[index]["price"]) for index in squad) <= budget + 1e-6
    )


def best_lineup(
    players: list[dict],
    squad: list[int],
    score_key: str,
    captain_key: str,
    enforce_fieldability: bool = False,
) -> tuple[list[int], list[int], int, int]:
    def lineup_value(index: int) -> float:
        if enforce_fieldability and hard_unavailable(players[index]):
            return -1e9
        value = float(players[index][score_key])
        if enforce_fieldability:
            value -= DEFAULT_FIELDABILITY_POLICY.risk_penalty * (
                1.0 - play_probability(players[index])
            )
        return value

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
                    key=lineup_value,
                    reverse=True,
                )[:count]
            ]
            for formation in options
        ),
        key=lambda selection: sum(lineup_value(index) for index in selection)
        if len(selection) == 11 else -1e9,
    )
    captain_pool = [
        index
        for index in xi
        if not enforce_fieldability
        or play_probability(players[index])
        >= DEFAULT_FIELDABILITY_POLICY.captain_min_play_probability
    ] or xi
    captain = max(captain_pool, key=lambda index: float(players[index][captain_key]))
    vice_pool = [
        index
        for index in xi
        if index != captain
        and (
            not enforce_fieldability
            or play_probability(players[index])
            >= DEFAULT_FIELDABILITY_POLICY.vice_min_play_probability
        )
    ] or [index for index in xi if index != captain]
    vice = max(
        vice_pool,
        key=lambda index: float(players[index][captain_key]),
    )
    bench = sorted(
        [index for index in squad if index not in xi],
        key=lambda index: (
            players[index]["position"] != "GK",
            play_probability(players[index]) if enforce_fieldability else 1.0,
            float(players[index][score_key]),
        ),
        reverse=True,
    )
    goalkeeper = next(index for index in bench if players[index]["position"] == "GK")
    bench = [goalkeeper] + [index for index in bench if index != goalkeeper]
    return xi, bench, captain, vice


def transfer_plan(
    players: list[dict],
    state: dict,
    score_key: str,
    hurdle: float,
    enforce_fieldability: bool = False,
    captain_key: str | None = None,
    package_search: bool = False,
) -> tuple[list[int], list[dict], float, int]:
    by_id = {int(row["id"]): index for index, row in enumerate(players)}
    squad = [by_id[player_id] for player_id in state["squadIds"] if player_id in by_id]
    purchase = {int(key): float(value) for key, value in state["purchasePrices"].items()}
    bank = float(state["bank"])
    transfers: list[dict] = []
    available = min(int(state.get("freeTransfers", 1)), 2)
    if package_search and available >= 2:
        current_captain = max(
            (float(players[index][captain_key]) for index in squad),
            default=0.0,
        ) if captain_key else 0.0
        incoming_by_position = {
            position: sorted(
                [
                    index
                    for index, row in enumerate(players)
                    if row["position"] == position
                    and index not in squad
                    and (not enforce_fieldability or not hard_unavailable(row))
                ],
                key=lambda index: float(players[index][score_key]),
                reverse=True,
            )[:10]
            for position in ("GK", "DEF", "MID", "FWD")
        }
        best_package = None
        for first_offset, first_out in enumerate(squad):
            for second_out in squad[first_offset + 1 :]:
                first_position = players[first_out]["position"]
                second_position = players[second_out]["position"]
                first_id = int(players[first_out]["id"])
                second_id = int(players[second_out]["id"])
                first_sale = selling_price(
                    float(players[first_out]["price"]),
                    purchase.get(first_id, float(players[first_out]["price"])),
                )
                second_sale = selling_price(
                    float(players[second_out]["price"]),
                    purchase.get(second_id, float(players[second_out]["price"])),
                )
                for first_in in incoming_by_position[first_position]:
                    for second_in in incoming_by_position[second_position]:
                        if first_in == second_in:
                            continue
                        candidate = [
                            index
                            for index in squad
                            if index not in (first_out, second_out)
                        ] + [first_in, second_in]
                        candidate_bank = round(
                            bank
                            + first_sale
                            + second_sale
                            - float(players[first_in]["price"])
                            - float(players[second_in]["price"]),
                            1,
                        )
                        if candidate_bank < -1e-6:
                            continue
                        if not legal(
                            players,
                            candidate,
                            sum(float(players[index]["price"]) for index in candidate)
                            + candidate_bank,
                        ):
                            continue
                        raw_gain = (
                            float(players[first_in][score_key])
                            + float(players[second_in][score_key])
                            - float(players[first_out][score_key])
                            - float(players[second_out][score_key])
                        )
                        fieldability_gain = (
                            play_probability(players[first_in])
                            + play_probability(players[second_in])
                            - play_probability(players[first_out])
                            - play_probability(players[second_out])
                        )
                        new_captain = max(
                            (float(players[index][captain_key]) for index in candidate),
                            default=0.0,
                        ) if captain_key else 0.0
                        captain_option = max(0.0, new_captain - current_captain)
                        utility = (
                            raw_gain
                            + (4.0 * fieldability_gain if enforce_fieldability else 0.0)
                            + 0.70 * captain_option
                        )
                        proposal = (
                            utility,
                            raw_gain,
                            fieldability_gain,
                            captain_option,
                            (first_out, second_out),
                            (first_in, second_in),
                            candidate,
                            candidate_bank,
                        )
                        if best_package is None or proposal[0] > best_package[0]:
                            best_package = proposal
        if best_package is not None and best_package[0] >= hurdle + 1.15:
            (
                _,
                raw_gain,
                fieldability_gain,
                captain_option,
                outgoing,
                incoming,
                squad,
                bank,
            ) = best_package
            for out_index, in_index in zip(outgoing, incoming):
                transfers.append(
                    {
                        "out": int(players[out_index]["id"]),
                        "outName": players[out_index]["name"],
                        "in": int(players[in_index]["id"]),
                        "inName": players[in_index]["name"],
                        "sixWeekGain": round(raw_gain, 2),
                        "fieldabilityGain": round(fieldability_gain, 3),
                        "captaincyOptionGain": round(captain_option, 3),
                        "packageMove": True,
                        "cost": 0,
                    }
                )
                purchase[int(players[in_index]["id"])] = float(
                    players[in_index]["price"]
                )
            available = 0
    for _ in range(available):
        club_counts = Counter(players[index]["team"] for index in squad)
        best = None
        for outgoing in squad:
            outgoing_id = int(players[outgoing]["id"])
            sale = selling_price(float(players[outgoing]["price"]), purchase.get(outgoing_id, float(players[outgoing]["price"])))
            for incoming, candidate in enumerate(players):
                if incoming in squad or candidate["position"] != players[outgoing]["position"]:
                    continue
                if enforce_fieldability and hard_unavailable(candidate):
                    continue
                if float(candidate["price"]) > bank + sale + 1e-6:
                    continue
                if candidate["team"] != players[outgoing]["team"] and club_counts[candidate["team"]] >= 3:
                    continue
                raw_gain = float(candidate[score_key]) - float(players[outgoing][score_key])
                fieldability_gain = play_probability(candidate) - play_probability(
                    players[outgoing]
                )
                emergency = bool(
                    enforce_fieldability and hard_unavailable(players[outgoing])
                )
                gain = raw_gain + (
                    4.0 * fieldability_gain if enforce_fieldability else 0.0
                )
                proposal = (gain, outgoing, incoming, sale, raw_gain, fieldability_gain, emergency)
                if best is None or (proposal[6], proposal[0]) > (best[6], best[0]):
                    best = proposal
        if best is None or (not best[6] and best[0] < hurdle):
            break
        gain, outgoing, incoming, sale, raw_gain, fieldability_gain, emergency = best
        bank = round(bank + sale - float(players[incoming]["price"]), 1)
        transfers.append(
            {
                "out": int(players[outgoing]["id"]),
                "outName": players[outgoing]["name"],
                "in": int(players[incoming]["id"]),
                "inName": players[incoming]["name"],
                "sixWeekGain": round(raw_gain, 2),
                "fieldabilityGain": round(fieldability_gain, 3),
                "emergencyRepair": emergency,
                "cost": 0,
            }
        )
        squad[squad.index(outgoing)] = incoming
        purchase[int(players[incoming]["id"])] = float(players[incoming]["price"])
    if not legal(players, squad, sum(float(players[index]["price"]) for index in squad) + bank):
        raise RuntimeError("Recursive transfer planner produced an illegal squad.")
    next_free = min(5, max(1, int(state.get("freeTransfers", 1)) - len(transfers)) + 1)
    return squad, transfers, bank, next_free


def player_card(player: dict) -> dict:
    return {
        "id": int(player["id"]),
        "name": player["name"],
        "team": player["team"],
        "position": player["position"],
        "price": player["price"],
        "projected": round(float(player["immediateForCaptain"]), 2),
        "deadlineMinutes": round(float(player["deadlineMinutes"]), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true", help="Commit decisions to recursive state.")
    args = parser.parse_args()
    status = json.loads((APP_DATA / "deadline-status.json").read_text(encoding="utf-8"))
    if args.lock and status["status"] != "locked":
        raise RuntimeError("Shadow decisions can only be committed from a --lock deadline snapshot.")
    snapshot = json.loads((ROOT / status["snapshotPath"]).read_text(encoding="utf-8"))
    chip = json.loads((APP_DATA / "chip-scenarios.json").read_text(encoding="utf-8"))
    frontier_path = APP_DATA / "frontier-scores.json"
    frontier = {}
    if frontier_path.exists():
        frontier = {int(row["id"]): row for row in json.loads(frontier_path.read_text(encoding="utf-8"))["players"]}
    listwise_path = APP_DATA / "listwise-scores.json"
    listwise = {}
    if listwise_path.exists():
        listwise = {int(row["id"]): row for row in json.loads(listwise_path.read_text(encoding="utf-8"))["players"]}
    regime_path = APP_DATA / "regime-scores.json"
    regime = {}
    if regime_path.exists():
        regime = {
            int(row["id"]): float(row["probability"])
            for row in json.loads(regime_path.read_text(encoding="utf-8"))["players"]
        }
    intelligence = {int(row["id"]): row for row in snapshot["deadlineIntelligence"]}
    raw = json.loads((APP_DATA / "current-players.json").read_text(encoding="utf-8"))
    players = []
    for row in raw:
        intel = intelligence.get(int(row["id"]), {})
        old_minutes = max(float(row["expectedMinutes"]), 8)
        new_minutes = float(intel.get("expectedMinutes", old_minutes))
        minute_ratio = max(0.20, min(1.25, new_minutes / old_minutes))
        immediate = float(row["projected"]) * (0.25 + 0.75 * minute_ratio)
        horizon = float(row["sixWeekProjected"]) * (0.75 + 0.25 * minute_ratio)
        upside = max(0.0, float(row["distribution"]["p90"]) - float(row["projected"]))
        frontier_immediate = float(frontier.get(int(row["id"]), {}).get("blend25", immediate))
        listwise_row = listwise.get(int(row["id"]), {})
        listwise_horizon = float(listwise_row.get("planBlend25", horizon)) * (0.75 + 0.25 * minute_ratio)
        listwise_captain = float(listwise_row.get("captainBlend50", row["captainRating"]))
        # Expected minutes are already inputs to both captain models.  Apply a
        # further penalty only when deadline intelligence is worse than the
        # model snapshot; using absolute minutes here double-counted rotation.
        deadline_captain_factor = 0.50 + 0.50 * min(1.0, minute_ratio)
        listwise_captain *= deadline_captain_factor
        route_captain = float(
            listwise_row.get("routeCaptainScore", listwise_captain)
        )
        route_captain *= deadline_captain_factor
        forecast_v2_captain = float(
            listwise_row.get("forecastV2CaptainScore", route_captain)
        )
        forecast_v2_captain *= deadline_captain_factor
        action_consensus = float(
            listwise_row.get("actionConsensusMapped", horizon)
        ) * (0.75 + 0.25 * minute_ratio)
        if not bool(listwise_row.get("actionPolicyActive", False)):
            action_consensus = horizon
        players.append(
            {
                **row,
                "deadlineMinutes": new_minutes,
                "immediateForCaptain": immediate,
                "horizonScore": horizon,
                "riskScore": horizon + 0.12 * upside,
                "frontierImmediateScore": frontier_immediate * (0.25 + 0.75 * minute_ratio),
                "listwiseHorizonScore": listwise_horizon,
                "listwiseCaptainScore": listwise_captain,
                "routeCaptainScore": route_captain,
                "forecastV2CaptainScore": forecast_v2_captain,
                "actionConsensusScore": action_consensus,
                "actionConsensusVote": float(listwise_row.get("actionVote", 0)),
                # Diagnostic only: fitted on all completed history after its
                # walk-forward gate passed.  Never add this probability to xPts.
                "regimeChangeProbability": float(regime.get(int(row["id"]), 0.0)),
            }
        )
    season_root = SHADOW_ROOT / status["season"]
    decision_managers = []
    for manager in MANAGERS:
        state_path = season_root / manager["id"] / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        manager_chip_plan = chip.get("managerPlans", {}).get(manager["id"], {})
        chip_use = (
            manager_chip_plan.get("recommendation", chip["recommendation"])
            if manager["chips"]
            else "Hold"
        )
        current_chip_key = chip_inventory_key(chip_use, int(status["gameweek"]))
        used_chips = used_chip_keys(state)
        if chip_use != "Hold" and current_chip_key in used_chips:
            chip_use = "Hold"
        if (
            chip_use == "Free Hit"
            and int(status["gameweek"]) == 20
            and state is not None
            and chip_inventory_key("Free Hit", 19) in used_chips
        ):
            # Official rule: FH19 cannot be followed immediately by FH20.
            chip_use = "Hold"
        transfers: list[dict] = []
        if state is None:
            squad, _, _, _ = optimise_squad(
                players,
                manager["planScore"],
                lineup_score_key=manager["lineupScore"],
                captain_score_key=manager["captainScore"],
                enforce_fieldability=bool(manager.get("fieldability", False)),
            )
            bank = round(100 - sum(float(players[index]["price"]) for index in squad), 1)
            free_transfers = 1
        elif chip_use == "Wildcard":
            available_budget = available_squad_budget(players, state)
            squad, _, _, _ = optimise_squad(
                players,
                manager["planScore"],
                lineup_score_key=manager["lineupScore"],
                captain_score_key=manager["captainScore"],
                minimum_spend=max(0.0, available_budget - 0.5),
                budget_limit=available_budget,
                enforce_fieldability=bool(manager.get("fieldability", False)),
            )
            bank = round(
                available_budget
                - sum(float(players[index]["price"]) for index in squad),
                1,
            )
            free_transfers = int(state.get("freeTransfers", 1))
        elif chip_use == "Free Hit":
            by_id = {int(row["id"]): index for index, row in enumerate(players)}
            squad = [
                by_id[player_id]
                for player_id in state["squadIds"]
                if player_id in by_id
            ]
            transfers = []
            bank = float(state["bank"])
            free_transfers = int(state.get("freeTransfers", 1))
        else:
            squad, transfers, bank, free_transfers = transfer_plan(
                players,
                state,
                manager["planScore"],
                float(manager["transferHurdle"]),
                enforce_fieldability=bool(manager.get("fieldability", False)),
                captain_key=manager["captainScore"],
                package_search=bool(manager.get("fieldability", False)),
            )
        if chip_use == "Free Hit":
            plan_squads = manager_chip_plan.get("squads", {})
            temporary_ids = set(
                plan_squads.get("freeHit", chip["squads"]["freeHit"])
            )
            decision_squad = [index for index, row in enumerate(players) if int(row["id"]) in temporary_ids]
        else:
            decision_squad = squad
        xi, bench, captain, vice = best_lineup(
            players,
            decision_squad,
            manager["lineupScore"],
            manager["captainScore"],
            enforce_fieldability=bool(manager.get("fieldability", False)),
        )
        projected = sum(float(players[index]["immediateForCaptain"]) for index in xi) + float(players[captain]["immediateForCaptain"])
        premium_candidates = sorted(
            [
                index
                for index, row in enumerate(players)
                if float(row["price"]) >= 11.0
            ],
            key=lambda index: float(players[index][manager["captainScore"]]),
            reverse=True,
        )[:5]
        premium_access = []
        current_package_value = sum(
            float(players[index][manager["planScore"]]) for index in decision_squad
        )

        def premium_package_value(premium_index: int) -> tuple[float, float]:
            if premium_index in decision_squad:
                return current_package_value, float(bank)
            best_value = -1e9
            best_bank = -1e9
            premium = players[premium_index]
            for outgoing in decision_squad:
                if players[outgoing]["position"] != premium["position"]:
                    continue
                direct_bank = (
                    float(bank)
                    + float(players[outgoing]["price"])
                    - float(premium["price"])
                )
                direct = [
                    premium_index if index == outgoing else index
                    for index in decision_squad
                ]
                if direct_bank >= -1e-6 and legal(
                    players,
                    direct,
                    sum(float(players[index]["price"]) for index in direct)
                    + direct_bank,
                ):
                    value = (
                        current_package_value
                        - float(players[outgoing][manager["planScore"]])
                        + float(premium[manager["planScore"]])
                    )
                    if value > best_value:
                        best_value, best_bank = value, direct_bank
                for funding_out in decision_squad:
                    if funding_out == outgoing:
                        continue
                    replacements = sorted(
                        [
                            index
                            for index, row in enumerate(players)
                            if row["position"] == players[funding_out]["position"]
                            and index not in decision_squad
                            and index != premium_index
                            and not hard_unavailable(row)
                        ],
                        key=lambda index: float(players[index][manager["planScore"]]),
                        reverse=True,
                    )[:12]
                    for funding_in in replacements:
                        candidate_bank = (
                            float(bank)
                            + float(players[outgoing]["price"])
                            + float(players[funding_out]["price"])
                            - float(premium["price"])
                            - float(players[funding_in]["price"])
                        )
                        if candidate_bank < -1e-6:
                            continue
                        candidate = [
                            index
                            for index in decision_squad
                            if index not in (outgoing, funding_out)
                        ] + [premium_index, funding_in]
                        if not legal(
                            players,
                            candidate,
                            sum(float(players[index]["price"]) for index in candidate)
                            + candidate_bank,
                        ):
                            continue
                        value = (
                            current_package_value
                            - float(players[outgoing][manager["planScore"]])
                            - float(players[funding_out][manager["planScore"]])
                            + float(premium[manager["planScore"]])
                            + float(players[funding_in][manager["planScore"]])
                        )
                        if value > best_value:
                            best_value, best_bank = value, candidate_bank
            return best_value, best_bank

        for premium_index in premium_candidates:
            owned = premium_index in decision_squad
            premium_package, premium_bank = premium_package_value(premium_index)
            captain_edge = max(
                0.0,
                float(players[premium_index][manager["captainScore"]])
                - float(players[captain][manager["captainScore"]]),
            )
            diagnostic = premium_access_diagnostic(
                premium_id=int(players[premium_index]["id"]),
                current_package_value=current_package_value,
                premium_package_value=premium_package,
                future_captain_probabilities=[0.65],
                future_captain_edges=[captain_edge],
                liquidity_value=(
                    0.10 * max(-2.0, premium_bank - float(bank))
                    if premium_package > -1e8
                    else -10.0
                ),
                model_disagreement=float(players[premium_index].get("uncertainty", 0.0)),
            )
            premium_access.append(
                {
                    "id": int(players[premium_index]["id"]),
                    "name": players[premium_index]["name"],
                    "owned": owned,
                    "captaincyOptionValue": diagnostic.captaincy_option_value,
                    "robustAccessAdvantage": diagnostic.robust_advantage,
                    "bestLegalPackageValue": (
                        round(premium_package, 3)
                        if premium_package > -1e8
                        else None
                    ),
                    "reviewRequired": diagnostic.access_failure and not owned,
                }
            )
        decision = {
            **manager,
            "chip": chip_use,
            "bank": bank,
            "freeTransfersNext": free_transfers,
            "transfers": transfers,
            "projectedPoints": round(projected, 1),
            "squad": [player_card(players[index]) for index in decision_squad],
            "xiIds": [int(players[index]["id"]) for index in xi],
            "benchIds": [int(players[index]["id"]) for index in bench],
            "captainId": int(players[captain]["id"]),
            "captain": players[captain]["name"],
            "viceId": int(players[vice]["id"]),
            "vice": players[vice]["name"],
            "fieldability": fieldability_audit(
                players, decision_squad, xi
            ),
            "premiumAccessAudit": premium_access,
            "highRegimeChangePlayers": [
                {
                    "id": int(players[index]["id"]),
                    "name": players[index]["name"],
                    "probability": round(float(players[index]["regimeChangeProbability"]), 3),
                    "owned": index in decision_squad,
                }
                for index in sorted(
                    range(len(players)),
                    key=lambda candidate: float(players[candidate]["regimeChangeProbability"]),
                    reverse=True,
                )[:8]
            ],
        }
        decision_managers.append(decision)
        if args.lock:
            permanent = squad
            previous_purchase = {} if state is None else state["purchasePrices"]
            purchase_prices = {
                str(int(players[index]["id"])): float(
                    previous_purchase.get(str(int(players[index]["id"])), players[index]["price"])
                )
                for index in permanent
            }
            used = [] if state is None else list(state.get("chipsUsed", []))
            if chip_use != "Hold":
                used.append(
                    {
                        "chip": chip_use,
                        "gameweek": int(status["gameweek"]),
                        "key": chip_inventory_key(
                            chip_use, int(status["gameweek"])
                        ),
                    }
                )
            atomic_json(
                state_path,
                {
                    "season": status["season"],
                    "lastGameweek": int(status["gameweek"]),
                    "squadIds": [int(players[index]["id"]) for index in permanent],
                    "purchasePrices": purchase_prices,
                    "bank": bank,
                    "freeTransfers": free_transfers,
                    "chipsUsed": used,
                    "lastSnapshotHash": status["snapshotHash"],
                },
            )
    core = {
        "schemaVersion": 1,
        "season": status["season"],
        "gameweek": int(status["gameweek"]),
        "decisionStatus": "locked" if args.lock else "provisional",
        "snapshotHash": status["snapshotHash"],
        "snapshotPath": status["snapshotPath"],
        "generatedAt": status["capturedAt"],
        "chipRecommendation": chip["recommendation"],
        "managers": decision_managers,
        "protocol": "All managers receive the same immutable deadline snapshot. Only locked decisions update recursive state; no post-deadline edit is permitted.",
    }
    decision_hash = payload_hash(core)
    core["decisionHash"] = decision_hash
    target = season_root / f"gw-{int(status['gameweek']):02d}" / f"{status['snapshotHash'][:12]}-{decision_hash[:12]}.json"
    if not target.exists():
        atomic_json(target, core)
    latest = {
        **core,
        "decisionPath": str(target.relative_to(ROOT)).replace("\\", "/"),
        "completedGameweeks": 0,
        "cumulative": {manager["id"]: 0 for manager in MANAGERS},
    }
    score_path = season_root / "scores.json"
    if score_path.exists():
        scores = json.loads(score_path.read_text(encoding="utf-8"))
        latest["completedGameweeks"] = len(scores.get("gameweeks", []))
        latest["cumulative"] = scores.get("cumulative", latest["cumulative"])
    atomic_json(APP_DATA / "shadow-status.json", latest)
    print(f"Created {core['decisionStatus']} shadow decision: {target.relative_to(ROOT)}")
    for row in decision_managers:
        print(f"{row['name']}: {row['projectedPoints']} pts, {row['captain']} captain, {row['chip']}")


if __name__ == "__main__":
    main()

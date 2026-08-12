"""Run three frozen prospective managers from the same deadline snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from prospective_common import APP_DATA, ROOT, SHADOW_ROOT, atomic_json, optimise_squad, payload_hash


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
) -> tuple[list[int], list[int], int, int]:
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
    captain = max(xi, key=lambda index: float(players[index][captain_key]))
    vice = max(
        (index for index in xi if index != captain),
        key=lambda index: float(players[index][captain_key]),
    )
    bench = sorted(
        [index for index in squad if index not in xi],
        key=lambda index: (players[index]["position"] != "GK", float(players[index][score_key])),
        reverse=True,
    )
    goalkeeper = next(index for index in bench if players[index]["position"] == "GK")
    bench = [goalkeeper] + [index for index in bench if index != goalkeeper]
    return xi, bench, captain, vice


def selling_price(current: float, purchase: float) -> float:
    if current <= purchase:
        return current
    return purchase + int((current - purchase) * 10 / 2) / 10


def transfer_plan(
    players: list[dict],
    state: dict,
    score_key: str,
    hurdle: float,
) -> tuple[list[int], list[dict], float, int]:
    by_id = {int(row["id"]): index for index, row in enumerate(players)}
    squad = [by_id[player_id] for player_id in state["squadIds"] if player_id in by_id]
    purchase = {int(key): float(value) for key, value in state["purchasePrices"].items()}
    bank = float(state["bank"])
    transfers: list[dict] = []
    available = min(int(state.get("freeTransfers", 1)), 2)
    for _ in range(available):
        club_counts = Counter(players[index]["team"] for index in squad)
        best = None
        for outgoing in squad:
            outgoing_id = int(players[outgoing]["id"])
            sale = selling_price(float(players[outgoing]["price"]), purchase.get(outgoing_id, float(players[outgoing]["price"])))
            for incoming, candidate in enumerate(players):
                if incoming in squad or candidate["position"] != players[outgoing]["position"]:
                    continue
                if float(candidate["price"]) > bank + sale + 1e-6:
                    continue
                if candidate["team"] != players[outgoing]["team"] and club_counts[candidate["team"]] >= 3:
                    continue
                gain = float(candidate[score_key]) - float(players[outgoing][score_key])
                proposal = (gain, outgoing, incoming, sale)
                if best is None or proposal[0] > best[0]:
                    best = proposal
        if best is None or best[0] < hurdle:
            break
        gain, outgoing, incoming, sale = best
        bank = round(bank + sale - float(players[incoming]["price"]), 1)
        transfers.append(
            {
                "out": int(players[outgoing]["id"]),
                "outName": players[outgoing]["name"],
                "in": int(players[incoming]["id"]),
                "inName": players[incoming]["name"],
                "sixWeekGain": round(gain, 2),
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
        listwise_captain *= 0.50 + 0.50 * min(1.0, new_minutes / 75)
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
        used_chips = set() if state is None else {row["chip"] for row in state.get("chipsUsed", [])}
        if chip_use in used_chips:
            chip_use = "Hold"
        transfers: list[dict] = []
        if state is None:
            squad, _, _, _ = optimise_squad(players, manager["planScore"])
            bank = round(100 - sum(float(players[index]["price"]) for index in squad), 1)
            free_transfers = 1
        elif chip_use == "Wildcard":
            squad, _, _, _ = optimise_squad(players, manager["planScore"])
            bank = round(float(state["bank"]) + sum(
                selling_price(float(players[next(i for i, p in enumerate(players) if int(p["id"]) == player_id)]["price"]), float(state["purchasePrices"].get(str(player_id), 0)))
                for player_id in state["squadIds"]
                if any(int(p["id"]) == player_id for p in players)
            ) - sum(float(players[index]["price"]) for index in squad), 1)
            free_transfers = int(state.get("freeTransfers", 1))
        else:
            squad, transfers, bank, free_transfers = transfer_plan(
                players, state, manager["planScore"], float(manager["transferHurdle"])
            )
        if chip_use == "Free Hit":
            temporary_ids = set(chip["squads"]["freeHit"])
            decision_squad = [index for index, row in enumerate(players) if int(row["id"]) in temporary_ids]
        else:
            decision_squad = squad
        xi, bench, captain, vice = best_lineup(
            players,
            decision_squad,
            manager["lineupScore"],
            manager["captainScore"],
        )
        projected = sum(float(players[index]["immediateForCaptain"]) for index in xi) + float(players[captain]["immediateForCaptain"])
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
                used.append({"chip": chip_use, "gameweek": int(status["gameweek"])})
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

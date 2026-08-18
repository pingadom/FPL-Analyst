"""Shared primitives for immutable prospective FPL research cycles."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "analysis" / "snapshots"
SHADOW_ROOT = ROOT / "analysis" / "shadow"
APP_DATA = ROOT / "app" / "data"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def chip_set_for_gameweek(gameweek: int) -> int:
    """Return the half-season chip set used by the 2025/26+ rules."""
    if not 1 <= int(gameweek) <= 38:
        raise ValueError(f"Invalid FPL gameweek for chip set: {gameweek}")
    return 1 if int(gameweek) <= 19 else 2


def chip_inventory_key(chip: str, gameweek: int) -> str:
    """Stable state key; unlike the display name it survives the GW20 refresh."""
    return f"{str(chip)}:H{chip_set_for_gameweek(gameweek)}"


def used_chip_keys(state: dict | None) -> set[str]:
    """Read new state and migrate legacy rows that stored only the chip name."""
    if state is None:
        return set()
    result: set[str] = set()
    for row in state.get("chipsUsed", []):
        gameweek = int(row.get("gameweek", 1))
        result.add(
            str(row.get("key") or chip_inventory_key(str(row["chip"]), gameweek))
        )
    return result


def selling_price(current: float, purchase: float) -> float:
    """FPL sale value after the manager receives half of any price rise."""
    if current <= purchase:
        return current
    return purchase + int((current - purchase) * 10 / 2) / 10


def available_squad_budget(players: list[dict], state: dict | None) -> float:
    """Return bank plus current selling value for a manager-specific rebuild."""
    if state is None:
        return 100.0
    by_id = {int(row["id"]): row for row in players}
    purchase = {
        int(key): float(value)
        for key, value in state.get("purchasePrices", {}).items()
    }
    return round(
        float(state.get("bank", 0.0))
        + sum(
            selling_price(
                float(by_id[player_id]["price"]),
                purchase.get(player_id, float(by_id[player_id]["price"])),
            )
            for player_id in state.get("squadIds", [])
            if int(player_id) in by_id
        ),
        1,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def official_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "FPL-Lens-prospective/1.0"}
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def season_label(bootstrap: dict) -> str:
    events = bootstrap.get("events", [])
    first_deadline = next(
        (event.get("deadline_time") for event in events if event.get("deadline_time")),
        None,
    )
    year = datetime.fromisoformat(str(first_deadline).replace("Z", "+00:00")).year
    return f"{year}-{str(year + 1)[-2:]}"


def next_event(bootstrap: dict) -> dict:
    events = bootstrap.get("events", [])
    event = next((item for item in events if item.get("is_next")), None)
    if event is None:
        event = next((item for item in events if not item.get("finished")), None)
    if event is None:
        raise RuntimeError("No unfinished official FPL event is available.")
    return event


def optimise_squad(
    players: list[dict],
    score_key: str,
    *,
    bench_weight: float = 0.08,
    captain_weight: float = 0.70,
    bench_premium_limit: float = 2.0,
    bench_premium_penalty: float = 0.22,
    minimum_spend: float = 99.5,
    budget_limit: float = 100.0,
) -> tuple[list[int], list[int], int, int]:
    """Jointly solve the legal squad, XI and captain in one integer program."""
    positions = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    count = len(players)
    scores = np.asarray([float(row[score_key]) for row in players])
    # x selects the XV, y the XI, and c the captain. The XI receives the full
    # forecast, the bench a small contingency value, and captaincy an extra
    # planning weight selected by the frozen policy audit.
    prices = np.asarray([float(row["price"]) for row in players])
    price_floors = {
        position: min(float(row["price"]) for row in players if row["position"] == position)
        for position in positions
    }
    bench_premium = np.asarray(
        [max(0.0, price - price_floors[row["position"]]) for price, row in zip(prices, players)]
    )
    # x-y identifies bench players. Charge premium cost only to that difference,
    # so an expensive starter is judged on points while unused bench money has
    # to justify itself through genuine contingency value.
    objective = -np.concatenate(
        [
            bench_weight * scores - bench_premium_penalty * bench_premium,
            (1 - bench_weight) * scores + bench_premium_penalty * bench_premium,
            captain_weight * scores,
        ]
    )
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    budget = np.asarray([int(round(float(row["price"]) * 10)) for row in players])
    rows.append(np.concatenate([budget, np.zeros(2 * count)]))
    effective_minimum_spend = min(float(minimum_spend), float(budget_limit))
    lower.append(int(round(effective_minimum_spend * 10)))
    upper.append(int(round(float(budget_limit) * 10)))
    for position, quota in positions.items():
        membership = np.asarray([1 if row["position"] == position else 0 for row in players])
        rows.append(np.concatenate([membership, np.zeros(2 * count)]))
        lower.append(quota)
        upper.append(quota)
    for club in sorted({str(row["team"]) for row in players}):
        membership = np.asarray([1 if str(row["team"]) == club else 0 for row in players])
        rows.append(np.concatenate([membership, np.zeros(2 * count)]))
        lower.append(0)
        upper.append(3)
    # The XI must be a subset of the squad; the captain must be in the XI.
    for index in range(count):
        link_xi = np.zeros(3 * count)
        link_xi[count + index] = 1
        link_xi[index] = -1
        rows.append(link_xi)
        lower.append(-np.inf)
        upper.append(0)
        link_captain = np.zeros(3 * count)
        link_captain[2 * count + index] = 1
        link_captain[count + index] = -1
        rows.append(link_captain)
        lower.append(-np.inf)
        upper.append(0)
    xi_total = np.zeros(3 * count)
    xi_total[count : 2 * count] = 1
    rows.append(xi_total)
    lower.append(11)
    upper.append(11)
    for position, minimum, maximum in (
        ("GK", 1, 1),
        ("DEF", 3, 5),
        ("MID", 2, 5),
        ("FWD", 1, 3),
    ):
        lineup = np.zeros(3 * count)
        lineup[count : 2 * count] = [1 if row["position"] == position else 0 for row in players]
        rows.append(lineup)
        lower.append(minimum)
        upper.append(maximum)
    captain_total = np.zeros(3 * count)
    captain_total[2 * count :] = 1
    rows.append(captain_total)
    lower.append(1)
    upper.append(1)
    premium_total = np.concatenate([bench_premium, -bench_premium, np.zeros(count)])
    rows.append(premium_total)
    lower.append(0)
    upper.append(bench_premium_limit)
    result = milp(
        c=objective,
        integrality=np.ones(3 * count),
        bounds=Bounds(np.zeros(3 * count), np.ones(3 * count)),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 20.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Legal squad optimisation failed: {result.message}")
    chosen = np.flatnonzero(result.x[:count] > 0.5).astype(int).tolist()
    best_xi = np.flatnonzero(result.x[count : 2 * count] > 0.5).astype(int).tolist()
    captain = int(np.flatnonzero(result.x[2 * count :] > 0.5)[0])
    vice = max(
        (index for index in best_xi if index != captain),
        key=lambda index: float(players[index][score_key]),
    )
    return chosen, best_xi, captain, vice


def normal_scenarios(mean: np.ndarray, std: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, size=(draws, len(mean))), 0, None)

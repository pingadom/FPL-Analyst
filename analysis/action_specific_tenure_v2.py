"""Action-specific forecast horizons for the FPL decision engine.

A single fixed look-ahead cannot serve captaincy, Free Hit, transfers and a
Wildcard equally.  This module exposes explicit value surfaces while keeping
all labels deadline-causal.  It is an interface and audit layer; no surface is
promoted until a paired recursive replay passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import calibrate_model as lens
from multiscale_horizon_validation import (
    HORIZONS,
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    expected_tenure,
    interpolate_horizon,
    structural_horizons,
)
from wildcard_freehit_ablation import champion_forecasts


@dataclass(frozen=True)
class ActionProfile:
    action: str
    horizon: str
    replacement_cost: float
    description: str


ACTION_PROFILES = (
    ActionProfile("captain", "h1", 0.0, "Only the next scoring event matters."),
    ActionProfile("starting_xi", "h1", 0.0, "Line-up choice is reset next deadline."),
    ActionProfile("free_hit", "h1", 0.0, "The temporary squad lasts one event."),
    ActionProfile("bench", "h3", 0.0, "Short cover value without premium overinvestment."),
    ActionProfile("transfer", "player-specific-1/3/6/10", 3.0, "Tenure follows durability and charges a future replacement."),
    ActionProfile("wildcard", "h10", 0.0, "Durable squad structure receives the longest audited horizon."),
)


def tenure_probabilities(tenure: np.ndarray) -> np.ndarray:
    """Return interpolation weights over the 1/3/6/10-GW horizon knots."""
    tenure = np.clip(np.asarray(tenure, float), HORIZONS[0], HORIZONS[-1])
    result = np.zeros((len(tenure), len(HORIZONS)), dtype=float)
    knots = np.asarray(HORIZONS, float)
    for row, value in enumerate(tenure):
        upper = int(np.searchsorted(knots, value, side="right"))
        if upper == 0:
            result[row, 0] = 1.0
        elif upper >= len(knots):
            result[row, -1] = 1.0
        else:
            lower = upper - 1
            fraction = (value - knots[lower]) / (knots[upper] - knots[lower])
            result[row, lower] = 1.0 - fraction
            result[row, upper] = fraction
    return result


def build_surfaces(data, immediate: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    structural = structural_horizons(data, immediate)
    static, static_audit = causal_ridge_horizons(data, structural)
    online, online_audit = causal_online_ridge_horizons(data, structural, static)
    tenure = expected_tenure(data)
    surfaces = {
        "captain": online[1],
        "starting_xi": online[1],
        "free_hit": online[1],
        "bench": online[3],
        "transfer": adaptive_value(data, online, exit_cost_scale=3.0),
        "wildcard": online[10],
    }
    weights = tenure_probabilities(tenure)
    position_audit = []
    for position in lens.SQUAD_QUOTAS:
        mask = data["position_id"].to_numpy(int) == position
        position_audit.append(
            {
                "position": int(position),
                "rows": int(mask.sum()),
                "meanExpectedTenure": round(float(tenure[mask].mean()), 3),
                "meanHorizonWeights": {
                    f"h{horizon}": round(float(weights[mask, index].mean()), 4)
                    for index, horizon in enumerate(HORIZONS)
                },
            }
        )
    audit = {
        "profiles": [profile.__dict__ for profile in ACTION_PROFILES],
        "positionTenure": position_audit,
        "staticFits": len(static_audit),
        "onlineFits": len(online_audit),
        "maturityRule": (
            "Each horizon model uses prior seasons plus only same-season labels "
            "whose full target horizon ended before the update checkpoint."
        ),
    }
    return surfaces, audit


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, _, _ = champion_forecasts(data)
    _, audit = build_surfaces(data, immediate)
    result = {
        "schemaVersion": 1,
        "status": "action-specific causal value surfaces available to challengers",
        **audit,
        "promotion": (
            "Surfaces are not automatically substituted into the recursive "
            "champion. Each action boundary requires a paired replay."
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "action_specific_tenure_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiles": result["profiles"], "positionTenure": result["positionTenure"]}, indent=2))


if __name__ == "__main__":
    main()

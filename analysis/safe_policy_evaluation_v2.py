"""Doubly-robust policy evaluation primitives and historical support audit.

The existing recursive replays provide deterministic counterfactual totals,
but they do not contain logged behaviour propensities.  This module refuses to
invent them.  It supplies a tested DR estimator for prospective shadow logs and
records exactly why the historical package frontier cannot support a valid DR
claim today.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import calibrate_model as lens


REQUIRED_COLUMNS = frozenset(
    {"action", "reward", "behaviourPropensity", "targetProbability", "qLogged", "qTarget"}
)


def doubly_robust_value(
    action: np.ndarray,
    reward: np.ndarray,
    behaviour_propensity: np.ndarray,
    target_probability: np.ndarray,
    q_logged: np.ndarray,
    q_target: np.ndarray,
) -> tuple[float, float]:
    """Estimate policy value and standard error for contextual-bandit logs."""
    del action  # The caller encodes action matching in target_probability.
    reward = np.asarray(reward, float)
    propensity = np.asarray(behaviour_propensity, float)
    target = np.asarray(target_probability, float)
    q_logged = np.asarray(q_logged, float)
    q_target = np.asarray(q_target, float)
    if not (
        len(reward)
        == len(propensity)
        == len(target)
        == len(q_logged)
        == len(q_target)
    ):
        raise ValueError("All DR inputs must have equal length")
    if np.any(propensity <= 0) or np.any(propensity > 1):
        raise ValueError("Behaviour propensities must be in (0, 1]")
    if np.any(target < 0) or np.any(target > 1):
        raise ValueError("Target probabilities must be in [0, 1]")
    influence = q_target + target / propensity * (reward - q_logged)
    estimate = float(influence.mean())
    standard_error = (
        float(influence.std(ddof=1) / np.sqrt(len(influence)))
        if len(influence) > 1
        else 0.0
    )
    return estimate, standard_error


def historical_support_audit(path: Path) -> dict:
    loaded = np.load(path)
    metadata = json.loads(str(loaded["metadata"].item()))
    available = set(metadata[0]) if metadata else set()
    missing = sorted(REQUIRED_COLUMNS - available)
    policies = sorted({str(row.get("sourcePolicy", "unknown")) for row in metadata})
    return {
        "rows": len(metadata),
        "sourcePolicies": policies,
        "availableColumns": sorted(available),
        "missingColumns": missing,
        "overlapIdentified": not missing,
        "validDoublyRobustEstimate": False if missing else None,
        "reason": (
            "Candidate actions and modelled horizon gains are present, but the "
            "chosen action reward and behaviour propensity were not logged. "
            "A doubly-robust estimate would therefore be unidentified."
            if missing
            else "Support fields exist; reward and propensity diagnostics are still required."
        ),
    }


def main() -> None:
    source = lens.CACHE / "breakthrough-action-states-v1.npz"
    audit = historical_support_audit(source)
    result = {
        "schemaVersion": 1,
        "status": "prospective DR evaluator implemented; historical inference blocked honestly",
        "requiredShadowLogColumns": sorted(REQUIRED_COLUMNS),
        "historicalSupport": audit,
        "promotionRule": (
            "Use paired recursive replay for historical selection. Use DR only "
            "after prospective shadow policies log actions, rewards, target "
            "probabilities and non-zero behaviour propensities."
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "safe_policy_evaluation_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

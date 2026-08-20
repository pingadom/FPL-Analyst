"""Expanded-state conservative action-value learning.

The rejected package learner saw only the champion's recursive states.  This
version collects the same complete legal package boundary from several frozen
policies, giving the learner difficult near-frontier examples from materially
different squads, banks and transfer inventories.  Fits remain walk-forward
and may abstain to the structural action.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric
from decision_focused_horizon_validation import target_and_maturity
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from package_action_value_validation import (
    FeatureBuilder,
    candidate_metrics,
    causal_fits,
    checkpoint_for_event,
)
from probabilistic_component_challenger import causal_route_predictions
from probabilistic_minutes_validation import season_summary
from transfer_action_ranker_validation import (
    CACHE_VERSION as ACTION_RANKER_CACHE_VERSION,
    agreed_action_plan,
    causal_action_predictions,
)
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1


def collection_strategies() -> list[lens.SimulationStrategy]:
    return [
        STRATEGY,
        replace(
            STRATEGY,
            name="Fieldability boundary",
            enforce_fieldability=True,
            fieldability_penalty=0.0,
        ),
        replace(STRATEGY, name="Lower transfer hurdle", transfer_hurdle=12.0),
        replace(STRATEGY, name="Higher transfer hurdle", transfer_hurdle=20.0),
        replace(
            STRATEGY,
            name="Broader legal beam",
            transfer_candidate_limit=16,
            transfer_beam_width=18,
        ),
    ]


def collect_multi_policy_packages(
    data: pd.DataFrame,
    scores: np.ndarray,
    plan: np.ndarray,
    captain: np.ndarray,
    builder: FeatureBuilder,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    cache = lens.CACHE / f"breakthrough-action-states-v{CACHE_VERSION}.npz"
    if cache.exists():
        loaded = np.load(cache)
        return (
            pd.DataFrame(json.loads(str(loaded["metadata"].item()))),
            loaded["features"],
            json.loads(str(loaded["collectionAudit"].item())),
        )

    records: dict[tuple, dict] = {}
    audit = []
    active_policy = ""

    def collector(context: dict) -> float:
        vector, metadata = builder.vector(context)
        base_signature = tuple(sorted(int(value) for value in context["baseSquad"]))
        key = (
            metadata["seasonOrder"],
            metadata["gw"],
            base_signature,
            metadata["outgoing"],
            metadata["incoming"],
        )
        if key not in records:
            residual, target_gain = builder.target_residual(context)
            records[key] = {
                **metadata,
                "sourcePolicy": active_policy,
                "baseSignature": list(base_signature),
                "predictedGain": float(context["predictedGain"]),
                "targetGain": float(target_gain),
                "targetResidual": float(residual),
                "feature": vector,
            }
        return 0.0

    for strategy in collection_strategies():
        active_policy = strategy.name
        before = len(records)
        print(f"Collecting action states: {strategy.name}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan,
            captain_scores=captain,
            package_action_adjustment=collector,
        )
        audit.append(
            {
                "policy": strategy.name,
                "evaluationAverage": round(float(totals[2:].mean()), 1),
                "newStates": len(records) - before,
                "cumulativeStates": len(records),
            }
        )

    ordered = list(records.values())
    features = np.vstack([row.pop("feature") for row in ordered]).astype(np.float32)
    metadata = pd.DataFrame(ordered)
    np.savez_compressed(
        cache,
        features=features,
        metadata=json.dumps(metadata.to_dict(orient="records")),
        collectionAudit=json.dumps(audit),
    )
    return metadata, features, audit


class ConservativeAdvantagePolicy:
    def __init__(
        self,
        builder: FeatureBuilder,
        fits: dict,
        *,
        probability_gate: float,
        residual_blend: float,
        correction_cap: float = 2.0,
    ) -> None:
        self.builder = builder
        self.fits = fits
        self.probability_gate = probability_gate
        self.residual_blend = residual_blend
        self.correction_cap = correction_cap
        self.calls = 0
        self.no_fit = 0
        self.vetoes = 0
        self.corrections = 0

    def __call__(self, context: dict) -> float:
        self.calls += 1
        vector, metadata = self.builder.vector(context)
        if metadata["goalkeeperPackage"]:
            return 0.0
        fitted = self.fits.get(
            (metadata["seasonOrder"], checkpoint_for_event(metadata["event"]))
        )
        if fitted is None:
            self.no_fit += 1
            return 0.0
        probabilities = np.asarray(
            [model.predict_probability(vector) for model in fitted.classifiers]
        )
        residuals = np.asarray([model.predict_one(vector) for model in fitted.models])
        target_predictions = float(context["predictedGain"]) + residuals
        disagreement = float(np.ptp(target_predictions))
        pessimistic = float(target_predictions.min() - 0.5 * disagreement)
        optimistic = float(target_predictions.max() + 0.5 * disagreement)
        if probabilities.max() < 1.0 - self.probability_gate or optimistic < 0.0:
            self.vetoes += 1
            return -100.0
        if probabilities.min() < self.probability_gate or pessimistic <= 0.0:
            return 0.0
        correction = float(
            np.clip(
                self.residual_blend * residuals.mean(),
                -self.correction_cap,
                self.correction_cap,
            )
        )
        if abs(correction) > 1e-9:
            self.corrections += 1
        return correction

    def audit(self) -> dict:
        return {
            "calls": self.calls,
            "noFit": self.no_fit,
            "vetoes": self.vetoes,
            "corrections": self.corrections,
            "abstentionRate": round(
                (self.calls - self.vetoes - self.corrections) / max(1, self.calls),
                4,
            ),
        }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    component, _ = causal_route_predictions(data, immediate)
    action_ranker_cache = (
        lens.CACHE / f"transfer-action-ranker-v{ACTION_RANKER_CACHE_VERSION}.npz"
    )
    if not action_ranker_cache.exists():
        print("Building missing causal transfer-action cache", flush=True)
        causal_action_predictions(data, plan, component)
    action_plan = agreed_action_plan(data, plan, 0.05)
    target, maturity = target_and_maturity(data)
    builder = FeatureBuilder(
        data, immediate, plan, action_plan, component, target, maturity
    )
    metadata, features, collection_audit = collect_multi_policy_packages(
        data, immediate, plan, captain, builder
    )
    fits, fit_audit = causal_fits(metadata, features)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    print("Running current audited control", flush=True)
    control_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    rows = []
    for probability, blend in ((0.58, 0.05), (0.60, 0.10), (0.62, 0.10)):
        strategy = replace(
            STRATEGY,
            name=f"Expanded action p{probability:.2f} b{blend:.2f}",
            # The separately tested historical fieldability strategy failed
            # its robustness gate, so it may diversify training states but may
            # not be bundled into this candidate's claimed action-value lift.
            enforce_fieldability=False,
            fieldability_penalty=0.0,
        )
        policy = ConservativeAdvantagePolicy(
            builder,
            fits,
            probability_gate=probability,
            residual_blend=blend,
        )
        print(f"Running {strategy.name}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            strategy,
            chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
            plan_scores=plan,
            captain_scores=captain,
            package_action_adjustment=policy,
        )
        delta = totals[2:] - control_totals[2:]
        rows.append(
            {
                "name": strategy.name,
                "parameters": {
                    "probabilityGate": probability,
                    "residualBlend": blend,
                },
                **season_summary(totals, seasons),
                "paired": {
                    "averageDelta": round(float(delta.mean()), 1),
                    "minimumDelta": int(delta.min()),
                    "positiveSeasons": int((delta > 0).sum()),
                    "negativeSeasons": int((delta < 0).sum()),
                    "seasonDeltas": delta.astype(int).tolist(),
                },
                "policyAudit": policy.audit(),
            }
        )

    selected = max(
        rows,
        key=lambda row: (
            row["developmentAverage"],
            row["holdoutAverage"],
            row["minimum"],
        ),
    )
    passed = bool(
        selected["paired"]["averageDelta"] >= 15
        and selected["paired"]["minimumDelta"] >= -10
        and selected["paired"]["positiveSeasons"] >= 6
        and selected["holdoutAverage"] >= 2170.5
    )
    result = {
        "status": (
            "expanded action challenger passed historical engineering gate"
            if passed
            else "expanded action challenger implemented; promotion gate failed"
        ),
        "method": (
            "Complete legal packages from five distinct recursive policies, "
            "prior-season/matured-label fits, two-model probability agreement, "
            "pessimistic action value and explicit Hold abstention."
        ),
        "trainingStates": int(len(metadata)),
        "sourcePolicies": collection_audit,
        "candidateMetrics": candidate_metrics(metadata, features, fits),
        "fitAudit": fit_audit,
        "control": season_summary(control_totals, seasons),
        "selected": selected,
        "experiments": rows,
        "passed": passed,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "breakthrough_action_value_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "trainingStates": result["trainingStates"],
                "candidateMetrics": result["candidateMetrics"],
                "controlAverage": result["control"]["average"],
                "experiments": [
                    {
                        "name": row["name"],
                        "average": row["average"],
                        "minimum": row["minimum"],
                        "paired": row["paired"],
                        "policyAudit": row["policyAudit"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

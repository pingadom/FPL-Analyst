"""Train transfer action value on realised legal fixed-squad rollouts.

The earlier action learner used discounted player targets.  This version labels
every collected legal package by replaying the base and candidate squads over
1/3/6/10 future scoring events.  At each future deadline the XI and captain are
chosen from that deadline's causal forecast, while realised points, autosubs
and captain-to-vice fallback supply the reward.  Labels are used only after
their complete horizon has matured.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import calibrate_model as lens
from breakthrough_action_value_validation import ConservativeAdvantagePolicy
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric
from decision_focused_horizon_validation import target_and_maturity
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets, expected_tenure
from package_action_value_validation import (
    FeatureBuilder,
    candidate_metrics,
    causal_fits,
)
from probabilistic_component_challenger import causal_route_predictions
from probabilistic_minutes_validation import season_summary
from transfer_action_ranker_validation import agreed_action_plan
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1
HORIZONS = (1, 3, 6, 10)
DISCOUNT = 0.86


def interpolate_rollout(tenure: float, horizon_values: dict[int, float]) -> float:
    knots = np.asarray(HORIZONS, float)
    values = np.asarray([horizon_values[horizon] for horizon in HORIZONS], float)
    return float(np.interp(np.clip(tenure, knots[0], knots[-1]), knots, values))


class FixedSquadRollout:
    def __init__(
        self,
        data: pd.DataFrame,
        immediate: np.ndarray,
        captain: np.ndarray,
    ) -> None:
        self.data = data
        self.immediate = immediate
        self.captain = captain
        self.actual = data["points"].to_numpy(float)
        self.minutes = data["minutes"].to_numpy(float)
        self.weeks: dict[str, list[int]] = {}
        self.rows: dict[tuple[str, int], dict[int, int]] = {}
        self.states: dict[str, dict[int, dict]] = {}
        self.week_points: dict[tuple[str, int, tuple[int, ...]], float] = {}
        for season, season_frame in data.groupby("season", sort=False):
            season_name = str(season)
            weeks = list(dict.fromkeys(season_frame["GW"].astype(int).tolist()))
            self.weeks[season_name] = weeks
            state: dict[int, dict] = {}
            for row in season_frame.drop_duplicates("element").itertuples():
                state[int(row.element)] = {
                    "position": int(row.position_id),
                    "team": int(row.team_id),
                }
            self.states[season_name] = state
            for gw, frame in season_frame.groupby("GW", sort=False):
                self.rows[(season_name, int(gw))] = {
                    int(row.element): int(row.Index) for row in frame.itertuples()
                }

    def squad_state(self, season: str, signature: tuple[int, ...]) -> dict[int, dict]:
        state = self.states[season]
        return {
            element: state[element].copy()
            for element in signature
            if element in state
        }

    def score_week(self, season: str, gw: int, signature: tuple[int, ...]) -> float:
        key = (season, gw, signature)
        cached = self.week_points.get(key)
        if cached is not None:
            return cached
        squad = self.squad_state(season, signature)
        if len(squad) != 15:
            self.week_points[key] = -20.0
            return -20.0
        row_by_element = self.rows[(season, gw)]
        xi, bench = lens.choose_xi(squad, row_by_element, self.immediate)
        captain_order = sorted(
            xi,
            key=lambda element: (
                self.captain[row_by_element[element]]
                if element in row_by_element
                else -1.0
            ),
            reverse=True,
        )
        captain, vice = captain_order[:2]
        score = lens.realised_week_breakdown(
            xi,
            bench,
            captain,
            vice,
            squad,
            row_by_element,
            self.actual,
            self.minutes,
        )["normal"]
        self.week_points[key] = float(score)
        return float(score)

    def label(
        self,
        season: str,
        gw: int,
        base: tuple[int, ...],
        candidate: tuple[int, ...],
        tenure: float,
        hit_cost: float,
    ) -> tuple[dict[int, float], float, int]:
        weeks = self.weeks[season]
        start = weeks.index(int(gw))
        horizon_values: dict[int, float] = {}
        cumulative = -float(hit_cost)
        end_gw = int(gw)
        for offset in range(max(HORIZONS)):
            if start + offset >= len(weeks):
                for horizon in HORIZONS:
                    horizon_values.setdefault(horizon, cumulative)
                break
            future_gw = weeks[start + offset]
            end_gw = int(future_gw)
            difference = self.score_week(season, future_gw, candidate) - self.score_week(
                season, future_gw, base
            )
            cumulative += (DISCOUNT**offset) * difference
            if offset + 1 in HORIZONS:
                horizon_values[offset + 1] = cumulative
        for horizon in HORIZONS:
            horizon_values.setdefault(horizon, cumulative)
        return horizon_values, interpolate_rollout(tenure, horizon_values), end_gw


def rollout_labels(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    features: np.ndarray,
    immediate: np.ndarray,
    captain: np.ndarray,
) -> dict[str, np.ndarray]:
    cache = lens.CACHE / f"rollout-action-labels-v{CACHE_VERSION}.npz"
    if cache.exists():
        loaded = np.load(cache)
        if len(loaded["adaptive"]) == len(metadata):
            return {key: loaded[key] for key in ("h1", "h3", "h6", "h10", "adaptive", "maturity", "tenure")}
    engine = FixedSquadRollout(data, immediate, captain)
    tenure_values = expected_tenure(data)
    row_lookup = {
        (str(row.season), int(row.GW), int(row.element)): int(row.Index)
        for row in data.itertuples()
    }
    labels = {f"h{horizon}": np.zeros(len(metadata), float) for horizon in HORIZONS}
    labels["adaptive"] = np.zeros(len(metadata), float)
    labels["maturity"] = np.zeros(len(metadata), int)
    labels["tenure"] = np.zeros(len(metadata), float)
    for index, row in enumerate(metadata.itertuples(index=False)):
        base = tuple(sorted(int(value) for value in row.baseSignature))
        outgoing = set(int(value) for value in row.outgoing)
        incoming = tuple(int(value) for value in row.incoming)
        candidate = tuple(sorted((set(base) - outgoing) | set(incoming)))
        incoming_rows = [
            row_lookup[(str(row.season), int(row.gw), element)]
            for element in incoming
            if (str(row.season), int(row.gw), element) in row_lookup
        ]
        tenure = (
            float(np.mean(tenure_values[incoming_rows])) if incoming_rows else 6.0
        )
        free_transfers = int(round(float(features[index, 2]) * 5.0))
        hit_cost = 4.0 * max(0, int(row.moves) - free_transfers)
        horizon, adaptive, maturity = engine.label(
            str(row.season),
            int(row.gw),
            base,
            candidate,
            tenure,
            hit_cost,
        )
        for value_horizon, value in horizon.items():
            labels[f"h{value_horizon}"][index] = value
        labels["adaptive"][index] = adaptive
        labels["maturity"][index] = maturity
        labels["tenure"][index] = tenure
        if (index + 1) % 10_000 == 0:
            print(f"Rollout-labelled {index + 1:,}/{len(metadata):,} actions", flush=True)
    np.savez_compressed(cache, **labels)
    return labels


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    component, _ = causal_route_predictions(data, immediate)
    action_plan = agreed_action_plan(data, plan, 0.05)
    target, maturity = target_and_maturity(data)
    builder = FeatureBuilder(
        data, immediate, plan, action_plan, component, target, maturity
    )
    source = np.load(lens.CACHE / "breakthrough-action-states-v1.npz")
    metadata = pd.DataFrame(json.loads(str(source["metadata"].item())))
    features = source["features"]
    labels = rollout_labels(data, metadata, features, immediate, captain)
    rollout_metadata = metadata.copy()
    rollout_metadata["targetGain"] = labels["adaptive"]
    rollout_metadata["targetResidual"] = (
        labels["adaptive"] - rollout_metadata["predictedGain"].to_numpy(float)
    )
    rollout_metadata["maturity"] = labels["maturity"].astype(int)
    fits, fit_audit = causal_fits(rollout_metadata, features)
    metrics = candidate_metrics(rollout_metadata, features, fits)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    print("Running rollout-action recursive control", flush=True)
    baseline, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    configurations = (
        (0.52, 0.00),
        (0.56, 0.00),
        (0.60, 0.00),
        (0.58, 0.02),
        (0.58, 0.05),
        (0.62, 0.05),
    )
    rows = []
    for probability, blend in configurations:
        policy = ConservativeAdvantagePolicy(
            builder,
            fits,
            probability_gate=probability,
            residual_blend=blend,
            correction_cap=2.0,
        )
        strategy = replace(
            STRATEGY,
            name=f"Rollout action p{probability:.2f} b{blend:.2f}",
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
        delta = totals[2:] - baseline[2:]
        rows.append(
            {
                "name": strategy.name,
                "probabilityGate": probability,
                "residualBlend": blend,
                **season_summary(totals, seasons),
                "averageDelta": round(float(delta.mean()), 1),
                "developmentDelta": round(float(delta[:-2].mean()), 1),
                "holdoutDelta": round(float(delta[-2:].mean()), 1),
                "worstSeasonDelta": int(delta.min()),
                "positiveSeasons": int((delta > 0).sum()),
                "negativeSeasons": int((delta < 0).sum()),
                "seasonDeltas": delta.astype(int).tolist(),
                "policyAudit": policy.audit(),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["developmentDelta"] - 0.20 * np.std(row["seasonDeltas"][:-2]),
            row["developmentDelta"],
            row["worstSeasonDelta"],
        ),
    )
    gate = {
        "developmentPositive": selected["developmentDelta"] > 0,
        "holdoutNonNegative": selected["holdoutDelta"] >= 0,
        "worstSeasonAtLeastMinusTen": selected["worstSeasonDelta"] >= -10,
        "positiveSeasonsAtLeastFive": selected["positiveSeasons"] >= 5,
    }
    result = {
        "schemaVersion": 1,
        "status": "realised fixed-squad rollout action challenger",
        "trainingStates": int(len(metadata)),
        "label": {
            "method": "Realised legal 1/3/6/10-event base-vs-transfer rollouts with autosubs, captain fallback, hit costs and player-specific tenure interpolation.",
            "meanTenure": round(float(labels["tenure"].mean()), 3),
            "meanAdaptiveGain": round(float(labels["adaptive"].mean()), 3),
            "positiveRate": round(float((labels["adaptive"] > 0).mean()), 4),
        },
        "candidateMetrics": metrics,
        "fitAudit": fit_audit,
        "baseline": season_summary(baseline, seasons),
        "selectedByDevelopmentOnly": selected,
        "gate": gate,
        "passed": all(gate.values()),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "rollout_action_value_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"label": result["label"], "metrics": metrics, "selected": selected, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()

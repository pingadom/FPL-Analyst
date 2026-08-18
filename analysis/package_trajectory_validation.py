"""Counterfactual multi-event replay for legal FPL transfer packages.

Each label forks the exact frozen manager state into Hold and one legal package,
then lets both branches make later transfers under the same frozen champion
policy.  The target is their discounted realised-point difference over a
predeclared three- or six-event horizon.  This captures bank, free-transfer and future-selection consequences
which a static player-value label cannot represent.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from decision_focused_horizon_validation import target_and_maturity
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from package_action_value_validation import (
    FeatureBuilder,
    FastLogistic,
    checkpoint_for_event,
    fit_fast_logistic,
)
from probabilistic_component_challenger import causal_route_predictions
from transfer_action_ranker_validation import agreed_action_plan
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1
HORIZON = int(os.environ.get("FPL_TRAJECTORY_HORIZON", "3"))
DISCOUNT = 0.90


def serialise_squad(squad: dict[int, dict]) -> list[list]:
    return [
        [
            int(element),
            int(state["position"]),
            int(state["team"]),
            int(state["purchase"]),
            int(state["last_price"]),
            str(state.get("nationality", "")),
        ]
        for element, state in sorted(squad.items())
    ]


def restore_squad(payload: list[list]) -> dict[int, dict]:
    return {
        int(element): {
            "position": int(position),
            "team": int(team),
            "purchase": int(purchase),
            "last_price": int(last_price),
            "nationality": str(nationality),
        }
        for element, position, team, purchase, last_price, nationality in payload
    }


def state_signature(squad: dict[int, dict]) -> tuple:
    return tuple(
        (
            int(element),
            int(state["position"]),
            int(state["team"]),
            int(state["purchase"]),
            int(state["last_price"]),
        )
        for element, state in sorted(squad.items())
    )


class TrajectoryEvaluator:
    def __init__(
        self,
        data: pd.DataFrame,
        scores: np.ndarray,
        plan: np.ndarray,
        captain: np.ndarray,
    ) -> None:
        self.data = data
        self.scores = scores
        self.plan = plan
        self.captain = captain
        self.actual = data["points"].to_numpy(float)
        self.minutes = data["minutes"].to_numpy(float)
        self.elements = data["element"].to_numpy(int)
        self.positions = data["position_id"].to_numpy(int)
        self.teams = data["team_id"].to_numpy(int)
        self.prices = data["price"].to_numpy(int)
        self.uncertainty = data["prediction_uncertainty"].to_numpy(float)
        self.price_rise = data["price_rise_probability"].to_numpy(float)
        self.price_fall = data["price_fall_probability"].to_numpy(float)
        self.nationality = data["nationality"].fillna("").to_numpy(str)
        self.context = {
            str(row["season"]): row
            for row in lens.simulation_context(data)["seasons"]
        }
        self.cache: dict[tuple, tuple[float, int]] = {}

    def exclusions(self, season: str, gw: int, indices: np.ndarray, squad: dict) -> set[int]:
        afcon_window = lens.AFCON_WINDOWS.get(season)
        active = bool(afcon_window and afcon_window[0] <= gw <= afcon_window[1])
        excluded = {
            int(self.elements[index])
            for index in indices
            if active and self.nationality[index] in lens.AFCON_NATIONS
        }
        if active:
            excluded.update(
                int(element)
                for element, state in squad.items()
                if str(state.get("nationality", "")) in lens.AFCON_NATIONS
            )
        return excluded

    def branch(
        self,
        season: str,
        start_gw: int,
        squad_payload: list[list],
        bank: int,
        free_transfers: int,
    ) -> tuple[float, int]:
        squad = restore_squad(squad_payload)
        signature = (
            season,
            int(start_gw),
            int(bank),
            int(free_transfers),
            state_signature(squad),
        )
        cached = self.cache.get(signature)
        if cached is not None:
            return cached
        season_context = self.context[season]
        weeks = season_context["weeks"]
        start = weeks.index(int(start_gw))
        selected_weeks = weeks[start : start + HORIZON]
        total = 0.0
        prior_gw = start_gw
        last_gw = start_gw
        for offset, gw in enumerate(selected_weeks):
            indices = season_context["weekIndices"][gw]
            row_by_element = dict(
                zip(self.elements[indices].tolist(), indices.tolist())
            )
            for element, state in squad.items():
                if element in row_by_element:
                    index = row_by_element[element]
                    state["team"] = int(self.teams[index])
                    state["last_price"] = int(self.prices[index])
                    state["nationality"] = str(self.nationality[index])
            excluded = self.exclusions(season, int(gw), indices, squad)
            bank_limit = min(
                STRATEGY.bank_limit,
                5 if season in {"2024-25", "2025-26"} else 2,
            )
            if offset > 0:
                if gw > prior_gw + 1:
                    free_transfers = min(
                        bank_limit, free_transfers + (gw - prior_gw - 1)
                    )
                if season == "2025-26" and gw == 16:
                    free_transfers = 5
                # Labels crossing an unlimited-rebuild deadline are not used;
                # preserving the incoming state here keeps the evaluator total
                # deterministic if such a row is inspected diagnostically.
                incoming: dict[int, np.ndarray] = {}
                for position in lens.SQUAD_QUOTAS:
                    local = indices[self.positions[indices] == position]
                    incoming[position] = local[np.argsort(self.plan[local])[::-1]][:40]
                squad, bank, changes, _ = lens.joint_transfer_plan(
                    squad=squad,
                    bank=bank,
                    free_transfers=free_transfers,
                    row_by_element=row_by_element,
                    incoming_by_position=incoming,
                    element_values=self.elements,
                    position_values=self.positions,
                    team_values=self.teams,
                    price_values=self.prices,
                    plan_scores=self.plan,
                    bench_scores=None,
                    captain_utility_scores=None,
                    price_rise_values=self.price_rise,
                    price_fall_values=self.price_fall,
                    uncertainty_values=self.uncertainty,
                    risk_scores=None,
                    excluded_elements=excluded,
                    team_option_score={},
                    strategy=STRATEGY,
                    gw=int(gw),
                )
                free_transfers = min(
                    bank_limit, max(0, free_transfers - changes) + 1
                )

            xi, bench = lens.choose_xi(
                squad, row_by_element, self.scores, excluded_elements=excluded
            )
            captain_order = sorted(
                xi,
                key=lambda element: self.captain[row_by_element[element]]
                if element in row_by_element and element not in excluded
                else -1e6,
                reverse=True,
            )
            if len(captain_order) >= 2:
                points = lens.realised_week_points(
                    xi,
                    bench,
                    captain_order[0],
                    captain_order[1],
                    squad,
                    row_by_element,
                    self.actual,
                    self.minutes,
                )
                total += (DISCOUNT**offset) * points
            if offset == 0:
                # The supplied state is after the current action. Advance its FT
                # state once current points have been scored.
                free_transfers = min(bank_limit, free_transfers + 1)
            prior_gw = gw
            last_gw = gw
        result = (float(total), int(last_gw))
        self.cache[signature] = result
        return result


def collect_trajectory_frontier(
    data: pd.DataFrame,
    scores: np.ndarray,
    plan: np.ndarray,
    captain: np.ndarray,
    builder: FeatureBuilder,
) -> tuple[pd.DataFrame, np.ndarray]:
    cache_path = lens.CACHE / (
        f"package-trajectories-h{HORIZON}-v{CACHE_VERSION}.npz"
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            pd.DataFrame(json.loads(str(cached["metadata"].item()))),
            cached["features"],
        )
    candidates: dict[tuple, dict] = {}

    def collector(context: dict) -> float:
        if int(context["moves"]) > 2:
            return 0.0
        vector, metadata = builder.vector(context)
        key = (
            metadata["seasonOrder"],
            metadata["gw"],
            metadata["outgoing"],
            metadata["incoming"],
        )
        if key not in candidates:
            candidates[key] = {
                **metadata,
                "predictedGain": float(context["predictedGain"]),
                "baseBank": int(context["baseBank"]),
                "candidateBank": int(context["candidateBank"]),
                "freeTransfers": int(context["freeTransfers"]),
                "baseSquad": serialise_squad(context["baseSquad"]),
                "candidateSquad": serialise_squad(context["candidateSquad"]),
                "feature": vector,
            }
        return 0.0

    print("Collecting compact counterfactual package frontier", flush=True)
    totals, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        package_action_adjustment=collector,
    )
    if round(float(totals[2:].mean()), 1) != 2174.9:
        raise AssertionError("Trajectory collector changed the frozen champion")

    season_orders = {
        str(season): int(order)
        for season, order in data[["season", "season_order"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    chosen: set[tuple] = set()
    for item in stats:
        for transfer in item["transferLog"]:
            chosen.add(
                (
                    season_orders[str(item["season"])],
                    int(transfer["gw"]),
                    tuple(sorted(int(value) for value in transfer["outElements"])),
                    tuple(sorted(int(value) for value in transfer["inElements"])),
                )
            )

    frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "feature"}
            for row in candidates.values()
        ]
    )
    feature_rows = np.vstack([row["feature"] for row in candidates.values()])
    frame["candidateKey"] = list(candidates)
    frame["chosenByChampion"] = frame["candidateKey"].isin(chosen)
    retained: set[int] = set(frame.index[frame["chosenByChampion"]].tolist())
    for _, group in frame.groupby(["seasonOrder", "gw", "moves"], sort=False):
        hurdle = STRATEGY.transfer_hurdle + STRATEGY.additional_move_hurdle * (
            int(group["moves"].iloc[0]) - 1
        )
        signed = group["predictedGain"] - hurdle
        below = group.loc[signed <= 0]
        above = group.loc[signed > 0]
        if len(below):
            retained.add(int((hurdle - below["predictedGain"]).idxmin()))
        if len(above):
            retained.add(int((above["predictedGain"] - hurdle).idxmin()))
    retained_indices = np.asarray(sorted(retained), dtype=int)
    frame = frame.loc[retained_indices].reset_index(drop=True)
    feature_rows = feature_rows[retained_indices]

    evaluator = TrajectoryEvaluator(data, scores, plan, captain)
    gains = []
    maturity = []
    valid = []
    for index, row in frame.iterrows():
        season = str(row["season"])
        base_ft = int(row["freeTransfers"])
        candidate_ft = max(0, base_ft - int(row["moves"]))
        base_points, base_end = evaluator.branch(
            season,
            int(row["gw"]),
            row["baseSquad"],
            int(row["baseBank"]),
            base_ft,
        )
        candidate_points, candidate_end = evaluator.branch(
            season,
            int(row["gw"]),
            row["candidateSquad"],
            int(row["candidateBank"]),
            candidate_ft,
        )
        gains.append(candidate_points - base_points)
        maturity.append(max(base_end, candidate_end))
        season_context = evaluator.context[season]
        selected_weeks = season_context["weeks"]
        start = selected_weeks.index(int(row["gw"]))
        horizon_weeks = selected_weeks[start : start + HORIZON]
        crosses_rebuild = any(
            gw in lens.UNLIMITED_TRANSFER_GWS.get(season, set())
            for gw in horizon_weeks[1:]
        )
        valid.append(not crosses_rebuild and len(horizon_weeks) == HORIZON)
        if (index + 1) % 100 == 0 or index + 1 == len(frame):
            print(f"Counterfactual trajectories {index + 1}/{len(frame)}", flush=True)
    frame["trajectoryGain"] = gains
    frame["trajectoryEndGw"] = maturity
    frame["validTrajectory"] = valid
    frame = frame.drop(columns=["candidateKey", "baseSquad", "candidateSquad"])
    valid_values = frame["validTrajectory"].to_numpy(bool)
    frame = frame.loc[valid_values].reset_index(drop=True)
    feature_rows = feature_rows[valid_values]
    np.savez_compressed(
        cache_path,
        metadata=json.dumps(frame.to_dict(orient="records")),
        features=feature_rows.astype(np.float32),
    )
    return frame, feature_rows


def causal_trajectory_fits(
    metadata: pd.DataFrame, features: np.ndarray
) -> tuple[dict[tuple[int, int], tuple[FastLogistic, FastLogistic]], list[dict]]:
    fits = {}
    audit = []
    orders = metadata["seasonOrder"].to_numpy(int)
    gws = metadata["gw"].to_numpy(int)
    maturity = metadata["trajectoryEndGw"].to_numpy(int)
    goalkeeper = metadata["goalkeeperPackage"].to_numpy(bool)
    target = metadata["trajectoryGain"].to_numpy(float) > 0
    for season_order in sorted(metadata["seasonOrder"].unique()):
        for checkpoint_index, checkpoint in enumerate((1, 13, 25)):
            end = 12 if checkpoint == 1 else 24 if checkpoint == 13 else 99
            test = (orders == season_order) & (gws >= checkpoint) & (gws <= end)
            train = (
                ((orders < season_order) | ((orders == season_order) & (maturity < checkpoint)))
                & ~goalkeeper
            )
            if not test.any() or train.sum() < 120 or len(np.unique(target[train])) < 2:
                continue
            age = season_order - orders[train]
            weights_a = np.power(0.88, np.maximum(age - 1, 0))
            weights_b = np.power(0.78, np.maximum(age - 1, 0))
            fits[(int(season_order), checkpoint)] = (
                fit_fast_logistic(features[train], target[train], weights_a, 0.04),
                fit_fast_logistic(features[train], target[train], weights_b, 0.02),
            )
            audit.append(
                {
                    "seasonOrder": int(season_order),
                    "checkpoint": checkpoint,
                    "trainingRows": int(train.sum()),
                    "maturedCurrentSeasonRows": int((train & (orders == season_order)).sum()),
                    "testRows": int(test.sum()),
                }
            )
    return fits, audit


class TrajectoryVeto:
    def __init__(
        self,
        builder: FeatureBuilder,
        fits: dict,
        threshold: float,
        penalty: float,
    ) -> None:
        self.builder = builder
        self.fits = fits
        self.threshold = threshold
        self.penalty = penalty
        self.calls = 0
        self.vetoes = 0
        self.abstentions = 0
        self.goalkeeper_abstentions = 0

    def __call__(self, context: dict) -> float:
        self.calls += 1
        vector, metadata = self.builder.vector(context)
        if metadata["goalkeeperPackage"]:
            self.goalkeeper_abstentions += 1
            return 0.0
        fitted = self.fits.get(
            (metadata["seasonOrder"], checkpoint_for_event(metadata["event"]))
        )
        if fitted is None:
            self.abstentions += 1
            return 0.0
        probability = np.asarray(
            [model.predict_probability(vector) for model in fitted]
        )
        if abs(probability[0] - probability[1]) > 0.18:
            self.abstentions += 1
            return 0.0
        if float(probability.max()) < self.threshold:
            self.vetoes += 1
            return -self.penalty
        return 0.0

    def audit(self) -> dict:
        return {
            "calls": self.calls,
            "vetoes": self.vetoes,
            "abstentions": self.abstentions,
            "goalkeeperAbstentions": self.goalkeeper_abstentions,
            "vetoRate": round(self.vetoes / max(1, self.calls), 4),
        }


def classification_audit(metadata: pd.DataFrame, features: np.ndarray, fits: dict) -> dict:
    probabilities = []
    targets = []
    chosen = []
    seasons = []
    for index, row in metadata.iterrows():
        fitted = fits.get(
            (int(row["seasonOrder"]), checkpoint_for_event(int(row["event"])))
        )
        if fitted is None or bool(row["goalkeeperPackage"]):
            continue
        probabilities.append(
            np.mean([model.predict_probability(features[index]) for model in fitted])
        )
        targets.append(float(row["trajectoryGain"]) > 0)
        chosen.append(bool(row["chosenByChampion"]))
        seasons.append(str(row["season"]))
    probability = np.asarray(probabilities)
    target = np.asarray(targets, dtype=bool)
    chosen_values = np.asarray(chosen, dtype=bool)
    evaluation = np.isin(seasons, lens.EVALUATION_SEASONS)

    def metrics(mask: np.ndarray) -> dict:
        return {
            "rows": int(mask.sum()),
            "positiveRate": round(float(target[mask].mean()), 4),
            "accuracy50": round(float(np.mean((probability[mask] >= 0.5) == target[mask])), 4),
            "alwaysMajorityAccuracy": round(float(max(target[mask].mean(), 1 - target[mask].mean())), 4),
            "brier": round(float(np.mean((probability[mask] - target[mask]) ** 2)), 4),
        }

    return {
        "evaluationFrontier": metrics(evaluation),
        "evaluationChampionChoices": metrics(evaluation & chosen_values),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, champion_plan, captain = champion_forecasts(data)
    component, _ = causal_route_predictions(data, scores)
    action_plan = agreed_action_plan(data, champion_plan, 0.05)
    target, maturity = target_and_maturity(data)
    builder = FeatureBuilder(
        data, scores, champion_plan, action_plan, component, target, maturity
    )
    metadata, features = collect_trajectory_frontier(
        data, scores, champion_plan, captain, builder
    )
    fits, fit_audit = causal_trajectory_fits(metadata, features)
    print(f"Trajectory rows {len(metadata):,}; causal fits {len(fits)}", flush=True)

    configs = ((0.15, 2.0), (0.20, 2.0), (0.25, 2.0), (0.30, 2.0), (0.35, 2.0), (0.30, 4.0))
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    print("Recursive trajectory champion", flush=True)
    baseline_totals, baseline_stats = lens.simulate_candidate(
        data, scores, STRATEGY, plan_scores=champion_plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    rows.append(
        {
            "name": "champion",
            "developmentStability": round(float(baseline_totals[2:6].mean() - 0.25 * baseline_totals[2:6].std()), 3),
            "holdoutAverage": round(float(baseline_totals[6:].mean()), 1),
            "summary": variant_summary(baseline_totals, baseline_stats, seasons),
            "policyAudit": None,
        }
    )
    for threshold, penalty in configs:
        name = f"trajectory-veto-p{int(threshold * 100):02d}-x{int(penalty):02d}"
        print(f"Recursive {name}", flush=True)
        policy = TrajectoryVeto(builder, fits, threshold, penalty)
        totals, stats = lens.simulate_candidate(
            data, scores, STRATEGY, plan_scores=champion_plan, captain_scores=captain,
            tracked_player_name="Salah", package_action_adjustment=policy,
        )
        rows.append(
            {
                "name": name,
                "parameters": {"threshold": threshold, "penalty": penalty},
                "developmentStability": round(float(totals[2:6].mean() - 0.25 * totals[2:6].std()), 3),
                "holdoutAverage": round(float(totals[6:].mean()), 1),
                "summary": variant_summary(totals, stats, seasons),
                "policyAudit": policy.audit(),
            }
        )

    baseline = rows[0]
    best_challenger = max(rows[1:], key=lambda row: row["developmentStability"])
    selected = max(rows, key=lambda row: row["developmentStability"])
    paired = [
        {
            "season": old["season"],
            "champion": old["points"],
            "challenger": new["points"],
            "delta": new["points"] - old["points"],
        }
        for old, new in zip(baseline["summary"]["seasons"], best_challenger["summary"]["seasons"])
    ]
    robust = bool(
        best_challenger["developmentStability"] > baseline["developmentStability"]
        and best_challenger["holdoutAverage"] >= baseline["holdoutAverage"]
        and best_challenger["summary"]["minimum"] >= baseline["summary"]["minimum"] - 10
        and sum(row["delta"] > 0 for row in paired) >= 5
    )
    result = {
        "status": "causal counterfactual package-trajectory challenger",
        "horizonEvents": HORIZON,
        "method": f"Fork Hold/package state, replay {HORIZON} legal events under the frozen policy, train two causal classifiers, and permit only an agreed negative-evidence veto. Goalkeeper packages abstain.",
        "selectionRule": "Select on 2018/19-2021/22 development stability, then require non-negative 2022/23-2025/26 holdout, downside and five-season breadth.",
        "trajectoryRows": len(metadata),
        "classificationAudit": classification_audit(metadata, features, fits),
        "fitAudit": fit_audit,
        "selected": selected,
        "bestChallenger": best_challenger,
        "pairedBestChallengerVsChampion": paired,
        "robustPromotion": robust,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / (
        f"package_trajectory_h{HORIZON}_validation.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "trajectoryRows": len(metadata),
                "classificationAudit": result["classificationAudit"],
                "selected": selected["name"],
                "bestChallenger": best_challenger["name"],
                "robustPromotion": robust,
                "paired": paired,
                "experiments": [
                    {
                        "name": row["name"],
                        "developmentStability": row["developmentStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                        "holdoutAverage": row["holdoutAverage"],
                        "policyAudit": row["policyAudit"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ), flush=True,
    )


if __name__ == "__main__":
    main()

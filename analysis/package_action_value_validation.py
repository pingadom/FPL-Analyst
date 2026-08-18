"""Causal value model for complete legal transfer packages.

The player rankers answer which individual looks stronger.  This experiment
answers the manager's actual question: is this whole one- or two-transfer bundle
better than holding, after accounting for the players sold, players bought,
bank, free-transfer state, formation, captain access and bench allocation?

Candidate packages are collected from the frozen champion path.  Their labels
use the already-audited player-specific 1/3/6/10-event holding target.  Every
deployed fit uses prior seasons plus only same-season labels whose complete
horizon has matured.  Two regularised models must agree, uncertain packages
abstain to the champion, and any goalkeeper package is left unchanged because
the preceding action audit found no goalkeeper regret gain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from decision_focused_horizon_validation import target_and_maturity
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from probabilistic_component_challenger import causal_route_predictions
from transfer_action_ranker_validation import agreed_action_plan
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 2
CHECKPOINTS = (1, 13, 25)
FEATURE_NAMES = (
    "predicted_gain",
    "moves",
    "free_transfers",
    "bank_before",
    "bank_after",
    "bank_change",
    "event",
    "incoming_uncertainty",
    "incoming_max_price",
    "outgoing_max_price",
    "premium_access",
    "incoming_club_concentration",
    "outgoing_club_concentration",
    "club_concentration_change",
    "changed_positions",
    "two_move_interaction",
)
PLAYER_METRIC_NAMES = (
    "immediate",
    "champion_plan",
    "action_plan",
    "route_total",
    "route_stacked",
    "route_sigma",
    "expected_minutes",
    "play_probability",
    "start_probability",
    "sixty_probability",
    "return5_probability",
    "haul8_probability",
    "goal_rate",
    "assist_rate",
    "team_attack",
    "team_defence",
    "team_clean",
    "opponent_defence",
    "price",
    "ownership",
    "fixture_count",
)


@dataclass(frozen=True)
class FastRidge:
    centre: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict_one(self, features: np.ndarray) -> float:
        normalised = (features - self.centre) / self.scale
        return float(self.intercept + normalised @ self.coefficient)


@dataclass(frozen=True)
class FastLogistic:
    centre: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict_probability(self, features: np.ndarray) -> float:
        normalised = (features - self.centre) / self.scale
        logit = float(self.intercept + normalised @ self.coefficient)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))


@dataclass(frozen=True)
class PackageFit:
    models: tuple[FastRidge, FastRidge]
    classifiers: tuple[FastLogistic, FastLogistic]
    magnitude_quantiles: dict[float, float]
    disagreement_limit: float
    training_rows: int


class FeatureBuilder:
    def __init__(
        self,
        data: pd.DataFrame,
        scores: np.ndarray,
        champion_plan: np.ndarray,
        action_plan: np.ndarray,
        component: dict,
        target: np.ndarray,
        maturity: np.ndarray,
    ) -> None:
        self.data = data
        self.scores = scores
        self.champion_plan = champion_plan
        self.target = target
        self.maturity = maturity
        self.events = event_number(data)
        self.positions = data["position_id"].to_numpy(int)
        ownership = np.log1p(data["selected"].clip(lower=0).to_numpy(float)) / 16.0
        self.player_metrics = np.column_stack(
            [
                scores / 8.0,
                champion_plan / 35.0,
                action_plan / 35.0,
                component["total"] / 8.0,
                component["stacked"] / 8.0,
                component["sigma"] / 6.0,
                data["expected_minutes"].to_numpy(float) / 90.0,
                data["play_probability"].to_numpy(float),
                data["start_probability"].to_numpy(float),
                data["sixty_probability"].to_numpy(float),
                data["return5_probability"].to_numpy(float),
                data["haul8_probability"].to_numpy(float),
                data["goal_rate"].to_numpy(float) / 0.5,
                data["assist_rate"].to_numpy(float) / 0.5,
                data["team_attack_rating"].to_numpy(float) / 2.0,
                data["team_defence_rating"].to_numpy(float) / 2.0,
                data["team_clean_probability"].to_numpy(float),
                data["opponent_defence_rating"].to_numpy(float) / 2.0,
                data["price"].to_numpy(float) / 150.0,
                ownership,
                data["fixture_count"].to_numpy(float) / 2.0,
            ]
        )

    @staticmethod
    def concentration(squad: dict[int, dict]) -> float:
        counts: dict[int, int] = {}
        for state in squad.values():
            team = int(state["team"])
            counts[team] = counts.get(team, 0) + 1
        return float(sum(count * (count - 1) / 2 for count in counts.values()))

    @staticmethod
    def indices(elements: tuple[int, ...], rows: dict[int, int]) -> np.ndarray:
        return np.asarray([rows[element] for element in elements if element in rows], dtype=int)

    def vector(self, context: dict) -> tuple[np.ndarray, dict]:
        rows = context["rowByElement"]
        incoming_elements = tuple(context["incomingElements"])
        outgoing_elements = tuple(context["outgoingElements"])
        incoming = self.indices(incoming_elements, rows)
        outgoing = self.indices(outgoing_elements, rows)
        any_index = next(iter(rows.values()))
        incoming_metrics = self.player_metrics[incoming]
        outgoing_metrics = self.player_metrics[outgoing]

        incoming_sum = incoming_metrics.sum(axis=0) if len(incoming) else np.zeros(len(PLAYER_METRIC_NAMES))
        outgoing_sum = outgoing_metrics.sum(axis=0) if len(outgoing) else np.zeros(len(PLAYER_METRIC_NAMES))
        delta = incoming_sum - outgoing_sum
        incoming_mean = incoming_metrics.mean(axis=0) if len(incoming) else np.zeros(len(PLAYER_METRIC_NAMES))
        outgoing_mean = outgoing_metrics.mean(axis=0) if len(outgoing) else np.zeros(len(PLAYER_METRIC_NAMES))

        incoming_prices = (
            self.data.loc[incoming, "price"].to_numpy(float) if len(incoming) else np.zeros(1)
        )
        outgoing_prices = (
            self.data.loc[outgoing, "price"].to_numpy(float) if len(outgoing) else np.zeros(1)
        )
        base_concentration = self.concentration(context["baseSquad"])
        candidate_concentration = self.concentration(context["candidateSquad"])
        incoming_positions = {int(self.positions[index]) for index in incoming}
        outgoing_positions = {int(self.positions[index]) for index in outgoing}
        changed_positions = len(incoming_positions | outgoing_positions)
        head = np.asarray(
            [
                float(context["predictedGain"]) / 30.0,
                float(context["moves"]) / 2.0,
                float(context["freeTransfers"]) / 5.0,
                float(context["baseBank"]) / 100.0,
                float(context["candidateBank"]) / 100.0,
                float(context["candidateBank"] - context["baseBank"]) / 100.0,
                float(self.events[any_index]) / 38.0,
                float(context["incomingUncertainty"]) / 10.0,
                float(incoming_prices.max()) / 150.0,
                float(outgoing_prices.max()) / 150.0,
                float(incoming_prices.max() - outgoing_prices.max()) / 100.0,
                candidate_concentration / 10.0,
                base_concentration / 10.0,
                (candidate_concentration - base_concentration) / 5.0,
                float(changed_positions) / 2.0,
                float(context["moves"] >= 2)
                * float(np.abs(delta[:6]).sum()),
            ],
            dtype=float,
        )
        # Delta carries the action; incoming/outgoing means let the learner
        # distinguish a premium swap from two cheap enablers with the same sum.
        vector = np.concatenate([head, delta, incoming_mean, outgoing_mean])
        union = set(context["baseSquad"]) | set(context["candidateSquad"])
        union_indices = self.indices(tuple(union), rows)
        metadata = {
            "seasonOrder": int(self.data.iloc[any_index]["season_order"]),
            "season": str(self.data.iloc[any_index]["season"]),
            "gw": int(context["gw"]),
            "event": int(self.events[any_index]),
            "maturity": int(self.maturity[union_indices].max()) if len(union_indices) else int(context["gw"]),
            "moves": int(context["moves"]),
            "goalkeeperPackage": bool(1 in (incoming_positions | outgoing_positions)),
            "incoming": incoming_elements,
            "outgoing": outgoing_elements,
        }
        return vector, metadata

    def target_residual(self, context: dict) -> tuple[float, float]:
        rows = context["rowByElement"]
        base = lens.squad_decision_utility(
            context["baseSquad"],
            rows,
            self.target,
            captain_weight=STRATEGY.squad_captain_weight,
            bench_weight=STRATEGY.squad_bench_weight,
        )
        candidate = lens.squad_decision_utility(
            context["candidateSquad"],
            rows,
            self.target,
            captain_weight=STRATEGY.squad_captain_weight,
            bench_weight=STRATEGY.squad_bench_weight,
        )
        target_gain = float(candidate - base)
        return target_gain - float(context["predictedGain"]), target_gain


def collect_packages(
    data: pd.DataFrame,
    scores: np.ndarray,
    champion_plan: np.ndarray,
    captain: np.ndarray,
    builder: FeatureBuilder,
) -> tuple[pd.DataFrame, np.ndarray]:
    cache = lens.CACHE / f"package-action-candidates-v{CACHE_VERSION}.npz"
    if cache.exists():
        loaded = np.load(cache)
        metadata = pd.DataFrame(json.loads(str(loaded["metadata"].item())))
        if loaded["features"].shape[1] == len(FEATURE_NAMES) + 3 * len(PLAYER_METRIC_NAMES):
            return metadata, loaded["features"]

    records: dict[tuple, dict] = {}

    def collector(context: dict) -> float:
        vector, metadata = builder.vector(context)
        key = (
            metadata["seasonOrder"],
            metadata["gw"],
            metadata["outgoing"],
            metadata["incoming"],
        )
        if key not in records:
            residual, target_gain = builder.target_residual(context)
            records[key] = {
                **metadata,
                "predictedGain": float(context["predictedGain"]),
                "targetGain": target_gain,
                "targetResidual": residual,
                "feature": vector,
            }
        return 0.0

    print("Collecting legal package frontier from frozen champion", flush=True)
    totals, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=champion_plan,
        captain_scores=captain,
        package_action_adjustment=collector,
    )
    # The zero-valued callback must be observational only.
    if round(float(totals[2:].mean()), 1) != 2174.9:
        raise AssertionError(f"Package collector changed frozen champion: {totals[2:].mean():.3f}")

    chosen_keys: set[tuple] = set()
    season_orders = {
        str(season): int(order)
        for season, order in data[["season", "season_order"]].drop_duplicates().itertuples(index=False)
    }
    for season_stats in stats:
        season = str(season_stats["season"])
        for move in season_stats["transferLog"]:
            chosen_keys.add(
                (
                    season_orders[season],
                    int(move["gw"]),
                    tuple(sorted(int(element) for element in move["outElements"])),
                    tuple(sorted(int(element) for element in move["inElements"])),
                )
            )
    ordered = list(records.values())
    for record in ordered:
        record["chosenByChampion"] = (
            record["seasonOrder"],
            record["gw"],
            record["outgoing"],
            record["incoming"],
        ) in chosen_keys
    features = np.vstack([record.pop("feature") for record in ordered]).astype(np.float32)
    metadata = pd.DataFrame(ordered)
    np.savez_compressed(
        cache,
        features=features,
        metadata=json.dumps(metadata.to_dict(orient="records")),
    )
    return metadata, features


def fit_fast_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> FastRidge:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    fitted = Ridge(alpha=alpha)
    fitted.fit(transformed, target, sample_weight=weights)
    scale = np.where(scaler.scale_ > 1e-8, scaler.scale_, 1.0)
    return FastRidge(
        centre=scaler.mean_.astype(float),
        scale=scale.astype(float),
        coefficient=fitted.coef_.astype(float),
        intercept=float(fitted.intercept_),
    )


def fit_fast_logistic(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    regularisation: float,
) -> FastLogistic:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    fitted = LogisticRegression(C=regularisation, max_iter=500)
    fitted.fit(transformed, target, sample_weight=weights)
    scale = np.where(scaler.scale_ > 1e-8, scaler.scale_, 1.0)
    return FastLogistic(
        centre=scaler.mean_.astype(float),
        scale=scale.astype(float),
        coefficient=fitted.coef_[0].astype(float),
        intercept=float(fitted.intercept_[0]),
    )


def causal_fits(metadata: pd.DataFrame, features: np.ndarray) -> tuple[dict, list[dict]]:
    fits: dict[tuple[int, int], PackageFit] = {}
    audit: list[dict] = []
    seasons = sorted(metadata["seasonOrder"].unique())
    target = metadata["targetResidual"].to_numpy(float).clip(-35, 35)
    target_gain = metadata["targetGain"].to_numpy(float)
    move_count = metadata["moves"].to_numpy(int)
    # First-stage question: does the package improve adaptive squad value at
    # all? The champion's existing hurdle continues to price the free-transfer
    # option; the classifier is only allowed to veto strong negative evidence.
    action_target = target_gain > 0.0
    orders = metadata["seasonOrder"].to_numpy(int)
    gws = metadata["gw"].to_numpy(int)
    maturity = metadata["maturity"].to_numpy(int)
    for season_order in seasons:
        if season_order == min(seasons):
            continue
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            end = CHECKPOINTS[checkpoint_index + 1] - 1 if checkpoint_index + 1 < len(CHECKPOINTS) else 99
            test_mask = season_mask & (gws >= checkpoint) & (gws <= end)
            if not test_mask.any():
                continue
            train_mask = (orders < season_order) | (
                season_mask & (maturity < checkpoint)
            )
            # Goalkeeper corrections were already falsified; exclude them from
            # both fitting and deployment rather than making the learner spend
            # capacity explaining a branch it may never use.
            train_mask &= ~metadata["goalkeeperPackage"].to_numpy(bool)
            if train_mask.sum() < 250:
                continue
            age = season_order - orders[train_mask]
            consequence = 1 + 0.03 * np.minimum(np.abs(target[train_mask]), 20)
            weights_a = np.power(0.88, np.maximum(age - 1, 0)) * consequence
            weights_b = np.power(0.78, np.maximum(age - 1, 0)) * consequence
            model_a = fit_fast_ridge(features[train_mask], target[train_mask], weights_a, 90.0)
            model_b = fit_fast_ridge(features[train_mask], target[train_mask], weights_b, 180.0)
            classifier_a = fit_fast_logistic(
                features[train_mask], action_target[train_mask], weights_a, 0.08
            )
            classifier_b = fit_fast_logistic(
                features[train_mask], action_target[train_mask], weights_b, 0.035
            )
            prediction_a = np.asarray([model_a.predict_one(row) for row in features[train_mask]])
            prediction_b = np.asarray([model_b.predict_one(row) for row in features[train_mask]])
            mean_prediction = 0.5 * (prediction_a + prediction_b)
            disagreement = np.abs(prediction_a - prediction_b)
            fits[(int(season_order), checkpoint)] = PackageFit(
                models=(model_a, model_b),
                classifiers=(classifier_a, classifier_b),
                magnitude_quantiles={
                    quantile: float(np.quantile(np.abs(mean_prediction), quantile))
                    for quantile in (0.50, 0.67, 0.80)
                },
                disagreement_limit=float(np.quantile(disagreement, 0.75)),
                training_rows=int(train_mask.sum()),
            )
            audit.append(
                {
                    "seasonOrder": int(season_order),
                    "checkpoint": checkpoint,
                    "trainingRows": int(train_mask.sum()),
                    "maturedCurrentSeasonRows": int((train_mask & season_mask).sum()),
                    "testRows": int(test_mask.sum()),
                }
            )
    return fits, audit


def checkpoint_for_event(event: int) -> int:
    if event >= 25:
        return 25
    if event >= 13:
        return 13
    return 1


class PackagePolicy:
    def __init__(
        self,
        builder: FeatureBuilder,
        fits: dict,
        probability_threshold: float,
        veto_penalty: float,
        max_moves: int = 2,
    ) -> None:
        self.builder = builder
        self.fits = fits
        self.probability_threshold = probability_threshold
        self.veto_penalty = veto_penalty
        self.max_moves = max_moves
        self.calls = 0
        self.abstained = 0
        self.goalkeeper_abstained = 0

    def __call__(self, context: dict) -> float:
        self.calls += 1
        vector, metadata = self.builder.vector(context)
        if metadata["goalkeeperPackage"]:
            self.goalkeeper_abstained += 1
            return 0.0
        if metadata["moves"] > self.max_moves:
            self.abstained += 1
            return 0.0
        fitted = self.fits.get(
            (metadata["seasonOrder"], checkpoint_for_event(metadata["event"]))
        )
        if fitted is None:
            self.abstained += 1
            return 0.0
        probabilities = np.asarray(
            [model.predict_probability(vector) for model in fitted.classifiers]
        )
        # This is a conservative veto, never a learned excuse to manufacture a
        # transfer that the champion would otherwise reject. Both causal models
        # must clear the action threshold; disagreement defaults to Hold.
        if abs(probabilities[0] - probabilities[1]) > 0.18:
            self.abstained += 1
            return 0.0
        if float(probabilities.min()) < self.probability_threshold:
            return -self.veto_penalty
        return 0.0

    def audit(self) -> dict:
        return {
            "calls": self.calls,
            "abstained": self.abstained,
            "abstentionRate": round(self.abstained / max(1, self.calls), 4),
            "goalkeeperAbstained": self.goalkeeper_abstained,
        }


def candidate_metrics(
    metadata: pd.DataFrame,
    features: np.ndarray,
    fits: dict,
) -> dict:
    actual = metadata["targetGain"].to_numpy(float)
    structural = metadata["predictedGain"].to_numpy(float)
    orders = metadata["seasonOrder"].to_numpy(int)
    events = metadata["event"].to_numpy(int)
    goalkeeper = metadata["goalkeeperPackage"].to_numpy(bool)
    corrected = structural.copy()
    probabilities = np.full(len(metadata), 0.5, dtype=float)
    covered = np.zeros(len(metadata), dtype=bool)
    for index, row in metadata.iterrows():
        if goalkeeper[index]:
            continue
        fitted = fits.get((orders[index], checkpoint_for_event(events[index])))
        if fitted is None:
            continue
        residual = np.mean([model.predict_one(features[index]) for model in fitted.models])
        corrected[index] += residual
        probabilities[index] = np.mean(
            [model.predict_probability(features[index]) for model in fitted.classifiers]
        )
        covered[index] = True
    evaluation = covered & metadata["season"].isin(lens.EVALUATION_SEASONS).to_numpy()
    action = actual > 0.0
    return {
        "rows": int(evaluation.sum()),
        "structuralMae": round(float(np.mean(np.abs(structural[evaluation] - actual[evaluation]))), 3),
        "correctedMae": round(float(np.mean(np.abs(corrected[evaluation] - actual[evaluation]))), 3),
        "structuralPositiveAccuracy": round(float(np.mean((structural[evaluation] > 0) == (actual[evaluation] > 0))), 4),
        "correctedPositiveAccuracy": round(float(np.mean((corrected[evaluation] > 0) == (actual[evaluation] > 0))), 4),
        "structuralCorrelation": round(float(np.corrcoef(structural[evaluation], actual[evaluation])[0, 1]), 4),
        "correctedCorrelation": round(float(np.corrcoef(corrected[evaluation], actual[evaluation])[0, 1]), 4),
        "actionRate": round(float(action[evaluation].mean()), 4),
        "actionBrier": round(float(np.mean((probabilities[evaluation] - action[evaluation]) ** 2)), 4),
        "actionAccuracy50": round(float(np.mean((probabilities[evaluation] >= 0.50) == action[evaluation])), 4),
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
    metadata, features = collect_packages(
        data, scores, champion_plan, captain, builder
    )
    fits, fit_audit = causal_fits(metadata, features)
    print(f"Collected {len(metadata):,} legal packages; fitted {len(fits)} causal windows", flush=True)

    configurations = [
        (0.15, 2.0, 1),
        (0.20, 2.0, 1),
        (0.20, 2.0, 2),
        (0.25, 2.0, 2),
        (0.30, 2.0, 2),
        (0.35, 2.0, 2),
        (0.30, 4.0, 2),
    ]
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []

    print("Recursive package baseline", flush=True)
    baseline_totals, baseline_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=champion_plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline_summary = variant_summary(baseline_totals, baseline_stats, seasons)
    rows.append(
        {
            "name": "champion",
            "developmentStability": round(float(baseline_totals[2:6].mean() - 0.25 * baseline_totals[2:6].std()), 3),
            "holdoutAverage": round(float(baseline_totals[6:].mean()), 1),
            "summary": baseline_summary,
            "policyAudit": None,
        }
    )
    for probability, penalty, max_moves in configurations:
        name = f"veto-p{int(probability * 100):02d}-x{int(penalty):02d}-m{max_moves}"
        print(f"Recursive {name}", flush=True)
        policy = PackagePolicy(builder, fits, probability, penalty, max_moves)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=champion_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
            package_action_adjustment=policy,
        )
        rows.append(
            {
                "name": name,
                "parameters": {"probabilityThreshold": probability, "vetoPenalty": penalty, "maxMoves": max_moves},
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
        for old, new in zip(
            baseline["summary"]["seasons"], best_challenger["summary"]["seasons"]
        )
    ]
    robust = bool(
        best_challenger["developmentStability"] > baseline["developmentStability"]
        and best_challenger["holdoutAverage"] >= baseline["holdoutAverage"]
        and best_challenger["summary"]["minimum"] >= baseline["summary"]["minimum"] - 10
        and sum(row["delta"] > 0 for row in paired) >= 5
    )
    result = {
        "status": "causal legal transfer-package challenger",
        "method": "Complete one/two-transfer packages are labelled by formation-aware adaptive holding value. Two causal classifiers must agree the package has positive value; otherwise strong negative evidence can only veto a marginal champion move. The champion hurdle still prices transfer option value, the learner can never manufacture a transfer, and goalkeeper packages abstain.",
        "selectionRule": "Choose on 2018/19-2021/22 development stability only; require non-negative 2022/23-2025/26 holdout, downside and five-season breadth for promotion.",
        "features": list(FEATURE_NAMES) + [f"delta_{name}" for name in PLAYER_METRIC_NAMES] + [f"incoming_{name}" for name in PLAYER_METRIC_NAMES] + [f"outgoing_{name}" for name in PLAYER_METRIC_NAMES],
        "candidateMetrics": candidate_metrics(metadata, features, fits),
        "fitAudit": fit_audit,
        "selected": selected,
        "bestChallenger": best_challenger,
        "pairedBestChallengerVsChampion": paired,
        "robustPromotion": robust,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "package_action_value_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidateMetrics": result["candidateMetrics"],
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

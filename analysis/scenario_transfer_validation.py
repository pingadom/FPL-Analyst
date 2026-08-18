"""Correlated-scenario transfer policy on the frozen champion forecast.

The forecast mean is not changed.  At every legal transfer-package boundary we
draw deterministic, antithetic scenarios whose scale reflects calibrated point
uncertainty and disagreement between the new causal minutes, team-defence and
tactical-role models.  A package may be vetoed when it does not clear the
ordinary transfer hurdle in enough plausible scenarios; optional lower-tail
penalties value downside without hindsight outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from tactical_role_challenger import baseline_attack, causal_prediction as role_prediction
from team_defence_challenger import causal_predictions as team_clean_predictions
from wildcard_freehit_ablation import champion_forecasts


SCENARIOS = 128
CONFIGS = (
    ("win45", 0.45, 0.00),
    ("win55", 0.55, 0.00),
    ("win65", 0.65, 0.00),
    ("win45-tail10", 0.45, 0.10),
    ("win55-tail05", 0.55, 0.05),
    ("win55-tail10", 0.55, 0.10),
    ("win65-tail10", 0.65, 0.10),
)


class CorrelatedScenarioGenerator:
    def __init__(
        self,
        data: pd.DataFrame,
        immediate: np.ndarray,
        plan: np.ndarray,
    ) -> None:
        self.data = data
        self.immediate = immediate
        self.plan = plan
        minute = minute_predictions(data)
        team_clean = team_clean_predictions(data)
        role_attack = role_prediction(data)
        old_attack = baseline_attack(data)
        old_minutes = data["expected_minutes"].to_numpy(float)
        minute_gap = (
            np.abs(minute["minutes"] - old_minutes)
            / 90.0
            * np.maximum(immediate, 1.0)
            * 2.0
        )
        clean_points = data["position_id"].map({1: 4.0, 2: 4.0, 3: 1.0, 4: 0.0}).to_numpy(float)
        clean_gap = (
            0.82
            * np.abs(team_clean - data["team_clean_probability"].to_numpy(float))
            * clean_points
            * data["sixty_probability"].to_numpy(float)
        )
        role_gap = np.abs(role_attack - old_attack)
        structural = (
            0.72 * data["prediction_uncertainty"].to_numpy(float) * np.sqrt(4.5)
        )
        ensemble = 1.8 * data["ensemble_disagreement"].to_numpy(float)
        self.sigma = np.sqrt(
            structural**2
            + ensemble**2
            + (2.2 * minute_gap) ** 2
            + (2.0 * clean_gap) ** 2
            + (1.8 * role_gap) ** 2
        ).clip(1.5, 14.0)
        self.week_cache: dict[tuple[str, int], tuple[dict[int, int], np.ndarray]] = {}

    def week(self, any_index: int) -> tuple[dict[int, int], np.ndarray]:
        season = str(self.data.at[any_index, "season"])
        gw = int(self.data.at[any_index, "GW"])
        key = (season, gw)
        cached = self.week_cache.get(key)
        if cached is not None:
            return cached
        indices = self.data.index[
            self.data["season"].eq(season) & self.data["GW"].eq(gw)
        ].to_numpy(int)
        half = SCENARIOS // 2
        season_order = int(self.data.at[any_index, "season_order"])
        rng = np.random.default_rng(920000 + 100 * season_order + gw)

        def antithetic(rows: int) -> np.ndarray:
            values = rng.standard_normal((rows, half))
            return np.concatenate([values, -values], axis=1)

        global_factor = antithetic(1)[0]
        position_factor = antithetic(4)
        teams = sorted(self.data.loc[indices, "team_id"].astype(int).unique())
        team_lookup = {team: offset for offset, team in enumerate(teams)}
        team_factor = antithetic(len(teams))
        independent = antithetic(len(indices))
        residual = np.zeros((len(indices), SCENARIOS), dtype=np.float32)
        for local, index in enumerate(indices):
            position = int(self.data.at[index, "position_id"])
            team = int(self.data.at[index, "team_id"])
            is_defence = position <= 2
            team_weight = 0.42 if is_defence else 0.28
            position_weight = 0.10
            global_weight = 0.08
            independent_weight = np.sqrt(
                1 - team_weight**2 - position_weight**2 - global_weight**2
            )
            standard = (
                global_weight * global_factor
                + position_weight * position_factor[position - 1]
                + team_weight * team_factor[team_lookup[team]]
                + independent_weight * independent[local]
            )
            residual[local] = (self.sigma[index] * standard).astype(np.float32)
        lookup = {int(index): local for local, index in enumerate(indices)}
        self.week_cache[key] = (lookup, residual)
        return self.week_cache[key]


@dataclass
class ScenarioRecord:
    season: str
    gw: int
    moves: int
    outgoing: tuple[int, ...]
    incoming: tuple[int, ...]
    mean_gain: float
    hurdle: float
    clear_probability: float
    lower_tail_mean: float
    scenario_std: float


class ScenarioTransferPolicy:
    def __init__(
        self,
        data: pd.DataFrame,
        plan: np.ndarray,
        generator: CorrelatedScenarioGenerator,
        clear_threshold: float,
        tail_share: float,
        observe_only: bool = False,
        clear_penalty: float = 0.0,
    ) -> None:
        self.data = data
        self.plan = plan
        self.generator = generator
        self.clear_threshold = clear_threshold
        self.tail_share = tail_share
        self.observe_only = observe_only
        self.clear_penalty = clear_penalty
        self.records: dict[tuple, ScenarioRecord] = {}
        self.weight_cache: dict[tuple[str, int, tuple[int, ...]], dict[int, float]] = {}

    def hurdle(self, context: dict) -> float:
        moves = int(context["moves"])
        hurdle = STRATEGY.transfer_hurdle + STRATEGY.additional_move_hurdle * (moves - 1)
        if STRATEGY.hold_option_value > 0:
            uncertainty = float(context["incomingUncertainty"])
            hurdle += STRATEGY.hold_option_value * min(1.5, uncertainty / max(1, moves) / 3.0)
            hurdle += STRATEGY.hold_option_value if int(context["freeTransfers"]) <= 1 else -0.35 * STRATEGY.hold_option_value
        return float(hurdle)

    def weights(self, squad: dict[int, dict], rows: dict[int, int]) -> dict[int, float]:
        any_index = next(iter(rows.values()))
        key = (
            str(self.data.at[any_index, "season"]),
            int(self.data.at[any_index, "GW"]),
            tuple(sorted(squad)),
        )
        cached = self.weight_cache.get(key)
        if cached is not None:
            return cached
        xi, bench = lens.choose_xi(squad, rows, self.plan)
        result = {int(element): STRATEGY.squad_bench_weight for element in bench}
        for element in xi:
            result[int(element)] = 1.0
        captain = max(xi, key=lambda element: self.plan[rows[element]])
        result[int(captain)] += STRATEGY.squad_captain_weight
        self.weight_cache[key] = result
        return result

    def __call__(self, context: dict) -> float:
        rows = context["rowByElement"]
        any_index = next(iter(rows.values()))
        lookup, residual = self.generator.week(any_index)
        base_weight = self.weights(context["baseSquad"], rows)
        candidate_weight = self.weights(context["candidateSquad"], rows)
        scenario_delta = np.zeros(SCENARIOS, dtype=float)
        for element in set(base_weight) | set(candidate_weight):
            index = rows.get(int(element))
            if index is None or index not in lookup:
                continue
            marginal = candidate_weight.get(int(element), 0.0) - base_weight.get(int(element), 0.0)
            if marginal:
                scenario_delta += marginal * residual[lookup[index]]
        mean_gain = float(context["predictedGain"])
        scenario_gain = mean_gain + scenario_delta
        hurdle = self.hurdle(context)
        clear_probability = float(np.mean(scenario_gain > hurdle))
        tail_count = max(1, SCENARIOS // 4)
        lower_tail_mean = float(np.sort(scenario_gain)[:tail_count].mean())
        record = ScenarioRecord(
            season=str(self.data.at[any_index, "season"]),
            gw=int(context["gw"]),
            moves=int(context["moves"]),
            outgoing=tuple(int(value) for value in context["outgoingElements"]),
            incoming=tuple(int(value) for value in context["incomingElements"]),
            mean_gain=mean_gain,
            hurdle=hurdle,
            clear_probability=clear_probability,
            lower_tail_mean=lower_tail_mean,
            scenario_std=float(np.std(scenario_gain)),
        )
        key = (record.season, record.gw, record.outgoing, record.incoming)
        self.records[key] = record
        if self.observe_only:
            return 0.0
        if clear_probability < self.clear_threshold:
            return -1_000_000.0
        robust_gain = (
            mean_gain
            - self.tail_share * max(0.0, mean_gain - lower_tail_mean)
            - self.clear_penalty * (1.0 - clear_probability)
        )
        return float(robust_gain - mean_gain)


def development_stability(totals: np.ndarray) -> float:
    values = totals[2:8]
    return float(values.mean() - 0.25 * values.std())


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    generator = CorrelatedScenarioGenerator(data, immediate, plan)

    print("Scenario observer on frozen champion", flush=True)
    observer = ScenarioTransferPolicy(data, plan, generator, 0.0, 0.0, observe_only=True)
    base_totals, base_stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
        package_action_adjustment=observer,
    )
    if round(float(base_totals[2:].mean()), 1) != 2174.9:
        raise AssertionError(f"Scenario observer changed frozen champion: {base_totals[2:].mean():.3f}")
    baseline = variant_summary(base_totals, base_stats, seasons)
    chosen_records = []
    for season_index, stats in enumerate(base_stats):
        season = str(stats["season"])
        for action in stats["transferLog"]:
            key = (
                season,
                int(action["gw"]),
                tuple(sorted(int(value) for value in action["outElements"])),
                tuple(sorted(int(value) for value in action["inElements"])),
            )
            record = observer.records.get(key)
            if record is not None:
                chosen_records.append(record)
    clear_values = np.asarray([record.clear_probability for record in chosen_records], dtype=float)
    scenario_audit = {
        "chosenPackages": len(chosen_records),
        "meanClearProbability": round(float(clear_values.mean()), 4),
        "quantiles": {
            str(q): round(float(np.quantile(clear_values, q)), 4)
            for q in [0.10, 0.25, 0.50, 0.75, 0.90]
        },
        "belowThreshold": {
            str(threshold): int(np.sum(clear_values < threshold))
            for threshold in [0.45, 0.55, 0.65]
        },
        "meanScenarioStd": round(float(np.mean([record.scenario_std for record in chosen_records])), 3),
    }
    print(json.dumps(scenario_audit, indent=2), flush=True)

    rows = []
    for name, threshold, tail_share in CONFIGS:
        print(f"Recursive scenario policy {name}", flush=True)
        policy = ScenarioTransferPolicy(data, plan, generator, threshold, tail_share)
        totals, stats = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
            package_action_adjustment=policy,
        )
        summary = variant_summary(totals, stats, seasons)
        deltas = [
            challenger["points"] - frozen["points"]
            for challenger, frozen in zip(summary["seasons"], baseline["seasons"])
        ]
        rows.append(
            {
                "name": name,
                "clearThreshold": threshold,
                "tailShare": tail_share,
                "developmentStability": round(development_stability(totals), 3),
                "holdoutAverage": round(float(totals[8:].mean()), 1),
                "summary": summary,
                "averageDelta": round(summary["average"] - baseline["average"], 1),
                "minimumDelta": summary["minimum"] - baseline["minimum"],
                "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
                "unchangedSeasons": int(sum(delta == 0 for delta in deltas)),
                "worseSeasons": int(sum(delta < 0 for delta in deltas)),
                "seasonDeltas": deltas,
            }
        )
        print(name, rows[-1]["averageDelta"], deltas, flush=True)

    baseline_stability = development_stability(base_totals)
    baseline_holdout = float(base_totals[8:].mean())
    selected = max(rows, key=lambda row: row["developmentStability"])
    robust_promotion = bool(
        selected["developmentStability"] > baseline_stability
        and selected["holdoutAverage"] >= baseline_holdout + 5
        and selected["minimumDelta"] >= 0
        and selected["improvedSeasons"] >= 5
    )
    result = {
        "status": "promoted" if robust_promotion else "research-only; robust promotion gate failed",
        "method": (
            "Frozen forecast means; 128 deterministic antithetic scenarios with global, "
            "position, club and player shocks. Uncertainty expands with causal minutes, "
            "team-defence and tactical-role disagreement. The callback is default-off."
        ),
        "scenarioAudit": scenario_audit,
        "baselineDevelopmentStability": round(baseline_stability, 3),
        "baselineHoldoutAverage": round(baseline_holdout, 1),
        "baseline": baseline,
        "variants": rows,
        "selectedByDevelopment": selected,
        "robustPromotion": robust_promotion,
    }
    output = lens.ROOT / "analysis" / "data" / "scenario_transfer_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": baseline["average"],
        "selected": selected["name"],
        "selectedAverage": selected["summary"]["average"],
        "selectedMinimum": selected["summary"]["minimum"],
        "selectedHoldout": selected["holdoutAverage"],
        "robustPromotion": robust_promotion,
    }, indent=2))


if __name__ == "__main__":
    main()

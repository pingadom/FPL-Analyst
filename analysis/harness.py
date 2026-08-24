"""Controlled-experiment harness for the FPL Lens decision model.

`calibrate_model.main()` interleaves five jobs — fitting the forecast, searching
weights, selecting a decision policy, evaluating seasons, and publishing the site
artifact — and every stage feeds the next. That makes it impossible to change one
thing: a single edit re-searches 2,400 weight mixtures, 80 candidates, 48 chip
policies and a strategy gate, and any of those can move in response.

That is not hypothetical. Five separate couplings were found this way, each
invisible until it cost points:

* decision thresholds were absolute point values, so a forecast with different
  dispersion silently redefined every one of them;
* the strategy gate judged policies under one candidate while seasons were scored
  under another, and the ranking flips between candidates;
* the gate drew its chip policy from the searched pool, so editing the chip
  search space moved a policy switch worth 200 points in a season;
* allowing a paid hit changed the incumbent's weekly profile enough to stop that
  same switch firing;
* a searched range excluded its own optimum, leaving 11 of 20 Wildcards unplayable.

This module exists so a change can be measured against a *pinned* reference. It
does not re-search anything. Given a configuration it replays deterministically,
and it reports differences with the uncertainty attached, because effects here are
routinely smaller than the noise in measuring them.

One more caveat: `reference_config()` reads the shipped artifact, so it tracks
production and therefore *moves when production moves*. Baseline and variants
within a single invocation share a configuration and are comparable; numbers from
two invocations spanning a pipeline run are not.

What this is not
----------------
A harness result is a *screen*, not a release decision. It replays one pinned
configuration; `main()` re-selects the candidate and the chip policy for every
season, so the two answer different questions and can disagree. The reference
here scores 2171.4 across the evaluation seasons where the published
walk-forward scores 2140.9, and neither is wrong.

That gap matters because it is the residual of the very problem this module
addresses. Pinning removes uncontrolled *variation*; it cannot remove pipeline
*interaction*. A change that looks good here can still lose end to end by
disturbing something downstream — enabling a paid hit measured +82.5 on the
training seasons with the strategy held fixed, then cost 20.8 a season through
the full pipeline because it stopped the decision gate switching policy. So:
screen here, confirm with a full run, and never ship on a harness number alone.

Typical use:

    python analysis/harness.py reference
    python analysis/harness.py sweep --field transfer_hurdle --values 5,3,1.5
    python analysis/harness.py compare --field max_hits --values 0,1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_model as lens


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "data" / "harness_runs.json"


# ---------------------------------------------------------------------------
# The reference configuration
# ---------------------------------------------------------------------------
# Everything an evaluation depends on, in one object. If a field is not here, it
# is not allowed to vary between two runs being compared.


@dataclass(frozen=True)
class Config:
    """A fully pinned evaluation. Two Configs differing in one field differ in
    exactly one thing — that is the whole point of this module."""

    candidate: lens.Candidate
    strategy: lens.SimulationStrategy
    chip_policy: lens.ChipPolicy | None
    robust_planning: bool = False
    schedule_censored: bool = True
    label: str = "reference"

    def with_field(self, name: str, value: object) -> "Config":
        """Return a copy with one field changed, wherever that field lives."""
        if name in {"robust_planning", "schedule_censored", "label"}:
            return replace(self, **{name: value})
        if hasattr(self.strategy, name):
            return replace(
                self,
                strategy=replace(self.strategy, **{name: value}),
                label=f"{name}={value}",
            )
        if self.chip_policy is not None and hasattr(self.chip_policy, name):
            return replace(
                self,
                chip_policy=replace(self.chip_policy, **{name: value}),
                label=f"{name}={value}",
            )
        if hasattr(self.candidate, name):
            return replace(
                self,
                candidate=replace(self.candidate, **{name: value}),
                label=f"{name}={value}",
            )
        raise KeyError(f"No field {name!r} on strategy, chip policy or candidate")


def reference_config() -> Config:
    """The configuration the shipped model actually uses.

    Read from the published artifact so the reference tracks production rather
    than drifting into a hand-maintained copy of it.
    """
    payload = json.loads(
        (ROOT / "app" / "data" / "model-results.json").read_text(encoding="utf-8-sig")
    )
    weights = payload["model"]["weights"]
    candidate = lens.Candidate(
        weights["performance"] / 100,
        weights["value"] / 100,
        weights["age"] / 100,
        weights["fixture"] / 100,
        weights.get("team", 0) / 100,
        weights["crowd"] / 100,
        weights["minutes"] / 100,
        weights["underlying"] / 100,
        weights["recent"] / 100,
    )
    strategy = (
        lens.JOINT_OPTION_STRATEGY
        if str(payload["model"].get("strategy", "")) == lens.JOINT_OPTION_STRATEGY.name
        else lens.WEEKLY_CHASE_STRATEGY
    )
    chips = (payload.get("chipStrategy") or {}).get("policy") or {}
    chip_policy = (
        lens.ChipPolicy(
            wildcard_gap=float(chips["wildcardGap"]),
            free_hit_gap=float(chips["freeHitGap"]),
            bench_score=float(chips["benchScore"]),
            triple_score=float(chips["tripleScore"]),
            afcon_bonus=float(chips["afconBonus"]),
            first_wildcard_min_gw=int(chips["firstWildcardMinGw"]),
            second_wildcard_min_gw=int(chips["secondWildcardMinGw"]),
        )
        if chips
        else None
    )
    return Config(
        candidate=candidate,
        strategy=strategy,
        chip_policy=chip_policy,
        robust_planning=bool(payload["model"].get("robustPlanningEnabled", False)),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    label: str
    totals: np.ndarray
    weekly: list[list[float]]
    stats: list[dict]

    @property
    def training(self) -> float:
        return float(self.totals[: len(lens.TRAINING_SEASONS)].mean())

    @property
    def evaluation(self) -> float:
        return float(self.totals[len(lens.TRAINING_SEASONS) :].mean())

    @property
    def overall(self) -> float:
        return float(self.totals.mean())


_SQUAD_CACHE: dict[tuple[int, bool], dict] = {}


def _fresh_squads(data: pd.DataFrame, scores: np.ndarray, one_week: bool) -> dict:
    """Cache the Free Hit / Wildcard rebuilds — they dominate sweep runtime and
    depend only on the score vector."""
    key = (int(pd.util.hash_pandas_object(pd.Series(scores)).sum()), one_week)
    cached = _SQUAD_CACHE.get(key)
    if cached is None:
        cached = lens.precompute_fresh_squads(data, scores, one_week_only=one_week)
        _SQUAD_CACHE[key] = cached
    return cached


def evaluate(config: Config, data: pd.DataFrame) -> Outcome:
    """Replay one pinned configuration. Deterministic: no search, no selection."""
    scores, plan, _ = lens.candidate_forecasts(
        data,
        config.candidate,
        robust_planning=config.robust_planning,
        schedule_censored=config.schedule_censored,
    )
    keywords: dict = {"plan_scores": plan}
    if config.chip_policy is not None:
        keywords.update(
            chip_policy=config.chip_policy,
            fresh_squads=_fresh_squads(data, plan, False),
            free_hit_squads=_fresh_squads(data, scores, True),
        )
    totals, stats = lens.simulate_candidate(data, scores, config.strategy, **keywords)
    return Outcome(
        label=config.label,
        totals=totals,
        weekly=[list(season["weeklyPoints"]) for season in stats],
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Comparison, with the uncertainty attached
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    label: str
    delta_training: float
    delta_evaluation: float
    delta_overall: float
    standard_error: float
    confidence: float

    @property
    def verdict(self) -> str:
        """Effects here are routinely smaller than the noise around them, so a
        point estimate on its own is not a result."""
        if self.standard_error == 0.0:
            # Every bootstrap draw was identical, so the configurations either
            # never diverged or diverged deterministically. Neither is "unresolved".
            if abs(self.delta_training) < 1e-9:
                return "no effect on the selecting seasons"
            return "better" if self.delta_training > 0 else "worse"
        if abs(self.delta_training) < self.standard_error:
            return "indistinguishable from noise"
        if self.confidence >= 0.75:
            return "better"
        if self.confidence <= 0.25:
            return "worse"
        return "unresolved"


def compare(
    baseline: Outcome, challenger: Outcome, training_only: bool = True
) -> Comparison:
    """Paired block bootstrap on weekly differences.

    Pairing cancels the shocks both configurations shared; four-Gameweek blocks
    keep a season's streakiness intact. Selection uses the training seasons by
    default, because choosing on evaluation seasons is how a backtest flatters
    itself.
    """
    count = len(lens.TRAINING_SEASONS) if training_only else len(baseline.weekly)
    draws = lens.block_bootstrap_season_delta(
        challenger.weekly[:count],
        baseline.weekly[:count],
        np.random.default_rng(lens.GATE_BOOTSTRAP_SEED),
    )
    return Comparison(
        label=challenger.label,
        delta_training=challenger.training - baseline.training,
        delta_evaluation=challenger.evaluation - baseline.evaluation,
        delta_overall=challenger.overall - baseline.overall,
        standard_error=float(draws.std()),
        confidence=float(np.mean(draws > 0)),
    )


def sweep(
    config: Config, name: str, values: list, data: pd.DataFrame
) -> tuple[Outcome, list[tuple[Outcome, Comparison]]]:
    """Vary exactly one field. Everything else stays pinned."""
    baseline = evaluate(config, data)
    results = []
    for value in values:
        variant = evaluate(config.with_field(name, value), data)
        results.append((variant, compare(baseline, variant)))
    return baseline, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_header(baseline: Outcome) -> None:
    print(
        "%-26s %9s %11s %9s %8s %7s  %s"
        % ("variant", "training", "evaluation", "all-10", "SE", "conf", "verdict")
    )
    print(
        "%-26s %9.1f %11.1f %9.1f %8s %7s  %s"
        % (
            "reference",
            baseline.training,
            baseline.evaluation,
            baseline.overall,
            "-",
            "-",
            "baseline",
        )
    )


def _print_row(outcome: Outcome, result: Comparison) -> None:
    print(
        "%-26s %+9.1f %+11.1f %+9.1f %8.1f %7.3f  %s"
        % (
            result.label,
            result.delta_training,
            result.delta_evaluation,
            result.delta_overall,
            result.standard_error,
            result.confidence,
            result.verdict,
        )
    )


def _record(payload: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if RESULTS.exists():
        history = json.loads(RESULTS.read_text(encoding="utf-8"))
    history.append(payload)
    RESULTS.write_text(json.dumps(history[-200:], indent=2) + "\n", encoding="utf-8")


def _parse(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["reference", "sweep", "compare"], help="what to run"
    )
    parser.add_argument("--field", help="the single field to vary")
    parser.add_argument("--values", help="comma-separated values for that field")
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="bootstrap on all seasons rather than training only (reporting only)",
    )
    arguments = parser.parse_args()

    data, _ = lens.load_or_build_prepared_history()
    config = reference_config()
    print(f"reference: {config.strategy.name}, robust={config.robust_planning}")

    if arguments.command == "reference":
        baseline = evaluate(config, data)
        _print_header(baseline)
        print()
        print("per season: " + " ".join("%d" % round(v) for v in baseline.totals))
        _record({"command": "reference", "overall": baseline.overall})
        return

    if not arguments.field or not arguments.values:
        parser.error("--field and --values are required for sweep/compare")
    values = [_parse(item) for item in arguments.values.split(",")]

    baseline, results = sweep(config, arguments.field, values, data)
    _print_header(baseline)
    for outcome, result in results:
        _print_row(outcome, result)
    _record(
        {
            "command": arguments.command,
            "field": arguments.field,
            "reference": baseline.overall,
            "variants": [
                {
                    "label": result.label,
                    "deltaTraining": round(result.delta_training, 1),
                    "deltaEvaluation": round(result.delta_evaluation, 1),
                    "deltaOverall": round(result.delta_overall, 1),
                    "standardError": round(result.standard_error, 1),
                    "confidence": round(result.confidence, 3),
                    "verdict": result.verdict,
                }
                for _, result in results
            ],
        }
    )


if __name__ == "__main__":
    main()

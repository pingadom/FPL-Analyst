"""Decision-focused primitives for the FPL breakthrough research stack.

The frozen historical champion remains the control.  This module contains the
new, explicitly probabilistic boundary between forecasts and legal FPL actions:

* deadline fieldability extraction and chance constraints;
* correlated player-route scenarios;
* paired, conservative action comparison against Hold;
* premium-access and regime-change diagnostics; and
* finite-horizon chip-sequence optimisation.

Nothing in this file reads future results.  Historical callers are responsible
for fitting/calibrating on completed prior seasons only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from forecast_routes import route_components


POSITIONS = ("GK", "DEF", "MID", "FWD")
POSITION_IDS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
HARD_OUT_STATUSES = frozenset({"i", "s", "u"})


def _nested(row: Mapping, *path: str, default=None):
    value = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _probability(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number > 1.0:
        number /= 100.0
    return float(np.clip(number, 0.0, 1.0))


def fixture_count(row: Mapping) -> int:
    """Return the current, deadline-known number of fixtures for a player."""
    value = row.get("fixtureCount")
    if value is None:
        value = _nested(row, "researchFeatures", "fixture_count")
    if value is None:
        # Live rows in older artifacts contain an opponent but no explicit count.
        value = 1 if row.get("opponent") not in (None, "", "-") else 0
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def play_probability(row: Mapping) -> float:
    value = row.get("playProbability")
    if value is None:
        value = _nested(row, "minutesModel", "playProbability")
    if value is None:
        value = _nested(row, "researchFeatures", "play_probability")
    if value is None:
        minutes = row.get("deadlineMinutes", row.get("expectedMinutes"))
        if minutes is not None:
            value = min(1.0, max(0.0, float(minutes) / 72.0))
    return _probability(value, 0.5)


def start_probability(row: Mapping) -> float:
    value = row.get("startProbability")
    if value is None:
        value = _nested(row, "minutesModel", "startProbability")
    if value is None:
        value = _nested(row, "researchFeatures", "start_probability")
    return _probability(value, min(play_probability(row), 0.5))


def sixty_probability(row: Mapping) -> float:
    value = row.get("sixtyProbability")
    if value is None:
        value = _nested(row, "minutesModel", "sixtyProbability")
    if value is None:
        value = _nested(row, "researchFeatures", "sixty_probability")
    return _probability(value, 0.8 * start_probability(row))


def official_chance(row: Mapping) -> float | None:
    value = row.get("chanceOfPlayingNextRound")
    if value is None:
        value = _nested(row, "minutesModel", "availabilityEvidence", "chance")
    if value is None:
        return None
    return _probability(value, 1.0)


def availability_status(row: Mapping) -> str:
    value = row.get("status")
    if value is None:
        value = _nested(row, "minutesModel", "availabilityEvidence", "status")
    return str(value or "a").lower()


@dataclass(frozen=True)
class FieldabilityPolicy:
    hard_exclude_no_fixture: bool = True
    hard_out_statuses: frozenset[str] = HARD_OUT_STATUSES
    hard_out_chance: float = 0.0
    minimum_expected_xi_appearances: float = 9.65
    captain_min_play_probability: float = 0.72
    vice_min_play_probability: float = 0.68
    risk_penalty: float = 0.45
    emergency_squad_target: int = 11


DEFAULT_FIELDABILITY_POLICY = FieldabilityPolicy()


def hard_unavailable(
    row: Mapping, policy: FieldabilityPolicy = DEFAULT_FIELDABILITY_POLICY
) -> bool:
    if policy.hard_exclude_no_fixture and fixture_count(row) <= 0:
        return True
    chance = official_chance(row)
    status = availability_status(row)
    return bool(
        status in policy.hard_out_statuses
        and chance is not None
        and chance <= policy.hard_out_chance
    )


def fieldability_vector(
    players: Sequence[Mapping],
    policy: FieldabilityPolicy = DEFAULT_FIELDABILITY_POLICY,
) -> np.ndarray:
    return np.asarray(
        [0.0 if hard_unavailable(row, policy) else play_probability(row) for row in players],
        dtype=float,
    )


def fieldability_audit(
    players: Sequence[Mapping], squad: Sequence[int], xi: Sequence[int]
) -> dict:
    squad_probabilities = fieldability_vector(players)[list(squad)]
    xi_probabilities = fieldability_vector(players)[list(xi)]
    hard_xi = [int(index) for index in xi if hard_unavailable(players[index])]
    return {
        "hardUnavailableXi": hard_xi,
        "expectedSquadAppearances": round(float(squad_probabilities.sum()), 4),
        "expectedXiAppearances": round(float(xi_probabilities.sum()), 4),
        "squadWithFixture": int(
            sum(fixture_count(players[index]) > 0 for index in squad)
        ),
        "xiWithFixture": int(sum(fixture_count(players[index]) > 0 for index in xi)),
        "meanXiPlayProbability": round(float(xi_probabilities.mean()), 4),
    }


@dataclass(frozen=True)
class ScenarioConfig:
    draws: int = 512
    seed: int = 20260820
    team_attack_sigma: float = 0.30
    player_attack_shape: float = 0.75
    residual_scale: float = 0.45
    clean_sheet_correlation: float = 1.0


@dataclass(frozen=True)
class ScenarioBundle:
    points: np.ndarray
    appearances: np.ndarray
    sixty: np.ndarray
    means: np.ndarray
    player_indices: np.ndarray
    metadata: dict = field(default_factory=dict)


def _column(frame: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), float(default), dtype=float)
    return frame[name].fillna(default).to_numpy(float)


def sample_correlated_player_scenarios(
    frame: pd.DataFrame,
    base_scores: np.ndarray,
    config: ScenarioConfig = ScenarioConfig(),
) -> ScenarioBundle:
    """Generate correlated route-level FPL point scenarios for one deadline.

    The route means contain only pre-deadline fields.  Appearance draws are
    explicit, team attack shocks are shared, and clean sheets are shared by
    club.  The result is intended for paired action comparisons, not as a new
    independently promotable point forecast.
    """
    if len(frame) != len(base_scores):
        raise ValueError("frame and base_scores must have identical lengths")
    if config.draws < 32:
        raise ValueError("at least 32 scenario draws are required")
    routes = route_components(frame, np.asarray(base_scores, dtype=float))
    rng = np.random.default_rng(config.seed)
    count = len(frame)
    p_play = np.clip(_column(frame, "play_probability", 0.5), 0.0, 1.0)
    p_sixty = np.minimum(
        p_play, np.clip(_column(frame, "sixty_probability", 0.35), 0.0, 1.0)
    )
    appearances = rng.random((config.draws, count)) < p_play
    conditional_sixty = np.divide(
        p_sixty,
        np.maximum(p_play, 1e-9),
        out=np.zeros_like(p_sixty),
        where=p_play > 0,
    )
    sixty = appearances & (rng.random((config.draws, count)) < conditional_sixty)

    appearance_points = appearances.astype(float) + sixty.astype(float)
    team = frame.get("team_id", pd.Series(np.arange(count))).to_numpy()
    deadline = (
        frame.get("season", pd.Series([""] * count)).astype(str)
        + ":"
        + frame.get("GW", pd.Series([0] * count)).astype(str)
        + ":"
        + pd.Series(team).astype(str)
    )
    groups, group_ids = np.unique(deadline.to_numpy(), return_inverse=True)
    team_attack = rng.lognormal(
        mean=-0.5 * config.team_attack_sigma**2,
        sigma=config.team_attack_sigma,
        size=(config.draws, len(groups)),
    )
    attack_mean = np.clip(routes["attack"], 0.0, None)
    attack_conditional_mean = np.divide(
        attack_mean,
        np.maximum(p_play, 1e-6),
        out=np.zeros_like(attack_mean),
        where=p_play > 0,
    )
    shape = max(config.player_attack_shape, 0.05)
    attack_independent = rng.gamma(
        shape=shape,
        scale=np.maximum(attack_conditional_mean, 1e-9) / shape,
        size=(config.draws, count),
    )
    attack_points = attack_independent * team_attack[:, group_ids] * appearances

    clean_probability = np.clip(
        _column(frame, "team_clean_probability", 0.25), 0.01, 0.85
    )
    team_clean_probability = np.zeros(len(groups), dtype=float)
    for group_id in range(len(groups)):
        team_clean_probability[group_id] = float(
            np.median(clean_probability[group_ids == group_id])
        )
    clean_team = rng.random((config.draws, len(groups))) < team_clean_probability
    clean_points = (
        clean_team[:, group_ids]
        * sixty
        * np.asarray(routes["cleanPoints"], dtype=float)[None, :]
    )

    bonus_mean = np.clip(routes["bonus"], 0.0, 3.0)
    bonus_conditional_mean = np.divide(
        bonus_mean,
        np.maximum(p_play, 1e-6),
        out=np.zeros_like(bonus_mean),
        where=p_play > 0,
    )
    bonus_points = (
        rng.poisson(bonus_conditional_mean, size=(config.draws, count))
        * appearances
    )
    known_mean = (
        np.asarray(routes["appearance"])
        + attack_mean
        + np.asarray(routes["clean"])
        + bonus_mean
    )
    residual_mean = np.asarray(base_scores, dtype=float) - known_mean
    residual_conditional_mean = np.divide(
        residual_mean,
        np.maximum(p_play, 1e-6),
        out=np.zeros_like(residual_mean),
        where=p_play > 0,
    )
    residual_std = np.maximum(
        _column(frame, "prediction_uncertainty", 1.5) * config.residual_scale,
        0.15,
    )
    residual = rng.normal(
        residual_conditional_mean, residual_std, size=(config.draws, count)
    )
    residual *= appearances
    points = np.clip(
        appearance_points + attack_points + clean_points + bonus_points + residual,
        -2.0,
        None,
    )
    return ScenarioBundle(
        points=points,
        appearances=appearances,
        sixty=sixty,
        means=points.mean(axis=0),
        player_indices=np.arange(count, dtype=int),
        metadata={
            "draws": config.draws,
            "seed": config.seed,
            "teamGroups": int(len(groups)),
            "meanAbsoluteMeanDrift": round(
                float(np.mean(np.abs(points.mean(axis=0) - base_scores))), 6
            ),
        },
    )


def _legal_outfield_sub(
    positions: Sequence[int], active: list[int], incoming: int
) -> bool:
    proposed = active + [incoming]
    counts = {position: sum(positions[index] == position for index in proposed) for position in (1, 2, 3, 4)}
    return (
        counts[1] == 1
        and 3 <= counts[2] <= 5
        and 2 <= counts[3] <= 5
        and 1 <= counts[4] <= 3
    )


def score_squad_scenarios(
    bundle: ScenarioBundle,
    positions: Sequence[int],
    xi: Sequence[int],
    bench: Sequence[int],
    captain: int,
    vice: int,
    *,
    bench_boost: bool = False,
    triple_captain: bool = False,
) -> np.ndarray:
    """Score a legal XI with autosubs and captain fallback for every draw."""
    xi = list(map(int, xi))
    bench = list(map(int, bench))
    positions = list(map(int, positions))
    if len(xi) != 11 or len(bench) != 4:
        raise ValueError("FPL scenarios require an XI and four-player bench")
    result = np.zeros(bundle.points.shape[0], dtype=float)
    for draw in range(bundle.points.shape[0]):
        if bench_boost:
            active = xi + bench
        else:
            active = [index for index in xi if bundle.appearances[draw, index]]
            missing_gk = not any(positions[index] == 1 for index in active)
            bench_gk = next((index for index in bench if positions[index] == 1), None)
            if (
                missing_gk
                and bench_gk is not None
                and bundle.appearances[draw, bench_gk]
            ):
                active.append(bench_gk)
            for incoming in bench:
                if positions[incoming] == 1 or not bundle.appearances[draw, incoming]:
                    continue
                if len(active) >= 11:
                    break
                if _legal_outfield_sub(positions, active, incoming):
                    active.append(incoming)
        result[draw] = float(bundle.points[draw, active].sum())
        armband = captain if bundle.appearances[draw, captain] else vice
        if bundle.appearances[draw, armband]:
            result[draw] += float(bundle.points[draw, armband])
            if triple_captain:
                result[draw] += float(bundle.points[draw, armband])
    return result


@dataclass(frozen=True)
class ActionRiskPolicy:
    minimum_win_probability: float = 0.57
    minimum_mean_advantage: float = 0.25
    minimum_cvar10: float = -4.0
    confidence_z: float = 0.75
    model_disagreement_penalty: float = 0.35
    emergency_fieldability_credit: float = 1.5


@dataclass(frozen=True)
class ActionEvaluation:
    action_id: str
    mean_advantage: float
    median_advantage: float
    win_probability: float
    q10: float
    cvar10: float
    standard_error: float
    lower_confidence_advantage: float
    utility: float
    passes_gate: bool
    metadata: dict = field(default_factory=dict)


def evaluate_paired_action(
    action_id: str,
    action_points: np.ndarray,
    hold_points: np.ndarray,
    *,
    transfer_cost: float = 0.0,
    continuation_value: float = 0.0,
    model_disagreement: float = 0.0,
    fieldability_gain: float = 0.0,
    policy: ActionRiskPolicy = ActionRiskPolicy(),
    metadata: dict | None = None,
) -> ActionEvaluation:
    action_points = np.asarray(action_points, dtype=float)
    hold_points = np.asarray(hold_points, dtype=float)
    if action_points.shape != hold_points.shape or action_points.ndim != 1:
        raise ValueError("paired action and Hold draws must be equal one-dimensional arrays")
    delta = action_points - hold_points - float(transfer_cost) + float(continuation_value)
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / math.sqrt(max(1, len(delta))))
    lower = mean - policy.confidence_z * standard_error
    q10 = float(np.quantile(delta, 0.10))
    tail = delta[delta <= q10]
    cvar = float(tail.mean()) if len(tail) else q10
    win = float(np.mean(delta > 0))
    emergency_credit = policy.emergency_fieldability_credit * max(0.0, fieldability_gain)
    utility = (
        lower
        - policy.model_disagreement_penalty * max(0.0, model_disagreement)
        + emergency_credit
    )
    passes = bool(
        mean >= policy.minimum_mean_advantage
        and win >= policy.minimum_win_probability
        and cvar >= policy.minimum_cvar10
        and utility > 0.0
    )
    return ActionEvaluation(
        action_id=str(action_id),
        mean_advantage=round(mean, 6),
        median_advantage=round(float(np.median(delta)), 6),
        win_probability=round(win, 6),
        q10=round(q10, 6),
        cvar10=round(cvar, 6),
        standard_error=round(standard_error, 6),
        lower_confidence_advantage=round(lower, 6),
        utility=round(float(utility), 6),
        passes_gate=passes,
        metadata=dict(metadata or {}),
    )


def choose_conservative_action(
    hold_points: np.ndarray,
    actions: Iterable[Mapping],
    policy: ActionRiskPolicy = ActionRiskPolicy(),
) -> tuple[str, list[ActionEvaluation]]:
    evaluations = [
        evaluate_paired_action(
            str(action["id"]),
            np.asarray(action["points"], dtype=float),
            np.asarray(hold_points, dtype=float),
            transfer_cost=float(action.get("transferCost", 0.0)),
            continuation_value=float(action.get("continuationValue", 0.0)),
            model_disagreement=float(action.get("modelDisagreement", 0.0)),
            fieldability_gain=float(action.get("fieldabilityGain", 0.0)),
            policy=policy,
            metadata=dict(action.get("metadata", {})),
        )
        for action in actions
    ]
    eligible = [row for row in evaluations if row.passes_gate]
    if not eligible:
        return "Hold", evaluations
    selected = max(eligible, key=lambda row: (row.utility, row.mean_advantage))
    return selected.action_id, evaluations


@dataclass(frozen=True)
class PremiumAccessDiagnostic:
    premium_id: int
    current_package_value: float
    premium_package_value: float
    captaincy_option_value: float
    liquidity_value: float
    model_disagreement: float
    robust_advantage: float
    access_failure: bool


def premium_access_diagnostic(
    *,
    premium_id: int,
    current_package_value: float,
    premium_package_value: float,
    future_captain_probabilities: Sequence[float],
    future_captain_edges: Sequence[float],
    liquidity_value: float,
    model_disagreement: float,
    uncertainty_multiplier: float = 0.5,
) -> PremiumAccessDiagnostic:
    probabilities = np.asarray(future_captain_probabilities, dtype=float)
    edges = np.asarray(future_captain_edges, dtype=float)
    if probabilities.shape != edges.shape:
        raise ValueError("captain probabilities and edges must align")
    captaincy = float(np.sum(np.clip(probabilities, 0.0, 1.0) * edges))
    advantage = (
        float(premium_package_value)
        - float(current_package_value)
        + captaincy
        + float(liquidity_value)
        - uncertainty_multiplier * max(0.0, float(model_disagreement))
    )
    return PremiumAccessDiagnostic(
        premium_id=int(premium_id),
        current_package_value=float(current_package_value),
        premium_package_value=float(premium_package_value),
        captaincy_option_value=round(captaincy, 6),
        liquidity_value=float(liquidity_value),
        model_disagreement=float(model_disagreement),
        robust_advantage=round(advantage, 6),
        access_failure=bool(advantage > 0.0),
    )


def regime_change_probability(
    recent_rate: float,
    long_rate: float,
    recent_minutes: float,
    long_minutes: float,
    team_shift: float,
    role_changed: bool,
    observations: int,
) -> float:
    """Bayesian-shrunk change signal; never a named-player bonus."""
    sample_weight = float(observations) / (float(observations) + 8.0)
    rate_scale = max(abs(float(long_rate)), 0.35)
    rate_change = np.clip((float(recent_rate) - float(long_rate)) / rate_scale, -2.5, 2.5)
    minute_change = np.clip((float(recent_minutes) - float(long_minutes)) / 30.0, -2.0, 2.0)
    logit = (
        -1.65
        + sample_weight * (0.90 * rate_change + 0.65 * minute_change)
        + 0.85 * np.clip(float(team_shift), 0.0, 1.0)
        + (0.90 if role_changed else 0.0)
    )
    return float(1.0 / (1.0 + math.exp(-float(logit))))


@dataclass(frozen=True)
class ChipState:
    week: int
    end_week: int
    available: frozenset[str]
    banked_transfers: int = 1
    permanent_state: str = "base"


@dataclass(frozen=True)
class ChipTransition:
    action: str
    immediate_value: float
    next_permanent_state: str
    setup_cost: float = 0.0
    risk_penalty: float = 0.0
    terminal_value: float = 0.0
    consumes_chip: str | None = None
    preserves_permanent_state: bool = False


@dataclass(frozen=True)
class ChipPlan:
    total_value: float
    actions: tuple[tuple[int, str], ...]
    terminal_state: str


def optimise_chip_sequence(
    initial: ChipState,
    transition_provider: Callable[[ChipState], Sequence[ChipTransition]],
    *,
    discount: float = 0.985,
) -> ChipPlan:
    """Optimise legal chip/action sequences over a finite half-season.

    The provider owns the squad/forecast model.  This routine enforces one
    action per week, chip inventory, FH non-persistence and receding-horizon
    continuation value without relying on a hard-coded chip calendar.
    """

    @lru_cache(maxsize=None)
    def solve(state: ChipState) -> ChipPlan:
        if state.week > state.end_week:
            return ChipPlan(0.0, tuple(), state.permanent_state)
        transitions = list(transition_provider(state))
        if not any(row.action == "Hold" for row in transitions):
            transitions.append(
                ChipTransition("Hold", 0.0, state.permanent_state)
            )
        best: ChipPlan | None = None
        for transition in transitions:
            chip = transition.consumes_chip
            if chip is not None and chip not in state.available:
                continue
            available = state.available - ({chip} if chip else set())
            next_permanent = (
                state.permanent_state
                if transition.preserves_permanent_state
                else transition.next_permanent_state
            )
            next_state = ChipState(
                week=state.week + 1,
                end_week=state.end_week,
                available=frozenset(available),
                banked_transfers=min(
                    5,
                    state.banked_transfers
                    + (1 if transition.action in {"Hold", "Free Hit", "Wildcard"} else 0),
                ),
                permanent_state=next_permanent,
            )
            continuation = solve(next_state)
            value = (
                float(transition.immediate_value)
                - float(transition.setup_cost)
                - float(transition.risk_penalty)
                + float(transition.terminal_value)
                + discount * continuation.total_value
            )
            candidate = ChipPlan(
                total_value=round(value, 8),
                actions=((state.week, transition.action),) + continuation.actions,
                terminal_state=continuation.terminal_state,
            )
            if best is None or candidate.total_value > best.total_value:
                best = candidate
        if best is None:
            raise RuntimeError("chip sequence has no legal transition")
        return best

    return solve(initial)

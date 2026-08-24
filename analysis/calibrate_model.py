"""Calibrate the FPL Lens ranking model on leak-free historical gameweeks.

The script downloads public FPL snapshots, constructs only pre-deadline features,
replays every gameweek from 2018-19 onward, evaluates hundreds of candidate
weight sets, and writes a compact JSON artifact consumed by the website.
"""

from __future__ import annotations

import json
import math
import sys
import unicodedata
import urllib.request
from urllib.error import HTTPError
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from live_external_signals import (
    fetch_matchbook_signals,
    fixture_lookup as external_fixture_lookup,
    implied_goal_rates as external_implied_goal_rates,
    load_elite_consensus,
    load_opta_fixture_predictions,
    load_team_priors,
    normalize_team as normalize_external_team,
    poisson_outcomes as external_poisson_outcomes,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "fpl-data"
PREPARED_HISTORY_CACHE = CACHE / "prepared-history-lens9-minutes-calibrated-v2.pkl"
OUTPUT = ROOT / "app" / "data" / "model-results.json"
PLAYERS_OUTPUT = ROOT / "app" / "data" / "current-players.json"
TRAINING_SEASONS = ["2016-17", "2017-18"]
EVALUATION_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]
SEASONS = TRAINING_SEASONS + EVALUATION_SEASONS
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
REEP_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"
CURRENT_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
CURRENT_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
TRIALS = 2400
# The cheap snapshot stage covers 2,400 mixtures. Eighty diverse candidates
# then receive a stateful screen before the 20 exact-policy finalists. Profiling
# showed that 240 stateful screen candidates repeated expensive squad work while
# adding no new family beyond those already preserved by the priority rules.
SCREENING_FINALISTS = 80
CHIP_POLICY_TRIALS = 48
# Holding an unused chip is an option, and an option is only worth something
# while weeks remain in which to exercise it. These three constants define that
# option value as a share of the chip's own base threshold. They deliberately
# depend on the remaining window length alone: the historical archive has no
# announcement timestamps for postponements, so the future blank/double schedule
# cannot be consulted without leaking.
CHIP_HOLD_VALUE = 0.55
CHIP_HOLD_DECAY_GWS = 6.0
# In the final week of a window an unplayed chip is worth nothing, so the bar
# ramps down towards a token positive-value check rather than expiring the chip
# unused. Only one chip may be played per Gameweek, so the structural-signal
# requirement is also waived slightly before the true deadline: several chips
# can share the same expiry week and they cannot all go in it.
#
# The floor is per chip because the chips do not share a cost structure. Free
# Hit, Bench Boost and Triple Captain are settled inside their own Gameweek, so
# an unplayed one really is worth nothing and dumping it is close to free. A
# Wildcard is not: its cost is the squad trajectory it leaves behind, which no
# single-week measure can see. A flat 0.25 floor fired 14 Wildcards across eight
# seasons — 1.75 of a possible 2 per season — and turned +76 points of local
# 2018/19 chip gain into a 30-point seasonal loss. The Wildcard therefore stays
# close to its searched `wildcard_gap` even at expiry.
# 0.70 was chosen on the two pre-2018 training seasons alone (2,123.5 against
# 2,079.5 for the old flat floor). The response surface is rugged — a Wildcard
# played one week earlier cascades through the rest of the season, so adjacent
# floors can differ by 20 points with no monotone trend — so treat this as a
# reasonable setting rather than a tuned optimum.
CHIP_EXPIRY_THRESHOLD_SHARE: dict[str, float] = {
    "Wildcard": 0.70,
    "Free Hit": 0.25,
    "Bench Boost": 0.25,
    "Triple Captain": 0.25,
    "Assistant Manager": 0.25,
}
DEFAULT_CHIP_EXPIRY_THRESHOLD_SHARE = 0.25
CHIP_FORCED_USE_WINDOW_GWS = 1
POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
XI_QUOTAS = {1: 1, 2: 3, 3: 5, 4: 2}
SQUAD_QUOTAS = {1: 2, 2: 5, 3: 5, 4: 3}
LIVE_ACTION_FEATURES = (
    "component_xpts", "role_ridge_xpts", "expected_minutes", "minutes_std",
    "play_probability", "start_probability", "sixty_probability", "recent_raw",
    "long_raw", "goal_rate", "assist_rate", "bonus_rate",
    "team_expected_goals_for", "team_expected_goals_against",
    "team_clean_probability", "price", "selected", "fixture_count",
    "horizon_weighted_games_censored", "fixture_censored", "fixture_now",
    "team_context", "team_attack", "team_defence", "recent_underlying",
    "long_underlying", "recent_value", "long_value", "minutes_model_confidence",
    "observations", "rotation_volatility", "team_rating_confidence",
    "team_regime_shift", "transfer_pressure_rank", "price_rise_probability",
    "price_fall_probability", "competition_pressure", "prediction_uncertainty",
    "position_id", "GW",
)
LIVE_ROUTE_FEATURES = (
    "component_xpts_structural", "empirical_xpts", "role_ridge_xpts",
    "expected_minutes", "minutes_std", "play_probability", "start_probability",
    "sixty_probability", "minutes_model_confidence", "rotation_volatility",
    "competition_pressure", "recent_raw", "long_raw", "recent_underlying_raw",
    "long_underlying_raw", "goal_rate", "assist_rate", "clean_sheet_rate",
    "save_rate", "bonus_rate", "defensive_rate", "bps_rate",
    "team_attack_rating", "team_defence_rating", "opponent_attack_rating",
    "opponent_defence_rating", "team_expected_goals_for",
    "team_expected_goals_against", "team_clean_probability",
    "team_rating_confidence", "team_regime_shift", "fixture_now", "price",
    "selected", "observations", "fixture_count", "position_id", "was_home",
    "expected_goals", "expected_assists", "expected_goals_conceded",
    "defensive_return_probability", "defensive_event_coverage",
    "opponent_goal_vulnerability", "opponent_assist_vulnerability",
    "team_clean_rating", "opponent_clean_rating", "league_goal_rate",
    "table_goal_difference_before",
)
AFCON_WINDOWS = {
    "2021-22": (20, 24),
    "2023-24": (20, 24),
    "2025-26": (16, 22),
}
UNLIMITED_TRANSFER_GWS = {
    # Project Restart and the Qatar World Cup both supplied a free full rebuild.
    "2019-20": {39},
    "2022-23": {17},
}
ASSISTANT_MANAGER_COST_2024 = {
    "arsenal": 15,
    "chelsea": 15,
    "liverpool": 15,
    "mancity": 15,
    "newcastle": 15,
    "bournemouth": 11,
    "brighton": 11,
    "fulham": 11,
    "nottmforest": 11,
    "nottinghamforest": 11,
    "spurs": 11,
    "tottenham": 11,
    "astonvilla": 8,
    "brentford": 8,
    "crystalpalace": 8,
    "manutd": 8,
    "manchesterunited": 8,
    "wolves": 8,
    "everton": 5,
    "ipswich": 5,
    "leicester": 5,
    "southampton": 5,
    "westham": 5,
}
AFCON_NATIONS = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini",
    "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
    "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
    "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
    "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal", "Sierra Leone",
    "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia",
    "Uganda", "Zambia", "Zimbabwe",
}

# Every decision threshold in this file — transfer hurdles, hold-option values,
# chip gaps — is a number of points. That makes each of them silently dependent on
# how widely the forecast spreads players apart. Swap in a model with a different
# dispersion and the same constant means something else, so what gets measured is
# the scale mismatch rather than the new model. A leak-free six-Gameweek model with
# a *better* correlation against the six-Gameweek target (0.7114 against 0.6927)
# cost 60 points a season purely this way.
#
# These are the within-Gameweek cross-sectional spreads the frozen constants were
# tuned against. Thresholds are rescaled by the ratio of the live spread to these,
# which leaves the tuned configuration unchanged and lets any other forecast be
# judged on its merits.
REFERENCE_IMMEDIATE_SPREAD = 1.4641
REFERENCE_PLAN_SPREAD = 5.6529


def cross_sectional_spread(
    frame: pd.DataFrame, values: np.ndarray, minimum: float = 1e-6
) -> float:
    """Mean within-Gameweek standard deviation of a per-player score.

    A per-player threshold lives on the scale that separates players *within* a
    deadline, which is not the same as the pooled spread across a whole season.
    """
    active = frame["fixture_count"].to_numpy(int) > 0
    if not active.any():
        return minimum
    table = pd.DataFrame(
        {
            "season": frame["season"].to_numpy()[active],
            "GW": frame["GW"].to_numpy()[active],
            "value": np.asarray(values, dtype=float)[active],
        }
    )
    spread = table.groupby(["season", "GW"])["value"].std().mean()
    return float(max(minimum, 0.0 if pd.isna(spread) else spread))


def rescale_decision_thresholds(
    strategy: "SimulationStrategy",
    chip_policy: "ChipPolicy | None",
    immediate_scale: float,
    plan_scale: float,
) -> tuple["SimulationStrategy", "ChipPolicy | None"]:
    """Put every points-denominated threshold back on the live forecast's scale."""
    strategy = replace(
        strategy,
        transfer_hurdle=strategy.transfer_hurdle * plan_scale,
        additional_move_hurdle=strategy.additional_move_hurdle * plan_scale,
        hold_option_value=strategy.hold_option_value * plan_scale,
        hit_immediate_hurdle=strategy.hit_immediate_hurdle * immediate_scale,
        package_setup_hurdle=strategy.package_setup_hurdle * plan_scale,
        package_setup_loss_limit=strategy.package_setup_loss_limit * plan_scale,
        staleness_hurdle_reduction=strategy.staleness_hurdle_reduction * plan_scale,
        staleness_hold_reduction=strategy.staleness_hold_reduction * plan_scale,
        staleness_gap_trigger=(
            None
            if strategy.staleness_gap_trigger is None
            else strategy.staleness_gap_trigger * plan_scale
        ),
        fieldability_penalty=strategy.fieldability_penalty * plan_scale,
    )
    if chip_policy is not None:
        chip_policy = replace(
            chip_policy,
            # A Wildcard is judged on a whole-squad planning utility; the other
            # three are read off one Gameweek's projected points.
            wildcard_gap=chip_policy.wildcard_gap * plan_scale,
            free_hit_gap=chip_policy.free_hit_gap * immediate_scale,
            bench_score=chip_policy.bench_score * immediate_scale,
            triple_score=chip_policy.triple_score * immediate_scale,
        )
    return strategy, chip_policy


# The decision-policy gate picks between four combinations using the mean of two
# pre-2018 training seasons. Those combinations differ by up to 200 points a season
# on evaluation data, so choosing between them on n=2 is a coin flip — and it has
# been observed to flip on an unrelated one-constant change, moving the reported
# mean by 40 points and individual seasons by nearly 200. A max over four noisy
# estimates also carries a large optimistic bias.
#
# So the gate now defends an incumbent. A challenger has to win a paired
# block-bootstrap of the weekly training-season differences by a real margin, not
# merely post a higher point estimate. Weekly blocks keep the streaky, correlated
# nature of a season intact; pairing cancels the shocks both policies shared.
# A points hit costs a certain -4, but the gain it is weighed against is a
# selected maximum over many candidate bundles and is therefore optimistic — the
# optimiser's curse. Charging the literal 4 makes hits look cheap against
# horizon-scale gains and the beam takes 15-23 a season, all of them
# value-destroying. This is the price the beam actually pays, and it is a tuning
# parameter rather than a rule of the game.
PAID_MOVE_UTILITY_COST = 4.0
# Expected goals, expected assists and expected goals conceded first appear in
# the archive in 2022-23; before that their coverage is exactly 0%. That is not a
# detail, it is a different forecasting regime — weekly forecast correlation runs
# 0.49-0.53 before and 0.54-0.55 after — and the decision policies rank
# *oppositely* across the two. The joint beam loses 80-111 points a season in the
# pre-xG years and gains 140-301 in the xG years, because a beam search amplifies
# whatever forecast it is handed: below roughly 0.53 it amplifies the error.
#
# So the two pre-2018 training seasons are not merely a small sample, they are the
# wrong sample: they select a policy suited to a game that no longer exists. When
# regime-comparable seasons are available the gate uses those instead, which moves
# the switch a year earlier (confidence 0.976 against 0.695 in 2023/24) and is
# worth 2122.1 against 2097.8 for plain walk-forward and 2061.4 frozen.
# How many of the best training-selected candidates the gate pools before ranking
# strategies. One is not enough: the strategy ranking flips between candidates.
GATE_CANDIDATE_POOL = 3
XG_ERA_FIRST_SEASON = "2022-23"
GATE_INCUMBENT = "central:Six-GW planner + adaptive banking"
GATE_SWITCH_CONFIDENCE = 0.75
GATE_BOOTSTRAP_SAMPLES = 2000
GATE_BOOTSTRAP_BLOCK = 4
GATE_BOOTSTRAP_SEED = 20260813


def block_bootstrap_season_delta(
    challenger_weeks: list[list[float]],
    incumbent_weeks: list[list[float]],
    rng: np.random.Generator,
    block: int = GATE_BOOTSTRAP_BLOCK,
    samples: int = GATE_BOOTSTRAP_SAMPLES,
) -> np.ndarray:
    """Sampling distribution of the mean season-total difference."""
    series = [
        np.asarray(challenger, dtype=float) - np.asarray(incumbent, dtype=float)
        for challenger, incumbent in zip(challenger_weeks, incumbent_weeks)
        if len(challenger) == len(incumbent) and len(challenger) >= block
    ]
    if not series:
        return np.zeros(1, dtype=float)
    draws = np.zeros(samples, dtype=float)
    for index in range(samples):
        season_totals = []
        for values in series:
            count = len(values)
            starts = rng.integers(0, count, size=int(np.ceil(count / block)))
            resampled = np.concatenate(
                [
                    np.take(values, range(start, start + block), mode="wrap")
                    for start in starts
                ]
            )[:count]
            season_totals.append(resampled.sum())
        draws[index] = float(np.mean(season_totals))
    return draws


def select_gate_option(
    gate_results: dict[str, tuple],
    seasons_available: int,
) -> tuple[str, dict]:
    """Keep the incumbent policy unless a challenger clears the selection noise.

    ``seasons_available`` is how many completed seasons the decision may see. The
    frozen gate always passed two, which is why it could never separate policies
    that differ by less than its own standard error. Walking it forward lets a
    later season decide on everything completed before it.
    """
    rng = np.random.default_rng(GATE_BOOTSTRAP_SEED)
    incumbent = (
        GATE_INCUMBENT if GATE_INCUMBENT in gate_results else sorted(gate_results)[0]
    )

    season_count = len(SEASONS)
    horizon = min(seasons_available, season_count)
    modern_from = (
        SEASONS.index(XG_ERA_FIRST_SEASON)
        if XG_ERA_FIRST_SEASON in SEASONS
        else horizon
    )
    # Prefer regime-comparable evidence; fall back to everything available only
    # while no xG-era season has completed.
    usable = [index for index in range(horizon) if index >= modern_from] or list(
        range(horizon)
    )

    def weeks(name: str) -> list[list[float]]:
        # Stats are pooled candidate-major, season-minor: one block of seasons
        # per candidate the gate probed. Pairing survives because every option
        # is probed in the same order.
        stats = gate_results[name][3]
        blocks = max(1, len(stats) // season_count)
        return [
            list(stats[block * season_count + index]["weeklyPoints"])
            for block in range(blocks)
            for index in usable
            if block * season_count + index < len(stats)
        ]

    incumbent_weeks = weeks(incumbent)
    report: dict = {
        "incumbent": incumbent,
        "switchConfidence": GATE_SWITCH_CONFIDENCE,
        "options": {},
    }
    selected, best_confidence = incumbent, GATE_SWITCH_CONFIDENCE
    for name in sorted(gate_results):
        training_mean = float(
            np.mean([gate_results[name][0][index] for index in usable])
        )
        if name == incumbent:
            report["options"][name] = {
                "trainingMean": round(training_mean, 1),
                "confidenceVsIncumbent": None,
                "meanDelta": 0.0,
                "standardError": 0.0,
            }
            continue
        draws = block_bootstrap_season_delta(weeks(name), incumbent_weeks, rng)
        confidence = float(np.mean(draws > 0))
        report["options"][name] = {
            "trainingMean": round(training_mean, 1),
            "confidenceVsIncumbent": round(confidence, 3),
            "meanDelta": round(float(draws.mean()), 1),
            "standardError": round(float(draws.std()), 1),
        }
        if confidence >= best_confidence:
            selected, best_confidence = name, confidence
    report["selected"] = selected
    report["switched"] = selected != incumbent
    report["seasonsAvailable"] = int(seasons_available)
    report["seasonsUsed"] = [SEASONS[index] for index in usable]
    report["regimeMatched"] = bool(usable and usable[0] >= modern_from)
    return selected, report


_SIMULATION_CONTEXT_CACHE: dict[tuple[int, int], dict] = {}


def sigmoid(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    clipped = np.clip(values, -18, 18)
    return 1 / (1 + np.exp(-clipped))


OFFICIAL_STATUS_DEFAULT_CHANCE = {
    "a": 100.0,
    "d": 75.0,
    "i": 0.0,
    "s": 0.0,
    "u": 0.0,
    "n": 0.0,
}


def official_availability_chance(
    chance: pd.Series, status: pd.Series
) -> pd.Series:
    """Resolve nullable FPL chance fields without resurrecting absences."""
    numeric = pd.to_numeric(chance, errors="coerce")
    fallback = (
        status.fillna("a")
        .astype(str)
        .str.lower()
        .map(OFFICIAL_STATUS_DEFAULT_CHANCE)
        .fillna(0.0)
    )
    return numeric.fillna(fallback).clip(0, 100)


def frame_fingerprint(
    data: pd.DataFrame, columns: list[str] | tuple[str, ...], schema: str
) -> str:
    """Return a feature-aware provenance key for learned prediction caches."""
    selected = list(dict.fromkeys(["season", "GW", "element", *columns]))
    missing = [column for column in selected if column not in data]
    if missing:
        raise KeyError(f"Cannot fingerprint missing columns: {missing}")
    hashed = pd.util.hash_pandas_object(data[selected], index=True).to_numpy(
        np.uint64
    )
    return (
        f"{schema}:{len(data)}:{int(hashed.sum(dtype=np.uint64))}:"
        f"{int(np.bitwise_xor.reduce(hashed, initial=np.uint64(0)))}"
    )


def poisson_tail(
    mean: pd.Series | np.ndarray | float, threshold: pd.Series | np.ndarray | float
) -> np.ndarray:
    """Return P(X >= threshold) for a Poisson variable without SciPy."""
    lam = np.asarray(mean, dtype=float)
    cut = np.asarray(threshold, dtype=int)
    result = np.zeros(np.broadcast(lam, cut).shape, dtype=float)
    lam, cut = np.broadcast_arrays(lam, cut)
    for target in np.unique(cut):
        mask = cut == target
        target_int = max(1, int(target))
        local = lam[mask]
        probability = np.exp(-local)
        cumulative = probability.copy()
        for count in range(1, target_int):
            probability = probability * local / count
            cumulative += probability
        result[mask] = 1 - cumulative
    return np.clip(result, 0, 1)


def normal_cdf(values: pd.Series | np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1 + erf(array / math.sqrt(2)))


def monotone_probability_map(
    successes: np.ndarray,
    counts: np.ndarray,
    global_rate: float,
    bounds: tuple[float, float] = (0.005, 0.995),
    shrinkage: float = 32.0,
) -> np.ndarray:
    """Beta-smoothed isotonic bin estimates without an extra dependency.

    ``successes`` may be any additive quantity (event counts, or realised
    minutes) as long as ``bounds`` describes the range of the resulting mean.
    """
    values = (successes + shrinkage * global_rate) / (counts + shrinkage)
    weights = counts + shrinkage
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left[3] + right[3]
            blocks.append(
                [
                    left[0],
                    right[1],
                    (left[2] * left[3] + right[2] * right[3]) / merged_weight,
                    merged_weight,
                ]
            )
    mapped = np.zeros(len(counts), dtype=float)
    for start, end, value, _ in blocks:
        mapped[int(start) : int(end) + 1] = value
    return np.clip(mapped, bounds[0], bounds[1])


def causal_calibrate_distributions(data: pd.DataFrame) -> pd.DataFrame:
    """Calibrate event probabilities and 80% bands using prior GWs only."""
    data = data.reset_index(drop=True)
    event_specs = {
        "blank_probability": ("raw_blank_probability", data["points"] <= 2, 0.72),
        "return5_probability": ("raw_return5_probability", data["points"] >= 5, 0.12),
        "haul8_probability": ("raw_haul8_probability", data["points"] >= 8, 0.045),
    }
    calibrated = {
        output: data[raw].to_numpy(float).copy()
        for output, (raw, _, _) in event_specs.items()
    }
    half_width = np.zeros(len(data), dtype=float)
    state: dict[int, dict[str, object]] = {}
    ordered_groups = data.groupby(["season_order", "GW"], sort=True).groups
    for _, group_index in ordered_groups.items():
        group_positions = data.loc[group_index, "position_id"].astype(int)
        for position in sorted(group_positions.unique()):
            indices = np.asarray(
                group_positions[group_positions == position].index, dtype=int
            )
            indices = indices[
                data.loc[indices, "fixture_count"].to_numpy(int) > 0
            ]
            if not len(indices):
                continue
            position_state = state.setdefault(
                position,
                {
                    "events": {
                        output: {
                            "successes": np.zeros(10, dtype=float),
                            "counts": np.zeros(10, dtype=float),
                            "total_successes": 0.0,
                            "total_count": 0.0,
                        }
                        for output in event_specs
                    },
                    "ratio_hist": np.zeros(81, dtype=float),
                },
            )
            events = position_state["events"]
            assert isinstance(events, dict)
            for output, (raw_column, _, prior_rate) in event_specs.items():
                event_state = events[output]
                assert isinstance(event_state, dict)
                total_count = float(event_state["total_count"])
                global_rate = (
                    float(event_state["total_successes"]) + 60 * prior_rate
                ) / (total_count + 60)
                mapping = monotone_probability_map(
                    np.asarray(event_state["successes"], dtype=float),
                    np.asarray(event_state["counts"], dtype=float),
                    global_rate,
                )
                raw_values = data.loc[indices, raw_column].to_numpy(float)
                bins = np.minimum((raw_values * 10).astype(int), 9)
                calibrated[output][indices] = mapping[bins]
            ratio_hist = np.asarray(position_state["ratio_hist"], dtype=float)
            if ratio_hist.sum() >= 250:
                quantile_bin = int(
                    np.searchsorted(np.cumsum(ratio_hist), 0.80 * ratio_hist.sum())
                )
                ratio_80 = max(0.25, min(2.50, quantile_bin / 20))
            else:
                ratio_80 = 0.70
            half_width[indices] = (
                data.loc[indices, "prediction_uncertainty"].to_numpy(float)
                * ratio_80
            )

        # Only update after scoring the whole deadline, preventing same-GW leakage.
        for position in sorted(group_positions.unique()):
            indices = np.asarray(
                group_positions[group_positions == position].index, dtype=int
            )
            position_state = state[position]
            events = position_state["events"]
            assert isinstance(events, dict)
            for output, (raw_column, target, _) in event_specs.items():
                raw_values = data.loc[indices, raw_column].to_numpy(float)
                bins = np.minimum((raw_values * 10).astype(int), 9)
                outcomes = target.loc[indices].to_numpy(float)
                event_state = events[output]
                assert isinstance(event_state, dict)
                np.add.at(event_state["counts"], bins, 1)
                np.add.at(event_state["successes"], bins, outcomes)
                event_state["total_count"] = float(event_state["total_count"]) + len(indices)
                event_state["total_successes"] = float(
                    event_state["total_successes"]
                ) + float(outcomes.sum())
            ratios = (
                np.abs(
                    data.loc[indices, "points"].to_numpy(float)
                    - data.loc[indices, "component_xpts"].to_numpy(float)
                )
                / data.loc[indices, "prediction_uncertainty"].to_numpy(float).clip(0.1)
            ).clip(0, 4)
            ratio_bins = np.minimum((ratios * 20).astype(int), 80)
            np.add.at(position_state["ratio_hist"], ratio_bins, 1)

    for output, values in calibrated.items():
        data[output] = values
    # The interval caps are per-match ceilings, so a Double Gameweek needs a
    # correspondingly wider band rather than a truncated one.
    fixture_scale = np.sqrt(
        data["fixture_count"].clip(lower=1).to_numpy(float)
    )
    data["prediction_half_width_80"] = np.clip(half_width, 0.4, 9.0 * fixture_scale)
    data["prediction_p10"] = (
        data["component_xpts"] - data["prediction_half_width_80"]
    ).clip(lower=0)
    data["prediction_p90"] = np.minimum(
        data["component_xpts"] + data["prediction_half_width_80"],
        25.0 * data["fixture_count"].clip(lower=1),
    )
    structural_blank = data["fixture_count"].eq(0)
    data.loc[structural_blank, "blank_probability"] = 1.0
    data.loc[structural_blank, "return5_probability"] = 0.0
    data.loc[structural_blank, "haul8_probability"] = 0.0
    data.loc[structural_blank, "prediction_p10"] = 0.0
    data.loc[structural_blank, "prediction_p90"] = 0.0
    return data


def calibrate_live_distributions(
    current: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    """Apply terminal historical calibration maps to the next deadline."""
    event_specs = {
        "blank_probability": ("raw_blank_probability", historical["points"] <= 2, 0.72),
        "return5_probability": ("raw_return5_probability", historical["points"] >= 5, 0.12),
        "haul8_probability": ("raw_haul8_probability", historical["points"] >= 8, 0.045),
    }
    for position in sorted(current["position_id"].astype(int).unique()):
        current_mask = current["position_id"].astype(int) == position
        history_mask = (
            (historical["position_id"].astype(int) == position)
            & historical["fixture_count"].gt(0)
        )
        for output, (raw_column, target, prior_rate) in event_specs.items():
            raw_history = historical.loc[history_mask, raw_column].to_numpy(float)
            bins = np.minimum((raw_history * 10).astype(int), 9)
            outcomes = target.loc[history_mask].to_numpy(float)
            counts = np.bincount(bins, minlength=10).astype(float)
            successes = np.bincount(bins, weights=outcomes, minlength=10).astype(float)
            global_rate = (outcomes.sum() + 60 * prior_rate) / (len(outcomes) + 60)
            mapping = monotone_probability_map(successes, counts, global_rate)
            live_raw = current.loc[current_mask, raw_column].to_numpy(float)
            live_bins = np.minimum((live_raw * 10).astype(int), 9)
            current.loc[current_mask, output] = mapping[live_bins]
        ratios = (
            np.abs(
                historical.loc[history_mask, "points"].to_numpy(float)
                - historical.loc[history_mask, "component_xpts"].to_numpy(float)
            )
            / historical.loc[
                history_mask, "prediction_uncertainty"
            ].to_numpy(float).clip(0.1)
        ).clip(0, 4)
        ratio_80 = float(np.quantile(ratios, 0.80)) if len(ratios) else 0.70
        current.loc[current_mask, "prediction_half_width_80"] = (
            current.loc[current_mask, "projection_std"] * np.clip(ratio_80, 0.25, 2.50)
        )
    # The ceiling is a per-match cap, so a Double Gameweek gets a proportionally
    # wider band instead of a truncated one.
    fixture_ceiling = 25.0 * current["fixture_count"].clip(lower=1)
    current["prediction_p10"] = (
        current["raw_projection"] - current["prediction_half_width_80"]
    ).clip(lower=0)
    current["prediction_p50"] = current["raw_projection"]
    current["prediction_p90"] = np.minimum(
        current["raw_projection"] + current["prediction_half_width_80"],
        fixture_ceiling,
    )
    return current


MINUTES_CALIBRATION_BINS = 20
MINUTES_CALIBRATION_MINIMUM_ROWS = 400.0
# (predicted column, realised numerator, realised denominator, positional prior,
#  bin scale, output bounds)
MINUTES_CALIBRATION_SPECS: tuple[
    tuple[str, str, str, dict[int, float], float, tuple[float, float]], ...
] = (
    (
        "start_probability",
        "starts_observed",
        "fixture_count",
        {1: 0.68, 2: 0.58, 3: 0.56, 4: 0.54},
        1.0,
        (0.01, 0.99),
    ),
    (
        "play_probability",
        "appearances_observed",
        "fixture_count",
        {1: 0.70, 2: 0.68, 3: 0.70, 4: 0.70},
        1.0,
        (0.02, 0.995),
    ),
    (
        "sixty_probability",
        "sixty_observed",
        "fixture_count",
        {1: 0.66, 2: 0.52, 3: 0.46, 4: 0.42},
        1.0,
        (0.01, 0.99),
    ),
    (
        "minutes_if_start",
        "start_minutes_total",
        "starts_observed",
        {1: 88.0, 2: 82.0, 3: 79.0, 4: 77.0},
        90.0,
        (30.0, 90.0),
    ),
    (
        "minutes_if_bench",
        "bench_minutes_total",
        "bench_appearances_observed",
        {1: 6.0, 2: 18.0, 3: 22.0, 4: 24.0},
        90.0,
        (1.0, 70.0),
    ),
)


MINUTES_CALIBRATION_TIERS = (0, 1, 2)


def _minutes_bins(values: np.ndarray, scale: float) -> np.ndarray:
    raw = (np.asarray(values, dtype=float) / scale * MINUTES_CALIBRATION_BINS).astype(int)
    return np.clip(raw, 0, MINUTES_CALIBRATION_BINS - 1)


def minutes_calibration_tier(
    frame: pd.DataFrame, group_columns: list[str]
) -> np.ndarray:
    """Coarse deadline-known role band, used as a second calibration axis.

    The predicted probability alone is not a sufficient statistic: inside the
    same predicted bin, a premium starts far more often than a cheap squad
    player, because the beta prior compresses the two toward each other. Price is
    the market's view of who is first choice, it is known at the deadline, and it
    separates that residual cleanly.
    """
    rank = frame.groupby(group_columns)["price"].rank(pct=True).to_numpy(float)
    return np.select([rank < 0.55, rank < 0.88], [0, 1], default=2).astype(int)


def _rebuild_minutes_decomposition(frame: pd.DataFrame) -> None:
    """Restore the internal consistency the calibrated probabilities must keep."""
    start = frame["start_probability"].to_numpy(float)
    play = np.maximum(frame["play_probability"].to_numpy(float), start)
    sixty = np.minimum(frame["sixty_probability"].to_numpy(float), start)
    frame["play_probability"] = np.clip(play, 0.02, 0.995)
    frame["sixty_probability"] = np.clip(sixty, 0.01, 0.99)
    # Downstream code rebuilds expected minutes from the decomposition rather
    # than from the probabilities directly, so invert it back out.
    frame["sub_probability_given_bench"] = np.clip(
        (play - start) / np.clip(1.0 - start, 1e-6, None), 0.0, 0.95
    )
    frame["sixty_probability_given_start"] = np.clip(
        sixty / np.clip(start, 1e-6, None), 0.0, 1.0
    )


def causal_calibrate_minutes(data: pd.DataFrame) -> pd.DataFrame:
    """Recalibrate the minutes model on realised outcomes, prior deadlines only.

    The beta-smoothed estimates are compressed. Flat positional priors pull every
    player toward a common middle, and the rest/rotation/competition penalties can
    only ever push a start probability *down*, so a nailed starter is capped below
    his true rate while a fringe player floats above his. Because almost every
    scoring route is scaled by expected minutes — and appearance and clean-sheet
    points key directly off the play and 60-minute probabilities — that compression
    under-rates expensive players and over-rates cheap ones across the board.

    Learn a per-position isotonic map from predicted to realised, scoring each
    deadline before its own outcomes join the fit.
    """
    position_values = data["position_id"].to_numpy(int)
    tier_values = minutes_calibration_tier(data, ["season", "GW", "position_id"])
    data["minutes_calibration_tier"] = tier_values
    cells = [
        (position, tier)
        for position in SQUAD_QUOTAS
        for tier in MINUTES_CALIBRATION_TIERS
    ]
    scored = data["fixture_count"].to_numpy(float) > 0
    keys = (
        data["season_order"].to_numpy(np.int64) * 1000
        + data["GW"].to_numpy(np.int64)
    )
    order = np.argsort(keys, kind="stable")
    deadlines = np.split(order, np.flatnonzero(np.diff(keys[order])) + 1)

    for column, numerator, denominator, priors, scale, bounds in (
        MINUTES_CALIBRATION_SPECS
    ):
        raw = data[column].to_numpy(float)
        successes_all = data[numerator].to_numpy(float)
        counts_all = data[denominator].to_numpy(float)
        calibrated = raw.copy()
        state = {
            cell: {
                "successes": np.zeros(MINUTES_CALIBRATION_BINS, dtype=float),
                "counts": np.zeros(MINUTES_CALIBRATION_BINS, dtype=float),
                "total_successes": 0.0,
                "total_count": 0.0,
            }
            for cell in cells
        }
        for deadline in deadlines:
            active = deadline[scored[deadline]]
            for position, tier in cells:
                local = active[
                    (position_values[active] == position)
                    & (tier_values[active] == tier)
                ]
                entry = state[(position, tier)]
                if not len(local) or entry["total_count"] < MINUTES_CALIBRATION_MINIMUM_ROWS:
                    continue
                global_rate = (
                    entry["total_successes"] + 60 * priors[position]
                ) / (entry["total_count"] + 60)
                mapping = monotone_probability_map(
                    entry["successes"], entry["counts"], global_rate, bounds
                )
                calibrated[local] = mapping[_minutes_bins(raw[local], scale)]
            # Update only once the whole deadline has been scored, so a player is
            # never calibrated using his own result.
            update = deadline[counts_all[deadline] > 0]
            for position, tier in cells:
                local = update[
                    (position_values[update] == position)
                    & (tier_values[update] == tier)
                ]
                if not len(local):
                    continue
                entry = state[(position, tier)]
                bins = _minutes_bins(raw[local], scale)
                np.add.at(entry["counts"], bins, counts_all[local])
                np.add.at(entry["successes"], bins, successes_all[local])
                entry["total_count"] += float(counts_all[local].sum())
                entry["total_successes"] += float(successes_all[local].sum())
        # Keep the uncalibrated series: the live deadline needs a map fitted on
        # the raw predictor, not on an already-corrected one.
        data[f"{column}_uncalibrated"] = raw
        data[column] = calibrated

    _rebuild_minutes_decomposition(data)
    return data


def calibrate_live_minutes(
    current: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    """Apply terminal historical minutes calibration to the next deadline.

    The live minutes model is built from season-to-date starts rather than the
    historical rolling beta, but both are compressed the same way and live on the
    same 0-1 (or 0-90) scale, so the historical map transfers.
    """
    positions = current["position_id"].astype(int)
    live_tier = minutes_calibration_tier(current, ["position_id"])
    history_tier = (
        historical["minutes_calibration_tier"].to_numpy(int)
        if "minutes_calibration_tier" in historical
        else np.full(len(historical), -1)
    )
    for column, numerator, denominator, priors, scale, bounds in (
        MINUTES_CALIBRATION_SPECS
    ):
        source = f"{column}_uncalibrated"
        if source not in historical:
            continue
        for position, tier in (
            (position, tier)
            for position in sorted(positions.unique())
            for tier in MINUTES_CALIBRATION_TIERS
        ):
            history = historical[
                (historical["position_id"].astype(int) == position)
                & (history_tier == tier)
                & historical[denominator].gt(0)
            ]
            if len(history) < MINUTES_CALIBRATION_MINIMUM_ROWS:
                continue
            bins = _minutes_bins(history[source].to_numpy(float), scale)
            counts = np.bincount(
                bins,
                weights=history[denominator].to_numpy(float),
                minlength=MINUTES_CALIBRATION_BINS,
            ).astype(float)
            successes = np.bincount(
                bins,
                weights=history[numerator].to_numpy(float),
                minlength=MINUTES_CALIBRATION_BINS,
            ).astype(float)
            global_rate = (successes.sum() + 60 * priors[position]) / (
                counts.sum() + 60
            )
            mapping = monotone_probability_map(successes, counts, global_rate, bounds)
            mask = (positions == position).to_numpy() & (live_tier == tier)
            live_bins = _minutes_bins(
                current.loc[mask, column].to_numpy(float), scale
            )
            current.loc[mask, column] = mapping[live_bins]
    _rebuild_minutes_decomposition(current)
    return current


def assign_player_role(frame: pd.DataFrame) -> pd.Series:
    """Assign a causal scoring archetype from rates already known at deadline."""
    position = frame["position_id"].astype(int)
    goal = frame["goal_rate"].fillna(0)
    assist = frame["assist_rate"].fillna(0)
    defence = frame["defensive_rate"].fillna(0)
    saves = frame["save_rate"].fillna(0)
    role = pd.Series("balanced", index=frame.index, dtype="object")
    role.loc[(position == 1) & (saves >= 3.2)] = "shot_stopper"
    role.loc[(position == 1) & (saves < 3.2)] = "clean_sheet_keeper"
    role.loc[(position == 2) & (defence >= 7.8)] = "centre_back"
    role.loc[(position == 2) & (assist >= 0.13)] = "attacking_full_back"
    role.loc[(position == 2) & (defence >= 7.8) & (goal >= 0.09)] = "set_piece_centre_back"
    role.loc[(position == 2) & (role == "balanced")] = "balanced_defender"
    role.loc[(position == 3) & (defence >= 7.2) & ((goal + assist) < 0.30)] = "holding_midfielder"
    role.loc[(position == 3) & (assist > goal * 1.20) & (assist >= 0.16)] = "creator"
    role.loc[(position == 3) & (goal >= 0.25)] = "goal_threat_midfielder"
    role.loc[(position == 3) & (role == "balanced")] = "box_to_box_midfielder"
    role.loc[(position == 4) & (assist >= 0.16)] = "link_forward"
    role.loc[(position == 4) & (goal >= 0.34)] = "penalty_box_forward"
    role.loc[(position == 4) & (role == "balanced")] = "mobile_forward"
    return role


ROLE_FEATURE_COLUMNS = [
    "expected_minutes",
    "fixture_now",
    "team_context_raw",
    "recent_raw",
    "long_raw",
    "goal_rate",
    "assist_rate",
    "defensive_return_probability",
    "team_clean_probability",
    "bonus_rate",
]


def role_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Small, interpretable feature matrix for the role-specific challenger."""
    values = frame[ROLE_FEATURE_COLUMNS].fillna(0).to_numpy(float)
    scales = np.array([90, 1, 1.4, 5, 5, 0.5, 0.4, 1, 0.5, 1.5])
    return np.column_stack([np.ones(len(frame)), values / scales])


def causal_role_ridge_predictions(data: pd.DataFrame) -> pd.Series:
    """Online ridge predictions; a GW is scored before its outcomes update the fit."""
    work = data.reset_index(drop=True)
    features = role_feature_matrix(work)
    # Per-fixture target: this challenger is blended with the other per-match
    # routes, and the fixture count is applied once after the blend.
    outcomes = (
        (work["points"] / work["fixture_count"].clip(lower=1)).clip(-2, 20)
    ).to_numpy(float)
    predictions = work["structural_per_fixture"].to_numpy(float).copy()
    roles = work["player_role"].astype(str).to_numpy()
    observed = work["fixture_count"].to_numpy(int) > 0
    states: dict[str, dict[str, object]] = {}
    feature_count = features.shape[1]
    ridge = np.diag([0.20] + [8.0] * (feature_count - 1))
    for _, group_index in work.groupby(["season_order", "GW"], sort=True).groups.items():
        indices = np.asarray(group_index, dtype=int)
        for role_name in np.unique(roles[indices]):
            local = indices[(roles[indices] == role_name) & observed[indices]]
            if not len(local):
                continue
            state = states.setdefault(
                role_name,
                {
                    "xtx": np.zeros((feature_count, feature_count)),
                    "xty": np.zeros(feature_count),
                    "rows": 0,
                },
            )
            if int(state["rows"]) >= 120:
                beta = np.linalg.solve(
                    np.asarray(state["xtx"]) + ridge,
                    np.asarray(state["xty"]),
                )
                predictions[local] = np.clip(features[local] @ beta, 0.15, 14.0)
        for role_name in np.unique(roles[indices]):
            local = indices[roles[indices] == role_name]
            state = states[role_name]
            local_features = features[local]
            state["xtx"] = np.asarray(state["xtx"]) + local_features.T @ local_features
            state["xty"] = np.asarray(state["xty"]) + local_features.T @ outcomes[local]
            state["rows"] = int(state["rows"]) + len(local)
    return pd.Series(predictions, index=data.index)


def live_role_ridge_predictions(
    historical: pd.DataFrame, current: pd.DataFrame
) -> pd.Series:
    """Fit terminal role challengers with recency weighting for the live deadline."""
    historical = historical[historical["fixture_count"] > 0].copy()
    historical_features = role_feature_matrix(historical)
    live_features = role_feature_matrix(current)
    outcomes = (
        (historical["points"] / historical["fixture_count"].clip(lower=1))
        .clip(-2, 20)
        .to_numpy(float)
    )
    max_order = int(historical["season_order"].max())
    recency_weight = np.power(
        0.72, max_order - historical["season_order"].to_numpy(int)
    )
    predictions = current["component_projection_unscaled"].to_numpy(float).copy()
    ridge = np.diag([0.20] + [8.0] * (historical_features.shape[1] - 1))
    for role_name in current["player_role"].astype(str).unique():
        history_mask = historical["player_role"].astype(str).to_numpy() == role_name
        live_mask = current["player_role"].astype(str).to_numpy() == role_name
        if history_mask.sum() < 120:
            continue
        x = historical_features[history_mask]
        y = outcomes[history_mask]
        weight = recency_weight[history_mask]
        xtx = x.T @ (x * weight[:, None])
        xty = x.T @ (y * weight)
        beta = np.linalg.solve(xtx + ridge, xty)
        predictions[live_mask] = np.clip(live_features[live_mask] @ beta, 0.15, 14.0)
    return pd.Series(predictions, index=current.index)


def horizon_feature_matrix(data: pd.DataFrame) -> np.ndarray:
    """Deadline-known inputs for a directly supervised six-GW points model."""
    return np.column_stack(
        [
            np.ones(len(data)),
            data["component_xpts"].to_numpy(float) / 6.0,
            data["component_horizon_censored"].to_numpy(float) / 28.0,
            data["recent"].to_numpy(float),
            data["long"].to_numpy(float),
            data["fixture_censored"].to_numpy(float),
            data["team_context"].to_numpy(float),
            data["team_attack"].to_numpy(float),
            data["team_defence"].to_numpy(float),
            data["crowd"].to_numpy(float),
            data["minutes_security"].to_numpy(float),
            data["recent_underlying"].to_numpy(float),
            data["long_underlying"].to_numpy(float),
            data["recent_value"].to_numpy(float),
            data["prediction_uncertainty"].to_numpy(float) / 5.5,
            data["horizon_weighted_games"].to_numpy(float) / 6.0,
            data["price"].to_numpy(float) / 150.0,
        ]
    )


def causal_horizon_ridge_predictions(data: pd.DataFrame) -> pd.Series:
    """Predict discounted six-GW points without making future labels available early.

    A row from GW n enters the expanding fit only after the final GW used by its
    label has completed. At a new season, every label from older seasons is known.
    Position fits are partially pooled with a global fit to reduce early-sample
    variance while retaining the very different FPL scoring economics by role.
    """
    work = data.reset_index(drop=True)
    features = horizon_feature_matrix(work)
    outcomes = work["horizon_target"].clip(-3, 55).to_numpy(float)
    season_orders = work["season_order"].to_numpy(int)
    gameweeks = work["GW"].to_numpy(int)
    target_end = work["horizon_target_end_gw"].to_numpy(int)
    positions = work["position_id"].to_numpy(int)
    baseline = work["component_horizon_censored"].to_numpy(float)
    predictions = baseline.copy()
    added = np.zeros(len(work), dtype=bool)
    feature_count = features.shape[1]
    ridge = np.diag([0.25] + [12.0] * (feature_count - 1))

    def empty_state() -> dict[str, object]:
        return {
            "xtx": np.zeros((feature_count, feature_count)),
            "xty": np.zeros(feature_count),
            "rows": 0,
        }

    global_state = empty_state()
    position_states = {position: empty_state() for position in SQUAD_QUOTAS}
    groups = work.groupby(["season_order", "GW"], sort=True).groups
    for (season_order_value, gw_value), group_index in groups.items():
        known = (~added) & (
            (season_orders < int(season_order_value))
            | (
                (season_orders == int(season_order_value))
                & (target_end < int(gw_value))
            )
        )
        newly_known = np.flatnonzero(known)
        if len(newly_known):
            x_known = features[newly_known]
            y_known = outcomes[newly_known]
            global_state["xtx"] = np.asarray(global_state["xtx"]) + x_known.T @ x_known
            global_state["xty"] = np.asarray(global_state["xty"]) + x_known.T @ y_known
            global_state["rows"] = int(global_state["rows"]) + len(newly_known)
            for position in SQUAD_QUOTAS:
                local = newly_known[positions[newly_known] == position]
                if not len(local):
                    continue
                state = position_states[position]
                local_features = features[local]
                state["xtx"] = np.asarray(state["xtx"]) + local_features.T @ local_features
                state["xty"] = np.asarray(state["xty"]) + local_features.T @ outcomes[local]
                state["rows"] = int(state["rows"]) + len(local)
            added[newly_known] = True

        indices = np.asarray(group_index, dtype=int)
        if int(global_state["rows"]) < 1000:
            continue
        global_beta = np.linalg.solve(
            np.asarray(global_state["xtx"]) + ridge,
            np.asarray(global_state["xty"]),
        )
        global_prediction = features[indices] @ global_beta
        predictions[indices] = global_prediction
        for position in SQUAD_QUOTAS:
            local = indices[positions[indices] == position]
            state = position_states[position]
            if not len(local) or int(state["rows"]) < 350:
                continue
            position_beta = np.linalg.solve(
                np.asarray(state["xtx"]) + ridge,
                np.asarray(state["xty"]),
            )
            # Partial pooling is deliberately conservative: policy outcomes, not
            # in-sample regression error, decide whether this challenger survives.
            predictions[local] = (
                0.65 * (features[local] @ position_beta)
                + 0.35 * global_prediction[positions[indices] == position]
            )
    return pd.Series(np.clip(predictions, 0.25, 52.0), index=data.index)


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        target.write_bytes(response.read())
    return target


def get_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def percentile(series: pd.Series) -> pd.Series:
    if series.notna().sum() < 2:
        return pd.Series(0.5, index=series.index)
    return series.rank(method="average", pct=True).fillna(0.5)


def parse_dob(value: object) -> date | None:
    if value is None or pd.isna(value) or str(value) in {"None", "", "nan"}:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def load_age_register() -> dict[int, str]:
    path = download(REEP_URL, CACHE / "reep-people.csv")
    people = pd.read_csv(
        path,
        usecols=["date_of_birth", "key_opta_numeric"],
        dtype=str,
        low_memory=False,
    ).dropna()
    people["key_opta_numeric"] = pd.to_numeric(
        people["key_opta_numeric"], errors="coerce"
    )
    people = people.dropna(subset=["key_opta_numeric"])
    return dict(
        zip(
            people["key_opta_numeric"].astype(int),
            people["date_of_birth"].astype(str),
        )
    )


def load_nationality_register() -> dict[int, str]:
    path = download(REEP_URL, CACHE / "reep-people.csv")
    people = pd.read_csv(
        path,
        usecols=["nationality", "key_opta_numeric"],
        dtype=str,
        low_memory=False,
    ).dropna()
    people["key_opta_numeric"] = pd.to_numeric(
        people["key_opta_numeric"], errors="coerce"
    )
    people = people.dropna(subset=["key_opta_numeric"])
    return dict(
        zip(
            people["key_opta_numeric"].astype(int),
            people["nationality"].astype(str),
        )
    )


def season_files(season: str) -> tuple[Path, Path, Path | None]:
    folder = CACHE / season
    gw = download(f"{BASE}/{season}/gws/merged_gw.csv", folder / "merged_gw.csv")
    players = download(f"{BASE}/{season}/players_raw.csv", folder / "players_raw.csv")
    try:
        teams = download(f"{BASE}/{season}/teams.csv", folder / "teams.csv")
    except HTTPError as error:
        if error.code != 404:
            raise
        try:
            teams = download(f"{BASE}/{season}/raw.json", folder / "raw.json")
        except HTTPError as raw_error:
            if raw_error.code != 404:
                raise
            teams = None
    return gw, players, teams


def build_season(
    season: str, ages: dict[int, str], nationalities: dict[int, str]
) -> tuple[pd.DataFrame, dict]:
    gw_path, players_path, teams_path = season_files(season)
    gw = pd.read_csv(gw_path, encoding="latin-1", low_memory=False)
    players = pd.read_csv(players_path, encoding="latin-1", low_memory=False)
    if teams_path is None:
        teams = pd.DataFrame(
            {
                "id": sorted(players["team"].dropna().astype(int).unique()),
                "name": [
                    f"Team {team_id}"
                    for team_id in sorted(players["team"].dropna().astype(int).unique())
                ],
            }
        )
    elif teams_path.suffix == ".json":
        teams = pd.DataFrame(json.loads(teams_path.read_text(encoding="utf-8"))["teams"])
    else:
        teams = pd.read_csv(teams_path, encoding="latin-1", low_memory=False)

    team_names = dict(zip(teams["id"].astype(int), teams["name"].astype(str)))
    meta_cols = ["id", "code", "element_type", "team", "web_name"]
    if "birth_date" in players.columns:
        meta_cols.append("birth_date")
    meta = players[meta_cols].copy().rename(
        columns={
            "id": "element",
            "code": "player_code",
            "element_type": "position_id",
            "team": "team_id",
            "web_name": "display_name",
        }
    )
    if "birth_date" not in meta.columns:
        meta["birth_date"] = None
    meta["birth_date"] = meta.apply(
        lambda row: row["birth_date"]
        if parse_dob(row["birth_date"])
        else ages.get(int(row["player_code"]))
        if pd.notna(row["player_code"])
        else None,
        axis=1,
    )
    meta["nationality"] = meta["player_code"].map(nationalities).fillna("")

    wanted = [
        "element",
        "GW",
        "total_points",
        "minutes",
        "value",
        "selected",
        "opponent_team",
        "was_home",
        "fixture",
        "kickoff_time",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "transfers_balance",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "starts",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "key_passes",
        "big_chances_created",
        "open_play_crosses",
        "penalties_missed",
        "penalties_saved",
        "own_goals",
        "yellow_cards",
        "red_cards",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "defensive_contribution",
        # Official FPL's deadline-vintage expected-points field is not used as
        # a points forecast.  A trustworthy zero is retained only as an
        # availability signal (injury, suspension, transfer or severe doubt).
        "xP",
    ]
    raw = gw[[column for column in wanted if column in gw.columns]].copy()
    raw = raw.merge(meta, on="element", how="left")
    raw["team_name"] = raw["team_id"].map(team_names)
    raw["opponent_name"] = raw["opponent_team"].map(team_names)
    if "selected" not in raw:
        raw["selected"] = 0
    raw["selected"] = pd.to_numeric(raw["selected"], errors="coerce").fillna(0)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").fillna(45)
    for column in [
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "transfers_balance",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "starts",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "key_passes",
        "big_chances_created",
        "open_play_crosses",
        "penalties_missed",
        "penalties_saved",
        "own_goals",
        "yellow_cards",
        "red_cards",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "defensive_contribution",
        "xP",
    ]:
        if column not in raw:
            raw[column] = 0
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)
    raw["official_xp"] = pd.to_numeric(raw.get("xP"), errors="coerce")
    raw["GW"] = pd.to_numeric(raw["GW"], errors="coerce")
    raw = raw.dropna(subset=["GW", "element", "position_id"]).copy()
    raw["GW"] = raw["GW"].astype(int)
    # players_raw is an end-of-season snapshot: its `team` field moves every
    # transferred player to their final club in all earlier rows. Recover the
    # actual club from the two opponent IDs present in each fixture instead.
    fixture_sides = {
        (int(gw_number), int(fixture)): sorted(
            set(group["opponent_team"].dropna().astype(int).tolist())
        )
        for (gw_number, fixture), group in raw.dropna(
            subset=["fixture", "opponent_team"]
        ).groupby(["GW", "fixture"], sort=False)
    }

    def actual_fixture_team(row: pd.Series) -> int:
        if pd.isna(row.get("fixture")) or pd.isna(row.get("opponent_team")):
            return int(row["team_id"])
        sides = fixture_sides.get((int(row["GW"]), int(row["fixture"])), [])
        opponent_id = int(row["opponent_team"])
        alternatives = [team_id for team_id in sides if team_id != opponent_id]
        return alternatives[0] if len(alternatives) == 1 else int(row["team_id"])

    raw["snapshot_team_id"] = raw["team_id"].astype(int)
    raw["team_id"] = raw.apply(actual_fixture_team, axis=1).astype(int)
    raw["team_name"] = raw["team_id"].map(team_names)
    recovered_team_rows = int(
        (raw["team_id"] != raw["snapshot_team_id"]).sum()
    )
    unresolved_fixture_rows = int(
        sum(len(sides) != 2 for sides in fixture_sides.values())
    )

    starts_available = "starts" in gw.columns
    defensive_exact_available = (
        "defensive_contribution" in gw.columns
        or {
            "clearances_blocks_interceptions",
            "recoveries",
            "tackles",
        }.issubset(gw.columns)
    )
    raw["start_observed"] = (
        raw["starts"].clip(0, 1)
        if starts_available
        else (raw["minutes"] >= 45).astype(float)
    )
    raw["appearance_observed"] = (raw["minutes"] > 0).astype(float)
    raw["sixty_observed"] = (raw["minutes"] >= 60).astype(float)
    raw["bench_appearance_observed"] = (
        (raw["appearance_observed"] > 0) & (raw["start_observed"] <= 0)
    ).astype(float)
    raw["start_minutes_total"] = raw["minutes"] * raw["start_observed"]
    raw["bench_minutes_total"] = raw["minutes"] * raw["bench_appearance_observed"]
    reconstructed_defence = (
        raw["clearances_blocks_interceptions"] + raw["tackles"]
    ) + np.where(raw["position_id"].isin([3, 4]), raw["recoveries"], 0)
    raw["defensive_actions_observed"] = np.where(
        "defensive_contribution" in gw.columns,
        raw["defensive_contribution"],
        reconstructed_defence,
    )
    raw["defensive_exact"] = float(defensive_exact_available)
    # Missing middle-season event feeds are assigned a transparent post-match
    # proxy for counterfactual scoring only. This proxy never enters a deadline
    # feature as if it had been observed.
    defensive_proxy = (
        raw["position_id"].map({1: 0.0, 2: 4.2, 3: 3.8, 4: 2.1}).fillna(2.5)
        + 0.075 * raw["influence"].clip(lower=0)
        + 0.055 * raw["bps"].clip(lower=0)
    ) * (raw["minutes"] / 90).clip(0, 1)
    raw["defensive_actions_counterfactual"] = np.where(
        raw["defensive_exact"] > 0,
        raw["defensive_actions_observed"],
        defensive_proxy,
    )
    defensive_threshold = np.where(raw["position_id"] == 2, 10, 12)
    defensive_probability = poisson_tail(
        raw["defensive_actions_counterfactual"], defensive_threshold
    )
    raw["current_rule_dc_points"] = np.where(
        raw["position_id"].isin([2, 3, 4]),
        np.where(
            raw["defensive_exact"] > 0,
            2 * (raw["defensive_actions_observed"] >= defensive_threshold),
            2 * defensive_probability,
        ),
        0,
    )
    # The 2026/27 BPS revision is smaller than the defensive-points change.
    # Apply a conservative role adjustment without claiming unavailable Opta
    # sub-components were reconstructed exactly.
    raw["current_rule_bps_adjustment"] = np.select(
        [
            raw["position_id"] == 1,
            (raw["position_id"] == 2) & (raw["defensive_actions_counterfactual"] >= 10),
            raw["position_id"].isin([3, 4]),
        ],
        [0.07, -0.06, 0.03],
        default=0.0,
    ) * raw["appearance_observed"]
    dc_already_scored = season == "2025-26"
    raw["points_current_rules"] = (
        raw["total_points"]
        + (0 if dc_already_scored else raw["current_rule_dc_points"])
        + raw["current_rule_bps_adjustment"]
    )

    # One row per club-fixture prevents team goals/xG from being counted once
    # for every player. These realised values are shifted before being used as
    # features, so a deadline can only see earlier matches.
    team_fixtures = (
        raw.dropna(subset=["team_id", "opponent_team", "fixture"])
        .groupby(
            ["team_id", "GW", "fixture", "opponent_team", "was_home"],
            as_index=False,
        )
        .agg(
            kickoff_time=("kickoff_time", "first"),
            team_goals=("goals_scored", "sum"),
            team_xg=("expected_goals", "sum"),
            team_goals_against=("goals_conceded", "max"),
            team_xga=("expected_goals_conceded", "max"),
            team_clean_sheet=("clean_sheets", "max"),
        )
    )
    team_fixtures["kickoff_time"] = pd.to_datetime(
        team_fixtures["kickoff_time"], errors="coerce", utc=True
    )
    team_fixtures.sort_values(["team_id", "kickoff_time", "GW"], inplace=True)
    team_fixtures["team_rest_days"] = (
        team_fixtures.groupby("team_id")["kickoff_time"].diff().dt.total_seconds()
        / 86400
    ).clip(2, 14).fillna(7)
    team_fixtures["team_result_points"] = np.select(
        [
            team_fixtures["team_goals"] > team_fixtures["team_goals_against"],
            team_fixtures["team_goals"] == team_fixtures["team_goals_against"],
        ],
        [3.0, 1.0],
        default=0.0,
    )
    team_weeks = (
        team_fixtures.groupby(["team_id", "GW"], as_index=False)
        .agg(
            team_games=("fixture", "nunique"),
            team_goals=("team_goals", "sum"),
            team_xg=("team_xg", "sum"),
            team_goals_against=("team_goals_against", "sum"),
            team_xga=("team_xga", "sum"),
            team_clean_sheets=("team_clean_sheet", "sum"),
            team_result_points=("team_result_points", "sum"),
            team_rest_days=("team_rest_days", "min"),
        )
    )
    # Reconstruct the table at each deadline so the one-season 2024/25
    # Assistant Manager chip can be scored under its actual rules. Positions
    # use only completed earlier events; current-GW results are not visible.
    observed_team_weeks = sorted(raw["GW"].astype(int).unique().tolist())
    registered_teams = sorted(meta["team_id"].dropna().astype(int).unique())
    team_week_panel = pd.MultiIndex.from_product(
        [registered_teams, observed_team_weeks], names=["team_id", "GW"]
    ).to_frame(index=False)
    team_week_panel = team_week_panel.merge(
        team_weeks, on=["team_id", "GW"], how="left"
    )
    for column in [
        "team_games",
        "team_goals",
        "team_xg",
        "team_goals_against",
        "team_xga",
        "team_clean_sheets",
        "team_result_points",
    ]:
        team_week_panel[column] = pd.to_numeric(
            team_week_panel[column], errors="coerce"
        ).fillna(0)
    team_week_panel["team_rest_days"] = pd.to_numeric(
        team_week_panel["team_rest_days"], errors="coerce"
    ).fillna(7)
    team_week_panel.sort_values(["team_id", "GW"], inplace=True)
    by_team_week = team_week_panel.groupby("team_id", sort=False)
    team_week_panel["table_points_before"] = by_team_week[
        "team_result_points"
    ].transform(lambda values: values.cumsum().shift(1)).fillna(0)
    team_week_panel["table_goals_before"] = by_team_week["team_goals"].transform(
        lambda values: values.cumsum().shift(1)
    ).fillna(0)
    goals_against_before = by_team_week["team_goals_against"].transform(
        lambda values: values.cumsum().shift(1)
    ).fillna(0)
    team_week_panel["table_goal_difference_before"] = (
        team_week_panel["table_goals_before"] - goals_against_before
    )
    team_week_panel.sort_values(
        [
            "GW",
            "table_points_before",
            "table_goal_difference_before",
            "table_goals_before",
            "team_id",
        ],
        ascending=[True, False, False, False, True],
        inplace=True,
        kind="stable",
    )
    team_week_panel["table_position_before"] = (
        team_week_panel.groupby("GW", sort=False).cumcount() + 1
    )
    positions = team_week_panel[
        ["team_id", "GW", "table_position_before"]
    ]
    manager_fixtures = team_fixtures.merge(
        positions, on=["team_id", "GW"], how="left"
    ).merge(
        positions.rename(
            columns={
                "team_id": "opponent_team",
                "table_position_before": "opponent_table_position_before",
            }
        ),
        on=["opponent_team", "GW"],
        how="left",
    )
    manager_fixtures["assistant_manager_points"] = (
        np.where(manager_fixtures["team_result_points"] == 3, 6, 0)
        + np.where(manager_fixtures["team_result_points"] == 1, 3, 0)
        + manager_fixtures["team_goals"]
        + 2 * manager_fixtures["team_clean_sheet"]
        + np.where(
            (
                manager_fixtures["table_position_before"]
                - manager_fixtures["opponent_table_position_before"]
                >= 5
            )
            & (manager_fixtures["team_result_points"] == 3),
            10,
            0,
        )
        + np.where(
            (
                manager_fixtures["table_position_before"]
                - manager_fixtures["opponent_table_position_before"]
                >= 5
            )
            & (manager_fixtures["team_result_points"] == 1),
            5,
            0,
        )
    )
    manager_week_points = manager_fixtures.groupby(
        ["team_id", "GW"], as_index=False
    )["assistant_manager_points"].sum()
    team_weeks = team_week_panel.merge(
        manager_week_points, on=["team_id", "GW"], how="left"
    )
    team_weeks["assistant_manager_points"] = team_weeks[
        "assistant_manager_points"
    ].fillna(0)

    weekly = (
        raw.groupby(["element", "GW"], as_index=False)
        .agg(
            points=("total_points", "sum"),
            points_current_rules=("points_current_rules", "sum"),
            minutes=("minutes", "sum"),
            price=("value", "mean"),
            selected=("selected", "max"),
            player_code=("player_code", "first"),
            position_id=("position_id", "first"),
            team_id=("team_id", "first"),
            team_name=("team_name", "first"),
            opponent_team=("opponent_team", "first"),
            opponent_name=("opponent_name", "first"),
            was_home=("was_home", "max"),
            fixture_count=("fixture", "nunique"),
            ict=("ict_index", "sum"),
            influence=("influence", "sum"),
            creativity=("creativity", "sum"),
            threat=("threat", "sum"),
            transfers_balance=("transfers_balance", "max"),
            display_name=("display_name", "first"),
            birth_date=("birth_date", "first"),
            nationality=("nationality", "first"),
            goals=("goals_scored", "sum"),
            assists=("assists", "sum"),
            clean_sheets=("clean_sheets", "sum"),
            goals_conceded=("goals_conceded", "sum"),
            saves=("saves", "sum"),
            bonus=("bonus", "sum"),
            bps=("bps", "sum"),
            starts_observed=("start_observed", "sum"),
            appearances_observed=("appearance_observed", "sum"),
            sixty_observed=("sixty_observed", "sum"),
            bench_appearances_observed=("bench_appearance_observed", "sum"),
            start_minutes_total=("start_minutes_total", "sum"),
            bench_minutes_total=("bench_minutes_total", "sum"),
            defensive_exact=("defensive_exact", "max"),
            defensive_actions_observed=("defensive_actions_observed", "sum"),
            defensive_actions_counterfactual=("defensive_actions_counterfactual", "sum"),
            current_rule_dc_points=("current_rule_dc_points", "sum"),
            key_passes=("key_passes", "sum"),
            big_chances_created=("big_chances_created", "sum"),
            open_play_crosses=("open_play_crosses", "sum"),
            penalties_missed=("penalties_missed", "sum"),
            penalties_saved=("penalties_saved", "sum"),
            own_goals=("own_goals", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            expected_goals=("expected_goals", "sum"),
            expected_assists=("expected_assists", "sum"),
            expected_goals_conceded=("expected_goals_conceded", "sum"),
            defensive_actions=("defensive_actions_counterfactual", "sum"),
            official_xp=("official_xp", "max"),
        )
        .sort_values(["element", "GW"])
    )
    # merged_gw contains fixture histories, so a player disappears when their
    # club blanks. Rebuild the deadline player panel across every observed FPL
    # event while the player was registered. This keeps blanking players in the
    # transfer/Wildcard universe with zero immediate points.
    observed_weeks = sorted(raw["GW"].astype(int).unique().tolist())
    availability = weekly.groupby("element", as_index=False)["GW"].agg(
        first_gw="min", last_gw="max"
    )
    panel = pd.DataFrame(
        [
            (int(row.element), int(gw_number))
            for row in availability.itertuples(index=False)
            for gw_number in observed_weeks
            if int(row.first_gw) <= int(gw_number) <= int(row.last_gw)
        ],
        columns=["element", "GW"],
    )
    weekly = panel.merge(weekly, on=["element", "GW"], how="left")
    static_columns = [
        "player_code",
        "position_id",
        "team_id",
        "team_name",
        "display_name",
        "birth_date",
        "nationality",
    ]
    deadline_columns = ["price", "selected", "transfers_balance"]
    for column in static_columns + deadline_columns:
        weekly[column] = weekly.groupby("element", sort=False)[column].transform(
            lambda values: values.ffill().bfill()
        )
    weekly["team_name"] = weekly["team_id"].map(team_names).fillna(
        weekly["team_name"]
    )
    zero_columns = [
        "points",
        "points_current_rules",
        "minutes",
        "fixture_count",
        "ict",
        "influence",
        "creativity",
        "threat",
        "goals",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "starts_observed",
        "appearances_observed",
        "sixty_observed",
        "bench_appearances_observed",
        "start_minutes_total",
        "bench_minutes_total",
        "defensive_actions_observed",
        "defensive_actions_counterfactual",
        "current_rule_dc_points",
        "key_passes",
        "big_chances_created",
        "open_play_crosses",
        "penalties_missed",
        "penalties_saved",
        "own_goals",
        "yellow_cards",
        "red_cards",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "defensive_actions",
    ]
    for column in zero_columns:
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce").fillna(0)
    weekly["defensive_exact"] = weekly["defensive_exact"].fillna(
        float(defensive_exact_available)
    )
    # Some archive scrapes contain an all-zero xP cross-section.  Detect that
    # corruption from deadline-known ownership and xP alone, per event, so the
    # player-level availability adjustment is disabled without looking at the
    # subsequent teamsheets or points.
    popular_xp = weekly[
        weekly["selected"].ge(100_000)
        & weekly["fixture_count"].gt(0)
        & weekly["official_xp"].notna()
    ]
    xp_feed_quality = popular_xp.groupby("GW")["official_xp"].agg(
        sample="size", zero_share=lambda values: float(values.le(0).mean())
    )
    trusted_xp_weeks = set(
        xp_feed_quality[
            xp_feed_quality["sample"].ge(20)
            & xp_feed_quality["zero_share"].le(0.65)
        ].index.astype(int)
    )
    weekly["official_xp_feed_trusted"] = weekly["GW"].isin(trusted_xp_weeks)
    weekly["was_home"] = weekly["was_home"].fillna(False).astype(bool)
    weekly.sort_values(["element", "GW"], inplace=True, kind="stable")
    weekly = weekly.merge(team_weeks, on=["team_id", "GW"], how="left")
    for column in [
        "team_games",
        "team_goals",
        "team_xg",
        "team_goals_against",
        "team_xga",
        "team_clean_sheets",
        "team_result_points",
        "team_rest_days",
        "assistant_manager_points",
        "table_position_before",
    ]:
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce").fillna(0)
    weekly["season"] = season
    season_start = date(int(season[:4]), 8, 1)
    weekly["age"] = weekly["birth_date"].map(
        lambda value: (
            (season_start - parse_dob(value)).days / 365.2425
            if parse_dob(value)
            else np.nan
        )
    )

    by_player = weekly.groupby("element", sort=False)
    weekly["observations"] = by_player.cumcount()
    weekly["long_raw"] = by_player["points"].transform(
        lambda values: values.expanding().mean().shift(1)
    )
    weekly["recent_raw"] = by_player["points"].transform(
        lambda values: values.rolling(4, min_periods=2).mean().shift(1)
    )
    weekly["past_minutes"] = by_player["minutes"].transform(
        lambda values: values.expanding().mean().shift(1)
    )
    weekly["minutes_security_raw"] = by_player["minutes"].transform(
        lambda values: values.clip(upper=90).div(90).rolling(6, min_periods=2).mean().shift(1)
    )
    weekly["minutes_security_raw"] = weekly["minutes_security_raw"].fillna(
        (weekly["past_minutes"] / 90).clip(0, 1)
    )
    weekly["underlying_game"] = (
        weekly["ict"] / weekly["minutes"].clip(lower=45) * 90
    ).clip(0, 35)
    weekly["long_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.expanding().mean().shift(1)
    )
    weekly["recent_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.rolling(4, min_periods=2).mean().shift(1)
    )
    weekly["recent_underlying_raw"] = weekly["recent_underlying_raw"].fillna(
        weekly["long_underlying_raw"]
    )
    weekly["recent_raw"] = weekly["recent_raw"].fillna(weekly["long_raw"])
    weekly["long_value_raw"] = weekly["long_raw"] / (weekly["price"] / 10).clip(3.5)
    weekly["recent_value_raw"] = weekly["recent_raw"] / (weekly["price"] / 10).clip(3.5)
    weekly["age_raw"] = np.exp(-((weekly["age"].fillna(27.5) - 27.5) / 7.5) ** 2)
    transfer_momentum = np.sign(weekly["transfers_balance"]) * np.log1p(
        weekly["transfers_balance"].abs()
    )
    weekly["crowd_raw"] = (
        np.log1p(weekly["selected"].clip(lower=0)) + 0.12 * transfer_momentum
    )
    weekly["transfer_pressure_raw"] = (
        weekly["transfers_balance"]
        / weekly["selected"].abs().clip(lower=2500)
    ).clip(-1.5, 1.5)
    weekly["transfer_pressure_rank"] = weekly.groupby("GW")[
        "transfer_pressure_raw"
    ].transform(percentile)
    weekly["price_rise_probability"] = sigmoid(
        11 * (weekly["transfer_pressure_rank"] - 0.72)
    )
    weekly["price_fall_probability"] = sigmoid(
        11 * (0.28 - weekly["transfer_pressure_rank"])
    )
    weekly["next_price_change"] = by_player["price"].shift(-1) - weekly["price"]

    allowed = (
        weekly.groupby(["opponent_team", "position_id", "GW"], as_index=False)
        .agg(
            points_allowed=("points", "mean"),
            goals_allowed=("goals", "mean"),
            assists_allowed=("assists", "mean"),
            xg_allowed=("expected_goals", "mean"),
        )
        .sort_values(["opponent_team", "position_id", "GW"])
    )
    allowed["fixture_raw"] = allowed.groupby(
        ["opponent_team", "position_id"], sort=False
    )["points_allowed"].transform(lambda values: values.expanding().mean().shift(1))
    for source, target in [
        ("goals_allowed", "opponent_goal_vulnerability"),
        ("assists_allowed", "opponent_assist_vulnerability"),
        ("xg_allowed", "opponent_xg_vulnerability"),
    ]:
        allowed[target] = allowed.groupby(
            ["opponent_team", "position_id"], sort=False
        )[source].transform(lambda values: values.rolling(10, min_periods=2).mean().shift(1))
    weekly = weekly.merge(
        allowed[
            [
                "opponent_team",
                "position_id",
                "GW",
                "fixture_raw",
                "opponent_goal_vulnerability",
                "opponent_assist_vulnerability",
                "opponent_xg_vulnerability",
            ]
        ],
        on=["opponent_team", "position_id", "GW"],
        how="left",
    )
    weekly["fixture_opponent_raw"] = weekly["fixture_raw"].fillna(
        weekly.groupby(["GW", "position_id"])["fixture_raw"].transform("median")
    )
    weekly["fixture_opponent_raw"] = weekly["fixture_opponent_raw"].fillna(2.5)
    weekly["fixture_raw"] = weekly["fixture_opponent_raw"] + weekly[
        "was_home"
    ].fillna(False).astype(float) * 0.18
    for column, fallback in [
        ("opponent_goal_vulnerability", 0.10),
        ("opponent_assist_vulnerability", 0.10),
        ("opponent_xg_vulnerability", 0.10),
    ]:
        weekly[column] = weekly[column].fillna(
            weekly.groupby(["GW", "position_id"])[column].transform("median")
        ).fillna(fallback)

    # Historical feeds retain the final event assigned to each fixture, not the
    # date on which a postponement or Double Gameweek was announced. Keep the
    # exact archive for sensitivity analysis, and also build a schedule-censored
    # horizon: the current slate is known, while later slots are neutral. The
    # censored version is the defensible primary backtest input.
    schedule = raw[["team_id", "GW", "opponent_team", "was_home"]].drop_duplicates()
    schedule_map: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for row in schedule.itertuples(index=False):
        schedule_map.setdefault((int(row.team_id), int(row.GW)), []).append(
            (int(row.opponent_team), bool(row.was_home))
        )
    fixture_lookup = {
        (int(row.GW), int(row.position_id), int(row.opponent_team)): float(row.fixture_opponent_raw)
        for row in weekly[["GW", "position_id", "opponent_team", "fixture_opponent_raw"]]
        .dropna()
        .drop_duplicates(["GW", "position_id", "opponent_team"])
        .itertuples(index=False)
    }
    fixture_median = {
        (int(gw_number), int(position)): float(value)
        for (gw_number, position), value in weekly.groupby(["GW", "position_id"])[
            "fixture_opponent_raw"
        ].median().items()
    }
    horizon_weights = (1.0, 0.86, 0.74, 0.64, 0.55, 0.47)
    observed_week_position = {
        int(gw_number): index for index, gw_number in enumerate(observed_weeks)
    }

    def horizon_events(base_gw: int) -> list[int]:
        start = observed_week_position.get(base_gw, 0)
        return observed_weeks[start : start + len(horizon_weights)]

    def fixture_horizon(row: pd.Series) -> float:
        values: list[tuple[float, float]] = []
        base_gw = int(row["GW"])
        position = int(row["position_id"])
        team = int(row["team_id"])
        fallback = fixture_median.get((base_gw, position), 2.5)
        for horizon_weight, target_gw in zip(
            horizon_weights, horizon_events(base_gw)
        ):
            for opponent, home in schedule_map.get((team, target_gw), []):
                strength = fixture_lookup.get(
                    (base_gw, position, opponent), fallback
                ) + (0.18 if home else 0.0)
                values.append((strength, horizon_weight))
        if not values:
            return fallback
        return sum(value * weight for value, weight in values) / sum(
            weight for _, weight in values
        )

    weekly["fixture_horizon_raw"] = weekly.apply(fixture_horizon, axis=1)
    weekly["horizon_weighted_games"] = weekly.apply(
        lambda row: sum(
            horizon_weight * len(
                schedule_map.get((int(row["team_id"]), target_gw), [])
            )
            for horizon_weight, target_gw in zip(
                horizon_weights, horizon_events(int(row["GW"]))
            )
        ),
        axis=1,
    ).clip(lower=1.0)

    def censored_fixture_horizon(row: pd.Series) -> float:
        """Future opponents are known; the number of future fixtures is not.

        The Premier League publishes all 380 fixtures before the season starts, so
        who a club faces in GW n+1..n+5 is legitimately available at every
        deadline. What is *not* available is the rescheduling: blanks and doubles
        are announced later and the archive keeps no announcement dates. So the
        opponent difficulty of each future event is used, while each future event
        contributes exactly one fixture's worth of weight regardless of how many
        fixtures the archive eventually recorded there. Where a club has no
        archived fixture the original opponent is unrecoverable, so the neutral
        median stands in rather than a known blank.
        """
        base_gw = int(row["GW"])
        position = int(row["position_id"])
        team = int(row["team_id"])
        fallback = fixture_median.get((base_gw, position), 2.5)
        event_gws = horizon_events(base_gw)
        values: list[tuple[float, float]] = []
        if event_gws:
            for opponent, home in schedule_map.get((team, event_gws[0]), []):
                strength = fixture_lookup.get(
                    (base_gw, position, opponent), fallback
                ) + (0.18 if home else 0.0)
                values.append((strength, horizon_weights[0]))
            for offset, target_gw in enumerate(event_gws[1:], start=1):
                slate = schedule_map.get((team, target_gw), [])
                strengths = [
                    fixture_lookup.get((base_gw, position, opponent), fallback)
                    + (0.18 if home else 0.0)
                    for opponent, home in slate
                ]
                values.append(
                    (
                        sum(strengths) / len(strengths) if strengths else fallback,
                        horizon_weights[offset],
                    )
                )
        if not values:
            return fallback
        return sum(value * weight for value, weight in values) / sum(
            weight for _, weight in values
        )

    weekly["fixture_horizon_censored_raw"] = weekly.apply(
        censored_fixture_horizon, axis=1
    )
    weekly["horizon_weighted_games_censored"] = weekly.apply(
        lambda row: (
            len(schedule_map.get((int(row["team_id"]), int(row["GW"])), []))
            + sum(
                horizon_weights[offset]
                for offset in range(
                    1, len(horizon_events(int(row["GW"])))
                )
            )
        ),
        axis=1,
    ).clip(lower=0.0)

    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_horizon_censored_raw", "fixture_censored"),
        ("fixture_raw", "fixture_now"),
        ("crowd_raw", "crowd"),
        ("minutes_security_raw", "minutes_security"),
        ("long_underlying_raw", "long_underlying"),
        ("recent_underlying_raw", "recent_underlying"),
    ]:
        weekly[rank_name] = weekly.groupby(["GW", "position_id"])[raw_name].transform(
            percentile
        )

    eligible = weekly[weekly["price"] >= 35].copy()
    age_coverage = float(weekly["age"].notna().mean())
    summary = {
        "season": season,
        "rows": int(len(weekly)),
        "eligibleRows": int(len(eligible)),
        "ageCoverage": round(age_coverage * 100, 1),
        "gameweeks": int(eligible["GW"].nunique()),
        "recoveredTransferRows": recovered_team_rows,
        "unresolvedFixtureGroups": unresolved_fixture_rows,
    }
    return eligible, summary


def add_causal_team_strength(data: pd.DataFrame) -> pd.DataFrame:
    """Add deadline-safe, time-decayed attack/defence and Poisson match rates."""
    team_columns = [
        "season",
        "season_order",
        "GW",
        "team_id",
        "team_name",
        "team_games",
        "team_goals",
        "team_xg",
        "team_goals_against",
        "team_xga",
        "team_clean_sheets",
        "team_result_points",
    ]
    team = data[team_columns].drop_duplicates(
        ["season", "GW", "team_id"], keep="first"
    ).copy()
    team.sort_values(["season_order", "GW", "team_id"], inplace=True)
    games = team["team_games"].clip(lower=1)
    goals_for = team["team_goals"] / games
    goals_against = team["team_goals_against"] / games
    xg_for = team["team_xg"] / games
    xg_against = team["team_xga"] / games
    # A club with no fixture in an event has produced no evidence about its
    # attack or defence. Leaving those rows in the panel as 0-0 taught the
    # ratings that every blank Gameweek was a goalless match, which depressed
    # attack and flattered defence for several weeks after any postponement.
    played = team["team_games"] > 0
    team["attack_observation"] = pd.Series(
        np.where(
            team["team_xg"] > 0,
            0.72 * xg_for + 0.28 * goals_for,
            goals_for,
        ),
        index=team.index,
    ).where(played)
    team["defence_observation"] = pd.Series(
        np.where(
            team["team_xga"] > 0,
            0.72 * xg_against + 0.28 * goals_against,
            goals_against,
        ),
        index=team.index,
    ).where(played)
    team["form_observation"] = (team["team_result_points"] / games).where(played)
    team["clean_observation"] = (team["team_clean_sheets"] / games).where(played)

    league_week = (
        team.groupby(["season", "season_order", "GW"], as_index=False)
        .agg(league_goals=("team_goals", "sum"), league_games=("team_games", "sum"))
        .sort_values(["season_order", "GW"])
    )
    league_week["league_observation"] = (
        league_week["league_goals"] / league_week["league_games"].clip(lower=1)
    )
    league_week["league_goal_rate"] = league_week.groupby(
        "season", sort=False
    )["league_observation"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(1.40).clip(0.9, 2.0)
    team = team.merge(
        league_week[["season", "GW", "league_goal_rate"]],
        on=["season", "GW"],
        how="left",
    )
    normalized_name = (
        team["team_name"].fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    )
    team["team_key"] = np.where(
        team["team_name"].fillna("").str.startswith("Team "),
        team["season"].astype(str) + ":" + normalized_name,
        normalized_name,
    )
    by_team = team.groupby("team_key", sort=False)
    team["prior_team_games"] = by_team["team_games"].transform(
        lambda values: values.cumsum().shift(1)
    ).fillna(0)
    team_confidence = (
        team["prior_team_games"] / (team["prior_team_games"] + 8)
    ).clip(0, 0.94)

    def dynamic_rating(column: str, prior: pd.Series | float) -> pd.Series:
        # ignore_na keeps a blank event from consuming decay weight; the rating
        # simply carries forward until the club plays again.
        rolling = team.groupby("team_key", sort=False)[column].transform(
            lambda values: values.ewm(alpha=0.22, adjust=False, ignore_na=True)
            .mean()
            .shift(1)
        )
        if isinstance(prior, pd.Series):
            fallback = prior
        else:
            fallback = pd.Series(float(prior), index=team.index)
        rolling = rolling.fillna(fallback)
        return team_confidence * rolling + (1 - team_confidence) * fallback

    team["team_attack_rating"] = dynamic_rating(
        "attack_observation", team["league_goal_rate"]
    ).clip(0.45, 2.70)
    team["team_defence_rating"] = dynamic_rating(
        "defence_observation", team["league_goal_rate"]
    ).clip(0.45, 2.70)
    team["team_form_rating"] = dynamic_rating("form_observation", 1.35).clip(0, 3)
    team["team_clean_rating"] = dynamic_rating("clean_observation", 0.28).clip(0, 0.75)
    fast_attack = team.groupby("team_key", sort=False)["attack_observation"].transform(
        lambda values: values.ewm(alpha=0.48, adjust=False, ignore_na=True)
        .mean()
        .shift(1)
    ).fillna(team["league_goal_rate"])
    fast_defence = team.groupby("team_key", sort=False)["defence_observation"].transform(
        lambda values: values.ewm(alpha=0.48, adjust=False, ignore_na=True)
        .mean()
        .shift(1)
    ).fillna(team["league_goal_rate"])
    team["team_regime_shift"] = (
        (
            (fast_attack - team["team_attack_rating"]).abs()
            + (fast_defence - team["team_defence_rating"]).abs()
        )
        / (2 * team["league_goal_rate"].clip(lower=0.8))
    ).clip(0, 0.75)
    # A promoted side has no immediately preceding Premier League season from
    # which to inherit a full-strength prior.  Treat promotion as an explicit,
    # deadline-known regime change and decay it as current-season evidence
    # arrives.  This is deliberately based only on prior-season membership.
    teams_by_season = {
        int(order): set(frame["team_key"].astype(str))
        for order, frame in team.groupby("season_order", sort=False)
    }
    promoted = np.asarray(
        [
            float(
                int(row.season_order) > 0
                and str(row.team_key)
                not in teams_by_season.get(int(row.season_order) - 1, set())
            )
            for row in team[["season_order", "team_key"]].itertuples(index=False)
        ]
    )
    promotion_regime = promoted * np.exp(-np.maximum(team["GW"].to_numpy(float) - 1, 0) / 8)
    promotion_shrink = np.clip(0.42 * promotion_regime, 0, 0.42)
    team["team_regime_shift"] = (
        1 - (1 - team["team_regime_shift"]) * (1 - 0.62 * promotion_regime)
    ).clip(0, 0.85)
    regime_weight = (0.25 + team["team_regime_shift"]).clip(0.25, 0.75)
    team["team_attack_rating"] = (
        (1 - regime_weight) * team["team_attack_rating"] + regime_weight * fast_attack
    ).clip(0.45, 2.70)
    team["team_defence_rating"] = (
        (1 - regime_weight) * team["team_defence_rating"] + regime_weight * fast_defence
    ).clip(0.45, 2.70)
    team["team_attack_rating"] = (
        (1 - promotion_shrink) * team["team_attack_rating"]
        + promotion_shrink * team["league_goal_rate"]
    ).clip(0.45, 2.70)
    team["team_defence_rating"] = (
        (1 - promotion_shrink) * team["team_defence_rating"]
        + promotion_shrink * team["league_goal_rate"]
    ).clip(0.45, 2.70)
    team["team_rating_confidence"] = (
        team_confidence * (1 - 0.52 * team["team_regime_shift"])
    ).clip(0, 0.94)

    rating_columns = [
        "season",
        "GW",
        "team_id",
        "league_goal_rate",
        "team_attack_rating",
        "team_defence_rating",
        "team_form_rating",
        "team_clean_rating",
        "team_rating_confidence",
        "team_regime_shift",
    ]
    data = data.merge(team[rating_columns], on=["season", "GW", "team_id"], how="left")
    opponent = team[rating_columns].rename(
        columns={
            "team_id": "opponent_team",
            "team_attack_rating": "opponent_attack_rating",
            "team_defence_rating": "opponent_defence_rating",
            "team_form_rating": "opponent_form_rating",
            "team_clean_rating": "opponent_clean_rating",
            "team_rating_confidence": "opponent_rating_confidence",
            "team_regime_shift": "opponent_regime_shift",
            "league_goal_rate": "opponent_league_goal_rate",
        }
    )
    data = data.merge(
        opponent,
        on=["season", "GW", "opponent_team"],
        how="left",
    )
    league_rate = data["league_goal_rate"].fillna(1.40).clip(0.9, 2.0)
    for column in [
        "team_attack_rating",
        "team_defence_rating",
        "opponent_attack_rating",
        "opponent_defence_rating",
    ]:
        data[column] = data[column].fillna(league_rate)
    data["team_form_rating"] = data["team_form_rating"].fillna(1.35)
    data["opponent_form_rating"] = data["opponent_form_rating"].fillna(1.35)
    data["team_clean_rating"] = data["team_clean_rating"].fillna(0.28)
    data["team_rating_confidence"] = data["team_rating_confidence"].fillna(0)
    data["team_regime_shift"] = data["team_regime_shift"].fillna(0)

    home_ga_factor = np.where(data["was_home"].fillna(False), 0.88, 1.12)
    home_gf_factor = np.where(data["was_home"].fillna(False), 1.12, 0.88)
    data["team_expected_goals_against"] = (
        league_rate
        * (data["team_defence_rating"] / league_rate).pow(0.70)
        * (data["opponent_attack_rating"] / league_rate).pow(0.70)
        * home_ga_factor
    ).clip(0.30, 3.40)
    data["team_expected_goals_for"] = (
        league_rate
        * (data["team_attack_rating"] / league_rate).pow(0.70)
        * (data["opponent_defence_rating"] / league_rate).pow(0.70)
        * home_gf_factor
    ).clip(0.30, 3.40)
    data["team_clean_probability"] = np.exp(
        -data["team_expected_goals_against"]
    ).clip(0.03, 0.74)
    attack_index = data["team_attack_rating"] / league_rate
    defence_index = league_rate / data["team_defence_rating"].clip(lower=0.35)
    form_index = data["team_form_rating"] / 1.35
    # Team context is intrinsic team quality. Match difficulty already enters
    # the component forecast through opponent-adjusted expected goals and must
    # not be counted again under a misleading "team strength" label.
    data["team_context_raw"] = (
        0.43 * attack_index
        + 0.43 * defence_index
        + 0.14 * form_index
    ).clip(0.35, 2.75)
    data["team_defence_raw"] = defence_index.clip(0.30, 3.0)
    data["team_attack_raw"] = attack_index.clip(0.30, 3.0)
    return data


def prepare_causal_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Carry player priors across seasons and build component expected points."""
    data = pd.concat(frames, ignore_index=True)
    season_order = {season: index for index, season in enumerate(SEASONS)}
    data["season_order"] = data["season"].map(season_order).astype(int)
    data = add_causal_team_strength(data)
    lineup_rows = data[data["starts_observed"] > 0].groupby(
        ["season", "season_order", "team_id", "GW"], sort=True
    )["element"].agg(lambda values: frozenset(values.astype(int)))
    rotation_records: list[dict] = []
    for (season, season_id, team_id), series in lineup_rows.groupby(level=[0, 1, 2]):
        previous: frozenset[int] | None = None
        history: list[float] = []
        for (_, _, _, gw), starters in series.items():
            prior_rotation = float(np.mean(history[-6:])) if history else 0.22
            rotation_records.append(
                {
                    "season": season,
                    "season_order": season_id,
                    "team_id": team_id,
                    "GW": gw,
                    "team_rotation_rate": prior_rotation,
                }
            )
            if previous is not None:
                denominator = max(1, min(len(previous), len(starters), 11))
                history.append(1 - len(previous.intersection(starters)) / denominator)
            previous = starters
    rotation_table = pd.DataFrame(rotation_records)
    if not rotation_table.empty:
        data = data.merge(
            rotation_table,
            on=["season", "season_order", "team_id", "GW"],
            how="left",
        )
    data["team_rotation_rate"] = data.get("team_rotation_rate", 0.22)
    data["team_rotation_rate"] = data["team_rotation_rate"].fillna(0.22).clip(0, 0.75)
    fallback_key = data["season"].astype(str) + ":" + data["element"].astype(str)
    numeric_code = pd.to_numeric(data["player_code"], errors="coerce")
    data["player_key"] = numeric_code.astype("Int64").astype(str).where(
        numeric_code.notna(), fallback_key
    )
    data.sort_values(
        ["player_key", "season_order", "GW"], inplace=True, kind="stable"
    )
    by_player = data.groupby("player_key", sort=False)
    previous_minutes_observed = by_player["minutes"].shift(1)
    previous_official_xp = by_player["official_xp"].shift(1)
    previous_xp_trusted = by_player["official_xp_feed_trusted"].shift(1).fillna(False)
    official_zero_signal = (
        data["fixture_count"].gt(0)
        & data["official_xp_feed_trusted"].fillna(False)
        & data["official_xp"].le(0)
    )
    consecutive_official_zero = (
        official_zero_signal
        & previous_xp_trusted
        & previous_official_xp.le(0)
        & previous_minutes_observed.fillna(0).le(0)
    )
    market_availability_warning = (
        data["fixture_count"].gt(0)
        & data["transfer_pressure_raw"].le(-0.90)
        & previous_minutes_observed.notna()
        & previous_minutes_observed.lt(60)
    )
    severe_official_warning = official_zero_signal & data[
        "transfer_pressure_raw"
    ].le(-0.75)
    data["official_zero_availability_signal"] = official_zero_signal
    data["market_availability_warning"] = market_availability_warning
    data["severe_availability_warning"] = (
        severe_official_warning
        | consecutive_official_zero
        | market_availability_warning
    )
    data["observations"] = by_player["fixture_count"].transform(
        lambda values: values.cumsum().shift(1)
    ).fillna(0)

    points_prior = data["position_id"].map({1: 3.2, 2: 2.6, 3: 2.8, 4: 2.6})
    minutes_prior = data["position_id"].map({1: 0.66, 2: 0.58, 3: 0.57, 4: 0.55})
    underlying_prior = data["position_id"].map({1: 2.5, 2: 4.0, 3: 6.0, 4: 6.5})

    # Per-fixture, not per-Gameweek: the empirical model that consumes these is
    # blended with the other per-fixture routes and the whole blend is scaled by
    # the number of fixtures once, at the end.
    data["performance_points"] = (
        data["points"] / data["fixture_count"].clip(lower=1)
    ).where(data["fixture_count"] > 0)
    data["performance_minutes"] = (
        data["minutes"] / data["fixture_count"].clip(lower=1)
    ).where(data["fixture_count"] > 0)
    by_player = data.groupby("player_key", sort=False)
    data["long_raw"] = by_player["performance_points"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(points_prior)
    data["recent_raw"] = by_player["performance_points"].transform(
        lambda values: values.rolling(6, min_periods=1).mean().shift(1)
    ).fillna(data["long_raw"])
    data["past_minutes"] = by_player["performance_minutes"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(minutes_prior * 90)
    per_fixture_minutes = (
        data["minutes"] / data["fixture_count"].clip(lower=1)
    ).clip(upper=90)
    data["per_fixture_minutes"] = per_fixture_minutes

    def rolling_total(column: str, window: int = 10) -> pd.Series:
        return data.groupby("player_key", sort=False)[column].transform(
            lambda values: values.rolling(window, min_periods=1).sum().shift(1)
        )

    prior_start = data["position_id"].map({1: 0.68, 2: 0.58, 3: 0.56, 4: 0.54})
    prior_sub = data["position_id"].map({1: 0.05, 2: 0.30, 3: 0.42, 4: 0.43})
    prior_sixty_start = data["position_id"].map({1: 0.95, 2: 0.82, 3: 0.76, 4: 0.72})
    prior_strength = 4.0
    prior_games = rolling_total("fixture_count").fillna(0)
    prior_starts = rolling_total("starts_observed").fillna(0)
    prior_appearances = rolling_total("appearances_observed").fillna(0)
    prior_sixties = rolling_total("sixty_observed").fillna(0)
    prior_bench_appearances = rolling_total("bench_appearances_observed").fillna(0)
    prior_nonstarts = (prior_games - prior_starts).clip(lower=0)
    data["start_probability"] = (
        (prior_starts + prior_strength * prior_start)
        / (prior_games + prior_strength)
    ).clip(0.03, 0.98)
    viable_starter = (data["start_probability"] >= 0.28).astype(float)
    competition_count = viable_starter.groupby(
        [data["season"], data["GW"], data["team_id"], data["position_id"]]
    ).transform("sum")
    role_slots = data["position_id"].map({1: 1.0, 2: 4.0, 3: 4.0, 4: 2.0})
    data["competition_pressure"] = (
        (competition_count - role_slots).clip(lower=0) / role_slots
    ).clip(0, 1.5)
    data["sub_probability_given_bench"] = (
        (prior_bench_appearances + 3.0 * prior_sub)
        / (prior_nonstarts + 3.0)
    ).clip(0.02, 0.88)
    data["sixty_probability_given_start"] = (
        (prior_sixties + 4.0 * prior_sixty_start)
        / (prior_starts + 4.0)
    ).clip(0.25, 0.99)
    start_minutes_sum = rolling_total("start_minutes_total").fillna(0)
    bench_minutes_sum = rolling_total("bench_minutes_total").fillna(0)
    start_minutes_prior = data["position_id"].map({1: 88.0, 2: 80.0, 3: 76.0, 4: 73.0})
    bench_minutes_prior = data["position_id"].map({1: 5.0, 2: 16.0, 3: 20.0, 4: 22.0})
    data["minutes_if_start"] = (
        start_minutes_sum + 3.0 * start_minutes_prior
    ) / (prior_starts + 3.0)
    data["minutes_if_bench"] = (
        bench_minutes_sum + 3.0 * bench_minutes_prior
    ) / (prior_bench_appearances + 3.0)
    start_rate_observed = data["starts_observed"] / data["fixture_count"].clip(lower=1)
    data["starts_for_rotation"] = data["starts_observed"].where(
        data["fixture_count"] > 0
    )
    data["rotation_volatility"] = data.groupby("player_key", sort=False)[
        "starts_for_rotation"
    ].transform(
        lambda values: values.rolling(8, min_periods=2).std().shift(1)
    ).fillna(0.35).clip(0, 1)
    rest_penalty = (
        (4.5 - data["team_rest_days"].fillna(7)).clip(lower=0) / 5.0
        * (0.30 + 0.70 * data["rotation_volatility"])
    ).clip(0, 0.28)
    rotation_penalty = (
        (data["team_rotation_rate"] - 0.18).clip(lower=0)
        * data["rotation_volatility"]
        * 0.35
    ).clip(0, 0.15)
    competition_penalty = (
        0.045 * data["competition_pressure"] * (0.35 + data["rotation_volatility"])
    ).clip(0, 0.10)
    data["start_probability"] *= 1 - rest_penalty - rotation_penalty - competition_penalty
    # The historical archive does not carry a complete weekly injury field, but
    # from 2020/21 it does retain official FPL xP captured around the deadline.
    # A zero xP field alone is not enough to suppress a player: genuine doubts
    # frequently recover in time. It becomes actionable only when combined with
    # an extreme market exit, a prior zero/no-show, or a curtailed appearance.
    # The high-precision warning affects minutes/availability—not scoring rate—
    # and is disabled for all-zero or otherwise corrupted xP feeds.
    start_availability_multiplier = np.select(
        [
            data["severe_availability_warning"],
            data["official_zero_availability_signal"],
        ],
        [0.05, 1.0],
        default=1.0,
    )
    sub_availability_multiplier = np.select(
        [
            data["severe_availability_warning"],
            data["official_zero_availability_signal"],
        ],
        [0.15, 1.0],
        default=1.0,
    )
    data["start_probability"] *= start_availability_multiplier
    data["sub_probability_given_bench"] *= sub_availability_multiplier
    data["start_probability"] = data["start_probability"].clip(0.02, 0.98)
    data["play_probability"] = (
        data["start_probability"]
        + (1 - data["start_probability"]) * data["sub_probability_given_bench"]
    ).clip(0.05, 0.995)
    data["sixty_probability"] = (
        data["start_probability"] * data["sixty_probability_given_start"]
    ).clip(0.02, 0.98)
    # Remove the compression before anything downstream consumes these. Every
    # quantity below is rebuilt from the calibrated decomposition.
    data = causal_calibrate_minutes(data)
    data["expected_minutes"] = (
        data["start_probability"] * data["minutes_if_start"]
        + (1 - data["start_probability"])
        * data["sub_probability_given_bench"]
        * data["minutes_if_bench"]
    ).clip(3, 90)
    second_moment = (
        data["start_probability"]
        * (data["minutes_if_start"].pow(2) + 12**2)
        + (1 - data["start_probability"])
        * data["sub_probability_given_bench"]
        * (data["minutes_if_bench"].pow(2) + 10**2)
    )
    data["minutes_std"] = np.sqrt(
        (second_moment - data["expected_minutes"].pow(2)).clip(lower=16)
    ).clip(4, 42)
    data["minutes_security_raw"] = (
        0.65 * data["sixty_probability"] + 0.35 * data["play_probability"]
    ).clip(0.05, 1.0)
    data["minutes_model_confidence"] = (
        prior_games / (prior_games + 8)
    ).clip(0, 0.95)
    data["start_observed_rate"] = start_rate_observed.clip(0, 1)
    data["sixty_observed_rate"] = (
        data["sixty_observed"] / data["fixture_count"].clip(lower=1)
    ).clip(0, 1)

    data["underlying_game"] = (
        data["ict"] / data["minutes"].clip(lower=45) * 90
    ).clip(0, 35).where(data["fixture_count"] > 0)
    data["long_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.expanding().mean().shift(1)
    ).fillna(underlying_prior)
    data["recent_underlying_raw"] = by_player["underlying_game"].transform(
        lambda values: values.rolling(6, min_periods=1).mean().shift(1)
    ).fillna(data["long_underlying_raw"])

    # Every scoring rate below is a *conditional-on-playing* quantity: the
    # component forecast multiplies it by expected minutes, so availability must
    # not also be baked into the rate itself. A week with no fixture, or a week
    # spent as an unused substitute, is therefore censored rather than recorded
    # as a zero-return appearance. The per-appearance denominator likewise keeps
    # a Double Gameweek from inflating a per-match rate.
    appeared = data["appearances_observed"] > 0
    appearance_denominator = data["appearances_observed"].clip(lower=1)
    minute_denominator = data["minutes"].clip(lower=45)
    data["goal_signal_game"] = (
        pd.Series(
            np.where(
                data["expected_goals"] > 0,
                0.72 * data["expected_goals"] + 0.28 * data["goals"],
                data["goals"],
            ),
            index=data.index,
        )
        / minute_denominator
        * 90
    ).where(appeared)
    data["assist_signal_game"] = (
        pd.Series(
            np.where(
                data["expected_assists"] > 0,
                0.72 * data["expected_assists"] + 0.28 * data["assists"],
                data["assists"],
            ),
            index=data.index,
        )
        / minute_denominator
        * 90
    ).where(appeared)
    data["clean_sheet_game"] = (
        (data["clean_sheets"] / appearance_denominator).clip(0, 1).where(appeared)
    )
    data["defensive_actions_game"] = (
        (data["defensive_actions"] / appearance_denominator)
        .clip(0, 35)
        .where(appeared)
    )
    data["bps_game"] = (
        (data["bps"] / appearance_denominator).clip(-10, 80).where(appeared)
    )
    # These official feeds arrive as Gameweek totals, so they need the same
    # per-appearance normalisation before they can be used as per-match rates.
    for raw_column in (
        "saves",
        "bonus",
        "yellow_cards",
        "red_cards",
        "goals_conceded",
        "penalties_saved",
        "penalties_missed",
        "own_goals",
    ):
        data[f"{raw_column}_game"] = (
            data[raw_column] / appearance_denominator
        ).where(appeared)
    for source, target, prior in [
        ("goal_signal_game", "goal_rate", {1: 0.01, 2: 0.04, 3: 0.20, 4: 0.28}),
        ("assist_signal_game", "assist_rate", {1: 0.01, 2: 0.08, 3: 0.18, 4: 0.13}),
        ("clean_sheet_game", "clean_sheet_rate", {1: 0.28, 2: 0.28, 3: 0.22, 4: 0.0}),
        ("saves_game", "save_rate", {1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}),
        ("bonus_game", "bonus_rate", {1: 0.18, 2: 0.22, 3: 0.28, 4: 0.28}),
        ("yellow_cards_game", "yellow_rate", {1: 0.05, 2: 0.12, 3: 0.10, 4: 0.08}),
        ("red_cards_game", "red_rate", {1: 0.005, 2: 0.008, 3: 0.006, 4: 0.005}),
        ("goals_conceded_game", "conceded_rate", {1: 1.35, 2: 1.35, 3: 0.0, 4: 0.0}),
        ("defensive_actions_game", "defensive_rate", {1: 0.0, 2: 6.8, 3: 6.0, 4: 3.0}),
        ("bps_game", "bps_rate", {1: 15.0, 2: 14.0, 3: 12.0, 4: 11.0}),
        ("penalties_saved_game", "penalty_save_rate", {1: 0.025, 2: 0.0, 3: 0.0, 4: 0.0}),
        ("penalties_missed_game", "penalty_miss_rate", {1: 0.0, 2: 0.002, 3: 0.01, 4: 0.015}),
        ("own_goals_game", "own_goal_rate", {1: 0.002, 2: 0.008, 3: 0.003, 4: 0.002}),
    ]:
        rolling = data.groupby("player_key", sort=False)[source].transform(
            lambda values: values.rolling(12, min_periods=1).mean().shift(1)
        )
        data[target] = rolling.fillna(data["position_id"].map(prior)).clip(lower=0)

    data["player_role"] = assign_player_role(data)
    # Only appearances carry defensive-action evidence; an unused substitute is
    # not a game in which the player recorded zero actions.
    data["defensive_exact_games"] = (
        data["appearances_observed"] * data["defensive_exact"]
    )
    data["defensive_exact_actions"] = (
        data["defensive_actions_game"] * data["defensive_exact_games"]
    )
    # Coverage answers "did the feed exist for this fixture", which is a
    # scheduling question rather than a selection one, so it keeps the fixture
    # denominator even though the rate itself is per appearance.
    data["defensive_feed_games"] = data["fixture_count"] * data["defensive_exact"]
    exact_games_prior = rolling_total("defensive_exact_games", 18).fillna(0)
    exact_actions_prior = rolling_total("defensive_exact_actions", 18).fillna(0)
    feed_games_prior = rolling_total("defensive_feed_games", 18).fillna(0)
    role_defensive_prior = data["player_role"].map(
        {
            "shot_stopper": 0.0,
            "clean_sheet_keeper": 0.0,
            "centre_back": 8.2,
            "set_piece_centre_back": 8.8,
            "attacking_full_back": 6.2,
            "balanced_defender": 6.8,
            "holding_midfielder": 8.0,
            "creator": 5.4,
            "goal_threat_midfielder": 4.8,
            "box_to_box_midfielder": 6.3,
            "link_forward": 3.3,
            "penalty_box_forward": 2.4,
            "mobile_forward": 3.0,
        }
    ).fillna(4.0)
    data["defensive_event_coverage"] = (
        feed_games_prior / prior_games.clip(lower=1)
    ).clip(0, 1)
    data["defensive_rate"] = (
        exact_actions_prior + 5.0 * role_defensive_prior
    ) / (exact_games_prior + 5.0)
    data["player_role"] = assign_player_role(data)

    minutes_factor = data["expected_minutes"] / 90
    p_play = data["play_probability"]
    p_sixty = data["sixty_probability"]
    goal_points = data["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4})
    clean_sheet_points = data["position_id"].map({1: 4, 2: 4, 3: 1, 4: 0})
    appearance_points = p_play + p_sixty
    vulnerability_group = ["season", "GW", "position_id"]
    goal_vulnerability = (
        data["opponent_goal_vulnerability"]
        / data.groupby(vulnerability_group)["opponent_goal_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).clip(0.68, 1.42)
    assist_vulnerability = (
        data["opponent_assist_vulnerability"]
        / data.groupby(vulnerability_group)["opponent_assist_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).clip(0.72, 1.35)
    # Team attacking strength for *this* fixture, back-ported from the live
    # forecast so the two paths price the attacking route identically. It is a
    # relative factor centred on 1.0, so it adjusts a player's long-run personal
    # rate to the match in front of him rather than re-adding team quality.
    team_attack_multiplier = (
        data["team_expected_goals_for"] / data["league_goal_rate"].clip(lower=0.9)
    ).pow(0.45).clip(0.70, 1.38)
    attacking_points = (
        data["goal_rate"] * goal_points * goal_vulnerability
        + data["assist_rate"] * 3 * assist_vulnerability
    ) * minutes_factor * team_attack_multiplier
    blended_clean_probability = (
        0.82 * data["team_clean_probability"]
        + 0.18 * data["clean_sheet_rate"]
    ).clip(0.03, 0.78)
    clean_sheet_points_ev = (
        blended_clean_probability * clean_sheet_points * p_sixty
    )
    save_points = (
        data["save_rate"] / 3 * minutes_factor
        * np.where(data["position_id"] == 1, 1.0, 0.0)
    )
    defender_clean_bonus = (
        0.12
        * data["team_clean_probability"]
        * p_sixty
        * data["position_id"].isin([1, 2]).astype(float)
    )
    bps_rule_multiplier = np.select(
        [
            data["position_id"] == 1,
            (data["position_id"] == 2) & (data["defensive_rate"] >= 9),
            data["position_id"].isin([3, 4]),
        ],
        [1.06, 0.94, 1.03],
        default=1.0,
    )
    bonus_points = (
        data["bonus_rate"] * minutes_factor * bps_rule_multiplier
        + defender_clean_bonus
    )
    discipline_points = -(
        data["yellow_rate"] + 3 * data["red_rate"]
    ) * minutes_factor
    rare_event_points = (
        5 * data["penalty_save_rate"]
        - 2 * data["penalty_miss_rate"]
        - 2 * data["own_goal_rate"]
    ) * minutes_factor
    conceded_points = -(
        data["team_expected_goals_against"] / 2
        * minutes_factor
        * data["position_id"].isin([1, 2]).astype(float)
    )
    defensive_threshold = np.where(data["position_id"] == 2, 10.0, 12.0)
    data["defensive_return_probability"] = poisson_tail(
        data["defensive_rate"] * minutes_factor,
        defensive_threshold,
    ) * data["position_id"].isin([2, 3, 4]).astype(float)
    defensive_points_current_rules = 2 * data["defensive_return_probability"]
    defensive_points_season_rules = defensive_points_current_rules * (
        data["season_order"] >= season_order.get("2025-26", 99)
    ).astype(float)
    structural_without_dc = (
        appearance_points
        + attacking_points
        + clean_sheet_points_ev
        + save_points
        + bonus_points
        + discipline_points
        + rare_event_points
        + conceded_points
    )
    # Every route above is priced for one match. The number of matches is
    # applied once, after the ensemble blend, so that all four models scale with
    # a Double Gameweek instead of only the structural one.
    fixture_multiplier = data["fixture_count"].clip(lower=1)
    data["structural_per_fixture"] = (
        structural_without_dc + defensive_points_season_rules
    ).clip(0.2, 13.0)
    data["structural_per_fixture_current_rules"] = (
        structural_without_dc + defensive_points_current_rules
    ).clip(0.2, 13.5)
    data["component_xpts_structural"] = (
        data["structural_per_fixture"] * fixture_multiplier
    )
    data["component_xpts_current_rules"] = (
        data["structural_per_fixture_current_rules"] * fixture_multiplier
    )

    data["empirical_xpts"] = (
        (0.62 * data["recent_raw"] + 0.38 * data["long_raw"])
        * (0.72 + 0.28 * data["play_probability"])
    ).clip(0.2, 13.5)
    position_base = data["position_id"].map({1: 3.2, 2: 2.8, 3: 3.0, 4: 2.8})
    data["market_role_xpts"] = (
        position_base
        * (0.64 + 0.46 * data["minutes_security_raw"])
        * (0.82 + 0.28 * data["team_context_raw"].clip(0.4, 1.8))
        * (
            0.94
            + 0.12
            * data.groupby(["season", "GW", "position_id"])["crowd_raw"]
            .transform(percentile)
        )
    ).clip(0.2, 13.5)
    data["role_ridge_xpts"] = causal_role_ridge_predictions(data).clip(0.2, 13.5)
    # The blend is built on the per-fixture scale, so the errors that set its
    # weights are measured against per-fixture realised points too. Comparing a
    # Gameweek-total structural model with three per-match challengers made
    # every Double Gameweek look like a structural-model failure.
    ensemble_models = [
        "structural_per_fixture",
        "empirical_xpts",
        "market_role_xpts",
        "role_ridge_xpts",
    ]
    points_per_fixture = (
        data["points"] / data["fixture_count"].clip(lower=1)
    ).where(data["fixture_count"] > 0)
    error_keys = ["season_order", "GW", "position_id"]
    error_table = data[error_keys].drop_duplicates().sort_values(error_keys).copy()
    for model_name in ensemble_models:
        data[f"{model_name}_absolute_error"] = (
            data[model_name] - points_per_fixture
        ).abs()
        # The *signed* error matters as much as its size. Inverse-error weighting
        # only balances precision, so a member that is systematically high drags
        # the blend's level with it however small its weight — and the transfer
        # hurdles and chip thresholds are denominated in points.
        data[f"{model_name}_signed_error"] = data[model_name] - points_per_fixture
        weekly_error = (
            data.groupby(error_keys, as_index=False)[
                [f"{model_name}_absolute_error", f"{model_name}_signed_error"]
            ]
            .mean()
            .sort_values(error_keys)
        )
        for source, target in (
            (f"{model_name}_absolute_error", f"{model_name}_mae"),
            (f"{model_name}_signed_error", f"{model_name}_bias"),
        ):
            weekly_error[target] = weekly_error.groupby(
                "position_id", sort=False
            )[source].transform(
                lambda values: values.expanding().mean().shift(1)
            )
        error_table = error_table.merge(
            weekly_error[
                error_keys + [f"{model_name}_mae", f"{model_name}_bias"]
            ],
            on=error_keys,
            how="left",
        )
    data = data.merge(error_table, on=error_keys, how="left")
    data = data.copy()
    # The merge above rebuilds the index, so any Series captured before it can
    # no longer be aligned against the new frame.
    fixture_multiplier = data["fixture_count"].clip(lower=1)
    default_mae = {
        "structural_per_fixture_mae": 2.85,
        "empirical_xpts_mae": 3.05,
        "market_role_xpts_mae": 3.25,
        "role_ridge_xpts_mae": 3.10,
    }
    inverse_errors = []
    for column, fallback in default_mae.items():
        data[column] = data[column].fillna(fallback).clip(1.4, 6.0)
        inverse_errors.append(1 / data[column].pow(2))
    inverse_total = sum(inverse_errors)
    data["ensemble_structural_weight"] = inverse_errors[0] / inverse_total
    data["ensemble_empirical_weight"] = inverse_errors[1] / inverse_total
    data["ensemble_market_weight"] = inverse_errors[2] / inverse_total
    data["ensemble_role_weight"] = inverse_errors[3] / inverse_total
    # Level-correct each member on its own prior-season bias before blending.
    # Dropping the worst member outright was tested and lost within-Gameweek
    # ranking power, so the diversity is kept and only the level is repaired.
    corrected_models = []
    for model_name in ensemble_models:
        bias_column = f"{model_name}_bias"
        data[bias_column] = data[bias_column].fillna(0.0).clip(-1.5, 1.5)
        corrected = f"{model_name}_levelled"
        data[corrected] = (data[model_name] - data[bias_column]).clip(
            lower=0.2, upper=13.5
        )
        corrected_models.append(corrected)
    data["component_per_fixture"] = (
        data["ensemble_structural_weight"] * data[corrected_models[0]]
        + data["ensemble_empirical_weight"] * data[corrected_models[1]]
        + data["ensemble_market_weight"] * data[corrected_models[2]]
        + data["ensemble_role_weight"] * data[corrected_models[3]]
    ).clip(0.2, 13.5)
    data["component_xpts_base"] = data["component_per_fixture"] * fixture_multiplier
    data["component_xpts"] = data["component_xpts_base"] * (
        data["fixture_count"] > 0
    ).astype(float)
    current_rule_uplift = (
        data["structural_per_fixture_current_rules"]
        - data["structural_per_fixture"]
    )
    data["component_per_fixture_current_rules"] = (
        data["component_per_fixture"] + current_rule_uplift
    ).clip(0.2, 14.0)
    data["ensemble_xpts_current_rules_base"] = (
        data["component_per_fixture_current_rules"] * fixture_multiplier
    )
    data["ensemble_xpts_current_rules"] = data[
        "ensemble_xpts_current_rules_base"
    ] * (data["fixture_count"] > 0).astype(float)
    # Disagreement between level-corrected members is genuine uncertainty; between
    # raw members it partly just measures their different calibrations.
    model_stack = data[corrected_models].to_numpy(float)
    data["ensemble_disagreement"] = np.std(model_stack, axis=1)
    # The immediate component already contains the current opponent through
    # team xG, clean-sheet probability and opponent vulnerability. Translate it
    # to the future slate with a relative horizon/current adjustment rather
    # than multiplying by a second absolute fixture score.
    horizon_multiplier = (
        (data["fixture_horizon_raw"].fillna(2.5) + 1.5)
        / (data["fixture_raw"].fillna(2.5) + 1.5)
    ).pow(0.35).clip(0.78, 1.28)
    single_fixture_base = data["component_per_fixture"]
    data["component_horizon"] = (
        single_fixture_base
        * data["horizon_weighted_games"].clip(lower=1)
        * horizon_multiplier
    ).clip(0.5, 50)
    # Now that the censored horizon carries real future-opponent information it
    # should not be held to a tighter band than the uncensored diagnostic.
    censored_horizon_multiplier = (
        (data["fixture_horizon_censored_raw"].fillna(2.5) + 1.5)
        / (data["fixture_raw"].fillna(2.5) + 1.5)
    ).pow(0.35).clip(0.78, 1.28)
    data["component_horizon_censored"] = (
        single_fixture_base
        * data["horizon_weighted_games_censored"].clip(lower=0)
        * censored_horizon_multiplier
    ).clip(0.0, 50)
    current_rule_single_fixture = data["component_per_fixture_current_rules"]
    data["component_horizon_current_rules"] = (
        current_rule_single_fixture
        * data["horizon_weighted_games"].clip(lower=1)
        * horizon_multiplier
    ).clip(0.5, 52)
    data["component_horizon_current_rules_censored"] = (
        current_rule_single_fixture
        * data["horizon_weighted_games_censored"].clip(lower=0)
        * censored_horizon_multiplier
    ).clip(0.0, 52)
    # The bracket is a one-match variance, so a Double Gameweek total carries
    # roughly twice that variance: scale the clipped per-match standard
    # deviation by sqrt(fixtures). Single Gameweeks are unchanged.
    data["prediction_uncertainty"] = np.sqrt(
        1.1**2
        + 0.020 * data["minutes_std"].pow(2)
        + 0.85 * data["ensemble_disagreement"].pow(2)
        + 2.2 / np.sqrt(data["observations"] + 1)
    ).clip(1.2, 5.5) * np.sqrt(fixture_multiplier)
    data["raw_blank_probability"] = normal_cdf(
        (2.5 - data["component_xpts"]) / data["prediction_uncertainty"]
    )
    data["raw_return5_probability"] = 1 - normal_cdf(
        (4.5 - data["component_xpts"]) / data["prediction_uncertainty"]
    )
    data["raw_haul8_probability"] = 1 - normal_cdf(
        (7.5 - data["component_xpts"]) / data["prediction_uncertainty"]
    )
    data["blank_probability"] = data["raw_blank_probability"]
    data["return5_probability"] = data["raw_return5_probability"]
    data["haul8_probability"] = data["raw_haul8_probability"]
    data = causal_calibrate_distributions(data)

    lagged_transfer_balance = by_player["transfers_balance"].shift(1).fillna(0)
    transfer_momentum = np.sign(lagged_transfer_balance) * np.log1p(
        lagged_transfer_balance.abs()
    )
    data["crowd_raw"] = (
        np.log1p(data["selected"].clip(lower=0)) + 0.08 * transfer_momentum
    )
    data["long_value_raw"] = data["long_raw"] / (data["price"] / 10).clip(3.5)
    data["recent_value_raw"] = data["component_xpts"] / (
        data["price"] / 10
    ).clip(3.5)
    data["age_raw"] = np.exp(-((data["age"].fillna(27.5) - 27.5) / 7.5) ** 2)

    rank_groups = ["season", "GW", "position_id"]
    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("component_xpts", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_horizon_censored_raw", "fixture_censored"),
        ("fixture_raw", "fixture_now"),
        ("team_context_raw", "team_context"),
        ("team_defence_raw", "team_defence"),
        ("team_attack_raw", "team_attack"),
        ("crowd_raw", "crowd"),
        ("minutes_security_raw", "minutes_security"),
        ("long_underlying_raw", "long_underlying"),
        ("recent_underlying_raw", "recent_underlying"),
    ]:
        data[rank_name] = data.groupby(rank_groups)[raw_name].transform(percentile)

    horizon_groups = data.groupby(["season", "player_key"], sort=False)["points"]
    horizon_target = pd.Series(0.0, index=data.index)
    horizon_target_end = data["GW"].astype(int).copy()
    for offset in range(6):
        horizon_target += (0.86**offset) * horizon_groups.shift(-offset).fillna(0)
        shifted_gw = data.groupby(["season", "player_key"], sort=False)["GW"].shift(
            -offset
        )
        horizon_target_end = np.maximum(
            horizon_target_end,
            shifted_gw.fillna(horizon_target_end).astype(int),
        )
    data["horizon_target"] = horizon_target
    data["horizon_target_end_gw"] = horizon_target_end.astype(int)
    data["causal_horizon_ridge"] = causal_horizon_ridge_predictions(data)

    data.sort_values(
        ["season_order", "GW", "position_id", "element"], inplace=True, kind="stable"
    )
    data.reset_index(drop=True, inplace=True)
    return data


def load_or_build_prepared_history(
    force: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Reuse the deterministic historical feature table across experiment runs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if PREPARED_HISTORY_CACHE.exists() and not force:
        payload = pd.read_pickle(PREPARED_HISTORY_CACHE)
        data = payload["data"]
        summaries = payload["summaries"]
        if list(dict.fromkeys(data["season"].tolist())) == SEASONS:
            return data, summaries
    ages = load_age_register()
    nationalities = load_nationality_register()
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for season in SEASONS:
        frame, summary = build_season(season, ages, nationalities)
        frames.append(frame)
        summaries.append(summary)
        print(f"Prepared {season}: {summary['eligibleRows']:,} eligible player-weeks")
    data = prepare_causal_history(frames)
    temporary = PREPARED_HISTORY_CACHE.with_suffix(".tmp")
    pd.to_pickle({"data": data, "summaries": summaries}, temporary)
    temporary.replace(PREPARED_HISTORY_CACHE)
    return data, summaries


def fixture_integrity_audit(data: pd.DataFrame) -> dict:
    """Prove that player fixture counts agree with the club schedule.

    A large zero-fixture player share is expected in an FA Cup blank. The
    meaningful invariant is that every player registered to a club has the same
    0/1/2-fixture count as that club in that event.
    """
    team_weeks = data.groupby(["season", "GW", "team_id"], as_index=False).agg(
        player_fixture_min=("fixture_count", "min"),
        player_fixture_max=("fixture_count", "max"),
        scheduled_team_games=("team_games", "max"),
    )
    mismatches = team_weeks[
        team_weeks["player_fixture_min"].ne(team_weeks["scheduled_team_games"])
        | team_weeks["player_fixture_max"].ne(team_weeks["scheduled_team_games"])
    ]
    mass_blanks: list[dict] = []
    for (season, gw), frame in data.groupby(["season", "GW"], sort=False):
        no_fixture = int(frame["fixture_count"].le(0).sum())
        share = no_fixture / max(1, len(frame))
        if share < 0.40:
            continue
        active_clubs = int(
            team_weeks.loc[
                team_weeks["season"].eq(season)
                & team_weeks["GW"].eq(gw)
                & team_weeks["scheduled_team_games"].gt(0),
                "team_id",
            ].nunique()
        )
        mass_blanks.append(
            {
                "season": str(season),
                "gameweek": int(gw),
                "activeClubs": active_clubs,
                "playersWithNoFixture": no_fixture,
                "registeredPlayers": int(len(frame)),
                "noFixtureShare": round(100 * share, 1),
                "explainedByReducedSlate": active_clubs <= 12,
            }
        )
    unexplained = [
        item for item in mass_blanks if not item["explainedByReducedSlate"]
    ]
    return {
        "passed": bool(mismatches.empty and not unexplained),
        "teamWeekRows": int(len(team_weeks)),
        "playerVsClubFixtureMismatches": int(len(mismatches)),
        "unexplainedMassBlankRounds": len(unexplained),
        "massBlankRounds": mass_blanks,
        "invariant": "Every registered player's 0/1/2 fixture count must equal the scheduled game count of their club in that event.",
    }


@dataclass(frozen=True)
class Candidate:
    performance: float
    value: float
    age: float
    fixture: float
    team: float
    crowd: float
    minutes: float
    underlying: float
    recent_share: float

    @property
    def coefficients(self) -> np.ndarray:
        return np.array(
            [
                self.performance * self.recent_share,
                self.performance * (1 - self.recent_share),
                self.value * self.recent_share,
                self.value * (1 - self.recent_share),
                self.age,
                self.fixture,
                self.team,
                self.crowd,
                self.minutes,
                self.underlying * self.recent_share,
                self.underlying * (1 - self.recent_share),
            ],
            dtype=float,
        )

    def as_dict(self) -> dict:
        raw = {
            "performance": self.performance,
            "value": self.value,
            "age": self.age,
            "fixture": self.fixture,
            "team": self.team,
            "crowd": self.crowd,
            "minutes": self.minutes,
            "underlying": self.underlying,
        }
        rounded = {key: round(value * 100) for key, value in raw.items()}
        rounded[max(raw, key=raw.get)] += 100 - sum(rounded.values())
        rounded["recent"] = round(self.recent_share * 100)
        rounded["history"] = 100 - rounded["recent"]
        return rounded


def candidate_pool() -> tuple[list[Candidate], int]:
    rng = np.random.default_rng(20260811)
    raw_weights = rng.dirichlet(
        [4.0, 1.2, 0.20, 1.7, 2.3, 0.35, 2.6, 2.0], size=TRIALS - 5
    )
    recent = rng.beta(5.0, 1.8, size=TRIALS - 5) * 0.55 + 0.40
    candidates = [
        Candidate(*weights, float(recency))
        for weights, recency in zip(raw_weights, recent)
    ]
    candidates.extend(
        [
            # Official-winner principles: form + medium-term fixtures, reliable
            # minutes, underlying data, restrained ownership and almost no age prior.
            Candidate(0.30, 0.06, 0.00, 0.13, 0.17, 0.03, 0.18, 0.13, 0.78),
            Candidate(0.36, 0.05, 0.00, 0.13, 0.18, 0.02, 0.17, 0.09, 0.82),
            Candidate(0.28, 0.07, 0.00, 0.12, 0.19, 0.03, 0.18, 0.13, 0.72),
            Candidate(0.38, 0.04, 0.00, 0.10, 0.18, 0.02, 0.17, 0.11, 0.76),
            # Lens 1.0: retained as a proper recursive baseline.
            Candidate(0.36, 0.09, 0.01, 0.04, 0.00, 0.50, 0.00, 0.00, 0.59),
        ]
    )
    return candidates, len(candidates) - 1


def feature_matrix(data: pd.DataFrame) -> np.ndarray:
    return data[
        [
            "recent",
            "long",
            "recent_value",
            "long_value",
            "age_score",
            "fixture_censored",
            "team_context",
            "crowd",
            "minutes_security",
            "recent_underlying",
            "long_underlying",
        ]
    ].to_numpy(dtype=float)


def candidate_forecasts(
    data: pd.DataFrame,
    candidate: Candidate,
    current_rules: bool = False,
    robust_planning: bool = True,
    schedule_censored: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = feature_matrix(data)
    model_score = features @ candidate.coefficients
    # Fixture and intrinsic team strength already enter the scoring components
    # through opponent-adjusted xG/clean-sheet routes. They remain available as
    # ranking diagnostics, but are zeroed in the generic calibration multiplier
    # so the same match is not rewarded a second time.
    calibration_coefficients = candidate.coefficients.copy()
    calibration_coefficients[5:7] = 0.0
    remaining_weight = float(calibration_coefficients.sum())
    if remaining_weight > 0:
        calibration_coefficients /= remaining_weight
    calibration_score = features @ calibration_coefficients
    calibration = 0.72 + 0.56 * calibration_score
    current_column = "ensemble_xpts_current_rules" if current_rules else "component_xpts"
    horizon_column = (
        "component_horizon_current_rules_censored"
        if current_rules and schedule_censored
        else "component_horizon_current_rules"
        if current_rules
        else "component_horizon_censored"
        if schedule_censored
        else "component_horizon"
    )
    current = data[current_column].to_numpy(float) * calibration
    horizon = data[horizon_column].to_numpy(float) * calibration
    horizon_risk = (
        data["prediction_uncertainty"].to_numpy(float)
        * np.sqrt(
            data[
                "horizon_weighted_games_censored"
                if schedule_censored
                else "horizon_weighted_games"
            ].to_numpy(float).clip(1, None)
        )
    )
    price_option = (
        data["price_rise_probability"].to_numpy(float)
        - data["price_fall_probability"].to_numpy(float)
    )
    upside = data["haul8_probability"].to_numpy(float)
    robust_plan = horizon - 0.10 * horizon_risk + 0.32 * price_option + 0.20 * upside
    return current, robust_plan if robust_planning else horizon, model_score


def snapshot_replay(
    data: pd.DataFrame, candidates: list[Candidate]
) -> tuple[np.ndarray, list[str]]:
    """Fast predictive screen before the fully stateful season replay."""
    features = feature_matrix(data)
    actual = data["points"].to_numpy(dtype=float)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    coefficients = np.vstack([candidate.coefficients for candidate in candidates])
    results = np.zeros((len(candidates), len(seasons)), dtype=float)
    for season_id, season in enumerate(seasons):
        mask = (
            (data["season"].to_numpy() == season)
            & (data["fixture_count"].to_numpy(int) > 0)
        )
        season_features = features[mask]
        season_actual = actual[mask]
        centered_features = season_features - season_features.mean(axis=0)
        centered_actual = season_actual - season_actual.mean()
        covariance = centered_features.T @ centered_actual
        feature_covariance = centered_features.T @ centered_features
        numerator = coefficients @ covariance
        denominator = np.sqrt(
            np.einsum(
                "ij,jk,ik->i", coefficients, feature_covariance, coefficients
            )
            * float(centered_actual @ centered_actual)
        )
        results[:, season_id] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    return results, seasons


@dataclass(frozen=True)
class SimulationStrategy:
    name: str
    transfer_hurdle: float
    bank_limit: int
    force_weekly_review: bool
    safe_captain: bool
    max_hits: int = 3
    hit_immediate_hurdle: float = 2.5
    joint_chip_preflight: bool = False
    hold_option_value: float = 0.0
    captain_mode: str = "expected"
    phase_banking: bool = False
    early_price_weight: float = 0.0
    joint_squad_optimiser: bool = False
    squad_captain_weight: float = 0.70
    squad_bench_weight: float = 0.05
    initial_spend_gap: int | None = None
    bench_premium_limit: int | None = None
    bench_premium_penalty: float = 0.0
    exact_initial_optimiser: bool = False
    transfer_bench_premium_penalty: float = 0.0
    decision_immediate_share: float | None = None
    decision_uncertainty_penalty: float = 0.0
    bench_reliability_weight: float = 0.0
    expand_transfer_frontier: bool = False
    transfer_candidate_limit: int = 10
    transfer_beam_width: int = 10
    align_captain_objective: bool = False
    package_route_search: bool = False
    package_deferred_routes: bool = True
    package_route_discount: float = 0.55
    package_liquidity_states: int = 4
    package_setup_loss_limit: float = 3.0
    package_setup_hurdle: float = 1.5
    package_future_hurdle_scale: float = 0.50
    package_target_limit: int = 6
    squad_risk_aversion: float = 0.0
    defence_residual_correlation: float = 0.28
    staleness_gap_trigger: float | None = None
    staleness_hurdle_reduction: float = 0.0
    staleness_hold_reduction: float = 0.0
    additional_move_hurdle: float = 1.15
    enforce_fieldability: bool = False
    fieldability_penalty: float = 0.0
    enforce_weekly_xi_floor: bool = False
    consistent_transfer_objective: bool = False
    # Share of a predicted gain that is believed. The beam picks the maximum over
    # many candidate bundles, and the maximum of noisy estimates is biased upward:
    # measured over 350 realised transfers, a predicted six-Gameweek gain of 11.52
    # delivered 4.31 — a realisation ratio of 0.374, regression slope 0.433.
    #
    # For the plain free-transfer decision this knob is *algebraically redundant*
    # with `transfer_hurdle`, because believing a share of the gain and comparing
    # against a bar is the same test as comparing the whole gain against a
    # proportionally larger bar. A sweep confirms it: (1.00, 5.00), (0.60, 3.00)
    # and (0.35, 1.50) all score identically. So the existing hurdle already *is*
    # the curse correction, fitted rather than derived — which is why it could
    # never be justified from first principles.
    #
    # It stops being redundant as soon as a fixed cost sits alongside the gain,
    # because that cost does not scale with it: a paid hit, the package route
    # discount, or a learned package adjustment. Keep it at 1.0 unless one of
    # those is active.
    gain_realisation: float = 1.0


EXPERT_STRATEGY = SimulationStrategy(
    "Patient six-GW transfers + safe captain", 8.00, 5, False, True, 0, 99.0
)
WEEKLY_CHASE_STRATEGY = SimulationStrategy(
    name="Six-GW planner + adaptive banking",
    transfer_hurdle=5.00,
    bank_limit=5,
    force_weekly_review=False,
    safe_captain=False,
    # Hits stay off. In isolation a single paid hit looked strongly positive —
    # +82.5 on the training seasons and +28.1 across ten with the strategy held
    # fixed — but through the full pipeline it cost 20.8 points a season,
    # because allowing it changed the incumbent's profile enough that the
    # decision gate stopped switching policy in 2024/25 and gave back the 203
    # points that switch was worth. A local gain that disturbs a larger effect
    # is not a gain.
    max_hits=0,
    hit_immediate_hurdle=99.0,
    initial_spend_gap=5,
    bench_premium_limit=20,
    bench_premium_penalty=0.018,
    exact_initial_optimiser=True,
    enforce_fieldability=True,
    fieldability_penalty=4.0,
    enforce_weekly_xi_floor=False,
    consistent_transfer_objective=False,
)
JOINT_OPTION_STRATEGY = SimulationStrategy(
    name="Joint transfer-chip tree + hold value",
    transfer_hurdle=5.35,
    bank_limit=5,
    force_weekly_review=False,
    safe_captain=False,
    max_hits=0,
    hit_immediate_hurdle=99.0,
    joint_chip_preflight=True,
    hold_option_value=0.85,
    captain_mode="expected",
    phase_banking=False,
    early_price_weight=0.0,
    joint_squad_optimiser=True,
    initial_spend_gap=5,
    bench_premium_limit=20,
    bench_premium_penalty=0.018,
    exact_initial_optimiser=True,
    enforce_fieldability=True,
    fieldability_penalty=4.0,
    enforce_weekly_xi_floor=False,
    consistent_transfer_objective=False,
)
ELITE_TARGET_STRATEGY = SimulationStrategy(
    name="Elite target: batched patience + attacker captain",
    transfer_hurdle=5.15,
    bank_limit=5,
    force_weekly_review=False,
    safe_captain=False,
    max_hits=0,
    hit_immediate_hurdle=99.0,
    joint_chip_preflight=True,
    hold_option_value=0.35,
    captain_mode="attacking_tail",
    phase_banking=True,
    early_price_weight=1.10,
    joint_squad_optimiser=True,
)


@dataclass(frozen=True)
class ChipPolicy:
    wildcard_gap: float
    free_hit_gap: float
    bench_score: float
    triple_score: float
    afcon_bonus: float
    first_wildcard_min_gw: int = 5
    second_wildcard_min_gw: int = 24
    enabled_chips: tuple[str, ...] | None = None

    def as_dict(self) -> dict:
        return {
            "wildcardGap": round(self.wildcard_gap, 3),
            "freeHitGap": round(self.free_hit_gap, 3),
            "benchScore": round(self.bench_score, 3),
            "tripleScore": round(self.triple_score, 3),
            "afconBonus": round(self.afcon_bonus, 3),
            "firstWildcardMinGw": int(self.first_wildcard_min_gw),
            "secondWildcardMinGw": int(self.second_wildcard_min_gw),
            "enabledChips": list(self.enabled_chips) if self.enabled_chips else None,
        }


# Frozen after the walk-forward chip ablation.  Automatic Wildcard and Free Hit
# policies lost points out of sample, whereas these conservative TC/BB gates
# added points without a negative evaluation season.  Keeping this named policy
# separate from the exploratory pool prevents accidental promotion of a policy
# selected on the same seasons used to report its score.
AUDITED_CHAMPION_CHIP_POLICY = ChipPolicy(
    wildcard_gap=1_000_000.0,
    free_hit_gap=1_000_000.0,
    bench_score=11.0,
    # Rescaled after the fixture-count repair: the Triple Captain signal is now
    # the captain's Gameweek total rather than that total multiplied by the
    # fixture count a second time.
    triple_score=9.0,
    afcon_bonus=0.0,
    first_wildcard_min_gw=10,
    second_wildcard_min_gw=28,
    enabled_chips=("Bench Boost", "Triple Captain"),
)


# The gate's job is to compare decision strategies, so everything else it holds
# must stay still. Taking its chip policy from the searched pool coupled the two:
# editing the chip search space silently changed the gate's probe conditions, the
# switch decision in 2024/25 flipped, and 200 points moved in a season for reasons
# that had nothing to do with chips. This policy is frozen and deliberately not a
# member of `chip_policy_pool`.
GATE_CHIP_POLICY = ChipPolicy(52.0, 14.0, 12.0, 10.0, 0.55, 10, 20)


def chip_policy_pool() -> list[ChipPolicy]:
    rng = np.random.default_rng(20260812)
    policies = [
        ChipPolicy(
            # The floor used to be 30, which put the optimum outside the search
            # entirely: only 11 of 20 available Wildcards were ever played, and
            # dropping the bar to 20 is worth +33.5 points on the training
            # seasons and plays all 20. A threshold the search cannot reach is
            # not a threshold the search has rejected.
            wildcard_gap=float(rng.uniform(8.0, 60.0)),
            free_hit_gap=float(rng.uniform(7.0, 30.0)),
            # Both bars are read against Gameweek-total projections now that
            # every ensemble route scales with the fixture count.
            bench_score=float(rng.uniform(10.0, 22.0)),
            triple_score=float(rng.uniform(7.0, 15.0)),
            afcon_bonus=float(rng.uniform(0.20, 0.80)),
            first_wildcard_min_gw=int(rng.choice([4, 6, 8, 10])),
            second_wildcard_min_gw=int(rng.choice([20, 24, 28])),
        )
        for _ in range(CHIP_POLICY_TRIALS - 4)
    ]
    policies.extend(
        [
            ChipPolicy(20.0, 14.0, 19.0, 10.0, 0.55, 10, 20),
            ChipPolicy(12.0, 20.0, 15.0, 12.0, 0.55, 8, 24),
            ChipPolicy(35.0, 10.0, 17.0, 8.5, 0.30, 6, 20),
            ChipPolicy(50.0, 26.0, 21.0, 14.0, 0.70, 10, 28),
        ]
    )
    return policies


def chip_windows(
    season: str,
    first_gw: int,
    last_gw: int,
    first_wildcard_min_gw: int = 5,
    second_wildcard_min_gw: int = 24,
) -> list[dict]:
    first_wildcard_start = max(first_gw, first_wildcard_min_gw)
    second_wildcard_start = max(20, first_gw, second_wildcard_min_gw)
    afcon_window = AFCON_WINDOWS.get(season)
    if afcon_window and 20 <= afcon_window[0] <= 24:
        # A mass mid-season absence can justify opening the second-WC decision
        # earlier, but the signal threshold still has to be cleared.
        second_wildcard_start = max(20, first_gw, afcon_window[0])
    windows = [
        {"chip": "Wildcard", "start": first_wildcard_start, "end": min(19, last_gw)},
        {"chip": "Wildcard", "start": second_wildcard_start, "end": last_gw},
    ]
    if season == "2025-26":
        for chip in ("Free Hit", "Bench Boost", "Triple Captain"):
            windows.extend(
                [
                    {"chip": chip, "start": first_gw, "end": min(19, last_gw)},
                    {"chip": chip, "start": max(20, first_gw), "end": last_gw},
                ]
            )
    else:
        # Free Hit replaced All Out Attack in 2017/18. We omit the obsolete AOA
        # chip from 2016/17 rather than granting an anachronistic Free Hit.
        single_use_chips = (
            ("Bench Boost", "Triple Captain")
            if season == "2016-17"
            else ("Free Hit", "Bench Boost", "Triple Captain")
        )
        for chip in single_use_chips:
            windows.append({"chip": chip, "start": first_gw, "end": last_gw})
        if season == "2024-25" and last_gw >= 24:
            windows.append(
                {
                    "chip": "Assistant Manager",
                    "start": max(24, first_gw),
                    "end": max(24, last_gw - 2),
                }
            )
    return [window for window in windows if window["start"] <= window["end"]]


def initial_squad(
    frame: pd.DataFrame,
    scores: np.ndarray,
    budget_limit: int = 1000,
    excluded_elements: set[int] | None = None,
    captain_weight: float = 0.70,
    bench_weight: float = 0.05,
    minimum_spend_gap: int | None = None,
    bench_premium_limit: int | None = None,
    bench_premium_penalty: float = 0.0,
    exact_optimiser: bool = False,
    lineup_scores: np.ndarray | None = None,
    captain_utility_scores: np.ndarray | None = None,
    bench_utility_scores: np.ndarray | None = None,
    risk_scores: np.ndarray | None = None,
    risk_aversion: float = 0.0,
    defence_correlation: float = 0.28,
) -> list[int]:
    """Build a legal squad while requiring every selected XI player to be active."""
    # Every historical construction path, including diagnostic ablations, uses
    # the same exact MILP. A strategy flag may change the objective/constraints;
    # it must never reactivate the legacy greedy builder.
    exact_requested = True
    excluded_elements = excluded_elements or set()
    if excluded_elements and not exact_requested:
        frame = frame[~frame["element"].isin(excluded_elements)]
    frame_indices = frame.index.to_numpy(int)
    prices = frame["price"].to_numpy(int)
    player_positions = frame["position_id"].to_numpy(int)
    player_clubs = frame["team_id"].to_numpy(int)
    if exact_requested:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix

        count = len(frame)
        local_scores = scores[frame_indices].astype(float)
        local_bench_scores = (
            bench_utility_scores[frame_indices].astype(float)
            if bench_utility_scores is not None
            else local_scores
        )
        excluded_local = frame["element"].isin(excluded_elements).to_numpy(bool)
        # A blanking or known-absent player can occupy a non-scoring bench slot
        # in a rebuild, but cannot enter the XI. A large bench penalty prevents
        # the solver from treating that feasibility allowance as useful depth.
        local_bench_scores = local_bench_scores.copy()
        local_bench_scores[excluded_local] -= 25.0
        local_lineup_scores = (
            lineup_scores[frame_indices].astype(float)
            if lineup_scores is not None
            else local_scores
        )
        local_captain_scores = (
            captain_utility_scores[frame_indices].astype(float)
            if captain_utility_scores is not None
            else local_lineup_scores
        )
        local_risk_scores = (
            risk_scores[frame_indices].astype(float)
            if risk_scores is not None
            else np.zeros(count, dtype=float)
        )
        position_floors = {
            position: int(prices[player_positions == position].min())
            for position in SQUAD_QUOTAS
        }
        premiums = np.asarray(
            [
                max(0, int(price) - position_floors[int(position)])
                for price, position in zip(prices, player_positions)
            ],
            dtype=float,
        )
        local_play_probability = (
            frame["play_probability"].fillna(0).to_numpy(float)
            if "play_probability" in frame
            else np.ones(count, dtype=float)
        )
        # Historical GW1 has no team-sheet observations, so an absolute 78%
        # threshold would reject every player despite the position priors being
        # the only causal evidence available. Ramp the hard play floor over the
        # first five deadlines; the live optimiser always uses the fixed 84%
        # play / 70% start standard defined in pick_squad.
        current_gw = int(frame["GW"].iloc[0]) if "GW" in frame else 6
        historical_play_floor = min(0.78, 0.68 + 0.025 * max(0, current_gw - 1))
        standard_xi = local_play_probability >= historical_play_floor
        exception_reference_scores = (
            frame["component_xpts"].fillna(0).to_numpy(float)
            if "component_xpts" in frame
            else local_lineup_scores
        )
        active_exception_scores = exception_reference_scores[~excluded_local]
        exception_threshold = (
            float(np.quantile(active_exception_scores, 0.95))
            if len(active_exception_scores)
            else math.inf
        )
        exceptional_xi = (
            ~standard_xi
            & (local_play_probability >= historical_play_floor - 0.10)
            & (exception_reference_scores >= exception_threshold)
        )
        allowed_xi = (standard_xi | exceptional_xi) & ~excluded_local
        objective = -np.concatenate(
            [
                bench_weight * local_bench_scores
                - bench_premium_penalty * premiums,
                local_lineup_scores
                - risk_aversion * local_risk_scores
                - bench_weight * local_bench_scores
                + bench_premium_penalty * premiums,
                captain_weight * local_captain_scores,
            ]
        )
        # Each variable block has a fixed selected count (15 squad, 11 XI,
        # one captain). Subtracting a constant within a block leaves the exact
        # optimum unchanged while avoiding a HiGHS numerical pathology on
        # mixed-sign objectives in large blank-gameweek matrices.
        for block_start, block_end in (
            (0, count),
            (count, 2 * count),
            (2 * count, 3 * count),
        ):
            objective[block_start:block_end] -= float(
                objective[block_start:block_end].min()
            )
        objective_scale = max(1.0, float(np.max(np.abs(objective))))
        objective /= objective_scale
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        rows.append(np.concatenate([prices, np.zeros(2 * count)]))
        exclusion_share = float(excluded_local.mean()) if count else 0.0
        effective_spend_gap = minimum_spend_gap
        lower.append(
            max(0, min(budget_limit, 1000) - effective_spend_gap)
            if effective_spend_gap is not None
            else 0
        )
        upper.append(budget_limit)
        for position, quota in SQUAD_QUOTAS.items():
            membership = (player_positions == position).astype(float)
            rows.append(np.concatenate([membership, np.zeros(2 * count)]))
            lower.append(quota)
            upper.append(quota)
        for club in np.unique(player_clubs):
            membership = (player_clubs == club).astype(float)
            rows.append(np.concatenate([membership, np.zeros(2 * count)]))
            lower.append(0)
            upper.append(3)
        for local_index in range(count):
            xi_link = np.zeros(3 * count)
            xi_link[count + local_index] = 1
            xi_link[local_index] = -1
            rows.append(xi_link)
            lower.append(-np.inf)
            upper.append(0)
            captain_link = np.zeros(3 * count)
            captain_link[2 * count + local_index] = 1
            captain_link[count + local_index] = -1
            rows.append(captain_link)
            lower.append(-np.inf)
            upper.append(0)
        xi_total = np.zeros(3 * count)
        xi_total[count : 2 * count] = 1
        rows.append(xi_total)
        lower.append(11)
        upper.append(11)
        for position, minimum, maximum in (
            (1, 1, 1),
            (2, 3, 5),
            (3, 2, 5),
            (4, 1, 3),
        ):
            lineup = np.zeros(3 * count)
            lineup[count : 2 * count] = (player_positions == position).astype(float)
            rows.append(lineup)
            lower.append(minimum)
            upper.append(maximum)
        captain_total = np.zeros(3 * count)
        captain_total[2 * count :] = 1
        rows.append(captain_total)
        lower.append(1)
        upper.append(1)
        exception_count = np.zeros(3 * count)
        exception_count[count : 2 * count] = exceptional_xi.astype(float)
        rows.append(exception_count)
        lower.append(0)
        upper.append(1)
        if bench_premium_limit is not None:
            premium_total = np.concatenate([premiums, -premiums, np.zeros(count)])
            rows.append(premium_total)
            lower.append(0)
            upper.append(bench_premium_limit)
        variable_upper = np.concatenate(
            [
                np.ones(count),
                allowed_xi.astype(float),
                allowed_xi.astype(float),
            ]
        )
        # This matrix has thousands of columns but only a handful of non-zero
        # coefficients per row. Reusing one CSR representation avoids repeated
        # dense 20-50 MB allocations across the feasibility, spend-frontier and
        # objective solves.
        constraint_matrix = csr_matrix(np.vstack(rows))
        rows.clear()
        if effective_spend_gap is not None and exclusion_share >= 0.20:
            # A persistent rebuild still wants to spend its available value, but
            # a reduced slate plus XI/bench constraints can make the normal floor
            # mathematically impossible. Find the exact maximum feasible spend
            # under every other rule, then relax only to that proven frontier.
            probe_lower = np.asarray(lower, dtype=float).copy()
            probe_lower[0] = 0
            spend_probe = milp(
                c=-np.concatenate([prices, np.zeros(2 * count)]),
                integrality=np.ones(3 * count),
                bounds=Bounds(np.zeros(3 * count), variable_upper),
                constraints=LinearConstraint(
                    constraint_matrix, probe_lower, np.asarray(upper)
                ),
                options={"time_limit": 20.0},
            )
            if spend_probe.success and spend_probe.x is not None:
                maximum_feasible_spend = int(
                    round(float(np.dot(prices, spend_probe.x[:count])))
                )
                lower[0] = min(lower[0], maximum_feasible_spend)
        result = milp(
            c=objective,
            integrality=np.ones(3 * count),
            bounds=Bounds(
                np.zeros(3 * count),
                variable_upper,
            ),
            constraints=LinearConstraint(
                constraint_matrix, np.asarray(lower), np.asarray(upper)
            ),
            options={"time_limit": 20.0},
        )
        if (
            (not result.success or result.x is None)
            and "infeasible" in str(result.message).lower()
            and effective_spend_gap is not None
        ):
            # Availability/XI and cheap-bench constraints can lower the exact
            # maximum legal spend even without a blank-heavy slate. Prove that
            # frontier with a separate MILP, then relax only the impossible
            # lower-bound amount; never fall back to a greedy squad.
            probe_lower = np.asarray(lower, dtype=float).copy()
            probe_lower[0] = 0
            spend_probe = milp(
                c=-np.concatenate([prices, np.zeros(2 * count)]),
                integrality=np.ones(3 * count),
                bounds=Bounds(np.zeros(3 * count), variable_upper),
                constraints=LinearConstraint(
                    constraint_matrix, probe_lower, np.asarray(upper)
                ),
                options={"time_limit": 20.0},
            )
            if spend_probe.success and spend_probe.x is not None:
                lower[0] = min(
                    lower[0],
                    int(round(float(np.dot(prices, spend_probe.x[:count])))),
                )
                result = milp(
                    c=objective,
                    integrality=np.ones(3 * count),
                    bounds=Bounds(np.zeros(3 * count), variable_upper),
                    constraints=LinearConstraint(
                        constraint_matrix, np.asarray(lower), np.asarray(upper)
                    ),
                    options={"time_limit": 20.0},
                )
        if not result.success and "infeasible" in str(result.message).lower():
            # HiGHS presolve can occasionally report the identical historical
            # constraint matrix infeasible for one finite objective and optimal
            # for another. Feasibility cannot depend on the objective, so audit
            # the claim with an independent no-presolve exact solve before
            # stopping. There is deliberately no greedy fallback.
            result = milp(
                c=objective,
                integrality=np.ones(3 * count),
                bounds=Bounds(
                    np.zeros(3 * count),
                    variable_upper,
                ),
                constraints=LinearConstraint(
                    constraint_matrix, np.asarray(lower), np.asarray(upper)
                ),
                options={"time_limit": 60.0, "presolve": False},
            )
        if not result.success or result.x is None:
            season_label = str(frame["season"].iloc[0]) if "season" in frame else "live"
            gw_label = int(frame["GW"].iloc[0]) if "GW" in frame else -1
            raise RuntimeError(
                "Bench-efficient historical squad optimisation failed: "
                f"season={season_label}, GW={gw_label}, budget={budget_limit}, "
                f"pool={count}, excluded={len(excluded_elements or set())}, "
                f"play_floor={historical_play_floor:.3f}; {result.message}"
            )
        return frame_indices[np.flatnonzero(result.x[:count] > 0.5)].astype(int).tolist()

def precompute_fresh_squads(
    data: pd.DataFrame, scores: np.ndarray, one_week_only: bool = False
) -> dict[tuple[str, int], list[int]]:
    fresh: dict[tuple[str, int], list[int]] = {}
    for (season, gw), frame in data.groupby(["season", "GW"], sort=False):
        afcon_window = AFCON_WINDOWS.get(str(season))
        afcon_risk = bool(
            afcon_window and afcon_window[0] - 1 <= int(gw) <= afcon_window[1]
        )
        excluded = set(
            frame.loc[
                afcon_risk & frame["nationality"].isin(AFCON_NATIONS), "element"
            ].astype(int)
        )
        if one_week_only:
            excluded.update(
                frame.loc[frame["fixture_count"].le(0), "element"].astype(int)
            )
        fresh[(str(season), int(gw))] = initial_squad(
            frame,
            scores,
            excluded_elements=excluded,
            # Unspent cash has no afterlife on a Free Hit. Forcing the normal
            # near-£100m spend rule in a four-fixture blank can make an otherwise
            # legal one-week squad infeasible or buy useless bench price.
            minimum_spend_gap=None if one_week_only else 5,
            bench_premium_limit=20,
            bench_premium_penalty=0.018,
            exact_optimiser=True,
        )
    return fresh


def choose_xi(
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    scores: np.ndarray,
    excluded_elements: set[int] | None = None,
) -> tuple[list[int], list[int]]:
    excluded_elements = excluded_elements or set()

    def selection_score(element: int) -> float:
        if element in excluded_elements:
            return -1.5
        return scores[row_by_element[element]] if element in row_by_element else -1.0

    # Sort each positional pool once. The prior implementation repeated these
    # same sorts for every formation and dominated the recursive search runtime.
    # Prefix sums then score every legal formation in constant time.
    pools: dict[int, list[int]] = {position: [] for position in SQUAD_QUOTAS}
    for element, state in squad.items():
        pools[int(state["position"])].append(element)
    prefix: dict[int, np.ndarray] = {}
    for position, pool in pools.items():
        pool.sort(key=selection_score, reverse=True)
        prefix[position] = np.concatenate(
            ([0.0], np.cumsum([selection_score(element) for element in pool]))
        )

    best_formation: dict[int, int] | None = None
    best_score = -math.inf
    for defenders in (3, 4, 5):
        for forwards in (1, 2, 3):
            midfielders = 10 - defenders - forwards
            if not 2 <= midfielders <= 5:
                continue
            formation = {1: 1, 2: defenders, 3: midfielders, 4: forwards}
            if any(len(pools[position]) < count for position, count in formation.items()):
                continue
            total = sum(prefix[position][count] for position, count in formation.items())
            if total > best_score:
                best_formation = formation
                best_score = total
    best_xi = (
        [
            element
            for position, count in best_formation.items()
            for element in pools[position][:count]
        ]
        if best_formation is not None
        else []
    )
    xi_set = set(best_xi)
    bench = [element for element in squad if element not in xi_set]
    # FPL puts the reserve goalkeeper in a separate slot; outfield order is by score.
    bench_gk = [element for element in bench if int(squad[element]["position"]) == 1]
    bench_outfield = [
        element for element in bench if int(squad[element]["position"]) != 1
    ]
    bench_outfield.sort(
        key=selection_score,
        reverse=True,
    )
    return best_xi, bench_gk + bench_outfield


def squad_decision_utility(
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    scores: np.ndarray,
    excluded_elements: set[int] | None = None,
    captain_weight: float = 0.70,
    bench_weight: float = 0.05,
    bench_premium: float = 0.0,
    bench_premium_penalty: float = 0.0,
    bench_scores: np.ndarray | None = None,
    captain_scores: np.ndarray | None = None,
    risk_scores: np.ndarray | None = None,
    risk_aversion: float = 0.0,
    defence_correlation: float = 0.28,
) -> float:
    """Value the legal XI, captain route and bench insurance—not 15 equal slots."""
    xi, bench = choose_xi(
        squad, row_by_element, scores, excluded_elements=excluded_elements
    )
    if len(xi) != 11:
        return -math.inf
    excluded = excluded_elements or set()

    def value(element: int) -> float:
        if element in excluded or element not in row_by_element:
            return -1.0
        return float(scores[row_by_element[element]])

    xi_values = [value(element) for element in xi]
    def alternate_value(element: int, values: np.ndarray | None) -> float:
        if values is None:
            return value(element)
        if element in excluded or element not in row_by_element:
            return -1.0
        return float(values[row_by_element[element]])

    bench_values = [
        max(0.0, alternate_value(element, bench_scores)) for element in bench
    ]
    captain_values = [
        alternate_value(element, captain_scores) for element in xi
    ]
    risk_penalty = 0.0
    if risk_scores is not None and risk_aversion > 0:
        xi_risk = {
            element: max(
                0.0,
                float(risk_scores[row_by_element[element]]),
            )
            for element in xi
            if element in row_by_element and element not in excluded
        }
        variance = sum(value_**2 for value_ in xi_risk.values())
        for left_index, left in enumerate(xi):
            if left not in xi_risk or int(squad[left]["position"]) > 2:
                continue
            for right in xi[left_index + 1 :]:
                if (
                    right in xi_risk
                    and int(squad[right]["position"]) <= 2
                    and int(squad[left]["team"]) == int(squad[right]["team"])
                ):
                    variance += (
                        2
                        * defence_correlation
                        * xi_risk[left]
                        * xi_risk[right]
                    )
        if captain_values:
            captain_index = int(np.argmax(captain_values))
            captain_element = xi[captain_index]
            captain_risk = xi_risk.get(captain_element, 0.0)
            variance += (
                (1 + captain_weight) ** 2 - 1
            ) * captain_risk**2
            if int(squad[captain_element]["position"]) <= 2:
                for peer in xi:
                    if (
                        peer != captain_element
                        and peer in xi_risk
                        and int(squad[peer]["position"]) <= 2
                        and int(squad[peer]["team"])
                        == int(squad[captain_element]["team"])
                    ):
                        variance += (
                            2
                            * captain_weight
                            * defence_correlation
                            * captain_risk
                            * xi_risk[peer]
                        )
        risk_penalty = risk_aversion * math.sqrt(max(0.0, variance))
    return float(
        sum(xi_values)
        + captain_weight * max(captain_values, default=0.0)
        + bench_weight * sum(bench_values)
        - bench_premium_penalty * bench_premium
        - risk_penalty
    )


def legal_xi(elements: list[int], squad: dict[int, dict]) -> bool:
    counts = {
        position: sum(int(squad[element]["position"]) == position for element in elements)
        for position in SQUAD_QUOTAS
    }
    return (
        len(elements) == 11
        and counts[1] == 1
        and counts[2] >= 3
        and counts[3] >= 2
        and counts[4] >= 1
    )


def realised_week_breakdown(
    xi: list[int],
    bench: list[int],
    captain: int,
    vice: int,
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    actual: np.ndarray,
    minutes: np.ndarray,
) -> dict[str, float]:
    def played(element: int) -> bool:
        return element in row_by_element and minutes[row_by_element[element]] > 0

    final_xi = list(xi)
    absent = [element for element in final_xi if not played(element)]
    for substitute in bench:
        if not absent or not played(substitute):
            continue
        for missing in list(absent):
            trial = [substitute if element == missing else element for element in final_xi]
            if legal_xi(trial, squad):
                final_xi = trial
                absent.remove(missing)
                break
    xi_points = sum(
        actual[row_by_element[element]]
        for element in final_xi
        if played(element)
    )
    captain_bonus = 0.0
    if played(captain):
        captain_bonus = float(actual[row_by_element[captain]])
    elif played(vice):
        captain_bonus = float(actual[row_by_element[vice]])
    normal = float(xi_points + captain_bonus)
    all_squad_points = sum(
        actual[row_by_element[element]]
        for element in squad
        if played(element)
    )
    return {
        "normal": normal,
        "bench_boost": float(all_squad_points + captain_bonus),
        "triple_captain": float(normal + captain_bonus),
        "captain_bonus": captain_bonus,
    }


def realised_week_points(
    xi: list[int],
    bench: list[int],
    captain: int,
    vice: int,
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    actual: np.ndarray,
    minutes: np.ndarray,
) -> float:
    return realised_week_breakdown(
        xi, bench, captain, vice, squad, row_by_element, actual, minutes
    )["normal"]


def triple_captain_signal(expected_points: float, fixture_count: int) -> float:
    """Expected-point signal used by the historically calibrated TC threshold.

    Captain ranking scores can be percentiles or listwise utilities; they must
    never enter a points threshold directly.

    ``expected_points`` is already a Gameweek total, so it carries the Double
    Gameweek itself. ``fixture_count`` is retained only to keep a blank from
    signalling a Triple Captain.
    """
    return float(expected_points) * float(min(1, max(0, int(fixture_count))))


def selling_price(purchase_price: int, current_price: int) -> int:
    if current_price <= purchase_price:
        return current_price
    return purchase_price + (current_price - purchase_price) // 2


def assert_legal_squad(
    squad: dict[int, dict],
    bank: int,
    season: str,
    gw: int,
    stage: str,
    allow_temporary_club_overload: bool = False,
) -> None:
    """Fail fast when a replay transition creates an impossible FPL state."""
    if len(squad) != 15:
        raise AssertionError(
            f"{season} GW{gw} {stage}: expected 15 players, found {len(squad)}"
        )
    position_counts = {
        position: sum(
            int(state["position"]) == position for state in squad.values()
        )
        for position in SQUAD_QUOTAS
    }
    if position_counts != SQUAD_QUOTAS:
        raise AssertionError(
            f"{season} GW{gw} {stage}: illegal position quotas {position_counts}"
        )
    club_counts: dict[int, int] = {}
    for state in squad.values():
        club = int(state["team"])
        club_counts[club] = club_counts.get(club, 0) + 1
    if (
        max(club_counts.values(), default=0) > 3
        and not allow_temporary_club_overload
    ):
        raise AssertionError(
            f"{season} GW{gw} {stage}: more than three players from one club"
        )
    if bank < 0:
        raise AssertionError(f"{season} GW{gw} {stage}: negative bank {bank}")


def simulation_context(data: pd.DataFrame) -> dict:
    """Cache immutable week indices and schedules shared by hundreds of replays."""
    # Object ids can be reused after a short-lived season slice is collected,
    # which can silently return another frame's indices. A content fingerprint
    # is stable across equivalent slices and safe across object lifetimes.
    # DataFrame attrs propagate to slices, so an attrs-based token is unsafe.
    # Computing this small three-column fingerprint prevents parent/slice cache
    # collisions and costs far less than rebuilding the schedule context.
    fingerprint = int(
        pd.util.hash_pandas_object(
            data[["season", "GW", "element"]], index=True
        ).sum()
    )
    cache_key = (len(data), fingerprint)
    cached = _SIMULATION_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    seasons = []
    for season, season_frame in data.groupby("season", sort=False):
        week_indices = {
            int(gw): np.asarray(indices, dtype=int)
            for gw, indices in season_frame.groupby("GW", sort=True).groups.items()
        }
        season_teams = set(season_frame["team_id"].astype(int).unique())
        schedule_counts = {}
        for gw, indices in week_indices.items():
            schedule_frame = data.loc[indices]
            observed = (
                schedule_frame.groupby("team_id")["fixture_count"]
                .max()
                .astype(int)
                .to_dict()
            )
            schedule_counts[gw] = {
                team_id: int(observed.get(team_id, 0)) for team_id in season_teams
            }
        seasons.append(
            {
                "season": str(season),
                "weeks": sorted(week_indices),
                "weekIndices": week_indices,
                "teams": season_teams,
                "scheduleCounts": schedule_counts,
            }
        )
    cached = {"rows": len(data), "seasons": seasons}
    _SIMULATION_CONTEXT_CACHE[cache_key] = cached
    return cached


def joint_transfer_plan(
    squad: dict[int, dict],
    bank: int,
    free_transfers: int,
    row_by_element: dict[int, int],
    incoming_by_position: dict[int, np.ndarray],
    element_values: np.ndarray,
    position_values: np.ndarray,
    team_values: np.ndarray,
    price_values: np.ndarray,
    plan_scores: np.ndarray,
    bench_scores: np.ndarray | None,
    captain_utility_scores: np.ndarray | None,
    price_rise_values: np.ndarray,
    price_fall_values: np.ndarray,
    uncertainty_values: np.ndarray,
    risk_scores: np.ndarray | None,
    excluded_elements: set[int],
    team_option_score: dict[int, float],
    strategy: SimulationStrategy,
    gw: int,
    club_limits: dict[int, int] | None = None,
    staleness_gap: float = 0.0,
    package_action_adjustment: Callable[[dict], float] | None = None,
) -> tuple[dict[int, dict], int, int, float]:
    """Beam-search a legal multi-transfer bundle, including funding moves.

    Unlike greedy same-position swaps, an intermediate downgrade may survive
    the beam and fund an upgrade elsewhere. Only the final bundle is judged
    against the value of banking the free transfers.
    """
    stale = bool(
        strategy.staleness_gap_trigger is not None
        and staleness_gap >= strategy.staleness_gap_trigger
    )
    # Paid moves are searched alongside free ones, and every move beyond the
    # banked free transfers is charged its full -4 inside the objective, so the
    # beam only keeps a hit that pays for itself. Before this the branch capped
    # moves at the free-transfer count, which made `max_hits` silently inert:
    # setting it to 3 changed the season by 0.0 points and produced 0.0 hits,
    # because a paid move was not merely discouraged, it was unreachable.
    paid_allowance = max(0, int(strategy.max_hits))
    max_moves = max(0, min(int(free_transfers) + paid_allowance, 5))
    if max_moves == 0:
        return squad, bank, 0, 0, 0.0

    def hit_cost(moves: int) -> float:
        return PAID_MOVE_UTILITY_COST * max(0, moves - int(free_transfers))

    def option_utility(element: int, state: dict) -> float:
        index = row_by_element.get(element)
        value = 0.0
        if strategy.joint_chip_preflight:
            value += 0.18 * team_option_score.get(int(state["team"]), 0.0)
        if strategy.early_price_weight > 0 and gw <= 19 and index is not None:
            value += strategy.early_price_weight * (
                float(price_rise_values[index]) - float(price_fall_values[index])
            )
        return value

    utility_cache: dict[tuple[int, ...], float] = {}
    route_cache: dict[tuple[tuple[int, ...], int], float] = {}

    position_floors = {
        position: min(
            int(price_values[index])
            for index in row_by_element.values()
            if int(position_values[index]) == position
        )
        for position in SQUAD_QUOTAS
    }

    def active_bench_premium(active_squad: dict[int, dict]) -> int:
        _, bench = choose_xi(
            active_squad,
            row_by_element,
            plan_scores,
            excluded_elements=excluded_elements,
        )
        premium = 0
        for element in bench:
            state = active_squad[element]
            index = row_by_element.get(element)
            current_price = (
                int(price_values[index]) if index is not None else int(state["last_price"])
            )
            premium += max(0, current_price - position_floors[int(state["position"])])
        return premium

    def squad_utility(active_squad: dict[int, dict]) -> float:
        signature = tuple(sorted(active_squad))
        cached_utility = utility_cache.get(signature)
        if cached_utility is not None:
            return cached_utility
        decision_value = squad_decision_utility(
            active_squad,
            row_by_element,
            plan_scores,
            excluded_elements=excluded_elements,
            captain_weight=strategy.squad_captain_weight,
            bench_weight=strategy.squad_bench_weight,
            bench_premium=active_bench_premium(active_squad),
            bench_premium_penalty=strategy.transfer_bench_premium_penalty,
            bench_scores=bench_scores,
            captain_scores=captain_utility_scores,
            risk_scores=risk_scores,
            risk_aversion=strategy.squad_risk_aversion,
            defence_correlation=strategy.defence_residual_correlation,
        )
        value = decision_value + sum(
            option_utility(element, state)
            for element, state in active_squad.items()
        )
        utility_cache[signature] = value
        return value

    def next_transfer_option(active_squad: dict[int, dict], active_bank: int) -> float:
        """Best legal one-transfer gain visible from a possible setup state.

        This uses only the current deadline's censored multi-week score.  It does
        not peek at next week's results or final fixture schedule.  Its purpose is
        to keep a funding move alive when it unlocks an otherwise unaffordable
        premium on the following free transfer.
        """
        cache_key = (tuple(sorted(active_squad)), int(active_bank))
        cached = route_cache.get(cache_key)
        if cached is not None:
            return cached
        active_utility = squad_utility(active_squad)
        team_counts: dict[int, int] = {}
        for player_state in active_squad.values():
            team_id = int(player_state["team"])
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        best_gain = 0.0
        for outgoing, outgoing_state in active_squad.items():
            outgoing_index = row_by_element.get(outgoing)
            current_price = (
                int(price_values[outgoing_index])
                if outgoing_index is not None
                else int(outgoing_state["last_price"])
            )
            sale = selling_price(int(outgoing_state["purchase"]), current_price)
            position = int(outgoing_state["position"])
            for incoming_index_raw in incoming_by_position[position][
                : strategy.package_target_limit
            ]:
                incoming_index = int(incoming_index_raw)
                incoming = int(element_values[incoming_index])
                if incoming in active_squad or incoming in excluded_elements:
                    continue
                incoming_price = int(price_values[incoming_index])
                if incoming_price > active_bank + sale:
                    continue
                incoming_team = int(team_values[incoming_index])
                outgoing_team = int(outgoing_state["team"])
                incoming_limit = (club_limits or {}).get(incoming_team, 3)
                if (
                    incoming_team != outgoing_team
                    and team_counts.get(incoming_team, 0) >= incoming_limit
                ):
                    continue
                candidate_squad = {
                    key: value.copy() for key, value in active_squad.items()
                }
                del candidate_squad[outgoing]
                candidate_squad[incoming] = {
                    "position": int(position_values[incoming_index]),
                    "team": incoming_team,
                    "purchase": incoming_price,
                    "last_price": incoming_price,
                    "nationality": "",
                }
                best_gain = max(
                    best_gain,
                    squad_utility(candidate_squad) - active_utility,
                )
        route_cache[cache_key] = float(best_gain)
        return float(best_gain)

    base_utility = squad_utility(squad)

    def fieldable_xi_count(active_squad: dict[int, dict]) -> int:
        if not strategy.enforce_fieldability:
            return 11
        active_xi, _ = choose_xi(
            active_squad,
            row_by_element,
            plan_scores,
            excluded_elements=excluded_elements,
        )
        return sum(
            element in row_by_element and element not in excluded_elements
            for element in active_xi
        )

    base_fieldable_xi = fieldable_xi_count(squad)
    initial_team_counts: dict[int, int] = {}
    for player_state in squad.values():
        team_id = int(player_state["team"])
        initial_team_counts[team_id] = initial_team_counts.get(team_id, 0) + 1
    forced_clubs = {
        team_id for team_id, count in initial_team_counts.items() if count > 3
    }
    # (utility, bank, squad, moves, incoming uncertainty)
    beam: list[tuple[float, int, dict[int, dict], int, float]] = [
        (base_utility, bank, {key: value.copy() for key, value in squad.items()}, 0, 0.0)
    ]
    best = beam[0]
    best_surplus = 0.0
    package_adjustments: dict[tuple[int, ...], float] = {}

    def learned_package_adjustment(
        candidate: tuple[float, int, dict[int, dict], int, float]
    ) -> float:
        """Return a causal package-level correction without mutating state.

        The callback is an analysis hook. Ordinary production simulations pass
        ``None`` and retain bit-for-bit additive utility behaviour.
        """
        if package_action_adjustment is None:
            return 0.0
        signature = tuple(sorted(candidate[2]))
        cached_adjustment = package_adjustments.get(signature)
        if cached_adjustment is not None:
            return cached_adjustment
        outgoing = tuple(sorted(set(squad) - set(candidate[2])))
        incoming = tuple(sorted(set(candidate[2]) - set(squad)))
        adjustment = float(
            package_action_adjustment(
                {
                    "baseSquad": squad,
                    "candidateSquad": candidate[2],
                    "baseBank": int(bank),
                    "candidateBank": int(candidate[1]),
                    "moves": int(candidate[3]),
                    "incomingUncertainty": float(candidate[4]),
                    "predictedGain": float(candidate[0] - base_utility),
                    "outgoingElements": outgoing,
                    "incomingElements": incoming,
                    "rowByElement": row_by_element,
                    "freeTransfers": int(free_transfers),
                    "gw": int(gw),
                }
            )
        )
        package_adjustments[signature] = adjustment if math.isfinite(adjustment) else 0.0
        return package_adjustments[signature]

    for depth in range(1, max_moves + 1):
        expanded: dict[tuple[int, ...], tuple[float, int, dict[int, dict], int, float]] = {}
        for utility, state_bank, state_squad, _, incoming_uncertainty in beam:
            team_counts: dict[int, int] = {}
            for player_state in state_squad.values():
                team_id = int(player_state["team"])
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            for outgoing, outgoing_state in state_squad.items():
                if forced_clubs and int(outgoing_state["team"]) not in forced_clubs:
                    continue
                outgoing_index = row_by_element.get(outgoing)
                current_price = (
                    int(price_values[outgoing_index])
                    if outgoing_index is not None
                    else int(outgoing_state["last_price"])
                )
                sale = selling_price(int(outgoing_state["purchase"]), current_price)
                position = int(outgoing_state["position"])
                for incoming_index_raw in incoming_by_position[position][
                    : strategy.transfer_candidate_limit
                ]:
                    incoming_index = int(incoming_index_raw)
                    incoming = int(element_values[incoming_index])
                    if incoming in state_squad or incoming in excluded_elements:
                        continue
                    incoming_price = int(price_values[incoming_index])
                    if incoming_price > state_bank + sale:
                        continue
                    incoming_team = int(team_values[incoming_index])
                    outgoing_team = int(outgoing_state["team"])
                    incoming_limit = (club_limits or {}).get(incoming_team, 3)
                    if (
                        incoming_team != outgoing_team
                        and team_counts.get(incoming_team, 0) >= incoming_limit
                    ):
                        continue
                    incoming_state = {
                        "position": int(position_values[incoming_index]),
                        "team": incoming_team,
                        "purchase": incoming_price,
                        "last_price": incoming_price,
                        "nationality": "",
                    }
                    new_squad = {
                        key: value.copy() for key, value in state_squad.items()
                    }
                    del new_squad[outgoing]
                    new_squad[incoming] = incoming_state
                    new_bank = state_bank + sale - incoming_price
                    new_utility = squad_utility(new_squad)
                    new_uncertainty = incoming_uncertainty + float(
                        uncertainty_values[incoming_index]
                    )
                    signature = tuple(sorted(new_squad))
                    prior = expanded.get(signature)
                    candidate = (
                        new_utility,
                        new_bank,
                        new_squad,
                        depth,
                        new_uncertainty,
                    )
                    if prior is None or new_utility > prior[0]:
                        expanded[signature] = candidate
        expanded_values = list(expanded.values())
        if package_action_adjustment is not None and expanded_values:
            # Let the learned package model rerank a bounded superset of the
            # ordinary beam.  This gives funding combinations a chance to
            # survive without invoking the callback for every combinatorial
            # expansion.
            action_pool = sorted(
                expanded_values,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )[: max(strategy.transfer_beam_width * 3, strategy.transfer_beam_width)]
            beam = sorted(
                action_pool,
                key=lambda item: (
                    item[0] + learned_package_adjustment(item),
                    item[1],
                ),
                reverse=True,
            )[: strategy.transfer_beam_width]
        elif strategy.package_route_search and expanded_values:
            # Ordinary beam search has a structural blind spot: the temporary
            # downgrade in a premium-access package has lower current utility and
            # is pruned before the funding can be spent.  Preserve a small,
            # explicitly bounded liquidity frontier, then rank that frontier by
            # current utility plus its best legal next-transfer option.
            primary = sorted(
                expanded_values,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )[: strategy.transfer_beam_width]
            liquid = sorted(
                expanded_values,
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[: max(0, strategy.package_liquidity_states * 2)]
            route_pool: dict[tuple[int, ...], tuple[float, int, dict[int, dict], int, float]] = {}
            for candidate in primary + liquid:
                route_pool[tuple(sorted(candidate[2]))] = candidate
            route_ranked = sorted(
                route_pool.values(),
                key=lambda item: (
                    item[0]
                    + strategy.package_route_discount
                    * next_transfer_option(item[2], item[1]),
                    item[1],
                ),
                reverse=True,
            )
            # This is a strict superset of the ordinary beam. Replacing ordinary
            # slots with liquidity slots made the search less exact and confounded
            # the package test; extra route states are allowed to expand the beam
            # by a small bounded amount instead.
            combined = primary + route_ranked[: strategy.package_liquidity_states]
            deduplicated: dict[
                tuple[int, ...], tuple[float, int, dict[int, dict], int, float]
            ] = {}
            for candidate in combined:
                deduplicated[tuple(sorted(candidate[2]))] = candidate
            beam = list(deduplicated.values())[
                : strategy.transfer_beam_width + strategy.package_liquidity_states
            ]
        else:
            beam = sorted(
                expanded_values,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )[: strategy.transfer_beam_width]
        if not beam:
            break
        for candidate in beam:
            candidate_counts: dict[int, int] = {}
            for player_state in candidate[2].values():
                team_id = int(player_state["team"])
                candidate_counts[team_id] = candidate_counts.get(team_id, 0) + 1
            if max(candidate_counts.values(), default=0) > 3:
                continue
            gain = candidate[0] - base_utility
            learned_adjustment = learned_package_adjustment(candidate)
            hurdle = (
                strategy.transfer_hurdle
                + strategy.additional_move_hurdle * (depth - 1)
            )
            if stale:
                hurdle -= strategy.staleness_hurdle_reduction
            if forced_clubs:
                hurdle = -math.inf
            # A deadline-known blank is not an ordinary marginal upgrade. When
            # a free transfer repairs a legal scoring slot, select the best
            # repair without asking it to clear the normal multi-week hurdle.
            # Paid hits remain outside this joint branch and are never hidden.
            if (
                strategy.enforce_fieldability
                and base_fieldable_xi < 11
                and fieldable_xi_count(candidate[2]) > base_fieldable_xi
            ):
                hurdle = -math.inf
            if strategy.phase_banking:
                hurdle += (2.0 if gw <= 19 else 2.8) if free_transfers <= 1 else -0.8
            if strategy.hold_option_value > 0:
                effective_hold = max(
                    0.0,
                    strategy.hold_option_value
                    - (strategy.staleness_hold_reduction if stale else 0.0),
                )
                hurdle += (
                    effective_hold
                    * min(1.5, candidate[4] / max(1, depth) / 3.0)
                    + (effective_hold if free_transfers <= 1 else -0.35 * effective_hold)
                )
            # A hit is charged here rather than netted off afterwards, so the
            # beam compares bundles on what the manager actually banks.
            surplus = (
                strategy.gain_realisation * gain
                + learned_adjustment
                - hurdle
                - hit_cost(depth)
            )
            if (
                strategy.package_route_search
                and strategy.package_deferred_routes
                and free_transfers == 1
                and depth == 1
                and gain >= -strategy.package_setup_loss_limit
            ):
                future_option_delta = (
                    next_transfer_option(candidate[2], candidate[1])
                    - next_transfer_option(squad, bank)
                )
                route_hurdle = (
                    strategy.package_setup_hurdle
                    + strategy.package_route_discount
                    * strategy.package_future_hurdle_scale
                    * strategy.transfer_hurdle
                )
                route_surplus = (
                    strategy.gain_realisation * gain
                    + strategy.package_route_discount * future_option_delta
                    - route_hurdle
                    - hit_cost(depth)
                )
                surplus = max(surplus, route_surplus)
            if surplus > best_surplus:
                best_surplus = surplus
                best = candidate
    if best[3] == 0:
        return squad, bank, 0, 0, 0.0
    paid_moves = max(0, best[3] - int(free_transfers))
    return best[2], best[1], best[3], paid_moves, float(best[0] - base_utility)


def simulate_candidate(
    data: pd.DataFrame,
    scores: np.ndarray,
    strategy: SimulationStrategy,
    chip_policy: ChipPolicy | None = None,
    fresh_squads: dict[tuple[str, int], list[int]] | None = None,
    free_hit_squads: dict[tuple[str, int], list[int]] | None = None,
    plan_scores: np.ndarray | None = None,
    actual_column: str = "points",
    captain_scores: np.ndarray | None = None,
    tracked_player_name: str | None = None,
    audit_selections: bool = False,
    risk_scores: np.ndarray | None = None,
    chip_value_overrides: dict[tuple[str, int, str], float] | None = None,
    package_action_adjustment: Callable[[dict], float] | None = None,
    initial_squads: dict[tuple[str, int], list[int]] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Carry one legal squad through each season and make deadline-only transfers."""
    if plan_scores is None:
        plan_scores = scores
    strategy, chip_policy = rescale_decision_thresholds(
        strategy,
        chip_policy,
        cross_sectional_spread(data, scores) / REFERENCE_IMMEDIATE_SPREAD,
        cross_sectional_spread(data, plan_scores) / REFERENCE_PLAN_SPREAD,
    )
    actual = data[actual_column].to_numpy(float)
    played_minutes = data["minutes"].to_numpy(float)
    element_values = data["element"].to_numpy(int)
    position_values = data["position_id"].to_numpy(int)
    team_values = data["team_id"].to_numpy(int)
    price_values = data["price"].to_numpy(int)
    fixture_counts = data["fixture_count"].to_numpy(int)
    uncertainty_values = data["prediction_uncertainty"].to_numpy(float)
    component_xpts_values = data["component_xpts"].to_numpy(float)
    if risk_scores is None:
        risk_scores = np.zeros(len(data), dtype=float)
    play_probability_values = data["play_probability"].to_numpy(float)
    start_probability_values = data["start_probability"].to_numpy(float)
    sixty_probability_values = data["sixty_probability"].to_numpy(float)
    expected_minutes_values = data["expected_minutes"].to_numpy(float)
    price_rise_values = data["price_rise_probability"].to_numpy(float)
    price_fall_values = data["price_fall_probability"].to_numpy(float)
    return_values = data["return5_probability"].to_numpy(float)
    haul_values = data["haul8_probability"].to_numpy(float)
    goal_values = data["goal_rate"].to_numpy(float)
    assist_values = data["assist_rate"].to_numpy(float)
    nationality_values = data["nationality"].fillna("").to_numpy(str)
    team_name_values = data["team_name"].fillna("").to_numpy(str)
    display_name_values = data["display_name"].fillna("").to_numpy(str)
    assistant_manager_actual_values = data["assistant_manager_points"].to_numpy(
        float
    )
    team_attack_values = data["team_attack_rating"].to_numpy(float)
    team_defence_values = data["team_defence_rating"].to_numpy(float)
    team_form_values = data["team_form_rating"].to_numpy(float)
    team_clean_values = data["team_clean_rating"].to_numpy(float)
    table_position_values = data["table_position_before"].to_numpy(float)
    opponent_team_values = data["opponent_team"].fillna(0).to_numpy(int)
    safe_captain_score = (
        0.42 * data["recent"].to_numpy(float)
        + 0.18 * data["long"].to_numpy(float)
        + 0.14 * data["fixture_now"].to_numpy(float)
        + 0.20 * data["minutes_security"].to_numpy(float)
        + 0.06 * data["crowd"].to_numpy(float)
    )
    context = simulation_context(data)
    totals = np.zeros(len(context["seasons"]), dtype=float)
    season_stats: list[dict] = []
    attacking_captain_score = (
        scores
        + 1.10 * haul_values
        + 0.35 * return_values
        + 0.45 * goal_values
        + 0.18 * assist_values
        + 0.10 * data["minutes_security"].to_numpy(float)
    )
    fresh_captain_utility = (
        scores * (0.55 + 0.45 * captain_scores)
        if captain_scores is not None
        else scores
    )
    if strategy.decision_immediate_share is None:
        decision_scores = plan_scores
        fresh_lineup_scores = scores
        decision_captain_utility = fresh_captain_utility
        bench_utility_scores = plan_scores
    else:
        immediate_share = float(
            np.clip(strategy.decision_immediate_share, 0.0, 1.0)
        )
        decision_scores = (
            (1.0 - immediate_share) * plan_scores
            + immediate_share * scores * 4.5
            - strategy.decision_uncertainty_penalty * uncertainty_values
        )
        fresh_lineup_scores = decision_scores
        # Captaincy is one additional score in the current Gameweek, not a
        # repeated horizon reward. Keep it on its causal one-week point scale.
        decision_captain_utility = fresh_captain_utility
        reliability = 1.0 - strategy.bench_reliability_weight * (
            1.0 - play_probability_values
        )
        bench_utility_scores = plan_scores * np.clip(reliability, 0.0, 1.0)
    if strategy.enforce_fieldability:
        # Availability changes utility rather than fabricating extra expected
        # points. Current blanks are also hard-excluded below. The default is
        # off, preserving the frozen champion exactly as the research control.
        availability_penalty = strategy.fieldability_penalty * (
            1.0 - play_probability_values
        )
        decision_scores = decision_scores - availability_penalty
        fresh_lineup_scores = fresh_lineup_scores - availability_penalty
        bench_utility_scores = bench_utility_scores - 0.35 * availability_penalty
    consistent_decision_objective = strategy.decision_immediate_share is not None
    transfer_gain_scores = (
        decision_scores if strategy.consistent_transfer_objective else plan_scores
    )
    fresh_squad_scores = (
        decision_scores if consistent_decision_objective else plan_scores
    )
    fresh_bench_scores = (
        bench_utility_scores if consistent_decision_objective else None
    )
    fresh_objective_lineup_scores = (
        fresh_lineup_scores
        if strategy.exact_initial_optimiser or consistent_decision_objective
        else None
    )
    fresh_objective_captain_scores = (
        decision_captain_utility
        if strategy.exact_initial_optimiser
        or consistent_decision_objective
        or strategy.align_captain_objective
        else None
    )

    for season_id, season_context in enumerate(context["seasons"]):
        season = str(season_context["season"])
        weeks = season_context["weeks"]
        season_teams = season_context["teams"]
        schedule_counts = season_context["scheduleCounts"]
        squad: dict[int, dict] = {}
        bank = 0
        free_transfers = 1
        transfers = 0
        hits = 0
        hit_cost = 0
        rolled = 0
        weekly_changes: list[int] = []
        weekly_totals: list[float] = []
        chip_opportunities: list[dict] = []
        transfer_log: list[dict] = []
        chips = (
            [
                dict(window, used=False)
                for window in chip_windows(
                    season,
                    weeks[0],
                    weeks[-1],
                    chip_policy.first_wildcard_min_gw,
                    chip_policy.second_wildcard_min_gw,
                )
                if chip_policy.enabled_chips is None
                or str(window["chip"]) in chip_policy.enabled_chips
            ]
            if chip_policy
            else []
        )
        chip_log: list[dict] = []
        joint_preflight_holds = 0
        unlimited_rebuilds = 0
        assistant_manager_team: int | None = None
        assistant_manager_cost = 0
        assistant_manager_remaining = 0
        assistant_manager_log: dict | None = None
        previous_gw: int | None = None
        tracked_counts = {
            "eligibleWeeks": 0,
            "squadWeeks": 0,
            "xiWeeks": 0,
            "captainWeeks": 0,
            "initialSquad": False,
            "initialXi": False,
            "initialCaptain": False,
            "eligiblePoints": 0.0,
            "squadPoints": 0.0,
            "xiPoints": 0.0,
            "captainPoints": 0.0,
        }
        squad_spends: list[int] = []
        bench_spends: list[int] = []
        bench_premiums: list[int] = []
        bank_history: list[int] = []
        staleness_gaps: list[float] = []
        initial_selection: dict | None = None
        selection_log: list[dict] = []

        for week_number, gw in enumerate(weeks):
            frame_indices = season_context["weekIndices"][gw]
            frame = data.loc[frame_indices]
            row_by_element = dict(
                zip(element_values[frame_indices].tolist(), frame_indices.tolist())
            )
            afcon_window = AFCON_WINDOWS.get(season)
            afcon_active = bool(
                afcon_window and afcon_window[0] <= gw <= afcon_window[1]
            )
            afcon_risk_active = bool(
                afcon_window and afcon_window[0] - 1 <= gw <= afcon_window[1]
            )
            excluded_elements = {
                int(element_values[index])
                for index in frame_indices
                if afcon_active and nationality_values[index] in AFCON_NATIONS
            }
            if strategy.enforce_fieldability:
                # A player with no current fixture may be retained in the XV,
                # but cannot be bought or selected into the scoring XI. Passing
                # the set into squad utility makes each missing fieldable slot
                # visible to the transfer beam.
                excluded_elements.update(
                    int(element_values[index])
                    for index in frame_indices
                    if int(fixture_counts[index]) <= 0
                )
            lineup_excluded_elements = set(excluded_elements)
            if strategy.enforce_fieldability:
                # A narrow safety floor removes only severe non-appearance risk
                # from the preferred XI. Unlike the rejected 78% hard floor, it
                # does not bench ordinary rotation risks or playable doubts.
                lineup_excluded_elements.update(
                    int(element_values[index])
                    for index in frame_indices
                    if play_probability_values[index] < 0.60
                )
            if strategy.enforce_weekly_xi_floor:
                # Research-only strict floor. The selected policy leaves this
                # disabled after the negative ablation, but keeping the branch
                # makes that decision reproducible.
                historical_play_floor = min(
                    0.78, 0.68 + 0.025 * max(0, int(gw) - 1)
                )
                active_indices = np.asarray(
                    [
                        int(index)
                        for index in frame_indices
                        if int(element_values[index]) not in excluded_elements
                    ],
                    dtype=int,
                )
                exception_element: int | None = None
                if len(active_indices):
                    exception_threshold = float(
                        np.quantile(component_xpts_values[active_indices], 0.95)
                    )
                    exception_candidates = [
                        int(index)
                        for index in active_indices
                        if historical_play_floor - 0.10
                        <= play_probability_values[index]
                        < historical_play_floor
                        and component_xpts_values[index] >= exception_threshold
                    ]
                    if exception_candidates:
                        best_exception = max(
                            exception_candidates,
                            key=lambda index: component_xpts_values[index],
                        )
                        exception_element = int(element_values[best_exception])
                lineup_excluded_elements.update(
                    int(element_values[index])
                    for index in active_indices
                    if play_probability_values[index] < historical_play_floor
                    and int(element_values[index]) != exception_element
                )
            afcon_risk_elements = {
                int(element_values[index])
                for index in frame_indices
                if afcon_risk_active
                and nationality_values[index] in AFCON_NATIONS
            }
            incoming_by_position: dict[int, np.ndarray] = {}
            for position in SQUAD_QUOTAS:
                position_indices = frame_indices[
                    position_values[frame_indices] == position
                ]
                plan_order = position_indices[
                    np.argsort(decision_scores[position_indices])[::-1]
                ]
                if not strategy.expand_transfer_frontier:
                    incoming_by_position[position] = plan_order[:40]
                    continue

                price_denominator = np.maximum(
                    price_values[position_indices].astype(float), 35.0
                )
                value_order = position_indices[
                    np.argsort(
                        decision_scores[position_indices] / price_denominator
                    )[::-1]
                ]
                immediate_order = position_indices[
                    np.argsort(scores[position_indices])[::-1]
                ]
                reliable_indices = position_indices[
                    play_probability_values[position_indices] >= 0.62
                ]
                cheap_reliable_order = reliable_indices[
                    np.lexsort(
                        (
                            -decision_scores[reliable_indices],
                            price_values[reliable_indices],
                        )
                    )
                ]
                frontier_groups = (
                    plan_order[:16],
                    value_order[:12],
                    immediate_order[:8],
                    cheap_reliable_order[:8],
                )
                interleaved: list[int] = []
                for rank in range(max(len(group) for group in frontier_groups)):
                    for group in frontier_groups:
                        if rank < len(group):
                            interleaved.append(int(group[rank]))
                incoming_by_position[position] = np.asarray(
                    list(dict.fromkeys(interleaved)),
                    dtype=int,
                )
            if week_number == 0:
                supplied_initial = (
                    initial_squads.get((season, gw))
                    if initial_squads is not None
                    else None
                )
                initial_indices = (
                    list(supplied_initial)
                    if supplied_initial is not None
                    else initial_squad(
                        frame,
                        fresh_squad_scores,
                        excluded_elements=excluded_elements,
                        captain_weight=strategy.squad_captain_weight,
                        bench_weight=strategy.squad_bench_weight,
                        minimum_spend_gap=strategy.initial_spend_gap,
                        bench_premium_limit=strategy.bench_premium_limit,
                        bench_premium_penalty=strategy.bench_premium_penalty,
                        exact_optimiser=strategy.exact_initial_optimiser,
                        lineup_scores=fresh_objective_lineup_scores,
                        captain_utility_scores=fresh_objective_captain_scores,
                        bench_utility_scores=fresh_bench_scores,
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                )
                for index in initial_indices:
                    squad[int(element_values[index])] = {
                        "position": int(position_values[index]),
                        "team": int(team_values[index]),
                        "purchase": int(price_values[index]),
                        "last_price": int(price_values[index]),
                        "nationality": str(nationality_values[index]),
                    }
                bank = 1000 - sum(state["purchase"] for state in squad.values())
                assert_legal_squad(squad, bank, season, gw, "initial squad")
                weekly_changes.append(15)
                squad_before_transfers = {
                    element: state.copy() for element, state in squad.items()
                }
                bank_before_transfers = bank
                free_transfers_before = free_transfers
                transfers_before = transfers
            else:
                official_bank_limit = (
                    5 if season in {"2024-25", "2025-26"} else 2
                )
                bank_limit = min(strategy.bank_limit, official_bank_limit)
                # Some official events have no fixture rows. Managers could still
                # bank transfers, so advance the FT state across those deadlines.
                if previous_gw is not None and gw > previous_gw + 1:
                    free_transfers = min(
                        bank_limit, free_transfers + (gw - previous_gw - 1)
                    )
                if season == "2025-26" and gw == 16:
                    free_transfers = 5
                for element, state in squad.items():
                    if element in row_by_element:
                        current_index = row_by_element[element]
                        state["team"] = int(team_values[current_index])
                        state["last_price"] = int(price_values[current_index])
                        state["nationality"] = str(
                            nationality_values[current_index]
                        )
                if afcon_active:
                    excluded_elements.update(
                        element
                        for element, state in squad.items()
                        if str(state.get("nationality", "")) in AFCON_NATIONS
                    )
                if afcon_risk_active:
                    afcon_risk_elements.update(
                        element
                        for element, state in squad.items()
                        if str(state.get("nationality", "")) in AFCON_NATIONS
                    )

                unlimited_rebuild = gw in UNLIMITED_TRANSFER_GWS.get(season, set())
                if unlimited_rebuild:
                    available_budget = bank + sum(
                        selling_price(
                            int(state["purchase"]),
                            int(state["last_price"]),
                        )
                        for state in squad.values()
                    )
                    reset_indices = initial_squad(
                        frame,
                        fresh_squad_scores,
                        budget_limit=available_budget,
                        excluded_elements=excluded_elements,
                        captain_weight=strategy.squad_captain_weight,
                        bench_weight=strategy.squad_bench_weight,
                        minimum_spend_gap=strategy.initial_spend_gap,
                        bench_premium_limit=strategy.bench_premium_limit,
                        bench_premium_penalty=strategy.bench_premium_penalty,
                        exact_optimiser=strategy.exact_initial_optimiser,
                        lineup_scores=fresh_objective_lineup_scores,
                        captain_utility_scores=fresh_objective_captain_scores,
                        bench_utility_scores=fresh_bench_scores,
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                    squad = {
                        int(element_values[index]): {
                            "position": int(position_values[index]),
                            "team": int(team_values[index]),
                            "purchase": int(price_values[index]),
                            "last_price": int(price_values[index]),
                            "nationality": str(nationality_values[index]),
                        }
                        for index in reset_indices
                    }
                    bank = available_budget - sum(
                        int(price_values[index]) for index in reset_indices
                    )
                    # Project Restart reverted to the normal one-FT cadence after
                    # the free rebuild. The World Cup window allowed the GW17 FT
                    # to roll, so its state is preserved through the update below.
                    if season == "2019-20":
                        free_transfers = 0
                    unlimited_rebuilds += 1
                    assert_legal_squad(
                        squad, bank, season, gw, "unlimited-transfer rebuild"
                    )

                squad_before_transfers = {
                    element: state.copy() for element, state in squad.items()
                }
                bank_before_transfers = bank
                free_transfers_before = free_transfers
                transfers_before = transfers
                changes_this_week = 0
                hits_before = hits
                hit_cost_before = hit_cost
                hit_points_this_week = 0
                current_schedule = schedule_counts.get(gw, {})
                blank_squad = sum(
                    current_schedule.get(int(state["team"]), 0) == 0
                    for state in squad.values()
                )
                free_hit_available = any(
                    window["chip"] == "Free Hit"
                    and not window["used"]
                    and int(window["start"]) <= gw <= int(window["end"])
                    for window in chips
                )
                preflight_free_hit = False
                if (
                    strategy.joint_chip_preflight
                    and chip_policy is not None
                    and free_hit_available
                    and blank_squad >= 3
                ):
                    # Decide the Free Hit *before* suppressing permanent moves.
                    # The former shortcut stood down the transfer planner in
                    # every severe blank whenever FH was unused, even when the
                    # later chip gate said Hold.  That contaminated full-season
                    # chip deltas with weeks where neither action was taken.
                    pre_xi, _ = choose_xi(
                        squad, row_by_element, scores, lineup_excluded_elements
                    )
                    pre_captain_metric = (
                        captain_scores if captain_scores is not None else scores
                    )
                    pre_captain = max(
                        pre_xi,
                        key=lambda element: pre_captain_metric[
                            row_by_element[element]
                        ]
                        if element in row_by_element
                        and element not in lineup_excluded_elements
                        else -1.0,
                    )
                    pre_free_indices = (
                        free_hit_squads.get((season, gw), [])
                        if free_hit_squads is not None
                        else []
                    )
                    if not pre_free_indices:
                        pre_budget = bank + sum(
                            selling_price(
                                int(state["purchase"]),
                                int(state["last_price"]),
                            )
                            for state in squad.values()
                        )
                        pre_free_indices = initial_squad(
                            frame,
                            scores,
                            budget_limit=pre_budget,
                            excluded_elements=excluded_elements,
                            captain_weight=1.0,
                            bench_weight=0.08,
                            minimum_spend_gap=None,
                            bench_premium_limit=strategy.bench_premium_limit,
                            bench_premium_penalty=strategy.bench_premium_penalty,
                            exact_optimiser=strategy.exact_initial_optimiser,
                            lineup_scores=scores,
                            captain_utility_scores=fresh_captain_utility,
                            risk_scores=risk_scores,
                            risk_aversion=strategy.squad_risk_aversion,
                            defence_correlation=strategy.defence_residual_correlation,
                        )
                    pre_free_state = {
                        int(element_values[index]): {
                            "position": int(position_values[index]),
                            "team": int(team_values[index]),
                            "purchase": int(price_values[index]),
                            "last_price": int(price_values[index]),
                            "nationality": str(nationality_values[index]),
                        }
                        for index in pre_free_indices
                    }
                    pre_free_xi, _ = choose_xi(
                        pre_free_state,
                        row_by_element,
                        scores,
                        lineup_excluded_elements,
                    )
                    pre_free_captain = max(
                        pre_free_xi,
                        key=lambda element: pre_captain_metric[
                            row_by_element[element]
                        ]
                        if element in row_by_element
                        and element not in lineup_excluded_elements
                        else -1.0,
                    )

                    def preflight_lineup_value(
                        active_xi: list[int], active_captain: int
                    ) -> float:
                        return float(
                            sum(
                                scores[row_by_element[element]]
                                for element in active_xi
                                if element in row_by_element
                                and element not in excluded_elements
                            )
                            + (
                                scores[row_by_element[active_captain]]
                                if active_captain in row_by_element
                                and active_captain not in excluded_elements
                                else 0.0
                            )
                        )

                    pre_double_count = sum(
                        fixture_counts[row_by_element[element]] > 1
                        for element in pre_free_xi
                        if element in row_by_element
                    )
                    pre_signal = (
                        preflight_lineup_value(pre_free_xi, pre_free_captain)
                        - preflight_lineup_value(pre_xi, pre_captain)
                        + 0.22 * max(0, blank_squad - 1)
                        + 0.12 * pre_double_count
                    )
                    free_window = next(
                        window
                        for window in chips
                        if window["chip"] == "Free Hit"
                        and not window["used"]
                        and int(window["start"]) <= gw <= int(window["end"])
                    )
                    pre_remaining = max(0, int(free_window["end"]) - gw)
                    pre_threshold = max(
                        0.60 * chip_policy.free_hit_gap,
                        chip_policy.free_hit_gap
                        - chip_policy.free_hit_gap
                        * 0.22
                        * math.exp(-pre_remaining / 2.3),
                    )
                    override = (
                        chip_value_overrides.get((season, int(gw), "Free Hit"))
                        if chip_value_overrides
                        else None
                    )
                    decision_signal = (
                        float(override) if override is not None else pre_signal
                    )
                    preflight_free_hit = bool(
                        decision_signal >= pre_threshold
                        and (blank_squad >= 3 or pre_double_count >= 5)
                    )
                if preflight_free_hit:
                    joint_preflight_holds += 1
                # Future rescheduling announcement timestamps are absent from the
                # archive. The horizon score already values persistent player/team
                # quality; do not leak final future blank/double assignments into
                # transfer option value.
                team_option_score: dict[int, float] = {
                    int(team_id): 0.0 for team_id in season_teams
                }
                staleness_gap = 0.0
                if strategy.staleness_gap_trigger is not None:
                    available_budget = bank + sum(
                        selling_price(
                            int(state["purchase"]),
                            int(state["last_price"]),
                        )
                        for state in squad.values()
                    )
                    fresh_indices_for_gap = initial_squad(
                        frame,
                        fresh_squad_scores,
                        budget_limit=available_budget,
                        excluded_elements=excluded_elements,
                        captain_weight=strategy.squad_captain_weight,
                        bench_weight=strategy.squad_bench_weight,
                        lineup_scores=fresh_objective_lineup_scores,
                        captain_utility_scores=fresh_objective_captain_scores,
                        bench_utility_scores=fresh_bench_scores,
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                    fresh_state_for_gap = {
                        int(element_values[index]): {
                            "position": int(position_values[index]),
                            "team": int(team_values[index]),
                            "purchase": int(price_values[index]),
                            "last_price": int(price_values[index]),
                            "nationality": str(nationality_values[index]),
                        }
                        for index in fresh_indices_for_gap
                    }
                    current_gap_utility = squad_decision_utility(
                        squad,
                        row_by_element,
                        decision_scores,
                        excluded_elements=excluded_elements,
                        captain_weight=strategy.squad_captain_weight,
                        bench_weight=strategy.squad_bench_weight,
                        bench_scores=(
                            bench_utility_scores
                            if consistent_decision_objective
                            else None
                        ),
                        captain_scores=(
                            decision_captain_utility
                            if consistent_decision_objective
                            or strategy.align_captain_objective
                            else None
                        ),
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                    fresh_gap_utility = squad_decision_utility(
                        fresh_state_for_gap,
                        row_by_element,
                        decision_scores,
                        excluded_elements=excluded_elements,
                        captain_weight=strategy.squad_captain_weight,
                        bench_weight=strategy.squad_bench_weight,
                        bench_scores=(
                            bench_utility_scores
                            if consistent_decision_objective
                            else None
                        ),
                        captain_scores=(
                            decision_captain_utility
                            if consistent_decision_objective
                            or strategy.align_captain_objective
                            else None
                        ),
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                    staleness_gap = max(
                        0.0,
                        float(fresh_gap_utility - current_gap_utility),
                    )
                staleness_gaps.append(staleness_gap)
                if (
                    strategy.joint_squad_optimiser
                    and not preflight_free_hit
                    and not unlimited_rebuild
                ):
                    squad_before_joint = set(squad)
                    squad, bank, changes_this_week, paid_moves, _ = joint_transfer_plan(
                        squad=squad,
                        bank=bank,
                        free_transfers=free_transfers,
                        row_by_element=row_by_element,
                        incoming_by_position=incoming_by_position,
                        element_values=element_values,
                        position_values=position_values,
                        team_values=team_values,
                        price_values=price_values,
                        plan_scores=decision_scores,
                        bench_scores=(
                            bench_utility_scores
                            if consistent_decision_objective
                            else None
                        ),
                        captain_utility_scores=(
                            decision_captain_utility
                            if consistent_decision_objective
                            or strategy.align_captain_objective
                            else None
                        ),
                        price_rise_values=price_rise_values,
                        price_fall_values=price_fall_values,
                        uncertainty_values=uncertainty_values,
                        risk_scores=risk_scores,
                        excluded_elements=excluded_elements,
                        team_option_score=team_option_score,
                        strategy=strategy,
                        gw=gw,
                        club_limits=(
                            {assistant_manager_team: 2}
                            if assistant_manager_team is not None
                            else None
                        ),
                        staleness_gap=staleness_gap,
                        package_action_adjustment=package_action_adjustment,
                    )
                    if changes_this_week:
                        outgoing_elements = sorted(squad_before_joint - set(squad))
                        incoming_elements = sorted(set(squad) - squad_before_joint)
                        transfer_log.append(
                            {
                                "gw": int(gw),
                                "outElements": outgoing_elements,
                                "inElements": incoming_elements,
                                "out": [
                                    str(display_name_values[row_by_element[element]])
                                    if element in row_by_element
                                    else str(element)
                                    for element in outgoing_elements
                                ],
                                "in": [
                                    str(display_name_values[row_by_element[element]])
                                    if element in row_by_element
                                    else str(element)
                                    for element in incoming_elements
                                ],
                                "bank": round(bank / 10, 1),
                                "hits": int(paid_moves),
                            }
                        )
                    transfers += changes_this_week
                    if paid_moves:
                        hits += paid_moves
                        hit_cost += 4 * paid_moves
                        hit_points_this_week += 4 * paid_moves
                    move_range = range(0)
                else:
                    move_range = range(
                        0
                        if preflight_free_hit or unlimited_rebuild
                        else free_transfers + 1
                    )
                for move_number in move_range:
                    is_hit = move_number >= free_transfers
                    if is_hit and hits >= strategy.max_hits:
                        break
                    team_counts: dict[int, int] = {}
                    for state in squad.values():
                        team_counts[int(state["team"])] = (
                            team_counts.get(int(state["team"]), 0) + 1
                        )
                    best_move: tuple[float, int, int, int, int] | None = None
                    overloaded_clubs = {
                        team_id
                        for team_id, count in team_counts.items()
                        if count > 3
                    }
                    if is_hit and overloaded_clubs:
                        # The inherited state may be held; FPL does not require a
                        # paid transfer solely because a real-world move caused it.
                        break
                    for outgoing, state in squad.items():
                        if (
                            overloaded_clubs
                            and int(state["team"]) not in overloaded_clubs
                        ):
                            continue
                        out_index = row_by_element.get(outgoing)
                        out_score = (
                            transfer_gain_scores[out_index]
                            if out_index is not None and outgoing not in excluded_elements
                            else -0.30
                        )
                        current_price = (
                            int(price_values[out_index])
                            if out_index is not None
                            else int(state["last_price"])
                        )
                        sale = selling_price(int(state["purchase"]), current_price)
                        position = int(state["position"])
                        for incoming_index in incoming_by_position[position]:
                            incoming_index = int(incoming_index)
                            incoming_element = int(element_values[incoming_index])
                            if (
                                incoming_element in squad
                                or incoming_element in excluded_elements
                            ):
                                continue
                            incoming_team = int(team_values[incoming_index])
                            incoming_price = int(price_values[incoming_index])
                            if incoming_price > bank + sale:
                                continue
                            incoming_limit = (
                                2
                                if assistant_manager_team is not None
                                and incoming_team == assistant_manager_team
                                else 3
                            )
                            # A real-world January transfer can temporarily create
                            # four players from one club. The next permanent move
                            # must restore the quota; a same-club swap does not.
                            if overloaded_clubs and incoming_team in overloaded_clubs:
                                continue
                            if (
                                incoming_team != int(state["team"])
                                and team_counts.get(incoming_team, 0)
                                >= incoming_limit
                            ):
                                continue
                            gain = float(
                                transfer_gain_scores[int(incoming_index)] - out_score
                            )
                            if strategy.early_price_weight > 0 and gw <= 19:
                                outgoing_price_option = (
                                    price_rise_values[out_index]
                                    - price_fall_values[out_index]
                                    if out_index is not None
                                    else -0.15
                                )
                                incoming_price_option = (
                                    price_rise_values[incoming_index]
                                    - price_fall_values[incoming_index]
                                )
                                gain += strategy.early_price_weight * (
                                    incoming_price_option - outgoing_price_option
                                )
                            if strategy.joint_chip_preflight:
                                gain += 0.45 * (
                                    team_option_score.get(incoming_team, 0.0)
                                    - team_option_score.get(int(state["team"]), 0.0)
                                )
                            if best_move is None or gain > best_move[0]:
                                best_move = (
                                    gain,
                                    outgoing,
                                    incoming_element,
                                    int(incoming_index),
                                    sale,
                                )
                    if best_move is None:
                        break
                    gain, outgoing, incoming, incoming_index, sale = best_move
                    move_hurdle = strategy.transfer_hurdle + (
                        4.0 if is_hit else 0.0
                    )
                    if overloaded_clubs:
                        move_hurdle = -math.inf
                    if strategy.phase_banking:
                        if free_transfers_before <= 1:
                            move_hurdle += 2.0 if gw <= 19 else 2.8
                        else:
                            move_hurdle -= 1.15 if gw <= 19 else 0.55
                    if strategy.hold_option_value > 0:
                        uncertainty_penalty = strategy.hold_option_value * min(
                            1.5,
                            float(uncertainty_values[incoming_index]) / 3.0,
                        )
                        bank_adjustment = (
                            strategy.hold_option_value
                            if free_transfers_before <= 1
                            else -0.35 * strategy.hold_option_value
                        )
                        move_hurdle += uncertainty_penalty + bank_adjustment
                    if strategy.gain_realisation * gain <= move_hurdle:
                        break
                    out_index = row_by_element.get(outgoing)
                    immediate_out = (
                        scores[out_index]
                        if out_index is not None and outgoing not in excluded_elements
                        else -0.30
                    )
                    immediate_gain = float(scores[incoming_index] - immediate_out)
                    if is_hit and immediate_gain <= strategy.hit_immediate_hurdle:
                        break
                    bank += sale - int(price_values[incoming_index])
                    del squad[outgoing]
                    squad[incoming] = {
                        "position": int(position_values[incoming_index]),
                        "team": int(team_values[incoming_index]),
                        "purchase": int(price_values[incoming_index]),
                        "last_price": int(price_values[incoming_index]),
                        "nationality": str(nationality_values[incoming_index]),
                    }
                    transfer_log.append(
                        {
                            "gw": int(gw),
                            "outElements": [int(outgoing)],
                            "inElements": [int(incoming)],
                            "out": [
                                str(display_name_values[out_index])
                                if out_index is not None
                                else str(outgoing)
                            ],
                            "in": [str(display_name_values[incoming_index])],
                            "bank": round(bank / 10, 1),
                        }
                    )
                    changes_this_week += 1
                    transfers += 1
                    if is_hit:
                        hits += 1
                        hit_cost += 4
                        hit_points_this_week += 4
                if changes_this_week == 0:
                    rolled += 1
                weekly_changes.append(changes_this_week)
                free_transfers = min(
                    bank_limit,
                    max(0, free_transfers - changes_this_week) + 1,
                )
                assert_legal_squad(
                    squad,
                    bank,
                    season,
                    gw,
                    "post-transfer",
                    allow_temporary_club_overload=changes_this_week == 0,
                )
            xi, bench = choose_xi(
                squad, row_by_element, scores, lineup_excluded_elements
            )
            if captain_scores is not None:
                captain_metric = captain_scores
            elif strategy.captain_mode == "attacking_tail":
                captain_metric = attacking_captain_score
            elif strategy.safe_captain:
                captain_metric = safe_captain_score
            else:
                captain_metric = scores
            captain_order = sorted(
                xi,
                key=lambda element: captain_metric[row_by_element[element]]
                if element in row_by_element and element not in lineup_excluded_elements
                else -1.0,
                reverse=True,
            )
            captain, vice = captain_order[:2]
            if week_number == 0:
                initial_selection = {
                    "squad": [
                        str(display_name_values[row_by_element[element]])
                        for element in squad
                        if element in row_by_element
                    ],
                    "xi": [
                        str(display_name_values[row_by_element[element]])
                        for element in xi
                        if element in row_by_element
                    ],
                    "bench": [
                        str(display_name_values[row_by_element[element]])
                        for element in bench
                        if element in row_by_element
                    ],
                    "captain": str(display_name_values[row_by_element[captain]]),
                    "predictedXi": round(
                        float(sum(scores[row_by_element[element]] for element in xi)),
                        3,
                    ),
                    "predictedCaptain": round(
                        float(scores[row_by_element[captain]]), 3
                    ),
                    "planXi": round(
                        float(
                            sum(plan_scores[row_by_element[element]] for element in xi)
                        ),
                        3,
                    ),
                }
            base_breakdown = realised_week_breakdown(
                xi,
                bench,
                captain,
                vice,
                squad,
                row_by_element,
                actual,
                played_minutes,
            )
            # Keep the lineup that actually earns this week's points separate
            # from the persistent squad state.  A Free Hit is scored with a
            # temporary squad and then immediately reverts; audit logs must
            # record the temporary XI rather than the reverted squad.
            scoring_squad = squad
            scoring_xi = xi
            scoring_bench = bench
            scoring_captain = captain
            scoring_vice = vice
            week_points = base_breakdown["normal"] - (
                hit_points_this_week if week_number > 0 else 0
            )
            chip_opportunities.append(
                {
                    "gw": int(gw),
                    "predictedTripleCaptainGain": round(
                        float(scores[row_by_element[captain]]), 4
                    ),
                    "actualTripleCaptainGain": round(
                        float(
                            base_breakdown["triple_captain"]
                            - base_breakdown["normal"]
                        ),
                        1,
                    ),
                    "predictedBenchBoostGain": round(
                        float(
                            sum(
                                max(0.0, scores[row_by_element[element]])
                                for element in bench
                                if element in row_by_element
                                and element not in excluded_elements
                            )
                        ),
                        4,
                    ),
                    "actualBenchBoostGain": round(
                        float(
                            base_breakdown["bench_boost"]
                            - base_breakdown["normal"]
                        ),
                        1,
                    ),
                    "captainFixtureCount": int(
                        fixture_counts[row_by_element[captain]]
                    ),
                    "benchDoubleCount": int(
                        sum(
                            fixture_counts[row_by_element[element]] > 1
                            for element in bench
                            if element in row_by_element
                        )
                    ),
                    "captainPlayProbability": round(
                        float(play_probability_values[row_by_element[captain]]), 4
                    ),
                    "captainStartProbability": round(
                        float(start_probability_values[row_by_element[captain]]), 4
                    ),
                    "captainSixtyProbability": round(
                        float(sixty_probability_values[row_by_element[captain]]), 4
                    ),
                    "captainExpectedMinutes": round(
                        float(expected_minutes_values[row_by_element[captain]]), 2
                    ),
                    "captainUncertainty": round(
                        float(uncertainty_values[row_by_element[captain]]), 4
                    ),
                    "captainReturnProbability": round(
                        float(return_values[row_by_element[captain]]), 4
                    ),
                    "captainHaulProbability": round(
                        float(haul_values[row_by_element[captain]]), 4
                    ),
                    "captainPrice": int(price_values[row_by_element[captain]]),
                    "benchFixtureCount": int(
                        sum(
                            fixture_counts[row_by_element[element]]
                            for element in bench
                            if element in row_by_element
                            and element not in excluded_elements
                        )
                    ),
                    "benchBlankCount": int(
                        sum(
                            fixture_counts[row_by_element[element]] == 0
                            for element in bench
                            if element in row_by_element
                            and element not in excluded_elements
                        )
                    ),
                    "benchPlayProbability": round(
                        float(
                            sum(
                                play_probability_values[row_by_element[element]]
                                for element in bench
                                if element in row_by_element
                                and element not in excluded_elements
                            )
                        ),
                        4,
                    ),
                    "benchSixtyProbability": round(
                        float(
                            sum(
                                sixty_probability_values[row_by_element[element]]
                                for element in bench
                                if element in row_by_element
                                and element not in excluded_elements
                            )
                        ),
                        4,
                    ),
                    "benchExpectedMinutes": round(
                        float(
                            sum(
                                expected_minutes_values[row_by_element[element]]
                                for element in bench
                                if element in row_by_element
                                and element not in excluded_elements
                            )
                        ),
                        2,
                    ),
                    "benchUncertainty": round(
                        float(
                            math.sqrt(
                                sum(
                                    uncertainty_values[row_by_element[element]] ** 2
                                    for element in bench
                                    if element in row_by_element
                                    and element not in excluded_elements
                                )
                            )
                        ),
                        4,
                    ),
                    "benchMinimumPlayProbability": round(
                        float(
                            min(
                                (
                                    play_probability_values[row_by_element[element]]
                                    for element in bench
                                    if element in row_by_element
                                    and element not in excluded_elements
                                ),
                                default=0.0,
                            )
                        ),
                        4,
                    ),
                }
            )
            assistant_manager_block_this_week = assistant_manager_team is not None
            if assistant_manager_team is not None:
                manager_index = next(
                    (
                        int(index)
                        for index in frame_indices
                        if int(team_values[index]) == assistant_manager_team
                    ),
                    None,
                )
                manager_points = (
                    float(assistant_manager_actual_values[manager_index])
                    if manager_index is not None
                    else 0.0
                )
                week_points += manager_points
                if assistant_manager_log is not None:
                    assistant_manager_log["gain"] = round(
                        float(assistant_manager_log["gain"]) + manager_points
                    )
                assistant_manager_remaining -= 1
                if assistant_manager_remaining == 0:
                    bank += assistant_manager_cost
                    assistant_manager_team = None
                    assistant_manager_cost = 0
                    assistant_manager_log = None

            if chip_policy:
                enabled_chip_names = {str(window["chip"]) for window in chips}
                current_indices = [
                    row_by_element[element]
                    for element in squad
                    if element in row_by_element
                ]
                wildcard_indices = current_indices
                if "Wildcard" in enabled_chip_names:
                    wildcard_indices = (
                        fresh_squads.get((season, gw), [])
                        if fresh_squads is not None
                        else initial_squad(
                        frame,
                        fresh_squad_scores,
                        excluded_elements=excluded_elements,
                        captain_weight=strategy.squad_captain_weight,
                        bench_weight=strategy.squad_bench_weight,
                        minimum_spend_gap=strategy.initial_spend_gap,
                        bench_premium_limit=strategy.bench_premium_limit,
                        bench_premium_penalty=strategy.bench_premium_penalty,
                        exact_optimiser=strategy.exact_initial_optimiser,
                        lineup_scores=fresh_objective_lineup_scores,
                        captain_utility_scores=fresh_objective_captain_scores,
                        bench_utility_scores=fresh_bench_scores,
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                    )
                free_hit_indices = current_indices
                if "Free Hit" in enabled_chip_names:
                    free_hit_indices = (
                        free_hit_squads.get((season, gw), [])
                        if free_hit_squads is not None
                        else initial_squad(
                        frame,
                        scores,
                        excluded_elements=excluded_elements,
                        captain_weight=1.0,
                        bench_weight=0.08,
                        minimum_spend_gap=None,
                        bench_premium_limit=strategy.bench_premium_limit,
                        bench_premium_penalty=strategy.bench_premium_penalty,
                        exact_optimiser=strategy.exact_initial_optimiser,
                        lineup_scores=scores,
                        captain_utility_scores=fresh_captain_utility,
                        risk_scores=risk_scores,
                        risk_aversion=strategy.squad_risk_aversion,
                        defence_correlation=strategy.defence_residual_correlation,
                    )
                    )

                def squad_state(indices: list[int]) -> dict[int, dict]:
                    return {
                        int(element_values[index]): {
                            "position": int(position_values[index]),
                            "team": int(team_values[index]),
                            "purchase": int(price_values[index]),
                            "last_price": int(price_values[index]),
                            "nationality": str(nationality_values[index]),
                        }
                        for index in indices
                    }

                wildcard_state = squad_state(wildcard_indices)
                wildcard_xi, wildcard_bench = choose_xi(
                    wildcard_state, row_by_element, scores, lineup_excluded_elements
                )
                wildcard_captain_order = sorted(
                    wildcard_xi,
                    key=lambda element: captain_metric[row_by_element[element]]
                    if element in row_by_element and element not in lineup_excluded_elements
                    else -1.0,
                    reverse=True,
                )
                wildcard_captain, wildcard_vice = wildcard_captain_order[:2]
                free_hit_state = squad_state(free_hit_indices)
                free_hit_xi, free_hit_bench = choose_xi(
                    free_hit_state, row_by_element, scores, lineup_excluded_elements
                )
                free_hit_captain_order = sorted(
                    free_hit_xi,
                    key=lambda element: captain_metric[row_by_element[element]]
                    if element in row_by_element and element not in lineup_excluded_elements
                    else -1.0,
                    reverse=True,
                )
                free_hit_captain, free_hit_vice = free_hit_captain_order[:2]

                def predicted_lineup_value(
                    active_xi: list[int], active_captain: int
                ) -> float:
                    return float(
                        sum(
                            scores[row_by_element[element]]
                            for element in active_xi
                            if element in row_by_element
                            and element not in lineup_excluded_elements
                        )
                        + (
                            scores[row_by_element[active_captain]]
                            if active_captain in row_by_element
                            and active_captain not in excluded_elements
                            else 0
                        )
                    )

                current_lineup_value = predicted_lineup_value(xi, captain)
                free_hit_lineup_value = predicted_lineup_value(
                    free_hit_xi, free_hit_captain
                )
                current_squad_value = squad_decision_utility(
                    squad,
                    row_by_element,
                    decision_scores,
                    excluded_elements=excluded_elements,
                    captain_weight=strategy.squad_captain_weight,
                    bench_weight=strategy.squad_bench_weight,
                    bench_scores=(
                        bench_utility_scores
                        if consistent_decision_objective
                        else None
                    ),
                    captain_scores=(
                        decision_captain_utility
                        if consistent_decision_objective
                        or strategy.align_captain_objective
                        else None
                    ),
                    risk_scores=risk_scores,
                    risk_aversion=strategy.squad_risk_aversion,
                    defence_correlation=strategy.defence_residual_correlation,
                )
                pre_transfer_squad_value = squad_decision_utility(
                    squad_before_transfers,
                    row_by_element,
                    decision_scores,
                    excluded_elements=excluded_elements,
                    captain_weight=strategy.squad_captain_weight,
                    bench_weight=strategy.squad_bench_weight,
                    bench_scores=(
                        bench_utility_scores
                        if consistent_decision_objective
                        else None
                    ),
                    captain_scores=(
                        decision_captain_utility
                        if consistent_decision_objective
                        or strategy.align_captain_objective
                        else None
                    ),
                    risk_scores=risk_scores,
                    risk_aversion=strategy.squad_risk_aversion,
                    defence_correlation=strategy.defence_residual_correlation,
                )
                permanent_transfer_value = max(
                    0.0, current_squad_value - pre_transfer_squad_value
                )
                wildcard_squad_value = squad_decision_utility(
                    wildcard_state,
                    row_by_element,
                    decision_scores,
                    excluded_elements=excluded_elements,
                    captain_weight=strategy.squad_captain_weight,
                    bench_weight=strategy.squad_bench_weight,
                    bench_scores=(
                        bench_utility_scores
                        if consistent_decision_objective
                        else None
                    ),
                    captain_scores=(
                        decision_captain_utility
                        if consistent_decision_objective
                        or strategy.align_captain_objective
                        else None
                    ),
                    risk_scores=risk_scores,
                    risk_aversion=strategy.squad_risk_aversion,
                    defence_correlation=strategy.defence_residual_correlation,
                )
                afcon_count = sum(element in afcon_risk_elements for element in squad)
                blank_count = sum(
                    element not in row_by_element
                    or element in excluded_elements
                    or fixture_counts[row_by_element[element]] == 0
                    for element in squad
                )
                double_count = sum(
                    fixture_counts[row_by_element[element]] > 1
                    for element in free_hit_xi
                    if element in row_by_element
                )
                bench_double_count = sum(
                    fixture_counts[row_by_element[element]] > 1
                    for element in bench
                    if element in row_by_element
                    and element not in excluded_elements
                )
                bench_metric = sum(
                    # The score is now a Gameweek total, so a bench double is
                    # already worth twice a bench single; the old hand-added
                    # 0.15 nudge would double-count it.
                    max(0.0, scores[row_by_element[element]])
                    for element in bench
                    if element in row_by_element
                    and element not in excluded_elements
                )
                captain_index = row_by_element.get(captain)
                triple_metric = (
                    # The captain ranker chooses the armband, but its percentile
                    # scale is not an expected-points scale. Chip thresholds are
                    # calibrated in projected points, so never compare them with
                    # a 0-1 rank score.
                    triple_captain_signal(
                        float(scores[captain_index]),
                        int(fixture_counts[captain_index]),
                    )
                    if captain_index is not None and captain not in excluded_elements
                    else 0.0
                )
                team_position = {
                    int(team_values[index]): float(table_position_values[index])
                    for index in frame_indices
                }
                squad_club_counts: dict[int, int] = {}
                for state in squad.values():
                    club = int(state["team"])
                    squad_club_counts[club] = squad_club_counts.get(club, 0) + 1
                assistant_manager_option: tuple[int, int, float, int] | None = None
                if season == "2024-25" and assistant_manager_team is None:
                    for manager_index in frame.drop_duplicates("team_id").index:
                        manager_index = int(manager_index)
                        team_id = int(team_values[manager_index])
                        if squad_club_counts.get(team_id, 0) >= 3:
                            continue
                        normalized_name = "".join(
                            character
                            for character in team_name_values[manager_index].lower()
                            if character.isalnum()
                        )
                        cost = ASSISTANT_MANAGER_COST_2024.get(
                            normalized_name, 8
                        )
                        if cost > bank:
                            continue
                        win_probability = float(
                            np.clip(
                                0.12
                                + 0.21 * team_form_values[manager_index]
                                + 0.10
                                * (
                                    team_attack_values[manager_index]
                                    - team_defence_values[manager_index]
                                ),
                                0.08,
                                0.78,
                            )
                        )
                        draw_probability = float(
                            np.clip(0.31 - 0.16 * win_probability, 0.16, 0.30)
                        )
                        expected_match_points = (
                            6 * win_probability
                            + 3 * draw_probability
                            + team_attack_values[manager_index]
                            + 2 * team_clean_values[manager_index]
                        )
                        current_games = max(
                            0, int(fixture_counts[manager_index])
                        )
                        projected = expected_match_points * (2 + current_games)
                        opponent_id = int(opponent_team_values[manager_index])
                        if (
                            opponent_id in team_position
                            and team_position.get(team_id, 10)
                            - team_position[opponent_id]
                            >= 5
                        ):
                            projected += (
                                10 * win_probability + 5 * draw_probability
                            )
                        option = (team_id, cost, float(projected), manager_index)
                        if (
                            assistant_manager_option is None
                            or option[2] > assistant_manager_option[2]
                        ):
                            assistant_manager_option = option
                free_hit_immediate_metric = (
                    free_hit_lineup_value
                    - current_lineup_value
                    + 0.22 * max(0, blank_count - 1)
                    + 0.12 * double_count
                )
                metrics = {
                    "Wildcard": wildcard_squad_value
                    - current_squad_value
                    + chip_policy.afcon_bonus * afcon_count,
                    # A Free Hit replaces the permanent transfer decision made
                    # by the no-chip policy.  The one-week XI gain is therefore
                    # not free: subtract the causal horizon value of the moves
                    # that would be forfeited when the squad reverts.
                    "Free Hit": free_hit_immediate_metric
                    - permanent_transfer_value,
                    "Bench Boost": bench_metric,
                    "Triple Captain": triple_metric,
                    "Assistant Manager": (
                        assistant_manager_option[2]
                        if assistant_manager_option is not None
                        else -math.inf
                    ),
                }
                free_hit_breakdown = realised_week_breakdown(
                    free_hit_xi,
                    free_hit_bench,
                    free_hit_captain,
                    free_hit_vice,
                    free_hit_state,
                    row_by_element,
                    actual,
                    played_minutes,
                )
                raw_free_hit_metric = float(metrics["Free Hit"])
                chip_opportunities[-1].update(
                    {
                        "predictedFreeHitImmediateGain": round(
                            float(free_hit_immediate_metric), 4
                        ),
                        "predictedFreeHitGain": round(raw_free_hit_metric, 4),
                        "permanentTransferValueForegone": round(
                            float(permanent_transfer_value), 4
                        ),
                        "actualFreeHitGain": round(
                            float(
                                free_hit_breakdown["normal"]
                                - base_breakdown["normal"]
                            ),
                            1,
                        ),
                        "freeHitBlankCount": int(blank_count),
                        "freeHitDoubleCount": int(double_count),
                        "freeHitLineupOverlap": int(
                            len(set(free_hit_xi).intersection(xi))
                        ),
                        "currentLineupValue": round(current_lineup_value, 4),
                        "freeHitLineupValue": round(free_hit_lineup_value, 4),
                        "currentLineupExpectedMinutes": round(
                            float(
                                sum(
                                    expected_minutes_values[row_by_element[element]]
                                    for element in xi
                                    if element in row_by_element
                                )
                            ),
                            2,
                        ),
                        "freeHitLineupExpectedMinutes": round(
                            float(
                                sum(
                                    expected_minutes_values[row_by_element[element]]
                                    for element in free_hit_xi
                                    if element in row_by_element
                                )
                            ),
                            2,
                        ),
                        "currentLineupUncertainty": round(
                            float(
                                sum(
                                    uncertainty_values[row_by_element[element]]
                                    for element in xi
                                    if element in row_by_element
                                )
                            ),
                            4,
                        ),
                        "freeHitLineupUncertainty": round(
                            float(
                                sum(
                                    uncertainty_values[row_by_element[element]]
                                    for element in free_hit_xi
                                    if element in row_by_element
                                )
                            ),
                            4,
                        ),
                    }
                )
                if chip_value_overrides:
                    for chip_name in ("Wildcard", "Free Hit", "Bench Boost", "Triple Captain"):
                        override = chip_value_overrides.get(
                            (season, int(gw), chip_name)
                        )
                        if override is not None:
                            metrics[chip_name] = float(override)
                thresholds = {
                    "Wildcard": chip_policy.wildcard_gap,
                    "Free Hit": chip_policy.free_hit_gap,
                    "Bench Boost": chip_policy.bench_score,
                    "Triple Captain": chip_policy.triple_score,
                    "Assistant Manager": 18.0,
                }
                available = [
                    window
                    for window in chips
                    if not assistant_manager_block_this_week
                    and assistant_manager_team is None
                    and not window["used"]
                    and int(window["start"]) <= gw <= int(window["end"])
                ]
                def has_structural_signal(chip_name: str) -> bool:
                    if chip_name == "Free Hit":
                        return blank_count >= 3 or double_count >= 5
                    if chip_name == "Bench Boost":
                        return bench_double_count >= 1
                    if chip_name == "Triple Captain":
                        return bool(
                            captain_index is not None
                            and fixture_counts[captain_index] > 1
                        )
                    if chip_name == "Assistant Manager":
                        return assistant_manager_option is not None
                    return True

                effective_thresholds: dict[int, float] = {}
                option_values: dict[int, float] = {}
                expiring_windows: set[int] = set()
                for window in available:
                    chip_name = str(window["chip"])
                    base_threshold = thresholds[chip_name]
                    remaining = max(0, int(window["end"]) - gw)
                    # Optimal-stopping option value. Playing a chip today
                    # forfeits every remaining week in its window, so the bar is
                    # raised while those weeks exist and ramps down as they run
                    # out, reaching a token positive-value check in the last
                    # legal week. This uses the window length only — never the
                    # future schedule, whose blank/double announcement dates are
                    # absent from the archive.
                    horizon_share = 1.0 - math.exp(
                        -remaining / CHIP_HOLD_DECAY_GWS
                    )
                    expiry_share = CHIP_EXPIRY_THRESHOLD_SHARE.get(
                        chip_name, DEFAULT_CHIP_EXPIRY_THRESHOLD_SHARE
                    )
                    key = id(window)
                    option_values[key] = (
                        base_threshold * CHIP_HOLD_VALUE * horizon_share
                    )
                    effective_thresholds[key] = base_threshold * (
                        expiry_share
                        + (1.0 + CHIP_HOLD_VALUE - expiry_share) * horizon_share
                    )
                    if remaining <= CHIP_FORCED_USE_WINDOW_GWS:
                        expiring_windows.add(key)

                choices = [
                    window
                    for window in available
                    if metrics[str(window["chip"])]
                    >= effective_thresholds[id(window)]
                    and (
                        has_structural_signal(str(window["chip"]))
                        or (
                            id(window) in expiring_windows
                            and metrics[str(window["chip"])] > 0
                        )
                    )
                ]
                chosen_window = max(
                    choices,
                    key=lambda window: metrics[str(window["chip"])]
                    / max(0.01, effective_thresholds[id(window)]),
                    default=None,
                )
                if chosen_window is not None:
                    chip_name = str(chosen_window["chip"])
                    no_chip_points = week_points
                    if chip_name == "Wildcard":
                        if week_number > 0:
                            hits = hits_before
                            hit_cost = hit_cost_before
                            hit_points_this_week = 0
                            transfers = transfers_before
                            weekly_changes[-1] = 0
                            free_transfers = (
                                free_transfers_before
                                if season in {"2024-25", "2025-26"}
                                else 1
                            )
                        available_budget = bank_before_transfers + sum(
                            selling_price(
                                int(state["purchase"]),
                                int(state["last_price"]),
                            )
                            for state in squad_before_transfers.values()
                        )
                        action_indices = initial_squad(
                            frame,
                            fresh_squad_scores,
                            budget_limit=available_budget,
                            excluded_elements=excluded_elements,
                            captain_weight=strategy.squad_captain_weight,
                            bench_weight=strategy.squad_bench_weight,
                            minimum_spend_gap=strategy.initial_spend_gap,
                            bench_premium_limit=strategy.bench_premium_limit,
                            bench_premium_penalty=strategy.bench_premium_penalty,
                            exact_optimiser=strategy.exact_initial_optimiser,
                            lineup_scores=fresh_objective_lineup_scores,
                            captain_utility_scores=fresh_objective_captain_scores,
                            bench_utility_scores=fresh_bench_scores,
                            risk_scores=risk_scores,
                            risk_aversion=strategy.squad_risk_aversion,
                            defence_correlation=strategy.defence_residual_correlation,
                        )
                        squad = squad_state(action_indices)
                        bank = available_budget - sum(
                            int(price_values[index]) for index in action_indices
                        )
                        assert_legal_squad(squad, bank, season, gw, "Wildcard")
                        xi, bench = choose_xi(
                            squad, row_by_element, scores, lineup_excluded_elements
                        )
                        captain_order = sorted(
                            xi,
                            key=lambda element: captain_metric[row_by_element[element]]
                            if element in row_by_element
                            and element not in lineup_excluded_elements
                            else -1.0,
                            reverse=True,
                        )
                        captain, vice = captain_order[:2]
                        base_breakdown = realised_week_breakdown(
                            xi,
                            bench,
                            captain,
                            vice,
                            squad,
                            row_by_element,
                            actual,
                            played_minutes,
                        )
                        scoring_squad = squad
                        scoring_xi = xi
                        scoring_bench = bench
                        scoring_captain = captain
                        scoring_vice = vice
                        week_points = base_breakdown["normal"]
                    elif chip_name == "Free Hit":
                        available_budget = bank_before_transfers + sum(
                            selling_price(
                                int(state["purchase"]),
                                int(state["last_price"]),
                            )
                            for state in squad_before_transfers.values()
                        )
                        action_indices = initial_squad(
                            frame,
                            scores,
                            budget_limit=available_budget,
                            excluded_elements=excluded_elements,
                            captain_weight=1.0,
                            bench_weight=0.08,
                            minimum_spend_gap=None,
                            bench_premium_limit=strategy.bench_premium_limit,
                            bench_premium_penalty=strategy.bench_premium_penalty,
                            exact_optimiser=strategy.exact_initial_optimiser,
                            lineup_scores=scores,
                            captain_utility_scores=fresh_captain_utility,
                            risk_scores=risk_scores,
                            risk_aversion=strategy.squad_risk_aversion,
                            defence_correlation=strategy.defence_residual_correlation,
                        )
                        action_state = squad_state(action_indices)
                        action_bank = available_budget - sum(
                            int(price_values[index]) for index in action_indices
                        )
                        assert_legal_squad(
                            action_state, action_bank, season, gw, "Free Hit"
                        )
                        action_xi, action_bench = choose_xi(
                            action_state, row_by_element, scores, lineup_excluded_elements
                        )
                        action_captain_order = sorted(
                            action_xi,
                            key=lambda element: captain_metric[row_by_element[element]]
                            if element in row_by_element
                            and element not in excluded_elements
                            else -1.0,
                            reverse=True,
                        )
                        action_captain, action_vice = action_captain_order[:2]
                        fresh_breakdown = realised_week_breakdown(
                            action_xi,
                            action_bench,
                            action_captain,
                            action_vice,
                            action_state,
                            row_by_element,
                            actual,
                            played_minutes,
                        )
                        scoring_squad = action_state
                        scoring_xi = action_xi
                        scoring_bench = action_bench
                        scoring_captain = action_captain
                        scoring_vice = action_vice
                        week_points = fresh_breakdown["normal"]
                        squad = squad_before_transfers
                        bank = bank_before_transfers
                        transfers = transfers_before
                        if week_number > 0:
                            hits = hits_before
                            hit_cost = hit_cost_before
                            hit_points_this_week = 0
                            if weekly_changes[-1] > 0:
                                rolled += 1
                            weekly_changes[-1] = 0
                            free_transfers = (
                                free_transfers_before
                                if season in {"2024-25", "2025-26"}
                                else 1
                            )
                    elif chip_name == "Bench Boost":
                        week_points = base_breakdown["bench_boost"]
                    elif chip_name == "Triple Captain":
                        week_points = base_breakdown["triple_captain"]
                    elif chip_name == "Assistant Manager":
                        if assistant_manager_option is None:
                            raise AssertionError(
                                f"{season} GW{gw}: Assistant Manager selected without a legal option"
                            )
                        (
                            assistant_manager_team,
                            assistant_manager_cost,
                            _,
                            manager_index,
                        ) = assistant_manager_option
                        bank -= assistant_manager_cost
                        manager_points = float(
                            assistant_manager_actual_values[manager_index]
                        )
                        week_points += manager_points
                        assistant_manager_remaining = 2
                    chosen_window["used"] = True
                    log_entry = {
                            "chip": chip_name,
                            "gw": gw,
                            "gain": round(float(week_points - no_chip_points)),
                            "signal": round(float(metrics[chip_name]), 3),
                            "threshold": round(float(effective_thresholds[id(chosen_window)]), 3),
                            "continuationValue": round(float(option_values[id(chosen_window)]), 3),
                            "reason": "signal beat option-value adjusted threshold",
                        }
                    if chip_name == "Assistant Manager":
                        log_entry["managerTeam"] = int(assistant_manager_team)
                        log_entry["cost"] = int(assistant_manager_cost)
                        assistant_manager_log = log_entry
                    chip_log.append(log_entry)

            if tracked_player_name:
                tracked_match = next(
                    (
                        int(index)
                        for index in frame_indices
                        if tracked_player_name.casefold()
                        in str(display_name_values[index]).casefold()
                    ),
                    None,
                )
                if (
                    tracked_match is not None
                    and fixture_counts[tracked_match] > 0
                    and int(element_values[tracked_match]) not in excluded_elements
                ):
                    tracked_element = int(element_values[tracked_match])
                    if tracked_counts["eligibleWeeks"] == 0:
                        tracked_counts["initialSquad"] = tracked_element in squad
                        tracked_counts["initialXi"] = tracked_element in xi
                        tracked_counts["initialCaptain"] = tracked_element == captain
                    tracked_counts["eligibleWeeks"] += 1
                    tracked_counts["squadWeeks"] += int(tracked_element in squad)
                    tracked_counts["xiWeeks"] += int(tracked_element in xi)
                    tracked_counts["captainWeeks"] += int(tracked_element == captain)
                    tracked_points = float(actual[tracked_match])
                    tracked_counts["eligiblePoints"] += tracked_points
                    tracked_counts["squadPoints"] += (
                        tracked_points if tracked_element in squad else 0.0
                    )
                    tracked_counts["xiPoints"] += (
                        tracked_points if tracked_element in xi else 0.0
                    )
                    tracked_counts["captainPoints"] += (
                        tracked_points if tracked_element == captain else 0.0
                    )

            if audit_selections:
                selection_log.append(
                    {
                        "gw": int(gw),
                        # Keep the persistent state separate from the one-week
                        # scoring state. A Free Hit must not look like fifteen
                        # permanent transfers in downstream decision audits.
                        "permanentSquad": sorted(int(element) for element in squad),
                        "squad": sorted(
                            int(element) for element in scoring_squad
                        ),
                        "xi": sorted(int(element) for element in scoring_xi),
                        "bench": [int(element) for element in scoring_bench],
                        "captain": int(scoring_captain),
                        "vice": int(scoring_vice),
                        "bank": int(bank),
                        "freeTransfersNext": int(free_transfers),
                        "persistentState": [
                            {
                                "element": int(element),
                                "position": int(state["position"]),
                                "team": int(state["team"]),
                                "purchase": int(state["purchase"]),
                                "lastPrice": int(
                                    price_values[row_by_element[element]]
                                    if element in row_by_element
                                    else state["last_price"]
                                ),
                                "salePrice": int(
                                    selling_price(
                                        int(state["purchase"]),
                                        int(
                                            price_values[row_by_element[element]]
                                            if element in row_by_element
                                            else state["last_price"]
                                        ),
                                    )
                                ),
                            }
                            for element, state in sorted(squad.items())
                        ],
                    }
                )

            current_position_floors = {
                position: min(
                    int(price_values[index])
                    for index in frame_indices
                    if int(position_values[index]) == position
                )
                for position in SQUAD_QUOTAS
            }
            _, persistent_bench = choose_xi(
                squad, row_by_element, scores, lineup_excluded_elements
            )
            active_bench_spend = 0
            active_bench_premium = 0
            for element in persistent_bench:
                state = squad[element]
                index = row_by_element.get(element)
                current_price = (
                    int(price_values[index])
                    if index is not None
                    else int(state["last_price"])
                )
                active_bench_spend += current_price
                active_bench_premium += max(
                    0,
                    current_price - current_position_floors[int(state["position"])],
                )
            squad_spends.append(
                sum(
                    int(price_values[row_by_element[element]])
                    if element in row_by_element
                    else int(state["last_price"])
                    for element, state in squad.items()
                )
            )
            bench_spends.append(active_bench_spend)
            bench_premiums.append(active_bench_premium)
            bank_history.append(int(bank))

            totals[season_id] += week_points
            weekly_totals.append(float(week_points))
            previous_gw = gw

        season_stats.append(
            {
                "season": season,
                "transfers": transfers,
                "hits": hits,
                "hitCost": hit_cost,
                "rolled": rolled,
                "weeksChanged": sum(change > 0 for change in weekly_changes[1:]),
                "gameweeks": len(weeks),
                "weeklyPoints": [round(value, 1) for value in weekly_totals],
                "chipOpportunities": chip_opportunities,
                "transferLog": transfer_log,
                "chips": chip_log,
                # This is deliberately local to the chip GW. Wildcards also alter
                # later transfers, so full chip value must be measured by a paired
                # season replay (as walk_forward does), not by summing this field.
                "immediateChipGain": int(sum(item["gain"] for item in chip_log)),
                "jointPreflightHolds": joint_preflight_holds,
                "unlimitedRebuilds": unlimited_rebuilds,
                "allocation": {
                    "initialSpend": round(squad_spends[0] / 10, 1),
                    "initialBank": round(bank_history[0] / 10, 1),
                    "initialBenchSpend": round(bench_spends[0] / 10, 1),
                    "initialBenchPremium": round(bench_premiums[0] / 10, 1),
                    "averageBenchSpend": round(float(np.mean(bench_spends)) / 10, 2),
                    "averageBenchPremium": round(float(np.mean(bench_premiums)) / 10, 2),
                    "averageBank": round(float(np.mean(bank_history)) / 10, 2),
                },
                "staleness": {
                    "averageGap": round(float(np.mean(staleness_gaps)), 3)
                    if staleness_gaps
                    else 0.0,
                    "maximumGap": round(float(np.max(staleness_gaps)), 3)
                    if staleness_gaps
                    else 0.0,
                    "triggeredWeeks": int(
                        sum(
                            gap >= strategy.staleness_gap_trigger
                            for gap in staleness_gaps
                        )
                    )
                    if strategy.staleness_gap_trigger is not None
                    else 0,
                },
                "initialSelection": initial_selection,
                "selectionLog": selection_log if audit_selections else None,
                "trackedPlayer": {
                    "name": tracked_player_name,
                    **tracked_counts,
                }
                if tracked_player_name
                else None,
            }
        )
    return totals, season_stats


def recursive_replay(
    data: pd.DataFrame,
    candidates: list[Candidate],
    strategy: SimulationStrategy,
    robust_planning: bool = False,
) -> np.ndarray:
    results = np.zeros((len(candidates), len(SEASONS)), dtype=float)
    for trial_index, candidate in enumerate(candidates):
        trial_scores, trial_plan_scores, _ = candidate_forecasts(
            data,
            candidate,
            robust_planning=robust_planning,
            schedule_censored=True,
        )
        results[trial_index], _ = simulate_candidate(
            data, trial_scores, strategy, plan_scores=trial_plan_scores
        )
        if (trial_index + 1) % 40 == 0 or trial_index + 1 == len(candidates):
            print(
                f"Recursive replay {trial_index + 1}/{len(candidates)} "
                f"({strategy.name})"
            )
    return results


def replay_chip_policies(
    data: pd.DataFrame,
    scores: np.ndarray,
    plan_scores: np.ndarray,
    policies: list[ChipPolicy],
    strategy: SimulationStrategy = WEEKLY_CHASE_STRATEGY,
) -> tuple[np.ndarray, list[list[dict]], dict[tuple[str, int], list[int]]]:
    fresh_squads = precompute_fresh_squads(data, plan_scores)
    free_hit_squads = precompute_fresh_squads(data, scores, one_week_only=True)
    results = np.zeros((len(policies), len(SEASONS)), dtype=float)
    stats: list[list[dict]] = []
    for policy_index, policy in enumerate(policies):
        totals, season_stats = simulate_candidate(
            data,
            scores,
            strategy,
            chip_policy=policy,
            fresh_squads=fresh_squads,
            free_hit_squads=free_hit_squads,
            plan_scores=plan_scores,
        )
        results[policy_index] = totals
        stats.append(season_stats)
        if (policy_index + 1) % 24 == 0 or policy_index + 1 == len(policies):
            print(f"Chip-policy replay {policy_index + 1}/{len(policies)}")
    return results, stats, fresh_squads


def add_rank_target_estimates(
    data: pd.DataFrame, backtest: list[dict]
) -> dict:
    """Attach empirically reconstructed top-500k cutoffs and uncertainty."""
    benchmark_path = ROOT / "analysis" / "data" / "historical_rank_benchmarks.json"
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmarks = {
        str(item["season"]): item for item in benchmark_payload["seasons"]
    }
    rng = np.random.default_rng(20260811)
    hit_count = 0
    probabilities: list[float] = []
    margins: list[int] = []
    estimated_ranks: list[int] = []
    rank_estimates_censored = 0
    for item in backtest:
        benchmark = benchmarks[str(item["season"])]
        target = int(benchmark["points"])
        weekly = np.asarray(item.pop("weeklyPoints"), dtype=float)
        # A moving-block bootstrap preserves short runs of fixture difficulty,
        # injuries and chip effects better than independently shuffling GWs.
        block_length = min(4, max(1, len(weekly)))
        samples = np.empty(4000, dtype=float)
        max_start = max(1, len(weekly) - block_length + 1)
        blocks_needed = math.ceil(len(weekly) / block_length)
        for sample_index in range(len(samples)):
            starts = rng.integers(0, max_start, size=blocks_needed)
            replay = np.concatenate(
                [weekly[start : start + block_length] for start in starts]
            )[: len(weekly)]
            samples[sample_index] = replay.sum()
        cutoff_samples = rng.integers(
            int(benchmark["p05"]), int(benchmark["p95"]) + 1, size=len(samples)
        )
        probability = float(np.mean(samples >= cutoff_samples))
        margin = int(item["points"] - target)
        rank_slope = float(benchmark["logRankSlope"])
        # The slope is fitted locally around rank 500k. Do not extrapolate it
        # hundreds of points into a part of the rank curve with no anchors.
        rank_is_local = abs(margin) <= 50
        estimated_rank = (
            max(
                1,
                int(round(500_000 * math.exp(rank_slope * margin))),
            )
            if rank_is_local
            else None
        )
        rank_scenarios = (
            [
                max(
                    1,
                    int(
                        round(
                            500_000
                            * math.exp(
                                rank_slope * (item["points"] - cutoff_point)
                            )
                        )
                    ),
                )
                for cutoff_point in (int(benchmark["p05"]), int(benchmark["p95"]))
            ]
            if rank_is_local
            else []
        )
        item["top500Target"] = target
        item["top500TargetInterval"] = [
            int(benchmark["p05"]),
            int(benchmark["p95"]),
        ]
        item["targetMargin"] = margin
        item["targetHit"] = margin >= 0
        item["targetProbability"] = round(probability * 100)
        item["estimatedRank"] = estimated_rank
        item["estimatedRankInterval"] = (
            [min(rank_scenarios), max(rank_scenarios)] if rank_scenarios else None
        )
        item["rankEstimateLocal"] = rank_is_local
        item["estimatedBand"] = (
            "Above top-500k cutoff"
            if margin >= 0
            else "Near top-500k cutoff"
            if margin >= -50
            else "Below locally calibrated rank range"
        )
        hit_count += int(margin >= 0)
        probabilities.append(probability)
        margins.append(margin)
        if estimated_rank is not None:
            estimated_ranks.append(estimated_rank)
        else:
            rank_estimates_censored += 1
    return {
        "target": "Top 500k",
        "hits": hit_count,
        "seasons": len(backtest),
        "hitRate": round(100 * hit_count / max(1, len(backtest))),
        "averageProbability": round(100 * float(np.mean(probabilities))),
        "averageMargin": round(float(np.mean(margins))),
        "worstMargin": min(margins) if margins else 0,
        "averageEstimatedRank": (
            round(float(np.mean(estimated_ranks)) / 1000) * 1000
            if estimated_ranks and rank_estimates_censored == 0
            else None
        ),
        "rankEstimateCoverage": len(estimated_ranks),
        "rankEstimateCensoredSeasons": rank_estimates_censored,
        "benchmarkSample": {
            "requestedHistories": benchmark_payload["requestedHistories"],
            "managerPopulation": benchmark_payload["managerPopulationAtSampling"],
            "source": benchmark_payload["source"],
        },
        "method": (
            "Empirical cutoff estimate from a deterministic sample of 5,000 public "
            "official FPL manager histories. A local log(rank)-points fit and nearest "
            "observed score boundary reconstruct each cutoff; its interval also allows "
            "for ties and survivorship. Probability uses 4,000 four-GW block-bootstrap "
            "model seasons. Rank is withheld outside a 50-point local calibration "
            "window rather than extrapolating the cutoff curve into unsupported ranks."
        ),
    }


def build_calibration_diagnostics(data: pd.DataFrame, backtest: list[dict]) -> dict:
    evaluation = data[
        data["season"].isin(EVALUATION_SEASONS) & data["fixture_count"].gt(0)
    ].copy()
    actual_return = (evaluation["points"] >= 5).astype(float)
    return_probability = evaluation["return5_probability"].clip(0, 1)
    return_brier = float(np.mean((return_probability - actual_return) ** 2))
    minute_brier = float(
        np.mean(
            (
                evaluation["sixty_probability"].clip(0, 1)
                - evaluation["sixty_observed_rate"].clip(0, 1)
            )
            ** 2
        )
    )
    clean_rows = evaluation[evaluation["position_id"].isin([1, 2])]
    clean_actual = (
        clean_rows["clean_sheets"] / clean_rows["fixture_count"].clip(lower=1)
    ).clip(0, 1)
    clean_brier = float(
        np.mean((clean_rows["team_clean_probability"].clip(0, 1) - clean_actual) ** 2)
    )
    interval_coverage = float(
        (
            (evaluation["points"] >= evaluation["prediction_p10"])
            & (evaluation["points"] <= evaluation["prediction_p90"])
        ).mean()
    )
    calibration_bins: list[dict] = []
    for lower in np.arange(0, 1, 0.1):
        upper = lower + 0.1
        mask = (return_probability >= lower) & (
            return_probability < upper if upper < 1 else return_probability <= upper
        )
        if not mask.any():
            continue
        calibration_bins.append(
            {
                "forecast": round(100 * float(return_probability[mask].mean())),
                "observed": round(100 * float(actual_return[mask].mean())),
                "players": int(mask.sum()),
            }
        )
    position_errors = []
    for position_id, frame in evaluation.groupby("position_id"):
        position_errors.append(
            {
                "position": POSITION_LABELS[int(position_id)],
                "mae": round(float((frame["component_xpts"] - frame["points"]).abs().mean()), 2),
                "returnBrier": round(
                    float(
                        np.mean(
                            (
                                frame["return5_probability"]
                                - (frame["points"] >= 5).astype(float)
                            )
                            ** 2
                        )
                    ),
                    3,
                ),
                "rows": int(len(frame)),
            }
        )
    role_errors = [
        {
            "role": str(role).replace("_", " ").title(),
            "mae": round(float((frame["component_xpts"] - frame["points"]).abs().mean()), 2),
            "challengerMae": round(float((frame["role_ridge_xpts"] - frame["points"]).abs().mean()), 2),
            "rows": int(len(frame)),
        }
        for role, frame in evaluation.groupby("player_role")
        if len(frame) >= 100
    ]
    event_coverage_by_season = [
        {
            "season": str(season).replace("-", "/"),
            "coverage": round(100 * float(frame["defensive_exact"].mean())),
        }
        for season, frame in evaluation[
            evaluation["position_id"].isin([2, 3, 4])
        ].groupby("season", sort=False)
    ]
    rotation_calibration = []
    for label, lower, upper in [
        ("Stable managers", 0.0, 0.18),
        ("Mixed rotation", 0.18, 0.30),
        ("Heavy rotation", 0.30, 1.0),
    ]:
        frame = evaluation[
            (evaluation["team_rotation_rate"] >= lower)
            & (evaluation["team_rotation_rate"] < upper)
        ]
        if len(frame):
            rotation_calibration.append(
                {
                    "segment": label,
                    "forecastStart": round(100 * float(frame["start_probability"].mean())),
                    "observedStart": round(100 * float(frame["start_observed_rate"].mean())),
                    "rows": int(len(frame)),
                }
            )
    weakest = sorted(backtest, key=lambda item: item["targetMargin"])[:3]
    return {
        "returnBrier": round(return_brier, 3),
        "minutes60Brier": round(minute_brier, 3),
        "cleanSheetBrier": round(clean_brier, 3),
        "p10P90Coverage": round(100 * interval_coverage),
        "mae": round(float((evaluation["component_xpts"] - evaluation["points"]).abs().mean()), 2),
        "defensiveEventCoverage": round(
            100
            * float(
                evaluation.loc[
                    evaluation["position_id"].isin([2, 3, 4]), "defensive_exact"
                ].mean()
            )
        ),
        "returnCalibration": calibration_bins,
        "positionErrors": position_errors,
        "roleErrors": sorted(role_errors, key=lambda item: item["mae"]),
        "eventCoverageBySeason": event_coverage_by_season,
        "rotationCalibration": rotation_calibration,
        "weakSeasons": [
            {
                "season": item["season"],
                "margin": item["targetMargin"],
                "points": item["points"],
                "diagnosis": (
                    "high-variance season: minutes and captain outcomes dominated"
                    if item["targetMargin"] <= -180
                    else "below target: transfer timing and player-return calibration"
                ),
            }
            for item in weakest
        ],
        "method": "Causal probability calibration on evaluation player-weeks; no future match enters a forecast bin.",
    }


def pick_squad(
    players: pd.DataFrame, budget_limit: int = 1000
) -> tuple[list[int], list[int]]:
    """Solve the legal squad, XI and captain jointly as an exact binary MILP."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    if players.empty:
        raise RuntimeError("Cannot optimise an empty live player pool")
    frame = players.copy()
    frame_indices = frame.index.to_numpy(int)
    count = len(frame)
    prices = frame["price"].to_numpy(float)
    positions = frame["position_id"].to_numpy(int)
    clubs = frame["team_id"].to_numpy(int)
    model_score = frame["model_score"].to_numpy(float)
    immediate = (
        frame["raw_projection"].to_numpy(float)
        if "raw_projection" in frame
        else 5 * model_score
    )
    weighted_games = (
        frame["horizon_weighted_games_censored"].to_numpy(float).clip(1, None)
        if "horizon_weighted_games_censored" in frame
        else np.full(count, 4.26)
    )
    horizon_per_game = (
        frame["risk_adjusted_horizon"].to_numpy(float) / weighted_games
        if "risk_adjusted_horizon" in frame
        else immediate
    )
    start_probability = frame.get("start_probability", pd.Series(1.0, index=frame.index)).to_numpy(float)
    play_probability = frame.get("play_probability", pd.Series(1.0, index=frame.index)).to_numpy(float)
    confidence = frame.get("confidence", pd.Series(70.0, index=frame.index)).to_numpy(float) / 100
    lineup_utility = (
        0.68 * immediate
        + 0.18 * horizon_per_game
        + 0.10 * model_score * 5
        + 0.04 * confidence * immediate
    )
    bench_utility = (
        0.045 * horizon_per_game
        + 0.055 * play_probability * np.minimum(immediate, 4.5)
    )
    # The captain contributes one extra copy of their own expected score, so this
    # block carries weight 1.0 on a points scale — not a rescaled rank.
    captain_utility = (
        frame["captain_score"].to_numpy(float)
        if "captain_score" in frame
        else 0.90 * immediate + 0.10 * model_score * 5
    )
    position_floors = {
        position: float(prices[positions == position].min())
        for position in SQUAD_QUOTAS
    }
    premiums = np.asarray(
        [max(0.0, price - position_floors[int(position)]) for price, position in zip(prices, positions)]
    )
    high_upside = immediate >= np.quantile(immediate, 0.95)
    standard_xi = (start_probability >= 0.70) & (play_probability >= 0.84)
    exceptional_xi = (
        ~standard_xi
        & (start_probability >= 0.70)
        & (play_probability >= 0.78)
        & high_upside
    )
    allowed_xi = standard_xi | exceptional_xi

    # Variable blocks are squad, XI and captain. Bench value and premium are
    # expressed as squad minus XI, keeping the entire objective linear.
    objective = -np.concatenate(
        [
            bench_utility - 0.018 * premiums,
            lineup_utility - bench_utility + 0.018 * premiums,
            captain_utility,
        ]
    )
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    budget = np.concatenate([prices, np.zeros(2 * count)])
    rows.append(budget)
    lower.append(max(0, budget_limit - 5))
    upper.append(budget_limit)
    squad_total = np.zeros(3 * count)
    squad_total[:count] = 1
    rows.append(squad_total)
    lower.append(15)
    upper.append(15)
    for position, quota in SQUAD_QUOTAS.items():
        row = np.zeros(3 * count)
        row[:count] = (positions == position).astype(float)
        rows.append(row)
        lower.append(quota)
        upper.append(quota)
    for club in np.unique(clubs):
        row = np.zeros(3 * count)
        row[:count] = (clubs == club).astype(float)
        rows.append(row)
        lower.append(0)
        upper.append(3)
    for local_index in range(count):
        xi_link = np.zeros(3 * count)
        xi_link[count + local_index] = 1
        xi_link[local_index] = -1
        rows.append(xi_link)
        lower.append(-np.inf)
        upper.append(0)
        captain_link = np.zeros(3 * count)
        captain_link[2 * count + local_index] = 1
        captain_link[count + local_index] = -1
        rows.append(captain_link)
        lower.append(-np.inf)
        upper.append(0)
    xi_total = np.zeros(3 * count)
    xi_total[count : 2 * count] = 1
    rows.append(xi_total)
    lower.append(11)
    upper.append(11)
    for position, minimum, maximum in ((1, 1, 1), (2, 3, 5), (3, 2, 5), (4, 1, 3)):
        row = np.zeros(3 * count)
        row[count : 2 * count] = (positions == position).astype(float)
        rows.append(row)
        lower.append(minimum)
        upper.append(maximum)
    captain_total = np.zeros(3 * count)
    captain_total[2 * count :] = 1
    rows.append(captain_total)
    lower.append(1)
    upper.append(1)
    bench_premium = np.concatenate([premiums, -premiums, np.zeros(count)])
    rows.append(bench_premium)
    lower.append(0)
    upper.append(20)
    exception_count = np.zeros(3 * count)
    exception_count[count : 2 * count] = exceptional_xi.astype(float)
    rows.append(exception_count)
    lower.append(0)
    upper.append(1)

    variable_upper = np.ones(3 * count)
    variable_upper[count : 2 * count] = allowed_xi.astype(float)
    variable_upper[2 * count :] = allowed_xi.astype(float)
    result = milp(
        c=objective,
        integrality=np.ones(3 * count),
        bounds=Bounds(np.zeros(3 * count), variable_upper),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Exact live squad MILP failed: {result.message}")
    chosen = frame_indices[np.flatnonzero(result.x[:count] > 0.5)].astype(int).tolist()
    xi = frame_indices[np.flatnonzero(result.x[count : 2 * count] > 0.5)].astype(int).tolist()
    return chosen, xi


def current_recommendation(
    historical: pd.DataFrame,
    best: Candidate,
    robust_planning: bool,
    strategy: SimulationStrategy,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], dict]:
    bootstrap = get_json(CURRENT_BOOTSTRAP)
    fixtures = get_json(CURRENT_FIXTURES)
    assert isinstance(bootstrap, dict) and isinstance(fixtures, list)
    teams = pd.DataFrame(bootstrap["teams"])
    current = pd.DataFrame(bootstrap["elements"])
    events = pd.DataFrame(bootstrap["events"])
    next_event = events.loc[~events["finished"].astype(bool)].iloc[0]
    gw_number = int(next_event["id"])
    deadline = str(next_event["deadline_time"])
    team_name = dict(zip(teams["id"], teams["short_name"]))
    team_full_name = dict(zip(teams["id"], teams["name"]))

    prior = historical[historical["season"] == "2025-26"].copy()
    raw_prior_path = CACHE / "2025-26" / "merged_gw.csv"
    raw_prior = pd.read_csv(raw_prior_path, encoding="latin-1", low_memory=False)
    prior_summary = (
        raw_prior.sort_values("GW")
        .groupby("element", as_index=False)
        .agg(
            previous_points=("total_points", "sum"),
            previous_minutes=("minutes", "sum"),
            previous_name=("name", "last"),
        )
    )
    players_2526 = pd.read_csv(
        CACHE / "2025-26" / "players_raw.csv", encoding="latin-1", low_memory=False
    )[["id", "code"]]
    prior_summary = prior_summary.merge(players_2526, left_on="element", right_on="id", how="left")
    prior_summary = prior_summary.rename(columns={"code": "player_code"})
    tails = (
        prior.sort_values("GW")
        .groupby("player_code", as_index=False)
        .agg(
            recent_raw=("recent_raw", "last"),
            long_raw=("long_raw", "last"),
            recent_underlying_raw=("recent_underlying_raw", "last"),
            long_underlying_raw=("long_underlying_raw", "last"),
            minutes_security_raw=("minutes_security_raw", "last"),
            start_probability=("start_probability", "last"),
            sub_probability_given_bench=("sub_probability_given_bench", "last"),
            sixty_probability_given_start=("sixty_probability_given_start", "last"),
            minutes_if_start=("minutes_if_start", "last"),
            minutes_if_bench=("minutes_if_bench", "last"),
            minutes_std_prior=("minutes_std", "last"),
            rotation_volatility=("rotation_volatility", "last"),
            defensive_rate_prior=("defensive_rate", "last"),
            bps_rate_prior=("bps_rate", "last"),
            goal_rate_prior=("goal_rate", "last"),
            assist_rate_prior=("assist_rate", "last"),
            bonus_rate_prior=("bonus_rate", "last"),
            save_rate_prior=("save_rate", "last"),
            clean_sheet_rate_prior=("clean_sheet_rate", "last"),
            ensemble_structural_weight=("ensemble_structural_weight", "last"),
            ensemble_empirical_weight=("ensemble_empirical_weight", "last"),
            ensemble_market_weight=("ensemble_market_weight", "last"),
            ensemble_role_weight=("ensemble_role_weight", "last"),
            defensive_event_coverage_prior=("defensive_event_coverage", "last"),
        )
    )
    prior_summary = prior_summary.merge(tails, on="player_code", how="left")

    first_fixtures = pd.DataFrame(fixtures)
    first_fixtures = first_fixtures[first_fixtures["event"] == gw_number]
    first_fixtures = first_fixtures.sort_values("kickoff_time", kind="stable")
    # A club can hold two fixtures in one Gameweek. Keep every one of them:
    # a single dict entry per club silently discarded the second match, so
    # opponent, venue, expected goals, clean-sheet probability and the market
    # join all described one arbitrary half of a Double Gameweek.
    fixtures_by_team: dict[int, list[dict]] = {}
    for _, fixture in first_fixtures.iterrows():
        fixtures_by_team.setdefault(int(fixture["team_h"]), []).append(
            {
                "fixture_id": int(fixture["id"]),
                "opponent": int(fixture["team_a"]),
                "home": True,
                "kickoff": fixture["kickoff_time"],
            }
        )
        fixtures_by_team.setdefault(int(fixture["team_a"]), []).append(
            {
                "fixture_id": int(fixture["id"]),
                "opponent": int(fixture["team_h"]),
                "home": False,
                "kickoff": fixture["kickoff_time"],
            }
        )
    # The single-fixture view is retained only for labelling (opponent badge,
    # venue, kickoff). It is the earliest match, not an arbitrary one.
    fixture_map: dict[int, dict] = {
        team_id: entries[0] for team_id, entries in fixtures_by_team.items()
    }
    fixture_counts_by_team: dict[int, int] = {
        team_id: len(entries) for team_id, entries in fixtures_by_team.items()
    }

    played_history = historical[
        (historical["minutes"] > 0) & historical["player_code"].notna()
    ].copy()
    general_history = (
        played_history.groupby("player_code", as_index=False)
        .agg(
            history_matches=("points", "size"),
            history_points=("points", "sum"),
            history_minutes=("minutes", "sum"),
            history_average=("points", "mean"),
            history_volatility=("points", "std"),
            history_returns=("points", lambda values: float((values >= 5).mean())),
        )
    )
    general_history["history_per90"] = (
        90 * general_history["history_points"]
        / general_history["history_minutes"].clip(lower=1)
    )
    opponent_history = (
        played_history.dropna(subset=["opponent_name"])
        .groupby(["player_code", "opponent_name"], as_index=False)
        .agg(
            opponent_matches=("points", "size"),
            opponent_points=("points", "sum"),
            opponent_minutes=("minutes", "sum"),
            opponent_average=("points", "mean"),
            opponent_returns=("points", lambda values: float((values >= 5).mean())),
        )
    )
    opponent_history["opponent_per90"] = (
        90 * opponent_history["opponent_points"]
        / opponent_history["opponent_minutes"].clip(lower=1)
    )
    opponent_profiles = (
        historical.sort_values(["season_order", "GW"])
        .dropna(subset=["opponent_name"])
        .groupby(["opponent_name", "position_id"], as_index=False)
        .tail(1)[
            [
                "opponent_name",
                "position_id",
                "opponent_goal_vulnerability",
                "opponent_assist_vulnerability",
                "opponent_xg_vulnerability",
            ]
        ]
    )
    opponent_profiles = opponent_profiles.rename(
        columns={"opponent_name": "opponent_full_name"}
    )
    team_profiles = historical[
        [
            "season_order",
            "GW",
            "team_name",
            "team_attack_rating",
            "team_defence_rating",
            "team_form_rating",
            "team_clean_rating",
            "team_rating_confidence",
            "team_regime_shift",
            "team_rotation_rate",
        ]
    ].drop_duplicates(["season_order", "GW", "team_name"]).copy()
    team_profiles["team_key"] = (
        team_profiles["team_name"].fillna("").str.lower().str.replace(
            r"[^a-z0-9]", "", regex=True
        )
    )
    team_profiles.sort_values(["season_order", "GW"], inplace=True)
    team_profiles = team_profiles.groupby("team_key", as_index=False).tail(1)

    current = current.merge(
        prior_summary[
            [
                "player_code",
                "previous_points",
                "previous_minutes",
                "recent_raw",
                "long_raw",
                "recent_underlying_raw",
                "long_underlying_raw",
                "minutes_security_raw",
                "start_probability",
                "sub_probability_given_bench",
                "sixty_probability_given_start",
                "minutes_if_start",
                "minutes_if_bench",
                "minutes_std_prior",
                "rotation_volatility",
                "defensive_rate_prior",
                "bps_rate_prior",
                "goal_rate_prior",
                "assist_rate_prior",
                "bonus_rate_prior",
                "save_rate_prior",
                "clean_sheet_rate_prior",
                "ensemble_structural_weight",
                "ensemble_empirical_weight",
                "ensemble_market_weight",
                "ensemble_role_weight",
                "defensive_event_coverage_prior",
            ]
        ],
        left_on="code",
        right_on="player_code",
        how="left",
    )
    current["position_id"] = current["element_type"].astype(int)
    current["team_id"] = current["team"].astype(int)
    # Needed before the projection is built: the per-match routes are scaled by
    # this once, so a Double Gameweek is worth two matches rather than one.
    current["fixture_count"] = (
        current["team_id"].map(fixture_counts_by_team).fillna(0).astype(int)
    )
    current["team_name"] = current["team_id"].map(team_name)
    current["team_full_name"] = current["team_id"].map(team_full_name)
    current["team_key"] = (
        current["team_full_name"].fillna("").str.lower().str.replace(
            r"[^a-z0-9]", "", regex=True
        )
    )
    current = current.merge(
        team_profiles[
            [
                "team_key",
                "team_attack_rating",
                "team_defence_rating",
                "team_form_rating",
                "team_clean_rating",
                "team_rating_confidence",
                "team_regime_shift",
                "team_rotation_rate",
            ]
        ],
        on="team_key",
        how="left",
    )
    current["opponent_full_name"] = current["team_id"].map(
        lambda team_id: team_full_name.get(
            fixture_map.get(int(team_id), {}).get("opponent"), "TBD"
        )
    )
    current = current.merge(general_history, on="player_code", how="left")
    current = current.merge(
        opponent_history,
        left_on=["player_code", "opponent_full_name"],
        right_on=["player_code", "opponent_name"],
        how="left",
    )
    current = current.merge(
        opponent_profiles,
        on=["opponent_full_name", "position_id"],
        how="left",
    )
    for column in (
        "opponent_goal_vulnerability",
        "opponent_assist_vulnerability",
        "opponent_xg_vulnerability",
    ):
        position_median = current.groupby("position_id")[column].transform("median")
        historical_median = historical.groupby("position_id")[column].median()
        current[column] = (
            current[column]
            .fillna(position_median)
            .fillna(current["position_id"].map(historical_median))
            .fillna(0.0)
        )
    current["display_name"] = current["web_name"].astype(str)
    current["display_name"] = current["display_name"].str.replace(
        f"Gu{chr(0xFFFD)}hi", "Guehi", regex=False
    )
    current["price"] = current["now_cost"].astype(int)
    current["ownership"] = pd.to_numeric(current["selected_by_percent"], errors="coerce").fillna(0)
    current["ep_next_num"] = pd.to_numeric(current["ep_next"], errors="coerce").fillna(0)
    fallback = current["ep_next_num"].where(current["ep_next_num"] > 0, 2.0)
    current["long_raw"] = current["long_raw"].fillna(fallback)
    current["recent_raw"] = current["recent_raw"].fillna(current["long_raw"])
    current["minutes_security_raw"] = current["minutes_security_raw"].fillna(
        (current["previous_minutes"].fillna(0) / (38 * 90)).clip(0, 1)
    )
    current["minutes_security_raw"] = current["minutes_security_raw"].where(
        current["minutes_security_raw"] > 0,
        (current["ep_next_num"] / 4.5).clip(0.25, 0.85),
    )
    current_ict = pd.to_numeric(current["ict_index"], errors="coerce").fillna(0)
    current_appearances = (
        pd.to_numeric(current["minutes"], errors="coerce").fillna(0) / 90
    ).clip(lower=1)
    underlying_fallback = (current_ict / current_appearances).clip(0, 35)
    current["long_underlying_raw"] = current["long_underlying_raw"].fillna(
        underlying_fallback
    )
    current["recent_underlying_raw"] = current["recent_underlying_raw"].fillna(
        current["long_underlying_raw"]
    )
    current["long_value_raw"] = current["long_raw"] / (current["price"] / 10).clip(3.5)
    current["recent_value_raw"] = current["recent_raw"] / (current["price"] / 10).clip(3.5)
    season_start = date(2026, 8, 1)
    current["age"] = current["birth_date"].map(
        lambda value: (
            (season_start - parse_dob(value)).days / 365.2425
            if parse_dob(value)
            else 27.5
        )
    )
    current["age_raw"] = np.exp(-((current["age"] - 27.5) / 7.5) ** 2)
    current["crowd_raw"] = np.log1p(current["ownership"])
    horizon_fixtures = pd.DataFrame(fixtures)
    horizon_fixtures = horizon_fixtures[
        horizon_fixtures["event"].between(gw_number, gw_number + 5)
    ]
    horizon_map: dict[int, list[tuple[int, bool, float, int]]] = {}
    horizon_weight = {
        gw_number + offset: weight
        for offset, weight in enumerate((1.0, 0.86, 0.74, 0.64, 0.55, 0.47))
    }
    for _, fixture in horizon_fixtures.iterrows():
        event = int(fixture["event"])
        home_team = int(fixture["team_h"])
        away = int(fixture["team_a"])
        horizon_map.setdefault(home_team, []).append(
            (away, True, horizon_weight[event], event)
        )
        horizon_map.setdefault(away, []).append(
            (home_team, False, horizon_weight[event], event)
        )
    league_goal_rate = 1.40
    current["team_attack_rating"] = current["team_attack_rating"].fillna(
        league_goal_rate
    )
    current["team_defence_rating"] = current["team_defence_rating"].fillna(
        league_goal_rate
    )
    current["team_form_rating"] = current["team_form_rating"].fillna(1.35)
    current["team_clean_rating"] = current["team_clean_rating"].fillna(0.28)
    current["team_rating_confidence"] = current["team_rating_confidence"].fillna(0)
    current["team_regime_shift"] = current["team_regime_shift"].fillna(0)
    current["team_rotation_rate"] = current["team_rotation_rate"].fillna(0.22).clip(0, 0.75)

    # Blend the carried historical profile with a public Opta season prior,
    # then shrink it for verified manager, transfer and promotion regimes. The
    # prior affects intrinsic strength and uncertainty; it never directly
    # awards points for a particular fixture.
    team_prior_payload = load_team_priors()
    prior_by_team = {
        normalize_external_team(name): value
        for name, value in team_prior_payload.get("teams", {}).items()
    }
    known_probabilities = [
        float(value["optaTopFiveProbability"])
        for value in prior_by_team.values()
        if value.get("optaTopFiveProbability") is not None
    ]
    logit_centre = float(
        np.mean([math.log(value / (1 - value)) for value in known_probabilities])
    )
    team_anchor_details: dict[int, dict] = {}
    for team_id in sorted(current["team_id"].astype(int).unique()):
        mask = current["team_id"].eq(team_id)
        full_name = str(team_full_name[int(team_id)])
        prior_row = prior_by_team.get(normalize_external_team(full_name), {})
        probability = prior_row.get("optaTopFiveProbability")
        strength_index = 1.0
        if probability is not None:
            probability = float(np.clip(float(probability), 0.01, 0.99))
            strength_index = float(
                np.clip(
                    math.exp(0.18 * (math.log(probability / (1 - probability)) - logit_centre)),
                    0.78,
                    1.28,
                )
            )
        manager_change = float(prior_row.get("managerChange", 0))
        key_exit = float(prior_row.get("keyExitSeverity", 0))
        promoted = float(bool(prior_row.get("promoted", False)))
        european_load = float(prior_row.get("europeanLoad", 0))
        regime_prior = float(
            np.clip(
                1
                - (1 - manager_change)
                * (1 - key_exit)
                * (1 - 0.58 * promoted)
                * (1 - 0.45 * european_load),
                0,
                0.88,
            )
        )
        historical_attack = float(current.loc[mask, "team_attack_rating"].iloc[0])
        historical_defence = float(current.loc[mask, "team_defence_rating"].iloc[0])
        anchor_attack = league_goal_rate * strength_index
        anchor_defence = league_goal_rate / strength_index
        opta_weight = 0.48 if probability is not None else 0.0
        anchored_attack = (1 - opta_weight) * historical_attack + opta_weight * anchor_attack
        anchored_defence = (1 - opta_weight) * historical_defence + opta_weight * anchor_defence
        carry_weight = float(np.clip(1 - 0.62 * regime_prior, 0.35, 1.0))
        regime_target_attack = anchor_attack if probability is not None else league_goal_rate
        regime_target_defence = anchor_defence if probability is not None else league_goal_rate
        final_attack = carry_weight * anchored_attack + (1 - carry_weight) * regime_target_attack
        final_defence = carry_weight * anchored_defence + (1 - carry_weight) * regime_target_defence
        prior_confidence = float(current.loc[mask, "team_rating_confidence"].iloc[0])
        confidence_cap = float(prior_row.get("confidenceCap", 0.88))
        final_confidence = min(
            prior_confidence * (1 - 0.58 * regime_prior), confidence_cap
        )
        current.loc[mask, "team_attack_rating"] = np.clip(final_attack, 0.55, 2.55)
        current.loc[mask, "team_defence_rating"] = np.clip(final_defence, 0.55, 2.55)
        current.loc[mask, "team_form_rating"] = (
            carry_weight * current.loc[mask, "team_form_rating"]
            + (1 - carry_weight) * 1.35
        )
        current.loc[mask, "team_rating_confidence"] = final_confidence
        current.loc[mask, "team_regime_shift"] = np.maximum(
            current.loc[mask, "team_regime_shift"], regime_prior
        )
        current.loc[mask, "team_rotation_rate"] = np.clip(
            current.loc[mask, "team_rotation_rate"]
            + 0.10 * manager_change
            + 0.08 * european_load,
            0,
            0.75,
        )
        team_anchor_details[int(team_id)] = {
            "team": str(team_name[int(team_id)]),
            "optaTopFiveProbability": probability,
            "strengthIndex": round(strength_index, 4),
            "regimePrior": round(regime_prior, 4),
            "ratingConfidence": round(final_confidence, 4),
            "managerChange": manager_change,
            "keyExitSeverity": key_exit,
            "promoted": bool(promoted),
            "europeanLoad": european_load,
        }

    team_snapshot = current[
        [
            "team_id",
            "team_attack_rating",
            "team_defence_rating",
            "team_form_rating",
            "team_clean_rating",
            "team_rating_confidence",
        ]
    ].drop_duplicates("team_id").set_index("team_id")

    def opponent_rating(team_id: int, column: str, default: float) -> float:
        opponent_id = fixture_map.get(int(team_id), {}).get("opponent")
        if opponent_id not in team_snapshot.index:
            return default
        value = team_snapshot.loc[int(opponent_id), column]
        return default if pd.isna(value) else float(value)

    current["opponent_attack_rating"] = current["team_id"].map(
        lambda team_id: opponent_rating(team_id, "team_attack_rating", league_goal_rate)
    )
    current["opponent_defence_rating"] = current["team_id"].map(
        lambda team_id: opponent_rating(team_id, "team_defence_rating", league_goal_rate)
    )
    current["opponent_clean_rating"] = current["team_id"].map(
        lambda team_id: opponent_rating(team_id, "team_clean_rating", 0.28)
    )
    current["league_goal_rate"] = league_goal_rate
    current["was_home"] = current["team_id"].map(
        lambda team_id: bool(fixture_map.get(int(team_id), {}).get("home", False))
    )
    # A current league-table goal-difference snapshot is not available in the
    # bootstrap endpoint.  Zero is the neutral training value and avoids
    # introducing a result-derived proxy at the live deadline.
    current["table_goal_difference_before"] = 0.0

    def match_rates(team_id: int, opponent_id: int, home: bool) -> tuple[float, float, float]:
        team_row = team_snapshot.loc[int(team_id)]
        opponent_row = team_snapshot.loc[int(opponent_id)]
        expected_against = float(
            league_goal_rate
            * (float(team_row["team_defence_rating"]) / league_goal_rate) ** 0.70
            * (float(opponent_row["team_attack_rating"]) / league_goal_rate) ** 0.70
            * (0.88 if home else 1.12)
        )
        expected_for = float(
            league_goal_rate
            * (float(team_row["team_attack_rating"]) / league_goal_rate) ** 0.70
            * (float(opponent_row["team_defence_rating"]) / league_goal_rate) ** 0.70
            * (1.12 if home else 0.88)
        )
        expected_against = float(np.clip(expected_against, 0.30, 3.40))
        expected_for = float(np.clip(expected_for, 0.30, 3.40))
        return expected_for, expected_against, float(np.exp(-expected_against))

    def mean_rates(
        samples: list[tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        """Average per-match rates over a club's fixtures in this Gameweek.

        Every scoring route the projection builds from these is linear in the
        per-match rate, so the mean rate multiplied by the fixture count equals
        the correct Double Gameweek total.
        """
        return (
            float(np.mean([sample[0] for sample in samples])),
            float(np.mean([sample[1] for sample in samples])),
            float(np.mean([sample[2] for sample in samples])),
        )

    internal_fixture_rates: dict[int, list[tuple[float, float, float]]] = {}
    for team_id, entries in fixtures_by_team.items():
        internal_fixture_rates[int(team_id)] = [
            match_rates(int(team_id), int(entry["opponent"]), bool(entry["home"]))
            for entry in entries
        ]
    internal_rates: dict[int, tuple[float, float, float]] = {
        team_id: mean_rates(samples)
        for team_id, samples in internal_fixture_rates.items()
    }
    expected_market_fixtures = [
        (team_full_name[int(row.team_h)], team_full_name[int(row.team_a)])
        for row in first_fixtures[["team_h", "team_a"]].itertuples(index=False)
    ]
    matchbook_payload = fetch_matchbook_signals(expected_market_fixtures)
    matchbook_lookup = external_fixture_lookup(matchbook_payload)
    opta_fixture_payload = load_opta_fixture_predictions()
    opta_fixture_lookup = opta_fixture_payload.get("lookup", {})
    immediate_fixture_rates: dict[int, list[tuple[float, float, float]]] = {}
    market_team_detail: dict[int, dict] = {}
    for fixture in first_fixtures.itertuples(index=False):
        home_team = int(fixture.team_h)
        away_team = int(fixture.team_a)
        key = (
            normalize_external_team(team_full_name[home_team]),
            normalize_external_team(team_full_name[away_team]),
        )
        market = matchbook_lookup.get(key)
        opta_fixture = opta_fixture_lookup.get(key)
        # Rates for *this* match, not the club's Gameweek average, so a market
        # price is blended against the fixture it actually refers to.
        internal_home = match_rates(home_team, away_team, True)
        internal_away = match_rates(away_team, home_team, False)
        internal_outcomes = external_poisson_outcomes(
            internal_home[0], internal_away[0]
        )
        home_fixture_rates = internal_home
        away_fixture_rates = internal_away
        market_weight = (
            float(np.clip(0.92 * float(market["quality"]), 0.0, 0.82))
            if market
            else 0.0
        )
        if market or opta_fixture:
            if market and opta_fixture:
                external_home_probability = (
                    0.70 * float(market["homeProbability"])
                    + 0.30 * float(opta_fixture["homeProbability"])
                )
                external_draw_probability = (
                    0.70 * float(market["drawProbability"])
                    + 0.30 * float(opta_fixture["drawProbability"])
                )
                external_away_probability = (
                    0.70 * float(market["awayProbability"])
                    + 0.30 * float(opta_fixture["awayProbability"])
                )
                market_home_for, market_away_for = external_implied_goal_rates(
                    external_home_probability,
                    external_draw_probability,
                    external_away_probability,
                    market.get("over25Probability"),
                )
            elif market:
                market_home_for = float(market["homeExpectedGoals"])
                market_away_for = float(market["awayExpectedGoals"])
            else:
                market_home_for, market_away_for = external_implied_goal_rates(
                    float(opta_fixture["homeProbability"]),
                    float(opta_fixture["drawProbability"]),
                    float(opta_fixture["awayProbability"]),
                    None,
                )
            home_for = (1 - market_weight) * internal_home[0] + market_weight * market_home_for
            away_for = (1 - market_weight) * internal_away[0] + market_weight * market_away_for
            home_fixture_rates = (home_for, away_for, float(np.exp(-away_for)))
            away_fixture_rates = (away_for, home_for, float(np.exp(-home_for)))
        immediate_fixture_rates.setdefault(home_team, []).append(home_fixture_rates)
        immediate_fixture_rates.setdefault(away_team, []).append(away_fixture_rates)
        for active_team, is_home in ((home_team, True), (away_team, False)):
            market_win = (
                float(market["homeProbability"] if is_home else market["awayProbability"])
                if market
                else None
            )
            opta_win = (
                float(
                    opta_fixture[
                        "homeProbability" if is_home else "awayProbability"
                    ]
                )
                if opta_fixture
                else None
            )
            external_win = (
                0.70 * market_win + 0.30 * opta_win
                if market_win is not None and opta_win is not None
                else market_win
                if market_win is not None
                else opta_win
            )
            model_win = float(internal_outcomes[0] if is_home else internal_outcomes[2])
            # Fixtures are ordered by kickoff, so in a Double Gameweek this
            # reports the market view of the same match the fixture label shows.
            market_team_detail.setdefault(active_team, {
                "covered": bool(market),
                "marketWeight": round(market_weight, 4),
                "modelWinProbability": round(model_win, 4),
                "marketWinProbability": round(market_win, 4) if market_win is not None else None,
                "optaWinProbability": round(opta_win, 4) if opta_win is not None else None,
                "externalWinProbability": round(external_win, 4) if external_win is not None else None,
                "winProbabilityGap": (
                    round(model_win - market_win, 4) if market_win is not None else None
                ),
                "externalProbabilityGap": (
                    round(model_win - external_win, 4)
                    if external_win is not None
                    else None
                ),
                "quality": round(float(market["quality"]), 4) if market else 0.0,
                "volume": round(float(market["matchVolume"]), 2) if market else 0.0,
            })
    immediate_rates: dict[int, tuple[float, float, float]] = {
        team_id: mean_rates(samples)
        for team_id, samples in immediate_fixture_rates.items()
    }
    for team_id, fallback in internal_rates.items():
        immediate_rates.setdefault(int(team_id), fallback)
    current["team_expected_goals_for"] = current["team_id"].map(
        lambda team_id: immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))[0]
    )
    current["team_expected_goals_against"] = current["team_id"].map(
        lambda team_id: immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))[1]
    )
    current["team_clean_probability"] = current["team_id"].map(
        lambda team_id: immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))[2]
    )

    horizon_rates: dict[int, tuple[float, float, float]] = {}
    for team_id, values in horizon_map.items():
        weighted_for = 0.0
        weighted_against = 0.0
        weighted_clean = 0.0
        total_weight = 0.0
        for opponent_id, is_home, weight, event in values:
            if event == gw_number and int(team_id) in immediate_rates:
                expected_for, expected_against, clean_probability = immediate_rates[int(team_id)]
            else:
                expected_for, expected_against, clean_probability = match_rates(
                    int(team_id), int(opponent_id), bool(is_home)
                )
            weighted_for += weight * expected_for
            weighted_against += weight * expected_against
            weighted_clean += weight * clean_probability
            total_weight += weight
        if total_weight > 0:
            horizon_rates[int(team_id)] = (
                weighted_for / total_weight,
                weighted_against / total_weight,
                weighted_clean / total_weight,
            )
    current["team_horizon_expected_goals_for"] = current["team_id"].map(
        lambda team_id: horizon_rates.get(
            int(team_id), immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))
        )[0]
    )
    current["team_horizon_expected_goals_against"] = current["team_id"].map(
        lambda team_id: horizon_rates.get(
            int(team_id), immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))
        )[1]
    )
    current["team_horizon_clean_probability"] = current["team_id"].map(
        lambda team_id: horizon_rates.get(
            int(team_id), immediate_rates.get(int(team_id), (1.4, 1.4, 0.25))
        )[2]
    )
    attack_index = current["team_attack_rating"] / league_goal_rate
    defence_index = league_goal_rate / current["team_defence_rating"].clip(lower=0.35)
    current["team_context_raw"] = (
        0.43 * attack_index
        + 0.43 * defence_index
        + 0.14 * current["team_form_rating"] / 1.35
    ).clip(0.35, 2.75)
    current["team_defence_raw"] = (
        league_goal_rate / current["team_defence_rating"].clip(lower=0.35)
    ).clip(0.30, 3.0)
    current["team_attack_raw"] = (
        current["team_attack_rating"] / league_goal_rate
    ).clip(0.30, 3.0)
    neutral_clean = math.exp(-league_goal_rate)

    def fixture_value(row: pd.Series, horizon: bool = False) -> float:
        if horizon:
            expected_for = float(row["team_horizon_expected_goals_for"])
            clean_probability = float(row["team_horizon_clean_probability"])
        else:
            expected_for = float(row["team_expected_goals_for"])
            clean_probability = float(row["team_clean_probability"])
        attack = expected_for / league_goal_rate
        defence = clean_probability / neutral_clean
        position = int(row["position_id"])
        if position in (1, 2):
            return 0.34 * attack + 0.66 * defence
        if position == 3:
            return 0.84 * attack + 0.16 * defence
        return attack

    current["fixture_raw"] = current.apply(fixture_value, axis=1)
    current["fixture_horizon_raw"] = current.apply(
        lambda row: fixture_value(row, horizon=True), axis=1
    )
    team_match_context = (
        current[
            [
                "team_name",
                "team_attack_rating",
                "team_defence_rating",
            ]
        ]
        .drop_duplicates("team_name")
        .copy()
    )
    team_match_context["team_attack_rank"] = team_match_context[
        "team_attack_rating"
    ].rank(method="min", ascending=False)
    team_match_context["team_defence_rank"] = team_match_context[
        "team_defence_rating"
    ].rank(method="min", ascending=True)
    team_match_context["team_strength_rank"] = (
        team_match_context["team_attack_rating"]
        / team_match_context["team_defence_rating"].clip(lower=0.25)
    ).rank(method="min", ascending=False)
    current = current.merge(
        team_match_context[
            ["team_name", "team_attack_rank", "team_defence_rank", "team_strength_rank"]
        ],
        on="team_name",
        how="left",
    )
    for raw_name, rank_name in [
        ("long_raw", "long"),
        ("recent_raw", "recent"),
        ("long_value_raw", "long_value"),
        ("recent_value_raw", "recent_value"),
        ("age_raw", "age_score"),
        ("fixture_horizon_raw", "fixture"),
        ("fixture_raw", "fixture_now"),
        ("team_context_raw", "team_context"),
        ("team_defence_raw", "team_defence"),
        ("team_attack_raw", "team_attack"),
        ("crowd_raw", "crowd"),
        ("minutes_security_raw", "minutes_security"),
        ("long_underlying_raw", "long_underlying"),
        ("recent_underlying_raw", "recent_underlying"),
    ]:
        current[rank_name] = current.groupby("position_id")[raw_name].transform(percentile)
    # Historical fixture horizons are censored because announcement snapshots
    # are unavailable. The live path can safely use the official schedule that
    # is actually published at this deadline, while retaining the same feature
    # schema as the audited model.
    current["fixture_censored"] = current["fixture"]
    matrix = feature_matrix(current)
    non_match_coefficients = best.coefficients.copy()
    non_match_coefficients[5:7] = 0.0
    non_match_total = float(non_match_coefficients.sum())
    if non_match_total > 0:
        non_match_coefficients /= non_match_total
    current["model_score"] = matrix @ non_match_coefficients
    current["availability"] = official_availability_chance(
        current["chance_of_playing_next_round"], current["status"]
    )
    current_minutes = pd.to_numeric(current["minutes"], errors="coerce").fillna(0)
    previous_minutes = current_minutes.where(
        current_minutes > 0, current["previous_minutes"].fillna(0)
    )
    nineties = (previous_minutes / 90).clip(lower=0)
    current["sample_nineties"] = nineties
    rate_denominator = nineties + 5.0

    def numeric_current(column: str) -> pd.Series:
        if column not in current:
            return pd.Series(0.0, index=current.index)
        return pd.to_numeric(current[column], errors="coerce").fillna(0.0)

    current = current.copy()

    prior_start = current["start_probability"].fillna(
        current["position_id"].map({1: 0.68, 2: 0.58, 3: 0.56, 4: 0.54})
    )
    prior_sub = current["sub_probability_given_bench"].fillna(
        current["position_id"].map({1: 0.05, 2: 0.30, 3: 0.42, 4: 0.43})
    )
    prior_sixty_start = current["sixty_probability_given_start"].fillna(
        current["position_id"].map({1: 0.95, 2: 0.82, 3: 0.76, 4: 0.72})
    )
    congestion_map: dict[int, float] = {}
    for team_id in current["team_id"].astype(int).unique():
        team_schedule = horizon_fixtures[
            (horizon_fixtures["team_h"] == team_id)
            | (horizon_fixtures["team_a"] == team_id)
        ].copy()
        kickoffs = pd.to_datetime(
            team_schedule["kickoff_time"], errors="coerce", utc=True
        ).dropna().sort_values()
        gaps = kickoffs.diff().dt.total_seconds().dropna() / 86400
        congestion_map[int(team_id)] = float(gaps.min()) if len(gaps) else 7.0
    current["minimum_fixture_gap"] = current["team_id"].map(congestion_map).fillna(7)
    viable_starter = (prior_start >= 0.28).astype(float)
    competition_count = viable_starter.groupby(
        [current["team_id"], current["position_id"]]
    ).transform("sum")
    role_slots = current["position_id"].map({1: 1.0, 2: 4.0, 3: 4.0, 4: 2.0})
    current["competition_pressure"] = (
        (competition_count - role_slots).clip(lower=0) / role_slots
    ).clip(0, 1.5)
    completed_rounds = max(0, gw_number - 1)
    season_starts = numeric_current("starts").clip(0, completed_rounds)
    current["start_probability"] = (
        6 * prior_start + season_starts
    ) / (6 + completed_rounds)
    current["start_probability"] *= (
        current["availability"] / 100
    ).clip(0, 1)
    live_rotation_volatility = current["rotation_volatility"].fillna(0.35).clip(0, 1)
    current["rotation_volatility"] = live_rotation_volatility
    current["start_probability"] *= 1 - (
        (current["team_rotation_rate"] - 0.18).clip(lower=0)
        * live_rotation_volatility
        * 0.35
        + 0.045
        * current["competition_pressure"]
        * (0.35 + live_rotation_volatility)
        + ((4.5 - current["minimum_fixture_gap"]).clip(lower=0) / 5)
        * (0.20 + 0.45 * live_rotation_volatility)
    ).clip(0, 0.28)
    current["start_probability"] = current["start_probability"].clip(0.01, 0.98)
    current["sub_probability_given_bench"] = prior_sub
    current["play_probability"] = (
        current["start_probability"]
        + (1 - current["start_probability"]) * prior_sub
    ).clip(0.02, 0.995)
    current["sixty_probability"] = (
        current["start_probability"] * prior_sixty_start
    ).clip(0.01, 0.99)
    # Resolve the missing-history defaults before calibrating: the isotonic map
    # bins on the predicted value and has no NaN bin.
    current["minutes_if_start"] = current["minutes_if_start"].fillna(
        current["position_id"].map({1: 88.0, 2: 80.0, 3: 76.0, 4: 73.0})
    )
    current["minutes_if_bench"] = current["minutes_if_bench"].fillna(
        current["position_id"].map({1: 5.0, 2: 16.0, 3: 20.0, 4: 22.0})
    )
    # Same compression repair as the historical path, using terminal maps fitted
    # on the uncalibrated historical predictor.
    current = calibrate_live_minutes(current, historical)
    minutes_if_start = current["minutes_if_start"]
    minutes_if_bench = current["minutes_if_bench"]
    current["minutes_if_start_forecast"] = minutes_if_start
    current["minutes_if_bench_forecast"] = minutes_if_bench
    expected_minutes = (
        current["start_probability"] * minutes_if_start
        + (1 - current["start_probability"])
        * current["sub_probability_given_bench"]
        * minutes_if_bench
    ).clip(1, 90)
    current["expected_minutes"] = expected_minutes
    second_moment = (
        current["start_probability"] * (minutes_if_start.pow(2) + 12**2)
        + (1 - current["start_probability"])
        * current["sub_probability_given_bench"]
        * (minutes_if_bench.pow(2) + 10**2)
    )
    current["minutes_std"] = np.sqrt(
        (second_moment - expected_minutes.pow(2)).clip(lower=16)
    ).clip(4, 42)
    current["minutes_security_raw"] = (
        0.65 * current["sixty_probability"] + 0.35 * current["play_probability"]
    )
    appearance_share = expected_minutes / 90
    sixty_probability = current["sixty_probability"]
    goals = numeric_current("goals_scored")
    assists = numeric_current("assists")
    expected_goals = numeric_current("expected_goals")
    expected_assists = numeric_current("expected_assists")
    clean_sheets = numeric_current("clean_sheets")
    saves = numeric_current("saves")
    bonus = numeric_current("bonus")
    yellow_cards = numeric_current("yellow_cards")
    red_cards = numeric_current("red_cards")
    goals_conceded = numeric_current("goals_conceded")
    defensive_points = numeric_current("defensive_contribution")
    defensive_threshold = current["position_id"].map({1: 10, 2: 10, 3: 12, 4: 12}).astype(float)
    goal_points = current["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4}).astype(float)
    clean_points = current["position_id"].map({1: 4, 2: 4, 3: 1, 4: 0}).astype(float)
    team_attack_multiplier = (
        current["team_expected_goals_for"] / league_goal_rate
    ).pow(0.45).clip(0.70, 1.38)
    goal_vulnerability = (
        current["opponent_goal_vulnerability"]
        / current.groupby("position_id")["opponent_goal_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).fillna(1).clip(0.68, 1.42)
    assist_vulnerability = (
        current["opponent_assist_vulnerability"]
        / current.groupby("position_id")["opponent_assist_vulnerability"]
        .transform("median")
        .clip(lower=0.01)
    ).fillna(1).clip(0.72, 1.35)
    goal_rate_live = (
        0.72 * expected_goals
        + 0.28 * goals
        + 5 * current["goal_rate_prior"].fillna(
            current["position_id"].map({1: 0.01, 2: 0.04, 3: 0.20, 4: 0.28})
        )
    ) / rate_denominator
    assist_rate_live = (
        0.72 * expected_assists
        + 0.28 * assists
        + 5 * current["assist_rate_prior"].fillna(
            current["position_id"].map({1: 0.01, 2: 0.08, 3: 0.18, 4: 0.13})
        )
    ) / rate_denominator
    penalties_order = numeric_current("penalties_order")
    free_kick_order = numeric_current("direct_freekicks_order")
    corner_order = numeric_current("corners_and_indirect_freekicks_order")
    penalty_role_probability = np.select(
        [penalties_order == 1, penalties_order == 2], [0.86, 0.12], default=0.0
    )
    # Opta prices a penalty at about 0.79 xG, and FPL's `expected_goals` field is
    # Opta xG, so an established taker's penalty return is *already inside*
    # `goal_rate_live`. Adding the full uplift on top double-counted it, worth
    # roughly half a point a week for exactly the premium forwards and
    # midfielders. The uplift is therefore worth only what the player's own
    # history cannot yet tell us — the same 5-appearance prior share the rate
    # itself uses — which is what makes it useful for a new signing or a taker
    # who has just been handed the job.
    set_piece_novelty = (5.0 / rate_denominator).clip(0, 1)
    set_piece_goal_rate = (
        0.075 * current["team_expected_goals_for"] * penalty_role_probability
        + 0.018 * (free_kick_order == 1).astype(float)
    ) * set_piece_novelty
    set_piece_assist_rate = (
        0.025 * (corner_order == 1).astype(float)
        + 0.010 * (corner_order == 2).astype(float)
    ) * set_piece_novelty
    appearance_component = 1.0 + appearance_share
    goal_component = (
        (goal_rate_live + set_piece_goal_rate)
        * appearance_share
        * goal_points
        * team_attack_multiplier
        * goal_vulnerability
    )
    assist_component = (
        (assist_rate_live + set_piece_assist_rate)
        * appearance_share
        * 3
        * team_attack_multiplier
        * assist_vulnerability
    )
    personal_clean_probability = (
        (
            clean_sheets
            + 5
            * current["clean_sheet_rate_prior"].fillna(
                current["position_id"].map({1: 0.28, 2: 0.28, 3: 0.22, 4: 0.0})
            )
        )
        / rate_denominator
    ).clip(0, 0.75)
    blended_clean_probability = (
        0.82 * current["team_clean_probability"]
        + 0.18 * personal_clean_probability
    ).clip(0.03, 0.78)
    clean_component = (
        blended_clean_probability * clean_points * sixty_probability
    )
    save_rate_live = (
        saves
        + 5
        * current["save_rate_prior"].fillna(
            current["position_id"].map({1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0})
        )
    ) / rate_denominator
    save_component = (save_rate_live / 3) * appearance_share
    bps_rule_multiplier = np.select(
        [
            current["position_id"] == 1,
            (current["position_id"] == 2)
            & (current["defensive_rate_prior"].fillna(0) >= 9),
            current["position_id"].isin([3, 4]),
        ],
        [1.06, 0.94, 1.03],
        default=1.0,
    )
    bonus_rate_live = (
        bonus
        + 5
        * current["bonus_rate_prior"].fillna(
            current["position_id"].map({1: 0.18, 2: 0.22, 3: 0.28, 4: 0.28})
        )
    ) / rate_denominator
    bonus_component = (
        bonus_rate_live * appearance_share * bps_rule_multiplier
        + 0.12
        * current["team_clean_probability"]
        * sixty_probability
        * current["position_id"].isin([1, 2]).astype(float)
    )
    defensive_rate_live = (
        defensive_points
        + 5
        * current["defensive_rate_prior"].fillna(
            current["position_id"].map({1: 0.0, 2: 6.8, 3: 6.0, 4: 3.0})
        )
    ) / rate_denominator
    current["goal_rate"] = goal_rate_live
    current["assist_rate"] = assist_rate_live
    current["save_rate"] = save_rate_live
    current["bonus_rate"] = bonus_rate_live
    current["defensive_rate"] = defensive_rate_live
    current["player_role"] = assign_player_role(current)
    role_defensive_prior = current["player_role"].map(
        {
            "shot_stopper": 0.0,
            "clean_sheet_keeper": 0.0,
            "centre_back": 8.2,
            "set_piece_centre_back": 8.8,
            "attacking_full_back": 6.2,
            "balanced_defender": 6.8,
            "holding_midfielder": 8.0,
            "creator": 5.4,
            "goal_threat_midfielder": 4.8,
            "box_to_box_midfielder": 6.3,
            "link_forward": 3.3,
            "penalty_box_forward": 2.4,
            "mobile_forward": 3.0,
        }
    ).fillna(4.0)
    exact_coverage_live = current["defensive_event_coverage_prior"].fillna(0).clip(0, 1)
    current["defensive_event_coverage_live"] = exact_coverage_live
    defensive_rate_live = (
        exact_coverage_live * defensive_rate_live
        + (1 - exact_coverage_live) * role_defensive_prior
    )
    current["defensive_rate"] = defensive_rate_live
    current["player_role"] = assign_player_role(current)
    current["defensive_return_probability"] = poisson_tail(
        defensive_rate_live * appearance_share,
        defensive_threshold,
    ) * current["position_id"].isin([2, 3, 4]).astype(float)
    defensive_component = 2 * current["defensive_return_probability"]
    discipline_component = -(
        (yellow_cards + 3 * red_cards) / rate_denominator * appearance_share
    )
    conceded_component = -pd.Series(
        np.where(
            current["position_id"].isin([1, 2]),
            current["team_expected_goals_against"] / 2 * appearance_share,
            0,
        ),
        index=current.index,
    )
    component_projection = (
        appearance_component
        + goal_component
        + assist_component
        + clean_component
        + save_component
        + bonus_component
        + defensive_component
        + discipline_component
        + conceded_component
    )
    current["component_projection_unscaled"] = component_projection
    own_projection = component_projection
    # `position_match_multiplier` used to scale both of these by the current
    # fixture rank. That is the second absolute fixture price the handbook says
    # was removed: the opponent is already inside the structural routes through
    # expected goals and vulnerability, and neither challenger has a historical
    # counterpart carrying it. Both are now built exactly as in
    # `prepare_causal_history`.
    empirical_projection = (
        (0.62 * current["recent_raw"] + 0.38 * current["long_raw"])
        * (0.72 + 0.28 * current["play_probability"])
    ).clip(0.2, 13.5)
    position_base = current["position_id"].map({1: 3.2, 2: 2.8, 3: 3.0, 4: 2.8})
    market_projection = (
        position_base
        * (0.64 + 0.46 * current["minutes_security_raw"])
        * (0.82 + 0.28 * current["team_context_raw"].clip(0.4, 1.8))
        * (0.94 + 0.12 * current["crowd_raw"].rank(pct=True))
    ).clip(0.2, 13.5)
    role_projection = live_role_ridge_predictions(historical, current)
    current["role_ridge_projection"] = role_projection
    # Level-correct each member against its terminal historical bias, matching
    # the causal correction in `prepare_causal_history`. Without it the blend
    # inherits whichever member happens to sit highest, and the hurdles and chip
    # thresholds it feeds are all denominated in points.
    scored_history = historical[historical["fixture_count"] > 0]
    history_per_fixture = scored_history["points"] / scored_history[
        "fixture_count"
    ].clip(lower=1)
    live_positions = current["position_id"].astype(int)

    def level_corrected(history_column: str, projection: pd.Series) -> pd.Series:
        if history_column not in scored_history:
            return projection
        bias = (
            (scored_history[history_column] - history_per_fixture)
            .groupby(scored_history["position_id"].astype(int))
            .mean()
        )
        offset = live_positions.map(bias).fillna(0.0).clip(-1.5, 1.5)
        return (projection - offset).clip(0.2, 13.5)

    own_projection = level_corrected("structural_per_fixture", own_projection)
    empirical_projection = level_corrected("empirical_xpts", empirical_projection)
    market_projection = level_corrected("market_role_xpts", market_projection)
    role_projection = level_corrected("role_ridge_xpts", role_projection)
    structural_weight = current["ensemble_structural_weight"].fillna(0.32)
    empirical_weight = current["ensemble_empirical_weight"].fillna(0.27)
    market_weight = current["ensemble_market_weight"].fillna(0.21)
    role_weight = current["ensemble_role_weight"].fillna(0.20)
    weight_total = structural_weight + empirical_weight + market_weight + role_weight
    structural_weight /= weight_total
    empirical_weight /= weight_total
    market_weight /= weight_total
    role_weight /= weight_total
    public_weight = pd.Series(
        np.where(current["ep_next_num"] > 0, 0.18, 0.0), index=current.index
    )
    internal_weight = 1 - public_weight
    current["raw_projection"] = (
        internal_weight
        * (
            structural_weight * own_projection
            + empirical_weight * empirical_projection
            + market_weight * market_projection
            + role_weight * role_projection
        )
        + public_weight * current["ep_next_num"]
    ).clip(0.4, 13.8)
    # Player-v-opponent history is a small, uncertainty-aware matchup
    # adjustment only.  It is heavily shrunk for small samples and regime
    # changes so a remembered brace cannot overpower current role and markets.
    opponent_sample = current["opponent_matches"].fillna(0).clip(lower=0)
    history_per90 = current["history_per90"].fillna(current["raw_projection"])
    opponent_per90 = current["opponent_per90"].fillna(history_per90)
    h2h_reliability = (
        opponent_sample / (opponent_sample + 14)
        * current["team_rating_confidence"].clip(0, 1)
        * (1 - current["team_regime_shift"].clip(0, 0.9))
    )
    h2h_signal = (
        (opponent_per90 - history_per90)
        / history_per90.abs().clip(lower=2.0)
    ).clip(-0.30, 0.30)
    current["h2h_adjustment"] = (0.10 * h2h_reliability * h2h_signal).clip(-0.025, 0.025)
    current["raw_projection_per_fixture"] = (
        current["raw_projection"] * (1 + current["h2h_adjustment"])
    ).clip(0.4, 13.8)
    # Every route above prices one match. Apply the number of matches once, at
    # the end, so a Double Gameweek is projected as two and a blank as zero.
    # The variance multiplier keeps a floor of one so a blank still has a
    # defined (and correctly pessimistic) return distribution.
    live_fixture_multiplier = current["fixture_count"].clip(lower=1)
    current["raw_projection"] = (
        current["raw_projection_per_fixture"] * current["fixture_count"].clip(lower=0)
    )
    ensemble_stack = np.vstack(
        [
            own_projection.to_numpy(float),
            empirical_projection.to_numpy(float),
            market_projection.to_numpy(float),
            role_projection.to_numpy(float),
            current["ep_next_num"].where(
                current["ep_next_num"] > 0, own_projection
            ).to_numpy(float),
        ]
    ).T
    current["ensemble_disagreement"] = np.std(ensemble_stack, axis=1)
    current["ensemble_structural_weight_live"] = structural_weight * internal_weight
    current["ensemble_empirical_weight_live"] = empirical_weight * internal_weight
    current["ensemble_market_weight_live"] = market_weight * internal_weight
    current["ensemble_role_weight_live"] = role_weight * internal_weight
    current["ensemble_public_weight_live"] = public_weight
    component_scale = current["raw_projection"] / component_projection.clip(lower=0.25)
    current["component_appearance"] = appearance_component * component_scale
    current["component_goals"] = goal_component * component_scale
    current["component_assists"] = assist_component * component_scale
    current["component_clean"] = clean_component * component_scale
    current["component_defence"] = (
        save_component + defensive_component
    ) * component_scale
    current["component_bonus"] = bonus_component * component_scale
    current["component_adjustment"] = current["raw_projection"] - (
        current["component_appearance"]
        + current["component_goals"]
        + current["component_assists"]
        + current["component_clean"]
        + current["component_defence"]
        + current["component_bonus"]
    )
    weighted_games = current["team_id"].map(
        lambda team_id: sum(
            weight for _, _, weight, _ in horizon_map.get(int(team_id), [])
        )
    ).clip(lower=1.0)
    horizon_attack_ratio = (
        (current["team_horizon_expected_goals_for"] + 0.40)
        / (current["team_expected_goals_for"] + 0.40)
    ).clip(0.70, 1.40)
    horizon_clean_ratio = (
        (current["team_horizon_clean_probability"] + 0.08)
        / (current["team_clean_probability"] + 0.08)
    ).clip(0.65, 1.50)
    team_horizon_multiplier = pd.Series(
        np.select(
            [
                current["position_id"].isin([1, 2]),
                current["position_id"] == 3,
            ],
            [
                0.65 * horizon_clean_ratio + 0.35 * horizon_attack_ratio,
                0.18 * horizon_clean_ratio + 0.82 * horizon_attack_ratio,
            ],
            default=horizon_attack_ratio,
        ),
        index=current.index,
    ).clip(0.72, 1.35)
    # weighted_games already counts both legs of a current Double Gameweek, so
    # the horizon multiplies the per-match projection, not the Gameweek total.
    current["horizon_projection"] = (
        current["raw_projection_per_fixture"]
        * weighted_games
        * team_horizon_multiplier
    )
    current["expected_minutes"] = expected_minutes
    official_disagreement = (
        (own_projection - current["ep_next_num"]).abs()
        / current["raw_projection_per_fixture"].clip(lower=1)
    ).clip(0, 1)
    current["projection_std_per_fixture"] = np.sqrt(
        1.05**2
        + 0.020 * current["minutes_std"].pow(2)
        + 0.90 * current["ensemble_disagreement"].pow(2)
        + 1.8 / np.sqrt(nineties + 1)
    ).clip(1.15, 5.8)
    # A two-match total carries roughly twice the one-match variance.
    current["projection_std"] = current[
        "projection_std_per_fixture"
    ] * np.sqrt(live_fixture_multiplier)
    current["uncertainty"] = (
        current["projection_std"]
        / (current["raw_projection"] + current["projection_std"]).clip(lower=1)
        + 0.08 * official_disagreement
        + 0.10 * (1 - current["availability"] / 100).clip(0, 1)
    ).clip(0.05, 1.0)
    current["confidence"] = (100 * (1 - current["uncertainty"])).clip(0, 95)
    current["raw_blank_probability"] = normal_cdf(
        (2.5 - current["raw_projection"]) / current["projection_std"]
    )
    current["raw_return5_probability"] = 1 - normal_cdf(
        (4.5 - current["raw_projection"]) / current["projection_std"]
    )
    current["raw_haul8_probability"] = 1 - normal_cdf(
        (7.5 - current["raw_projection"]) / current["projection_std"]
    )
    current["blank_probability"] = current["raw_blank_probability"]
    current["return5_probability"] = current["raw_return5_probability"]
    current["haul8_probability"] = current["raw_haul8_probability"]
    current = calibrate_live_distributions(current, historical)
    transfer_pressure = (
        numeric_current("transfers_in_event")
        - numeric_current("transfers_out_event")
    ) / numeric_current("selected").clip(lower=2500)
    transfer_pressure_rank = transfer_pressure.rank(pct=True)
    current["transfer_pressure_rank"] = transfer_pressure_rank
    current["price_rise_probability"] = sigmoid(
        11 * (transfer_pressure_rank - 0.72)
    )
    current["price_fall_probability"] = sigmoid(
        11 * (0.28 - transfer_pressure_rank)
    )
    # Preserve the full deadline-known research matrix in the player artifact.
    # These fields feed prospective shadow models only; the production squad
    # score above remains unchanged.
    current["component_xpts"] = current["raw_projection"]
    # Match the historical frame's scales: the structural column is a Gameweek
    # total, the three challengers are per-match.
    current["structural_per_fixture"] = own_projection
    current["component_per_fixture"] = current["raw_projection_per_fixture"]
    current["component_xpts_structural"] = own_projection * current[
        "fixture_count"
    ].clip(lower=0)
    current["empirical_xpts"] = empirical_projection
    current["role_ridge_xpts"] = current["role_ridge_projection"]
    current["horizon_weighted_games_censored"] = weighted_games
    current["fixture_censored"] = current["fixture"]
    current["minutes_model_confidence"] = (
        current["history_matches"].fillna(0)
        / (current["history_matches"].fillna(0) + 8)
    ).clip(0, 0.95)
    current["observations"] = current["history_matches"].fillna(0)
    current["prediction_uncertainty"] = current["uncertainty"]
    current["GW"] = gw_number
    current["selected"] = numeric_current("selected").where(
        numeric_current("selected") > 0,
        current["ownership"] * 50_000,
    )
    current["clean_sheet_rate"] = personal_clean_probability
    current["bps_rate"] = current["bps_rate_prior"].fillna(0)
    current["defensive_event_coverage"] = current[
        "defensive_event_coverage_live"
    ]
    current["expected_goals"] = numeric_current("expected_goals")
    current["expected_assists"] = numeric_current("expected_assists")
    current["expected_goals_conceded"] = numeric_current(
        "expected_goals_conceded"
    )
    current["risk_adjusted_projection"] = (
        current["raw_projection"] - 0.10 * current["projection_std"]
    ).clip(lower=0.2)
    robust_horizon = (
        current["horizon_projection"]
        - 0.10
        * current["projection_std_per_fixture"]
        * np.sqrt(weighted_games)
        + 0.32
        * (current["price_rise_probability"] - current["price_fall_probability"])
        + 0.20 * current["haul8_probability"]
    ).clip(lower=0.2)
    current["risk_adjusted_horizon"] = (
        robust_horizon if robust_planning else current["horizon_projection"]
    )
    current["value_projection"] = (
        current["risk_adjusted_horizon"] / (current["price"] / 10).clip(lower=3.5)
    )
    current["model_score"] = (
        0.42 * current["model_score"].rank(pct=True)
        + 0.38 * current["risk_adjusted_horizon"].rank(pct=True)
        + 0.20 * current["risk_adjusted_projection"].rank(pct=True)
    )
    pool = current[
        (current["status"].isin(["a", "d"]))
        & (current["availability"] >= 75)
        & (current["price"] >= 35)
        & (
            (current["previous_minutes"].fillna(0) >= 180)
            | (current["ownership"] >= 0.5)
            | (current["ep_next_num"] >= 2.5)
        )
    ].copy()
    pool.reset_index(drop=True, inplace=True)
    pool["fixture_id"] = pool["team_id"].map(
        lambda team_id: fixture_map.get(int(team_id), {}).get("fixture_id", -1)
    )
    pool["position_rank"] = pool.groupby("position_id")[
        "risk_adjusted_horizon"
    ].rank(method="min", ascending=False)
    pool["position_count"] = pool.groupby("position_id")["id"].transform("size")
    pool["projection_percentile"] = pool.groupby("position_id")[
        "risk_adjusted_horizon"
    ].rank(pct=True)
    pool["balanced_utility"] = pool["model_score"]
    pool["protect_utility"] = (
        0.40 * pool["prediction_p10"].rank(pct=True)
        + 0.30 * pool["risk_adjusted_horizon"].rank(pct=True)
        + 0.20 * pool["sixty_probability"].rank(pct=True)
        + 0.10 * pool["confidence"].rank(pct=True)
    )
    pool["chase_utility"] = (
        0.38 * pool["prediction_p90"].rank(pct=True)
        + 0.30 * pool["haul8_probability"].rank(pct=True)
        + 0.22 * pool["horizon_projection"].rank(pct=True)
        + 0.10 * pool["team_attack"].rank(pct=True)
    )
    # Captaincy is worth exactly one extra copy of the player's Gameweek score,
    # so the armband must be decided in points. The previous blend of four
    # percentile ranks threw away the only thing that matters — the size of the
    # gap between the best option and the next — and gave ownership a vote in a
    # decision ownership has no bearing on. Replayed from 2018/19, it disagreed
    # with the expected-points choice in 87% of weeks and returned 5.42 against
    # 6.96 a week.
    pool["captain_score"] = pool["risk_adjusted_projection"]
    # Kept separately for display: the rank blend is still a readable "how safe
    # is this armband" summary, it just must not decide anything.
    pool["captain_safety"] = (
        0.56 * pool["risk_adjusted_projection"].rank(pct=True)
        + 0.18 * pool["fixture_now"]
        + 0.20 * pool["minutes_security"]
        + 0.06 * pool["crowd"]
    )
    chosen, xi = pick_squad(pool)
    strategy_profiles: list[dict] = []
    for profile_name, utility_column in [
        ("Protect", "protect_utility"),
        ("Balanced", "balanced_utility"),
        ("Chase", "chase_utility"),
    ]:
        profile_pool = pool.copy()
        profile_pool["model_score"] = profile_pool[utility_column]
        profile_chosen, profile_xi = pick_squad(profile_pool)
        strategy_profiles.append(
            {
                "name": profile_name,
                "squadIds": profile_pool.loc[profile_chosen, "id"].astype(int).tolist(),
                "expectedXI": round(float(profile_pool.loc[profile_xi, "raw_projection"].sum()), 1),
                "downsideXI": round(float(profile_pool.loc[profile_xi, "prediction_p10"].sum()), 1),
                "upsideXI": round(float(profile_pool.loc[profile_xi, "prediction_p90"].sum()), 1),
                "spend": round(float(profile_pool.loc[profile_chosen, "price"].sum()) / 10, 1),
            }
        )
    xi_set = set(xi)
    selected = pool.loc[chosen].sort_values(
        ["position_id", "model_score"], ascending=[True, False]
    )
    captain_order = pool.loc[xi].sort_values("captain_score", ascending=False).index.tolist()
    captain = captain_order[0]
    vice = captain_order[1]
    scenario_rng = np.random.default_rng(20260813)
    scenario_count = 5000
    xi_frame = pool.loc[xi]
    independent = scenario_rng.normal(size=(scenario_count, len(xi)))
    scenario_values = np.zeros_like(independent)
    for column_index, (_, player_row) in enumerate(xi_frame.iterrows()):
        same_team_columns = [
            local_index
            for local_index, (_, peer_row) in enumerate(xi_frame.iterrows())
            if int(peer_row["team_id"]) == int(player_row["team_id"])
        ]
        if column_index == same_team_columns[0]:
            common_shock = scenario_rng.normal(size=scenario_count)
            for same_team_index in same_team_columns:
                peer = xi_frame.iloc[same_team_index]
                scenario_values[:, same_team_index] = np.clip(
                    float(peer["raw_projection"])
                    + float(peer["projection_std"])
                    * (
                        math.sqrt(0.78) * independent[:, same_team_index]
                        + math.sqrt(0.22) * common_shock
                    ),
                    0,
                    25,
                )
    captain_column = xi.index(int(captain))
    scenario_totals = scenario_values.sum(axis=1) + scenario_values[:, captain_column]
    scenario_summary = {
        "simulations": scenario_count,
        "p10": round(float(np.quantile(scenario_totals, 0.10)), 1),
        "median": round(float(np.quantile(scenario_totals, 0.50)), 1),
        "p90": round(float(np.quantile(scenario_totals, 0.90)), 1),
        "probability70": round(100 * float((scenario_totals >= 70).mean())),
        "probability80": round(100 * float((scenario_totals >= 80).mean())),
        "correlation": "Team clean-sheet outcomes share a 22% scenario shock.",
    }

    def player_payload(index: int, row: pd.Series) -> dict:
        fixture = fixture_map.get(int(row["team_id"]), {})
        opponent_id = fixture.get("opponent")
        fixture_peers = pool[
            (pool["fixture_id"] == int(row["fixture_id"]))
            & (pool["id"] != int(row["id"]))
        ]
        popular_rival = (
            fixture_peers.nlargest(1, "ownership").iloc[0]
            if not fixture_peers.empty
            else row
        )
        fixture_rank = int(
            1 + (fixture_peers["model_score"] > float(row["model_score"])).sum()
        )

        def clean_number(name: str, default: float = 0.0) -> float:
            value = row.get(name, default)
            return default if pd.isna(value) else float(value)

        def research_number(name: str) -> float | None:
            value = row.get(name, np.nan)
            return None if pd.isna(value) else round(float(value), 8)

        set_pieces: list[str] = []
        for label, column in [
            ("Penalties", "penalties_order"),
            ("Direct free-kicks", "direct_freekicks_order"),
            ("Corners", "corners_and_indirect_freekicks_order"),
        ]:
            order = clean_number(column, 0)
            if 0 < order <= 2:
                set_pieces.append(f"{label} #{int(order)}")
        risk_flags: list[str] = []
        if float(row["expected_minutes"]) < 60:
            risk_flags.append("Minutes risk")
        if float(row["availability"]) < 100:
            risk_flags.append("Fitness flag")
        if float(row["sample_nineties"]) < 8:
            risk_flags.append("Small sample")
        if float(row["uncertainty"]) >= 0.42:
            risk_flags.append("Wide projection")
        if not risk_flags:
            risk_flags.append("No major flag")
        confidence = round(float(row["confidence"]))
        projection_percentile = float(row["projection_percentile"])
        verdict = (
            "Priority"
            if projection_percentile >= 0.90 and confidence >= 65
            else "Strong"
            if projection_percentile >= 0.75 and confidence >= 55
            else "Watch"
            if projection_percentile >= 0.45
            else "Fade"
        )
        archetype = {
            "shot_stopper": "Shot-stopping goalkeeper",
            "clean_sheet_keeper": "Clean-sheet goalkeeper",
            "centre_back": "Defensive centre-back",
            "set_piece_centre_back": "Set-piece centre-back",
            "attacking_full_back": "Attacking full-back",
            "balanced_defender": "Balanced defender",
            "holding_midfielder": "Defensive-contribution midfielder",
            "creator": "Creative midfielder",
            "goal_threat_midfielder": "Goal-threat midfielder",
            "box_to_box_midfielder": "Box-to-box midfielder",
            "link_forward": "Link forward",
            "penalty_box_forward": "Penalty-box forward",
            "mobile_forward": "Mobile forward",
        }.get(str(row["player_role"]), "Balanced role")
        return {
            "id": int(row["id"]),
            "name": str(row["display_name"]),
            "team": str(row["team_name"]),
            "position": POSITION_LABELS[int(row["position_id"])],
            "price": round(float(row["price"]) / 10, 1),
            "ownership": round(float(row["ownership"]), 1),
            "projected": round(float(row["raw_projection"]), 1),
            "sixWeekProjected": round(float(row["horizon_projection"]), 1),
            "expectedMinutes": round(float(row["expected_minutes"])),
            "uncertainty": round(float(row["uncertainty"]), 2),
            "confidence": confidence,
            "valueProjected": round(float(row["value_projection"]), 2),
            "verdict": verdict,
            "setPieces": set_pieces,
            "riskFlags": risk_flags,
            "archetype": archetype,
            "minutesModel": {
                "startProbability": round(100 * float(row["start_probability"])),
                "playProbability": round(100 * float(row["play_probability"])),
                "sixtyProbability": round(100 * float(row["sixty_probability"])),
                "minutesIfStart": round(clean_number("minutes_if_start_forecast", 75)),
                "minutesIfBench": round(clean_number("minutes_if_bench_forecast", 18)),
                "minutesStd": round(float(row["minutes_std"]), 1),
                "rotationVolatility": round(100 * clean_number("rotation_volatility", 0.35)),
                "competitionPressure": round(100 * float(row["competition_pressure"])),
                "managerRotation": round(100 * float(row["team_rotation_rate"])),
                "minimumFixtureGap": round(float(row["minimum_fixture_gap"]), 1),
                "scenarios": [
                    {
                        "label": "Starts",
                        "probability": round(100 * float(row["start_probability"])),
                        "minutes": round(clean_number("minutes_if_start_forecast", 75)),
                    },
                    {
                        "label": "Bench appearance",
                        "probability": round(
                            100
                            * (1 - float(row["start_probability"]))
                            * float(row["sub_probability_given_bench"])
                        ),
                        "minutes": round(clean_number("minutes_if_bench_forecast", 18)),
                    },
                    {
                        "label": "No appearance",
                        "probability": round(100 * (1 - float(row["play_probability"]))),
                        "minutes": 0,
                    },
                ],
                "availabilityEvidence": {
                    "status": str(row.get("status", "a")),
                    "chance": round(float(row["availability"])),
                    "officialNews": str(row.get("news", "") or "No official flag"),
                },
            },
            "distribution": {
                "p10": round(float(row["prediction_p10"]), 1),
                "median": round(float(row["prediction_p50"]), 1),
                "p90": round(float(row["prediction_p90"]), 1),
                "blankProbability": round(100 * float(row["blank_probability"])),
                "return5Probability": round(100 * float(row["return5_probability"])),
                "haul8Probability": round(100 * float(row["haul8_probability"])),
                "standardDeviation": round(float(row["projection_std"]), 2),
            },
            "defenderModel": {
                "actionRate": round(clean_number("defensive_rate_prior"), 1),
                "contributionProbability": round(
                    100 * float(row["defensive_return_probability"])
                ),
                "bpsRate": round(clean_number("bps_rate_prior"), 1),
                "goalRoute": round(clean_number("goal_rate_prior"), 3),
                "assistRoute": round(clean_number("assist_rate_prior"), 3),
                "exactEventCoverage": round(
                    100 * clean_number("defensive_event_coverage_live", 0)
                ),
            },
            "ensemble": {
                "structural": round(100 * float(row["ensemble_structural_weight_live"])),
                "empirical": round(100 * float(row["ensemble_empirical_weight_live"])),
                "marketRole": round(100 * float(row["ensemble_market_weight_live"])),
                "roleChallenger": round(100 * float(row["ensemble_role_weight_live"])),
                "official": round(100 * float(row["ensemble_public_weight_live"])),
                "disagreement": round(float(row["ensemble_disagreement"]), 2),
                "roleProjection": round(clean_number("role_ridge_projection"), 2),
            },
            "marketForecast": {
                "priceRiseProbability": round(100 * float(row["price_rise_probability"])),
                "priceFallProbability": round(100 * float(row["price_fall_probability"])),
            },
            "teamContext": {
                "expectedGoalsFor": round(float(row["team_expected_goals_for"]), 2),
                "expectedGoalsAgainst": round(float(row["team_expected_goals_against"]), 2),
                "cleanSheetProbability": round(100 * float(row["team_clean_probability"])),
                "horizonExpectedGoalsAgainst": round(
                    float(row["team_horizon_expected_goals_against"]), 2
                ),
                "horizonCleanSheetProbability": round(
                    100 * float(row["team_horizon_clean_probability"])
                ),
                "attackRank": round(float(row["team_attack_rank"])),
                "defenceRank": round(float(row["team_defence_rank"])),
                "strengthRank": round(float(row["team_strength_rank"])),
                "ratingConfidence": round(100 * float(row["team_rating_confidence"])),
                "regimeShift": round(100 * float(row["team_regime_shift"])),
                "marketWinProbability": (
                    round(
                        100
                        * float(
                            market_team_detail.get(int(row["team_id"]), {}).get(
                                "marketWinProbability"
                            )
                        )
                    )
                    if market_team_detail.get(int(row["team_id"]), {}).get(
                        "marketWinProbability"
                    )
                    is not None
                    else None
                ),
                "modelWinProbability": round(
                    100
                    * float(
                        market_team_detail.get(int(row["team_id"]), {}).get(
                            "modelWinProbability", 0
                        )
                    )
                ),
                "marketWeight": round(
                    100
                    * float(
                        market_team_detail.get(int(row["team_id"]), {}).get(
                            "marketWeight", 0
                        )
                    )
                ),
                "marketDisagreement": (
                    round(
                        100
                        * float(
                            market_team_detail.get(int(row["team_id"]), {}).get(
                                "winProbabilityGap"
                            )
                        )
                    )
                    if market_team_detail.get(int(row["team_id"]), {}).get(
                        "winProbabilityGap"
                    )
                    is not None
                    else None
                ),
                "optaWinProbability": (
                    round(
                        100
                        * float(
                            market_team_detail.get(int(row["team_id"]), {}).get(
                                "optaWinProbability"
                            )
                        )
                    )
                    if market_team_detail.get(int(row["team_id"]), {}).get(
                        "optaWinProbability"
                    )
                    is not None
                    else None
                ),
                "externalWinProbability": (
                    round(
                        100
                        * float(
                            market_team_detail.get(int(row["team_id"]), {}).get(
                                "externalWinProbability"
                            )
                        )
                    )
                    if market_team_detail.get(int(row["team_id"]), {}).get(
                        "externalWinProbability"
                    )
                    is not None
                    else None
                ),
            },
            "components": {
                "appearance": round(float(row["component_appearance"]), 2),
                "goals": round(float(row["component_goals"]), 2),
                "assists": round(float(row["component_assists"]), 2),
                "cleanSheet": round(float(row["component_clean"]), 2),
                "defence": round(float(row["component_defence"]), 2),
                "bonus": round(float(row["component_bonus"]), 2),
                "adjustment": round(float(row["component_adjustment"]), 2),
            },
            "history": {
                "matches": round(clean_number("history_matches")),
                "average": round(clean_number("history_average"), 2),
                "per90": round(clean_number("history_per90"), 2),
                "returnRate": round(100 * clean_number("history_returns")),
                "volatility": round(clean_number("history_volatility"), 2),
            },
            "opponentHistory": {
                "matches": round(clean_number("opponent_matches")),
                "average": round(clean_number("opponent_average"), 2),
                "per90": round(clean_number("opponent_per90"), 2),
                "returnRate": round(100 * clean_number("opponent_returns")),
            },
            "comparison": {
                "fixtureRank": fixture_rank,
                "fixturePlayers": int(len(fixture_peers) + 1),
                "positionRank": int(row["position_rank"]),
                "positionPlayers": int(row["position_count"]),
                "projectionRank": round(100 * projection_percentile),
                "popularRival": str(popular_rival["display_name"]),
                "popularRivalOwnership": round(float(popular_rival["ownership"]), 1),
                "popularRivalProjection": round(float(popular_rival["raw_projection"]), 1),
                "edgeVsPopular": round(
                    float(row["raw_projection"] - popular_rival["raw_projection"]), 1
                ),
            },
            # A 0-100 readability score. The armband itself is decided on
            # expected points via `captain_score`, not on this.
            "captainRating": round(float(row["captain_safety"]) * 100),
            "score": round(float(row["model_score"]) * 100),
            "strategyScores": {
                "protect": round(float(row["protect_utility"]), 4),
                "balanced": round(float(row["balanced_utility"]), 4),
                "chase": round(float(row["chase_utility"]), 4),
            },
            "features": {
                "recent": round(float(row["recent"]), 4),
                "history": round(float(row["long"]), 4),
                "recentValue": round(float(row["recent_value"]), 4),
                "historyValue": round(float(row["long_value"]), 4),
                "age": round(float(row["age_score"]), 4),
                "fixture": round(float(row["fixture"]), 4),
                "team": round(float(row["team_context"]), 4),
                "crowd": round(float(row["crowd"]), 4),
                "minutes": round(float(row["minutes_security"]), 4),
                "underlying": round(
                    float(
                        best.recent_share * row["recent_underlying"]
                        + (1 - best.recent_share) * row["long_underlying"]
                    ),
                    4,
                ),
            },
            "researchFeatures": {
                name: research_number(name)
                for name in dict.fromkeys(
                    LIVE_ACTION_FEATURES + LIVE_ROUTE_FEATURES
                )
            },
            "opponent": team_name.get(opponent_id, "TBD"),
            "venue": "H" if fixture.get("home") else "A",
            "starter": index in xi_set,
            "captain": index == captain,
            "vice": index == vice,
            "trend": "up"
            if float(row["recent_raw"]) > float(row["long_raw"]) + 0.35
            else "down"
            if float(row["recent_raw"]) + 0.35 < float(row["long_raw"])
            else "flat",
        }

    squad = [player_payload(int(index), row) for index, row in selected.iterrows()]
    squad.sort(key=lambda item: (not item["starter"], ["GK", "DEF", "MID", "FWD"].index(item["position"]), -item["score"]))

    top_players = pool.nlargest(12, "model_score")
    watchlist = [player_payload(int(index), row) for index, row in top_players.iterrows()]
    all_players = [
        player_payload(int(index), row)
        for index, row in pool.sort_values("model_score", ascending=False).iterrows()
    ]
    matchups: list[dict] = []
    for _, fixture in first_fixtures.iterrows():
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        match_pool = pool[pool["team_id"].isin([home, away])]
        if match_pool.empty:
            continue
        model_pick = match_pool.nlargest(1, "model_score").iloc[0]
        popular_pick = match_pool.nlargest(1, "ownership").iloc[0]
        matchups.append(
            {
                "fixture": f"{team_name[home]}  v  {team_name[away]}",
                "modelPick": str(model_pick["display_name"]),
                "popularPick": str(popular_pick["display_name"]),
                "modelProjection": round(float(model_pick["raw_projection"]), 1),
                "popularProjection": round(float(popular_pick["raw_projection"]), 1),
                "popularOwnership": round(float(popular_pick["ownership"]), 1),
                "modelConfidence": round(float(model_pick["confidence"])),
                "edge": round(
                    float(
                        model_pick["raw_projection"]
                        - popular_pick["raw_projection"]
                    ),
                    1,
                ),
            }
        )
    matchups.sort(key=lambda item: item["edge"], reverse=True)

    headline = {
        "gameweek": gw_number,
        "season": "2026/27",
        "deadline": deadline,
        "budget": round(float(selected["price"].sum()) / 10, 1),
        "projected": round(float(pool.loc[xi, "raw_projection"].sum()) + float(pool.loc[captain, "raw_projection"]), 1),
        "formation": f"{sum(pool.loc[xi, 'position_id'] == 2)}-{sum(pool.loc[xi, 'position_id'] == 3)}-{sum(pool.loc[xi, 'position_id'] == 4)}",
        "captain": str(pool.loc[captain, "display_name"]),
        "vice": str(pool.loc[vice, "display_name"]),
        "scenario": scenario_summary,
    }
    def normalise_player_name(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", str(value))
        return "".join(character for character in decomposed if character.isalnum()).lower()

    consensus_payload = load_elite_consensus()
    pool_by_name = {
        normalise_player_name(str(row["display_name"])): (int(index), row)
        for index, row in pool.iterrows()
    }
    chosen_set = set(chosen)
    xi_index_set = set(xi)
    elite_diagnostics: list[dict] = []
    consensus_names: set[str] = set()
    for consensus_row in consensus_payload.get("players", []):
        aliases = [consensus_row.get("name", ""), *consensus_row.get("aliases", [])]
        normalised_aliases = [normalise_player_name(alias) for alias in aliases]
        consensus_names.update(normalised_aliases)
        matched = next(
            (pool_by_name[alias] for alias in normalised_aliases if alias in pool_by_name),
            None,
        )
        if matched is None:
            elite_diagnostics.append(
                {
                    "player": str(consensus_row.get("name", "Unknown")),
                    "tier": str(consensus_row.get("tier", "watch")),
                    "status": "not-in-eligible-pool",
                    "selected": False,
                    "starter": False,
                }
            )
            continue
        matched_index, matched_row = matched
        selected_by_model = matched_index in chosen_set
        elite_diagnostics.append(
            {
                "player": str(matched_row["display_name"]),
                "tier": str(consensus_row.get("tier", "watch")),
                "status": (
                    "agreement"
                    if selected_by_model
                    else "core-disagreement"
                    if str(consensus_row.get("tier")) == "core"
                    else "watch-disagreement"
                ),
                "selected": selected_by_model,
                "starter": matched_index in xi_index_set,
                "projection": round(float(matched_row["raw_projection"]), 2),
                "horizonProjection": round(float(matched_row["horizon_projection"]), 2),
                "ownership": round(float(matched_row["ownership"]), 1),
                "positionRank": int(matched_row["position_rank"]),
            }
        )
    model_only = [
        {
            "player": str(pool.loc[index, "display_name"]),
            "projection": round(float(pool.loc[index, "raw_projection"]), 2),
            "ownership": round(float(pool.loc[index, "ownership"]), 1),
        }
        for index in xi
        if normalise_player_name(str(pool.loc[index, "display_name"])) not in consensus_names
    ]
    market_disagreements = sorted(
        [
            {
                "team": str(team_name[team_id]),
                **detail,
            }
            for team_id, detail in market_team_detail.items()
            if detail.get("winProbabilityGap") is not None
        ],
        key=lambda item: abs(float(item["winProbabilityGap"])),
        reverse=True,
    )
    exceptions = [
        str(pool.loc[index, "display_name"])
        for index in xi
        if not (
            float(pool.loc[index, "start_probability"]) >= 0.70
            and float(pool.loc[index, "play_probability"]) >= 0.84
        )
    ]
    current_meta = {
        "playersScored": int(len(pool)),
        "fixturesScored": int(len(first_fixtures)),
        "historicalSeasons": int(historical["season"].nunique()),
        "componentModel": "Single-count fixture components + Opta/regime team priors + Matchbook Poisson + role ensemble + coverage-aware defender DC/BPS",
        "decisionEngine": strategy.name,
        "managerPopulation": int(bootstrap.get("total_players", 0)),
        "officialRankImport": True,
        "publicProjectionEndpoint": "/api/projections",
        "defensiveEventCoverage": round(
            100 * float(historical.loc[historical["position_id"].isin([2, 3, 4]), "defensive_exact"].mean())
        ),
        "strategyProfiles": strategy_profiles,
        "fixtureSchema": {
            "version": "single-count-v1",
            "currentFixtureApplications": 1,
            "horizonMethod": "relative current-to-horizon rate ratio",
        },
        "squadOptimiser": {
            "type": "exact-binary-milp",
            "status": "optimal",
            "optimalityGap": 0,
            "budgetMinimum": 99.5,
            "budgetMaximum": 100.0,
            "benchPremiumMaximum": 2.0,
            "xiStartProbabilityFloor": 70,
            "xiPlayProbabilityFloor": 84,
            "exceptionMaximum": 1,
            "exceptions": exceptions,
        },
        "externalTeamSignals": {
            "optaAsOf": team_prior_payload.get("asOf"),
            "optaFixtureAsOf": opta_fixture_payload.get("asOf"),
            "optaFixtureCoverage": f"{len(opta_fixture_payload.get('fixtures', []))}/{len(first_fixtures)}",
            "matchbookStatus": matchbook_payload.get("status"),
            "matchbookCapturedAt": matchbook_payload.get("capturedAt"),
            "matchbookCoverage": f"{matchbook_payload.get('fixtureCount', 0)}/{matchbook_payload.get('expectedFixtureCount', len(first_fixtures))}",
            "marketDisagreements": market_disagreements[:8],
            "teamAnchors": list(team_anchor_details.values()),
        },
        "eliteConsensus": {
            "asOf": consensus_payload.get("asOf"),
            "method": "Diagnostic only; never enters the forecast or optimiser objective.",
            "players": elite_diagnostics,
            "modelOnlyXI": model_only,
        },
        "sourceUpdated": datetime.now().astimezone().isoformat(timespec="minutes"),
    }
    return headline, squad, watchlist, matchups[:6], all_players, current_meta


def main() -> None:
    data, data_summary = load_or_build_prepared_history()
    candidates, baseline_index = candidate_pool()
    print("Validating the live recommendation pipeline before long replays")
    current_recommendation(data, candidates[-5], False, WEEKLY_CHASE_STRATEGY)
    snapshot_scores, seasons = snapshot_replay(data, candidates)
    gameweeks = np.array(
        [data.loc[data["season"] == season, "GW"].nunique() for season in seasons],
        dtype=float,
    )
    snapshot_per_gameweek = snapshot_scores / gameweeks
    snapshot_stability = snapshot_per_gameweek.mean(axis=1) - snapshot_per_gameweek.std(axis=1) * 0.18

    # Select the recursive search space using training-only seasons. Reported
    # 2018/19 onward results cannot influence which candidates are evaluated.
    shortlist_indices: list[int] = []
    priority = [baseline_index] + list(range(len(candidates) - 5, len(candidates)))
    training_count = len(TRAINING_SEASONS)
    for season_id in range(training_count):
        priority.extend(
            np.argsort(snapshot_per_gameweek[:, season_id])[-8:][::-1].astype(int).tolist()
        )
    training_snapshot_stability = (
        snapshot_per_gameweek[:, :training_count].mean(axis=1)
        - snapshot_per_gameweek[:, :training_count].std(axis=1) * 0.25
    )
    priority.extend(np.argsort(training_snapshot_stability)[::-1].astype(int).tolist())
    for index in priority:
        if index not in shortlist_indices:
            shortlist_indices.append(index)
        if len(shortlist_indices) == SCREENING_FINALISTS:
            break
    recursive_candidates = [candidates[index] for index in shortlist_indices]
    recursive_scores = recursive_replay(
        data, recursive_candidates, WEEKLY_CHASE_STRATEGY
    )
    per_gameweek = recursive_scores / gameweeks
    stability = per_gameweek.mean(axis=1) - per_gameweek.std(axis=1) * 0.18
    best_local_index = int(np.argmax(stability))
    best_index = shortlist_indices[best_local_index]
    best = candidates[best_index]
    baseline_local_index = shortlist_indices.index(baseline_index)

    walk_forward: list[dict] = []
    # Current recommendations may use all completed history, but the policy gate
    # that governs every historical evaluation season must not use a candidate
    # selected on those evaluation outcomes.
    training_recursive_stability = (
        per_gameweek[:, :training_count].mean(axis=1)
        - per_gameweek[:, :training_count].std(axis=1) * 0.25
    )
    # Which strategy wins depends on the weights it is judged with: under one
    # candidate the joint tree trails the incumbent by 8.5 points on training,
    # under another it leads by 32.7 across ten seasons. Judging on a single
    # candidate therefore answers a question about a model that is not the one
    # the walk-forward will score, so the gate pools several of the best
    # training-selected candidates and compares strategies across all of them.
    gate_order = np.argsort(training_recursive_stability)[::-1]
    gate_candidates = [
        recursive_candidates[int(index)]
        for index in gate_order[:GATE_CANDIDATE_POOL]
    ]
    gate_candidate = gate_candidates[0]
    chip_policies = chip_policy_pool()
    gate_policy = GATE_CHIP_POLICY
    gate_scores, robust_plan_scores, _ = candidate_forecasts(data, gate_candidate)
    _, central_plan_scores, _ = candidate_forecasts(
        data, gate_candidate, robust_planning=False
    )
    gate_results: dict[str, tuple[np.ndarray, SimulationStrategy, np.ndarray, list]] = {}
    for candidate_index, candidate in enumerate(gate_candidates):
        probe_scores, probe_robust_plan, _ = candidate_forecasts(data, candidate)
        _, probe_central_plan, _ = candidate_forecasts(
            data, candidate, robust_planning=False
        )
        probe_free_hits = precompute_fresh_squads(
            data, probe_scores, one_week_only=True
        )
        for objective_name, plan_values in [
            ("robust", probe_robust_plan),
            ("central", probe_central_plan),
        ]:
            probe_fresh = precompute_fresh_squads(data, plan_values)
            for strategy_option in [WEEKLY_CHASE_STRATEGY, JOINT_OPTION_STRATEGY]:
                totals, gate_stats = simulate_candidate(
                    data,
                    probe_scores,
                    strategy_option,
                    chip_policy=gate_policy,
                    fresh_squads=probe_fresh,
                    free_hit_squads=probe_free_hits,
                    plan_scores=plan_values,
                )
                key = f"{objective_name}:{strategy_option.name}"
                if candidate_index == 0:
                    gate_results[key] = (
                        totals,
                        strategy_option,
                        (
                            robust_plan_scores
                            if objective_name == "robust"
                            else central_plan_scores
                        ),
                        list(gate_stats),
                    )
                else:
                    previous = gate_results[key]
                    gate_results[key] = (
                        previous[0] + totals,
                        previous[1],
                        previous[2],
                        previous[3] + list(gate_stats),
                    )
    for key, payload in gate_results.items():
        gate_results[key] = (
            payload[0] / len(gate_candidates),
            payload[1],
            payload[2],
            payload[3],
        )
    robust_gate_fresh = precompute_fresh_squads(data, robust_plan_scores)
    central_gate_fresh = precompute_fresh_squads(data, central_plan_scores)
    gate_free_hits = precompute_fresh_squads(
        data, gate_scores, one_week_only=True
    )
    selected_gate_name, gate_selection_report = select_gate_option(
        gate_results, training_count
    )
    print(
        f"Decision gate: {selected_gate_name}"
        f" ({'switched from' if gate_selection_report['switched'] else 'held'}"
        f" {gate_selection_report['incumbent']})"
    )
    selected_probe_totals, active_strategy, gate_plan_scores, _ = gate_results[
        selected_gate_name
    ]
    robust_planning_enabled = selected_gate_name.startswith("robust:")

    # The cheap recursive pass only screens weights. Re-score a compact,
    # training-selected set under the exact decision policy that will be used
    # in evaluation; otherwise weights and optimiser are tuned for different
    # objectives. Selection into this set uses only the two pre-2018 seasons.
    policy_priority = [baseline_local_index] + list(
        np.argsort(training_recursive_stability)[-16:][::-1].astype(int)
    )
    policy_priority.extend(
        index
        for index in range(len(recursive_candidates))
        if shortlist_indices[index] >= len(candidates) - 5
    )
    policy_local_indices: list[int] = []
    for index in policy_priority:
        if index not in policy_local_indices:
            policy_local_indices.append(index)
        if len(policy_local_indices) == 20:
            break
    policy_global_indices = [shortlist_indices[index] for index in policy_local_indices]
    recursive_candidates = [candidates[index] for index in policy_global_indices]
    recursive_scores = recursive_replay(
        data,
        recursive_candidates,
        active_strategy,
        robust_planning=robust_planning_enabled,
    )
    shortlist_indices = policy_global_indices
    per_gameweek = recursive_scores / gameweeks
    stability = per_gameweek.mean(axis=1) - per_gameweek.std(axis=1) * 0.18
    best_local_index = int(np.argmax(stability))
    best_index = shortlist_indices[best_local_index]
    best = candidates[best_index]
    baseline_local_index = shortlist_indices.index(baseline_index)

    chip_scores, chip_policy_stats, best_fresh_squads = replay_chip_policies(
        data, gate_scores, gate_plan_scores, chip_policies, active_strategy
    )
    no_chip_best, _ = simulate_candidate(
        data, gate_scores, active_strategy, plan_scores=gate_plan_scores
    )
    chip_gains = chip_scores - no_chip_best
    chip_stability = chip_gains.mean(axis=1) - chip_gains.std(axis=1) * 0.18
    best_chip_policy_index = int(np.argmax(chip_stability))
    best_chip_policy = chip_policies[best_chip_policy_index]
    best_scores, best_plan_scores, _ = candidate_forecasts(
        data,
        best,
        robust_planning=robust_planning_enabled,
    )

    def blend_candidates(indices: np.ndarray) -> Candidate:
        values = np.array(
            [
                [
                    recursive_candidates[int(index)].performance,
                    recursive_candidates[int(index)].value,
                    recursive_candidates[int(index)].age,
                    recursive_candidates[int(index)].fixture,
                    recursive_candidates[int(index)].team,
                    recursive_candidates[int(index)].crowd,
                    recursive_candidates[int(index)].minutes,
                    recursive_candidates[int(index)].underlying,
                    recursive_candidates[int(index)].recent_share,
                ]
                for index in indices
            ],
            dtype=float,
        )
        return Candidate(*values.mean(axis=0).tolist())

    def blend_chip_policies(indices: np.ndarray) -> ChipPolicy:
        values = np.array(
            [
                [
                    chip_policies[int(index)].wildcard_gap,
                    chip_policies[int(index)].free_hit_gap,
                    chip_policies[int(index)].bench_score,
                    chip_policies[int(index)].triple_score,
                    chip_policies[int(index)].afcon_bonus,
                    chip_policies[int(index)].first_wildcard_min_gw,
                    chip_policies[int(index)].second_wildcard_min_gw,
                ]
                for index in indices
            ],
            dtype=float,
        )
        averaged = values.mean(axis=0)
        return ChipPolicy(
            *averaged[:5].tolist(),
            first_wildcard_min_gw=int(round(averaged[5])),
            second_wildcard_min_gw=int(round(averaged[6])),
        )

    walk_forward_gate: list[dict] = []
    for season_id, season in enumerate(seasons):
        # Decide this season's policy on every season completed before it. The
        # frozen gate saw two seasons for all eight evaluations; by 2024/25 there
        # are eight, and the extra evidence is what lets a real difference clear
        # the selection noise instead of drowning in it.
        season_gate_name, season_gate_report = select_gate_option(
            gate_results, max(training_count, season_id)
        )
        season_strategy = gate_results[season_gate_name][1]
        season_robust_planning = season_gate_name.startswith("robust:")
        walk_forward_gate.append(
            {
                "season": str(season).replace("-", "/"),
                "selected": season_gate_name,
                "seasonsAvailable": int(max(training_count, season_id)),
                "switched": bool(season_gate_report["switched"]),
            }
        )
        if season_id == 0:
            trial_candidate = candidates[-5]
            mode = "fixed preseason seed"
            trial_policy = GATE_CHIP_POLICY
            chip_mode = "fixed preseason seed"
        else:
            prior = per_gameweek[:, :season_id]
            train_score = prior.mean(axis=1) - prior.std(axis=1) * 0.25
            ensemble_indices = np.argsort(train_score)[-12:]
            trial_candidate = blend_candidates(ensemble_indices)
            mode = f"12-model ensemble trained on {season_id} prior season{'s' if season_id != 1 else ''}"
            prior_chip_gain = chip_gains[:, :season_id]
            chip_train_score = (
                prior_chip_gain.mean(axis=1)
                - prior_chip_gain.std(axis=1) * 0.25
            )
            chip_ensemble_indices = np.argsort(chip_train_score)[-12:]
            trial_policy = blend_chip_policies(chip_ensemble_indices)
            chip_mode = (
                f"12-policy ensemble trained on {season_id} prior "
                f"season{'s' if season_id != 1 else ''}"
            )
        trial_scores, trial_plan_scores, _ = candidate_forecasts(
            data,
            trial_candidate,
            robust_planning=season_robust_planning,
        )
        season_mask = data["season"].to_numpy() == season
        season_data = data.loc[season_mask].reset_index(drop=True)
        season_scores = trial_scores[season_mask]
        season_plan_scores = trial_plan_scores[season_mask]
        no_chip_totals, no_chip_stats = simulate_candidate(
            season_data,
            season_scores,
            season_strategy,
            plan_scores=season_plan_scores,
        )
        trial_fresh_squads = precompute_fresh_squads(
            season_data, season_plan_scores
        )
        trial_free_hit_squads = precompute_fresh_squads(
            season_data, season_scores, one_week_only=True
        )
        trial_totals, trial_stats = simulate_candidate(
            season_data,
            season_scores,
            season_strategy,
            chip_policy=trial_policy,
            fresh_squads=trial_fresh_squads,
            free_hit_squads=trial_free_hit_squads,
            plan_scores=season_plan_scores,
        )
        current_rule_scores, current_rule_plan_scores, _ = candidate_forecasts(
            data,
            trial_candidate,
            current_rules=True,
            robust_planning=season_robust_planning,
        )
        current_rule_fresh_squads = precompute_fresh_squads(
            season_data, current_rule_plan_scores[season_mask]
        )
        current_rule_free_hit_squads = precompute_fresh_squads(
            season_data, current_rule_scores[season_mask], one_week_only=True
        )
        current_rule_totals, _ = simulate_candidate(
            season_data,
            current_rule_scores[season_mask],
            season_strategy,
            chip_policy=trial_policy,
            fresh_squads=current_rule_fresh_squads,
            free_hit_squads=current_rule_free_hit_squads,
            plan_scores=current_rule_plan_scores[season_mask],
            actual_column="points_current_rules",
        )
        points = round(float(trial_totals[0]))
        baseline = round(float(no_chip_totals[0]))
        season_transfer_stats = trial_stats[0]
        if season not in EVALUATION_SEASONS:
            continue
        walk_forward.append(
            {
                "season": season.replace("-", "/"),
                "points": points,
                "baseline": baseline,
                "uplift": round((points / baseline - 1) * 100, 1) if baseline else 0,
                "mode": mode,
                "chipMode": chip_mode,
                "weights": trial_candidate.as_dict(),
                "transfers": season_transfer_stats["transfers"],
                "weeksChanged": season_transfer_stats["weeksChanged"],
                "rolled": season_transfer_stats["rolled"],
                "hits": season_transfer_stats["hits"],
                "hitCost": season_transfer_stats["hitCost"],
                "gameweeks": season_transfer_stats["gameweeks"],
                "weeklyPoints": season_transfer_stats["weeklyPoints"],
                "chipPoints": points - baseline,
                "currentRulePoints": round(float(current_rule_totals[0])),
                "currentRuleDelta": round(float(current_rule_totals[0] - points)),
                "chips": season_transfer_stats["chips"],
                "legacyBaseline": round(
                    float(recursive_scores[baseline_local_index, season_id])
                ),
            }
        )

    top_indices = np.argsort(stability)[-5:][::-1]
    leaderboard = [
        {
            "rank": rank + 1,
            "trial": shortlist_indices[int(index)] + 1,
            "pointsPerGameweek": round(float(per_gameweek[index].mean()), 2),
            "stability": round(float(stability[index]), 3),
            "weights": recursive_candidates[int(index)].as_dict(),
        }
        for rank, index in enumerate(top_indices)
    ]
    curve_indices = np.linspace(0, len(recursive_candidates) - 1, 16).astype(int)
    sorted_scores = np.sort(stability)
    calibration_curve = [
        {
            "percentile": round(int(index) / (len(candidates) - 1) * 100),
            "score": round(float(sorted_scores[index]), 2),
        }
        for index in curve_indices
    ]

    best_totals, best_stats = simulate_candidate(
        data, best_scores, active_strategy, plan_scores=best_plan_scores
    )
    weekly_safe_captain = SimulationStrategy(
        "Six-GW planner + safe captain", 5.00, 5, False, True, 0, 99.0
    )
    patient_model_captain = SimulationStrategy(
        "Patient six-GW transfers + model captain", 8.00, 5, False, False, 0, 99.0
    )
    permissive_hit_strategy = SimulationStrategy(
        "Six-GW planner + three paid hits", 5.00, 5, False, False, 3, 2.5
    )
    safe_captain_totals, _ = simulate_candidate(
        data, best_scores, weekly_safe_captain, plan_scores=best_plan_scores
    )
    patient_totals, patient_stats = simulate_candidate(
        data, best_scores, patient_model_captain, plan_scores=best_plan_scores
    )
    permissive_hit_totals, _ = simulate_candidate(
        data, best_scores, permissive_hit_strategy, plan_scores=best_plan_scores
    )
    non_team_scale = 1 / max(1 - best.team, 0.01)
    no_team_candidate = Candidate(
        best.performance * non_team_scale,
        best.value * non_team_scale,
        best.age * non_team_scale,
        best.fixture * non_team_scale,
        0.0,
        best.crowd * non_team_scale,
        best.minutes * non_team_scale,
        best.underlying * non_team_scale,
        best.recent_share,
    )
    no_team_scores, no_team_plan_scores, _ = candidate_forecasts(
        data, no_team_candidate, robust_planning=robust_planning_enabled
    )
    no_team_totals, _ = simulate_candidate(
        data,
        no_team_scores,
        active_strategy,
        plan_scores=no_team_plan_scores,
    )
    best_model_score = feature_matrix(data) @ best.coefficients
    best_calibration = 0.72 + 0.56 * best_model_score
    structural_only_scores = (
        data["component_xpts_structural"].to_numpy(float) * best_calibration
    )
    structural_only_plan = (
        data["component_horizon_censored"].to_numpy(float)
        * (
            # Per-fixture ratio: both terms are strictly positive, so a blank
            # Gameweek cannot blow the ablation's horizon up.
            data["structural_per_fixture"]
            / data["component_per_fixture"].clip(lower=0.2)
        ).to_numpy(float)
        * best_calibration
    )
    structural_only_totals, _ = simulate_candidate(
        data,
        structural_only_scores,
        active_strategy,
        plan_scores=structural_only_plan,
    )
    non_role_weight = (1 - data["ensemble_role_weight"]).clip(lower=0.05)
    no_role_component = (
        data["component_xpts"]
        - data["ensemble_role_weight"] * data["role_ridge_xpts"]
    ) / non_role_weight
    no_role_scores = no_role_component.to_numpy(float) * best_calibration
    no_role_plan = (
        data["component_horizon_censored"].to_numpy(float)
        * (no_role_component / data["component_xpts"].clip(lower=0.2)).to_numpy(float)
        * best_calibration
    )
    no_role_totals, _ = simulate_candidate(
        data,
        no_role_scores,
        active_strategy,
        plan_scores=no_role_plan,
    )
    ranked_gate_names = sorted(
        gate_results,
        key=lambda name: float(np.mean(gate_results[name][0][:training_count])),
        reverse=True,
    )
    rejected_plan_totals = gate_results[
        next(name for name in ranked_gate_names if name != selected_gate_name)
    ][0]
    selected_plan_totals = selected_probe_totals
    baseline_totals = recursive_scores[baseline_local_index]

    def advice_test(label: str, improved: np.ndarray, comparison: np.ndarray, detail: str) -> dict:
        delta = float(np.mean(improved - comparison))
        return {
            "label": label,
            "delta": round(delta, 1),
            "result": "helped" if delta > 0.05 else "hurt" if delta < -0.05 else "neutral",
            "detail": detail,
        }

    expert_tests = [
        advice_test(
            "Paid-hit restraint",
            best_totals,
            permissive_hit_totals,
            "The calibration allows a paid-hit alternative, but keeps it only if the historical replay beats patient free transfers.",
        ),
        advice_test(
            "Patience over churn",
            patient_totals,
            best_totals,
            "Bank a transfer unless the best same-position upgrade clears the model hurdle.",
        ),
        advice_test(
            "Safety-first captaincy",
            safe_captain_totals,
            best_totals,
            "Blend proven output, the immediate fixture and 60-minute security; ownership is only a tiebreaker.",
        ),
        advice_test(
            "Expert data layer",
            best_totals,
            baseline_totals,
            "Add component expected points, six-GW fixtures, minutes security and underlying involvement to the original Lens feature set.",
        ),
        advice_test(
            "Team-strength signal",
            best_totals,
            no_team_totals,
            "Keep all structural clean-sheet logic fixed, then test whether the separately learned causal team attack/defence feature improves recursive squad decisions.",
        ),
        advice_test(
            "Causal position ensemble",
            best_totals,
            structural_only_totals,
            "Compare the dynamically error-weighted structural, empirical and market-role blend with the same structural model on its own.",
        ),
        advice_test(
            "Role-specific ridge challenger",
            best_totals,
            no_role_totals,
            "Fit an online regularised model separately for centre-backs, full-backs, creators, holding midfielders and forward roles; every deadline is predicted before its outcome updates the fit.",
        ),
        advice_test(
            "Joint decision gate",
            selected_plan_totals,
            rejected_plan_totals,
            "Use only the two pre-2018 training seasons to choose the transfer/chip tree and central-versus-robust objective together, then freeze both before evaluation.",
        ),
    ]

    best_chip_totals = chip_scores[best_chip_policy_index]
    best_chip_stats = chip_policy_stats[best_chip_policy_index]
    chip_gains_by_type: dict[str, list[int]] = {
        "Wildcard": [],
        "Free Hit": [],
        "Bench Boost": [],
        "Triple Captain": [],
        "Assistant Manager": [],
    }
    for season_stat in walk_forward:
        for chip in season_stat["chips"]:
            chip_gains_by_type[str(chip["chip"])].append(int(chip["gain"]))
    chip_breakdown = [
        {
            "chip": chip,
            "uses": len(gains),
            "averageGain": round(float(np.mean(gains)), 1) if gains else 0.0,
            "totalGain": int(sum(gains)),
        }
        for chip, gains in chip_gains_by_type.items()
    ]

    rank_target = add_rank_target_estimates(data, walk_forward)
    calibration_diagnostics = build_calibration_diagnostics(data, walk_forward)
    current_rules_replay = {
        "averagePoints": round(
            float(np.mean([item["currentRulePoints"] for item in walk_forward])), 1
        ),
        "averageScoringDelta": round(
            float(np.mean([item["currentRuleDelta"] for item in walk_forward])), 1
        ),
        "seasons": [
            {
                "season": item["season"],
                "points": item["currentRulePoints"],
                "deltaVsHistoricalRules": item["currentRuleDelta"],
            }
            for item in walk_forward
        ],
        "eventCoverage": calibration_diagnostics["defensiveEventCoverage"],
        "method": "Counterfactual 2026/27 scoring replay. Exact CBIT/CBIRT where public event data exists; coverage-labelled post-match proxy elsewhere. It is not used as a historical rank estimate.",
    }
    challenger_average = round(
        float(np.mean([item["points"] for item in walk_forward])), 1
    )
    frozen_audit_path = ROOT / "analysis" / "data" / "audited_policy_validation.json"
    frozen_audit = (
        json.loads(frozen_audit_path.read_text(encoding="utf-8"))
        if frozen_audit_path.exists()
        else None
    )
    frozen_average = (
        float(frozen_audit["average"]) if frozen_audit else challenger_average
    )
    frozen_hits = int(frozen_audit["targetHits"]) if frozen_audit else rank_target["hits"]
    frozen_margin = (
        float(frozen_audit["averageMargin"])
        if frozen_audit
        else float(rank_target["averageMargin"])
    )
    target_average = round(
        float(np.mean([item["top500Target"] for item in walk_forward])), 1
    )
    decision_promoted = bool(frozen_hits >= 6 and frozen_margin >= 0)
    headline, squad, watchlist, matchups, all_players, current_meta = current_recommendation(
        data, best, robust_planning_enabled, active_strategy
    )
    result = {
        "product": "FPL Lens",
        "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
        "model": {
            "version": "Lens 8.0",
            "trials": len(candidates),
            "recursiveTrials": len(recursive_candidates),
            "seasons": len(EVALUATION_SEASONS),
            "trainingSeasons": len(TRAINING_SEASONS),
            "playerWeeks": int(len(data)),
            "bestTrial": best_index + 1,
            "weights": best.as_dict(),
            "method": "Leak-free walk-forward replay with single-count fixtures, lineup-scenario minutes, exact squad MILP, role-specific online ridge challengers, regime-aware team Poisson rates, coverage-aware defender events and a jointly gated transfer-chip tree. Live team rates add timestamped Opta and Matchbook anchors.",
            "objective": (
                "Maximise legal autosubbed XI, captain and chip points; a training-only gate selected the downside/price/upside six-GW objective."
                if robust_planning_enabled
                else "Maximise legal autosubbed XI, captain and chip points; a training-only gate rejected the risk overlay and retained central six-GW expected points."
            ),
            "robustPlanningEnabled": robust_planning_enabled,
            "strategy": active_strategy.name,
        },
        "headline": headline,
        "currentMeta": current_meta,
        "squad": squad,
        "watchlist": watchlist,
        "fixtureMatchups": matchups,
        "backtest": walk_forward,
        "rankTarget": rank_target,
        "decisionGate": {
            **gate_selection_report,
            "walkForward": walk_forward_gate,
        },
        "championGovernance": {
            "decisionChampion": "Lens 8.0" if decision_promoted else "Research baseline",
            "decisionChallenger": "Frozen audited policy",
            "decisionPromoted": decision_promoted,
            "reason": (
                "Promoted: the frozen pre-2018 policy cleared the estimated top-500k line in at least six of eight seasons with a non-negative average margin."
                if decision_promoted
                else "Research-only: the frozen pre-2018 audit has not demonstrated consistent top-500k performance."
            ),
            "incumbentAveragePoints": target_average,
            "challengerAveragePoints": round(frozen_average, 1),
            "incumbentTop500Hits": 6,
            "challengerTop500Hits": frozen_hits,
            "playerLayerPromoted": decision_promoted,
            "incumbentPlayerMae": None,
            "challengerPlayerMae": calibration_diagnostics["mae"],
            "promotionRule": "Promotion requires at least 6/8 top-500k cutoff hits and a non-negative average cutoff margin under a policy frozen on 2016/17 and 2017/18; later searches remain diagnostics.",
        },
        "frozenAudit": {
            "available": frozen_audit is not None,
            "selection": frozen_audit["selection"] if frozen_audit else "Not generated",
            "averagePoints": round(frozen_average, 1),
            "top500Hits": frozen_hits,
            "averageMargin": round(frozen_margin, 1),
            "minimumPoints": int(frozen_audit["minimum"]) if frozen_audit else None,
            "averageChipDelta": (
                float(frozen_audit["averageChipDelta"]) if frozen_audit else None
            ),
            "researchSearchAverage": challenger_average,
            "method": "The promotion benchmark is selected only on 2016/17 and 2017/18. The broader recursive search is shown for research transparency but cannot promote itself after exposure to later-season results.",
        },
        "rankEngine": {
            "officialManagerPopulation": current_meta["managerPopulation"],
            "personalisedImport": True,
            "exactCurrentRank": "Imported directly from the official FPL entry endpoint",
            "forecast": "Correlated squad score scenarios are translated into a rank-movement range from the manager's exact current rank and percentile.",
            "historicalLimitation": "Historical cutoffs are estimates from 5,000 anonymised official manager histories with explicit tie, bootstrap and survivorship allowances; complete official cutoff tables are not published.",
        },
        "currentRulesReplay": current_rules_replay,
        "calibrationDiagnostics": calibration_diagnostics,
        "probabilisticEngine": {
            "playerIntervals": "10th, median and 90th percentile forecasts for every current player",
            "squadScenarios": headline["scenario"]["simulations"],
            "riskProfiles": current_meta["strategyProfiles"],
            "minutesModel": "Start, bench and no-appearance scenarios with manager rotation, competition pressure, congestion and official availability evidence",
            "ensemble": "Causally error-weighted structural, empirical, market-role and role-specific ridge models plus the official current projection when available",
            "defenderModel": "Coverage-aware exact CBIT/CBIRT events, role shrinkage where feeds are missing, Poisson thresholds, clean-sheet correlation, set pieces and 2026/27 BPS adjustment",
            "planningObjective": {
                "selected": "downside/price/upside" if robust_planning_enabled else "central expected points",
                "gate": "Frozen using only the two pre-2018 training seasons",
            },
            "decisionTree": active_strategy.name,
        },
        "chipStrategy": {
            "policyTrials": len(chip_policies),
            "policy": best_chip_policy.as_dict(),
            "averageGain": round(
                float(np.mean([item["chipPoints"] for item in walk_forward])), 1
            ),
            "diagnosticBestPolicyAverageGain": round(
                float(np.mean(best_chip_totals - best_totals)), 1
            ),
            "walkForwardAverageGain": round(
                float(np.mean([item["chipPoints"] for item in walk_forward])), 1
            ),
            "breakdown": chip_breakdown,
            "seasonPlans": [
                {
                    "season": stat["season"],
                    "chipPoints": stat["chipPoints"],
                    "chips": stat["chips"],
                }
                for stat in walk_forward
            ],
            "current": {
                "chip": "Hold",
                "gameweek": headline["gameweek"],
                "reason": "Single-Gameweek slate and a freshly optimised squad. Preserve the first-half chips for a larger fixture or availability edge.",
                "nextReview": "Re-score after every deadline; blank clashes, double fixtures, injuries and rotation are explicit triggers.",
            },
            "rules": "Two of each chip: one set through GW19 and one from GW20, with one chip permitted per Gameweek.",
        },
        "expertTests": expert_tests,
        "simulationSummary": {
            "averageTransfers": round(float(np.mean([item["transfers"] for item in best_stats])), 1),
            "averageWeeksChanged": round(float(np.mean([item["weeksChanged"] for item in best_stats])), 1),
            "averageRolled": round(float(np.mean([item["rolled"] for item in best_stats])), 1),
            "averageHits": round(float(np.mean([item["hits"] for item in best_stats])), 1),
            "patientAverageTransfers": round(float(np.mean([item["transfers"] for item in patient_stats])), 1),
            "averageJointPreflightHolds": round(float(np.mean([item["jointPreflightHolds"] for item in best_stats])), 1),
        },
        "leaderboard": leaderboard,
        "calibrationCurve": calibration_curve,
        "dataSummary": [
            item for item in data_summary if item["season"] in EVALUATION_SEASONS
        ],
        "fixtureIntegrity": fixture_integrity_audit(data),
        "sources": [
            {
                "label": "Historical FPL dataset",
                "url": "https://github.com/vaastav/Fantasy-Premier-League",
            },
            {
                "label": "Dixon-Coles dynamic score model",
                "url": "https://www.research.lancs.ac.uk/portal/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-football-betting-market%28d16276a2-d6e0-483b-a708-1d29663f1992%29.html",
            },
            {
                "label": "Bayesian hierarchical football model",
                "url": "https://discovery.ucl.ac.uk/id/eprint/16040/",
            },
            {
                "label": "Goal-chance team-strength evidence",
                "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0104647",
            },
            {
                "label": "OpenFPL forecasting + optimisation",
                "url": "https://arxiv.org/abs/2508.09992",
            },
            {
                "label": "Role challenger forecasting evidence",
                "url": "https://arxiv.org/abs/2405.02412",
            },
            {
                "label": "Official defender contribution analysis",
                "url": "https://www.premierleague.com/en/news/4361968/which-defenders-will-get-the-most-defensive-contribution-points-in-fpl",
            },
            {
                "label": "Official 2026/27 defensive contributions",
                "url": "https://www.premierleague.com/en/news/4361991/whats-happening-with-defensive-contribution-points-in-202627-fantasy",
            },
            {
                "label": "Official 2026/27 BPS changes",
                "url": "https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system",
            },
            {
                "label": "Robust FPL integer optimisation",
                "url": "https://arxiv.org/abs/2505.02170",
            },
            {
                "label": "Probability calibration survey",
                "url": "https://link.springer.com/article/10.1007/s10994-023-06336-7",
            },
            {
                "label": "Event-sequence xG research",
                "url": "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0312278",
            },
            {
                "label": "Official FPL API",
                "url": "https://fantasy.premierleague.com/api/bootstrap-static/",
            },
            {
                "label": "Opta 2026/27 season projections",
                "url": "https://theanalyst.com/articles/premier-league-predictions-2026-27-opta-supercomputer",
            },
            {
                "label": "Opta current match predictions",
                "url": "https://theanalyst.com/articles/premier-league-match-predictions",
            },
            {
                "label": "Matchbook exchange events",
                "url": "https://api.matchbook.com/edge/rest/events?exchange-type=back-lay&sport-ids=15",
            },
            {
                "label": "Reep identity register",
                "url": "https://github.com/withqwerty/reep",
            },
            {
                "label": "FPL champion: squad + transfers",
                "url": "https://www.premierleague.com/en/news/4671982/fpl-champion-how-to-build-the-perfect-squad-and-make-the-best-transfers",
            },
            {
                "label": "FPL champion: captaincy + chips",
                "url": "https://www.premierleague.com/en/news/4672128/fpl-champion-how-to-pick-your-captain-and-maximise-your-chips",
            },
            {
                "label": "FPL champion: 4–6 GW planning",
                "url": "https://www.premierleague.com/en/news/4025381",
            },
            {
                "label": "Official 2026/27 FPL changes",
                "url": "https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627",
            },
            {
                "label": "Official rank percentile feature",
                "url": "https://www.premierleague.com/en/news/4675893/see-how-your-fpl-career-history-compares-with-all-other-managers-in-the-world",
            },
            {
                "label": "FPL Review projection methodology",
                "url": "https://docs.fplreview.com/getting-started/about-fplreview/",
            },
            {
                "label": "Published top-500k points benchmark",
                "url": "https://www.fantasyfootballscout.co.uk/2019/11/28/quantifying-the-impact-of-fpl-decisions-with-chip-season-arriving-earlier/",
            },
            {
                "label": "Official FPL chip rules",
                "url": "https://fantasy.premierleague.com/help/",
            },
            {
                "label": "FPL chip strategy basics",
                "url": "https://www.premierleague.com/en/news/2174900/fpl-basics-chips",
            },
        ],
        "notes": [
            "Every historical GW is recursive: the prior squad, bank and season-correct free-transfer cap carry forward; transfers use contemporaneous prices and FPL selling-price rules.",
            "Minutes are explicit lineup scenarios: start, bench appearance and no appearance are conditioned on manager rotation, positional competition, fixture congestion, availability and substitution history.",
            "Four causal forecasts are blended by prior out-of-sample error, including an online ridge challenger fitted separately by scoring role; current official xPts joins only as a live external vote.",
            "The replay selects a legal formation, orders the bench, applies autosubs and hands the armband to the vice-captain when required.",
            "Transfers and chips share a six-GW decision layer. Historical future blank/double assignments are censored because announcement snapshots are unavailable; only the confirmed current slate can trigger a structural Free Hit, Bench Boost or Triple Captain signal.",
            "The transfer planner looks six Gameweeks ahead and can bank up to the cap that applied in that season; paid-hit variants were tested and rejected when they reduced replay points.",
            "A personalised import reads exact current overall rank and manager population from the official FPL API; historical top-500k bands remain estimates because complete cutoff tables are not public.",
            "Player analysis decomposes expected points by scoring route, labels sample size, estimates minutes and uncertainty, and treats opponent history as descriptive rather than predictive on its own.",
            "Team attack and defence are shifted, exponentially weighted and shrunk toward the league mean; current xG/xGA is blended with goals where available, so promoted and low-sample teams are not overconfidently rated.",
            "Defender and goalkeeper clean-sheet points are driven primarily by the opponent-adjusted team Poisson rate, then combined with expected minutes, attacking involvement, defensive contributions and bonus routes.",
            "Defensive-contribution forecasts use exact event counts where public feeds contain them and role-level shrinkage where they do not; the post-match proxy is isolated to the current-rules counterfactual.",
            "Live squads are evaluated in 5,000 correlated scenarios and exposed as Protect, Balanced and Chase profiles; the deterministic legal squad constraints remain binding in every profile.",
            "The current XV is solved as an exact binary mixed-integer programme with a zero optimality gap, £99.5m minimum spend, £2.0m maximum bench premium and an XI availability floor; at most one top-five-percent upside exception is permitted and disclosed.",
            "Fixture difficulty has a neutral opponent component and one venue component. Opponent difficulty enters each forecast once; the horizon uses a relative current-to-future adjustment rather than a second absolute multiplier.",
            "Current team rates blend causal historical strength, published Opta season and match forecasts, verified manager/transfer/promotion regimes and no-vig Matchbook probabilities. Opta-market and elite-manager disagreements are displayed diagnostically and never silently force a pick.",
            "Price-rise and fall probabilities use transfer pressure as an option-value tiebreaker, not as a substitute for expected points.",
            "Age is an availability/consistency prior, not a claim that younger or older players are inherently better.",
            "Current projections are decision support, not guarantees; late team news should override the model.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PLAYERS_OUTPUT.write_text(
        json.dumps(all_players, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Best trial {best_index + 1}/{len(candidates)} from "
        f"{len(recursive_candidates)} recursive finalists; wrote {OUTPUT.relative_to(ROOT)}"
    )


def refresh_current_artifact() -> None:
    """Refresh live recommendations without rerunning the expensive calibration."""
    if not OUTPUT.exists():
        raise FileNotFoundError("Run the full calibration before --refresh-current")
    result = json.loads(OUTPUT.read_text(encoding="utf-8-sig"))
    # Older artifacts embedded the full player pool. Keep refresh compatible
    # while preserving the split payload that makes the initial page smaller.
    result.pop("currentPlayers", None)
    weights = result["model"]["weights"]
    best = Candidate(
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
    historical, _ = load_or_build_prepared_history()
    result["fixtureIntegrity"] = fixture_integrity_audit(historical)
    stored_strategy = str(result["model"].get("strategy", ""))
    active_strategy = (
        JOINT_OPTION_STRATEGY
        if stored_strategy == JOINT_OPTION_STRATEGY.name
        else WEEKLY_CHASE_STRATEGY
    )
    headline, squad, watchlist, matchups, all_players, current_meta = (
        current_recommendation(
            historical,
            best,
            bool(result["model"].get("robustPlanningEnabled", False)),
            active_strategy,
        )
    )
    result.update(
        {
            "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
            "headline": headline,
            "squad": squad,
            "watchlist": watchlist,
            "fixtureMatchups": matchups,
            "currentMeta": current_meta,
        }
    )
    PLAYERS_OUTPUT.write_text(
        json.dumps(all_players, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Refreshed current recommendations in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--refresh-current" in sys.argv:
        refresh_current_artifact()
    else:
        main()

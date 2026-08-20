"""Fit the frozen listwise horizon/captain challengers to the live pool."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from captain_route_consensus_validation import (
    DEFENDER_TIE_BREAK,
    SEED_OFFSETS,
    SELECTED_SHARE,
    SELECTED_SIGMA,
)
from captain_ranker_validation import CAPTAIN_FEATURES, captain_matrix
from decision_focused_horizon_validation import target_and_maturity
from frontier_ranker_validation import matrix, selectable_frontier
from live_action_feature_ablation import (
    LIVE_EXTENDED_FEATURES,
    action_matrix as live_action_matrix,
)
from listwise_ranker_validation import fitted_ranker
from multiscale_horizon_validation import add_targets
from forecast_layer_v2 import minutes_mixture
from probabilistic_component_challenger import (
    causal_route_predictions,
    terminal_live_route_predictions,
)
from probabilistic_minutes_validation import terminal_live_predictions
from transfer_action_ranker_validation import SHIFTS, model as action_model, query_order
from wildcard_freehit_ablation import champion_forecasts


def current_frame(current: list[dict]) -> pd.DataFrame:
    rows = []
    for row in current:
        minutes = row["minutesModel"]
        history = row["history"]
        team = row["teamContext"]
        components = row["components"]
        expected_minutes = max(float(row["expectedMinutes"]), 8)
        record = {
                "component_xpts": row["projected"],
                "role_ridge_xpts": row["ensemble"]["roleProjection"],
                "expected_minutes": row["expectedMinutes"],
                "minutes_std": minutes["minutesStd"],
                "play_probability": minutes["playProbability"] / 100,
                "start_probability": minutes["startProbability"] / 100,
                "sixty_probability": minutes["sixtyProbability"] / 100,
                "recent_raw": history["average"],
                "long_raw": history["per90"],
                "goal_rate": components["goals"] / expected_minutes,
                "assist_rate": components["assists"] / expected_minutes,
                "bonus_rate": components["bonus"] / expected_minutes,
                "team_expected_goals_for": team["expectedGoalsFor"],
                "team_expected_goals_against": team["expectedGoalsAgainst"],
                "team_clean_probability": team["cleanSheetProbability"] / 100,
                "price": row["price"] * 10,
                "selected": row["ownership"] * 50_000,
                "fixture_count": 1,
                "haul8_probability": row["distribution"]["haul8Probability"] / 100,
                "return5_probability": row["distribution"]["return5Probability"] / 100,
                "team_attack_rating": team["expectedGoalsFor"],
                "opponent_defence_rating": 1.4 / max(float(row["features"]["fixture"]), 0.15),
                "minutes_security": 0.65 * minutes["sixtyProbability"] / 100 + 0.35 * minutes["playProbability"] / 100,
                "fixture_now": row["features"]["fixture"],
                "position_id": {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[row["position"]],
            }
        record.update(row.get("researchFeatures", {}))
        # Preserve categorical fields after the numeric research payload has
        # been merged, and let pandas represent unavailable JSON nulls as NaN.
        record["position_id"] = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[
            row["position"]
        ]
        rows.append(record)
    return pd.DataFrame(rows)


def horizon_scores(data: pd.DataFrame, live: pd.DataFrame, current: list[dict]) -> np.ndarray:
    frontier = selectable_frontier(data)
    output = np.zeros(len(live))
    for position in lens.SQUAD_QUOTAS:
        train_mask = (
            (data["position_id"].to_numpy(int) == position)
            & frontier
            & (data["fixture_count"].to_numpy(int) > 0)
        )
        test_mask = live["position_id"].to_numpy(int) == position
        train = data.loc[train_mask].sort_values(["season_order", "GW"], kind="stable")
        train_x, medians = matrix(train)
        test_x, _ = matrix(live.loc[test_mask], medians)
        query = pd.factorize(
            train["season_order"].astype(str) + "-" + train["GW"].astype(str), sort=False
        )[0]
        relevance = np.rint(train["horizon_target"].clip(0, 15)).to_numpy(np.int32)
        model = fitted_ranker(520000 + int(position))
        model.fit(train_x, relevance, qid=query)
        output[test_mask] = model.predict(test_x)
    return output


def captain_scores(data: pd.DataFrame, live: pd.DataFrame) -> np.ndarray:
    structural = data["component_xpts"].to_numpy(float)
    group_rank = pd.Series(structural, index=data.index).groupby(
        [data["season"], data["GW"]]
    ).rank(method="first", ascending=False)
    train = data[
        (group_rank <= 40)
        & (data["play_probability"] >= 0.45)
        & (data["fixture_count"] > 0)
    ].sort_values(["season_order", "GW"], kind="stable")
    train_x, medians = captain_matrix(train)
    test_x, _ = captain_matrix(live, medians)
    query = pd.factorize(
        train["season_order"].astype(str) + "-" + train["GW"].astype(str), sort=False
    )[0]
    relevance = np.rint(train["points"].clip(0, 15)).to_numpy(np.int32)
    model = fitted_ranker(530000)
    model.set_params(eval_metric="ndcg@5", lambdarank_num_pair_per_sample=8)
    model.fit(train_x, relevance, qid=query)
    return model.predict(test_x)


def map_within_position(raw: np.ndarray, reference: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.zeros(len(raw))
    percentile = np.zeros(len(raw))
    for position in lens.SQUAD_QUOTAS:
        indices = np.flatnonzero(positions == position)
        order = indices[np.argsort(raw[indices], kind="stable")]
        mapped[order] = np.sort(reference[indices])
        percentile[order] = np.linspace(0, 100, len(indices))
    return mapped, percentile


def map_local_rank(raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mapped = np.zeros(len(raw), dtype=float)
    order = np.argsort(raw, kind="stable")
    mapped[order] = np.sort(reference)
    return mapped


def forecast_v2_captain_scores(
    current: list[dict],
    route_captain: np.ndarray,
    live_minutes: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict]:
    """Apply the historically selected match boundary when exact odds exist."""
    market_path = lens.ROOT / "app" / "data" / "market-scores-v2.json"
    if not market_path.exists():
        return route_captain.copy(), {"status": "market artifact missing", "coveredPlayers": 0}
    artifact = json.loads(market_path.read_text(encoding="utf-8"))
    fixtures = {
        (row["team"], row["opponent"], row["venue"]): row
        for row in artifact.get("fixtures", [])
    }
    dynamic = np.asarray([float(row["projected"]) for row in current])
    covered = 0
    for index, row in enumerate(current):
        match = fixtures.get((row["team"], row["opponent"], row["venue"]))
        if match is None:
            continue
        covered += 1
        team = row["teamContext"]
        components = row["components"]
        attack_ratio = np.clip(
            float(match["dynamicExpectedGoalsFor"])
            / max(float(team["expectedGoalsFor"]), 0.25),
            0.60,
            1.55,
        )
        attack_points = float(components["goals"]) + float(components["assists"])
        attack_delta = attack_points * (attack_ratio - 1.0)
        clean_probability = max(float(team["cleanSheetProbability"]) / 100.0, 0.03)
        clean_delta = float(components["cleanSheet"]) * np.clip(
            float(match["dynamicCleanProbability"]) / clean_probability - 1.0,
            -0.70,
            1.20,
        )
        conceded_delta = (
            -0.5
            * (
                float(match["dynamicExpectedGoalsAgainst"])
                - float(team["expectedGoalsAgainst"])
            )
            * (float(row["expectedMinutes"]) / 90.0)
            * (row["position"] in {"GK", "DEF"})
        )
        dynamic[index] += 0.70 * (attack_delta + clean_delta + conceded_delta)
    if covered == 0:
        return route_captain.copy(), {
            "status": artifact.get("status", "no matching market fixtures"),
            "coveredPlayers": 0,
        }
    mixture = minutes_mixture(live_minutes)
    old_play = np.asarray(
        [float(row["minutesModel"]["playProbability"]) / 100 for row in current]
    )
    old_sixty = np.asarray(
        [float(row["minutesModel"]["sixtyProbability"]) / 100 for row in current]
    )
    new_reliability = 0.45 * (1.0 - mixture.no_show) + 0.55 * mixture.sixty_plus
    old_reliability = 0.45 * old_play + 0.55 * old_sixty
    downside = np.minimum(new_reliability - old_reliability, 0.0)
    multiplier = 1.0 + 0.50 * downside * (0.75 + 0.25 * mixture.entropy)
    dynamic *= np.clip(multiplier, 0.55, 1.0)
    market_rank = pd.Series(dynamic).rank(method="average", pct=True).to_numpy(float) * 100
    return 0.80 * route_captain + 0.20 * market_rank, {
        "status": "exact-capture prospective shadow",
        "coveredPlayers": covered,
        "marketSnapshotHash": artifact.get("sourceSnapshotHash"),
        "historicalSelection": "dynamicCaptain0.70-share0.20-minuteDownside0.50",
        "minutesModel": "terminal causal probabilistic challenger",
    }


def terminal_action_scores(
    data: pd.DataFrame,
    live: pd.DataFrame,
    historical_plan: np.ndarray,
    historical_component: dict,
    live_reference: np.ndarray,
    live_component: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Five-seed, two-band terminal fit for the pre-registered action policy."""
    target, _ = target_and_maturity(data)
    frontier = selectable_frontier(data)
    observed = data["fixture_count"].to_numpy(int) > 0
    historical_positions = data["position_id"].to_numpy(int)
    live_positions = live["position_id"].to_numpy(int)
    mapped = np.tile(live_reference[:, None, None], (1, len(SHIFTS), 5))
    fits = []
    for position in lens.SQUAD_QUOTAS:
        train_mask = observed & frontier & (historical_positions == position)
        test_mask = live_positions == position
        if not test_mask.any():
            continue
        train = data.loc[train_mask]
        test = live.loc[test_mask]
        train_x, medians = live_action_matrix(
            train,
            LIVE_EXTENDED_FEATURES,
            True,
            historical_plan,
            historical_component,
        )
        test_x, _ = live_action_matrix(
            test,
            LIVE_EXTENDED_FEATURES,
            True,
            live_reference,
            live_component,
            medians,
        )
        local_reference = live_reference[test_mask]
        for seed_index in range(5):
            for shift_index, shift in enumerate(SHIFTS):
                order, qid = query_order(train, shift)
                fitted = action_model(
                    860000 + 1000 * seed_index + 10 * position + shift_index
                )
                fitted.fit(train_x[order], target[train_mask][order], qid=qid)
                raw = fitted.predict(test_x)
                mapped[test_mask, shift_index, seed_index] = map_local_rank(
                    raw, local_reference
                )
        fits.append(
            {
                "position": int(position),
                "trainingRows": int(train_mask.sum()),
                "liveRows": int(test_mask.sum()),
            }
        )

    delta = mapped - live_reference[:, None, None]
    within_seed_agreement = np.sign(delta[:, 0, :]) == np.sign(delta[:, 1, :])
    seed_delta = delta.mean(axis=1)
    seed_sign = np.sign(seed_delta)
    positive = ((seed_sign > 0) & within_seed_agreement).mean(axis=1)
    negative = ((seed_sign < 0) & within_seed_agreement).mean(axis=1)
    vote = np.maximum(positive, negative)
    consensus = np.median(mapped.mean(axis=1), axis=1)
    agreed = vote >= 0.80
    score = live_reference + 0.05 * agreed * (consensus - live_reference)
    agreement_count = np.maximum(
        ((seed_sign > 0) & within_seed_agreement).sum(axis=1),
        ((seed_sign < 0) & within_seed_agreement).sum(axis=1),
    )
    return score, vote, agreement_count, {
        "seeds": 5,
        "priceBands": len(SHIFTS),
        "requiredDirectionalAgreement": 0.80,
        "blend": 0.05,
        "fits": fits,
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = add_targets(data.reset_index(drop=True))
    current = json.loads((lens.ROOT / "app" / "data" / "current-players.json").read_text(encoding="utf-8"))
    live = current_frame(current)
    horizon_raw = horizon_scores(data, live, current)
    captain_raw = captain_scores(data, live)
    positions = live["position_id"].to_numpy(int)
    reference_horizon = np.asarray([float(row["sixWeekProjected"]) for row in current])
    horizon_mapped, horizon_percentile = map_within_position(horizon_raw, reference_horizon, positions)
    captain_percentile = pd.Series(captain_raw).rank(pct=True).to_numpy(float) * 100
    structural_captain_percentile = pd.Series([float(row["projected"]) for row in current]).rank(pct=True).to_numpy(float) * 100
    historical_immediate, _, _ = champion_forecasts(data)
    live_immediate = np.asarray([float(row["projected"]) for row in current])
    live_route_ranks = []
    route_audits = []
    for seed_offset in SEED_OFFSETS:
        live_component, route_audit = terminal_live_route_predictions(
            data,
            live,
            historical_immediate,
            live_immediate,
            seed_offset=seed_offset,
        )
        route_raw = (
            live_component["stacked"] + SELECTED_SIGMA * live_component["sigma"]
        )
        live_route_ranks.append(
            pd.Series(route_raw).rank(pct=True).to_numpy(float) * 100
        )
        route_audits.append(route_audit)
    route_rank_mean = np.column_stack(live_route_ranks).mean(axis=1)
    frozen_captain = (
        0.50 * structural_captain_percentile + 0.50 * captain_percentile
    )
    position_adjustment = 100 * DEFENDER_TIE_BREAK * (positions == 2)
    position_adjustment += 10 * DEFENDER_TIE_BREAK * (positions == 1)
    route_captain = (
        (1 - SELECTED_SHARE) * frozen_captain
        + SELECTED_SHARE * route_rank_mean
        - position_adjustment
    )
    live_minutes = terminal_live_predictions(data, live)
    forecast_v2_captain, forecast_v2_audit = forecast_v2_captain_scores(
        current, route_captain, live_minutes
    )
    # The former action-transfer challenger failed its leak-free revalidation.
    # Keep compatibility fields inert so no stale consumer can activate it.
    action_score = reference_horizon.copy()
    action_vote = np.zeros(len(live), dtype=float)
    action_agreement = np.zeros(len(live), dtype=int)
    current_gw = int(round(float(live["GW"].median()))) if "GW" in live else 1
    action_policy_active = np.zeros(len(live), dtype=bool)
    players = []
    for index, row in enumerate(current):
        players.append(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "position": row["position"],
                "horizonRaw": round(float(horizon_raw[index]), 4),
                "horizonMapped": round(float(horizon_mapped[index]), 2),
                "horizonPercentile": round(float(horizon_percentile[index]), 1),
                "planBlend25": round(0.75 * reference_horizon[index] + 0.25 * horizon_mapped[index], 2),
                "captainRaw": round(float(captain_raw[index]), 4),
                "captainPercentile": round(float(captain_percentile[index]), 1),
                "captainBlend50": round(0.50 * structural_captain_percentile[index] + 0.50 * captain_percentile[index], 1),
                "routeCaptainScore": round(float(route_captain[index]), 2),
                "routeCaptainRankMean": round(float(route_rank_mean[index]), 2),
                "forecastV2CaptainScore": round(float(forecast_v2_captain[index]), 2),
                "actionConsensusMapped": round(float(action_score[index]), 2),
                "actionVote": round(float(action_vote[index]), 2),
                "actionAgreementSeeds": int(action_agreement[index]),
                "actionPolicyActive": bool(action_policy_active[index]),
            }
        )
    validation_path = lens.ROOT / "analysis" / "data" / "listwise_ranker_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    captain_validation_path = lens.ROOT / "analysis" / "data" / "captain_ranker_validation.json"
    captain_validation = json.loads(captain_validation_path.read_text(encoding="utf-8")) if captain_validation_path.exists() else {}
    result = {
        "schemaVersion": 1,
        "status": "shadow challenger",
        "model": "Causal position LambdaMART horizon rank + captain NDCG rank",
        "promotionRule": "Historical improvement can enter the shadow manager, but only frozen prospective decisions can promote it to production.",
        "historicalValidation": validation.get("models", {}),
        "captainValidation": captain_validation.get("models", {}),
        "actionChallenger": {
            "status": "invalidated; leak-free seed and ensemble gates failed",
            "policy": "disabled",
            "historicalValidation": "analysis/data/live_action_ensemble_validation.json",
            "currentGW": current_gw,
            "activePlayerCount": 0,
        },
        "captainRouteChallenger": {
            "status": "prospective captain shadow",
            "policy": "85% frozen captain rank + 15% five-seed route rank + 0.005 defender tie-break; minimum three completed training seasons",
            "historicalValidation": "analysis/data/captain_route_consensus_validation.json",
            "routeFits": route_audits,
        },
        "forecastV2Challenger": {
            **forecast_v2_audit,
            "policy": "80% route-consensus captain rank + 20% dynamic-match rank at 0.70 route strength with 0.50 new-minutes-downside protection",
            "historicalValidation": "analysis/data/captain_surface_search_v2.json",
        },
        "players": players,
    }
    output = lens.ROOT / "app" / "data" / "listwise-scores.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(lens.ROOT)} with {len(players)} live scores")


if __name__ == "__main__":
    main()

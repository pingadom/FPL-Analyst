"""Frozen research integration point for the selected forecast-v2 challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from captain_route_consensus_validation import selected_consensus_metric, weekly_percentile
from forecast_layer_v2 import dynamic_route_score
from wildcard_freehit_ablation import champion_forecasts


MODEL_ID = "forecast-v2-dynamic-captain-030-share-010"
MODEL_STATUS = "research-only pending prospective locked shadow sample"
DYNAMIC_ROUTE_STRENGTH = 0.30
CAPTAIN_SHARE = 0.10


def selected_forecasts(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frozen immediate, plan and selected captain policy arrays."""
    required = {
        "dynamic_expected_goals_for",
        "dynamic_expected_goals_against",
        "dynamic_clean_probability",
        "market_covered",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Forecast-v2 data is missing dynamic match fields: {missing}")
    immediate, plan, frozen_captain = champion_forecasts(data)
    route_captain = selected_consensus_metric(data, immediate, frozen_captain)
    dynamic, _ = dynamic_route_score(data, immediate, DYNAMIC_ROUTE_STRENGTH)
    selected_captain = (
        (1.0 - CAPTAIN_SHARE) * route_captain
        + CAPTAIN_SHARE * weekly_percentile(data, dynamic)
    )
    return immediate, plan, selected_captain

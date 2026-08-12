"""Estimate historical FPL rank cutoffs from official, public manager histories.

The official API exposes each active manager's past season points and overall
rank.  A deterministic random sample gives season-specific (points, rank)
anchors without retaining manager IDs or names.  We fit log(rank) locally
around the requested cutoff and publish both the estimate and bootstrap range.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "data" / "historical_rank_benchmarks.json"
SEASONS = [
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
    "2025/26",
]
TARGET_RANK = 500_000
USER_AGENT = "FPL-Lens-rank-benchmark/1.0"


def official_json(url: str, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=12) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 404:
                return {}
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (TimeoutError, URLError):
            if attempt == retries:
                raise
        time.sleep(0.35 * (attempt + 1))
    return {}


def fetch_history(entry: int) -> list[tuple[str, int, int]]:
    payload = official_json(
        f"https://fantasy.premierleague.com/api/entry/{entry}/history/"
    )
    rows: list[tuple[str, int, int]] = []
    for season in payload.get("past", []):
        name = str(season.get("season_name", ""))
        points = int(season.get("total_points", 0) or 0)
        rank = int(season.get("rank", 0) or 0)
        if name in SEASONS and points > 0 and rank > 0:
            rows.append((name, points, rank))
    return rows


def local_cutoff(
    pairs: list[tuple[int, int]], target_rank: int, rng: np.random.Generator
) -> dict:
    # Points-rank curves are close to linear in log-rank over a local interval.
    unique = sorted(set(pairs))
    target_log = math.log(target_rank)
    local = sorted(
        unique, key=lambda pair: abs(math.log(pair[1]) - target_log)
    )[: min(120, len(unique))]
    points = np.array([pair[0] for pair in local], dtype=float)
    log_ranks = np.log(np.array([pair[1] for pair in local], dtype=float))
    slope, intercept = np.polyfit(points, log_ranks, 1)
    estimate = float((target_log - intercept) / slope)
    fitted = slope * points + intercept
    r_squared = 1 - float(np.sum((log_ranks - fitted) ** 2)) / max(
        float(np.sum((log_ranks - log_ranks.mean()) ** 2)), 1e-9
    )

    bootstraps: list[float] = []
    for _ in range(1200):
        indices = rng.integers(0, len(local), len(local))
        sample_points = points[indices]
        sample_ranks = log_ranks[indices]
        sample_slope, sample_intercept = np.polyfit(
            sample_points, sample_ranks, 1
        )
        if sample_slope < -1e-5:
            bootstraps.append((target_log - sample_intercept) / sample_slope)
    lower, upper = np.percentile(bootstraps, [5, 95])
    closest = sorted(unique, key=lambda pair: abs(pair[1] - target_rank))[:8]
    below_cutoff = [point for point, rank in local if rank > target_rank]
    above_cutoff = [point for point, rank in local if rank <= target_rank]
    observed_lower = max(below_cutoff) if below_cutoff else math.floor(estimate)
    observed_upper = min(above_cutoff) if above_cutoff else math.ceil(estimate)
    conservative_target = max(round(estimate), observed_upper)
    # Sampling error is not the only uncertainty: equal points can straddle the
    # cutoff through the transfers-made tiebreak, and the active-manager sample
    # has survivorship bias. Keep an explicit two-point systematic allowance
    # instead of presenting a spuriously one-point regression interval.
    interval_lower = math.floor(min(lower, observed_lower, estimate - 2.0))
    interval_upper = math.ceil(max(upper, observed_upper, estimate + 2.0))
    return {
        "points": int(conservative_target),
        "fitPoints": round(float(estimate), 2),
        "p05": int(interval_lower),
        "p95": int(interval_upper),
        "observedBoundary": [int(observed_lower), int(observed_upper)],
        "logRankSlope": round(float(slope), 8),
        "logRankIntercept": round(float(intercept), 8),
        "anchors": len(unique),
        "localAnchors": len(local),
        "localRSquared": round(r_squared, 3),
        "closestOfficialSamples": [
            {"points": point, "rank": rank} for point, rank in closest
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    bootstrap = official_json(
        "https://fantasy.premierleague.com/api/bootstrap-static/"
    )
    total_managers = int(bootstrap.get("total_players", 0))
    if total_managers < 1:
        raise RuntimeError("The official FPL manager population was unavailable.")

    chooser = random.Random(20260812)
    entries = chooser.sample(
        range(1, total_managers + 1), min(args.samples, total_managers)
    )
    season_pairs: dict[str, list[tuple[int, int]]] = {
        season: [] for season in SEASONS
    }
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as executor:
        futures = {executor.submit(fetch_history, entry): entry for entry in entries}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                for season, points, rank in future.result():
                    season_pairs[season].append((points, rank))
            except Exception:
                failures += 1
            if completed % 100 == 0 or completed == len(futures):
                print(f"Official histories {completed}/{len(futures)}")

    rng = np.random.default_rng(20260812)
    estimates = []
    for season in SEASONS:
        pairs = season_pairs[season]
        if len(set(pairs)) < 25:
            raise RuntimeError(
                f"Only {len(set(pairs))} unique anchors for {season}; increase --samples."
            )
        estimates.append(
            {
                "season": season,
                **local_cutoff(pairs, TARGET_RANK, rng),
            }
        )

    result = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="minutes"),
        "source": "Official FPL public entry history endpoint",
        "managerPopulationAtSampling": total_managers,
        "requestedHistories": len(entries),
        "failedHistories": failures,
        "targetRank": TARGET_RANK,
        "method": "Deterministic random sample of active manager histories; local log(rank)-points fit plus the nearest observed score boundary. The interval includes bootstrap, points-tiebreak and survivorship allowances. No names or manager IDs are retained.",
        "seasons": estimates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for item in estimates:
        print(
            f"{item['season']}: {item['points']} "
            f"({item['p05']}-{item['p95']}), n={item['anchors']}"
        )


if __name__ == "__main__":
    main()

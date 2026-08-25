"""Average-score analysis for one or more model runs.

A single mean hides most of what matters. A model that scores 2140 by averaging
2300 and 1980 is a different proposition from one that scores 2140 every year,
because the objective is a *rank* threshold that must be cleared in the season you
actually play — and the top-500k cutoff moves from year to year too.

So this reports the mean, but alongside the things that decide whether it is any
good: the spread, the margin to that season's real cutoff, and how much of the
total came from chips rather than from picking players.

Note what `baseline` in the artifact is *not*. It is `points - chipPoints` — this
same model with its chips removed — so "uplift over baseline" is chip points
restated and says nothing about squad quality. The only genuine external
benchmark here is `top500Target`.

Usage:

    python analysis/score_analysis.py
    python analysis/score_analysis.py path/to/a.json path/to/b.json --labels A B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "app" / "data" / "model-results.json"


def load_run(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return payload["backtest"]


def summarise(seasons: list[dict]) -> dict:
    points = np.array([float(s["points"]) for s in seasons], dtype=float)
    # `baseline` is this same model with its chips removed, *not* an average
    # manager — it is exactly `points - chipPoints`, so any "uplift over
    # baseline" is chip points restated and says nothing about player picking.
    # The one genuine external benchmark in the artifact is `top500Target`, an
    # empirical per-season cutoff from FPL's public entry histories. It moves a
    # lot year to year (2151 in 2018/19, 2425 in 2022/23), so comparing against a
    # single fixed number would flatter or punish the model by season difficulty.
    baseline = np.array([float(s.get("baseline", np.nan)) for s in seasons])
    target = np.array([float(s.get("top500Target", np.nan)) for s in seasons])
    chips = np.array([float(s.get("chipPoints", 0)) for s in seasons])
    margin = points - target
    return {
        "seasons": [str(s["season"]) for s in seasons],
        "points": points,
        "baseline": baseline,
        "target": target,
        "margin": margin,
        "chips": chips,
        "mean": float(points.mean()),
        "median": float(np.median(points)),
        "std": float(points.std(ddof=1)) if len(points) > 1 else 0.0,
        "worst": float(points.min()),
        "best": float(points.max()),
        "no_chip": float(np.nanmean(baseline)),
        "mean_margin": float(np.nanmean(margin)),
        "worst_margin": float(np.nanmin(margin)),
        "hit_rate": float((margin >= 0).mean()),
        "chip_share": float(chips.mean()),
    }


def print_run(label: str, summary: dict) -> None:
    print(f"=== {label} ===")
    header = (
        f"{'season':<10}{'points':>8}{'no chips':>10}"
        f"{'cutoff':>8}{'margin':>8}{'chips':>7}"
    )
    print(header)
    for index, season in enumerate(summary["seasons"]):
        baseline = summary["baseline"][index]
        target = summary["target"][index]
        print(
            f"{season:<10}{summary['points'][index]:>8.0f}{baseline:>10.0f}"
            f"{target:>8.0f}{summary['margin'][index]:>+8.0f}"
            f"{summary['chips'][index]:>7.0f}"
        )
    print(
        f"\n  mean {summary['mean']:.1f}   median {summary['median']:.1f}   "
        f"spread {summary['std']:.1f}   range {summary['worst']:.0f}-{summary['best']:.0f}"
    )
    print(
        f"  without chips {summary['no_chip']:.1f}   chips contribute "
        f"{summary['chip_share']:+.1f}"
    )
    print(
        f"  margin to the top-500k cutoff {summary['mean_margin']:+.1f} mean, "
        f"{summary['worst_margin']:+.0f} worst, cleared {summary['hit_rate']:.0%} of seasons"
    )
    # The mean is not the thing being optimised. Clearing a threshold in a single
    # season is, and a model one standard deviation short of the cutoff on average
    # still clears it sometimes.
    if summary["std"] > 0:
        z = summary["mean_margin"] / summary["std"]
        print(
            f"  the mean sits {abs(z):.2f} season-to-season standard deviations "
            f"{'above' if z >= 0 else 'below'} the cutoff"
        )
    print()


def compare(labels: list[str], summaries: list[dict]) -> None:
    print("=== comparison ===")
    print(
        f"{'run':<26}{'mean':>9}{'spread':>9}{'no chips':>10}"
        f"{'margin':>9}{'chips':>8}"
    )
    for label, summary in zip(labels, summaries):
        print(
            f"{label:<26}{summary['mean']:>9.1f}{summary['std']:>9.1f}"
            f"{summary['no_chip']:>10.1f}{summary['mean_margin']:>+9.1f}"
            f"{summary['chip_share']:>+8.1f}"
        )

    base_label, base = labels[0], summaries[0]
    if len(summaries) < 2:
        return
    print(f"\nper-season delta against {base_label}:")
    common = base["seasons"]
    row = f"{'season':<10}"
    for label in labels[1:]:
        row += f"{label:>14}"
    print(row)
    for index, season in enumerate(common):
        row = f"{season:<10}"
        for summary in summaries[1:]:
            if season in summary["seasons"]:
                other = summary["points"][summary["seasons"].index(season)]
                row += f"{other - base['points'][index]:>+14.0f}"
            else:
                row += f"{'':>14}"
        print(row)
    print()
    for label, summary in zip(labels[1:], summaries[1:]):
        if summary["seasons"] != common:
            print(f"  {label}: season set differs, deltas above are partial")
            continue
        delta = summary["points"] - base["points"]
        # Paired across seasons: both runs replay the same weeks, so the pairing
        # removes season difficulty from the comparison.
        se = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else 0.0
        detail = f"{delta.mean():+.1f}"
        if se:
            detail += f" +/- {se:.1f} ({delta.mean() / se:+.1f} SE)"
        print(
            f"  {label:<24}{detail}   improved "
            f"{int((delta > 0).sum())}/{len(delta)} seasons"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=None)
    arguments = parser.parse_args()

    paths = [Path(p) for p in arguments.artifacts] or [DEFAULT_ARTIFACT]
    labels = arguments.labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        parser.error("--labels must match the number of artifacts")

    summaries = []
    kept_labels = []
    for label, path in zip(labels, paths):
        if not path.exists():
            print(f"{label}: missing {path}")
            continue
        summary = summarise(load_run(path))
        summaries.append(summary)
        kept_labels.append(label)
        print_run(label, summary)
    if len(summaries) > 1:
        compare(kept_labels, summaries)


if __name__ == "__main__":
    main()

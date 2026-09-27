"""What separates good FPL managers from bad ones, and where this model sits.

The elite snapshot in `elite_manager_audit.py` samples fifty managers at one
deadline. That is enough to describe a squad and far too little to identify a
habit. This reads a published study of **24,041 managers** stratified from the
world champion to rank six million, with a season of per-Gameweek scores and a
behavioural autopsy for each, and asks which behaviours actually track finishing
position.

Source: github.com/zakariae-boui/fpl-luck-or-skill. Third-party data, so it is
treated as evidence to be checked rather than accepted: the loader verifies the
row count, the rank ordering and that points fall monotonically down the bands
before any conclusion is drawn from it.

Mechanical versus behavioural
-----------------------------
Most of the strongest correlations here are not lessons. `transfer_gain_net` is
denominated in points, so correlating it with total points is close to a
tautology. `auto_sub_rescues` counts the times a starter did not play, and
`zero_min_starters` counts the same failure directly — a manager scores badly
*because* of those, so they describe the outcome rather than a choice that caused
it. `final_value` rises when picks are good.

Only a few columns record a decision made before the outcome was known: when the
transfer was made, how many were made, how much was paid in hits, and how heavily
the XI overlapped the crowd. Those are the ones worth acting on, and they are not
the largest correlations — which is exactly why the ranking has to be split rather
than read off the top.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "external" / "luck-or-skill"
SOURCE = "https://raw.githubusercontent.com/zakariae-boui/fpl-luck-or-skill/main/data"
FILES = ("managers_summary.csv", "autopsy_all.csv")

# Bands in finishing order, best first. `baseline` and the unranked groups are
# excluded because they are not a position on the ladder.
BANDS = [
    "top100", "top1k", "top10k", "band_50k", "band_100k",
    "band_250k", "band_500k", "band_1m", "band_2m", "band_5m",
]

# Columns that record a decision taken before the outcome was known.
BEHAVIOURAL = [
    "median_hours_before_deadline",
    "pct_transfers_last_3h",
    "n_transfers",
    "hits_cost",
    "xi_ownership",
]
# Columns that largely restate having scored well.
MECHANICAL = [
    "transfer_gain_net",
    "auto_sub_rescues",
    "zero_min_starters",
    "auto_sub_points",
    "final_value",
    "value_gain",
    "bench_points",
    "rebuys",
]


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = CACHE / name
        if target.exists() and target.stat().st_size > 0:
            continue
        request = urllib.request.Request(
            f"{SOURCE}/{name}", headers={"User-Agent": "FPL-Lens/1.0"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            target.write_bytes(response.read())


def load() -> pd.DataFrame:
    """Merge the two tables and check the data says what it claims to."""
    download()
    summary = pd.read_csv(CACHE / "managers_summary.csv", low_memory=False)
    autopsy = pd.read_csv(CACHE / "autopsy_all.csv", low_memory=False)
    overlap = [c for c in autopsy.columns if c in summary.columns and c != "entry"]
    frame = summary.merge(autopsy.drop(columns=overlap), on="entry", how="inner")

    if len(frame) < 20000:
        raise RuntimeError(f"Expected ~24,000 managers, merged {len(frame)}")
    ranked = frame[frame["group"].isin(BANDS)]
    medians = ranked.groupby("group", observed=True)["total_points"].median()
    ordered = [medians[b] for b in BANDS if b in medians]
    if ordered != sorted(ordered, reverse=True):
        raise RuntimeError(
            "Band median points are not monotonic in rank; the bands are not what "
            f"they claim to be: {dict(zip(BANDS, ordered))}"
        )
    return frame


def band_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ranked = frame[frame["group"].isin(BANDS)].copy()
    ranked["group"] = pd.Categorical(ranked["group"], BANDS, ordered=True)
    present = [c for c in columns if c in ranked.columns]
    return ranked.groupby("group", observed=True)[present].median().round(3)


def correlations(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, float]]:
    points = pd.to_numeric(frame["total_points"], errors="coerce")
    out = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.notna() & points.notna()
        if mask.sum() < 500:
            continue
        out.append((column, float(np.corrcoef(values[mask], points[mask])[0, 1])))
    return sorted(out, key=lambda item: -abs(item[1]))


if __name__ == "__main__":
    frame = load()
    print(f"managers: {len(frame):,}\n")

    print("Behaviour by finishing band (median):")
    print(band_table(frame, ["total_points"] + BEHAVIOURAL).to_string())
    print()
    print("Outcome measures by band (median) — these describe scoring well,")
    print("they do not explain it:")
    print(band_table(frame, MECHANICAL).to_string())
    print()

    print("Correlation with final points — decisions made before the outcome:")
    for name, value in correlations(frame, BEHAVIOURAL):
        print(f"  {name:<32}{value:+.3f}")
    print()
    print("Correlation with final points — largely mechanical:")
    for name, value in correlations(frame, MECHANICAL):
        print(f"  {name:<32}{value:+.3f}")

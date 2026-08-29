"""Official FPL availability as it stood *before* each historical deadline.

The archive this model replays carries no availability field at all — no
`chance_of_playing`, no `status`, no `news` — because `players_raw.csv` is an
end-of-season snapshot. That absence was treated here as permanent, and the
conclusion drawn was that a backtest could never see team news. Both halves were
wrong.

Roughly 58% of missed Gameweeks are continuations of an absence already visible in
the minutes record, and `absence_run` already exploits those. The other 42% are
first weeks — someone pulls up on the Thursday — and those are the ones an
archive is needed for. It exists: `Randdalf/fplcache` snapshotted the live FPL API,
and SmartPlayFPL published a deadline-anchored extract of it.

Measured against this model's own frame, on 35,118 joined established-player rows
across five seasons:

    sig_chance_playing   first-week absence rate
    100 (fit)                    5.8%
     75                         20.5%
     50                         21.1%
     25                         27.2%
      0                         24.5%

    has_news = 1  ->  23.5%      has_news = 0  ->  5.8%      lift 4.07x

A player carrying news is four times more likely to miss the very Gameweek the
minutes record gives no warning about.

Timestamp discipline
--------------------
Every row records `snapshot_at` and `deadline_time`. The snapshots sit a median
4.35 hours before their deadline and none is later than it, so this is information
a manager genuinely had. `load_deadline_news` re-checks that per row and refuses
anything captured at or after its own deadline, rather than trusting the
publisher — the whole value of the feed is that it predates the decision, so that
is the one property worth verifying locally.

Coverage is 2021-22 to 2025-26: five of the ten replayed seasons, and all four in
the xG era. Earlier seasons have no snapshots and must fall back to `absence_run`
alone, which makes availability a feature whose strength varies by season — worth
stating plainly rather than discovering later as an unexplained regime shift.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "external" / "dastan"
SOURCE = (
    "https://raw.githubusercontent.com/qazybekb/smartplayfpl-dastan/main/data/"
    "pre_deadline_signals.parquet"
)
COVERED_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

SIGNAL_COLUMNS = [
    "sig_chance_playing",
    "sig_status_risk",
    "sig_has_news",
    "sig_pens_order",
    "sig_fk_order",
    "sig_corners_order",
]


def download() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / "pre_deadline_signals.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())
    return target


def load_deadline_news() -> pd.DataFrame:
    """Availability signals keyed by season, Gameweek and FPL player code."""
    frame = pd.read_parquet(download())
    snapshot = pd.to_datetime(frame["snapshot_at"], utc=True, errors="coerce")
    deadline = pd.to_datetime(frame["deadline_time"], utc=True, errors="coerce")
    # Verify locally rather than trusting the publisher's own age_hours: a row
    # captured after its deadline would be a leak wearing the shape of a feature.
    late = (snapshot >= deadline).fillna(True)
    if late.any():
        raise RuntimeError(
            f"{int(late.sum())} snapshots are not earlier than their deadline; "
            "refusing to publish post-deadline information as a pre-deadline signal"
        )
    frame = frame.assign(
        gameweek=frame["gameweek"].astype(int),
        fpl_code=pd.to_numeric(frame["fpl_code"], errors="coerce"),
    )
    return frame[["season", "gameweek", "fpl_code", *SIGNAL_COLUMNS]]


def attach_deadline_news(data: pd.DataFrame) -> pd.DataFrame:
    """Join the signals onto a prepared frame by season, Gameweek and player code.

    Seasons without snapshots keep the "unknown" sentinel rather than a zero:
    zero is a real value on these scales — `chance_playing = 0` means ruled out —
    and filling an absent season with it would tell the model every player in
    2016/17 was injured.
    """
    news = load_deadline_news()
    frame = data.copy()
    frame["_gw"] = frame["GW"].astype(int)
    frame["_code"] = pd.to_numeric(frame["player_code"], errors="coerce")
    merged = frame.merge(
        news,
        left_on=["season", "_gw", "_code"],
        right_on=["season", "gameweek", "fpl_code"],
        how="left",
    )
    for column in SIGNAL_COLUMNS:
        merged[column] = merged[column].fillna(-1.0)
    merged["has_deadline_news_feed"] = (
        merged["season"].isin(COVERED_SEASONS).astype(float)
    )
    return merged.drop(columns=["_gw", "_code", "gameweek", "fpl_code"])


def coverage_report(data: pd.DataFrame) -> pd.DataFrame:
    """How much of each season the feed reaches. A silent join failure looks
    exactly like a season in which nobody was ever injured."""
    scored = data[data["fixture_count"] > 0]
    return (
        scored.assign(matched=scored["sig_chance_playing"] >= 0)
        .groupby("season")["matched"]
        .agg(rows="size", matched="sum")
        .assign(share=lambda f: (100 * f["matched"] / f["rows"]).round(1))
    )


if __name__ == "__main__":
    news = load_deadline_news()
    print(f"rows {len(news):,}  seasons {sorted(news['season'].unique())}")
    print(news.head(3).to_string())

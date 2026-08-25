"""FA Cup and EFL Cup fixtures for Premier League clubs.

The European module answers "is there a midweek game in Munich". This answers the
other half: the domestic cups, which the archive is equally blind to and which
produce the heaviest rotation in English football — a big club in an early EFL Cup
round often changes eight or nine players.

The direction is genuinely unclear here, more so than for Europe. Resting a
starter *before* a Champions League quarter-final costs an FPL manager a fit
player. A cup tie may do the opposite: the reserves played on Tuesday, so the
first eleven is *fresher* on Saturday than it would otherwise have been. This
module therefore supplies dates only and asserts nothing about the effect.

Format notes
------------
`openfootball/england` uses the same layout as the European files with one
difference that matters: no country tags, because every club is English. The FA
Cup field runs to 124 clubs including non-league, so matching is against the
Premier League club set rather than a country filter — anything that does not
resolve to a Premier League club key is simply dropped.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from european_fixtures import (
    _DATE_LINE,
    _MONTHS,
    _TIME_PREFIX,
    _anchor_date,
    club_key,
    club_key_or_blank,
)


ROOT = Path(__file__).resolve().parents[1]
CUP_CACHE = ROOT / "work" / "fpl-data" / "cups"
SOURCE_BASE = "https://raw.githubusercontent.com/openfootball/england/master"

COMPETITIONS = ("eflcup", "facup")

# Cup files carry the full club field, so these cover clubs the European files
# never mention. Keys are the source spelling with "FC"/"AFC" already stripped.
CUP_ALIASES = {
    "lutontown": "luton",
    "norwichcity": "norwich",
    "ipswichtown": "ipswich",
    "stokecity": "stoke",
    "swanseacity": "swansea",
    "hullcity": "hull",
    "cardiffcity": "cardiff",
    "huddersfieldtown": "huddersfield",
    "sunderland": "sunderland",
    "sheffieldunited": "sheffutd",
    "sheffieldwednesday": "sheffieldwed",
    "queensparkrangers": "qpr",
    "bournemouth": "bournemouth",
    "birminghamcity": "birmingham",
    "blackburnrovers": "blackburn",
    "boltonwanderers": "bolton",
}


def cup_club_key(name: str) -> str:
    """Club key for a cup entrant, applying the European aliases first."""
    key = club_key(name)
    return CUP_ALIASES.get(key, key)


def _download(season: str, competition: str) -> str:
    CUP_CACHE.mkdir(parents=True, exist_ok=True)
    target = CUP_CACHE / f"{season}-{competition}.txt"
    if target.exists() and target.stat().st_size > 0:
        return target.read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(
        f"{SOURCE_BASE}/{season}/{competition}.txt",
        headers={"User-Agent": "FPL-Lens/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8", errors="replace")
    target.write_text(payload, encoding="utf-8")
    return payload


def parse(payload: str, season: str) -> list[tuple[date, str]]:
    """Return (date, club key) rows, one per club per match."""
    rows: list[tuple[date, str]] = []
    current: date | None = None
    start_year = int(season.split("-")[0])
    anchor = _anchor_date(payload, start_year)
    for line in payload.splitlines():
        stamp = _DATE_LINE.match(line)
        if stamp:
            month_name, day, _ = stamp.groups()
            month = _MONTHS.get(month_name)
            if month is None:
                continue
            # Same anchoring rule as the European files: the year is stated once
            # and the sections are not in chronological order.
            for candidate_year in (start_year, start_year + 1):
                try:
                    candidate = date(candidate_year, month, int(day))
                except ValueError:
                    continue
                if candidate >= anchor - pd.Timedelta(days=10).to_pytimedelta():
                    current = candidate
                    break
            continue
        if current is None or " v " not in line:
            continue
        left, _, right = line.partition(" v ")
        home = _TIME_PREFIX.sub("", left).strip()
        # The away name runs up to the run of spaces before the score.
        away = re.split(r"\s{2,}", right.strip())[0].strip()
        for name in (home, away):
            if name:
                rows.append((current, cup_club_key(name)))
    return rows


def load_cup_matches(
    seasons: list[str], competitions: tuple[str, ...] = COMPETITIONS
) -> pd.DataFrame:
    """One row per club per cup match: season, club, date, competition."""
    records: list[dict] = []
    for season in seasons:
        for competition in competitions:
            try:
                payload = _download(season, competition)
            except Exception:
                continue
            for match_date, key in parse(payload, season):
                records.append(
                    {
                        "season": season,
                        "club": key,
                        "date": pd.Timestamp(match_date),
                        "competition": competition,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.drop_duplicates(["season", "club", "date", "competition"])
    return frame


def attach_cup_proximity(
    frame: pd.DataFrame,
    season: str,
    club_by_team_id: dict[int, str],
    competitions: tuple[str, ...] = COMPETITIONS,
) -> pd.DataFrame:
    """Days to and from the nearest domestic cup tie, per league fixture."""
    far = 99.0
    frame = frame.copy()
    matches = load_cup_matches([season], competitions)
    by_club: dict[str, np.ndarray] = {}
    if not matches.empty:
        by_club = {
            club: np.sort(group["date"].to_numpy().astype("datetime64[ns]"))
            for club, group in matches.groupby("club")
        }
    kickoff = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    kickoff = kickoff.dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    clubs = frame["team_id"].map(club_by_team_id).map(club_key_or_blank)

    days_to = np.full(len(frame), far)
    days_since = np.full(len(frame), far)
    for index, (club, when) in enumerate(zip(clubs, kickoff)):
        dates = by_club.get(club)
        if dates is None or np.isnat(when):
            continue
        deltas = (dates - when) / np.timedelta64(1, "D")
        after = deltas[deltas > 0]
        before = deltas[deltas < 0]
        if len(after):
            days_to[index] = float(after.min())
        if len(before):
            days_since[index] = float(-before.max())
    frame["cup_days_to"] = days_to
    frame["cup_days_since"] = days_since
    return frame


def coverage_report(frame: pd.DataFrame, premier_clubs: set[str]) -> pd.DataFrame:
    """Matches found per season, restricted to clubs that were in the top flight.

    The cup field includes the whole pyramid, so the useful number is how many
    ties involve a Premier League club — not the raw row count.
    """
    if frame.empty:
        return pd.DataFrame()
    top = frame[frame["club"].isin(premier_clubs)]
    return top.groupby(["season", "competition"]).agg(
        matches=("club", "size"),
        clubs=("club", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )


if __name__ == "__main__":
    import historical_odds as odds

    seasons = list(odds.SEASON_TAGS)
    frame = load_cup_matches(seasons)
    premier: set[str] = set()
    for season in seasons:
        premier.update(odds.load_market_fixtures([season])["home_key"].unique())
    print(coverage_report(frame, premier).to_string())
    unmatched = sorted(set(frame["club"]) - premier)
    print(f"\nnon-Premier-League entrants dropped: {len(unmatched)}")

"""Midweek European fixtures for Premier League clubs.

The archive contains Premier League matches and nothing else, so every fixture
looks equally rested. It is not. A club that played in Munich on the Wednesday
picks a different side on the Saturday than one that trained all week, and the
model has no way to know which is which: `team_rest_days`, `team_rotation_rate`
and `rotation_volatility` are all built from league fixtures alone.

The direction is not fixed, which is why this module only supplies the *fact* of a
European match and its date, and leaves the effect to be estimated. A deep squad
rotates and barely notices; a thin one fields the same eleven and tires; a club
knocked out in February suddenly has clear midweeks for a run-in. Encoding
"congested = worse" would assume the answer.

Source and coverage
-------------------
`openfootball/champions-league` publishes dated results as plain text, free and
unauthenticated, with a country code against every club. Coverage is uneven and
the unevenness matters:

* Champions League: every season from 2016/17. Consistent.
* Europa League: 2020/21 onward. Conference League: 2021/22 onward.

So a Europa club in 2017/18 is invisible here and its congested weeks look free.
That biases any *estimate* toward zero rather than inventing an effect, so a
result that survives it is real, while a null is ambiguous. It also means the
competitions available differ by season, which is exactly the kind of quiet
train/serve skew this project has been repairing — so anything shipped as a
feature should use `CORE_COMPETITIONS` (Champions League only), which is
available for every replayed season, rather than whatever each season happens to
have.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EUROPEAN_CACHE = ROOT / "work" / "fpl-data" / "european"
SOURCE_BASE = (
    "https://raw.githubusercontent.com/openfootball/champions-league/master"
)

# Champions League only. Present for every replayed season, so a feature built on
# it means the same thing in 2016/17 as in 2025/26.
CORE_COMPETITIONS = ("cl",)
# Everything the source publishes, for measurement where power matters more than
# cross-season consistency.
ALL_COMPETITIONS = ("cl", "el", "conf")

# Competitions actually published per season; anything else 404s.
SEASON_COMPETITIONS: dict[str, tuple[str, ...]] = {
    "2016-17": ("cl",),
    "2017-18": ("cl",),
    "2018-19": ("cl",),
    "2019-20": ("cl",),
    "2020-21": ("cl", "el"),
    "2021-22": ("cl", "el", "conf"),
    "2022-23": ("cl", "el", "conf"),
    "2023-24": ("cl", "el", "conf"),
    "2024-25": ("cl", "el", "conf"),
    "2025-26": ("cl",),
}

# The source writes full club names; the model keys on FPL's short ones. Only the
# clubs that differ after "FC"/"AFC" is stripped need an entry here.
CLUB_ALIASES = {
    "manchestercity": "mancity",
    "manchesterunited": "manutd",
    "tottenhamhotspur": "spurs",
    "tottenham": "spurs",
    "leicestercity": "leicester",
    "westhamunited": "westham",
    "newcastleunited": "newcastle",
    "nottinghamforest": "nottmforest",
    "brightonhovealbion": "brighton",
    "wolverhamptonwanderers": "wolves",
    "westbromwichalbion": "westbrom",
    "leedsunited": "leeds",
    "astonvilla": "astonvilla",
    "crystalpalace": "crystalpalace",
}

_MONTHS = {
    month: index + 1
    for index, month in enumerate(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    )
}

# The year is written on the first date line of a file and omitted on every one
# after it ("Tue Sep 19 2023", then "Wed Sep 20"). Requiring it drops all but the
# first date, which silently truncates each season at its first knockout round.
_DATE_LINE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$"
)
_HOME = re.compile(r"(.+?)\s*\(([A-Z]{3})\)\s*$")
_AWAY = re.compile(r"^\s*(.+?)\s*\(([A-Z]{3})\)")
_TIME_PREFIX = re.compile(r"^\s*\d{1,2}[:.]\d{2}\s*")


def club_key(name: str) -> str:
    """Normalise a source club name onto the model's club key."""
    cleaned = re.sub(r"\b(FC|AFC|CF)\b", " ", str(name))
    key = "".join(ch for ch in cleaned.lower() if ch.isalnum())
    return CLUB_ALIASES.get(key, key)


def _download(season: str, competition: str) -> str:
    EUROPEAN_CACHE.mkdir(parents=True, exist_ok=True)
    target = EUROPEAN_CACHE / f"{season}-{competition}.txt"
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


def _anchor_date(payload: str, start_year: int) -> date:
    """The season's opening date, from the one line that states its year.

    Every other date line omits the year, so this is the fixed point the rest are
    resolved against.
    """
    for line in payload.splitlines():
        stamp = _DATE_LINE.match(line)
        if not stamp:
            continue
        month_name, day, stated_year = stamp.groups()
        month = _MONTHS.get(month_name)
        if month is None:
            continue
        year = int(stated_year) if stated_year else start_year
        return date(year, month, int(day))
    return date(start_year, 7, 1)


def parse(payload: str, season: str) -> list[tuple[date, str, str]]:
    """Return (date, club key, country) rows for English clubs.

    `season` supplies the year, which the source only writes once per file.

    Emits one row per club per match, so a tie between two English clubs produces
    two rows and each is credited with the fixture.
    """
    rows: list[tuple[date, str, str]] = []
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
            # Pick the first year that puts this date at or after the season's
            # opening date. Tracking the year while reading drifts, because the
            # sections are not in chronological order — a knockout round restarts
            # earlier than the group stage printed above it. A plain July-to-June
            # rule is also wrong: COVID pushed the 2019/20 knockouts to August
            # 2020, and dating those to August 2019 drops European fixtures right
            # on top of that season's opening Gameweeks.
            for candidate_year in (start_year, start_year + 1):
                try:
                    candidate = date(candidate_year, month, int(day))
                except ValueError:  # 29 February in a non-leap year
                    continue
                # A few days of slack, because a matchday spans Tuesday and
                # Wednesday and the first date line in the file is not reliably
                # the earlier of the two — without it every such Tuesday is
                # pushed a full year forward. The slack stays far short of the
                # ~six weeks separating the 2019/20 anchor from its August
                # restart, so the COVID case still resolves correctly.
                if candidate >= anchor - timedelta(days=10):
                    current = candidate
                    break
            continue
        if current is None or " v " not in line:
            continue
        left, _, right = line.partition(" v ")
        left = _TIME_PREFIX.sub("", left)
        home_match = _HOME.search(left)
        away_match = _AWAY.match(right)
        if not home_match or not away_match:
            continue
        for name, country in (home_match.groups(), away_match.groups()):
            if country == "ENG":
                rows.append((current, club_key(name), country))
    return rows


def load_european_matches(
    seasons: list[str], competitions: tuple[str, ...] = ALL_COMPETITIONS
) -> pd.DataFrame:
    """One row per English club per European match: season, club, date."""
    records: list[dict] = []
    for season in seasons:
        available = SEASON_COMPETITIONS.get(season, ())
        for competition in competitions:
            if competition not in available:
                continue
            try:
                payload = _download(season, competition)
            except Exception:
                continue
            for match_date, key, _ in parse(payload, season):
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
        frame = frame.drop_duplicates(["season", "club", "date"])
    return frame


# Months in which a European tie is a knockout rather than a group game. The
# distinction is the whole effect: measured within European clubs against their
# own free weeks, a tie within four days costs 0.075 of start probability in these
# months (6.5 SE) and a statistically indistinguishable 0.004 in the group months.
# Nobody rests a first-choice forward for a dead November group game.
KNOCKOUT_MONTHS = (2, 3, 4, 5)


def attach_european_proximity(
    frame: pd.DataFrame,
    season: str,
    club_by_team_id: dict[int, str],
    competitions: tuple[str, ...] = ALL_COMPETITIONS,
) -> pd.DataFrame:
    """Days to and from the nearest European tie, per league fixture.

    Needs `team_id` and a timezone-aware `kickoff_time`. Clubs with no European
    football get the "far away" sentinel rather than a null, so downstream
    arithmetic does not have to special-case them.

    Everything here is known at the deadline: European draws are made weeks ahead,
    so a manager filling in a team on Saturday already knows about Tuesday.
    """
    far = 99.0
    frame = frame.copy()
    matches = load_european_matches([season], competitions)
    by_club: dict[str, np.ndarray] = {}
    if not matches.empty:
        by_club = {
            club: np.sort(group["date"].to_numpy().astype("datetime64[ns]"))
            for club, group in matches.groupby("club")
        }

    kickoff = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    # Compare like with like: the source dates are naive calendar days.
    kickoff = kickoff.dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    clubs = frame["team_id"].map(club_by_team_id).map(club_key_or_blank)

    days_to = np.full(len(frame), far)
    days_since = np.full(len(frame), far)
    knockout_soon = np.zeros(len(frame), dtype=float)
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
        near = dates[np.abs(deltas) <= 4]
        if len(near) and any(
            pd.Timestamp(value).month in KNOCKOUT_MONTHS for value in near
        ):
            knockout_soon[index] = 1.0

    frame["european_days_to"] = np.minimum(days_to, far)
    frame["european_days_since"] = np.minimum(days_since, far)
    frame["european_knockout_soon"] = knockout_soon
    return frame


def club_key_or_blank(name: object) -> str:
    """Club key for a league club name, or "" when the name is missing."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Clubs and matches found per season. A silent parse failure looks like a
    season in which no English club played in Europe, which never happens."""
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("season")
        .agg(
            matches=("club", "size"),
            clubs=("club", "nunique"),
            first=("date", "min"),
            last=("date", "max"),
        )
    )


if __name__ == "__main__":
    seasons = list(SEASON_COMPETITIONS)
    frame = load_european_matches(seasons)
    print(coverage_report(frame).to_string())
    print()
    print("club keys seen:", sorted(frame["club"].unique()))

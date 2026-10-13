"""Our own record of anytime-goalscorer prices, captured before each deadline.

No free historical archive of player scorer odds exists: Betfair's needs a
betting account, and the Wayback Machine holds only a handful of copies of any
odds page (8 of Fantasy Football Pundit's table, 95 Oddschecker match pages in
ten seasons). So the record is built forward instead. The Odds API's free plan
serves live Premier League `player_goal_scorer_anytime` prices (US bookmakers)
at one credit per match per region; a snapshot of a whole gameweek costs about
ten of the plan's 500 monthly credits.

Each snapshot is the full API response for every match of the coming
gameweek, stamped with when it was taken, so whatever is later done with it can
prove the price predates the deadline. Snapshots stay under `work/` (git-ignored):
the repository is public and the prices are the API's data, not ours to publish.

Setup: put your free key from the-odds-api.com in `work/odds-api-key.txt` (or the
ODDS_API_KEY environment variable). Without a key every call here is a no-op.

    python analysis/scorer_odds.py --status     # key present? credits left? snapshots so far
    python analysis/scorer_odds.py --snapshot   # capture the coming gameweek now
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "work" / "odds-api-key.txt"
STORE = ROOT / "work" / "scorer-odds"
API = "https://api.the-odds-api.com/v4/sports/soccer_epl"
MARKET = "player_goal_scorer_anytime"
REGIONS = "us"
# Stop well before the free plan runs out, so a busy month never fails a
# deadline snapshot halfway through a gameweek.
MIN_CREDITS_LEFT = 40


def api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def _get(path: str, params: dict) -> tuple[object, dict]:
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/8.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        headers = {k.lower(): v for k, v in response.headers.items()}
        return json.load(response), headers


def gameweek_dir(season: str, gameweek: int) -> Path:
    return STORE / season.replace("/", "-") / f"gw{int(gameweek):02d}"


def has_snapshot(season: str, gameweek: int) -> bool:
    folder = gameweek_dir(season, gameweek)
    return folder.exists() and any(folder.glob("*.json"))


def snapshot(season: str, gameweek: int, first_kickoff: datetime, last_kickoff: datetime) -> Path | None:
    """Capture every match kicking off in the gameweek's window. Returns the file."""
    key = api_key()
    if not key:
        print("scorer odds: no API key (work/odds-api-key.txt); skipped")
        return None
    events, headers = _get("/events", {"apiKey": key, "dateFormat": "iso"})
    window_start = first_kickoff - timedelta(hours=12)
    window_end = last_kickoff + timedelta(hours=12)
    chosen = [
        event for event in events
        if window_start <= datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")) <= window_end
    ]
    captured = datetime.now(timezone.utc)
    matches = []
    remaining = headers.get("x-requests-remaining")
    for event in chosen:
        if remaining is not None and float(remaining) < MIN_CREDITS_LEFT:
            print(f"scorer odds: only {remaining} credits left; stopping early")
            break
        odds, headers = _get(
            f"/events/{event['id']}/odds",
            {"apiKey": key, "regions": REGIONS, "markets": MARKET, "oddsFormat": "decimal", "dateFormat": "iso"},
        )
        remaining = headers.get("x-requests-remaining", remaining)
        matches.append(odds)
    folder = gameweek_dir(season, gameweek)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{captured:%Y%m%dT%H%M%SZ}.json"
    target.write_text(
        json.dumps(
            {
                "capturedAt": captured.isoformat(timespec="seconds"),
                "season": season,
                "gameweek": int(gameweek),
                "market": MARKET,
                "regions": REGIONS,
                "creditsRemaining": remaining,
                "matches": matches,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    priced = sum(1 for match in matches if match.get("bookmakers"))
    print(f"scorer odds: GW{gameweek} {len(matches)} matches ({priced} priced), {remaining} credits left -> {target.name}")
    return target


def player_prices(snapshot_payload: dict) -> list[dict]:
    """Flatten a snapshot to one row per bookmaker price for a player to score.

    Outcomes carry the player in `description` and "Yes"/"No" in `name`; only
    the "Yes" side is the scorer price.
    """
    rows = []
    for match in snapshot_payload.get("matches") or []:
        for bookmaker in match.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                if market.get("key") != MARKET:
                    continue
                for outcome in market.get("outcomes") or []:
                    if str(outcome.get("name", "Yes")).lower() not in ("yes", "over"):
                        continue
                    player = outcome.get("description") or outcome.get("name")
                    price = float(outcome.get("price") or 0)
                    if not player or price <= 1.0:
                        continue
                    rows.append(
                        {
                            "home": match.get("home_team"),
                            "away": match.get("away_team"),
                            "kickoff": match.get("commence_time"),
                            "bookmaker": bookmaker.get("key"),
                            "player": player,
                            "price": price,
                        }
                    )
    return rows


def status() -> None:
    key = api_key()
    print(f"API key: {'present' if key else 'missing (put it in work/odds-api-key.txt)'}")
    snapshots = sorted(STORE.glob("*/gw*/*.json")) if STORE.exists() else []
    print(f"snapshots: {len(snapshots)}" + (f", latest {snapshots[-1].relative_to(STORE)}" if snapshots else ""))
    if key:
        _, headers = _get("/events", {"apiKey": key})
        print(f"credits used {headers.get('x-requests-used')}, remaining {headers.get('x-requests-remaining')}")


def coming_gameweek() -> tuple[str, int, datetime, datetime] | None:
    """Season label, Gameweek and its kickoff window, from the FPL API."""
    import pick_history

    bootstrap = pick_history._get(pick_history.BOOTSTRAP)
    event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if not event:
        return None
    fixtures = pick_history._get(f"https://fantasy.premierleague.com/api/fixtures/?event={event['id']}")
    kickoffs = [
        datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
        for f in fixtures if f.get("kickoff_time")
    ]
    if not kickoffs:
        return None
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    start_year = deadline.year if deadline.month >= 7 else deadline.year - 1
    season = f"{start_year}/{str(start_year + 1)[2:]}"
    return season, int(event["id"]), min(kickoffs), max(kickoffs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    arguments = parser.parse_args()
    if arguments.snapshot:
        coming = coming_gameweek()
        if coming is None:
            print("No upcoming gameweek found")
            return
        snapshot(*coming)
    else:
        status()


if __name__ == "__main__":
    main()

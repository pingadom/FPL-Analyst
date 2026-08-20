"""Capture a sanitised, timestamped pre-deadline 1X2/totals market snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

import calibrate_model as lens
from market_lineup_challenger import SOURCE_TEMPLATE, first_available, no_vig_probabilities, season_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--input", type=Path, help="Optional locally supplied current-season CSV")
    args = parser.parse_args()
    deadline_status = json.loads(
        (lens.ROOT / "app" / "data" / "deadline-status.json").read_text(encoding="utf-8")
    )
    season = str(deadline_status["season"])
    gameweek = int(deadline_status["gameweek"])
    captured = datetime.now(timezone.utc)
    deadline = datetime.fromisoformat(str(deadline_status["deadline"]).replace("Z", "+00:00"))
    if captured > deadline:
        raise RuntimeError("Market snapshot was requested after the FPL deadline")
    url = SOURCE_TEMPLATE.format(code=season_code(season))
    try:
        if args.input:
            payload = args.input.read_bytes()
            url = str(args.input.resolve())
        else:
            with urlopen(url, timeout=30) as response:
                payload = response.read()
    except Exception as error:
        unavailable = {
            "schemaVersion": 1,
            "season": season,
            "gameweek": gameweek,
            "capturedAt": captured.isoformat(),
            "deadline": deadline.isoformat(),
            "status": "unavailable",
            "source": url,
            "reason": f"{type(error).__name__}: {error}",
            "fixtureCount": 0,
        }
        (lens.ROOT / "app" / "data" / "market-deadline-status-v2.json").write_text(
            json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(unavailable, indent=2))
        return
    raw = pd.read_csv(io.BytesIO(payload))
    home = first_available(raw, ("AvgH", "BbAvH", "B365H"))
    draw = first_available(raw, ("AvgD", "BbAvD", "B365D"))
    away = first_available(raw, ("AvgA", "BbAvA", "B365A"))
    over = first_available(raw, ("Avg>2.5", "BbAv>2.5", "B365>2.5"))
    under = first_available(raw, ("Avg<2.5", "BbAv<2.5", "B365<2.5"))
    home_p, draw_p, away_p = no_vig_probabilities(home, draw, away)
    over_p, _ = no_vig_probabilities(over, under)
    records = []
    for index, row in raw.iterrows():
        probabilities = (home_p[index], draw_p[index], away_p[index], over_p[index])
        if not all(pd.notna(value) for value in probabilities):
            continue
        records.append(
            {
                "date": str(row.get("Date", "")),
                "homeTeam": str(row["HomeTeam"]),
                "awayTeam": str(row["AwayTeam"]),
                "homeProbability": round(float(home_p[index]), 8),
                "drawProbability": round(float(draw_p[index]), 8),
                "awayProbability": round(float(away_p[index]), 8),
                "over25Probability": round(float(over_p[index]), 8),
            }
        )
    if not records:
        raise RuntimeError(f"No usable pre-closing market rows were available at {url}")
    core = {
        "schemaVersion": 1,
        "season": season,
        "gameweek": gameweek,
        "capturedAt": captured.isoformat(),
        "deadline": deadline.isoformat(),
        "status": "locked" if args.lock else "provisional",
        "source": url,
        "sourceSha256": hashlib.sha256(payload).hexdigest(),
        "oddsPolicy": "No-vig probabilities from first non-closing 1X2 and O/U 2.5 columns only.",
        "forbiddenClosingColumnsConsumed": [],
        "fixtures": records,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    core["snapshotHash"] = digest
    folder = lens.ROOT / "analysis" / "snapshots" / season / f"gw-{gameweek:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"market-{captured.strftime('%Y%m%dT%H%M%SZ')}-{digest[:12]}.json"
    path.write_text(json.dumps(core, indent=2) + "\n", encoding="utf-8")
    status = {
        "schemaVersion": 1,
        "season": season,
        "gameweek": gameweek,
        "capturedAt": core["capturedAt"],
        "deadline": core["deadline"],
        "status": core["status"],
        "snapshotHash": digest,
        "snapshotPath": path.relative_to(lens.ROOT).as_posix(),
        "fixtureCount": len(records),
    }
    (lens.ROOT / "app" / "data" / "market-deadline-status-v2.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

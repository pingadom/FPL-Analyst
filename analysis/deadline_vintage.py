"""Provenance boundary for deadline-safe forecast inputs.

Historical reconstruction is useful for research, but only a source captured
at or before the FPL deadline may qualify a model for prospective promotion.
This module makes that distinction machine-readable instead of leaving it in
comments spread across individual challengers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import calibrate_model as lens


FORBIDDEN_REALIZED_FIELDS = frozenset(
    {
        "points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "bps",
        "full_time_home_goals",
        "full_time_away_goals",
    }
)
FORBIDDEN_CLOSING_MARKET_FIELDS = frozenset(
    {
        "B365CH",
        "B365CD",
        "B365CA",
        "AvgCH",
        "AvgCD",
        "AvgCA",
        "PSCH",
        "PSCD",
        "PSCA",
    }
)


def parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class VintageSource:
    source_id: str
    source_type: str
    captured_at: str | None
    deadline_at: str | None
    timing: str
    path: str
    sha256: str
    fields: tuple[str, ...]
    seasons: tuple[str, ...] = ()
    decision_status: str = "unknown"

    @property
    def captured_before_deadline(self) -> bool:
        if self.captured_at is None or self.deadline_at is None:
            return False
        return parse_utc(self.captured_at) <= parse_utc(self.deadline_at)

    @property
    def forbidden_fields(self) -> tuple[str, ...]:
        forbidden = FORBIDDEN_REALIZED_FIELDS | FORBIDDEN_CLOSING_MARKET_FIELDS
        return tuple(sorted(set(self.fields) & forbidden))

    @property
    def promotion_eligible(self) -> bool:
        return (
            self.timing == "exact-capture"
            and self.captured_before_deadline
            and not self.forbidden_fields
            and self.decision_status == "locked"
        )

    @property
    def shadow_eligible(self) -> bool:
        return (
            self.timing == "exact-capture"
            and self.captured_before_deadline
            and not self.forbidden_fields
            and self.decision_status in {"provisional", "locked"}
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources(sources: Iterable[VintageSource]) -> dict[str, Any]:
    rows = []
    for source in sources:
        path = lens.ROOT / source.path
        errors = []
        if not path.exists():
            errors.append("missing-file")
        elif source.sha256 and sha256(path) != source.sha256:
            errors.append("hash-mismatch")
        if source.forbidden_fields:
            errors.append("forbidden-field")
        if source.timing == "exact-capture" and not source.captured_before_deadline:
            errors.append("captured-after-deadline")
        rows.append(
            {
                **asdict(source),
                "capturedBeforeDeadline": source.captured_before_deadline,
                "forbiddenFields": list(source.forbidden_fields),
                "shadowEligible": source.shadow_eligible and not errors,
                "promotionEligible": source.promotion_eligible and not errors,
                "errors": errors,
            }
        )
    return {
        "sources": rows,
        "valid": all(not row["errors"] for row in rows),
        "shadowEligibleSources": sum(row["shadowEligible"] for row in rows),
        "promotionEligibleSources": sum(row["promotionEligible"] for row in rows),
        "researchOnlySources": sum(not row["promotionEligible"] for row in rows),
    }


def build_manifest() -> dict[str, Any]:
    sources: list[VintageSource] = []
    market_root = lens.CACHE / "market-archive"
    for path in sorted(market_root.glob("*-E0.csv")):
        relative = path.relative_to(lens.ROOT).as_posix()
        sources.append(
            VintageSource(
                source_id=f"football-data:{path.stem}",
                source_type="pre-closing-market-archive",
                captured_at=None,
                deadline_at=None,
                timing="archive-first-odds-unknown-capture",
                path=relative,
                sha256=sha256(path),
                fields=("home_odds", "draw_odds", "away_odds", "over_under_odds"),
                seasons=(path.stem.split("-")[0],),
                decision_status="research-only",
            )
        )

    deadline_status_path = lens.ROOT / "app" / "data" / "deadline-status.json"
    if deadline_status_path.exists():
        status = json.loads(deadline_status_path.read_text(encoding="utf-8"))
        snapshot_path = lens.ROOT / status["snapshotPath"]
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        sources.append(
            VintageSource(
                source_id=f"official-fpl:{status['season']}:gw{int(status['gameweek']):02d}",
                source_type="official-fpl-deadline-snapshot",
                captured_at=str(status["capturedAt"]),
                deadline_at=str(status["deadline"]),
                timing="exact-capture",
                path=snapshot_path.relative_to(lens.ROOT).as_posix(),
                sha256=sha256(snapshot_path),
                fields=(
                    "official_player_status",
                    "official_chance_of_playing",
                    "fixtures",
                    "prices",
                    "ownership",
                    "transfers",
                ),
                seasons=(str(status["season"]),),
                decision_status=str(status["status"]),
            )
        )

    market_status_path = lens.ROOT / "app" / "data" / "market-deadline-status-v2.json"
    if market_status_path.exists():
        market_status = json.loads(market_status_path.read_text(encoding="utf-8"))
        if market_status.get("status") in {"provisional", "locked"}:
            market_snapshot_path = lens.ROOT / market_status["snapshotPath"]
            sources.append(
                VintageSource(
                    source_id=(
                        f"football-data-live:{market_status['season']}:"
                        f"gw{int(market_status['gameweek']):02d}"
                    ),
                    source_type="sanitised-pre-closing-market-snapshot",
                    captured_at=str(market_status["capturedAt"]),
                    deadline_at=str(market_status["deadline"]),
                    timing="exact-capture",
                    path=market_snapshot_path.relative_to(lens.ROOT).as_posix(),
                    sha256=sha256(market_snapshot_path),
                    fields=(
                        "home_probability",
                        "draw_probability",
                        "away_probability",
                        "over25_probability",
                    ),
                    seasons=(str(market_status["season"]),),
                    decision_status=str(market_status["status"]),
                )
            )

    validation = validate_sources(sources)
    result = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "historicalUse": "Unknown-capture archives may train research challengers but cannot independently justify production promotion.",
            "prospectiveUse": "Exact captures at or before the deadline may advance locked shadow evidence.",
            "forbiddenRealizedFields": sorted(FORBIDDEN_REALIZED_FIELDS),
            "forbiddenClosingMarketFields": sorted(FORBIDDEN_CLOSING_MARKET_FIELDS),
        },
        **validation,
    }
    return result


def main() -> None:
    result = build_manifest()
    output = lens.ROOT / "analysis" / "data" / "deadline_vintage_manifest.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sources": len(result["sources"]),
                "valid": result["valid"],
                "promotionEligibleSources": result["promotionEligibleSources"],
                "shadowEligibleSources": result["shadowEligibleSources"],
                "researchOnlySources": result["researchOnlySources"],
                "output": str(output.relative_to(lens.ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

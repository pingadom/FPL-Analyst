# Prospective FPL research runbook

This protocol exists to stop post-deadline knowledge from leaking into model claims. Historical replay remains useful for rejecting weak ideas, but only the frozen shadow season can promote the selectable-frontier challenger.

## Weekly sequence

1. Run `npm run research:cycle` whenever the official schedule changes and again after meaningful injury or press-conference news. It refreshes the live listwise horizon/captain ranks before the chip and shadow decisions. These runs are provisional and do not change recursive manager state.
2. Put documented expected-minutes changes in `analysis/inputs/deadline_overrides.json`. Every row requires a player ID, expected minutes from 0–90, start probability from 0–1, confidence from 0–1, source, reason, and a timezone-aware `updatedAt`. Future-dated overrides are rejected.
3. Immediately before the official deadline, run:

   ```powershell
   python analysis/capture_deadline_snapshot.py --lock
   python analysis/fit_live_listwise.py
   python analysis/chip_scenario_planner.py
   python analysis/run_shadow_cycle.py --lock
   ```

   The lock is rejected after the deadline. Each snapshot contains the official player availability fields, fixture schedule, game settings, expected-minutes intelligence, schedule fingerprint, and SHA-256 content hash.
4. After the gameweek is complete, run `npm run research:score -- <gameweek>`. The scorer reads only the locked decision and official event points, applies captain/vice and legal autosubs, and appends the result. A scored gameweek cannot be overwritten.

## Pre-registered managers

- **Structural control:** Lens projections, exact legal optimiser, recursive transfer hurdle, no chips.
- **Structural + scenarios:** identical player evidence with modest risk utility and paired Monte Carlo chip gates.
- **Hybrid challenger:** a frozen 25% selectable-frontier next-GW rerank, 25% listwise six-week transfer rerank, 50% captain rerank and the same chip gates.

All three receive the same deadline snapshot. Provisional decisions are displayed for inspection but are never included in results.

## Chip gates

The planner uses 5,000 paired draws with common player outcomes and correlated team shocks. A positive mean is insufficient:

- Free Hit also requires a materially broken current structure, a 10-point mean, non-negative P10, and at least 75% probability of a gain.
- Bench Boost requires a bench double, a 10-point mean, and at least four points at P10.
- Triple Captain requires a captain double, a 12-point mean, and at least four points at P10.
- Wildcard is unavailable in GW1 and requires a 25-point six-week mean, at least five points at P10, and at least 75% probability of a gain.
- A qualifying chip must still beat the estimated value of saving it for an already-announced future blank or double.

## Promotion gate

The tree challenger is research-only. Do not promote it on its historical average. Review after a meaningful frozen sample using total points, points per decision, frontier ranking, high-return miss rate, transfer regret, calibration, minimum-season/downside behaviour, and operational failures. Any feature or threshold changed after observing a result starts a new challenger version; prior prospective weeks stay attached to the old version.

## Current status

The first 2026/27 GW1 snapshot and decisions are provisional. No prospective gameweek has been scored yet, so neither the top-500k target nor the challenger has new forward evidence.

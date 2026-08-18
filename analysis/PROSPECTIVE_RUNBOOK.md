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
- **Scoring-route captain:** the same squad, lineup and chip layers as the hybrid challenger, with captaincy ranked by 85% frozen captain rank and 15% five-seed causal scoring-route rank plus a tiny defender tie-break. Raw same-fixture xG/xA/xGA are forbidden. The former late-action manager was disabled after its leak-free revalidation failed.

All four receive the same deadline snapshot. Provisional decisions are displayed for inspection but are never included in results.

The historical pre-closing market captain boundary is not yet a fifth manager.
It can be enrolled only after market prices are captured and hashed inside the
same immutable pre-deadline snapshot. Closing prices, prices collected after the
FPL deadline and manually reconstructed historical prices are forbidden.

## Chip gates

The planner uses 5,000 paired two-part draws with explicit no-shows, legal
autosubs, captain-to-vice fallback and correlated club shocks. A positive mean is
insufficient:

- Every manager has distinct H1/H2 inventories. Only one chip may be used in a
  Gameweek; reservations decay as GW19/GW38 approaches and collapse when expiry
  creates a forced chip collision.
- Free Hit is decided before permanent transfers, uses the manager's selling
  budget, subtracts the value of the permanent move it displaces, and requires
  announced structural damage plus a conservative downside gate. It is a shadow
  challenger, not an automatic production rule.
- Bench Boost uses points above normal autosubs. It may qualify on a double or on
  a highly reliable four-player bench, but must beat its causal historical
  reservation value with a non-negative lower tail.
- Triple Captain may qualify on a double or an elite premium single fixture. It
  must beat its causal reservation value with strong minutes and downside.
- Wildcard is scored over the horizon at the manager's actual budget. Automatic
  activation remains disabled after negative historical tests; any live WC
  recommendation needs manual review of injuries, roles, AFCON, fixture swing,
  setup cost and terminal squad value.
- WC -> BB, FH blank and TC double sequences are enumerated only after the
  relevant fixtures are officially announced. The hold sequence always remains
  available.

## Promotion gate

The tree challenger is research-only. Do not promote it on its historical average. Review after a meaningful frozen sample using total points, points per decision, frontier ranking, high-return miss rate, transfer regret, calibration, minimum-season/downside behaviour, and operational failures. Any feature or threshold changed after observing a result starts a new challenger version; prior prospective weeks stay attached to the old version.

## Current status

The first 2026/27 GW1 snapshot and decisions are provisional. No prospective gameweek has been scored yet, so neither the top-500k target nor the challenger has new forward evidence.

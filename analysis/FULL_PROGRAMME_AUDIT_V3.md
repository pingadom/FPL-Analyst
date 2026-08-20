# Full FPL modelling programme audit v3 — 20 August 2026

## Corrected outcome

The earlier claim that every planned stage was complete was wrong. This audit
finishes the omitted work and makes each accepted or rejected branch explicit.

The fully integrated research candidate averages **2,212.0 points** across
2018/19–2025/26. That is **+21.4** over the 2,190.6 route control and **+17.8**
over the earlier 2,194.2 forecast-v2 model. It improves all eight seasons over
the original control, including **+34.5** on the two untouched recent holdouts.

This is the strongest recursive result in the project, but it is not yet a
consistent top-500k model. It clears 2 of 8 reconstructed cutoffs and remains
84.8 points below the 2,296.8 average pace. The correct status is a frozen
research shadow candidate, not production promotion and not a rank guarantee.

## Like-for-like performance

| Stage | Average | Change vs control | Holdout average | Top-500k hits |
|---|---:|---:|---:|---:|
| Original route control | 2,190.6 | — | 2,170.5 | 0/8 |
| Earlier forecast v2 | 2,194.2 | +3.6 | 2,175.0 | 1/8 |
| New captain, old chips | 2,197.8 | +7.1 | 2,179.0 | 1/8 |
| New captain, no chips | 2,186.9 | −3.8 | 2,164.5 | 1/8 |
| Fully integrated winner | **2,212.0** | **+21.4** | **2,205.0** | **2/8** |

Integrated season totals are 2,135, 2,186, 2,271, 2,312, 2,274, 2,108,
2,365 and 2,045. Relative to the original control, the changes are +6, +30,
+33, +7, +16, +10, +56 and +13. No evaluation season is worse.

## Work that was actually completed

### 1. Real rollout-return learning

The rollout label replays **70,247** legal historical action packages over
1/3/6/10-event futures with autosubs, captain-to-vice fallback, hits and
player-specific tenure. The trained correction reduces action-value MAE from
13.042 to 10.460, but its best recursive policy loses 13.1 points per season,
loses 39.5 on holdout and is negative in six seasons. It is rejected.

This replaces the earlier generic off-policy evaluator with an actual realised
rollout target. A doubly robust historical claim is still intentionally refused
because logged behaviour propensities do not exist.

### 2. Action-specific horizons

The recursive test now applies h1 to captain/XI/Free Hit, h3 to the bench,
player-specific 1/3/6/10 tenure to transfers, and h10 to initial/Wildcard squad
construction. The best transfer-after-120-minutes variant loses 11.8 points on
average and 18.7 on development, despite +9.0 on holdout. It is rejected.

The six-week horizon is therefore no longer a hard universal assumption, but
the tested replacement is not allowed to alter the transfer policy.

### 3. Large captain search

**1,655** captain configurations were screened on fixed historical XIs. Only
the top 12 by first-six-season stability received exact recursive replays; the
last two seasons were retained for the gate.

The winner is:

- 80% scoring-route consensus;
- 20% dynamic match-route rank;
- 0.70 dynamic-route strength;
- 0.50 downside-only causal minutes protection;
- no OpenFPL player-ensemble share.

It averages 2,197.8 with the old chip policy: +7.1 overall, +6.7 on development,
+8.5 on holdout, a −5 worst-season delta and six positive seasons. It passes
the predeclared gate and replaces the earlier 90/10, strength-0.30 captain.

### 4. Chip retuning and Wildcard falsification

**756** causal BB/TC/FH threshold policies were screened and the top eight were
exact-replayed. Selection used development stability only. The winner uses:

- Bench Boost signal threshold: 9.0;
- Triple Captain signal threshold: 10.0;
- Free Hit learned net threshold: 3.0;
- Free Hit residual-risk discount: 0.0;
- automatic Wildcard: disabled.

Against the identical no-chip path, it adds 25.1 points per season, adds 20.0
on development and 40.5 on holdout, and improves all eight seasons. Its exact
season gains are +3, +20, +44, +30, +16, +7, +67 and +14.

Six h10 recursive Wildcard variants were then tested only after BB/TC/FH was
frozen. The best loses 15.8 points per season, loses 20.5 on holdout and has an
−85 failure. Automatic Wildcards remain disabled; the live planner may expose
a manual review signal but cannot activate one except unavoidable chip-expiry
handling.

### 5. Live integration repairs

The historical winner is frozen in `forecast_champion_v2.py`. The prospective
captain path now fits the probabilistic minutes model through completed history,
uses the selected 80/20 and 0.70 parameters, and falls back to route consensus
when an exact current market snapshot is unavailable.

The chip planner now includes the forecast-v2 shadow manager. Previously it
omitted that manager and used the structural control XI/captain inside every
manager-specific simulation. It now consumes each manager's own planning,
lineup and captain score, and the forecast manager receives the frozen 9/10/3
threshold profile.

## Remaining barrier

The improvement reduces the average top-500k gap from 106.1 to 84.8 points, but
the remaining loss is concentrated in 2022/23, 2023/24 and 2025/26. That pattern
does not support a simple “surprising teams” explanation. The more credible
causes are forecast calibration under regime change, incomplete deadline-vintage
market information, minutes/availability misses, and transfer paths that still
cannot exploit the weekly-rebuild ceiling consistently.

The next honest breakthrough target is not another broad hyperparameter sweep.
It is a locked prospective dataset containing exact market capture timestamps,
predicted lineups and news revisions, followed by residual attribution for every
missed starter, captain and transfer. Historical improvements remain vulnerable
to dataset shift until that sample exists.

## Reproduction and artifacts

Run:

```powershell
python analysis/rollout_action_value_v2.py
python analysis/action_specific_recursive_v2.py
python analysis/captain_surface_search_v2.py
python analysis/chip_surface_search_v2.py
python analysis/final_breakthrough_validation_v3.py
python -m unittest discover analysis -p "test_*.py"
pnpm test
pnpm lint
```

The decisive integrated result is
`analysis/data/final_breakthrough_validation_v3.json`. Supporting artifacts are
`rollout_action_value_v2.json`, `action_specific_recursive_v2.json`,
`captain_surface_search_v2.json` and `chip_surface_search_v2.json` in the same
directory.

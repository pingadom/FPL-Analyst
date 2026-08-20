# Availability and held-player audit

## What this audit asked

The model can be wrong for two very different reasons:

1. a good, healthy player receives fewer FPL returns than expected; or
2. information available before the deadline already showed that the player
   was unlikely to play, had left the club, or had become a poor use of a
   transfer.

Only the second class is automatically a model failure. Selling every player
after three blanks would chase noise, spend scarce transfers and often miss the
next return. This audit therefore keeps future replacement points separate from
the evidence the model actually possessed at the deadline.

## Headline result

Using the same final Lens 8 weights, chip policy and recursive decision policy
in every evaluation season, the selected repair scored **2,103.4 points on
average**, compared with **2,096.6** before the repair.

This is a fixed-policy forensic diagnostic, not the published walk-forward
benchmark and not a new rank estimate. The average gain is only 6.8 points and
the season deltas are uneven, so the result supports the causal-integrity repair
but does not prove a universal performance breakthrough.

| Audit measure | Before | Selected repair | Change |
|---|---:|---:|---:|
| Average points | 2,096.6 | 2,103.4 | +6.8 |
| Starter no-shows | 249 | 227 | -22 (-8.8%) |
| XI selections below 70% play probability | 33 | 26 | -7 (-21.2%) |
| Four-start return droughts | 113 | 107 | -6 |
| Starting weeks inside those droughts | 531 | 496 | -35 |
| Forecast shortfall inside droughts | 1,774.4 | 1,664.8 | -109.6 |

The season point changes were `0, -21, -63, +8, +30, -30, +201, -71`.
That variance is why the +6.8 average is reported cautiously.

## The serious errors found

### 1. Stale registered players could survive in the squad

The clearest example was Harry Kane in 2023/24. The old replay selected him for
five starting XIs after his transfer to Bayern. The archive contained official
FPL `xP=0` and extreme transfers out, but the model discarded the official
field and trusted Kane's older elite performance prior.

The selected repair uses an official zero only when it is corroborated by one
of the following deadline-known signals:

- extreme transfers out;
- a previous official zero plus a no-show; or
- a curtailed previous appearance plus extreme selling.

The 2023/24 Kane drought and the 2019/20 Laporte injury drought disappear from
the repaired held-player audit.

### 2. Some archived official fields are corrupted

Several gameweek snapshots contain `xP=0` for almost the entire player pool.
Treating those rows as injuries would be disastrous. A per-gameweek source
quality gate now checks the share of zero values among widely owned players. If
the cross-section is implausibly all-zero, the signal is disabled before any
player is scored.

This check uses only information present at that deadline. It does not inspect
the later teamsheet or points to decide whether the feed was trustworthy.

### 3. Live status could resurrect an unavailable player

The live generator and deadline API treated a missing
`chance_of_playing_next_round` value as 100. That is correct for status `a`
(available), but wrong for injured, suspended, unavailable or not-in-squad
statuses.

The fallback is now:

| Official status | Chance when the numeric field is null |
|---|---:|
| Available (`a`) | 100% |
| Doubtful (`d`) | 75% |
| Injured, suspended, unavailable or not in squad | 0% |

The live overlay now propagates the change through expected minutes, projected
points, six-week value, components, return/haul distributions, captain rating
and the browser optimiser's features. Previously it changed only the displayed
probabilities.

### 4. Free Hit bookkeeping mixed temporary and permanent squads

An audit path exposed a crash where bench-spend diagnostics priced the temporary
Free Hit bench against the reverted permanent squad. Scoring was already kept
separate, but the post-chip structure diagnostic was not. It now recomputes the
persistent bench after the Free Hit reverts.

### 5. Exact optimisation repeatedly allocated dense matrices

The full all-gameweek audit exhausted memory because the exact MILP rebuilt the
same mostly-empty constraint matrix several times per solve. It now constructs
one sparse matrix and reuses it for the feasibility, maximum-spend and objective
solves. The constraints and optimum are unchanged.

### 6. Sensible constraints can become jointly infeasible

At one Wildcard deadline, the £99.5m minimum spend, cheap-bench cap and XI
availability floor had no joint feasible solution. The optimiser now proves the
maximum legal spend with a separate MILP and relaxes only the impossible part
of the spend floor. There is still no greedy fallback.

## Which long holds were actually predictable?

The repaired audit finds 107 runs of at least four consecutive starting-XI
scores below five points:

- **64 rough patches / outcome variance.** Minutes and projection stayed
  competitive. A sale would mostly be hindsight chasing.
- **37 transfers prioritised elsewhere.** A local replacement eventually
  cleared the hurdle, but the model used the finite transfer on a different
  squad problem. These are path-allocation cases worth further joint-transfer
  research.
- **5 mixed warnings.** Deterioration was visible but no affordable replacement
  independently cleared the configured hurdle.
- **1 foreseeable availability decay.** Repeated absences and the causal
  minutes signal aligned, yet the response remained too slow.

There are no cases where an affordable replacement cleared the actual planning
hurdle, no competing transfer was made, and the player was nevertheless held.

Examples:

- **Teemu Pukki, 2019/20 GW6-12:** seven starts, no no-shows, secure minutes and
  no replacement clearing the hurdle. This was a genuine scoring drought, not
  an obvious pre-deadline sale.
- **James Rodríguez, 2020/21 GW5-11:** the retrospective source identifies a
  groin injury in GW7. Grealish only cleared the causal planning hurdle from
  GW9, when the model allocated its transfer elsewhere. The early injury flag
  is precisely where archived official status snapshots would improve the
  replay.
- **Phil Foden, 2025/26 GW17-23:** played throughout the drought. A replacement
  route only became strong late; this is mainly transfer prioritisation, not an
  injury miss.
- **Bruno Fernandes, 2025/26 GW17-20:** three no-shows are retrospectively
  explained by a hamstring injury. That season's archived `xP` feed is corrupted
  for many weeks, demonstrating why a genuine pre-deadline status archive is
  still needed.

The retrospective availability source explains an injury or suspension inside
24 of the 107 droughts. It is not used to claim those absences were predictable;
many injuries began after the drought started.

## Injury and availability sources

### Live decisions: official FPL API

Use `bootstrap-static` immediately before every deadline. It provides player
status, chance of playing, news and the news timestamp. The repository already
captures these fields in immutable deadline snapshots.

`https://fantasy.premierleague.com/api/bootstrap-static/`

### Historical causal replay: archived official snapshots

For a proper historical injury feature, use the latest archived
`players_raw.csv` commit strictly before the simulated deadline and reject any
news timestamp later than that deadline. The vaastav FPL repository is the most
practical public archive for this reconstruction.

`https://github.com/vaastav/Fantasy-Premier-League`

This is the remaining recommended data-engineering task. A final end-of-season
`players_raw.csv` must never be backfilled into earlier decisions.

### Retrospective explanation: availability-data

The audit integrates `withqwerty/availability-data`, which covers Premier
League matchdays from 2015/16 onward and labels starts, substitute appearances,
bench, injury, suspension, international duty, not-in-squad and not-at-club.
Injury rows include a description and expected return.

`https://github.com/withqwerty/availability-data`

It is Transfermarkt-derived, crowd-sourced and scraped after matches. It is
useful for explaining a no-show, but feeding it into a historical deadline
forecast would leak future knowledge.

## Rejected fixes

The audit deliberately tested and rejected attractive-looking overcorrections:

| Variant | Average points | Decision |
|---|---:|---|
| Original fixed-policy diagnostic | 2,096.6 | Comparison |
| Broad downgrade for every official `xP=0` | 2,066.8 | Reject |
| Broad downgrade + 78% weekly XI floor | 2,050.5 | Reject |
| Broad downgrade + penalised transfer hurdle | 2,073.6 | Reject |
| Broad downgrade + both changes | 2,057.2 | Reject |
| High-precision corroborated signal | **2,103.4** | Select |
| High-precision signal + penalised transfer hurdle | 2,077.0 | Reject |

The lesson is important: a warning can be valuable for screening candidates
without being safe to apply twice in expected minutes, XI eligibility and the
transfer hurdle. The selected policy acts only on high-confidence absence risk.

## Remaining work

1. Build weekly, pre-deadline official status history from archived snapshots.
2. Calibrate injury recurrence and expected-return priors by injury type without
   using future matchday labels.
3. Evaluate exact multi-transfer allocation, because 37 droughts had a good
   local exit while the transfer was used elsewhere.
4. Re-run the full season-by-season walk-forward calibration before changing
   the published ranking benchmark.
5. Require several live shadow deadlines before promoting any injury-news model
   into the public recommendation policy.

## Reproduction

```bash
python analysis/availability_repair_validation.py
python analysis/held_player_audit.py
python analysis/retrospective_availability_audit.py
python -m unittest discover -s analysis -p "test_availability_repair.py"
```

Generated evidence:

- `analysis/data/availability_repair_validation.json`
- `analysis/data/held_player_audit.json`
- `analysis/data/retrospective_availability_audit.json`

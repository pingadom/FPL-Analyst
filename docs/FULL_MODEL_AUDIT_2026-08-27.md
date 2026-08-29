# Full model audit — 27 August 2026

Updated through 29 August after deadline-selection and browser-objective parity
checks.

## Bottom line

The model is materially safer and more honest than the version that entered this
audit, but it has **not** demonstrated consistent top-500k performance. Two numbers
must remain separate:

| evaluation | average | top-500k pace | meaning |
|---|---:|---:|---|
| broad walk-forward research search | 2,175.5 | 2/8 seasons | useful diagnostic; evaluation-era choices were compared |
| fingerprinted frozen policy | 2,119.9 | 0/8 seasons | promotion benchmark; selected before the eight evaluation seasons |

The research result is 2.4 points below the preceding 2,177.9 run. That is noise,
not a breakthrough. The frozen result is 23.1 points above the old 2,096.8 file,
but the old file was stale and cannot support a clean causal comparison.

## What was wrong and what changed

### 1. A fit current player could inherit last season's absence streak

The live path used the final 2025/26 absence run. That made an available Haaland,
who had played 90 minutes in 2026/27 GW1, look like a major minutes risk.

The streak is now rebuilt from official **current-season** event histories before
every deadline. A blank is neutral; a current-season appearance resets the run;
and a new player with no current-season sample receives a neutral streak. In the
29 August GW3 refresh, Haaland has 89% start probability, 92% play probability
and 79 expected minutes.

### 2. The decision gate counted correlated variants as extra evidence

Several weight variants produced paths through the same football seasons. Treating
each path as independent shrank the standard error without adding independent
seasons. The gate now averages candidate variants within each season-week and
bootstraps season-level evidence. The artifact records the evidence unit and the
number of averaged variants.

### 3. A stale frozen audit could be published

The previous public build could load an old audit after the engine or feature frame
changed. A frozen audit now contains SHA-256 fingerprints for the engine, prepared
frame, cache schema, evaluation seasons and its own canonical content. Calibration
and live refresh fail closed if any fingerprint differs.

### 4. Captaincy differed between Python and the browser

Python selected the captain by projected points, but the browser re-sorted the XI
using a percentile blend containing ownership and safety. Captaincy is worth one
extra copy of the player's realised Gameweek score, so ownership cannot increase
its expected value. Both paths now choose the highest projected points in the XI;
the old blend remains display-only.

### 5. European rest was directionally wrong and evaluation-fitted

The old flag used an absolute day gap, penalising league matches both before and
after a European tie. The measured effect was anticipatory rotation before a tie,
not fatigue after it. The flag now means an upcoming tie only. Its production
coefficient is zero because 0.10 was estimated on seasons also used for evaluation;
it can return only after a pre-2018 or prospective frozen test.

### 6. Exact calibrated weights were rounded in the experiment harness

The artifact stores exact and display-rounded weights. The harness reconstructed
the reference configuration from the rounded percentages, so an alleged one-change
ablation did not actually reproduce the reference. It now requires `weights.exact`
and refuses a rounded fallback. The fast live-refresh path now does the same; an
old artifact must receive a full calibration instead of an approximate refresh.

### 7. Governance labels overstated what had been achieved

The old governance object called a target threshold an incumbent result and mixed
research-search performance with production evidence. The schema now distinguishes
target points, target hits, frozen-audit performance, research performance and
forecast error.

### 8. The UI described an underperforming shadow model as accepted

The causal shadow trailed the current research replay by 92.7 points, yet static
copy said it added points and listed it under accepted experiments. Audit generation
now derives accepted/rejected status and signed wording from the actual comparison.

### 9. Opta fixture coverage was stale

The site reported 10/10 Opta coverage from a ten-match **GW1** file while displaying
GW2. None of those fixtures matched the live slate. The loader now requires the
snapshot season and Gameweek to match the deadline and reports matched fixture keys,
not the number of rows in the file. The GW2 file was refreshed from Opta's published
27 August probabilities. On 29 August the model moved to GW3 before Opta had
published MD3 probabilities, correctly reported 0/10 and excluded the stale file.

### 10. The current-player count was easy to misread as broken fixture coverage

Official FPL listed 614 registered players at the original audited deadline. The old
published set contained 368 because it required minutes, ownership or an official
xP signal in addition to availability. That excluded 146 nominally available
players, including late signings and GW1 participants, and was not defensible: a
forecast should not require popularity before it will consider a player. The
screen is now availability-only (status `a`/`d`, at least 75%, legal price), and
registered, eligible and excluded counts are published separately. Every eligible
candidate must have a current fixture.

### 11. The optimizer test forced a famous player instead of testing correctness

A test required Haaland in every squad. That is not a model invariant: at £15.5m he
can be rationally excluded if a cheaper player offers the same captain ceiling and
the redistributed budget adds more XI points. The replacement test requires his
availability to be correct, his projection to remain near the slate maximum, the
squad to be legal and exact, and the selected captain to be the XI projection leader.

### 12. Free Hit and chip reporting needed sharper semantics

Free Hit construction is explicitly one-week-only and restores the persistent squad.
Historical chip windows match the rules of their seasons; 2025/26 and 2026/27 have
two sets of four chips, while older seasons have the appropriate earlier rules.

The `gain` beside an individual chip is its realised one-Gameweek increment. A
season's `chipPoints` is the difference between two independently evolving recursive
policies, with and without chips. It includes downstream squad-path effects and
therefore does not equal the sum of the logged one-week gains. A negative season
delta is possible even when each immediate chip gain is positive.

### 13. The browser still optimised a near-copy of the Python objective

The website divided every planning horizon by six, even though Python divides by
the actual censored fixture weight. It also charged 0.22 points per £1m of bench
premium where Python charged 0.18, reconstructed utility from rounded display
fields and independently rounded the XI-eligibility boundary. Those differences
can change a squad around blanks, doubles and close price trade-offs.

Every player payload now carries the exact risk-adjusted horizon, weighted fixture
count, lineup utility for all three risk modes, bench utility, captain utility and
Python's starter/exception flags. The untouched browser preset consumes those exact
values. The human sliders deliberately leave the preset and calculate a custom
model. A regression test requires the default browser XV and captain to match the
generated Python selection.

### 14. A locked live Gameweek was still treated as selectable

The pipeline previously chose the first event whose matches were not all finished.
Once a deadline passed, it could therefore keep re-optimising a team users could no
longer submit. It now targets the first future official deadline. Only fully
finished events enter absence evidence, so an in-progress round is neither treated
as prospective nor smuggled in as a partial result.

## Live squad checks

The 29 August GW3 refresh publishes 506 eligible candidates from 622 registered
players, with no missing fixture. Its 3-5-2 squad:

- costs £100.0m; the exact
  optimiser permits £99.5m–£100.0m so it never downgrades expected points merely
  to spend a final price increment;
- spends £16.5m on Dovin, Mendy, Ajayi and Walle Egeli—the positional price floor,
  with zero bench premium;
- contains no XI start/play exceptions;
- satisfies 2 GK, 5 DEF, 5 MID, 3 FWD and at most three players per club;
- starts eleven players with at least 70% start and 84% play probability;
- captains Haaland at the 5.7-point slate ceiling, with Cherki vice-captain; and
- contains no missing fixture among the eligible candidates.

Haaland is selected on merit rather than by a named-player rule. The model is free
to exclude him in a different slate if his price prevents a stronger XI, but it must
retain him in the eligible pool whenever his official availability clears the same
rules as every other player.

## What the result still says is weak

The frozen policy is 176.9 points per season below estimated top-500k pace and has
zero hits in eight seasons. The research path is volatile: 2,352 in 2020/21 but
1,987 in 2025/26. Chip-policy trajectory deltas range from -52 to +227, showing that
the long-run interaction between Wildcards, transfers and later squads remains much
less stable than the immediate chip-value estimates.

The next modelling work should therefore be a pre-registered prospective test, not
another evaluation-era parameter search. Freeze the current policy and a small set
of challengers before future deadlines; archive every input snapshot; score forecast,
minutes, captain, transfer and chip regret separately; and promote only on a declared
season-level rule.

## Reproduction and release gate

The release is accepted only when:

1. the frozen audit fingerprints match the current engine and prepared frame;
2. the full Python regression suite passes;
3. all Node/API/optimizer tests pass;
4. the production build succeeds; and
5. generated public metrics agree with the source artifacts.

The machine-readable sources are `analysis/data/audited_policy_validation.json`,
`app/data/model-results.json`, `app/data/current-players.json` and
`app/data/model-audit.json`.

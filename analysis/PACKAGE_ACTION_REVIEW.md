# Transfer-package action review - 2026-08-13

## Decision

The package-action challenger is rejected. The frozen 2,174.9 no-chip and
2,192.4 chip champions remain unchanged. No package challenger is sent to the
prospective shadow, and no chip interaction test is justified because the
transfer model failed first.

## What was implemented

- `joint_transfer_plan` now has an optional research callback that can score a
  complete legal package. The default is `None`, preserving normal production
  behaviour. A regression test proves the hook can rerank a package, and the
  zero-valued collector reproduced the frozen 2,174.9 baseline exactly.
- 17,125 legal bundles were collected from the champion's recursive weekly
  state. A bundle contains its complete outgoing/incoming set, one or two moves,
  current bank, resulting bank, free transfers, formation, captain/bench access,
  club concentration, player forecasts, route distributions and uncertainty.
- Labels use formation-aware, player-specific adaptive holding value rather than
  a fixed six-Gameweek target. Fits are causal at GW1/GW13/GW25: prior seasons
  plus only same-season labels whose complete horizon has matured.
- Goalkeeper packages always abstain because the earlier action audit found no
  goalkeeper regret improvement.
- Two differently regularised and recency-weighted models must agree. The model
  was tested first as a residual value correction, then as a classifier allowed
  only to veto a champion transfer; it could never manufacture a new transfer.

## Results

Across 11,586 evaluation packages with a causal fit:

- package MAE improved from 11.953 to 9.544 adaptive points;
- correlation improved from 0.2299 to 0.2703;
- positive-value classification was only 57.86%;
- Brier score was 0.2466, barely informative for a binary outcome.

Those broad metrics concealed the decisive failure. Only 169 of 17,125 generated
packages were selected by the champion across all ten seasons. On 118 causal,
non-goalkeeper evaluation decisions at that optimiser boundary:

- 72.88% of chosen packages had positive adaptive value;
- always predicting positive was therefore 72.88% accurate;
- the learned classifier was only 65.25% accurate;
- its probability/value correlation was 0.0436.

The learner could distinguish obviously poor packages in the broad pool but not
the difficult, already-filtered decisions where it was needed. This is the
offline-policy version of optimiser's curse.

## Recursive tests

The direct value overlay was strongly negative. Its best development-selected
variant averaged 2,147.8 versus 2,174.9. A conservative veto-only redesign was
then tested:

- 15% probability, one-move-only veto: exactly 2,174.9; it changed no scored
  path and added no value;
- every veto threshold that changed meaningful decisions reduced the holdout;
- the 20% one-move veto averaged 2,173.5;
- less conservative two-move gates fell as low as 2,161.4 with a 1,996 minimum.

The package audit also found that two-move candidates were overvalued by 13.53
adaptive points on average, versus 5.26 for one-move candidates. This motivated
a clean isolation of the hard-coded extra-move hurdle. Raising it from 1.15 to
3/5/8/12 was uniformly harmful; the least-bad challenger averaged 2,127.4. The
structural champion's apparently cheap second-transfer hurdle is therefore not a
simple bug: its scale is already mediated by the joint squad utility and beam.

## Why the experiment failed

1. The action labels are counterfactual only at the player-value level. They do
   not replay the entire future manager state after each alternative package.
2. The champion chooses only around 17 packages per season, leaving very sparse
   on-policy training evidence.
3. Packages deep in the rejected pool are easy to classify but not relevant to
   the decision boundary.
4. A transfer changes bank, later free transfers, future choices and chip setup.
   A static adaptive target captures only part of that continuation value.

## Counterfactual trajectory follow-up

The proposed forked-state experiment was subsequently implemented rather than
left as future work. At every frozen deadline it retained the champion package
plus the closest Hold-side and Act-side one/two-transfer alternatives. Hold and
Package branches were replayed under the frozen policy with separate squads,
banks and free transfers. Labels include legal later transfers, selling prices,
XI, autosubs, captain fallback and realised points. Windows crossing an unlimited
rebuild were excluded.

Two horizons were evaluated independently:

| horizon | valid rows | frontier classifier | chosen-package classifier | best recursive effect |
| --- | ---: | ---: | ---: | ---: |
| 3 events | 773 | 52.51% vs 56.27% majority | 60.75% vs 63.55% majority | +0.3 average, no development gain |
| 6 events | 696 | 51.10% vs 52.51% majority | 54.64% vs 64.95% majority | +0.3 average, no development gain |

Both true continuation-value models still fail to discriminate at the decision
boundary. Their safest thresholds reproduce the champion; thresholds strong
enough to alter its development path lose points. The +0.3 effects are tiny,
selected only after historical exposure and fail the predeclared development
gate. They are not promotions or shadows.

This closes the present package-policy avenue. The limiting factor is no longer
the absence of a continuation label: it is the small number of difficult
on-policy actions and the inability of deadline features to predict their noisy
realised difference. Further threshold tuning on these seasons is prohibited.

## Next credible route

Shift effort back to information quality—especially starts/minutes, tactical
role changes and causal team-strength regimes—and gather frozen prospective
decision data. A future offline-RL revisit would need many behaviour policies
to create diverse manager states, season-blocked cross-fitting and a much larger
on-policy action sample; the current ten-season champion path is insufficient.

## Reproduction

```powershell
python analysis/package_action_value_validation.py
python analysis/package_boundary_audit.py
python analysis/package_hurdle_validation.py
python analysis/package_trajectory_validation.py
$env:FPL_TRAJECTORY_HORIZON='6'; python analysis/package_trajectory_validation.py
python -m unittest discover analysis -p "test_*.py"
```

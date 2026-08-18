# Robust transfer-decision review

> **Leakage correction, 2026-08-14:** the route feature set used by this phase
> contained same-fixture raw xG, xA and xGA. Those are post-match observations,
> so every action result in this document is invalidated. Version-2 leak-free
> retraining produced a five-seed median of -3.4 points per season and no stable
> ensemble; the action shadow has been disabled. The unaffected frozen control
> remains 2,174.9 without chips and 2,185.5 with audited chips. The replacement
> captain-only experiment is documented in `CAPTAIN_DECISION_REVIEW.md`.

## Outcome

The robust-decision phase found a live-computable transfer challenger, but did
not promote it to production. The frozen policy remains the control. A fourth,
exploratory prospective manager now records the new policy from the same locked
deadline snapshots.

The selected research policy is deliberately narrow:

- no action before GW25;
- no goalkeeper reranks;
- compare players within two overlapping near-price bands;
- fit five independently seeded causal LambdaMART action models;
- require at least four of five models to agree on direction and require each
  agreeing seed to agree across both price bands;
- blend only 5% of the consensus rank into the structural transfer horizon.

This directly addresses the earlier failure mode where a tiny point-estimate
change caused a completely different recursive squad path.

## What failed first

Correlated 128-scenario transfer packages were audited before any new tree
policy. Chosen packages cleared the structural control in 64.7% of scenarios on
average, but hard 55% and 65% win gates reduced the recursive average by 93.0
and 128.1 points. Soft tail and consensus penalties also failed. Scenario
dispersion is useful evidence, but the existing transfer utility does not yet
turn it into a better hard decision rule.

The first deployable action model also failed. Restricting the ranker to the 18
features already crossing the old live JSON boundary scored 2,165.5 in its
original validation. The feature-family rerun scored 2,173.6 with a 2,017
minimum, still below the 2,174.9 / 2,031 frozen control.

## Feature-boundary diagnosis

The live recommendation builder was already calculating most of the required
information, but discarded it when serialising player rows. The audit tested
the missing feature families causally and recursively.

| Variant | Average | Minimum | Recent holdout average | Decision |
| --- | ---: | ---: | ---: | --- |
| Frozen structural plan | 2,174.9 | 2,031 | 2,154.5 | Control |
| 18 live fields | 2,173.6 | 2,017 | 2,163.2 | Reject |
| 18 fields + learned routes | 2,170.6 | 2,032 | 2,158.8 | Reject |
| Extended live context | 2,171.0 | 2,017 | 2,154.0 | Reject |
| Extended context + routes, one seed | 2,182.1 | 2,067 | 2,184.2 | Seed audit required |
| Extended context + routes, 80% seed consensus | 2,186.1 | 2,067 | 2,172.5 | Exploratory shadow |

Adding either historical horizon forecast to the extended route model in
isolation reduced performance. The raw component horizon scored 2,169.4 and
the causal horizon scored 2,172.4. They were excluded from the live policy.

## Seed and consensus audit

Five independent fits of the extended route ranker produced average deltas of
approximately +7.2, +5.2, -1.3, -1.3 and +11.0 points. The median was +5.2,
but the lower 20th percentile was -1.2, so a single stochastic fit failed the
declared seed-robustness gate.

Simple mean and median ensembles remained unstable. Four-of-five directional
voting at the fixed 5% blend recovered the strong recursive path:

- no-chip average: 2,186.1, +11.2;
- minimum: 2,067, +36;
- season deltas: 0, 0, -13, 0, 0, +67, 0, +36;
- recent two-season holdout: 2,172.5 versus 2,154.5.

This result is still concentrated: two seasons improve, one declines and five
are unchanged. The stricter development/holdout historical selection rule in
`live_action_ensemble_validation.json` therefore selected no production
candidate. Enrolling the policy in shadow is an experiment, not a claim that
the historical gate was passed.

## Final chip and captain interaction

The exact live-compatible consensus policy was replayed with the unchanged
audited chips and the fixed 15% captain-ceiling challenger.

| Policy | Average | Minimum | Holdout | Average top-500k gap |
| --- | ---: | ---: | ---: | ---: |
| Frozen, no chips | 2,174.9 | 2,031 | 2,154.5 | -121.9 |
| Live action, no chips | 2,186.1 | 2,067 | 2,172.5 | -110.6 |
| Frozen + audited chips | 2,185.5 | 2,031 | 2,169.0 | -111.2 |
| Live action + audited chips | 2,197.9 | 2,069 | 2,188.0 | -98.9 |
| Live action + captain 15% + chips | 2,200.2 | 2,068 | 2,193.5 | -96.5 |

Compared with the frozen chip policy, the action policy adds 12.4 points and
raises the minimum by 38. The full action/captain/chip combination adds 14.7
points and raises the minimum by 37. It still records zero reconstructed
top-500k cutoff hits, so it is progress rather than the final target.

## Live implementation

- `current-players.json` now preserves 64 deadline-known research fields for
  every live player. These fields do not alter the production score.
- Terminal scoring-route models train on 74,277 completed, selectable-frontier
  player-weeks. Variance and the route stack are calibrated from causal
  historical out-of-fold predictions, not in-sample residuals.
- The action ranker fits five seeds and two price-band queries separately by
  position. Repeated complete fits produced a byte-identical
  `listwise-scores.json` artifact.
- GW1 correctly has zero active action players. The action manager matches the
  structural plan until the pre-registered GW25 boundary.
- Four prospective managers now receive the same immutable snapshot:
  structural control, structural scenarios, hybrid challenger and late action
  consensus.
- All 24 Python integrity tests pass.

## Governance and next evidence

No website-facing production recommendation was switched to the action model.
No historical result can promote it. The next evidence must come from locked
2026/27 decisions, with changes after observing a scored week versioned as a
new challenger.

Review the action manager on transfer regret, points per changed decision,
abstention rate, price-band agreement, calibration and operational failures.
Because the action gate begins in GW25, headline season totals alone will be
weak evidence; report the paired GW25-38 difference and every transfer the
challenger changed.

## Primary artifacts

- `scenario_transfer_validation.py` and `data/scenario_transfer_validation.json`
- `live_action_feature_ablation.py` and `data/live_action_feature_ablation.json`
- `live_action_seed_sensitivity.py` and `data/live_action_seed_sensitivity.json`
- `live_action_ensemble_validation.py` and `data/live_action_ensemble_validation.json`
- `live_action_final_interaction.py` and `data/live_action_final_interaction.json`
- `fit_live_listwise.py`
- `run_shadow_cycle.py`

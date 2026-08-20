# Lens 8 performance and reproducibility audit

## Bottom line

Lens 8 is a real improvement over the previous published model, but the old
2,212-point Breakthrough V3 headline was not a valid benchmark.

| Model or replay | Average points | Status |
|---|---:|---|
| Previous published model (Lens 7) | 2,048.1 | Valid comparison baseline |
| Lens 8 repaired model | 2,087.0 | +38.9 points; current deployed research model |
| Retrained causal shadow | 2,119.2 | +32.2 over Lens 8; research-only |
| Old Breakthrough V3 headline | 2,212.0 | Retired; not reproducible |
| Old stack through the repaired engine | 2,122.1 | Fair diagnostic replay; -89.9 from the old claim |
| Lens 8 chip-policy hybrid | 2,112.2 | Rejected; lost 7.0 to the coherent causal shadow |

Lens 8 recorded 2,103, 2,044, 2,076, 2,152, 2,158, 2,261, 1,968 and
1,934 points across 2018/19–2025/26. Its mean is 2,087.0. It cleared the
estimated top-500k pace in zero of eight seasons, so this audit does not claim
consistent top-500k performance or a dependable historical average rank.

## Why the old result fell

The old challenger loaded three learned prediction arrays from local `.npz`
caches. It checked only that the number of rows matched the current historical
frame. Lens 8 changed what those rows and features meant without changing the
total row count. Old-schema predictions could therefore be silently paired with
new-schema player-weeks.

That is a reproducibility failure: the replay appeared to use a stronger model,
but some of its inputs belonged to a different feature definition.

The repair adds a fingerprint containing the feature and target schema plus a
hash of the ordered historical frame. Frontier, horizon-listwise and captain
caches are accepted only when their version and fingerprint match. Otherwise
the models retrain. The repaired run retrained 36 frontier models, 36 horizon
models and nine captain models.

## What is genuinely better

The +38.9-point Lens 8 lift over Lens 7 survives the corrected full recursive
replay. It includes the fixture, availability, team-strength, regime, optimiser,
blank-week and club-transfer repairs described in the decision audit.

The retrained causal shadow reaches 2,119.2 without the stale-cache problem. It
is the strongest reproducible player-ordering challenger currently available.
It is not yet the website's live champion for two reasons:

1. it was inspected on the eight evaluation seasons, so those seasons cannot be
   reused as an untouched promotion test; and
2. its historical walk-forward estimators still need a separately tested final
   fit for scoring the current live player pool.

The model therefore separates three states:

- **deployed research:** Lens 8 and its current live recommendations;
- **shadow challenger:** the 2,119.2 causal stack; and
- **retired evidence:** the unreproducible 2,212 V3 headline.

## Why the repaired stack does not automatically replace Lens 8

A higher exposed historical average is evidence for further work, not automatic
permission to ship. Repeatedly choosing whichever system scores best on the same
eight seasons would overfit the policy to those seasons.

The next promotion test must freeze the causal model, live feature mapping,
optimiser and chip policy before scoring new deadlines or a genuinely untouched
historical archive. Prediction caches must carry matching fingerprints, and all
historical market data must have a verifiable pre-deadline timestamp.

## The failed hybrid matters

Combining the causal ordering stack with the Lens 8 chip policy produced 2,112.2
points. That is better than Lens 8 but worse than the complete causal replay.
This shows that model components are not independently additive: a chip policy
learned around one transfer and player-ranking process can conflict with another.
Future tests must compare whole decision systems as well as individual modules.

## Reproduction files

- `analysis/compare_lens8_shadow.py` runs the fair comparison.
- `analysis/data/lens8_shadow_comparison.json` contains the season totals and
  fixture-integrity audit.
- `analysis/frontier_ranker_validation.py`,
  `analysis/listwise_ranker_validation.py` and
  `analysis/captain_ranker_validation.py` own the fingerprinted learned caches.
- `app/data/model-audit.json` is the compact website-facing result hierarchy.

The full calibration remains:

```powershell
python analysis/calibrate_model.py
python analysis/compare_lens8_shadow.py
python -m unittest discover -s analysis -p "test_*.py"
pnpm test
pnpm lint
```

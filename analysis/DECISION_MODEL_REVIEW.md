# Decision-model review - 2026-08-13

> **Leakage correction, 2026-08-14:** the historical probabilistic-route and
> transfer-action results below used same-fixture raw xG/xA/xGA as predictors.
> They are invalid and must not be used for model comparison. Version-2 route
> features remove those columns; the action model then fails its seed and
> ensemble gates. See `CAPTAIN_DECISION_REVIEW.md` for the corrected use of
> leak-free route distributions in captaincy only.

> Update: the subsequent information-model phase is documented in
> `INFORMATION_MODEL_REVIEW.md`. Causal minutes, tactical-role and team-defence
> challengers improved their forecast targets but failed recursive promotion.
> The key remaining design issue is deterministic optimiser amplification, not
> a lack of additional player features.

## Outcome

The frozen historical champion remains unchanged. Two causal challengers were
built and replayed through legal weekly squads:

| model | decision result | recursive result | decision |
| --- | ---: | ---: | --- |
| frozen multi-timescale champion, no chips | 65.05% near-price accuracy | 2,174.9 average, 2,031 minimum | retain |
| probabilistic scoring routes | 67.57% near-price accuracy | 2,119.8 for the development-selected blend | reject as selector |
| two-band transfer-action consensus, 5% | 67.90% near-price accuracy | 2,177.6 average, 2,067 minimum | prospective shadow only |
| action consensus + audited TC/BB/FH | not independently selected | 2,202.6 average, 2,080 minimum | interaction diagnostic only |

The 2,202.6 result is **not** a new unbiased expected score. The transfer-action
blend missed its predeclared development promotion gate, and the chip interaction
was then inspected after that historical exposure. It is evidence that the
components are compatible, not permission to promote them.

Against the reconstructed top-500k boundaries, that combined diagnostic records
one hit in eight seasons and an average gap of -94.1 points (target pace 2,296.8).
That is 10.3 points closer than the frozen 2,192.4 chip champion, but still far
from evidence of consistent top-500k performance.

## What changed

### Proper component distributions

The earlier component challenger predicted five bundled point totals with
squared-error regression. It had no coherent event probabilities or variance.
`probabilistic_component_challenger.py` instead fits prior-season Poisson models
for appearances, 60-minute appearances, goals, assists, clean sheets, saves,
goals conceded, bonus and defensive-contribution points. A signed robust model
covers cards, own goals and penalty residuals. Position-specific FPL scoring
rules convert those events to points; exact Poisson floor moments handle saves
per three and goals conceded per two.

The new route model is informative but not a safe standalone selector:

- near-price accuracy: 63.06% -> 67.57%;
- affordable top-player regret: 7.78 -> 6.40 points;
- raw event-route MAE: 1.440 versus champion 1.350;
- 80% interval coverage: 94.72%, showing that the first-order independent-event
  distribution remains too wide after causal scale calibration;
- every tested recursive blend lost to the frozen champion on the predeclared
  development stability objective.

This is an important negative result. Better one-row ordering is insufficient
when a recursive optimiser amplifies a small ranking change into a different
season-long transfer path.

### Legal transfer-action learning

`transfer_action_ranker_validation.py` trains position-specific LambdaMART
models on adaptive holding value inside two overlapping 1.5m price bands. It
fits at GW13 and GW25; current-season labels are admitted only when their full
player-specific horizon has matured. Champion forecasts and all scoring-route
means are inputs, so the learner corrects the established model instead of
discarding it.

The ranker improves the metrics it is trained to solve:

- near-price accuracy: 65.05% -> 67.90%;
- affordable top-player regret: 11.96 -> 11.36 discounted holding points;
- affordable top-three regret: 27.20 -> 25.88;
- defenders: 61.66% -> 64.49%, regret 12.59 -> 11.77;
- midfielders: 60.72% -> 65.04%, regret 13.63 -> 12.76;
- forwards: 67.28% -> 69.83%, regret 11.46 -> 11.05;
- goalkeepers show no useful regret gain, so future versions should consider
  leaving goalkeeper action values entirely on the champion scale.

The conservative 5% two-band-consensus plan scores 2,177.6, +2.7 versus the
no-chip champion, and raises the minimum by 36. Its recent four-season holdout
is +8.7 on average, but its development stability is 2.9 points below champion.
It therefore fails promotion and remains a shadow candidate.

The larger 10% ungated version scores 2,177.4 and a 2,049 minimum; a post-hoc
80th-percentile gate reaches 2,183.8 but has a 1,992 minimum. Both are recorded
for diagnosis only and cannot be selected after evaluation exposure.

## Chip interaction

`action_chip_combination_validation.py` refits corrected Free Hit values on the
action manager's own recursive path, then adds the audited TC/BB/FH policies.
The combined diagnostic averages 2,202.6, versus 2,192.4 for the frozen champion
plus chips. Chips add 25.0 points on the action path, but 2024/25 supplies 124 of
those points and 2019/20 loses 16. This concentration is too high for a claim of
consistent elite performance.

## Governance conclusion

1. Production and the website remain untouched.
2. The frozen 2,174.9 no-chip / 2,192.4 chip champion remains the benchmark.
3. The 5% two-band-consensus action model is the only credible new prospective
   shadow. It is not historically promoted.
4. The raw probabilistic component selector is rejected. Its route outputs may
   remain features for decision and simulation models.
5. The next action-model version must be preregistered before any new results.
   The most defensible changes are: no goalkeeper overlay; uncertainty-aware
   abstention calibrated entirely on development data; and package-level labels
   for downgrade-plus-upgrade moves rather than independent player ranks.

## Package-action follow-up - 2026-08-13

- A complete legal-package learner was implemented with an optional, default-off
  optimiser callback, 17,125 one/two-transfer candidates, causal matured labels,
  two-model agreement, goalkeeper abstention and an explicit Hold fallback.
- It improved broad package MAE from 11.953 to 9.544 but failed at the optimiser
  boundary: on 118 causal non-goalkeeper packages the champion actually selected,
  its classification accuracy was 65.25% versus 72.88% for always positive, and
  probability/value correlation was only 0.0436.
- The safest veto reproduced the champion exactly; every policy that materially
  changed the path lost points. A separate 1.15/3/5/8/12 extra-move hurdle audit
  also selected the 1.15 champion default.
- Package learning is rejected, chips were not retested, and production remains
  untouched. Forked three- and six-event counterfactual trajectories were also
  completed: both classifiers lost to their majority baselines at the champion
  decision boundary, and their best recursive effect was only +0.3 with no
  development gain. The package avenue is closed pending materially richer
  information or a much larger prospective/multi-policy state sample. See
  `PACKAGE_ACTION_REVIEW.md`.

## Reproduction

```powershell
python analysis/probabilistic_component_challenger.py
python analysis/transfer_action_ranker_validation.py
python analysis/action_chip_combination_validation.py
python -m unittest discover analysis -p "test_*.py"
```

Machine-readable outputs are in:

- `analysis/data/probabilistic_component_challenger.json`
- `analysis/data/transfer_action_ranker_validation.json`
- `analysis/data/action_chip_combination_validation.json`

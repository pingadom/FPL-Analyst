# FPL model-engine review

Date: 2026-08-13

Scope: historical forecasting, recursive squad management, transfers, captaincy,
chips and evaluation. The website was deliberately excluded.

## Executive result

### 2026-08-13 multi-timescale update

The frozen historical research champion now averages **2,185.5 points** over
2018/19-2025/26, with a minimum of **2,031**. Without chips it averages
**2,174.9**. The change replaces the assumption that every player has one
six-GW value with causal 1/3/6/10-GW value functions, player-specific expected
tenure, an exit-cost term, and two within-season refits. The already validated
Bench Boost and Triple Captain gates are unchanged.

This is a **historical research result, not an unbiased forecast or rank
guarantee**. The architecture is walk-forward causal, but the 15% blend was
chosen after inspecting evaluation-season results. Against the localized
historical point cutoffs in `historical_rank_benchmarks.json`, it still clears
zero of eight targets and is 111.2 points short on average. It is materially
better than the preceding audited champion (2,164.1 average, 1,999 minimum), but
prospective shadow evidence remains mandatory.

| Season | New audited points | Local target | Margin |
|---|---:|---:|---:|
| 2018/19 | 2,129 | 2,151 | -22 |
| 2019/20 | 2,154 | 2,160 | -6 |
| 2020/21 | 2,232 | 2,270 | -38 |
| 2021/22 | 2,284 | 2,340 | -56 |
| 2022/23 | 2,258 | 2,425 | -167 |
| 2023/24 | 2,089 | 2,373 | -284 |
| 2024/25 | 2,307 | 2,411 | -104 |
| 2025/26 | 2,031 | 2,244 | -213 |

The no-chip lift over the fixed-horizon champion is +25.4 points per season:
`[+41, +42, +23, +68, -18, +27, -12, +32]`. A deterministic season bootstrap
gives a 95% interval of **+6.6 to +43.2** and a 99.6% probability that the mean
lift is positive. This quantifies historical stability; it does not undo model
selection exposure.

The original executive result below is retained as the pre-redesign benchmark.

The audited historical champion averages **2,164.1 points** over 2018/19–2025/26,
with a minimum of **1,999**. The same forecasting and transfer stack without chips
averages **2,149.5**. The validated Bench Boost and Triple Captain policy adds
**14.6 points per evaluation season**, never loses points in an evaluation season,
and raises the best season by 40 points.

This is an improvement, but it is not yet a top-500k model. Against the localized
historical point cutoffs in `historical_rank_benchmarks.json`, it clears zero of
eight targets and is 132.6 points short on average. The largest deficits are
2023/24 (-301) and 2025/26 (-245). Those misses are priorities for diagnosis, not
results to conceal or convert into a speculative rank claim.

| Season | Audited points | Local target | Margin |
|---|---:|---:|---:|
| 2018/19 | 2,097 | 2,151 | -54 |
| 2019/20 | 2,112 | 2,160 | -48 |
| 2020/21 | 2,210 | 2,270 | -60 |
| 2021/22 | 2,216 | 2,340 | -124 |
| 2022/23 | 2,277 | 2,425 | -148 |
| 2023/24 | 2,072 | 2,373 | -301 |
| 2024/25 | 2,330 | 2,411 | -81 |
| 2025/26 | 1,999 | 2,244 | -245 |

## Correctness fixes made

1. **Triple Captain units.** A 0–1 captain-ranking percentile was compared with a
   15–24 expected-points threshold, so Triple Captain could never trigger when the
   ranker was active. The policy now uses captain expected points times fixture
   count; the captain ranker only chooses the player.
2. **Chip isolation.** `ChipPolicy.enabled_chips` now permits clean, paired
   ablations. Disabled wildcard/free-hit branches no longer build unused fresh
   squads every gameweek.
3. **Objective consistency.** Starting-squad and transfer utilities can now use
   distinct XI, bench and captain utilities. This prevents a percentile captain
   score or a six-week total from silently entering an incompatible scale.
4. **Recursive observability.** Simulations now record weekly squads, XI, captain,
   transfer path, spend, bank, bench spend, chip opportunities and named-player
   exposure. This makes selection failures traceable rather than anecdotal.
5. **Transfer-search controls.** Candidate-frontier size and beam width are
   configurable, and an optional diversified frontier includes immediate,
   planning, value and reliable-budget routes.
6. **Bench allocation controls.** Exact initial optimisation, minimum spend,
   maximum bench premium and a transfer-time bench-premium cost were implemented
   and tested. They are controls, not blindly enabled champion rules.

## What was promoted

### Forecast and decision stack

- 75% causal immediate forecast + 25% causal frontier rerank.
- 75% stable 4.5-week plan + 25% causal listwise horizon rerank.
- 50% causal captain rerank.
- Existing recursively changing squad, legal transfer accounting and no-hindsight
  deadline features.

Training-only selection still chose this full stack. Its no-chip evaluation score
is 2,149.5, versus 2,141.1 for rolling prior-seasons-only model selection and
2,184.6 for an invalid per-season oracle. The oracle gap quantifies remaining
regime sensitivity.

### Chips

The named `AUDITED_CHAMPION_CHIP_POLICY` enables only:

- Bench Boost when a confirmed-double structural signal clears 11 points.
- Triple Captain when a confirmed-double captain signal clears 15 points.

Automatic Wildcard and Free Hit are disabled in the historical champion. They
remain valid human/scenario decisions, but the tested generic triggers were not
safe enough to automate.

## What was rejected

| Challenger | Evaluation result | Decision |
|---|---:|---|
| Exact multi-horizon opening optimiser | 2,088.2 | Reject |
| Training-selected consistent 65/35 horizon blend | 2,129.6 | Reject |
| Training-selected diversified transfer frontier | 2,146.5 | Reject |
| Captain-access squad bonus | 2,087.9 | Reject |
| Position-specific calibrated horizon policy | 2,031.1 | Reject |
| Training-selected automatic Wildcard | -11.4 points/season | Disable |
| Training-selected automatic Free Hit | -17.6 points/season | Disable |
| Legacy combined chip policy | 2,139.9 | Reject |

Two post-exposure results are retained only as shadows: diversified transfer
frontier 16/beam 10 scored 2,154.2, and full immediate isotonic calibration scored
2,151.0. They were not selected by the frozen training rule, so promoting them
would be test-set overfitting.

## Why the expensive bench and premium omissions happened

The original opening-squad comparison was confounded: it changed both solver and
bench rule. A clean decomposition showed that a cheap bench can improve GW1 but
destroy later rotation value. In one six-week check, the original bench supplied
81 later XI starts and 292 points; the hard-cheap bench supplied 44 starts and 109
points. Therefore “cheapest possible bench” is not a universal optimum.

The real allocation rule should price bench players by probability-weighted
substitution and rotation value, then penalize only premium above that insurance
value. Minimum budget use is a diagnostic guardrail, not an objective: spending
the last £0.5m–£4m is good only when it increases total squad decision value.

The named-player replay found genuine premium access failures:

- Salah squad exposure was 0% in 2019/20 and 2022/23 even though his average
  position-forecast ranks were 2.9 and 3.3.
- Haaland exposure was 0% in 2025/26 despite a 2.6 average position rank.
- In contrast, the model correctly faded 2025/26 Salah when his forecast rank
  dropped to 16.8, and held 2022/23 Haaland for 73.5% of eligible weeks.

This says not to hard-code Salah or Haaland. It says the transfer engine has a
**restructuring/access problem**: a premium may be recognized as elite while the
short beam cannot see a two- or three-transfer route that releases the money, and
the hold option then preserves a locally good balanced squad. The next transfer
engine should search packages and delayed routes explicitly, with a value-of-
information penalty for unnecessary early commitments.

## Forecast diagnostics

- Immediate MAE improved from 1.350 to 1.185 under causal position-specific
  isotonic calibration, and horizon MAE from 4.107 to 3.348. Yet full horizon
  calibration reduced FPL points. Better error metrics do not guarantee better
  downstream decisions when the optimiser depends on ordering and threshold
  scale.
- The top three forecast-ranked players average a +0.817 point optimism bias,
  versus +0.212 at ranks 16–30. The optimiser repeatedly selects the most
  overestimated tail: classic optimiser's curse.
- Goalkeepers have the largest position bias (+0.423). Rotation players are
  especially weak: predicted 0.747 versus actual 0.446 points per row.
- Strong-team defenders do score more: the top team-strength quintile averages
  1.549 points per row versus 0.874 in the bottom quintile. But the top quintile is
  already overpredicted by +0.426, so an extra generic “big team” boost would
  double count strength. Defender improvement must model clean-sheet probability,
  minutes, set pieces and attacking involvement separately.

## Evaluation design

The following rules are now mandatory for a promotion:

1. Features and schedules must be available at the deadline being simulated.
2. Hyperparameters are selected on prior seasons only; evaluation seasons cannot
   decide the winner.
3. Every rule is compared by paired, recursive season replay—not a static GW1
   squad or a one-week forecast metric.
4. Report average, minimum, season margins, transfer/hit counts, chip use and
   failure seasons.
5. Named stars are sanity checks, never labels or constraints.
6. A historical challenger becomes production only after a frozen prospective
   shadow period.

## Highest-value next research

### Six-week-horizon redesign findings

- The old label assumed the same six-GW tenure for an injury-risk punt, secure
  defender and premium captain. It also contained no cost for needing to reverse
  a short-lived transfer.
- The replacement learns cumulative values at 1, 3, 6 and 10 events. A player's
  expected tenure is 2-10 GWs based only on deadline-known availability, starts,
  60-minute probability, rotation, evidence depth and team stability. Tenure is
  capped at season end, including the non-contiguous 2019/20 restart event IDs.
- Prior-season-only ridge models improved correlation and MAE at every longer
  horizon. Online GW13/GW25 refits improved them again; a label is admitted only
  after its complete horizon ends.
- The new forecast's accuracy on over one million near-price player pairs is
  67.21%, versus 63.70% for the fixed plan. Mean regret for the best affordable
  option falls from 12.59 to 11.15 discounted points. Defender regret improves
  from 13.45 to 11.55.
- A hindsight-only adaptive oracle averages 2,380.1. This is not attainable or
  promotable, but it proves the legal recursive optimiser has substantial ceiling
  when future-value ordering is correct.
- Applying the new forecast too strongly still triggers optimiser's curse. The
  frozen policy keeps the old plan through GW12, then uses an 85/15 blend of the
  old plan and an equal ridge/nonlinear decision-focused ensemble. Larger generic
  weights and sparse consensus gates were worse.
- A finalized-schedule diagnostic marginally raised rank correlation (0.7603 to
  0.7650) but materially reduced recursive points. Reconstructing old fixture
  announcement vintages is therefore not the current priority.
- A dormant staleness option that could have added an uncharged transfer was
  removed. Staleness may lower a hurdle, but paid moves must have explicit hit
  accounting.

The next credible improvement is not another global horizon blend. It is a
pre-registered prospective decision model with calibrated uncertainty on feasible
transfer packages, plus richer but causal role/minutes and team-defence inputs.

1. **Decision-focused training.** Train listwise/decision losses on legal squad and
   transfer alternatives so the model optimizes ordering and opportunity cost,
   rather than pointwise MAE alone.
2. **Package transfer search.** Generate premium-access packages (one downgrade +
   one upgrade), rolling two-week routes and wildcard-like restructures without
   spending a chip. Use stochastic beam search or MILP with terminal squad value.
3. **Predictive distributions.** Estimate minutes, appearance, goals, assists,
   bonus and clean sheets as components, then sample correlated player/team
   outcomes. Optimize expected points with downside and model-uncertainty terms.
4. **Hierarchical team defence.** Use causal rolling team attacking/defensive
   strength, opponent strength, home advantage and projected minutes. Shrink early
   season estimates toward promoted-team/league priors.
5. **Chip option value.** At each deadline compare use-now with simulated future
   announced opportunities. Historical future schedules cannot be read before
   they were announced; announcement snapshots or reconstructed vintages are
   required before Wildcard/Free Hit can be evaluated faithfully.
6. **Prospective registry.** Freeze forecasts, squad, captain, transfers, chips and
   model version before every deadline. Evaluate calibration, points, decision
   regret and counterfactual packages after results arrive.

## Reproducible artefacts

- `multiscale_horizon_validation.py/json`: 1/3/6/10-GW targets, tenure, exit cost,
  causal forecasts and oracle information ceiling.
- `multiscale_phase_validation.py/json`: opening-vs-in-season phase gates.
- `feasible_decision_audit.py/json`: near-price and affordable-choice regret.
- `decision_focused_horizon_validation.py/json`: direct holding-value learner.
- `decision_focused_champion_audit.py/json`: frozen no-chip/chip champion and
  causal/selection-exposure audit.
- `schedule_information_bound.py/json`: non-promotable finalized-schedule bound.
- `sequential_chip_value_validation.py/json`: causal TC/BB optimal stopping and
  empirical half-season reservation values.
- `freehit_value_validation.py/json`: prior-season FH value model and corrected
  recursive decision-order ablation.
- `combined_chip_policy_validation.py/json`: TC/BB/FH collision and interaction
  test on the frozen multi-timescale champion.
- `CHIP_STRATEGY_REVIEW.md`: official rules, elite-manager evidence, model
  defects, results and the prospective sequence-optimiser specification.

- `championship_stack_validation.py/json`: consolidated champion.
- `walk_forward_stack_audit.py/json`: static and rolling model selection.
- `tc_bb_policy_validation.py/json`: corrected chip validation.
- `wildcard_freehit_ablation.py/json`: isolated negative chip results.
- `forecast_selection_bias_audit.py/json`: optimiser's-curse audit.
- `forecast_calibration_challenger.py/json`: causal calibration.
- `premium_asset_audit.py/json`: named asset and season-leader exposure.
- `player_segment_audit.py/json`: position, rotation, price, popularity and team
  strength diagnostics.
- `test_model_engine.py`: model-engine regression tests.

## Chip engine redesign - 2026-08-13

- TC/BB no longer require a confirmed double. Prior-season-only value models
  compare use-now value with chip-specific reservation values that decay toward
  GW19/GW38 expiry and force distinct final-week assignments when necessary.
- The causal sequential TC/BB challenger averages 11.9 chip points, versus 10.6
  for the old audited policy. Its total is 2,186.8 on the frozen 2,174.9 no-chip
  path, but the +1.3 paired improvement is too small and uneven to auto-promote.
- Free Hit contained two process defects: a preflight branch suppressed normal
  transfers even when the later chip decision was Hold, and its one-week XI gain
  ignored the permanent transfer action displaced by FH. Both are fixed.
- A learned FH value gate raises the combined TC/BB manager from 2,185.5 to
  2,192.4. The +6.9 evaluation lift persists at +6.5 over the 2022/23-2025/26
  holdout, but with only two non-zero holdout activations and a -15 paired worst
  season it remains a prospective shadow challenger.
- Automatic Wildcard remains rejected. The best training-selected policy lost
  49.6 points out of sample and even the least-bad threshold lost 16.9. Live WC
  advice therefore requires manager-specific damage, multiweek gain and announced
  fixture structure; it is not part of the historical champion.

## Probabilistic and transfer-action redesign - 2026-08-13

- A true event-route challenger now fits prior-season Poisson models for
  appearances, 60-minute appearances, goals, assists, clean sheets, saves, goals
  conceded, bonus and defensive contributions, plus a signed residual model.
- Route ordering improved near-price accuracy from 63.06% to 67.57% and reduced
  affordable top-player regret from 7.78 to 6.40, but every recursively selected
  route blend lost to the champion. It is rejected as a squad selector and kept
  only as a source of component features and uncertainty.
- A separate transfer-action ranker learns adaptive holding value only among
  same-position players in two overlapping 1.5m price bands. It raises action
  accuracy from 65.05% to 67.90% and reduces holding-value regret from 11.96 to
  11.36. Defender, midfielder and forward regret improve; goalkeeper regret does
  not.
- The conservative 5% two-band-consensus challenger scores 2,177.6 without
  chips, versus 2,174.9, while its minimum improves from 2,031 to 2,067. It still
  misses the predeclared development gate, so it is prospective-shadow only.
- A post-exposure action/chip interaction reaches 2,202.6 versus the frozen
  chip champion's 2,192.4. This is a compatibility diagnostic, not a promotion:
  gains are concentrated in 2024/25 and one season is harmed.
- Full method, ablations and governance are recorded in
  `DECISION_MODEL_REVIEW.md`.

# FPL Lens process and architecture review

> **Information-boundary addendum, 2026-08-14:** raw xG, xA and xGA on a
> historical player-gameweek row are post-match values. They have been removed
> from the scoring-route learner, all dependent caches were versioned, and the
> previously reported late-action gains were invalidated. The action shadow is
> disabled. A leak-free five-seed captain route consensus survives fixed-XI,
> recursive, chip, seed, neighbourhood and holdout checks; details are in
> `CAPTAIN_DECISION_REVIEW.md`.

This is the post-audit record for Lens 7. It distinguishes implemented fixes,
rejected challengers and work that still needs genuinely new evidence.

## Corrected

- Historical club identity now comes from each fixture rather than a player's
  end-of-season club, removing final-club leakage after real transfers.
- Every registered player remains in each observed deadline universe. A known
  blank scores zero without being treated as poor form, low availability or a
  missing future label.
- Historical future blanks and doubles are censored because archive data does
  not contain their announcement timestamps. The current app still uses the
  official schedule that is actually published at the live deadline.
- Six-Gameweek targets are calendar aligned and only enter causal fits after
  their full outcome window has completed.
- Project Restart, the World Cup unlimited window, Free Hit introduction,
  changing transfer caps, chip preservation rules, Assistant Manager and the
  two-chip-set season are represented separately rather than backfilled under
  one modern rule set.
- FPL selling value, bank, position quotas, three-per-club limits, real-world
  club transfers and AFCON exclusions are checked at every squad transition.
- Wildcard impact is evaluated with a paired full-season replay. The sum of
  same-week chip gains is now explicitly labelled `immediateChipGain` and is
  never presented as the full chip effect.
- The top-500k promotion gate uses the compact policy selected only on 2016/17
  and 2017/18. Searches exposed to later seasons remain research diagnostics.
- Historical rank cutoffs use 5,000 public official histories, nearest observed
  boundaries, tie/survivorship allowances and four-GW block bootstrap bands.
- The current FPL team endpoint computes the legal XI/captain from the imported
  15, enforces club limits on suggestions and anchors rank to the exact official
  rank instead of fabricating a future rank from one projection.
- Current player detail is split out of the initial page artifact. The main JSON
  fell from about 4.5 MB to about 0.17 MB; the player API loads the full pool.
- The lineup selector sorts each position once and uses prefix sums. A full joint
  ten-season replay fell from roughly 30 seconds to 14 seconds, with 1,000
  randomized equivalence checks. The intermediate stateful screen fell from 240
  redundant candidates to 80 while retaining 2,400 broad mixes and 20 exact
  finalists.
- Simulation cache keys now use a content fingerprint rather than a reusable
  Python object id. Walk-forward evaluation simulates only the season being read;
  full-versus-sliced replay was checked for exact equality.
- The live recommendation pipeline runs as a preflight before long calibration,
  and local site scripts now work on Windows and Unix.

## Challenger results

- A causal position-specific Random Forest/XGBoost ensemble, inspired by
  OpenFPL, improved player-week MAE from 1.3332 to 1.2277 and mean weekly
  Spearman correlation from 0.6376 to 0.6871.
- Using the raw tree predictions for the whole squad reduced recursive points.
  The loss came from optimizing the full player distribution rather than the
  thin, price-constrained FPL frontier.
- Quantile-mapping the tree ordering back onto the structural forecast scale and
  using a 25% immediate blend raised the fixed-policy research average from
  2,106.5 to 2,120.1 points. It cleared two estimated cutoffs, but this is
  post-exposure research and therefore is not promoted.
- Direct tree forecasts of the six-GW target improved MAE and rank correlation
  but reduced recursive squad points. The existing structural planner stays in
  place.

## Current honest benchmark

- Promotion audit: settings frozen using 2016/17 and 2017/18 only.
- Evaluation average, 2018/19 through 2025/26: 2,096.8 points.
- Estimated top-500k cutoff hits: 0 of 8.
- Average cutoff margin: -200.0 points.
- The app is research-grade decision support, not yet an elite manager model.

## Residual limitations

- There is no untouched historical season left after this research cycle. A
  prospective shadow season is required before the tree blend can be promoted.
- Exact historical schedule-announcement snapshots, injury news, predicted
  lineups, bookmaker markets and crowd expected-minutes forecasts are absent.
  Censoring avoids leakage but understates the planning information a real
  manager would have had.
- AFCON availability is nationality/window based and intentionally conservative;
  verified call-ups and return dates should replace it prospectively.
- Rank estimates are only exposed within 50 points of the reconstructed cutoff.
  Totals outside that local calibration range retain points, margins and target
  probabilities but no invented rank number.
- Chip continuation value is conservative because historical future schedule
  announcements are censored. This prevents hindsight but weakens proactive
  double-Gameweek planning.
- Model selection has substantial researcher degrees of freedom. Future changes
  need a preregistered hypothesis, a frozen shadow baseline and promotion only
  after prospective results.

## Next promotion experiment

Run one full season in shadow mode with three frozen managers: the structural
champion, the 25% rank-calibrated tree challenger and a no-chip control. Record
every deadline input and decision before lock. Promote only if the challenger
improves decision regret, calibration on the selectable frontier and season
points without increased illegal-state or late-news override rates.

## Prospective implementation — 2026-08-12

- The proposed shadow experiment now exists. Official player/fixture/game-rule
  payloads, schedule fingerprints, expected-minutes overrides and SHA-256
  hashes are captured in append-only deadline artifacts. Locked capture is
  rejected after the official deadline.
- The selectable-frontier challenger trains on 74,520 historically plausible
  player-deadlines. Its frozen 25% blend improved its paired recursive control
  from 2,056.2 to 2,103.0 points and improved frontier MAE, Spearman ordering,
  top-15 returns and blank rate. It still produced 0/8 estimated top-500k hits
  and an average -193.8 point cutoff margin, so it remains unpromoted.
- A fixed historical chip calendar reduced that challenger from 2,103.0 to
  2,090.6. Live chip decisions therefore use paired Monte Carlo distributions,
  structural blank/double requirements, downside thresholds and wait value.
  Each recursive shadow squad is evaluated separately after GW1 state exists.
- The live optimiser now solves squad, XI and captain together in one integer
  program. Bench value and captain value are explicit, avoiding the earlier
  mistake of valuing all 15 squad slots equally.
- Three recursive states are pre-registered: structural/no chips,
  structural/scenario chips and frontier/scenario chips. Provisional decisions
  are idempotent and never update state; only a locked decision can be scored.
- The official post-GW scorer applies appearance-based captain fallback, legal
  autosubs, Bench Boost, Triple Captain and transfer costs, then appends a
  non-overwritable result.
- The first 2026/27 GW1 artifact is deliberately **provisional**. It tracks 581
  official players, carries full projections for the 390-player selectable
  pool, recommends holding every chip, and has zero scored prospective weeks.

## Performance push — 2026-08-13

- Decision regret identifies transfer-path persistence as the dominant attainable
  loss. Unlimited fresh weekly squads average 2,242.9 points versus a 2,296.8
  reconstructed pace line; the paired recursive control averages 2,106.5.
- A causal LambdaMART ranker trained on the six-week selectable frontier raises
  the recursive average to 2,141.6 and the minimum from 1,897 to 1,970.
- Combining that transfer horizon with the 25% next-GW frontier rerank reaches
  2,147.6. A 50% captain rerank raises the full stack to 2,149.5 and the minimum
  to 1,999: +43.0 average and +102 downside versus control.
- The stack still records 0/8 top-500k cutoff hits and remains 147.3 points below
  average pace. It replaces only the pre-GW1 shadow challenger, not production.
- The original transfer hurdle survived a predeclared interaction grid. Both
  more aggressive and more passive transfer policies lost points.
- A percentile-relevance version was rejected because it discarded the magnitude
  of six-week returns. An extra big-team defender boost was also rejected: for
  expected 45+ minute defenders, the strongest clean-sheet quintile is already
  close to calibrated and slightly overpredicted.
- The legacy historical chip threshold reduces the combined stack from 2,149.5
  to 2,132.5. It remains a diagnostic only; prospective managers retain the
  structural Monte Carlo gates and option value of waiting.

## Multi-timescale redesign — 2026-08-13

- The fixed six-GW target is no longer treated as a complete transfer objective.
  The research engine now constructs causal 1/3/6/10-event value functions and
  assigns each player a deadline-known 2-10 GW expected tenure. Short-lived
  transfers pay an explicit future-replacement cost; every horizon is capped at
  the number of events left in the season.
- The calendar audit found no missing event transitions in the historical player
  panel. A regression test covers non-contiguous event IDs such as the 2019/20
  restart, so horizon length is based on event order rather than GW subtraction.
- Prior-season ridge forecasts improve every longer-horizon MAE and rank metric;
  causal GW13/GW25 refits improve them again. Same-season outcomes enter only
  after the complete player-specific target window has matured.
- On affordable, near-price decisions, accuracy improves from 63.70% to 67.21%
  and mean best-option regret falls from 12.59 to 11.15 discounted points. The
  largest position gain is for defenders (13.45 to 11.55 regret).
- A phase-gated 10% multi-timescale overlay, selected by the two calibration
  seasons, raises the no-chip historical result from 2,149.5 to 2,156.6 and the
  minimum from 1,999 to 2,025. With the audited TC/BB gates it scores 2,167.9.
  This is the defensible historical promotion candidate.
- A later 15% equal ridge/direct ensemble scores 2,174.9 without chips and
  2,185.5 with chips, improving six of eight seasons. Because that strength was
  chosen after evaluation exposure, it is frozen as a prospective research
  challenger rather than an unbiased expected future score.
- The hindsight-only adaptive oracle scores 2,380.1. It is not a model result;
  it establishes that forecast discrimination, rather than legal optimisation,
  is now the dominant remaining ceiling.
- Finalized historical fixture schedules were tested only as a non-promotable
  information bound. They slightly improved rank correlation but reduced season
  points, so archived schedule-vintage reconstruction is deprioritized.
- An unused staleness option capable of adding a transfer without hit accounting
  was removed. No model branch may create a paid move without explicit cost.

## Chip process audit - 2026-08-13

- The live and historical engines now represent the 2026/27 two-set inventory,
  GW19 expiry, one-chip-per-GW collision, GW1 transfer-chip restrictions and
  FH19/FH20 separation.
- Bench Boost is incremental to legal autosubs; Triple Captain has vice-captain
  fallback; WC/FH use each manager's actual selling budget; FH never mutates the
  permanent squad.
- The former Free Hit preflight was invalid: chip availability alone could stop
  ordinary transfers before the gate ran. The gate now runs first, and FH value
  is charged for the permanent transfer plan it displaces.
- The corrected combined TC/BB/FH challenger averages 2,192.4, +17.5 over no
  chips and +6.9 over TC/BB alone. It is registered for prospective shadow only.
- Wildcard remains manually supervised because all automatic historical policies
  lost points. WC -> BB sequencing will be evaluated from frozen official
  schedule announcements with explicit setup and terminal-squad value.

## Robust late-transfer phase - 2026-08-14

- Hard scenario-win and tail-risk transfer gates were rejected after reducing
  recursive points. Uncertainty remains diagnostic rather than a transfer veto.
- The apparent +11.2 late-action gain could not be reproduced by the old 18
  live fields. The weekly builder now preserves 64 deadline-known research
  fields instead of silently dropping team, role, price-pressure, uncertainty
  and scoring-route evidence at the JSON boundary.
- Five independent fits exposed meaningful tree-seed variance. A single model
  failed the seed gate; four-of-five directional consensus across both
  overlapping price bands recovered 2,186.1 no-chip points with a 2,067 minimum.
- With the unchanged audited chips the exact live-compatible action policy
  scores 2,197.9. Adding the fixed captain challenger reaches 2,200.2, +14.7
  over the frozen chip control, but still misses every reconstructed top-500k
  cutoff.
- The strict historical development/holdout selection rule selected no
  production candidate. The policy is therefore a fourth exploratory shadow
  manager, inactive through GW24 and prohibited from production promotion
  without frozen prospective evidence.
- The terminal live fit uses 74,277 completed frontier rows, is byte-stable on
  repeated execution, and passes all 24 integrity tests.

## Leakage correction and breakthrough information phase - 2026-08-14

- The late-action result above was invalidated after raw same-fixture xG, xA
  and xGA were found in the route feature set. Leak-free five-seed retraining
  had a -3.4 median delta and no stable ensemble. The action shadow was removed.
- Its leak-free replacement is captain-only: a five-seed scoring-route consensus
  adds 5.1 points with audited chips, reaching the honest 2,190.6 reference with
  no negative season delta against its frozen captain control.
- An explicit information-ceiling tournament shows that better current-week
  minutes have a +17.8 selection-only ceiling, while perfect team outcome
  knowledge has +137.4. Their combined +153.5 ceiling is complementary, but all
  values are hindsight diagnostics rather than attainable forecast claims.
- A prior-season-only market model built from non-closing 1X2 and over/under
  prices improves team goals-for MAE from 0.9607 to 0.9055, goals-against MAE
  from 0.9729 to 0.9141 and clean-sheet Brier from 0.23587 to 0.22495.
- Direct market/lineup transfer integration loses 80.5 points in the
  development-selected row, confirming that deterministic recursive path
  amplification remains the central design failure.
- With the transfer path frozen, a conservative market captain boundary adds
  3.6 points, reaching 2,194.2 with audited chips. It is research-only: only four
  seasons improve and exact historical odds timestamps cannot be verified
  against every FPL deadline.
- The next architecture is a decision-focused legal-action model over joint
  market/lineup scenarios with an explicit hold/abstain option. No external
  point estimate may directly rewrite the transfer path.

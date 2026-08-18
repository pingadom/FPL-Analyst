# Breakthrough model phase review

## Outcome

The phase found the remaining wall and produced one small but credible external-
information challenger. It did **not** find a safe market-driven transfer model.

The current honest historical reference remains the leak-free scoring-route
captain with audited Bench Boost and Triple Captain: **2,190.6 points per
season**. A pre-closing market captain boundary reaches **2,194.2**, a gain of
3.6 points, but remains research-only because exact archive capture times cannot
be proved against every historical FPL deadline.

No website or production recommendation was changed.

## Information-ceiling tournament

Each hindsight source replaced only its identifiable point routes. The
selection-only replay kept the original transfer plan; recursive-path allowed
only the known current-GW delta to affect transfers. It never multiplied one
known result across the multiweek horizon.

| Information source | Selection-only lift | Recursive lift | Interpretation |
| --- | ---: | ---: | --- |
| Perfect minutes | +17.8 | +49.5 | Useful, but not sufficient alone and path use is unstable |
| Perfect team result | +137.4 | +177.4 | Large team-environment ceiling; most realised-result knowledge is unattainable |
| Perfect minutes + team | +153.5 | +305.4 | Strong complementarity and the target architecture |
| Perfect scorer involvement | +261.6 | +604.2 | Primarily hindsight/football variance |
| Perfect total points | +389.2 | +996.0 | Pure oracle control |

These are not expected gains. They allocate research attention. In particular,
the team-result oracle does not imply that bookmaker probabilities can recover
137 points.

## Market-plus-lineup challenger

Fourteen seasons of Football-Data.co.uk first-set prices were archived with
SHA-256 hashes. Only non-closing `Avg`, `BbAv` and `B365` columns are allowed;
columns containing the closing `C` marker are rejected by code and tests.

Prior-season-only Poisson and logistic models convert no-vig 1X2 and over/under
prices into team goal and clean-sheet forecasts. Evaluation fixture coverage is
100% across 5,643 single-fixture team-weeks.

| Team target | Structural model | Market model |
| --- | ---: | ---: |
| Goals-for MAE | 0.9607 | 0.9055 |
| Goals-against MAE | 0.9729 | 0.9141 |
| Clean-sheet Brier | 0.23587 | 0.22495 |

The market model clearly improves its own targets. Directly inserting those
better estimates into transfer utility nevertheless failed every tested blend.
The development-selected full-strength market plus 30% downside-lineup row lost
80.5 points per season and seven of eight seasons. This is another clean example
of deterministic path amplification, not evidence that the odds are useless.

## Frozen-path boundary result

The transfer path was then held fixed byte-for-byte while market information was
tested on whole-XI, defender, attacker and captain surfaces independently.

Development-only selection chose a conservative captain policy:

- 30% market correction when constructing the market rank;
- 10% market rank blended into the leak-free route captain rank;
- no market changes to squad selection, transfers, XI score or chips;
- no-chip average 2,183.4 versus 2,179.8, +3.6;
- audited-chip average 2,194.2 versus 2,190.6, +3.6;
- season deltas: 0, +10, +9, 0, +4, -3, +9, 0;
- holdout gain: +4.5 points per season;
- four positive, three unchanged and one negative season.

It fails the five-positive-season gate and the historical odds archive cannot
prove exact FPL-deadline timestamps. It is therefore not production-promoted.

## Architecture decision

The next model must preserve the following boundaries:

1. Market/team probabilities may inform captaincy and correlated match scenarios.
2. The lineup model may provide downside and no-show uncertainty without freely
   boosting player means.
3. Neither source may directly rewrite the transfer path through a small change
   in a point estimate.
4. Transfer decisions must be learned/evaluated as legal actions against holding,
   with abstention when source models disagree.
5. Prospective market prices must be captured inside the immutable FPL deadline
   snapshot. Historical first-set prices can support research but not promotion.

The fundamental next build is therefore a decision-focused action layer over
joint market/lineup scenarios, not another deterministic forecast blend.

## Verification

- 36 integrity tests pass.
- All new Python files compile.
- The market model is deterministic across complete refits; repeated arrays are
  identical with SHA-256 `6437f07c4815b0e67ca36c388735c06ba18e4cd74d400babdea11bd0eab8d1ea`.
- `git diff --check` reports no patch errors.

## Artifacts

- `information_ceiling_tournament.py`
- `data/information_ceiling_tournament.json`
- `forecast_routes.py`
- `market_lineup_challenger.py`
- `data/market_lineup_challenger.json`
- `market_decision_boundary_validation.py`
- `data/market_decision_boundary_validation.json`
- `test_information_ceiling.py`
- `test_market_lineup_challenger.py`

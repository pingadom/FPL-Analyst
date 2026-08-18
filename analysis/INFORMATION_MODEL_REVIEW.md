# Information-model phase review

## Decision

The frozen no-chip champion remains the production model at **2,174.9 points per
season**, with a **2,031-point minimum** across 2018/19-2025/26. No player,
team-strength or minutes challenger passed every promotion gate. The best new
research candidate is a 15% captain-ceiling blend, but it is not automatically
promoted because only four of eight seasons improved and three declined.

This is not a null result. Three independently trained causal challengers
substantially improved their own forecast target, then lost points when their
small differences were fed into the deterministic transfer optimiser. The
repeated result identifies the next design bottleneck much more precisely than
another round of feature tuning would.

## Frozen decision-boundary autopsy

The audit in `data/decision_boundary_error_autopsy.json` replays the actual
champion path and weights players according to whether they were credible
frontier candidates, owned, started, captained or involved in a transfer.

- Starting-XI forecast: 5.129 versus 4.606 actual; bias +0.523; MAE 3.521.
- Captain forecast: 6.765 versus 6.481 actual; MAE 4.503.
- Transfer-boundary forecast: 5.387 versus 4.600 actual; bias +0.788.
- Attacking-return outcomes account for 43.7% of weighted XI absolute error.
- Unexpected minutes account for 16.3%, forecast no-shows for 10.1%, and
  reduced minutes/benching for 4.2%.
- Team clean sheets account for 30.7% of defender error and 40.6% of goalkeeper
  error, confirming that defender selection must be team-led.

## Probabilistic minutes and availability

`probabilistic_minutes_validation.py` trains prior-season-only gradient models
for appearance, start, 60-minute and expected-minute probabilities. Every
short-term lineup signal is shifted before the target Gameweek.

| Target | Baseline Brier | Challenger Brier | Change |
| --- | ---: | ---: | ---: |
| Appearance | 0.08818 | 0.06063 | -31.2% |
| Start | 0.13182 | 0.10971 | -16.8% |
| 60 minutes | 0.15833 | 0.10216 | -35.5% |
| Minutes / 90 | 0.10569 | 0.07815 | -26.1% |

Symmetric transfer integration was unstable. The safest 25% downside-only
lineup application scored 2,168.1, compared with 2,174.9, although both holdout
seasons improved by one point. The model is therefore useful as a source of
uncertainty/no-show evidence, not yet as a direct expected-points replacement.

## Tactical role and inferred set pieces

`tactical_role_challenger.py` uses shifted 2- versus 6-match changes in starts,
minutes, xG, xA, threat, creativity, key passes, big chances, open-play crosses,
penalty-miss evidence and BPS. It never treats an inferred set-piece role as an
official fact.

- Decision-weighted attacking-route MAE improved from 1.6920 to 1.5447.
- Correlation improved from 0.3143 to 0.3648.
- Recursive variants nevertheless scored below the frozen champion and were
  rejected.

## Hierarchical team defence

`team_defence_challenger.py` trains at team-game level on home advantage, rest,
league scoring environment, opponent-adjusted attack/defence/form, table state,
regime shifts and the existing Poisson expectation. It scores a season only
from earlier-season training data.

- Defender/GK decision-weighted clean-sheet Brier improved from 0.24995 to
  0.24082.
- Bias improved from -0.06673 to -0.00350.
- Recursive symmetric and downside-only integrations all lost points, so none
  was promoted.

## Premiums and captaincy

Generic premium transfer bonuses were strongly harmful. Price is evidence of
role and ceiling, not free points; paying for a premium is only justified by the
captaincy and squad opportunity it creates.

Captain-only blends were more promising because they do not redirect the
transfer path. A 15% blend of the frozen captain rank with a haul, return,
team-attack and probabilistic-minutes ceiling rank produced:

- No chips: 2,178.6 average, +3.7; development +2.3; holdout +8.0; minimum
  season delta -13.
- With the current audited chip policy: 2,189.1 versus 2,185.5, +3.6;
  development +2.2; holdout +8.0; minimum improved from 2,031 to 2,035.
- Chip-paired season deltas: 0, -13, +12, +18, -1, -3, +12, +4.

This is the phase's strongest result, but only four seasons improved, one was
unchanged and three declined. It remains a shadow challenger rather than a
silent production change.

## Root design error exposed by the phase

The current optimiser consumes point estimates as if a 0.05-point ordering were
known with certainty. A deterministic argmax turns a small calibration change
into a different 15-player squad, different transfers and a different season
trajectory. That is why three better information models can all make the
policy worse.

The next credible improvement is therefore not another unconstrained forecast
blend. It is a robust decision layer:

1. Preserve forecast distributions and ensemble disagreement through squad
   selection instead of collapsing them to one point value.
2. Evaluate candidate squads across correlated minutes, clean-sheet and
   attacking-return scenarios.
3. Require a transfer to win across a majority of plausible scenarios, with a
   downside/CVaR constraint, rather than merely winning the mean by a fraction.
4. Use separate utilities for current XI, captain, transfer horizon and bench
   insurance. Do not extrapolate a one-week clean-sheet correction over an
   entire holding period.
5. Let the new minutes, team and role models control confidence and abstention.
   Retain the frozen forecast order when the models disagree.
6. Validate the scenario policy on development seasons, then report the final
   two seasons once. Promote only if average, holdout, minimum and cross-season
   consistency all improve.

## Artifacts

- `decision_boundary_error_autopsy.py`
- `probabilistic_minutes_validation.py`
- `tactical_role_challenger.py`
- `team_defence_challenger.py`
- `premium_captain_validation.py`
- `captain_consensus_refinement.py`
- `captain_chip_interaction_validation.py`
- Corresponding JSON results in `analysis/data/`

No website code or live production model was changed during this phase.

# Captain-decision review - 2026-08-14

## Outcome

The selected captain challenger improves the full recursive model by 4.9
points per season without chips and 5.1 with the unchanged audited chip policy.
Across the eight evaluation seasons, no recursive season becomes worse.

| replay | average delta | worst delta | positive / negative seasons |
| --- | ---: | ---: | ---: |
| fixed squad and XI | +4.5 | -4 | 6 / 1 |
| full recursive, no chips | +4.9 | 0 | 6 / 0 |
| full recursive, audited chips | +5.1 | 0 | 6 / 0 |

The honest current reference is therefore 2,179.8 without chips and 2,190.6
with audited chips. Earlier 2,197.9 and 2,200.2 action-based figures are
invalidated by the information-boundary correction described below.

## Selected decision rule

The model keeps 85% of the frozen captain rank and adds 15% of the mean rank
from five independently seeded causal scoring-route models. The route score is
the modelled expected points plus 0.30 times its calibrated standard deviation.
A 0.005 rank penalty for defenders, and one tenth of that for goalkeepers,
breaks only near-ties; it does not ban defensive captains or Double-Gameweek
captains.

The challenger abstains until three complete prior seasons are available. This
data-sufficiency rule was added after the seed audit showed that a model trained
on only two seasons was unstable. It is based on training volume, not a named
season or player.

## Information boundary

The first route experiment included `expected_goals`, `expected_assists` and
`expected_goals_conceded` from the same historical fixture. Those fields are
post-match observations. They were removed, the route cache was moved to
version 2, and every dependent action cache was invalidated and retrained.

The corrected action result is negative: the five-seed median is -3.4 points
per season, zero seeds are positive on average, and no ensemble passes. The
late-action prospective manager is therefore disabled before its first active
Gameweek.

An automated test now fails if any of the three forbidden columns re-enters the
route feature set.

## Opponent history

Historical performance against the same opponent is calculated causally and
shrunk strongly toward the player's general record. It remains weak evidence:
its Spearman relationship with next-match points is roughly 0.06-0.08 versus
roughly 0.16-0.18 for the structural projection. Direct and learned H2H blends
failed their stability gates, so opponent history is descriptive and is not in
the selected captain rule.

This is an important fine margin: fixture history can explain a recommendation,
but it should not overrule expected scoring routes, team strength, minutes and
the current fixture model.

## Stability evidence

- Development delta: +5.0 points per season.
- Holdout delta: +3.0 points per season.
- Five-seed 20th-percentile average delta: +2.6.
- Local coefficient-neighbourhood 20th-percentile average delta: +2.9.
- Fixed-XI worst season: -4.
- Recursive worst season: 0.

The exact evidence and changed-decision log are stored in
`data/captain_route_consensus_validation.json`. Historical success grants only
prospective-shadow status; production promotion still requires frozen live
decisions scored after their deadlines.

## Live correction

The deadline pipeline formerly multiplied captain rank by an absolute expected-
minutes factor even though expected minutes were already model inputs. That
double-counted rotation risk and briefly moved Gabriel above Haaland in the GW1
shadow. The adjustment now responds only when new deadline intelligence is
worse than the model snapshot. With unchanged information, Haaland is captain
and Gabriel vice-captain in all current shadow managers.

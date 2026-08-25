# FPL Lens — independent model logic audit

Scope: `analysis/calibrate_model.py` (9,178 lines), `app/lib/squad-optimizer.mjs`,
and the generated `app/data/model-results.json`. Every quantitative claim below was
verified against the cached feature frame (252,337 player-weeks), not inferred from
reading alone. "Before" figures come from
`work/fpl-data/prepared-history-lens8-availability-v3.pkl`; "after" figures from the
repaired `work/fpl-data/prepared-history-lens9-minutes-calibrated-v2.pkl`.

Headline: the replay averages **2,087 points/season** and gains only **~50 points/season
from all chips combined**. Both numbers are explained by a small number of concrete
defects, not by a missing feature. The five in Tier 1 are worth, conservatively,
**150–300 points a season**.

## Status

**Tier 1 (1-5) and Round 2 (26-30) are fixed, and the full calibration has been
re-run.** The feature cache is now `prepared-history-lens9-teams-blankfree-v3.pkl`;
earlier frames are not compatible.

### Walk-forward result

| season | before | after | delta | no-chip before | no-chip after |
|---|---|---|---|---|---|
| 2018/19 | 2103 | 2010 | **-93** | 2065 | 2040 |
| 2019/20 | 2044 | 2086 | +42 | 2045 | 2091 |
| 2020/21 | 2076 | 2185 | +109 | 2049 | 2183 |
| 2021/22 | 2152 | 2163 | +11 | 2105 | 2141 |
| 2022/23 | 2158 | 2168 | +10 | 2026 | 2136 |
| 2023/24 | 2261 | 2293 | +32 | 2235 | 2242 |
| 2024/25 | 1968 | 2336 | **+368** | 1901 | 2197 |
| 2025/26 | 1934 | 1937 | +3 | 1868 | 1868 |
| **mean** | **2087.0** | **2147.2** | **+60.2** | 2036.8 | **2112.2** |

Seven of eight seasons improved. The estimated top-500k shortfall roughly halved
(average margin -210 to -150, worst -443 to -307), though the target is still not
reached in any season.

Three things to read carefully before attributing this number:

- **It bundles everything.** The training-only gate flipped the decision policy from
  "Six-GW planner + adaptive banking" to "Joint transfer-chip tree + hold value", so the
  +60.2 mixes the forecast repairs with a strategy switch the repairs happened to trigger.
  Individual fixes cannot be attributed from this run.
- **The gain is in the forecast, not the chips.** The no-chip baseline rose 75.4 a season
  — *more* than the headline. Chips now contribute 35.0 a season against 50.2 before,
  despite being played 40 times against 17.
- **2024/25's +368 is genuine but was a low base.** Its no-chip baseline alone rose 296.
  The old model scored that season worst of all eight (1968) despite it being unremarkable.
  A leak check on the un-censored horizon feature (finding 28) is clean: it correlates
  0.011 with hidden future fixture counts and -0.058 with hidden future blanks, against
  0.866 and -0.386 for the deliberately leaky diagnostic.

### The chip policy now over-fires — a regression introduced by finding 3's repair

| season | with chips | no chips | season delta | sum of local chip gains |
|---|---|---|---|---|
| 2018/19 | 2010 | 2040 | **-30** | +76 |
| 2019/20 | 2086 | 2091 | **-5** | -6 |
| 2020/21 | 2185 | 2183 | +2 | +53 |
| 2023/24 | 2293 | 2242 | +51 | +112 |
| 2024/25 | 2336 | 2197 | +139 | +147 |

In 2018/19 the chips locally "gained" 76 points and cost 30 over the season. The
use-before-expiry ramp fires 14 Wildcards across eight seasons — 1.75 per season against a
maximum of 2 — so it is effectively always dumping both, and a Wildcard's cost is not the
week it is played but the squad trajectory afterwards, which the local gain cannot see.
`CHIP_EXPIRY_THRESHOLD_SHARE = 0.25` is too permissive for the Wildcard specifically. Free
Hit, by contrast, is now the strongest chip in the book at 7 uses and +27.0 average.

The obvious test is a per-chip expiry floor: keep the aggressive ramp for the one-week
scoring chips (Free Hit, Bench Boost, Triple Captain), whose cost really is local, and hold
the Wildcard to something much closer to its searched `wildcard_gap`.

Forecast quality across the whole repair, on the same 252,337 player-weeks:

| | original | after Tier 1 | after Round 2 | truth |
|---|---|---|---|---|
| correlation with realised points | 0.5156 | 0.5219 | **0.5243** | - |
| mean absolute error | 1.3318 | 1.3318 | **1.2289** | - |
| mean projection | 1.465 | 1.465 | **1.296** | 1.293 |
| Double Gameweek ratio | 1.21x | 1.91x | 1.91x | 1.93x |
| `goal_rate` vs true per-90, lowest start-probability quartile | 0.45x | 1.00x | 1.00x | 1.00x |
| team attack rating within 3 GWs of a blank | 1.135 | 1.443 | 1.443 | 1.438 |

Round 2 specifically:

| | before | after |
|---|---|---|
| E[minutes] error, < GBP 4.5m | +5.80 | **+0.58** |
| E[minutes] error, GBP 9.0m+ | -8.10 | **-3.89** |
| P(60+) error, GBP 9.0m+ | -0.139 | **-0.064** |
| P(play) error, < GBP 4.5m | +0.100 | **+0.006** |
| blend bias (per fixture) | +0.1655 | **+0.0024** |
| blend MAE (per fixture) | 1.2916 | **1.1920** |
| `corr(fixture_censored, fixture_now)` | 0.990 | **0.538** |
| horizon residual fixture signal | -0.0031 | **+0.0353** |

**Still open.** Two things the repair did not fully reach:

- The **GBP 9.0m+ tier is still under-projected by 0.48 a week** (was 0.51). The minutes
  repair fixed the cheap end almost exactly but only halved the premium end; a third of
  what is left is the residual -3.89 minute gap, and the rest is the scoring rates. Goal
  rates are now well calibrated for premiums (ratio 0.99) and assists nearly so (0.94),
  but **bonus is under-predicted at every tier** (0.73-0.89 of the true per-90), because a
  12-week rolling mean of a right-skewed variable lags. That is the obvious next target.
- Tier 2 and Tier 3 below are otherwise untouched.

---

## Tier 1 — defects that directly cost points *(fixed)*

### 1. Double Gameweeks are under-forecast by ~40% *(fixed)*

`prepare_causal_history` multiplies **only** `component_xpts_structural` by
`fixture_count` (line 2172). `empirical_xpts`, `market_role_xpts` and
`role_ridge_xpts` are per-gameweek quantities and are blended in un-scaled, then the
whole blend is re-clipped to `(0.2, 13.5)`.

Measured on the cached frame:

| | single GW | double GW | ratio |
|---|---|---|---|
| realised `points` | 1.250 | 2.408 | **1.93** |
| `component_xpts_structural` | 1.180 | 2.232 | 1.89 |
| `empirical_xpts` | 1.312 | 1.287 | 0.98 |
| `market_role_xpts` | 2.654 | 2.651 | 1.00 |
| `role_ridge_xpts` | 1.268 | 1.236 | 0.98 |
| **`component_xpts` (blend)** | **1.405** | **1.699** | **1.21** |

Restricted to fieldable premiums (`price >= 6.0`, `play_probability > 0.8`):
DGW forecast **4.47** vs realised **7.26** — a **2.79 point per player per DGW**
under-forecast.

Consequences, all observable in `model-results.json`:

- Triple Captain is gated on `fixture_count > 1` *and* on
  `triple_captain_signal(scores[captain], fixture_count) >= triple_score (≈21.8)`.
  Because the captain's DGW score is ~40% too low, that gate is almost never cleared:
  **3 TC uses in 8 seasons, averaging 8 points.**
- Bench Boost's `bench_metric` is a sum of the same deflated scores: **3 uses, 16.3 avg.**
- The transfer beam never prices in a DGW, so the squad is never built toward one.

**Repaired.** The blend now happens entirely on a per-fixture scale
(`structural_per_fixture`, `empirical_xpts`, `market_role_xpts`, `role_ridge_xpts`), and
`fixture_count` is applied once to the finished blend. The ensemble weights are fitted
against per-fixture realised points so a Double Gameweek no longer looks like a structural
failure; the role ridge and the empirical form terms were moved onto the same scale;
uncertainty and the P10/P90 band scale by sqrt(fixtures). Measured ratio 1.21x -> 1.91x
against a truth of 1.93x.

Two thresholds moved with it: `triple_captain_signal` no longer multiplies by
`fixture_count` a second time (the projection already carries it, and the argument is now
only used to veto a blank), and `bench_metric` dropped its hand-added `0.15 x extra
fixtures` nudge. `chip_policy_pool` bench/triple ranges were rescaled to the corrected
metric.

### 2. The live path ignores Double Gameweeks entirely *(fixed)*

Two separate bugs in `current_recommendation`:

- `fixture_map` is a `dict` keyed by team id (line ~6545). Both loop bodies write
  `fixture_map[team_h]` / `fixture_map[team_a]`, so **in a DGW the second fixture
  silently overwrites the first**. Opponent, home/away, `team_expected_goals_for/against`,
  clean-sheet probability and the Opta/Matchbook market join therefore all describe one
  arbitrary match.
- `raw_projection` (line 7519) is never multiplied by `fixture_count`. `fixture_count`
  *is* computed correctly at line 7673, but only reaches `horizon_projection`
  indirectly via `weighted_games`. So on the live site, a player with two fixtures shows
  the **same GW xPts** as a player with one.

This makes the shipped product unusable in exactly the weeks that decide a season.

**Repaired.** `fixtures_by_team` keeps every fixture; `fixture_map` retains a single entry
per club for labelling only, and is now deterministically the earliest kickoff rather than
whichever row happened to be last. Per-club match rates are averaged across the club's
fixtures (every scoring route is linear in the per-match rate, so mean x count is the
correct total), and the market/Opta blend is applied per fixture rather than to a
club-level average. `fixture_count` is computed before the projection is built, and
`raw_projection` is multiplied by it once — so a double is worth two matches and a blank
is zero. `horizon_projection` keeps using the per-fixture value because `weighted_games`
already counts both legs.

### 3. Chip timing has no forward-looking value, and more than half of all chips expire unused *(fixed)*

In `simulate_candidate`, `continuation_value = 0.0` is hard-coded (line ~5591), which makes
`option_cost = 0.30 * max(0, continuation_value - current_schedule_signal)` identically zero.
`schedule_opportunity()` is computed and then discarded. The effective threshold collapses to
`max(0.60*base, base - expiry_relief)`, so a chip fires **the first week its metric crosses a
static bar**, with no comparison against better weeks ahead. Every `chip_log` entry in the
shipped artifact reads `"continuationValue": 0.0`.

There is also no forced-use-before-expiry rule: `expiry_relief` only shaves 22% off the
threshold in the final week, floored at 60% of base.

Result across the 8 evaluation seasons: **17 chips played out of ~41 available.**
2019/20 played one chip all season; 2023/24 played no Wildcard at all; 2025/26 played one.
Total chip contribution ≈ 50 points/season against 150–250 for competent human play.

`team_option_score` is likewise initialised to `{team_id: 0.0}` and never populated
(line ~4589), so every `joint_chip_preflight` option-value term in `joint_transfer_plan`
and the greedy transfer loop is multiplied by zero. That one is still open (Tier 2).

**Repaired.** The dead `schedule_opportunity` helper — which read the final schedule and
would have leaked — is gone. In its place is a real optimal-stopping bar: playing a chip
forfeits every remaining week of its window, so the threshold starts at
`1 + CHIP_HOLD_VALUE` times the policy threshold and ramps smoothly down to
`CHIP_EXPIRY_THRESHOLD_SHARE` in the last legal week. The structural-signal requirement
(needs a double, needs a blank) is waived in the final `CHIP_FORCED_USE_WINDOW_GWS` weeks,
because only one chip may be played per Gameweek and several can share an expiry week. The
ramp depends on the remaining window length alone, never on the future schedule.

Replaying one fixed candidate and one fixed policy across all ten seasons:

| | chips played | mean season | chip gain |
|---|---|---|---|
| before | 17 of ~41 | — | ~50 |
| discontinuous expiry cliff (rejected) | 38 | 2,127.7 | +77.6 |
| **smooth ramp (shipped)** | **43** | **2,145.0** | **+94.9** |

No season is chip-negative any more, against two under the old policy. This is a single
frozen candidate/policy pair, not the walk-forward selection, so it is a mechanism
comparison rather than a new headline.

### 4. Scoring rates are contaminated by non-playing weeks — availability is counted twice *(fixed)*

`underlying_game`, `performance_points` and `performance_minutes` are correctly masked with
`.where(fixture_count > 0)`. The **scoring rates are not**:

```python
minute_denominator = data["minutes"].clip(lower=45)      # 0 minutes -> denominator 45
data["goal_signal_game"] = (...) / minute_denominator * 90
```

A week with 0 minutes contributes an observation of `0/45*90 = 0` to the 12-week rolling
mean for `goal_rate`, `assist_rate`, `clean_sheet_rate`, `defensive_rate`, `bps_rate`,
`save_rate`, `bonus_rate`, `conceded_rate`, cards and penalties. Blank gameweeks
(7,912 rows) do the same.

These are then multiplied by `minutes_factor = expected_minutes / 90` in the component
forecast — so availability is applied twice, roughly squared.

Measured deflation of `goal_rate` versus a play-conditional recomputation, by
start-probability quartile:

| start-prob quartile | rate vs. true per-90 |
|---|---|
| lowest | **0.45** |
| second | 0.59 |
| third | 0.78 |
| highest | 0.92 |

Even nailed starters are 8–11% low; rotation risks and players returning from injury are
under-rated by more than half before the minutes model is even applied. This is why
returning differentials are invisible to the model.

**Repaired.** Every rate source is now censored to weeks the player actually appeared
(`.where(appearances_observed > 0)`) and divided by appearances rather than fixtures, so a
Double Gameweek no longer inflates a per-match rate either. The eight official feeds that
arrive as Gameweek totals (`saves`, `bonus`, cards, `goals_conceded`, penalties, own goals)
gained the same per-appearance normalisation they were missing. `defensive_exact_games`
counts appearances; a separate `defensive_feed_games` keeps `defensive_event_coverage` on
its original fixture denominator, since feed availability is a scheduling question rather
than a selection one. All four start-probability quartiles now sit at 1.00x true per-90.

### 5. Team attack/defence ratings are corrupted by blank gameweeks *(fixed)*

`add_causal_team_strength` builds its team panel from `data`, which (correctly) contains
one row per registered player per event including blanks. For those team-weeks
`team_games = 0`, so:

```python
games  = team["team_games"].clip(lower=1)       # 0 -> 1
gf     = team["team_goals"] / games             # 0/1 = 0
attack_observation = ... = 0.0
```

A blank gameweek is fed into the `ewm(alpha=0.22)` attack **and** defence series as a match
in which the club scored zero and conceded zero. 233 team-weeks (3.1%) are affected —
61 of them in 2021/22 alone.

Measured against a recomputation that skips blank weeks, over the 773 team-weeks within
three GWs of a blank:

- attack rating **1.161** (corrupted) vs **1.438** (correct) — a 19% error, max 0.96 goals/game
- and the same magnitude of error, in the flattering direction, on defence rating

So immediately after a postponement the model believes a club has forgotten how to score
and become defensively elite. Those are precisely the fixture-swing weeks around
blanks/doubles where chip and transfer timing is decided.

**Repaired.** Exactly that: the four observations are censored to events the club
played, and every EWM (both the slow alpha=0.22 ratings and the fast alpha=0.48 regime
detectors) uses `ignore_na=True` so a blank carries the rating forward instead of consuming
decay weight. Attack rating within three GWs of a blank moved from 1.135 to 1.443 against a
1.438 reference.

---

## Round 2 — found after the Tier 1 repair

Measured on the repaired frame (`prepared-history-lens9-fixture-scaled-v1.pkl`),
ordered by expected value.

### 26. The minutes model is systematically biased against expensive players *(fixed)*

The largest single remaining defect. Calibration of the minutes probabilities against
realised outcomes, all fixtures, 2016/17-2025/26:

| predicted 60+ | rows | predicted | actual | error |
|---|---|---|---|---|
| 0.4-0.5 | 18,092 | 0.448 | 0.495 | -0.047 |
| 0.6-0.7 | 15,663 | 0.649 | 0.720 | -0.071 |
| 0.8-0.9 | 12,686 | 0.836 | 0.914 | **-0.079** |

and `play_probability` fails in the other direction at the bottom: the 0.2-0.3 bucket
holds **47,996 rows predicted at 0.236 against a realised 0.061**.

By price tier the bias is monotone:

| price | E[minutes] | actual min/fixture | error | P(60+) pred | actual | error |
|---|---|---|---|---|---|---|
| < GBP 4.5m | 23.0 | 17.2 | **+5.8** | 0.223 | 0.182 | +0.041 |
| GBP 5.5-6.5m | 41.9 | 43.9 | -2.0 | 0.409 | 0.463 | -0.054 |
| GBP 7.5-9.0m | 46.8 | 51.7 | -4.8 | 0.464 | 0.562 | -0.098 |
| **GBP 9.0m+** | **52.8** | **60.9** | **-8.1** | **0.530** | **0.672** | **-0.143** |

A premium is credited with 13% fewer minutes than he plays and a 60-minute probability 14
points too low. `component_xpts` scales nearly every route by `expected_minutes / 90`, and
the appearance and clean-sheet routes are driven directly by `p_play` and `p_sixty`, so
this is the root cause of the residual price-tier bias in finding 6 - premiums under-rated
by 0.51 points a week, fodder over-rated by 0.34.

The cause is compression. Flat positional beta priors (`prior_start` 0.54-0.68 with
`prior_strength = 4`, `minutes_if_start` 73-88 with strength 3) pull everyone toward a
common middle, and the three multiplicative penalties (`rest_penalty`, `rotation_penalty`,
`competition_penalty`) can only ever push a start probability *down*, so a nailed starter
is capped below his true rate while a fringe player floats up. `expected_minutes.clip(3,
90)` then hands every unused substitute three minutes of scoring rates.

**Repaired** in `causal_calibrate_minutes`, with `calibrate_live_minutes` as its
terminal twin. An expanding-window isotonic map is learned from predicted to realised for
all five quantities - the three probabilities plus `minutes_if_start` and
`minutes_if_bench` - scoring each deadline before its own outcomes join the fit.

The map is keyed on **position and a deadline-known price band**, not position alone. That
second axis turned out to be essential: a first attempt keyed only on position fixed the
aggregate but made premiums *worse*, because the isotonic curve is dominated by the fringe
players who make up most of the pool. Within a single predicted bin, cheap players started
27% of the time and premiums 44% - a 17-point residual that predicted probability alone
cannot express. Price is the market's view of who is first choice and is known before the
deadline, so it separates that cleanly.

Result: minutes error at the cheap end +5.80 to +0.58 and at the premium end -8.10 to
-3.89; P(60+) error for premiums -0.139 to -0.064; P(play) error for fodder +0.100 to
+0.006. The raw predictors are retained as `*_uncalibrated` so the live deadline fits its
map on the uncorrected series rather than on an already-corrected one.

### 27. The live captain is chosen on a rank scale, costing about 59 points a season *(fixed)*

`pick_squad` uses `captain_utility = frame["captain_score"] * 5`, and `captain_score` is

```python
0.56 * risk_adjusted_projection.rank(pct=True)
+ 0.18 * fixture_now + 0.20 * minutes_security + 0.06 * crowd
```

Every term is a percentile. Ranking destroys magnitude, which is the only thing that
matters for an armband: the gap between an 8.5-xPts captain and a 6.0-xPts captain
collapses to a rank difference. Ownership gets a 0.06 vote in a decision where ownership
has no bearing on expected points.

Replaying 2018/19 onward, choosing from each week's top-15 by `component_xpts` as a proxy
for a strong squad:

- the rank blend picks a **different player from the expected-points argmax in 87% of weeks**
- mean captain return **6.96** by expected points against **5.42** by the rank blend
- **1.54 points per week, about 59 points per season** - and since the captain is doubled,
  that is the whole armband delta

The historical replay uses `captain_metric = scores`, real expected points, so the backtest
never sees this. It is live-only.

**Repaired.** `captain_score` is now `risk_adjusted_projection` - expected points - and
`pick_squad` carries it at weight 1.0 rather than rescaling a rank by five, because the
armband is worth exactly one extra copy of the player's score. The old rank blend survives
as `captain_safety`, which is what the site's 0-100 `captainRating` now displays; it is a
readability summary and decides nothing. Verified against the live endpoint: the chosen
captain is the highest-xPts member of the XI.

### 28. The production six-Gameweek fixture outlook contains no fixture information *(fixed)*

`fixture_censored` is the horizon feature the backtest actually uses.

- `corr(fixture_censored, fixture_now) = 0.988` - it is the current fixture wearing a
  different label
- the censored horizon multiplier correlates **-0.82** with the current fixture: pure mean
  reversion, not an outlook
- it spans 0.942 to 1.052 at the 5th/95th percentiles, so the entire six-week fixture
  apparatus moves a valuation by under 6%

Residual correlation with the realised six-Gameweek target, after removing immediate xPts,
minutes security and long form:

| feature | residual corr |
|---|---|
| `fixture_censored` (production) | -0.0031 |
| `fixture` (real future opponents) | **+0.0331** |
| `horizon_weighted_games_censored` | +0.2210 |
| `horizon_weighted_games` | +0.2482 |

The real future-opponent feature carries **10.7x the residual signal** of the censored one.
Note too that the *number* of future fixtures is by far the strongest horizon feature
(+0.22), and that one is already correctly in the production path.

The censoring is over-applied. The stated reason - the archive carries no announcement
timestamps for reschedulings - is true, but that only makes blank and double **counts**
uncertain. The 380-fixture base schedule is published before the season starts, so future
**opponent identity** is legitimately known at every deadline. Keep the censored counts and
un-censor the opponent difficulty.

This also puts the live site and the backtest on different footings: `horizon_map` in
`current_recommendation` reads the real fixtures endpoint for GW..GW+5, so the live horizon
has the signal the backtest was denied. The six-Gameweek transfer horizon the whole
strategy rests on has never been tested with fixture information in it.

**Repaired.** `censored_fixture_horizon` now reads the actual opponent for each future
event while still counting each future event as exactly one fixture, so blank and double
*counts* stay censored and opponent identity does not. Where a club has no archived fixture
the original opponent is unrecoverable, so the neutral median still stands in rather than a
known blank. The multiplier's clip was widened from (0.82, 1.22) to (0.78, 1.28) to match
the uncensored diagnostic, now that the censored version carries real information.

`corr(fixture_censored, fixture_now)` fell from 0.990 to 0.538 and the residual six-week
signal rose from -0.0031 to +0.0353, matching the uncensored feature's +0.0331. The
transfer horizon is testable for the first time.

### 29. The live forecast is a different model from the backtested one *(fixed)*

Beyond the optimiser mismatch in finding 13, the live *forecast* diverges from the
historical one in four ways, none of which the 2,087-point replay can validate:

- **Set pieces and penalties are live-only.** `set_piece_goal_rate` adds
  `0.075 * team_expected_goals_for * 0.86` for a first-choice penalty taker, plus free-kick
  and corner terms, with no historical counterpart. Worse, it is added on top of
  `goal_rate_live`, which is built from FPL's `expected_goals` field - and Opta xG already
  prices a penalty at about 0.79. Penalties are therefore **double-counted live**, worth
  roughly +0.4 to +0.5 points a week for a first-choice taker who is also a high-xG player:
  exactly the premium forwards and midfielders.
- **`team_attack_multiplier`**, `(team xGF / league rate) ** 0.45`, multiplies the live goal
  route. The historical component forecast has no such term.
- **`position_match_multiplier`** is applied to the live empirical and market projections
  only.
- **Rate construction differs.** Live rates are season-to-date totals with a fixed
  five-game prior; historical rates are 12-week rolling per-appearance means.

**Repaired**, one of each:

- **Penalties are no longer double-counted.** The set-piece uplift is multiplied by
  `5 / (nineties + 5)` - precisely the share of the rate that comes from the prior rather
  than from the player's own record. A new signing or a newly appointed taker gets the full
  uplift; a taker with 25 appearances behind him gets 17% of it, because the other 83% is
  already inside his expected goals.
- **`position_match_multiplier` is deleted.** It was a second absolute fixture price on the
  empirical and market challengers, with no historical counterpart, in direct contradiction
  of the handbook's own "price the current fixture exactly once".
- **`team_attack_multiplier` is back-ported** into the historical attacking route, so both
  paths price team attacking strength identically. It is a relative factor centred on 1.0,
  so it adjusts a long-run personal rate to the match in hand rather than re-adding team
  quality.
- **Level correction now runs on both sides**, the causal expanding version historically
  and a terminal version live.

The rate construction still differs - live rates are season-to-date with a five-appearance
prior, historical rates are 12-week rolling per-appearance means. That one is left as is:
the live model has no rolling window to work with mid-season.

### 30. The ensemble is now worse than its best single member *(fixed)*

Re-measured per fixture (realised mean 1.249):

| model | corr | MAE | mean | bias |
|---|---|---|---|---|
| `structural_per_fixture` | 0.4788 | 1.2431 | 1.271 | +0.023 |
| `empirical_xpts` | 0.4862 | 1.1991 | 1.272 | +0.023 |
| **`market_role_xpts`** | 0.4482 | **2.2896** | 2.652 | **+1.404** |
| **`role_ridge_xpts`** | **0.5144** | **1.1675** | 1.242 | **-0.006** |
| blend `component_per_fixture` | 0.5096 | 1.2916 | 1.414 | +0.165 |

The role ridge - eleven features and an online least-squares fit - beats the blend on
correlation, on MAE and on bias, and is essentially unbiased on its own.
`market_role_xpts` is biased **+1.404**, more than double the truth, and still holds 11% of
the weight; 0.11 x 1.404 = +0.154, which is 93% of the blend's entire +0.165 bias. Deleting
it, or mean-correcting it, improves all three metrics at once.

Two structural notes. Inverse-squared-**MAE** weighting under-penalises a biased model, so
a bias-corrected precision weighting would be better. And the elaborate structural
component forecast is now the *weakest* of the three usable members (corr 0.479), so it is
not earning its complexity.

**Repaired - but not by deleting the bad member.** Dropping `market_role_xpts` outright was
tested and *lost* ranking power: within-Gameweek, within-position Spearman among playable
players fell from 0.3530 to 0.3493. Its poor standalone metrics hide genuine diversity. So
each member is instead **level-corrected against its own expanding prior-season bias**
before blending, per position, causally. Blend correlation 0.5096 to 0.5115, MAE 1.2916 to
1.1920, bias +0.1655 to +0.0024. Ensemble disagreement is now measured between the
corrected members, so it reflects genuine uncertainty rather than differing calibrations.

The weighting scheme is unchanged: inverse-squared MAE was measured to give the best
within-Gameweek Spearman of the variants tried, and precision weighting on total MSE turned
out to discriminate poorly because roughly 95% of the error is irreducible noise.

### 31. Earlier findings re-confirmed on the repaired frame

- **Finding 8 still holds.** Across 300 sampled candidates the rank-multiplier search spans
  corr 0.51927-0.52357 (spread 0.0043), and **all 300 are worse on MAE** than not applying
  the multiplier at all (1.359-1.375 against 1.332).
- **Finding 11 is confirmed by the shipped `simulationSummary`**: 35.6 transfers over 34.2
  changed weeks, 2.7 rolls, 0.0 hits per season. The model makes exactly one transfer in 34
  of 38 weeks, so `free_transfers` is 1 almost always and the entire beam search,
  `package_route_search` and liquidity-frontier machinery is dead code in production.
- **Finding 6 is largely a symptom of finding 26.** Fixing the minutes calibration should
  remove most of the remaining price-tier bias without a separate recalibration layer.

---

## Tier 2 — structural ceilings on the search

### 6. The forecast is compressed; premiums are systematically under-rated

Decile calibration of `component_xpts` against realised points (fixtures only):

| decile | forecast | realised | bias |
|---|---|---|---|
| 0 | 0.396 | 0.019 | **+0.377** |
| 4 | 0.819 | 0.618 | +0.202 |
| 9 | 3.786 | 3.920 | −0.134 |

By price tier:

| price | forecast | realised | bias |
|---|---|---|---|
| < £4.5m | 0.896 | 0.602 | +0.294 |
| £7.5–9.0m | 2.917 | 3.247 | −0.330 |
| **£9.0m+** | **3.794** | **4.441** | **−0.646** |

The model under-rates every £9m+ player by 0.65 points a week and over-rates fodder by
0.3. An optimiser fed this will chronically under-invest at the top and over-invest on the
bench. A single monotone recalibration (isotonic or a two-parameter stretch) against
realised points, fitted causally, would recover most of this.

### 7. `market_role_xpts` is a bad model that poisons the ensemble's level

| model | corr | MAE | mean (realised = 1.293) |
|---|---|---|---|
| `component_xpts_structural` | 0.5005 | 1.221 | 1.219 |
| `empirical_xpts` | 0.4836 | 1.247 | 1.311 |
| `role_ridge_xpts` | 0.5105 | 1.212 | 1.267 |
| **`market_role_xpts`** | **0.4460** | **2.316** | **2.654** |
| blend `component_xpts` | 0.5156 | **1.312** | 1.416 |

`market_role_xpts` is biased **2× high** and has nearly double the MAE of the others, yet
inverse-squared-MAE weighting still hands it 10.9%. The blend therefore has a *worse* MAE
than three of its four members. Because transfer hurdles (`transfer_hurdle`,
`hit_immediate_hurdle`) and chip thresholds (`bench_score`, `triple_score`) are all
expressed in points, that +10% level bias propagates straight into every decision gate.

### 8. The 2,400-trial weight search is optimising a dimension with no signal

`candidate_forecasts` converts a candidate's weights into a multiplier
`calibration = 0.72 + 0.56 * (rank features @ normalised weights)`, i.e. a value in
`[0.72, 1.28]`, and multiplies the calibrated points forecast by it.

Sampling 250 candidates from the actual `candidate_pool()` Dirichlet:

- correlation with realised points spans **0.51201 → 0.51637** — a total spread of 0.0044
- **4 of 250** candidates beat simply *not applying the multiplier at all* (0.51560)
- the champion candidate's multiplier adds a **+21% level bias** (mean score 1.566 vs
  realised 1.293) and lowers correlation from 0.5156 to 0.5142

So the headline "2,400 historical weight trials" is searching a near-null direction. Any
differences that show up in replay points are simulation noise, and the multiplier's only
reliable effect is to break the points scale the hurdles depend on.

### 9. Walk-forward blending collapses to the pool mean

```python
ensemble_indices = np.argsort(train_score)[-12:]
trial_candidate = blend_candidates(ensemble_indices)
```

The finalist pool is 20 candidates; the walk-forward averages the **top 12 of 20** weight
vectors every season. Averaging 60% of a Dirichlet pool returns something very close to the
pool centroid regardless of the training signal. `blend_chip_policies` does the same with
12 of 48 policies. Combined with (8), the per-season "trained" model is effectively the
same model every year — which is exactly the plateau symptom.

### 10. The transfer search can only buy the ten most expensive players per position *(overstated — measured inert)*

**Correction.** The heading is wrong for the shipped configuration, and the measurement
says so. `transfer_candidate_limit` is read *only* inside `joint_transfer_plan`, and the
champion strategy runs `joint_squad_optimiser = False`, so that slice is never reached.
The path that actually makes transfers in production iterates the full `plan_order[:40]`.

Swept 10 / 16 / 24 / 32: **+0.0 on every season, to the decimal**. Identical totals across
four settings is not a small effect — it is the knob failing to reach the code, which is
what sent me looking. Reordering the frontier was tested separately and also failed:
`expand_transfer_frontier = True` scored +0.0 on training and **−9.1** on evaluation, so
the price-aware interleave is neutral at best.

What survives of the finding is narrower: the incoming universe is the top 40 by
*absolute* horizon score with price ignored, so a cheap enabler outside that 40 is
unreachable. That is a real ceiling, but it is a 40-deep one, not the 10-deep one written
below, and widening the order it is drawn in does not pay. The original text follows.



With `expand_transfer_frontier = False` (the default on every production strategy),
`incoming_by_position[position] = plan_order[:40]` — sorted by **absolute** horizon score,
price ignored — and `joint_transfer_plan` then slices `[:transfer_candidate_limit]` = **10**.

So the multi-move beam's entire incoming universe is the ten highest-scoring (i.e. most
expensive) players per position. Most are unaffordable and get skipped by the
`incoming_price > bank + sale` guard. Budget enablers, mid-price form picks and rotation
fodder are unreachable by construction. Note the greedy fallback loop iterates all 40, so
the two transfer paths do not even search the same space.

### 11. The multi-move machinery never actually fires

`max_moves = min(free_transfers, 5)`. From the shipped artifact, every season shows
`transfers ≈ weeksChanged` (e.g. 2018/19: 31 transfers over 30 changed weeks), so the model
makes exactly one transfer almost every week and `free_transfers` is essentially always 1.
The beam search, `package_route_search`, liquidity frontier, `next_transfer_option` route
valuation and `package_liquidity_states` are therefore dead in production — the model is a
one-transfer-a-week bot with an elaborate unused planner attached.

Root cause is the hurdle scale: `transfer_hurdle = 5.0–5.35` against horizon-scale plan
scores of ~20–40, i.e. under one point per week. Nothing ever gets banked.

### 12. Paid hits are permanently disabled

Every production strategy sets `max_hits=0, hit_immediate_hurdle=99.0`, and
`joint_transfer_plan` never models a hit at all. All eight seasons report `hits: 0`.
Given (11) — the model already burns its free transfer every week — the joint effect is
that the model can never respond to more than one problem per week, ever.

### 13. The backtested policy is not the policy that produces the live squad

Three different objectives are in play for the same decision:

| | objective |
|---|---|
| historical `initial_squad` | `lineup_scores − risk·risk_scores`, `bench_weight·bench`, `captain_weight·captain` (weights come from the strategy) |
| live `pick_squad` | `0.68·immediate + 0.18·horizon_per_game + 0.10·model_score·5 + 0.04·confidence·immediate` (hard-coded) |
| browser `squad-optimizer.mjs` | `0.68·projected + 0.18·(sixWeek/6) + 0.14·(liveScore·5)` (hard-coded, different again) |

The browser also divides the horizon by 6 while Python divides by `weighted_games`
(mean 4.03), under-weighting the horizon by ~33% relative to the Python solver, so the two
"exact" optimisers return different squads for the same inputs. And `pick_squad`'s
`captain_utility = captain_score * 5` is a rank-derived quantity, not expected points —
so the live captain is not chosen on the points scale.

**The 2,087-point backtest validates none of this.** Whatever the replay proves, it does
not apply to the squad the site actually shows.

---

## Tier 3 — correctness bugs

14. **Blank rows contaminate the calibration maps.** *(fixed)* In `causal_calibrate_distributions`,
    the scoring loop filters `fixture_count > 0` but the *update* loop (line ~322) does not.
    7,912 structural blanks (always 0 points) are folded into the isotonic
    blank/return5/haul8 bins and the uncertainty ratio histogram. The live twin
    `calibrate_live_distributions` **does** filter — so causal and live calibration are
    trained on different populations.

15. **Same bug in the role ridge.** *(fixed)* `causal_role_ridge_predictions` filters `observed` when
    predicting but not when updating `xtx`/`xty`, so every blank row teaches the model
    "these features → 0 points". `live_role_ridge_predictions` filters correctly.
    Same train/serve skew.

16. **Latent `KeyError` in `causal_role_ridge_predictions`.** *(fixed)* The update loop does
    `state = states[role_name]`, but `states` is only populated in the prediction loop,
    which `continue`s before `setdefault` when a role has no observed rows. A rare role
    (`set_piece_centre_back`) with zero active players in a small Blank Gameweek slate
    crashes the run.

17. **Bench Boost and Triple Captain silently refund a points hit.** *(fixed)*
    `week_points = base_breakdown["normal"] - hit_points_this_week`, but the chip branches
    then do `week_points = base_breakdown["bench_boost"]` / `["triple_captain"]`, dropping
    the `-4`. Latent today only because `max_hits = 0`; it will fire the moment hits are
    re-enabled.

18. **Wildcard over-credits free transfers.** After a Wildcard the code sets
    `free_transfers = 1` (pre-2024/25), then the end-of-week rule applies
    `min(bank_limit, max(0, 1 - 0) + 1) = 2`. Real FPL gives you 1 the following week.

19. **Look-ahead in the horizon ridge features.** *(fixed)* `horizon_feature_matrix` uses
    `horizon_weighted_games` (the **uncensored** count, built from the final future
    schedule) while everything else uses `..._censored`. They differ on 23% of rows.
    Contained today — `causal_horizon_ridge` is computed at line 2360 and never consumed
    by the production scorer — but it invalidates any research script that reads it.

20. **Inconsistent per-fixture normalisation.** *(already fixed)* `goal_rate`, `assist_rate`,
    `clean_sheet_rate`, `defensive_rate` and `bps_rate` divide by `fixture_count`;
    `save_rate`, `bonus_rate`, `yellow_rate`, `red_rate`, `conceded_rate`,
    `penalty_save_rate`, `penalty_miss_rate` and `own_goal_rate` use the raw per-GW sum.
    Measured: GKs average 0.81 saves in a single GW and 1.53 in a double; bonus 0.096 vs
    0.186. So a DGW inflates half the rate table by ~2× and leaves the other half alone.

21. **AFCON exclusion is a blanket nationality ban.** `precompute_fresh_squads` and
    `simulate_candidate` exclude *every* player holding one of 51 AFCON nationalities for
    the whole window, called up or not — so South African, Namibian and Zimbabwean players
    are benched alongside the actual absentees. There is no squad-list check.

22. **`refresh_current_artifact` reconstructs weights from rounded integer percentages.** *(fixed)*
    `Candidate.as_dict()` rounds to whole percent and dumps the residual onto the largest
    weight; `refresh_current_artifact` reads those back and divides by 100. `pnpm
    research:refresh` therefore runs on measurably different weights than the calibration
    that produced them.

23. **`calibration_curve` percentiles are wrong.** *(fixed)* `round(index / (len(candidates) - 1) * 100)`
    uses 2,399 as the denominator while `index` ranges over the 20 finalists, so every
    published percentile is 0–0.8%.

24. **`snapshot_scores / gameweeks` is meaningless.** `snapshot_replay` returns a
    normalised correlation in `[-1, 1]`; dividing it by the gameweek count doesn't produce
    a per-gameweek anything. Harmless in practice (all seasons are 37–38 GWs) but it makes
    the mean/std stability blend across seasons hard to reason about. `snapshot_stability`
    is computed on line 8339 and never used.

25. **`exception_threshold` is a quantile of the whole registered pool.** Both
    `initial_squad` and `pick_squad` take `quantile(immediate, 0.95)` over ~700 players,
    most of whom are third-choice. The "top 5% for immediate projected points" exception in
    the handbook is therefore a much lower bar than it reads.

---

## Tier 3 repairs — what was done

Findings 14, 15 and 16 turned out to be one defect wearing three hats. Both
`causal_calibrate_distributions` and `causal_role_ridge_predictions` filter
`fixture_count > 0` when *scoring* but not when *updating*, so structural blanks —
guaranteed zeros, because there was no fixture — taught both models that these
projections produce nothing. Both live twins already filtered, so the causal and live
paths were trained on different populations. And both had the same latent crash: the
scoring loop `continue`s before `setdefault`, so the update loop's unconditional
`states[key]` could raise `KeyError` for a rare role or position in a small Blank
Gameweek. Filtering the update loop and switching to `setdefault` fixes all three.

Findings 17, 19, 22 and 23 are repaired and inert or display-only:

* **17** — Bench Boost and Triple Captain overwrote `week_points` with a gross
  breakdown, handing back a `-4` that had already been paid. Wildcard and Free Hit
  are correct because both zero the hit first: they make the week's transfers free.
  Latent while `max_hits = 0`, and it would have fired the moment hits were enabled.
* **19** — the horizon ridge read the uncensored fixture count, built from the final
  schedule. Nothing in production consumes that ridge, which is the only reason the
  look-ahead never reached a score.
* **22** — the exact candidate weights are now stored alongside the rounded display
  percentages, so `--refresh-current` reconstructs the calibrated model instead of an
  approximation of it. Older artifacts without the field still load.
* **23** — the percentile denominator now matches what the index ranges over.

**20 needed no repair.** Every rate already divides by `appearance_denominator`, or by
minutes per 90 for goals and assists; the DGW inflation the finding describes was
removed by the Tier 1 work.

Findings 21, 24 and 25 remain open.

## Recommended order of work

Tier 1 and Round 2 are done. Remaining, in expected-value order:

1. **Bonus-point rates.** Under-predicted at every price tier (0.73-0.89 of the true
   per-90) because a 12-week rolling mean of a right-skewed variable lags. Together with
   the residual premium minutes gap this is most of the remaining -0.48 per week on
   GBP 9.0m+ players.
2. **Widen the transfer frontier to a price-aware candidate set, raise the hurdle so
   transfers actually bank, and re-enable hits behind the existing gate** (10, 11, 12).
   The model still makes exactly one transfer in 34 of 38 weeks and never takes a hit, so
   the entire beam search and package machinery is dead code.
3. **Replace the rank-multiplier weight search** (8, 9), or delete it and spend the compute
   on the decision policy instead. Re-measured after every repair, it still spans only
   0.0043 in correlation and every sampled candidate is worse on MAE.
4. **Reconcile the two optimisers** (13) - the live `pick_squad`, the browser
   `squad-optimizer.mjs` and the backtested `initial_squad` still use three different
   hand-coded objectives.
5. Work through Tier 3.

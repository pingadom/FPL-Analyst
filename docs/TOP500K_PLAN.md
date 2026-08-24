# Plan: closing the gap to top 500k

Target: **2,297 points a season** (mean of the eight measured top-500k cutoffs, each
anchored to sampled official manager ranks — 2024/25's 2,411 sits at rank 500,469).
Current model: **2,173.1**. Gap: **124 points**.

Everything below is priced from replay experiments, not estimated. Where a number is a
guess it says so.

---

## 0. The conversion rates, measured

Blending the shipped forecast towards the truth in small steps and replaying gives the
exchange rate between forecast skill and season points:

| forecast | correlation | season points | vs shipped |
|---|---|---|---|
| shipped | 0.5228 | 2117.4 | — |
| 2% truth | 0.5470 | 2126.3 | +8.9 |
| 5% truth | 0.5827 | 2165.1 | +47.7 |
| 10% truth | 0.6397 | 2203.1 | +85.7 |
| 20% truth | 0.7418 | 2279.1 | +161.7 |

**Roughly +7 season points per +0.01 of weekly forecast correlation.** Closing 150 points
by forecast accuracy alone would need correlation to go from 0.52 to about 0.73. That is
not achievable — it is most of the way to knowing the score in advance. **The gap has to
come mostly from the decision layer.**

The same experiment on the planning horizon:

| blend | season points | vs shipped |
|---|---|---|
| horizon 2% truth | 2141.6 | +24.2 |
| horizon 5% truth | 2171.1 | +53.7 |
| horizon 10% truth | 2237.4 | +120.0 |
| horizon 20% truth | 2393.2 | +275.8 |

The horizon is worth **1.5–1.7x** the immediate forecast per unit of skill. Perfect
foresight, same decision engine: one week ahead 2537, six weeks ahead 2797, both 3577.

The engine converts information into points extremely well — hand it the truth and it
scores 3577 against a 2297 target. Nothing here is an optimiser problem.

---

## Progress

### Closed: the objective is not the problem

A manager whose only signal is ownership, run through the identical decision engine —
same optimiser, same transfer rules, same chips:

| | mean | vs cutoff |
|---|---|---|
| template (ownership only) | 1924.4 | **-372.4** |
| model | 2171.4 | -125.4 |
| top-500k cutoff | 2296.8 | — |

The template misses the cutoff in all eight seasons, and **the model beats it by 247
points a season**. That closes the one genuinely fundamental question that was open: it is
not worth reworking the objective from expected points to something rank-aware. If the
crowd scored near the cutoff you would need differential play to beat it; the crowd scores
1,924, and no rank trick converts a 372-point deficit into a surplus. You get there by
being better, which is what maximising expected points already does.

It also vindicates the handbook's refusal to treat ownership as evidence, and retires the
earlier suggestion to weight it more heavily. Ownership correlates 0.38 with realised
points against the model's 0.47; the +0.056 residual correlation with model error is real
but small enough that the replays correctly measured it as noise.

Caveat on the number: this proxy builds the highest-total-ownership legal squad within
budget, which is harsher than a real template — actual managers own the popular premiums
and fill in with value picks. The direction is unambiguous at -372; 1,924 is not a claim
about what a typical manager scores.

### Result: +80.9 from repairing the forecast and the decision gate

| run | change | mean |
|---|---|---|
| — | original published model | 2087.0 |
| 3 | Tier 1 + Round 2 forecast repairs | 2110.1 |
| 6 | candidate-pooled walk-forward gate | 2154.5 |
| 7 | + widened chip range **and** paid hits | 2133.8 (-20.8) |
| 8 | + widened chip range only | 2140.9 (-13.6) |
| 9 | + gate decoupled from the chip pool | 2167.9 (+13.4) |
| **10** | **+ chips ranked by points, not ratio-to-threshold** | **2173.1 (+5.2)** |

Runs 7 and 8 look like failures and were not. The chip-range change was gaining **+153 in
2022/23 the entire time**, while simultaneously destroying **+203 in 2024/25** through a
coupling that had nothing to do with chips: `gate_policy` was drawn from the searched chip
pool, so editing that pool changed the conditions under which strategies were compared and
the policy switch stopped firing.

The net read as a small loss, and reverting on the net would have thrown away the good
half with the bad. Only the per-season breakdown showed 2024/25 collapsing in a way the
change could not plausibly explain. **A net number can hide two large opposing effects.**

`GATE_CHIP_POLICY` is now frozen and explicitly outside the pool. With the dependency gone
both effects coexist: 2022/23 keeps its +153 and 2024/25 recovers to 2317.

### Earlier: +42.9 from repairing the decision gate

| run | change | mean |
|---|---|---|
| — | original published model | 2087.0 |
| 3 | Tier 1 + Round 2 forecast repairs, incumbent-defending gate | 2110.1 |
| 4 | + walk-forward gate | 2110.1 (+0.0) |
| 5 | + regime-matched seasons | 2111.6 (+1.5) |
| **6** | **+ candidate-pooled gate** | **2154.5 (+42.9)** |

Concentrated in 2023/24 (+140) and 2024/25 (+203), and it comes from the walk-forward
gate finally switching policy in the later seasons.

The mechanism is visible in the gate's own error bars. Pooling three candidates roughly
halved every standard error — 47.7 to 29.3 on the leading challenger, 54.9 to 31.2, 22.2
to 19.3. Averaging across candidates removed a variance component that single-candidate
evaluation had been mistaking for signal.

**Three of the four gate changes were worth nothing on their own.** The walk-forward
structure had nothing to detect until the comparison feeding it was precise enough to
detect it. That is the argument for section 1: enabling work scores zero until the thing
it enables arrives.

The root cause, for the record: the gate judged strategies using a single
training-selected candidate, but the walk-forward then scored seasons with per-season
blended candidates. Whether the joint tree beats the incumbent *depends on the weights it
is judged with* — it trailed by 8.5 under one candidate and led by 32.7 under another. A
ranking that flips with the weighting is not a ranking.



**1.1 and 1.2 are done.** Both were enabling work: neither was expected to add points,
and neither did. What they bought is the ability to measure anything at all.

The gate now reports its own uncertainty, and the result settles the argument:

| option | training mean | mean delta | standard error | confidence |
|---|---|---|---|---|
| central: Six-GW planner *(incumbent)* | 2094.0 | — | — | held |
| central: Joint transfer-chip tree | 2085.5 | -6.5 | **47.7** | 0.446 |
| robust: Joint transfer-chip tree | 2080.0 | -13.6 | **54.9** | 0.398 |
| robust: Six-GW planner | 2072.5 | -21.0 | **22.2** | 0.166 |

The policies are separated by 6-21 points and the noise in estimating that separation is
22-55. **No challenger comes close to 75% confidence**, so the incumbent holds. The old
`max()` over these four numbers was selecting on noise two to three times larger than the
signal.

That reframes the reported score. The 2,147.2 recorded earlier was a lucky draw from this
coin flip; the stable estimate with the incumbent held is **2,110.1**. The model did not
get worse — the measurement got honest, and the headline should be read as
2,110 with a selection standard error of roughly 48, not as a point estimate.

Threshold rescaling validated on the experiment that motivated it: a leak-free supervised
six-Gameweek model went from **-60.2 to -24.5**, with the tuned baseline unchanged at
2117.4. So 36 of those 60 points were pure scale mismatch.

The remaining -24.5 is an open question and matters for step 3. The ridge is better on
overall correlation (0.5250 against 0.4998) *and* on within-Gameweek/position Spearman
(0.4188 against 0.3963), and its cross-position spread matches production (0.624 against
0.623). So it is not scale, not ranking quality, and not budget allocation. The only
distortion found so far is a relative one: the ridge over-rates goalkeepers by +0.48
against defenders at +0.14. Until this is understood, forecast work still cannot be
trusted to convert.

---

## 1. Fix the measurement apparatus first

Nothing else can be evaluated until these two are done. Both were discovered by trying to
make improvements and watching them fail to register.

### 1.1 The selection gate is a coin flip *(highest priority)*

Changing one chip constant — the Wildcard expiry floor — flipped the training-only gate
from "Joint transfer-chip tree + hold value" to "Six-GW planner + adaptive banking", and
the evaluation mean moved **2147.2 → 2107.1**, with individual seasons swinging +138,
−198, −196.

The gate chooses between decision policies using **two** pre-2018 training seasons. Those
policies differ by ±200 points per season on evaluation data. Selecting between them on
n=2 is noise, and it currently dominates the reported score. The headline number is
substantially determined by a coin flip.

**Do:**
- Score the gate on many bootstrapped season replays, not two seasons — the codebase
  already has four-GW block-bootstrap machinery in `add_rank_target_estimates`.
- Report the reported mean with its selection standard error. Right now ±40 is invisible.
- Stop re-selecting the strategy on every run. Freeze it, or require a new candidate to
  beat the incumbent by more than the selection noise.

*Expected value: 0 points directly. Without it every number below is unmeasurable.*

### 1.2 Decision thresholds are welded to the forecast's scale

A **strictly better** horizon model made the season worse:

| plan score | corr with 6-GW truth | season points |
|---|---|---|
| production horizon | 0.6927 | 2117.4 |
| supervised ridge (leak-free) | **0.7114** | **2057.2 (−60.2)** |
| 50/50 blend | — | 2104.3 (−13.1) |

A better forecast, worse decisions. The cause is that `transfer_hurdle = 5.35`,
`hold_option_value`, `additional_move_hurdle` and the chip gaps are **absolute constants in
points**. Change the forecast's dispersion and every one of them silently means something
different, so you measure the scale mismatch rather than the forecast.

**Do:**
- Re-express every threshold in units of the forecast's own spread (e.g. hurdle = k ×
  cross-sectional SD of the planning score) so it is invariant to forecast changes.
- Until that is done, **re-tune the thresholds jointly with any forecast change**, never
  in isolation.

*Expected value: unlocks the rest. Probably +30–50 on its own, because the current
constants were tuned to a forecast that has since been repaired four times.*

---

## 2. The decision layer — where the points actually are

### 2.1 Chips: the single largest addressable pool

Measured: **35–64 points a season**. Competent human play: **150–250**. This is a
100+ point pool and the biggest single item on the list.

The repair already made this better (chips played 17 → 40, contribution 35 → 64 with the
per-chip expiry floor), but the timing logic is still deliberately blind: it may not look
at the future schedule at all, because the historical archive has no announcement dates
for postponements.

**That constraint does not apply live.** The live path already reads the real fixtures
endpoint. So:
- Build a proper chip-timing model for the **live** path that uses the known future
  schedule to value Bench Boost / Triple Captain / Free Hit against the best remaining
  week in the window.
- Keep the archive-safe optimal-stopping ramp for the backtest, and accept that the
  backtest *understates* live chip value. Document the asymmetry rather than crippling
  the live model to match the backtest.
- Wildcard timing needs a trajectory-aware value, not a one-week gain. Its measured
  "local gain" was +76 in a season where it cost 30.

*Expected value: +60–85, the best-evidenced large item.*

### 2.0 The optimiser's curse is the binding constraint *(new, and it reframes 2.1-2.3)*

Across the 350 transfers the model actually makes:

| | |
|---|---|
| mean **predicted** six-Gameweek gain | 11.52 |
| mean **realised** six-Gameweek gain | 4.31 |
| **realisation ratio** | **0.374** |
| regression | realised = 0.433 x predicted - 0.68 (corr 0.247) |

**Only 37% of a predicted transfer gain materialises.** The beam picks the maximum over
many candidate bundles, and the maximum of noisy estimates is biased upward. This is not a
forecast-accuracy problem — the same inflation would occur with an unbiased per-player
forecast, because selection is what creates it.

It explains every decision-layer failure measured today:

- **Hits lose points** because the beam weighs a certain -4 against a gain inflated 2.7x.
- **The forced Wildcard** locally "gained" 76 points in a season it cost 30.
- **A better horizon model lost 24.5 points** even after the scale repair — improving the
  forecast does not help while the *gain* estimates are still inflated.
- `transfer_hurdle = 5.35` is not really a hurdle. It is an empirical fudge absorbing the
  curse, which is why it could not be derived and why it breaks whenever the forecast
  changes.

**Tried, and it does not work as stated — the shrinkage is redundant with the hurdle.**
Believing a share of the gain and comparing against a bar is the same test as comparing
the whole gain against a proportionally larger bar: `lambda * gain > hurdle` is
`gain > hurdle / lambda`. A joint sweep confirms it exactly — (1.00, 5.00), (0.60, 3.00)
and (0.35, 1.50) all score 2058.8, 2058.8 and 2058.9.

So `transfer_hurdle = 5.35` **already is** the curse correction. It was fitted rather than
derived, which is precisely why it could never be justified from first principles and why
it breaks whenever the forecast's dispersion changes. The 0.374 realisation ratio is the
explanation for the hurdle's existence, not a new lever on top of it.

The knob is retained at 1.0 (a no-op) because it stops being redundant the moment a fixed
cost sits beside the gain and does not scale with it — a paid hit, the package route
discount, a learned package adjustment. That is the only setting in which shrinking the
gain and moving the bar differ.

What the sweep *did* show, with the incumbent policy: the hurdle is set too high. Dropping
it from 5.00 to 1.50 moves the training seasons 2048.5 -> 2062.0 and all ten 2058.8 ->
2087.5, and it saturates below 1.5 (0.50 scores identically), meaning the binding
constraint disappears entirely. **This is not shipped**: the effect is +14 on training
against a measured selection standard error of roughly 48, so it is not distinguishable
from noise on two seasons, and "transfer every single week" is a degenerate policy that
deserves better evidence than this before being adopted.

*Expected value: 0 as originally conceived. The real lesson is that the hurdle is the
curse correction, so any future forecast change must re-derive it — which is what 1.2 now
does automatically.*

### 2.2 Paid hits are structurally impossible, not merely disabled

Setting `max_hits=3` changed the season score by **+0.0** and produced **0.0 hits**.
`joint_transfer_plan` caps moves at `max_moves = min(free_transfers, 5)` and never models a
paid transfer, so on the joint strategy the `max_hits` flag is inert. The greedy branch
supports hits; the branch actually used does not.

**Done, and it disproved the estimate.** The beam now searches paid moves, prices each at
its full -4 inside the objective and charges it against weekly points. `max_hits` finally
means what it says. Measured:

| hit priced at | mean | hits/season |
|---|---|---|
| 4.0 (the true rule) | 2039.2 | 21.4 |
| 8.0 | 2101.6 | 7.2 |
| 24.0 | 2105.5 | 0.3 |
| **hits off** | **2117.4** | 0.0 |

**No price makes hits profitable.** Every setting is below the hits-off baseline, and the
series only returns toward it as hits approach zero. The earlier "+20-40" estimate was
wrong, and hits should stay off until 2.0 is fixed — at which point they become worth
re-testing, because the machinery now exists and production is bit-unchanged with
`max_hits = 0`.

*Expected value: 0 today. Revisit after 2.0.*

### 2.3 Transfer cadence

35.2 transfers over 27.4 changed weeks, 9.5 rolls, 0 hits. The multi-move beam,
`package_route_search` and the liquidity frontier are all dead code in production because
`free_transfers` is almost always 1.

Widening the candidate frontier alone did nothing (**−1.3**), so this is not a search
problem. It is the hurdle scale again — see 1.2.

---

## 3. Forecast work — bounded, and only after section 1

At +7 points per +0.01 correlation, and with realistically +0.02–0.04 available from
better inputs, this is a **+15–30 point** bucket. Worth doing, but it is not the answer,
and doing it before section 1 means the gain will not register.

Ranked by expected correlation gain:

1. **Live team-news / predicted lineups.** The minutes model is the highest-leverage
   component and currently infers availability from history plus an official flag. Real
   press-conference and predicted-lineup data is the single biggest available input the
   model does not have.
2. **Player-level bookmaker markets** (anytime scorer, assists). Currently only team-level
   match odds are used. Player markets are the sharpest public forecast that exists.
3. **Residual premium gap.** £9.0m+ players are still under-projected 0.48/week and their
   expected minutes 3.9 short per fixture, after the calibration repair.

**Do not** bother with the bonus-point rates. They are under-predicted at every tier
(0.73–0.89 of true), which looks like a defect, but correcting them exactly is worth
**+0.3 points a season** — measured. A level error in a shared direction moves every player
together and changes no decisions. Only *differential* errors matter.

---

## 4. Measured dead ends — do not spend time here

| idea | measured effect |
|---|---|
| bonus-rate correction | **+0.3** |
| paid hits enabled (current code) | **+0.0** (structurally inert) |
| wider / price-aware transfer frontier | **−1.3** |
| ownership tilt into the forecast | −12.7 to +1.2 (noise) |
| better-correlated horizon model, thresholds unchanged | **−60.2** |
| the 2,400-trial weight search | 0.0043 correlation spread across 300 candidates; every one worse on MAE |

The ownership result is worth a note: ownership does carry residual signal
(`corr(ownership, model error) = +0.056`), but naively adding it to the projection does not
convert. If it is worth anything it is as a *risk* term for rank chasing, not as a points
term.

---

## 5. Budget

| item | evidence | points |
|---|---|---|
| 1.1 stabilise the selection gate | measured ±40 swing | 0 (enabling) |
| 1.2 scale-free thresholds | measured −60 on a better model | +30–50 |
| 2.1 chip timing, live-aware | measured 35→64 already; pool is 100+ | +60–85 |
| 2.0 shrink predicted gains | measured: algebraically redundant with the hurdle | 0 |
| 2.2 hits in the beam | measured: no price is profitable | 0 until 2.0 |
| 3 forecast inputs | +7/0.01 corr, +0.02–0.04 available | +15–30 |
| **total** | | **+105–165** |

The gap is **150–190**. So it closes, but only if the work goes into the decision layer and
the measurement apparatus. The instinct that "better data analysis should walk this" is
half right: the analysis is not the binding constraint — the machinery that converts
analysis into decisions is, and it is currently unable to register an improvement even when
one is handed to it.

---

## 6. Sequencing

1. **Selection gate** (1.1) — until this is stable, nothing else is measurable.
2. **Scale-free thresholds** (1.2) — until this is done, forecast work is wasted.
3. **Re-measure the repairs already made.** The Wildcard floor, the horizon un-censoring
   and the minutes calibration were each evaluated through a noisy gate; at least one
   (the Wildcard floor) looks good in controlled test and bad through the gate.
4. **Chips** (2.1) — the big one.
5. **Hits** (2.2).
6. **Forecast inputs** (3) — last, because that is when the gains will finally register.

## Current state

The working tree has `CHIP_EXPIRY_THRESHOLD_SHARE["Wildcard"] = 0.70`, which is better on
controlled evidence (+16.3 with the strategy held fixed, and chosen on training seasons
alone) but scores worse through the gate (2107.1 against 2147.2). It is kept because
reverting it to recover the higher published number would be selecting on the evaluation
seasons — the exact error this codebase is otherwise careful to avoid. Step 1.1 resolves
the ambiguity honestly.

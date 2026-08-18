# FPL chip strategy review

Date frozen: 2026-08-13

## Decision

The chip engine is now a constrained, manager-specific decision layer rather
than four independent threshold rules. Triple Captain and Bench Boost use
causal historical reservation values. A corrected learned-value Free Hit is
promoted only to prospective shadow. Automatic Wildcard remains disabled.

The strongest fully recursive historical combination tested is:

| Policy | 2018/19-2025/26 average | Gain over no chips | Status |
|---|---:|---:|---|
| No chips | 2,174.9 | - | frozen control |
| Audited TC + BB | 2,185.5 | +10.6 | historical control |
| Causal sequential TC + BB | 2,186.8 | +11.9 | research challenger |
| Audited TC + BB + corrected FH | 2,192.4 | +17.5 | prospective shadow |

The corrected Free Hit adds 6.9 points per evaluation season relative to the
audited TC/BB manager. On the later 2022/23-2025/26 holdout it adds 6.5 points;
its paired worst season is -15. The combined policy never falls below the
no-chip manager on that four-season holdout, but the FH activation sample is
small. This is not enough to claim a stable future +6.9 points.

The combined policy still records 0/8 reconstructed top-500k cutoff hits. Its
average cutoff gap improves from -121.9 points for the no-chip champion to
-104.4, so chips help but do not solve the larger player-ranking and recursive
transfer gap.

Automatic Wildcard is rejected. The training-selected rule lost 49.6 points per
evaluation season and the least-bad tested threshold still lost 16.9. This does
not mean the chip is bad; it means the automatic trigger and rebuild forecasts
do not yet value its persistent counterfactual well enough.

## Official constraints encoded

- There are two complete chip sets: Wildcard, Free Hit, Bench Boost and Triple
  Captain once in each half, for eight chips total.
- The first set expires at the GW19 deadline and cannot be carried forward.
- Only one chip may be used in a Gameweek.
- Wildcard and Free Hit are unavailable in GW1. A GW19 Free Hit cannot be
  followed by the refreshed Free Hit in GW20.
- Wildcard and Free Hit keep previously banked transfers under the current
  rules, although the transfer received for the activation Gameweek is consumed.
- Triple Captain passes to the vice-captain if the captain does not play.
- Free Hit uses the manager's current selling budget and restores the permanent
  squad and bank after the Gameweek.

Primary rules source:
https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627

Detailed FAQ:
https://www.premierleague.com/en/news/4661030

## Evidence from strong managers

The evidence supports opportunity-based use, not a universal calendar:

- The 2025/26 champion described patience as essential, then used a coordinated
  WC32 -> BB33 -> FH34 -> TC36 sequence around the largest announced double and
  blank. The Wildcard was valuable because it simplified every later decision,
  not merely because its activation-week score was high.
- Top-50 usage clustered Bench Boost around the major double, while Triple
  Captain was also used successfully on elite premium single fixtures. This is
  why the new model permits a strong single-Gameweek TC and a reliable early BB
  rather than requiring doubles mechanically.
- Academic evidence on roughly one million managers found that high-ranked
  managers waited for major blank/double opportunities more often and that
  planning around these weeks separated skill tiers. It also found substantial
  luck, so realized chip points alone are a noisy policy label.

Sources:

- https://www.premierleague.com/en/news/4672128
- https://www.fantasyfootballfix.com/blog-index/fpl-top-50-tips-chips-25-26/
- https://arxiv.org/abs/2009.01206

## Engine defects fixed

1. **Zero option value.** The historical engine described a continuation value
   but hard-coded it to zero. TC/BB now compare use-now value with an empirical
   prior-season reservation value that decays toward half-season expiry.
2. **One chip name instead of two inventories.** State now stores keys such as
   `Triple Captain:H1` and `Triple Captain:H2`.
3. **Bench Boost overstatement.** The old live calculation added all four bench
   forecasts. The new paired simulation subtracts points that would already
   arrive through legal autosubs. On the frozen GW1 example this reduced the BB
   estimate from 14.8 to 10.5.
4. **Captain fallback.** TC scenarios now transfer the extra captain multiplier
   to the vice-captain when required.
5. **Generic £100m chip squads.** WC/FH now use each manager's sale value plus
   bank. The FH bench is explicitly cheap because only the starting XI scores.
6. **Free Hit persistence bug.** FH no longer runs permanent transfers and then
   accidentally saves them. The decision is made before transfers and the
   permanent state remains unchanged.
7. **Free Hit preflight bug.** The old engine suppressed transfers whenever FH
   was available in a large blank, even when the later gate recommended Hold.
   It now suppresses the transfer planner only after the FH gate passes.
8. **Missing opportunity cost.** FH now competes against the best permanent
   transfer action. Its one-week XI edge is charged for long-term transfer value
   foregone.
9. **Shared wait heuristic.** Each chip now has its own reservation value. TC/BB
   values come from causal history; FH/WC retain conservative live gates until
   stronger validation exists.
10. **Expiry collision.** Remaining chip count is compared with remaining weeks;
    reservations collapse when chips must be used in distinct final weeks.

## Current policy

### Triple Captain

Use the additional captain points as the chip's marginal value. Require strong
minutes and downside. A double is valuable but not mandatory: an elite premium
single fixture may beat the decaying reservation value, especially before GW19.
Do not use names as rules; premium price, return probability, expected minutes,
set-piece role and opponent quality must generate the signal.

### Bench Boost

Use incremental points above normal autosubs, not raw bench points. Require four
credible appearances, adequate expected minutes and a positive lower tail. For
a WC -> BB plan, subtract the setup cost: transfers/hits, starter value lost by
funding the bench, and the value of the post-BB squad. A double raises ceiling
but does not rescue non-starting bench players.

### Free Hit

Use only as an alternative action to the permanent transfer plan. Require
announced blank/double structure, a manager-specific temporary squad, positive
paired downside, and learned net value after foregone permanent moves. The
corrected policy is a shadow challenger because its signal correlation is still
only 0.231 and its MAE is 13.41 points.

### Wildcard

Keep automatic activation disabled. Prospectively, trigger a review when there
are multiple injuries/non-starters, large role changes, a fixture swing, AFCON
departures, or an announced WC -> BB structure. Score WC across at least 8-10
weeks plus terminal squad value and transfer/hit savings. Do not score it by the
activation Gameweek.

## Sequence optimiser specification

For every announced schedule revision, enumerate legal chip sequences over the
remaining half-season. Each node stores permanent squad, purchase prices, bank,
banked transfers, used chip-set keys and model version. Each transition compares:

`expected points + terminal squad value - hit cost - setup cost - risk penalty`

Constraints include one chip per Gameweek, FH non-persistence, GW19/GW20 FH
separation and expiry. Use common random draws across sequences. Re-optimize
weekly from the locked state, but never revise past decisions.

The WC -> BB, FH blank and TC double patterns are candidates generated from the
announced schedule, not hard-coded mandatory plays. The no-chip/hold sequence is
always present.

## Remaining uncertainty

- Historical fixture announcement vintages are unavailable, so exact WC -> BB
  planning cannot be replayed without hindsight. It must be evaluated
  prospectively from frozen announcements.
- The FH learned-value model improves raw correlation from 0.166 to 0.231, still
  weak. It is useful as a gate, not as a precise point forecast.
- The sequential TC/BB challenger improves the old policy by only 1.3 points on
  average and has mixed paired season deltas. Keep both in shadow.
- There is no credible guarantee of top 500k. Chip gains are real but small
  relative to the remaining player-ranking and transfer-path gap.

## Reproducible artifacts

- `sequential_chip_value_validation.py/json`
- `wildcard_freehit_ablation.py/json`
- `freehit_value_validation.py/json`
- `combined_chip_policy_validation.py/json`
- `chip_scenario_planner.py`
- `run_shadow_cycle.py`
- `test_prospective_pipeline.py`

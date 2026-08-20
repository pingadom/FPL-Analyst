# Breakthrough implementation review — 20 August 2026

## Outcome

All planned research stages were implemented and evaluated against the frozen,
leak-free 2018/19–2025/26 recursive control. The historical control remains
2,190.6 points because none of the new recursive transfer/fieldability policies
passed the declared promotion gates. The work nevertheless closes several live
engineering gaps and establishes a much stronger, auditable experimentation
boundary.

The average reconstructed top-500k pace is 2,296.8 points, leaving the control
106.2 points short. A refreshed unlimited-weekly-rebuild audit scores 2,242.9,
which decomposes the gap into roughly 53.9 forecast/ranking points and 52.3
recursive decision-path points. Optimisation changes alone therefore cannot
honestly guarantee top 500k.

## Stage results

| Stage | Result | Decision |
|---|---:|---|
| Frozen recursive benchmark | 2,190.6 average; 2,032 minimum | Retain as control |
| Weekly-rebuild forecast ceiling | 2,242.9 average | Forecast layer still needs about 54 points |
| Correlated generative scenarios | 93.56% central-80 coverage; 0.178 within-club GK/DEF correlation; 0.19 mean drift | Use for paired action/chip comparisons, not as a replacement mean |
| Fieldability policy | best structural variant −8.1 average; −81 worst season | Reject historical promotion; retain exact-news live shadow only |
| Expanded action-value learner | 70,247 states; best candidate −41.1 average | Reject learned action corrections |
| Causal regime detector | 0.895 walk-forward AUC versus 0.550 baseline proxy | Export as diagnostic only; never add directly to xPts |
| Premium access | detected Salah 2019/20 and Haaland 2025/26 access failures | Audit legal funded packages and captain option value; no named-player bonus |
| TC + BB | +10.6 average historical gain | Retain audited policy |
| Corrected Free Hit | +6.9 incremental average; −15 holdout worst | Retain in prospective shadow with downside gate |
| Automatic Wildcard | −49.6 average; −140 worst | Reject |
| Joint chip sequence DP | inventory, one-chip-per-GW, FH non-persistence and WC persistence implemented | Diagnostic receding horizon pending prospective evidence |

## Implemented architecture

`breakthrough_engine.py` is the shared decision boundary. It contains:

- nested official availability and fixture extraction;
- fieldability audits and hard-unavailable rules;
- explicit appearance and 60-minute scenario draws;
- shared club attack shocks and clean-sheet outcomes;
- legal autosubs and captain-to-vice fallback;
- common-random-number action comparisons with downside gates;
- legal premium-access diagnostics; and
- finite-horizon chip inventory/state optimisation.

Historical validation scripts are deliberately separate from live generation.
Every rejected idea writes its evidence artifact but cannot mutate the frozen
control. The live regime artifact is labelled `diagnostic-only`, and the failed
action learner is not loaded by the weekly process.

## Live weekly decision changes

The prospective pipeline now has five pre-registered managers, adding an
isolated `breakthrough-decision` shadow. The weekly cycle:

1. captures a new official provisional/locked snapshot;
2. rebuilds causal live rankers and regime diagnostics;
3. creates a base chip plan;
4. jointly optimises each manager's XV, XI and captain;
5. reruns chip scenarios on each exact XV; and
6. regenerates the final manager decisions.

This two-pass squad/chip cycle fixes the former mismatch where GW1 chip advice
could be calculated for a different XV from the eventual shadow squad.

The squad integer programme now uses separate horizon, immediate-lineup and
captain objectives. It still enforces £99.5m minimum spend (or the manager's
sale-value budget), the £100m cap, positional quotas, club limits, legal XI,
captain membership, bench premium limits and fieldability constraints.

## Current provisional GW1 audit

Snapshot `1be44aa9a995…` is provisional and has not updated recursive state.
The breakthrough manager currently:

- spends exactly £100.0m;
- owns and captains Haaland;
- has no hard-unavailable player in the XI and all 11 have a fixture;
- benches Verbruggen £4.5m, Shaw £4.5m, Justin £4.5m and Yarmoliuk £5.0m;
- flags Bruno Fernandes for premium-package review rather than forcing him; and
- provisionally recommends Bench Boost after the manager-specific second pass.

The Bench Boost call is narrow: its risk-adjusted value is about 0.9 points
above the past-only reservation value. It is not locked and must be regenerated
from the final deadline snapshot.

## Honest limitations and next research priority

The recent-season gap is real: average margin changed from −23.2 across the
first four evaluation seasons to −189.0 across the last four. The weekly ceiling
shows that player/team/minutes forecasting is now the primary bottleneck, while
the failed action learner shows that small one-step prediction gains can become
large recursive losses.

The next high-value research programme should therefore focus on forecast
calibration by route—team attacking strength, clean-sheet probability, player
minutes/start probability and role/set-piece changes—using deadline-vintage
market and team-news data. New action models should be trained against policy
rollouts or doubly robust long-term value targets, not one-step residual MAE.
Production promotion still requires prospective locked decisions and cannot be
claimed from these eight research-exposed seasons alone.

## Reproduction

```powershell
python analysis/availability_leak_audit.py
python analysis/breakthrough_benchmark.py
python analysis/breakthrough_fieldability_validation.py
python analysis/breakthrough_generative_validation.py
python analysis/breakthrough_action_value_validation.py
python analysis/breakthrough_premium_regime_validation.py
python analysis/breakthrough_chip_sequence_validation.py
python -m unittest discover analysis -p "test_*.py"
pnpm test
pnpm lint
```

The generated evidence lives in `analysis/data/breakthrough_*.json`.

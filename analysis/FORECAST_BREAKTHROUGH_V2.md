# Forecast breakthrough v2 — 20 August 2026

> Superseded later on 20 August 2026 by `FULL_PROGRAMME_AUDIT_V3.md`.
> The statement below that the programme was implemented end to end was too
> strong: action-specific recursive tests, a large captain surface, chip
> retuning and the final integrated replay had not yet been completed.

## Outcome

The forecast programme is implemented end to end. The frozen recursive control
scores **2,190.6** points across 2018/19–2025/26. The development-selected
challenger scores **2,194.2** (+3.6), with a −3 worst-season delta and +4.5 on
the two untouched recent holdouts. A predeclared combined player/match model
scores **2,195.8** (+5.1), including +11.5 on the holdouts, but it did not beat
the selected model's development-stability criterion and was not promoted after
seeing the holdout.

The selected research configuration is frozen as
`forecast-v2-dynamic-captain-030-share-010`: the existing recursive squad and
transfer model remains unchanged, while 10% of captain rank comes from a 0.30
strength dynamic match-route score. Historical odds lack exact capture times,
so this remains research-only until a sufficient locked prospective sample is
scored.

## What changed

- `deadline_vintage.py` enforces source hashes, deadline timestamps, forbidden
  realised/closing fields, and separate provisional-shadow vs locked-promotion
  eligibility.
- `dynamic_match_model_v2.py` combines structural team ratings with no-vig
  first-market odds using walk-forward, prior-season-selected weights.
- `forecast_layer_v2.py` decomposes changes through attack, clean-sheet and
  goals-conceded routes and represents no-show/cameo/start/60+ minutes mixtures.
- `openfpl_position_ensemble_v2.py` fits a causal position-specific ensemble of
  structural, role, selectable-frontier and scoring-route forecasts.
- `forecast_breakthrough_tournament_v2.py` runs 15 paired full recursive
  candidates with identical transfer, optimiser and audited chip rules.
- `action_specific_tenure_v2.py` exposes h1 captain/XI/Free-Hit, h3 bench,
  player-specific transfer and h10 Wildcard value surfaces.
- `safe_policy_evaluation_v2.py` provides a tested doubly-robust estimator and
  refuses a historical DR claim because the 70,247 action states do not contain
  logged behaviour propensities and realised chosen-action rewards.
- `forecast_champion_v2.py` is the single frozen research integration point.

## Forecast evidence

| Model | Goals-for MAE | Goals-against MAE | Clean-sheet Brier |
|---|---:|---:|---:|
| Structural | 0.9607 | 0.9729 | 0.23587 |
| First market | 0.9055 | 0.9141 | 0.22495 |
| Causal dynamic blend | 0.9065 | 0.9161 | 0.22564 |

The position ensemble improves selectable-frontier Spearman correlation from
0.4219 to 0.4289, top-15 realised points from 4.9282 to 4.9787, and missed-haul
rate from 0.8163 to 0.8096. Those gains are real forecast improvements, but the
broad recursive selection variants were unstable. They remain challenger
evidence rather than silent changes to transfers.

## Recursive tournament

| Candidate | Average | Delta | Development | Holdout | Worst season |
|---|---:|---:|---:|---:|---:|
| Frozen control | 2,190.6 | — | — | — | — |
| Selected dynamic captain | 2,194.2 | +3.6 | +3.3 | +4.5 | −3 |
| Dynamic captain, stronger | 2,194.5 | +3.9 | +2.8 | +7.0 | −5 |
| Predeclared combined | 2,195.8 | +5.1 | +3.0 | +11.5 | −3 |
| OpenFPL selection 5% | 2,192.1 | +1.5 | −0.3 | +7.0 | −2 |
| OpenFPL defence 10% | 2,189.2 | −1.4 | −2.5 | +2.0 | −13 |

This is progress, not the claimed top-500k breakthrough. Against the currently
reconstructed top-500k pace of 2,296.8, the selected model remains about 102.6
points short on average. The result narrows the gap without overstating rank or
using hindsight.

## Prospective pipeline

The weekly cycle now attempts a sanitised, timestamped market capture before
fitting the live rankers. When exact current odds exist, the sixth pre-registered
`forecast-breakthrough-v2` shadow uses the selected captain boundary. When the
source is unavailable, it explicitly falls back to route consensus. It never
reuses a stale prior-season market file.

At this provisional GW1 capture the current-season source returned HTTP 300, so
the fallback is active. The shadow squad remains legal, spends £100.0m, owns and
captains Haaland, and has 11/11 starters with fixtures. No recursive state is
updated until the official snapshot and shadow decision are locked.

## Reproduction

```powershell
python analysis/deadline_vintage.py
python analysis/dynamic_match_model_v2.py
python analysis/openfpl_position_ensemble_v2.py
python analysis/forecast_breakthrough_tournament_v2.py
python analysis/action_specific_tenure_v2.py
python analysis/safe_policy_evaluation_v2.py
python -m unittest discover analysis -p "test_*.py"
pnpm test
pnpm lint
```

The decisive artifact is
`analysis/data/forecast_breakthrough_tournament_v2.json`; it contains every
season total and selection gate, not only the winning average.

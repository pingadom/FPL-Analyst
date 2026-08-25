# FPL Lens

An interactive Fantasy Premier League analyst that calibrates player-ranking
weights across every completed season from 2018/19 through 2025/26.

The product includes:

- an exact mixed-integer £100m squad/XI/captain optimiser;
- adjustable performance, value, age, fixture, market, and form-memory weights;
- current 2026/27 FPL player prices, ownership, and fixtures;
- 2,400 historical weight trials with walk-forward season checks; and
- a reproducible, leak-free analysis pipeline in `analysis/calibrate_model.py`.

## Local development

Requires Node.js 22.13 or later and Python with pandas/numpy for recalibration.

```bash
npm install
npm run dev
npm run build
python analysis/calibrate_model.py
pnpm research:refresh
```

Historical data is cached under `work/` and is not committed. The generated
model artifact lives at `app/data/model-results.json`.

Three supporting modules sit alongside the pipeline:

| module | what it does |
|---|---|
| `analysis/harness.py` | replays a *pinned* configuration so one change can be measured without the pipeline re-searching around it. A screen, not a release decision — confirm with a full run. |
| `analysis/historical_odds.py` | free *opening* betting prices for all ten replayed seasons, de-vigged into implied team goals. The backtest previously had no market view at all. |
| `analysis/score_analysis.py` | average-score analysis for one or more runs — mean, spread, margin to that season's real top-500k cutoff, and how much of the total came from chips rather than from picking players. |
| `analysis/european_fixtures.py` | Champions League, Europa and Conference fixtures for English clubs, so the minutes model can see the midweek games the Premier League archive does not contain. |
| `analysis/team_identity.py` | recovers club names for 2016/17 and 2017/18, whose archive ships no team list, and verifies them against an independent match record (380/380 on both seasons). |

Older breakthrough implementation notes are retained as research history, but
their 2,212-point headline is retired because it did not reproduce under the
repaired schema and cache rules.

For a non-technical explanation of every analytics layer, read
[`docs/MODEL_HANDBOOK.md`](docs/MODEL_HANDBOOK.md). The explanation-driven
legacy review and Lens 8 acceptance gate are in
[`docs/DECISION_AUDIT_LENS8.md`](docs/DECISION_AUDIT_LENS8.md).
The fair Lens 7/Lens 8/causal-shadow comparison and the stale-cache root cause
are in [`docs/PERFORMANCE_AUDIT_LENS8.md`](docs/PERFORMANCE_AUDIT_LENS8.md).
The causal injury-source, no-show and long-held-player review is in
[`docs/AVAILABILITY_AND_HELD_PLAYER_AUDIT.md`](docs/AVAILABILITY_AND_HELD_PLAYER_AUDIT.md).
The independent logic audit, its repaired defects and the open backlog are in
[`docs/MODEL_LOGIC_AUDIT.md`](docs/MODEL_LOGIC_AUDIT.md). The costed plan for reaching a
top-500k season average is in [`docs/TOP500K_PLAN.md`](docs/TOP500K_PLAN.md).

## Data

Historical player-gameweeks come from the public vaastav FPL dataset. Current
players and fixtures come from the official FPL API. Birth-date coverage for
older seasons is completed through the open Reep identity register.

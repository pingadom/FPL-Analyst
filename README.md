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

## Data

Historical player-gameweeks come from the public vaastav FPL dataset. Current
players and fixtures come from the official FPL API. Birth-date coverage for
older seasons is completed through the open Reep identity register.

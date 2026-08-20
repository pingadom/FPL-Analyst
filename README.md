# FPL Lens

An interactive Fantasy Premier League analyst that calibrates player-ranking
weights across every completed season from 2018/19 through 2025/26.

The product includes:

- a legal £100m, 15-player squad optimiser;
- adjustable performance, value, age, fixture, market, and form-memory weights;
- current 2026/27 FPL player prices, ownership, and fixtures;
- 640 historical weight trials with walk-forward season checks; and
- a reproducible, leak-free analysis pipeline in `analysis/calibrate_model.py`.

## Local development

Requires Node.js 22.13 or later and Python with pandas/numpy for recalibration.

```bash
npm install
npm run dev
npm run build
python analysis/calibrate_model.py
```

Historical data is cached under `work/` and is not committed. The generated
model artifact lives at `app/data/model-results.json`.

The latest full breakthrough implementation and promotion decisions are
documented in `analysis/BREAKTHROUGH_IMPLEMENTATION_2026-08-20.md`.
The forecast-first follow-up, including the full recursive tournament and the
frozen v2 research challenger, is documented in
`analysis/FORECAST_BREAKTHROUGH_V2.md`.

## Data

Historical player-gameweeks come from the public vaastav FPL dataset. Current
players and fixtures come from the official FPL API. Birth-date coverage for
older seasons is completed through the open Reep identity register.

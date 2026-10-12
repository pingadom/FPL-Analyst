# Backend audit, October 2026

Scope: everything that runs without a browser. That is the model pipeline
(`analysis/calibrate_model.py` and the modules it imports), the frozen promotion audit,
the daily pick-log job, the API routes under `app/api/`, the Cloudflare worker
scaffolding and the local caches. The front end was audited separately
(`docs/AUDIT_2026-10.md`).

## How the backend fits together

| Piece | Runs | Writes |
|---|---|---|
| `audited_policy_validation.py` then `calibrate_model.py` (full run, ~65 min) | by hand after any engine change | `analysis/data/audited_policy_validation.json`, `app/data/model-results.json`, `app/data/current-players.json` |
| `weekly_cycle.py` (score, refresh, record, report) | daily scheduled task `fpl-weekly-pick-log` | `app/data/pick-history.json` and the two app files |
| `app/api/projections` | per request | nothing; overlays live FPL injury news on the artifact |
| `app/api/team/[entry]` | per request | nothing; reads a public FPL entry |
| `app/api/fixtures` | per request | nothing; live FPL fixture list plus the model's team ratings |
| `app/api/research` | per request | nothing; serves frozen research files |

The production engine imports seven local modules: `live_external_signals`,
`team_identity`, `understat_history`, `european_fixtures`, `historical_odds` and
`deadline_news`. Fifteen or so further modules are imported only by tests and harnesses.
The other ~150 scripts in `analysis/` are one-off research and are not on any production
path.

## Findings

Severity: **high** means it can silently produce wrong numbers; **medium** means wrong or
misleading output, or wasted resources; **low** means hygiene.

| # | Sev | Finding | Status |
|---|---|---|---|
| 1 | high | The prepared-history cache (the feature table every run reads) was accepted whenever its season list matched. Changing feature code without hand-renaming the cache file silently replayed the old features. | **Fixed**: the cache stores a hash of the six source files that build it and rebuilds on mismatch. |
| 2 | high | The live pick log did not apply automatic substitutions. GW2 had three starters who did not play, so the log under-reported the model's real FPL score. | **Fixed**: FPL autosub rules (formation minimums, goalkeeper for goalkeeper, bench order stored per entry). Rescored by code with `pick_history.py --rescore`: GW2 82 → 97, GW3 44 → 52. |
| 3 | high | The pick log scored a Gameweek as soon as FPL marked it `finished`, before bonus points are confirmed, and never rescored it. A provisional total could be frozen in. | **Fixed**: waits for `data_checked`. |
| 4 | medium | `chipStrategy.current` was hard-coded to "Hold" with fixed wording and never touched by the daily refresh, so it could contradict the chip the model was playing (it showed GW6 while the model was on GW7). | **Fixed**: built from the live decision in both the full run and the refresh. |
| 5 | medium | The frozen-audit fingerprint covered `domestic_cups.py`, which nothing imports, and omitted `deadline_news.py`, which feeds the backtest frame. | **Fixed**: list now matches the imports. |
| 6 | medium | `european_fixtures` swallowed failed downloads. Harmless while the files are cached, but a fresh checkout would silently lose fixture-congestion features. | **Fixed**: failures are reported on stderr. |
| 6b | medium | The new warning fired on its first run: openfootball has no 2026/27 Champions League file (HTTP 404), so the **live** model currently has no European congestion data for this season. Until now that was invisible. | **Open**: needs another source for the current season's European fixtures, or the file once openfootball adds it. |
| 7 | medium | `/api/projections` downloaded FPL's 1.5 MB bootstrap file on every request, and the fixtures and team routes fetched it again. | **Fixed**: one shared 60-second cache (`app/lib/fpl-api.ts`). |
| 8 | medium | Every page load pulled 2.3 MB of player data, 38% of it research feature vectors the site never reads. | **Fixed**: `view=app` drops them (1.3 MB); the public API default is unchanged. |
| 9 | medium | The daily job runs as an agent session. Today's run has been waiting since 13:42 on a Bash permission prompt nobody can answer, so on such days the pick is not recorded or scored. | **Open**: needs a permission allow-rule for that task's commands (your decision, not changed here). |
| 10 | medium | 37 prepared-history caches (0.5 GB each, 18 GB in total) sit in `work/fpl-data`, and only one is current. The folder is inside OneDrive, so all of it syncs. | **Open**: safe to delete all but the current cache; better, move `work/` out of OneDrive. Not deleted without your say-so. |
| 11 | medium | The per-player P10/P90 was `xPts ± a symmetric residual half-width`, not a quantile. For skewed FPL outcomes it was wrong exactly for premiums (Haaland 4.1–7.9 with a 44% blank chance); 45 players had an 8+ chance above 10% but a P90 below 8. | **Fixed**: `outcome_quantiles` builds P10/median/P90 from the probabilities the model already estimates (not playing, under 60 minutes, ≤2, 5+, 8+). Haaland now 0.9 / 3.1 / 14.5; inconsistent players 0. Coverage of the 80% interval on history is 87% (integer points make it slightly wide), against 79% for the old band. |
| 12 | low | No `requirements.txt`. The pipeline needs numpy 2.4, pandas 3.0, scipy 1.18 (milp) and pyarrow 25; nothing records that. | **Fixed**: `analysis/requirements.txt`. |
| 13 | low | `/api/research` serves research files frozen on 29 August (Lens 7/8, shadow managers, chip scenarios). | **Open**: nothing on the site uses it; delete the route and files when convenient. |
| 14 | low | `worker/`, `db/`, `drizzle/` and `app/chatgpt-auth.ts` are hosting-template scaffolding: the D1 binding is null, the schema is empty, nothing imports the auth helper. | **Open**: harmless; keep while the hosting template expects them. |
| 15 | low | `domestic_cups.py` has no data for 2016/17, 2017/18 or 2025/26 (the source returns 404) and is not imported. | Noted: if it is ever wired in, its feature would exist in only seven of ten seasons, including neither training season. |
| 16 | low | Domestic cup, European and news coverage all vary by season (news only from 2020/21). | Noted: a feature whose strength varies by season is fitted on uneven evidence. |

## Checked and fine

- No credentials or tokens in the repository; no `.env` files tracked.
- Every outbound HTTP call in the Python pipeline has a timeout; there are no bare `except:` clauses, and only three broad `except Exception` blocks. Of those, Matchbook's records an "unavailable" status, European fixtures now report, and domestic cups are unused.
- API inputs are validated: the team ID is checked against `^\d{1,10}$` before any outbound call; `limit` and `position` are bounded.
- The pick log refuses to record after a deadline and refuses to overwrite a scored week; both are tested.
- Every `npm run research:*` script points at a file that exists; no caches, logs or bytecode are tracked in git.

## Verification

- Engine changes 1, 4, 5, 6 and 11 alter no backtest number. Two full audit + pipeline reruns (the first on a freshly rebuilt feature cache) reproduced the published backtest, 2,208.1, and the frozen audit, 2,141.8, season by season.
- Python tests: pick log 13/13 (five new autosub and `data_checked` tests); engine contract tests for the chip call and fingerprint coverage.
- JS: build, `tsc`, eslint and 8/8 node tests.

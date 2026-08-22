# FPL Lens model handbook

## What the model is trying to do

FPL Lens has two separate jobs:

1. estimate the points each player could score, including the chance that they do not start; and
2. turn those estimates into a legal 15-player squad, starting XI, captain and transfer/chip plan.

Those jobs must stay separate. A good forecast can still produce a bad team if the optimiser wastes money on the bench, and an exact optimiser can confidently magnify a bad forecast. Lens therefore tests the player forecast and the decision engine independently before judging total season points.

The model is decision support, not a promise of rank. Football contains injuries, red cards, deflections and tactical changes that no pre-deadline dataset can know.

## The journey from data to a team

### 1. Take a deadline snapshot

Immediately before a deadline, the model reads the official FPL player list, prices, teams, positions, ownership, fixtures, injury flags and news. The public projection endpoint adds a short-lived official availability overlay so a late flag is not hidden by an older generated file.

Historical tests use only information that existed before the simulated deadline. A result is shifted back before it can become a feature. If the timestamp of a historical announcement is unknown, the model censors it instead of pretending that the manager knew it.

### 2. Estimate whether the player will play

The minutes model has three branches:

- starts;
- appears from the bench; or
- does not play.

It uses past starts, substitute appearances, minutes when starting, team rotation, positional competition, rest time and official availability. These probabilities affect appearance points and every other scoring route.

Those raw estimates are then **recalibrated against what actually happened**. Smoothing a small sample toward a positional average, and then applying rotation, rest and competition penalties that can only ever push a start probability down, compresses everybody toward the middle: a nailed starter comes out below his true rate and a fringe player above his. Left uncorrected that under-rated expensive players on every minutes-scaled route — a £9m+ player was credited with 13% fewer minutes than he actually played and a 60-minute probability 14 points too low — while flattering cheap bench fodder.

The correction is an isotonic map from predicted to realised, learned separately for each position and each of three deadline-known price bands, and fitted only on earlier Gameweeks. The price band matters because the predicted probability on its own is not enough: among players the raw model rated identically, the expensive ones started far more often. Price is the market's view of who is first choice, it is known before the deadline, and it separates that residual. The live deadline uses the same maps, fitted on all completed history.

For the live XI, an ordinary starter needs at least a 70% start probability and an 84% play probability. At most one exceptional player may miss the play floor, and only if they remain in the top 5% for immediate projected points, have at least a 70% start probability and at least a 78% play probability. The exception is always named in the output.

Historical GW1 has no observed team sheets. Its causal play floor therefore begins at 68% and rises to 78% by GW5. This is a Bayesian cold-start rule, not a claim that old GW1 lineups were known with modern certainty.

### 3. Rate the teams

Team quality is split into attack and defence. The historical model updates those ratings only from completed earlier matches and shrinks small samples toward the league average. Promoted teams receive extra shrinkage because Championship dominance does not translate one-for-one into Premier League strength.

A Gameweek in which a club has no fixture is not a match. It produces no evidence and leaves the rating unchanged. Treating it as a 0-0 draw — which an earlier version did — made a postponed club look as if it had forgotten how to score and become defensively elite for several weeks afterwards, exactly during the fixture swings that decide chip and transfer timing.

The live model adds three external checks:

- Opta season probabilities for slow-moving team quality;
- published Opta match probabilities for the current fixture; and
- no-vig Matchbook exchange probabilities and expected goals for the latest market view.

Manager changes, major exits, promotion and European workload reduce confidence in carry-over ratings. They do not directly add or subtract fantasy points.

Opta and Matchbook can disagree. That disagreement is displayed. It is not resolved by pretending one source is always right: the exchange receives more current-fixture weight, while Opta supplies an independent model vote.

### 4. Price the current fixture exactly once

This was a major Lens 8 repair.

Opponent difficulty has a neutral opponent component. Venue is then applied once. Expected goals, expected goals conceded and clean-sheet probability feed the relevant scoring routes. The same absolute fixture score is not multiplied into the finished player projection a second time.

For the planning horizon, the model uses the ratio between the future slate and the current fixture. This changes the timing of value without paying for the current opponent twice.

The future slate is censored, but only where censoring is warranted. The Premier League publishes all 380 fixtures before a ball is kicked, so **who** a club faces in the next five Gameweeks is legitimately known at every deadline. What is not known is the rescheduling: blanks and doubles are announced later and the archive keeps no announcement dates. So future opponent difficulty is used, while each future Gameweek contributes exactly one fixture's worth of weight no matter how many fixtures the archive eventually recorded there. An earlier version censored the opponents as well, which left the "six-Gameweek fixture outlook" correlating 0.99 with the current fixture — a mean-reversion term wearing a planning label, with no forward information in it at all.

In a Double Gameweek, the number of fixtures is counted once — but it is counted for the whole forecast, not just part of it. Every scoring route and every rival model in the blend is priced for a single match; the match count is applied once, at the end. An earlier version scaled only the structural model, which left a Double Gameweek projected at roughly 1.2 times a single week instead of the 1.9 that actually occurs, and quietly suppressed every Triple Captain and Bench Boost decision that depended on it. A blank produces zero immediate fixture points.

### Why a large part of the pool can have no fixture

`No fixture` describes that Gameweek, not a missing player record. In an ordinary
10-match round every club is active. In a four-match Blank Gameweek only eight of
20 clubs are active, so roughly 60% of the registered player pool should have zero
fixtures. For example, the Premier League's official GW29 schedule in 2020/21 had
Fulham–Leeds, Brighton–Newcastle, West Ham–Arsenal and Aston Villa–Tottenham only.

The audit now checks the club set as well as the raw player percentage. A high
zero-fixture share is accepted only when the official schedule also contains fewer
active clubs. Eleven active starters are still mandatory. A persistent squad may
retain future-value blank players on its bench; a Free Hit instead optimises the
one-week XI and active autosub depth. Because Free Hit cash disappears after the
Gameweek, its solver does not force near-£100m expenditure in a tiny blank slate.

### 5. Build points from scoring routes

The structural forecast adds the expected value of:

- playing;
- goals;
- assists;
- clean sheets;
- goalkeeper saves;
- defensive contributions;
- bonus;
- cards, goals conceded, own goals and penalties.

The rates depend on position, expected minutes, team attack/defence, opponent vulnerability, set-piece role and player history. The live and historical forecasts build them the same way, which they did not always do: the live model used to add a set-piece and penalty uplift on top of a goal rate derived from Opta expected goals — and Opta already prices a penalty at about 0.79 xG, so an established taker's penalties were counted twice. That uplift now carries only the weight the player's own history cannot yet supply, which is exactly what makes it useful for a new signing or a taker who has just been handed the job, and worthless as a bonus for someone whose record already shows it. The live model also applied a second fixture-rank multiplier that had no historical counterpart, in direct contradiction of "price the current fixture exactly once"; that has been removed, and the fixture-specific team attack term the live model did have has been back-ported so both paths price the attacking route identically. Defenders therefore benefit from strong team defence, but only through clean-sheet and related bonus routes—not through a generic club badge bonus.

Every one of those rates is measured *conditional on the player appearing*, per appearance. A week the player missed is censored, not recorded as an appearance with no returns, and a Double Gameweek is divided by two rather than counted as one enormous match. This matters because the forecast then multiplies each rate by expected minutes: if absence were also inside the rate, availability would be charged twice and compound. Under the old definition a fringe player's goal rate came out at roughly 45% of his true per-90 figure before the minutes model had even been applied, which made returning and rotation-risk players close to invisible.

### 6. Ask rival models

The structural forecast is blended with:

- an empirical form/history estimate;
- a market-and-role baseline;
- an online ridge model fitted separately for player archetypes; and
- the official FPL next-game projection when it is available.

Weights are based on prior errors, and each rival is **level-corrected against its own prior-season bias before it is blended**. Weighting by error size alone only balances precision; a member that reads systematically high still drags the blend's level with it however small its weight, and the transfer hurdles and chip thresholds it feeds are all denominated in points. One member was reading more than double the truth and holding 11% of the weight, which accounted for almost all of the blend's overall bias. Removing that member outright was tested and lost within-Gameweek ranking power, so the diversity is kept and only the level is repaired.

Model disagreement increases uncertainty. A wide interval is information, not something to hide.

### 7. Use player-versus-opponent history carefully

Head-to-head history is a small tiebreaker. It is shrunk by sample size, team-rating confidence and regime change, and is capped at ±2.5%. A few old goals against the same badge cannot override a changed manager, role or squad.

### 8. Solve the squad exactly

The production squad, XI and captain are one binary mixed-integer optimisation problem. It enforces:

- £100.0m maximum spend and normally at least £99.5m used;
- exactly 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards;
- no more than 3 players from one club;
- a legal starting formation;
- captain in the XI;
- no more than £2.0m of price premium above positional minimums on the bench; and
- the XI availability rules above.

The Python production solver uses the full eligible pool and must report an optimal solution with zero gap. The interactive browser first retains the immediate, captain, horizon, value, minutes and cheap-enabler frontiers, then solves that declared candidate set exactly. It never falls back to the old greedy builder.

Bench players still matter for autosubs and Bench Boost, but ordinary bench points are heavily discounted. The optimiser is not rewarded for storing £6m–£8m players outside the XI.

### 9. Plan transfers and chips recursively

Each simulated season carries its squad, bank, free transfers and purchase prices into the next Gameweek. Decisions are made again from that new state.

If a real-world January transfer temporarily moves a held player into a club for
which the manager already owns three players, the squad may be held. Official FPL
requires the manager to return below the quota when making the next permanent
transfer. The replay mirrors that transition and does not invent a forced points hit.

The six-Gameweek transfer horizon remains because tested action-specific replacements lost points out of sample. It is no longer treated as a universal truth: captaincy is immediate, the bench is shorter-term, and long-lived squad construction receives a longer view in the separate frozen breakthrough research path.

Automatic Wildcards remain disabled in the audited policy because tested automatic variants lost points. Bench Boost and Triple Captain require a sufficiently large causal opportunity. Blank/Double Gameweek and AFCON signals can trigger review, but a plausible story is not enough to pass the replay gate.

Holding an unused chip is treated as an option. Playing it today forfeits every remaining week of its window, so the bar it must clear starts about 55% above the policy threshold and ramps down as the window closes, reaching a token positive-value check in the last legal week. A chip is therefore never simply lost to expiry, and it is not spent on the first week that happens to clear a static bar. That ramp depends on the remaining window length alone: the historical archive carries no announcement dates for postponements, so the future blank/double schedule cannot be consulted without leaking.

### Cached predictions are part of the model

A learned prediction cache is valid only for the exact ordered player-week frame,
feature schema and target definition that created it. Lens 8 fingerprints all
three. A matching row count is not sufficient. Any mismatch forces a retrain.
This rule was added after the old 2,212-point research result failed to reproduce
under the repaired schema.

## How historical testing works

The walk-forward replay starts before 2018/19 for model selection, then reports 2018/19 onward as evaluation seasons. The squad changes week by week using only the evidence available at that point. It scores legal formations, autosubs, captain/vice fallback, transfers, hits and chips under the rules that applied in that season.

Hundreds or thousands of weight settings can be screened cheaply, but only a smaller frozen set receives the expensive recursive replay. A result cannot promote itself merely because it looked good after seeing the evaluation seasons.

Top-500k cutoffs are estimates rather than complete official historical tables. The model publishes the uncertainty and does not translate out-of-range totals into a made-up precise rank.

The repaired replay averages **2,154.5 points**, against 2,087.0 for the
previous published model on the same eight evaluation seasons. Roughly a third of
that comes from the forecast repairs and the rest from the decision gate, which
now pools several candidate weightings before ranking strategies and walks its
choice forward season by season. Pooling halved the gate's own standard errors,
which is what let a real difference between policies become visible instead of
drowning in selection noise. The estimated top-500k shortfall roughly halved, from 210 points to 150
on average and from 443 to 307 at worst, but the target is still not reached in
any season. The former 2,212 V3 result stays retired because stale learned-model
caches made it unreproducible. The earlier Lens 8 comparison is in
[`PERFORMANCE_AUDIT_LENS8.md`](PERFORMANCE_AUDIT_LENS8.md); the repairs behind
this number are in [`MODEL_LOGIC_AUDIT.md`](MODEL_LOGIC_AUDIT.md).

## Reading the website

- **GW xPts** is the expected score for the immediate Gameweek.
- **Six-GW xPts** is a discounted planning total, not six identical repetitions of GW1.
- **Start/play/60+** separates three different kinds of minutes risk.
- **P10/median/P90** shows the forecast range.
- **Team context** shows intrinsic strength, fixture expected goals and rating confidence separately.
- **Market disagreement** shows where the internal team view conflicts with Opta or Matchbook.
- **Captain rating** is a 0-100 readability score, not the decision. The armband goes to the highest expected Gameweek score, because captaincy is worth exactly one extra copy of that score. An earlier version chose it from a blend of four percentile ranks, which threw away the size of the gap between the best option and the next and gave ownership a vote in a decision ownership has no bearing on; replayed from 2018/19 it disagreed with the expected-points choice in 87% of weeks.
- **Elite disagreement** compares the model with published expert squads but never forces a consensus player into the team.
- **Backtest points** are recursive simulated points, not the sum of perfect hindsight weekly teams.

## What the model deliberately does not claim

- It does not have proprietary Opta event data; it uses public Opta outputs with source dates.
- It does not insert today’s betting odds into old seasons without timestamped historical snapshots.
- It does not treat ownership as proof that a player is good.
- It does not treat age as a direct source of points; age is a small availability and consistency prior.
- It does not guarantee that £100m must always be spent when a documented strategic-bank mode is active, although the initial live squad normally uses at least £99.5m.
- It does not claim consistent top-500k performance until the frozen promotion rule is actually cleared.

## Reproducing the model

```powershell
python analysis/calibrate_model.py
python -m unittest discover -s analysis -p "test_*.py"
pnpm test
pnpm lint
```

For a fast live-only refresh after a completed full calibration:

```powershell
pnpm research:refresh
```

The generated website inputs are `app/data/model-results.json`,
`app/data/current-players.json` and `app/data/model-audit.json`. Historical caches
under `work/` are local and are rebuilt when their version or frame fingerprint
changes.

## Public methodology sources

- [Premier League: reduced GW29 schedule in 2020/21](https://www.premierleague.com/en/news/2068574)
- [Premier League: blank and double Gameweek planning caused by the FA Cup](https://www.premierleague.com/news/2035867)
- [Premier League: temporary four-player club squads after real-world transfers](https://www.premierleague.com/en/news/4536241/what-happens-when-you-have-four-players-from-one-club-in-your-fpl-squad)
- [Opta Analyst match predictions](https://theanalyst.com/articles/premier-league-match-predictions)
- [Matchbook football exchange](https://www.matchbook.com/events/soccer)

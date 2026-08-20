# Lens 8 explanation-driven decision audit

## Audit rule

Every production choice must answer four questions:

1. What pre-deadline input does it use?
2. How does that input change a forecast or decision?
3. What failure is the rule intended to prevent?
4. What walk-forward or prospective evidence allows it into production?

If a rule cannot answer all four, it is removed, constrained to a diagnostic, or left in research shadow.

## Decisions repaired in Lens 8

| Decision | Audit finding | Lens 8 action |
|---|---|---|
| FPL fixture-strength field | The live code treated an official 2–5 scale as if it were a roughly 1,400-point rating. This made the transformation meaningless. | Removed. Fixtures now come from opponent-adjusted goal rates and clean-sheet probabilities. |
| Historical venue schema | A lookup could inherit a home bonus from an arbitrary prior row and add a future home bonus again. | Split neutral opponent difficulty from venue; apply venue once. |
| Finished-projection fixture multiplier | The component routes already knew the opponent, then an absolute fixture score could multiply them again. | Removed. The horizon uses a relative adjustment only. |
| “Team strength” label | The old context blended immediate opponent difficulty with club quality, making strong clubs appear weak in hard fixtures and vice versa. | Team context is now intrinsic attack, defence and form. Match difficulty is separate. |
| Carry-over after upheaval | New managers, major exits and promoted teams inherited too much confidence. | Added explicit regime priors, league-mean shrinkage and confidence caps. |
| Greedy live squad construction | Local swaps could miss the global combination, leave money unused and buy an expensive bench. | Replaced with exact binary MILP and explicit solver status/gap. |
| Bench value | Bench projections were valued too similarly to XI points. | Discounted ordinary bench value and capped bench price premium at £2m. |
| XI availability | A high projection could enter the XI despite a low chance of playing. | Added hard start/play floors and one disclosed exceptional-upside slot. |
| H2H missing-data fallback | New signings could receive NaN because opponent history was filled before general history. | Reordered the fallback; missing H2H now means zero H2H adjustment. |
| Hidden model/expert conflict | A plausible squad could conceal that it rejected several elite-manager core players. | Added diagnostic-only elite-consensus agreement and model-only XI lists. |
| Hidden model/market conflict | Internal team ratings could overrate a side without displaying the contradiction. | Added Opta, Matchbook and combined external probabilities plus ranked disagreement. |
| Large `no fixture` share | The raw percentage looked like a broken join, but both investigated cases were genuine four-match FA Cup blank rounds with only eight active clubs. | Validate active club IDs against the official schedule, not a misleading player-row percentage. Keep blank players out of the XI. |
| Blank-week Free Hit spend | The one-week rebuild inherited the persistent squad's near-full-spend rule, so the exact solver could become infeasible when only eight clubs played. | Free Hit has no minimum-spend constraint; it requires 11 active starters and values active autosub depth. Persistent Wildcard/spend rules are unchanged. |
| Objective-dependent availability | The exceptional-upside eligibility set was accidentally recomputed from each risk objective, meaning constraints changed when only the objective should change. | Freeze exception eligibility from causal immediate component xPts before comparing objectives. |
| January transfer club overload | A real-world move can temporarily give an FPL squad four players from one club. The replay treated the held state as illegal, even though official FPL rules permit it until the manager's next transfer. | Permit the inherited state when holding or Free Hitting; the next permanent transfer must restore the three-player quota, and no points hit is forced merely to repair it. |
| Learned prediction caches | Old challenger caches checked row count only. The Lens 8 frame had the same length but a different schema, so stale forecasts could be silently reused. | Version and fingerprint the ordered frame, features and target; retrain on any mismatch. Retire the unreproducible 2,212 result. |

## Legacy decisions retained—but narrowed

| Decision | Why it remains | Constraint that prevents overreach |
|---|---|---|
| Six-GW transfer planning | Tested action-specific transfer horizons lost points in the frozen replay. | It is not reused as the captain horizon, and future fixtures adjust relative to the current fixture. |
| Ownership signal | It can contain price, role and collective-news information. | Small ensemble weight; it cannot override minutes, price and forecast routes alone. |
| Age curve | It can weakly inform reliability where history is sparse. | Low prior weight; never described as direct footballing quality. |
| Player-v-opponent history | It can break genuine stylistic ties. | Sample/regime shrinkage and ±2.5% cap. |
| Near-full initial spend | Premium starters usually dominate idle bank at GW1. | Minimum is £99.5m rather than a forced £100m equality; documented strategic-bank mode remains possible later. |
| One upside minutes exception | Some elite captains remain worthwhile despite measured rotation risk. | Top 5% immediate projection, ≥70% start, ≥78% play, at most one, always disclosed. |

## Decisions kept out of production

| Idea | Reason |
|---|---|
| Elite consensus as an optimiser reward | Circular and vulnerable to groupthink. It remains a disagreement diagnostic. |
| Automatic Wildcard storytelling | AFCON, injuries or bad structure can justify review, but tested automatic Wildcard variants lost out of sample. |
| Closing odds in historical forecasts | They would leak post-deadline information unless a timestamped pre-deadline archive exists. |
| Unrestricted H2H boost | Old matches often belong to different managers, roles and squads. |
| Exact historical 84% play floor from GW1 | No team-sheet evidence exists at a cold start; it makes the entire player universe infeasible. The historical floor ramps causally while the live floor stays strict. |
| Greedy fallback after solver failure | It hides an infeasible design behind a plausible team. Lens 8 raises an error and stops publication. |

## Back-checks prompted by writing the handbook

1. **Could “fixture” be explained without counting it twice?** No. This exposed and removed the neutral/venue lookup duplication and the second absolute multiplier.
2. **Could “team strength” be explained independently of the next opponent?** Previously no. It is now an intrinsic rating with a separate match layer.
3. **Could every live starter’s availability be justified?** Previously no hard rule existed. The floor and named exception are now solver constraints.
4. **Could a missing H2H record be explained?** The intended answer was “no adjustment,” but the implementation produced NaN for 157 current players. The fallback was repaired before the replay.
5. **Could the browser claim exact optimisation over 391 players inside an interactive render?** The pure-JavaScript solver timed out. The UI now declares a multi-frontier presolve candidate universe and solves that universe exactly; the Python artifact still solves the full eligible pool.
6. **Could the Opta claim be verified for every live fixture?** A season top-five table was insufficient. Published Matchday 1 Opta probabilities were added for all 10 fixtures and dated separately from Matchbook.
7. **Could 60% of players really have no fixture?** Yes in a four-match blank: only eight of 20 clubs are active. The real bug was applying normal-squad spend semantics to the resulting Free Hit, not the fixture join itself.
8. **Could feasibility legitimately change when only the risk objective changed?** No. That exposed an objective-dependent exceptional-player mask; the mask is now fixed before optimisation.
9. **Can a legal held squad contain four players from one club?** Temporarily, yes, when a real-world Premier League transfer creates it. The prior unconditional assertion was stricter than FPL itself and has been replaced by a state-transition rule.
10. **Did the old 2,212-point result survive the repaired engine?** No. Its fair replay was 2,122.1, and fresh causal retraining produced 2,119.2. The old headline is retired rather than used to make Lens 8 look worse.

## Acceptance checklist

Lens 8 is accepted only if all of the following are true:

- Python and JavaScript tests pass;
- all historical deadline MILPs are feasible without greedy fallback;
- live budget is between £99.5m and £100m;
- live bench premium is at most £2m;
- XI availability exceptions are no more than one and are named;
- Opta and Matchbook source dates and coverage are present;
- the eight-season walk-forward run completes;
- the new totals are compared with the prior frozen result rather than reported in isolation; and
- the website builds from the generated artifact.

Passing engineering checks does not automatically mean the historical policy improved. Lens 8 does improve on Lens 7 by 38.9 points per season, but its 2,087.0 average remains below the 2,119.2 exposed causal shadow and clears zero of eight estimated top-500k pace lines. The shadow is not promoted until a locked test and live final-fit are complete. See [`PERFORMANCE_AUDIT_LENS8.md`](PERFORMANCE_AUDIT_LENS8.md).

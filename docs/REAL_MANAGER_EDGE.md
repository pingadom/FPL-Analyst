# What copying real managers could add

*FPL Lens research note, 27 September 2026*

The question: which decisions do strong real managers make that this model cannot see, and
how many points would replicating them be worth?

The answer depends on how "strong managers" are chosen. Choose them by how their season
*finished* and the gap looks enormous: 203 points in 2025/26 against the top-10k median.
But that group was selected on its outcome, so the gap is mostly luck that has already
happened. The fair comparison is with managers who were proven *before* the season, whose
decisions could actually have been copied at each deadline. On that basis the model is
already close, and the prize is concentrated in a few specific decisions.

## Data

* **project-fpl2526.** Squads, captains and chips for about 100,000 real managers across
  GW1–6 of 2025/26, each labelled with their best rank in earlier seasons. The 897
  managers with a previous best inside the top 10k form the **ex-ante elite**: a group you
  could have identified in August.
* The same source's rank history (2019/20–2024/25), used to test whether elite
  performance persists.
* The model's pinned replay of 2025/26, scored with the same rules: XI points plus captain
  extra, before autosubs and hits.

## Does elite skill persist?

Managers ranked by their 2023/24 finish, and where they finished in 2024/25:

| 2023/24 finish | managers | median 2024/25 rank | next top 10k | next top 100k | next top 1m |
|---|---|---|---|---|---|
| top 10k | 90 | 411,462 | 6.7% | 25.6% | 80.0% |
| 10k–100k | 818 | 323,886 | 2.8% | 20.9% | 78.6% |
| 100k–1m | 7,661 | 1,214,396 | 0.4% | 5.8% | 43.7% |
| 1m+ | 42,569 | 3,851,687 | 0.0% | 0.6% | 10.8% |

Skill is real. A previous top-10k manager is about 40 times likelier than a 1m+ manager to
finish top 100k. But it regresses hard: the median proven manager finishes around 400k the
next season. On the 2025/26 points distribution, a 400k finish is roughly 2,255 points,
against this model's 2,219 in its 2025/26 walk-forward. **The repeatable edge of a proven
manager over this model is on the order of 35–40 points a season, not 200.**

## GW1–6 2025/26, decision by decision

Mean per manager per Gameweek:

| group | score | XI points | captain extra | non-playing starters |
|---|---|---|---|---|
| **model** | **57.5** | **52.2** | 5.3 | 0.17 |
| ex-ante elite (prev top 10k) | 54.3 | 45.9 | **8.4** | **0.05** |
| prev 10k–100k | 53.5 | 45.7 | 7.8 | 0.07 |
| prev 100k–1m | 52.1 | 45.1 | 7.0 | 0.13 |
| prev 1m+ | 50.2 | 44.0 | 6.2 | 0.28 |
| no history | 49.0 | 43.2 | 5.8 | 0.42 |

Over these six weeks the model out-scored the ex-ante elite by 3.2 points a week. It picks a
better XI and captains worse. Squad composition was a wash: the model's 15 scored within 1.1
points, over the six weeks, of the elite's ownership-weighted 15.

### Captaincy: the one decision the elite clearly won

| GW | model captain | pts | elite consensus | share | pts | model owned him |
|---|---|---|---|---|---|---|
| 1 | Salah | 8 | Salah | 69% | 8 | yes |
| 2 | Salah | 5 | Salah | 44% | 5 | yes |
| 3 | Salah | 3 | Haaland | 20% | 9 | no |
| 4 | Salah | 9 | Salah | 64% | 9 | yes |
| 5 | Salah | 5 | Salah | 67% | 5 | yes |
| 6 | Salah | 2 | **Haaland** | **82%** | **16** | no |

The model agrees with the elite consensus whenever it owns the consensus player. The whole
difference is Haaland, whom the elite moved to as he started the season in form and the
model rated as a near-tie with Salah (see TOP500K_PLAN section 8). Over the full season the
captaincy gap against the hindsight top 10k is about 52 points (armband 251 against 303);
against proven managers it is smaller, but it is the largest single item.

## What the model cannot see, and what each is worth

| decision | what the elite have | evidence | plausible value | can the backtest test it? |
|---|---|---|---|---|
| **Consensus switch on a premium** | a crowd of proven managers moving together (82% on Haaland by GW6) | GW3 +6, GW6 +14 in six weeks | 10–30 a season | no: elite ownership is not in the archive |
| **Late team news** | deciding hours before the deadline, after press conferences | 0.05 non-playing starters a week against 0.17 | about 10–15 a season | partly: pre-deadline flags are already used |
| **Early-season transfers** | reading new roles, set pieces and pre-season | their GW2–6 moves realise 8.4 over four weeks against the model's 4.3 prediction for the same moves | real but hard to separate from luck | no |
| **Wildcard timing** | a coordinated GW6 rebuild | forcing the model's own Wildcard to GW6 gains +1.5 | ~0 with the model's squad | yes, and it does not help |
| **Hits** | 4 points a season for the top 10k | paid hits lose at every price in replay | ~0 | yes, and they do not help |
| **Chips** | all eight used | model leaves some unused; threshold changes are flat | small | yes, flat |

The pattern is consistent with every earlier test in this project. Where the elite's edge
comes from **timing or policy** (Wildcards, hits, banking, chips), copying the behaviour
without their information does not help, and the backtest confirms it. Where it comes from
**information** (who is in form, who is starting, where the crowd of proven managers is
moving), it is real, but the historical archive does not contain it, so it can only be
captured and measured live.

## Recommendation

1. **Add an elite-ownership signal to the live path, as a warning rather than a forecast
   input.** Each week, compare the model's squad and captain with a sample of proven managers
   (for example the elite sample on plan.livefpl.net, or FPL Review's Elite 1000). When at
   least 75% of them own or captain a player the model rates within a hurdle of its own
   pick, flag it in the recommendation. That would have raised Haaland at GW6. It cannot be
   backtested, so it should be measured through the pick log before it is allowed to change
   picks.
2. **Keep the decision layer as it is.** Every elite *behaviour* tested so far (anchoring,
   banking, bench, early Wildcards, hits, chip thresholds) has failed out of sample, and
   `harness.py walkforward` confirms the shipped settings.
3. **Judge it on the live log.** Scored weeks so far: GW2 82 (average 81), GW3 44 (average
   51). Proven managers' edge of roughly 35–40 points a season is about one point a week, so
   it takes most of a season of live weeks to see it.

## Caveats

Six Gameweeks of one season. GW1–6 scores exclude autosubs and hits for both sides (the
elite paid few hits). The persistence table covers only managers present in the 2025/26
sample. The 400k-to-points conversion uses the 2025/26 distribution for a 2024/25 rank. The
model figures come from the pinned replay; the walk-forward finished 2025/26 on 2,219.

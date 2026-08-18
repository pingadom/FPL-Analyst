import { readFile, writeFile } from "node:fs/promises";

import { buildOptimizedSquad, evaluateSquad } from "../app/lib/squad-optimizer.mjs";

const players = JSON.parse(
  await readFile(new URL("../app/data/current-players.json", import.meta.url), "utf8"),
);
const results = JSON.parse(
  await readFile(new URL("../app/data/model-results.json", import.meta.url), "utf8"),
);

function liveScore(player) {
  const weights = results.model.weights;
  const recent = weights.recent / 100;
  const performance =
    player.features.recent * recent + player.features.history * (1 - recent);
  const value =
    player.features.recentValue * recent + player.features.historyValue * (1 - recent);
  const lens =
    performance * (weights.performance / 100) +
    value * (weights.value / 100) +
    player.features.age * (weights.age / 100) +
    player.features.fixture * (weights.fixture / 100) +
    player.features.team * (weights.team / 100) +
    player.features.crowd * (weights.crowd / 100) +
    player.features.minutes * (weights.minutes / 100) +
    player.features.underlying * (weights.underlying / 100);
  return (
    0.58 * lens +
    0.14 * (player.comparison.projectionRank / 100) +
    0.10 * (player.confidence / 100) +
    0.18 * player.strategyScores.balanced
  );
}

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normal(next) {
  const first = Math.max(next(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * next());
}

const base = players.map((player) => ({ ...player, liveScore: liveScore(player) }));
const baseline = buildOptimizedSquad(base);
const baselineIds = new Set(baseline.squad.map((player) => player.id));
const trials = [];

for (let seed = 1; seed <= 40; seed += 1) {
  const next = random(260813 + seed);
  const noisy = base.map((player) => {
    const shock = normal(next);
    return {
      ...player,
      projected: Math.max(0, Number(player.projected) + 0.04 * shock),
      sixWeekProjected: Math.max(0, Number(player.sixWeekProjected) + 0.18 * shock),
      liveScore: Number(player.liveScore) + 0.004 * shock,
    };
  });
  const production = buildOptimizedSquad(noisy);
  const relaxed = buildOptimizedSquad(noisy, {
    allowStrategicBank: true,
    benchPremiumLimit: 100,
  });
  const relaxedUnderProduction = evaluateSquad(relaxed.squad, noisy);
  const displayedCaptain = [...production.xi].sort(
    (a, b) => b.captainRating - a.captainRating,
  )[0];
  trials.push({
    overlap: production.squad.filter((player) => baselineIds.has(player.id)).length,
    captainMatches: production.captain.id === displayedCaptain.id,
    relaxedFeasible: Boolean(relaxedUnderProduction),
    searchGap: relaxedUnderProduction
      ? relaxedUnderProduction.score - production.score
      : null,
  });
}

const feasible = trials.filter((trial) => trial.relaxedFeasible);
const positiveGaps = feasible.filter((trial) => trial.searchGap > 1e-8);
const audit = {
  perturbation:
    "40 deterministic draws; shared Gaussian shock of 0.04 next-GW points, 0.18 six-week points and 0.004 live-score units per player",
  meanSquadOverlapOutOf15:
    trials.reduce((sum, trial) => sum + trial.overlap, 0) / trials.length,
  minimumSquadOverlapOutOf15: Math.min(...trials.map((trial) => trial.overlap)),
  captainDisplayMatchRate:
    trials.filter((trial) => trial.captainMatches).length / trials.length,
  relaxedCandidateFeasibleTrials: feasible.length,
  productionSearchBeatenTrials: positiveGaps.length,
  meanGapWhenBeaten:
    positiveGaps.reduce((sum, trial) => sum + trial.searchGap, 0) /
    Math.max(1, positiveGaps.length),
  maximumSearchGap: Math.max(0, ...positiveGaps.map((trial) => trial.searchGap)),
};
await writeFile(
  new URL("./data/current_optimizer_stability.json", import.meta.url),
  `${JSON.stringify(audit, null, 2)}\n`,
);
console.log(JSON.stringify(audit, null, 2));

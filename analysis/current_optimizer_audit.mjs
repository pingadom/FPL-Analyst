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

const scored = players.map((player) => ({ ...player, liveScore: liveScore(player) }));
function summarize(selection) {
  if (!selection) throw new Error("Production optimiser did not return a legal squad");
  const displayedCaptain = [...selection.xi].sort(
    (a, b) => b.captainRating - a.captainRating,
  )[0];
  return {
    score: selection.score,
    spend: selection.spend,
    benchSpend: selection.benchSpend,
    benchPremium: selection.benchPremium,
    optimizerCaptain: selection.captain.name,
    displayedCaptain: displayedCaptain.name,
    captainMatches: selection.captain.id === displayedCaptain.id,
    xi: selection.xi.map((player) => player.name),
    bench: selection.bench.map((player) => ({ name: player.name, price: player.price })),
  };
}

const productionSelection = buildOptimizedSquad(scored);
const relaxedSelection = buildOptimizedSquad(scored, {
    allowStrategicBank: true,
    benchPremiumLimit: 100,
  });
const production = summarize(productionSelection);
const relaxed = summarize(relaxedSelection);
const relaxedReevaluatedUnderProductionRules = summarize(
  evaluateSquad(relaxedSelection.squad, scored),
);
const moderateBench = summarize(buildOptimizedSquad(scored, { benchPremiumLimit: 4 }));

const audit = { production, relaxed, relaxedReevaluatedUnderProductionRules, moderateBench };
await writeFile(
  new URL("./data/current_optimizer_audit.json", import.meta.url),
  `${JSON.stringify(audit, null, 2)}\n`,
);
console.log(JSON.stringify(audit, null, 2));

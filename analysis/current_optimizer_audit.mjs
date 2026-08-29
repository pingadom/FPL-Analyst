import { readFile, writeFile } from "node:fs/promises";

import { buildOptimizedSquad, evaluateSquad } from "../app/lib/squad-optimizer.mjs";

const players = JSON.parse(
  await readFile(new URL("../app/data/current-players.json", import.meta.url), "utf8"),
);
function liveScore(player) {
  // The artifact keeps full-precision fitted coefficients. Reconstructing this
  // value from the rounded UI percentages makes the audit test a different model.
  return player.strategyScores.balanced;
}

const scored = players.map((player) => ({
  ...player,
  liveScore: liveScore(player),
  optimizerUtility: player.optimization.lineupUtility.balanced,
}));
function summarize(selection) {
  if (!selection) throw new Error("Production optimiser did not return a legal squad");
  const displayedCaptain = [...selection.xi].sort(
    (a, b) => b.projected - a.projected,
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

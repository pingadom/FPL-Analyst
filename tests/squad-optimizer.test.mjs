import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BENCH_PREMIUM_LIMIT,
  buildOptimizedSquad,
} from "../app/lib/squad-optimizer.mjs";

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

test("joint optimiser spends the initial budget on the XI and captain", () => {
  const scored = players.map((player) => ({ ...player, liveScore: liveScore(player) }));
  const selection = buildOptimizedSquad(scored);
  assert.ok(selection, "expected a legal squad");
  assert.equal(selection.squad.length, 15);
  assert.equal(selection.xi.length, 11);
  assert.ok(selection.spend >= 99.5 && selection.spend <= 100);
  assert.ok(selection.benchPremium <= BENCH_PREMIUM_LIMIT + 1e-6);
  assert.equal(selection.solver.type, "exact-binary-milp");
  assert.equal(selection.solver.status, "optimal");
  assert.equal(selection.solver.optimalityGap, 0);
  assert.ok(selection.solver.candidatePlayers < selection.solver.inputPlayers);
  assert.ok(selection.xi.some((player) => player.id === selection.captain.id));
  const positionCounts = Object.fromEntries(
    ["GK", "DEF", "MID", "FWD"].map((position) => [
      position,
      selection.squad.filter((player) => player.position === position).length,
    ]),
  );
  assert.deepEqual(positionCounts, { GK: 2, DEF: 5, MID: 5, FWD: 3 });
  const clubCounts = new Map();
  for (const player of selection.squad) {
    clubCounts.set(player.team, (clubCounts.get(player.team) ?? 0) + 1);
  }
  assert.ok([...clubCounts.values()].every((count) => count <= 3));
  const exceptional = selection.xi.filter(
    (player) =>
      player.minutesModel.startProbability < 70 ||
      player.minutesModel.playProbability < 84,
  );
  assert.ok(exceptional.length <= 1, "only one exceptional-upside minutes exception is legal");
  assert.ok(
    exceptional.every(
      (player) =>
        player.minutesModel.startProbability >= 70 &&
        player.minutesModel.playProbability >= 78,
    ),
  );
  assert.ok(
    selection.squad.some((player) => player.name === "Haaland"),
    "the highest immediate projection and captain option should not be displaced by premium substitutes",
  );
});

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

test("joint optimiser spends the initial budget on the XI and captain", () => {
  const scored = players.map((player) => ({
    ...player,
    liveScore: player.strategyScores.balanced,
    optimizerUtility: player.optimization.lineupUtility.balanced,
  }));
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
  assert.equal(
    selection.captain.projected,
    Math.max(...selection.xi.map((player) => player.projected)),
    "captaincy must go to the selected XI player with the highest projected points",
  );
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
    (player) => !player.optimization.standardStarterEligible,
  );
  assert.ok(exceptional.length <= 1, "only one exceptional-upside minutes exception is legal");
  assert.ok(
    exceptional.every((player) => player.optimization.exceptionalStarterEligible),
  );
  const haaland = scored.find((player) => player.name === "Haaland");
  assert.ok(haaland, "Haaland must survive the candidate eligibility screen");
  assert.ok(
    haaland.minutesModel.startProbability >= 70 &&
      haaland.minutesModel.playProbability >= 84,
    "a current-season starter must not inherit an absence streak from the prior season",
  );
  assert.ok(
    haaland.projected >= Math.max(...scored.map((player) => player.projected)) - 0.2,
    "an available premium talisman should remain near the top of the immediate forecast",
  );
  assert.deepEqual(
    selection.squad.map((player) => player.id).toSorted((a, b) => a - b),
    results.squad.map((player) => player.id).toSorted((a, b) => a - b),
    "the untouched calibrated preset must reproduce the generated Python squad",
  );
  assert.equal(selection.captain.id, results.squad.find((player) => player.captain).id);
});

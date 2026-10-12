import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const modelResults = JSON.parse(
  await readFile(new URL("../app/data/model-results.json", import.meta.url), "utf8"),
);
const publicModelAudit = JSON.parse(
  await readFile(new URL("../app/data/model-audit.json", import.meta.url), "utf8"),
);
const currentPlayers = JSON.parse(
  await readFile(new URL("../app/data/current-players.json", import.meta.url), "utf8"),
);
const currentBacktestAverage =
  Math.round(
    (modelResults.backtest.reduce((sum, season) => sum + season.points, 0) /
      modelResults.backtest.length) *
      10,
  ) / 10;

async function request(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the FPL Lens tool", async () => {
  const response = await request();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>FPL Lens<\/title>/i);
  // Every section is reachable from the tab bar.
  for (const tab of ["This week", "Players", "Captain", "Fixtures", "My team", "Wildcard planner", "Track record"]) {
    assert.match(html, new RegExp(`>${tab}<`), `missing tab ${tab}`);
  }
  // The default view is the model's own team for the coming deadline.
  assert.match(html, new RegExp(`GW${modelResults.headline.gameweek}(?:<!-- -->)? projection`));
  assert.match(html, /Transfers this week/);
  assert.match(html, /Chips/);
  for (const move of modelResults.headline.transfers ?? []) {
    assert.ok(html.includes(move.out) && html.includes(move.in), `transfer ${move.out} -> ${move.in} not shown`);
  }
  const captain = modelResults.squad.find((player) => player.captain);
  assert.ok(captain && html.includes(captain.name), "captain missing from the pitch");
  // The retired research copy is gone from the page.
  assert.doesNotMatch(html, /Build a squad|Tune the lens|causal shadow|Lens 8 model audit/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("serves the frozen prospective research audit", async () => {
  const response = await request("/api/research");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), "*");
  const payload = await response.json();
  assert.match(payload.deadline.snapshotHash, /^[a-f0-9]{64}$/);
  assert.equal(payload.deadline.status, "provisional");
  assert.equal(payload.shadows.managers.length, 6);
  assert.equal(payload.chips.simulationCount, 5000);
  assert.equal(payload.frontier.status, "shadow challenger");
  assert.equal(payload.listwise.status, "shadow challenger");
  assert.match(payload.performance.status, /retired legacy artifact/i);
  assert.match(payload.breakthrough.status, /retired legacy artifact/i);
  assert.deepEqual(payload.modelAudit, publicModelAudit);
  // model-audit.json is a frozen record of the Lens 7/8 research comparison; the
  // published backtest has moved on since, so it is no longer tied to it.
  assert.ok(Number.isFinite(currentBacktestAverage));
  assert.equal(
    payload.modelAudit.causalChallenger.deltaVsLens8,
    Math.round(
      (payload.modelAudit.causalChallenger.average - payload.modelAudit.lens8.average) * 10,
    ) / 10,
  );
  assert.match(payload.modelAudit.legacyBreakthrough.status, /retired/i);
  assert.equal(payload.breakthrough.seasons.length, 8);
  assert.equal(modelResults.frozenAudit.valid, true);
  assert.match(modelResults.frozenAudit.contentFingerprint, /^[a-f0-9]{64}$/);
  const optaSignals = modelResults.currentMeta.externalTeamSignals;
  const [optaCovered, optaTotal] = optaSignals.optaFixtureCoverage
    .split("/")
    .map(Number);
  assert.equal(optaTotal, modelResults.currentMeta.fixturesScored);
  if (optaSignals.optaFixtureSnapshotMatchesDeadline) {
    assert.ok(optaCovered > 0 && optaCovered <= optaTotal);
  } else {
    assert.equal(optaCovered, 0, "a stale Opta snapshot must contribute zero fixtures");
  }
  assert.equal(currentPlayers.length, modelResults.currentMeta.playersScored);
  assert.equal(
    modelResults.currentMeta.playersScored +
      modelResults.currentMeta.playersExcludedByAvailability,
    modelResults.currentMeta.officialPlayersRegistered,
  );
  assert.match(modelResults.currentMeta.playerEligibility, /no minutes, ownership or public-xP/i);
  assert.ok(
    currentPlayers.every(
      (player) =>
        player.opponent && player.researchFeatures.fixture_count > 0,
    ),
    "every published optimisation candidate must have a current fixture",
  );
  assert.ok(
    currentPlayers.every(
      (player) =>
        player.horizonWeightedGames > 0 &&
        Number.isFinite(player.riskAdjustedHorizonProjected) &&
        Number.isFinite(player.optimization.lineupUtility.balanced) &&
        Number.isFinite(player.optimization.benchUtility) &&
        Number.isFinite(player.optimization.captainUtility) &&
        typeof player.optimization.standardStarterEligible === "boolean" &&
        typeof player.optimization.exceptionalStarterEligible === "boolean",
    ),
    "every candidate must publish the exact Python optimiser inputs",
  );
  assert.equal(
    payload.chips.managerPlans["forecast-breakthrough-v2"].policyProfile,
    "forecast-v2 756-policy recursive winner",
  );
  for (const season of modelResults.backtest) {
    assert.equal(
      new Set(season.chips.map((chip) => chip.gw)).size,
      season.chips.length,
      `${season.season} must use at most one chip in a Gameweek`,
    );
    const uses = Object.groupBy(season.chips, (chip) => chip.chip);
    assert.ok((uses.Wildcard?.length ?? 0) <= 2);
    for (const chip of ["Free Hit", "Bench Boost", "Triple Captain"]) {
      const limit = season.season === "2025/26" ? 2 : 1;
      assert.ok(
        (uses[chip]?.length ?? 0) <= limit,
        `${season.season} exceeds its ${chip} allowance`,
      );
    }
    assert.ok((uses["Assistant Manager"]?.length ?? 0) <= 1);
    assert.equal(
      (uses["Assistant Manager"]?.length ?? 0) > 0,
      season.season === "2024/25" && (uses["Assistant Manager"]?.length ?? 0) > 0,
      "Assistant Manager is legal only in 2024/25",
    );
  }
});

test("serves public Lens 8 projections with CORS", async () => {
  const response = await request("/api/projections?position=DEF&limit=2");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), "*");
  const payload = await response.json();
  assert.equal(payload.model, "Lens 8.0");
  assert.equal(payload.count, 2);
  assert.ok(payload.players.every((player) => player.position === "DEF"));
  assert.ok(payload.players.every((player) => typeof player.ensemble.roleChallenger === "number"));
});

test("serves a fixture grid, or a well-formed empty one when FPL is unreachable", async () => {
  const response = await request("/api/fixtures");
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.ok(["official-fpl", "unavailable"].includes(payload.source));
  assert.ok(Array.isArray(payload.events) && Array.isArray(payload.teams));
  assert.ok(Object.keys(payload.modelRatings).length >= 18, "model team ratings are always present");
  if (payload.source === "official-fpl") {
    assert.equal(payload.teams.length, 20);
    assert.ok(payload.events.length >= 1 && payload.events.length <= 6);
  }
});

test("rejects malformed FPL team IDs before calling the official API", async () => {
  const response = await request("/api/team/not-a-team");
  assert.equal(response.status, 400);
  const payload = await response.json();
  assert.match(payload.error, /valid numeric FPL team ID/i);
});

test("removes the disposable starter preview", async () => {
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});

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

test("server-renders the FPL Lens decision room", async () => {
  const response = await request();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>FPL Lens — Lens 8 model audit<\/title>/i);
  assert.match(html, /Build a squad/);
  assert.match(html, /2,400(?:<!-- -->)? candidate mixes/i);
  assert.match(
    html,
    new RegExp(`${modelResults.model.recursiveTrials}(?:<!-- -->)? recursive finalists`, "i"),
  );
  assert.match(html, /Tune the lens/);
  assert.match(html, /Optimal XV/i);
  assert.match(html, /Chip desk/i);
  assert.match(html, /48(?:<!-- -->)? chip policies/i);
  assert.match(html, /Lens 8\.0[\s\S]{0,24}\+ chips/i);
  assert.match(html, /AUDITED PROMOTION GATE/i);
  assert.match(html, /Frozen pre-2018 audit/i);
  assert.match(html, /RANK OUTSIDE LOCAL CALIBRATION/i);
  assert.match(html, /Average rank withheld/i);
  assert.match(html, /Top-500k consistency test/i);
  assert.match(html, /Player Lab/i);
  assert.match(html, /Projection anatomy/i);
  assert.match(html, /Point distribution/i);
  assert.match(html, /Minutes tree/i);
  assert.match(html, /Personalised decision room/i);
  assert.match(html, /Open projections/i);
  assert.match(html, /Role challenger/i);
  assert.match(html, /Same-fixture test/i);
  assert.match(html, /Performance evidence/i);
  assert.match(html, /Team context/i);
  assert.match(html, /Poisson probability/i);
  assert.match(html, /Replay the past/i);
  assert.match(html, /Probability calibration/i);
  assert.match(html, /Current-rules counterfactual/i);
  assert.match(html, /Champion advice, tested/i);
  assert.match(html, /correlated squad scenarios/i);
  assert.match(html, /Frozen research season/i);
  assert.match(html, /6(?:<!-- -->)? pre-registered managers/i);
  assert.match(html, /Chip scenario gates/i);
  assert.match(html, /Retrained causal challenger/i);
  assert.match(html, /Performance ladder/i);
  assert.match(html, /Experiment ledger/i);
  assert.match(html, /The repaired model improves/i);
  assert.match(html, /Model governance/i);
  assert.match(html, /Legacy replay/i);
  assert.match(html, /not reproducible/i);
  assert.match(html, /Automatic Wildcard/i);
  assert.match(html, /Hybrid challenger/i);
  assert.match(html, /provisional/i);
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
  assert.equal(payload.modelAudit.lens8.average, currentBacktestAverage);
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

test("rejects malformed FPL team IDs before calling the official API", async () => {
  const response = await request("/api/team/not-a-team");
  assert.equal(response.status, 400);
  const payload = await response.json();
  assert.match(payload.error, /valid numeric FPL team ID/i);
});

test("removes the disposable starter preview", async () => {
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});

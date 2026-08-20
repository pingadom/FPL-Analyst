import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

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
  assert.match(html, /<title>FPL Lens — Breakthrough v3<\/title>/i);
  assert.match(html, /Build a squad/);
  assert.match(html, /2,400(?:<!-- -->)? candidate mixes/i);
  assert.match(html, /20(?:<!-- -->)? recursive finalists/i);
  assert.match(html, /Tune the lens/);
  assert.match(html, /Optimal XV/i);
  assert.match(html, /Chip desk/i);
  assert.match(html, /48(?:<!-- -->)? chip policies/i);
  assert.match(html, /Lens 7\.0[\s\S]{0,24}\+ chips/i);
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
  assert.match(html, /Championship stack challenger/i);
  assert.match(html, /Performance ladder/i);
  assert.match(html, /Experiment ledger/i);
  assert.match(html, /21\.4 points found/i);
  assert.match(html, /Season-by-season replay/i);
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
  assert.equal(payload.performance.stackLift, 21.4);
  assert.equal(payload.performance.targetHits, 2);
  assert.equal(payload.breakthrough.headline.averagePoints, 2212);
  assert.equal(payload.breakthrough.headline.holdoutLift, 34.5);
  assert.equal(payload.breakthrough.seasons.length, 8);
  assert.equal(payload.breakthrough.seasons.filter((season) => season.hit).length, 2);
  assert.equal(
    payload.chips.managerPlans["forecast-breakthrough-v2"].policyProfile,
    "forecast-v2 756-policy recursive winner",
  );
});

test("serves public Lens 7 projections with CORS", async () => {
  const response = await request("/api/projections?position=DEF&limit=2");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), "*");
  const payload = await response.json();
  assert.equal(payload.model, "Lens 7.0");
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

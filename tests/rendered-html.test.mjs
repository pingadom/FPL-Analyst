import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the FPL Lens decision room", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>FPL Lens — Data-led squad decisions<\/title>/i);
  assert.match(html, /Build a squad/);
  assert.match(html, /2,400(?:<!-- -->)? candidate mixes/i);
  assert.match(html, /240(?:<!-- -->)? recursive finalists/i);
  assert.match(html, /Tune the lens/);
  assert.match(html, /Optimal XV/i);
  assert.match(html, /Chip desk/i);
  assert.match(html, /144(?:<!-- -->)? chip policies/i);
  assert.match(html, /Lens 4\.0[\s\S]{0,24}\+ chips/i);
  assert.match(html, /Top-500k consistency test/i);
  assert.match(html, /Replay the past/i);
  assert.match(html, /Champion advice, tested/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("removes the disposable starter preview", async () => {
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});

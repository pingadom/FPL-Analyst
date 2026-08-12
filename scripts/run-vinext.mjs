import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const command = resolve("node_modules/vinext/dist/cli.js");
const result = spawnSync(process.execPath, [command, ...process.argv.slice(2)], {
  cwd: process.cwd(),
  env: { ...process.env, WRANGLER_LOG_PATH: ".wrangler/wrangler.log" },
  stdio: "inherit",
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);

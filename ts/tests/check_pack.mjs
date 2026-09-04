// What `npm pack` would actually ship, asserted rather than assumed.
//
//   npm pack --dry-run --json > pack.json && node tests/check_pack.mjs pack.json
//
// Apache-2.0 requires the licence text to travel with the artifact, and
// `npm pack` collects only files under `ts/` — the LICENSE at the repo root
// reaches no tarball on its own. The wasm has a second, invisible failure:
// `wasm-pack` writes its own `.gitignore` (containing `*`) into `ts/pkg/`,
// which npm honours when no `.npmignore` sits beside it, and that once
// published a package carrying no wasm at all. A published version cannot be
// taken back, so both are checked here.

import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) {
  console.error("usage: check_pack.mjs <npm pack --json output>");
  process.exit(2);
}

const files = JSON.parse(readFileSync(path, "utf8"))[0].files.map((f) => f.path);
const fail = (msg) => {
  console.error(`error: ${msg}:\n${files.join("\n")}`);
  process.exit(1);
};

if (!files.some((p) => p === "LICENSE")) fail("npm tarball ships no LICENSE");

const wasm = files.filter((p) => p.endsWith(".wasm"));
if (wasm.length !== 1) {
  fail(`expected exactly one .wasm in the npm tarball, got ${wasm.length}`);
}

console.log(`npm tarball ok: ${files.length} files, LICENSE, ${wasm[0]}`);

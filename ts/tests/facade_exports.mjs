// Everything the bundle exports has to reach the package entry point.
//
// `index.js` names them one by one rather than re-exporting `*`, which is what
// lets a new binding ship in the wasm and be unreachable through the package —
// 0.7.0 published `tapewasmVersion` that way. Nothing else noticed, because
// every test in here imports from `../index.js` and so only sees the names it
// already knows to ask for.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const bundle = await import(resolve(here, "..", "pkg", "stanwasm.js"));
const facade = await import(resolve(here, "..", "index.js"));

// `initSync` and the wasm-bindgen internals are not part of the surface; the
// default export is `init`, under another name.
const internal = new Set(["default", "initSync", "init_panic_hook"]);
const missing = Object.keys(bundle).filter((k) => !internal.has(k) && !(k in facade));

if (missing.length) {
  console.error(`FAIL: the bundle exports ${missing.join(", ")}, and index.js does not`);
  process.exit(1);
}

const types = await readFile(resolve(here, "..", "index.d.ts"), "utf8");
const untyped = Object.keys(facade).filter((k) => k !== "default" && !types.includes(k));
if (untyped.length) {
  console.error(`FAIL: index.d.ts does not name ${untyped.join(", ")}`);
  process.exit(1);
}

console.log(`facade ok: ${Object.keys(facade).length} exports, all typed`);

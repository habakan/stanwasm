#!/usr/bin/env bash
# Build stan-wasm-api as wasm32 + run wasm-bindgen via wasm-pack.
# Output goes to ts/pkg/, ready for `import { StanModel } from "stanwasm"`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "error: wasm-pack not found. Install with: cargo install wasm-pack" >&2
  exit 1
fi

wasm-pack build crates/stan-wasm-api \
  --target web \
  --out-dir "$ROOT/ts/pkg" \
  --release

# wasm-pack drops its own `.gitignore` (just `*`) into ts/pkg/. That's
# redundant here — the repo root .gitignore already excludes /ts/pkg — and
# actively harmful: npm's ignore-file resolution honors a nested .gitignore
# with no matching .npmignore, so `npm publish`/`npm pack` from ts/ would
# silently ship an empty pkg/ (no wasm, no glue JS) despite package.json's
# `"files": ["pkg/"]` saying to include it. Remove it so the package we'd
# actually publish is the one we tested.
rm -f "$ROOT/ts/pkg/.gitignore"

echo
echo "build ok:"
ls -la ts/pkg/

#!/usr/bin/env bash
# Build stan-wasm-api as wasm32 + run wasm-bindgen via wasm-pack.
# Output goes to ts/pkg/, ready for `import { StanModel } from "stan-wasm-rs"`.
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

echo
echo "build ok:"
ls -la ts/pkg/

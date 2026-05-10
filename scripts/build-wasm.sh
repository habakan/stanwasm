#!/usr/bin/env bash
# Build stan-wasm-api as wasm32 + run wasm-bindgen to produce TS-friendly output.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "error: wasm-bindgen CLI not found. Install with: cargo install wasm-bindgen-cli" >&2
  exit 1
fi

cargo build --release --target wasm32-unknown-unknown -p stan-wasm-api

WASM_IN="target/wasm32-unknown-unknown/release/stan_wasm_api.wasm"
PKG_OUT="ts/pkg"

mkdir -p "$PKG_OUT"
wasm-bindgen "$WASM_IN" \
  --out-dir "$PKG_OUT" \
  --target web \
  --no-typescript=false

# Optional: optimize with wasm-opt if available.
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz -o "$PKG_OUT/stan_wasm_api_bg.wasm" "$PKG_OUT/stan_wasm_api_bg.wasm"
fi

cp "$PKG_OUT/stan_wasm_api_bg.wasm" www/stan.wasm 2>/dev/null || true

echo
echo "build ok:"
ls -la "$PKG_OUT"

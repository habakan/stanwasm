# MoonBit と Rust で同じ Stan 推論エンジンを書き比べた話

ブラウザで Bayesian 推論を走らせる Stan 実装 ([stan-wasm](https://github.com/habakan/stan-wasm)) を MoonBit で書いた後、同じものを Rust で書き直した（このリポジトリ `stan-wasm-rs`）。両者を本番に近い構成で動かして得た知見をまとめる。

> **要旨**: MoonBit を採用した最大の根拠は wasm-gc backend のはずだったが、ホットパスには wasm-gc が効かない構造だった。Rust に移植しても性能はパリティ。残ったのは「ツールチェイン・コミュニティの厚さ」と「外部 Rust エコシステム (nuts-rs 等) と同じ言語で統合できる」という言語選択の素朴な利点だけ。**「wasm 性能のために MoonBit」は本プロジェクトには成立しなかった**。

---

## 背景: stan-wasm のアーキテクチャ

Stan 言語のモデルをブラウザで実行するには、大きく 3 段階の処理が要る:

1. **パーサ + 制約変換** — Stan ソース → AST → 制約 transform 付き AST
2. **AOT codegen** — モデル固有の `log_prob_grad` を wasm32 として生成（forward + backward を完全 unroll）
3. **NUTS サンプラ** — 適応的 leapfrog で `log_prob_grad` を呼びながら chain を回す

stan-wasm の MoonBit 版はこれを以下のように実装していた:

```
┌─────────────────────────────────┐    ┌─────────────────────┐
│  wasm_api.wasm (MoonBit, wasm-gc) │   │  nuts_rs.wasm        │
│  - パーサ・bytecode VM             │   │  (Rust, wasm32)      │
│  - AOT codegen → WAT 文字列        │   │  - NUTS コア         │
│  - 制約 transform                  │   │  - mass matrix 適応  │
└─────────────────────────────────┘    └─────────────────────┘
                          ▲                       │
                          │ JS callback で per-leapfrog
                          │ memcpy: nuts.memory ↔ model.memory
                          ▼
                  ┌─────────────────────┐
                  │  AOT model.wasm     │
                  │  (wasm32, 動的生成) │
                  └─────────────────────┘
```

**注目点:** 3 つの wasm モジュールが存在し、JS ブリッジ越しに連携する。`wasm_api.wasm` だけが wasm-gc。`nuts_rs.wasm` と動的生成される `model.wasm` は **どちらも wasm32**。

## なぜ MoonBit を選んだか（当時の主張）

stan-wasm の README と ARCHITECTURE.md は wasm-gc backend をこう謳っていた:

> wasm-gc ネイティブ GC 命令が wasm32 独自ヒープより約 4× 速い

これは MoonBit の `wasm-gc` バックエンドと `wasm32` バックエンドを比較した内部ベンチに基づく。前者は V8 の GC を使い、後者は自前のヒープを線形メモリに置く。MoonBit の wasm32 codegen が遅いのは確かにその通りで、wasm-gc にすると 4× 速くなる。

加えて wasm-gc は理論上:
- バイナリサイズが小さい（GC が無いので alloc/dealloc コードを持たない）
- 起動が速い（ヒープのセットアップが要らない）
- 文字列・配列・ADT が host GC 上にネイティブに乗る

これだけ見れば「Stan のパーサのような ADT/String を多用する処理には wasm-gc が最適」と判断するのは自然だった。

## ところが本番ホットパスでは wasm-gc が効かない

サンプリング中に毎ステップ走るのは **AOT model wasm の `log_prob_grad`** だけ。これは forward + backward を `f64` ローカルだけで完全 unroll した wasm32 モジュールで:

- alloc は 0
- 関数呼び出しは Math.exp/log/lgamma 等のインポートのみ
- ループも構造体も使わない（unroll 済みなので）

wasm-gc が活きる場面（オブジェクト確保・ループ内の配列アクセス）が**ホットパスに一切存在しない**。MoonBit 版でも、ここは MoonBit の wasm-gc コンパイラを使わず WAT を生成 → wabt → wasm32 として実行していた。

つまり**性能を決めるホットパスは MoonBit を採用しても採用しなくても wasm32 で同じ**。wasm-gc backend が活きるのは:

- パーサ（実行時間: ~10ms、1モデル 1 回）
- AOT codegen そのもの（実行時間: ~10ms、1モデル 1 回）
- 制約 transform のセットアップ

つまり**1モデルあたり 1 回しか走らない部分**。サンプリング 2000 draws の合計 ~10ms から見れば誤差。

## 実バイナリサイズも変わらない

「wasm-gc は GC 不要なのでバイナリが小さい」も実際には成立しなかった:

| バンドル | サイズ |
|---|---:|
| MoonBit `wasm_api.wasm` (wasm-gc) | 278 KB |
| MoonBit `nuts_rs.wasm` (wasm32) | 245 KB |
| **MoonBit 合計（2 モジュール）** | **523 KB** |
| Rust `stan_wasm_api.wasm` (wasm32, wasm-pack 出力) | **365 KB** |

Rust 単一バンドルの方が **MoonBit ペアより 30% 小さい**。`wasm-opt -Oz` と LTO + `panic=abort` がかなり効いた。「wasm-gc なら小さくなる」は単純には言えなかった。

## Rust への移植判断

判断材料を整理した結果:

| 観点 | MoonBit | Rust |
|---|---|---|
| ホットパス性能 | wasm32 (= V8 JIT) | wasm32 (= V8 JIT) — 同条件 |
| パーサ性能 | wasm-gc 4× 速い | dlmalloc — でも 1 回しか走らない |
| バンドルサイズ | 523 KB (2 モジュール) | 365 KB (単一) |
| nuts-rs との統合 | 別 wasm + JS コールバック | 同 wasm に内蔵可能 |
| ツール成熟度 | 言語仕様が動く・LSP 等まだ薄い | rustc / cargo / 各種 lsp 完備 |
| 利用者プール | 小さい | 大きい |
| 既存コード資産 | MoonBit 11k 行 | 既存 nuts-wasm bridge |

ホットパスの優位性が消え、サイズ・統合・ツール・コミュニティで全敗となった時点で MoonBit を残す技術的な根拠は無くなった。

唯一残るのは「**MoonBit のショーケースとしての価値**」だが、それは事業戦略であって工学判断ではない。

## 移植後の構成

```
┌──────────────────────────────────────────┐    ┌─────────────────────────┐
│  stan_wasm_api.wasm (Rust, wasm32)        │    │  AOT model.wasm         │
│  - パーサ・AST 評価                        │    │  (動的生成 wasm32)       │
│  - autodiff tape                          │    │  - 完全 unroll log_prob │
│  - AOT codegen (wasm-encoder で直接 binary) │   │                         │
│  - nuts-rs (内蔵 Rust crate)               │    │  imports memory from   │
│  - サンプリング driver                     │    │  "stan" namespace       │
└──────────────────────────────────────────┘    └─────────────────────────┘
                          ▲                                 │
                          └── 5 行の JS shim ───────────────┘
                              （V8 が hot 化後 inline）
```

主要な変更点:

1. **2 モジュール → 1 モジュール + 動的生成 1 個**: nuts-rs を Rust crate として直接依存に入れ、サンプリング driver と同じ wasm に統合。
2. **共有メモリ**: AOT model wasm が `(import "stan" "memory" (memory 1))` でホストのメモリを取り込む。MoonBit 版で必要だった `memcpy: nuts.memory ↔ model.memory` がゼロになる。
3. **wabt JS 依存撤廃**: `wasm-encoder` クレートで wasm バイナリを直接出力。WAT 文字列経由でのブラウザコンパイルが不要に。

## 性能（Node.js V8、Apple Silicon）

n_warmup=1000, n_draws=1000 のサンプリング全体の wall time:

| モデル | MoonBit (WAT AOT) | Rust (sampleViaAot) |
|---|---:|---:|
| logistic / poisson regression (2 params) | 6.0 ms | 5.5 ms |
| eight_schools_ncp (10 params) | 5.5 ms | 6.0 ms |

**実態はパリティ**。Rust が勝っているケースもあれば負けているケースもあり、計測ノイズ（5–10%）の範囲内。差を生んでいるのは:

- **+** 共有メモリで lpg 1 回あたり `2 * n_params * 8 byte` の memcpy が消える
- **+** 単一バンドルの方が V8 の JIT 単位が一つで済む
- **−** Rust libstd + dlmalloc のオーバーヘッドが小さく入り込む

これらは互いに打ち消し合う規模。

## 「Rust が wasm-gc を使えてない」のは事実

念のため検証した。`wasmparser` で GC 機能を OFF にしても両方の wasm が validate する:

```rust
let no_gc = WasmFeatures::default() - WasmFeatures::GC;
Validator::new_with_features(no_gc).validate_all(&bytes)?;  // 通る
```

(`crates/stan-codegen/tests/no_wasm_gc.rs` でテスト化済)

stable Rust に wasm-gc ターゲットは存在しない（experimental も限定的）。Rust→wasm32 は線形メモリ + dlmalloc。MoonBit と同じく **AOT model 側は wasm-gc 不使用なので等条件**。差が出るのは parser/compiler 側だけで、上述の通りホットパスではない。

## 教訓

1. **「wasm-gc が速い」は文脈依存**。ループ内アロケーションがあるパースのような処理には効くが、HMC/NUTS のホットパスのように `f64` ローカルだけで完結する unroll コードには関係ない。**自プロジェクトのホットパスがどっちのカテゴリかを早めに見極める**べきだった。

2. **「言語選択」と「ホットパス言語」は分離して考える**。stan-wasm の場合、サンプリングのホットパス（AOT model）はどの言語を選んでも結局 wasm32 として生成される。**ホストとなる言語**は parser/compiler という cold path しか担当しない。

3. **エコシステム > 言語の僅差な性能**。nuts-rs を Rust crate として `Cargo.toml` に書くだけで取り込めるのと、JS bridge で別 wasm として渡すのとでは、実装複雑度が大きく違う。今回は後者を前者にできたので、コードベースが体感 30% 小さくなった。

4. **「MoonBit を選んだ理由」を文章化していて助かった**。ARCHITECTURE.md の「wasm-gc 4×」は言語選択の論拠だったが、実際にはホットパスに関係しないことを後で確認できた。文章化していなければ、なんとなく「MoonBit の方が速いかも」と思い込んだまま続けていた可能性がある。

## ベンチを再現する

```bash
# 全テスト
cargo test --workspace

# ネイティブ
cargo run --release -p stan-cli -- bench all

# Node.js V8 + wasm32（MoonBit と apples-to-apples）
./scripts/build-wasm.sh
cd ts && node --experimental-strip-types tests/bench.ts
```

## 関連ドキュメント

- `docs/MIGRATION.md` — フェーズ別の移植計画
- `docs/BENCHMARKS.md` — 計測値と再現手順
- `crates/stan-codegen/tests/no_wasm_gc.rs` — wasm-gc 不使用の検証テスト

## ライセンス

Apache-2.0

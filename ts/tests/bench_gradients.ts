// Per-gradient cost of both paths, across the model set in `bench_models.ts`.
//
//   node --experimental-strip-types tests/bench_gradients.ts [N] [--emit DIR]
//
// Gradient granularity, not sampling wall clock: the two paths can take
// different NUTS trajectories, and so can any other implementation.
//
// `--emit DIR` also writes each model out as `<name>.stan`, its data as
// `<name>.data.json`, and the unconstrained point it is evaluated at as
// `<name>.params.json`, so another implementation can be pointed at exactly
// the same model, data and point.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, { StanModel } from "../index.js";
import { benchModels } from "./bench_models.ts";

const here = dirname(fileURLToPath(import.meta.url));
await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "stanwasm_bg.wasm")) });

const lgamma = (x: number): number => {
  let z = x, r = 0;
  while (z < 10) { r -= Math.log(z); z += 1; }
  const zi = 1 / z, zi2 = zi * zi;
  return r + (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
       + zi * (1 / 12 + zi2 * (-1 / 360 + zi2 / 1260));
};
const digamma = (x: number): number => {
  let xx = x, r = 0;
  while (xx < 6) { r -= 1 / xx; xx += 1; }
  const inv = 1 / xx, inv2 = inv * inv;
  return r + Math.log(xx) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
};
const phi = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
    + t * (-1.821255978 + t * 1.330274429))));
  const cdf = 1 - (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x) * poly;
  return x >= 0 ? cdf : 1 - cdf;
};
const mathImports = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos,
  pow: Math.pow, lgamma, digamma, phi,
};

const args = process.argv.slice(2);
const N = Number(args.find((a) => /^\d+$/.test(a)) ?? 5000);
const emitAt = args.indexOf("--emit");
const emitDir = emitAt >= 0 ? resolve(process.cwd(), args[emitAt + 1]) : null;
if (emitDir) await mkdir(emitDir, { recursive: true });

type Row = {
  name: string; nParams: number; scratch: number;
  replay: number; aot: number; ok: boolean;
};

type Cand = { name: string; run: () => void };
const cands: Cand[] = [];
const rows: Row[] = [];

for (const model of benchModels(N)) {
  const m = new StanModel(model.src, JSON.stringify(model.data));
  const params = new Float64Array(model.init);
  if (params.length !== m.n_params) {
    throw new Error(`${model.name}: init has ${params.length}, model wants ${m.n_params}`);
  }
  const ref = m.logProbGrad(params);
  if (!Number.isFinite(ref[0])) throw new Error(`${model.name}: log_prob is ${ref[0]}`);

  const bytes = m.compileToWasm();
  const scratch = m.aotScratchInit();
  const mem = new WebAssembly.Memory({
    initial: Math.ceil((m.n_params * 16 + scratch.length * 8) / 65536) + 4,
  });
  const aot = await WebAssembly.instantiate(bytes, { stan: { memory: mem }, Math: mathImports });
  const view = new Float64Array(mem.buffer);
  view.set(scratch, m.n_params * 2);
  view.set(params, 0);
  const lpg = aot.instance.exports.log_prob_grad as
    (p: number, g: number, n: number, s: number) => number;
  const lp = lpg(0, m.n_params * 8, m.n_params, m.n_params * 16);
  const grads = [...view.subarray(m.n_params, 2 * m.n_params)];
  const ok = Math.abs(lp - ref[0]) < 1e-8 * (1 + Math.abs(ref[0]))
    && grads.every((v, i) => Math.abs(v - ref[i + 1]) < 1e-8 * (1 + Math.abs(ref[i + 1])));

  rows.push({
    name: model.name, nParams: m.n_params, scratch: scratch.length,
    replay: 0, aot: 0, ok,
  });
  cands.push({ name: `${model.name}/replay`, run: () => { m.logProbGrad(params); } });
  cands.push({
    name: `${model.name}/aot`,
    run: () => { lpg(0, m.n_params * 8, m.n_params, m.n_params * 16); },
  });

  if (emitDir) {
    await writeFile(resolve(emitDir, `${model.name}.stan`), model.src + "\n");
    await writeFile(resolve(emitDir, `${model.name}.data.json`), JSON.stringify(model.data));
    // Two points, so a comparison can tell a gradient that disagrees from a
    // log density that only differs by the normalising constants Stan's `~`
    // drops — those are the same at every point, the gradient is not.
    const second = model.init.map((v, i) => v + 0.05 * Math.cos(i + 1));
    for (const [tag, point] of [["", model.init], ["2", second]] as const) {
      const r = m.logProbGrad(new Float64Array(point));
      await writeFile(
        resolve(emitDir, `${model.name}.params${tag}.json`),
        JSON.stringify({ params_r: point }),
      );
      await writeFile(
        resolve(emitDir, `${model.name}.ref${tag}.json`),
        JSON.stringify({ lp: r[0], grads: [...r.slice(1)] }),
      );
    }
  }
}

// Interleaved: every candidate is measured once per round, in a rotating order,
// so a drift in machine state hits all of them rather than whichever ran last.
const ITERS = Number(process.env.ITERS ?? 200);
const ROUNDS = Number(process.env.ROUNDS ?? 30);
const best = new Map(cands.map((c) => [c.name, Infinity]));
for (const c of cands) for (let i = 0; i < 500; i++) c.run();
for (let r = 0; r < ROUNDS; r++) {
  for (let i = 0; i < cands.length; i++) {
    const c = cands[(i + r) % cands.length];
    const t = performance.now();
    for (let j = 0; j < ITERS; j++) c.run();
    best.set(c.name, Math.min(best.get(c.name)!, ((performance.now() - t) * 1000) / ITERS));
  }
}

console.log(`\nN=${N}, ${process.version}, µs per gradient, min of ${ROUNDS} interleaved rounds\n`);
console.log("model         | params | scratch |  replay |     AOT | speedup | grads");
console.log("-".repeat(74));
for (const row of rows) {
  const replay = best.get(`${row.name}/replay`)!;
  const aot = best.get(`${row.name}/aot`)!;
  console.log(
    [
      row.name.padEnd(13),
      String(row.nParams).padStart(6),
      String(row.scratch).padStart(7),
      replay.toFixed(2).padStart(7),
      aot.toFixed(2).padStart(7),
      (replay / aot).toFixed(2).padStart(6) + "x",
      row.ok ? "agree" : "MISMATCH",
    ].join(" | "),
  );
}
if (emitDir) {
  await writeFile(
    resolve(emitDir, "stanwasm.json"),
    JSON.stringify(
      Object.fromEntries(rows.map((r) => [r.name, {
        params: r.nParams,
        scratch: r.scratch,
        replay: best.get(`${r.name}/replay`),
        aot: best.get(`${r.name}/aot`),
      }])),
      null,
      1,
    ),
  );
  console.log(`\nwrote ${rows.length} models to ${emitDir}`);
}
if (rows.some((r) => !r.ok)) process.exit(1);

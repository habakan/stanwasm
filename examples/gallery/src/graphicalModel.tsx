import { memo, useId, useMemo, type ReactNode } from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";

/** Wrap once at the app root. MathJax is served from a locally-copied
 *  bundle (see `copy-mathjax` in package.json), not a CDN — every other
 *  part of this app runs offline, and a runtime fetch to a third-party
 *  CDN just to typeset a formula would quietly break that. Uses the SVG
 *  output component specifically (not the more common CHTML one): CHTML
 *  draws glyphs via @font-face files referenced relative to the loaded
 *  script, which we'd have to vendor and keep path-matched separately —
 *  miss that and it silently falls back to the browser's default font
 *  instead of MathJax's. SVG embeds glyph outlines directly in the output,
 *  so a single self-contained JS file is enough. */
export function MathJaxProvider({ children }: { children: ReactNode }) {
  const src = `${import.meta.env.BASE_URL}mathjax-tex-svg.js`;
  return (
    <MathJaxContext version={3} src={src} config={{ tex: { packages: { "[+]": ["ams"] } } }}>
      {children}
    </MathJaxContext>
  );
}

// ============================================================================
// Stan source -> graphical-model graph. Every diagram in this app (including
// Get Started's live editor) is derived from this parser rather than
// hand-drawn, so the diagram can never drift out of sync with the actual
// model being sampled.
// ============================================================================

const GREEK: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", zeta: "ζ",
  eta: "η", theta: "θ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ",
  nu: "ν", xi: "ξ", omicron: "ο", pi: "π", rho: "ρ", sigma: "σ",
  tau: "τ", upsilon: "υ", phi: "φ", chi: "χ", psi: "ψ", omega: "ω",
};
const LATEX_GREEK: Record<string, string> = Object.fromEntries(
  Object.keys(GREEK).map((k) => [k, "\\" + k]),
);
const SUB_UNICODE: Record<string, string> = {
  i: "ᵢ", j: "ⱼ", k: "ₖ", n: "ₙ", m: "ₘ",
  "0": "₀", "1": "₁", "2": "₂", "3": "₃",
};

function toUnicodeLabel(name: string, subscript?: string | null): string {
  const tildeMatch = name.match(/^(.+)_tilde$/);
  let sym = tildeMatch ? (GREEK[tildeMatch[1]] ?? tildeMatch[1]) : (GREEK[name] ?? name);
  if (tildeMatch) sym += "̃";
  if (subscript) sym += [...subscript].map((c) => SUB_UNICODE[c] ?? c).join("");
  return sym;
}

function toLatexIdent(name: string): string {
  const tildeMatch = name.match(/^(.+)_tilde$/);
  if (tildeMatch) return `\\tilde{${LATEX_GREEK[tildeMatch[1]] ?? tildeMatch[1]}}`;
  if (!LATEX_GREEK[name]) {
    const m = name.match(/^(.+)_(.+)$/);
    if (m) return `${LATEX_GREEK[m[1]] ?? m[1]}_{\\mathrm{${m[2]}}}`;
  }
  return LATEX_GREEK[name] ?? name;
}

const DIST_LATEX: Record<string, (a: string[]) => string> = {
  normal: (a) => `\\mathcal{N}(${a[0]}, ${a[1]})`,
  std_normal: () => `\\mathcal{N}(0, 1)`,
  half_normal: (a) => `\\text{HalfNormal}(${a[0]})`,
  exponential: (a) => `\\text{Exponential}(${a[0]})`,
  cauchy: (a) => `\\text{Cauchy}(${a[0]}, ${a[1]})`,
  student_t: (a) => `\\text{StudentT}(${a[0]}, ${a[1]}, ${a[2]})`,
  lognormal: (a) => `\\text{LogNormal}(${a[0]}, ${a[1]})`,
  gamma: (a) => `\\text{Gamma}(${a[0]}, ${a[1]})`,
  beta: (a) => `\\text{Beta}(${a[0]}, ${a[1]})`,
  bernoulli: (a) => `\\text{Bernoulli}(${a[0]})`,
  bernoulli_logit: (a) => `\\text{Bernoulli}(\\mathrm{logit}^{-1}(${a[0]}))`,
  poisson: (a) => `\\text{Poisson}(${a[0]})`,
  neg_binomial_2: (a) => `\\text{NegBinomial2}(${a[0]}, ${a[1]})`,
  multi_normal_cholesky: (a) => `\\mathcal{N}(${a[0]}, ${a[1]}${a[1]}^{\\mathsf{T}})`,
  dirichlet: (a) => `\\text{Dirichlet}(${a[0]})`,
  lkj_corr_cholesky: (a) => `\\text{LKJCholesky}(${a[0]})`,
  uniform: (a) => `\\text{Uniform}(${a[0]}, ${a[1]})`,
};
function distLatex(name: string, args: string[]): string {
  return (DIST_LATEX[name] ?? ((a: string[]) => `\\text{${name}}(${a.join(", ")})`))(args);
}

function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/.*$/gm, "");
}

function extractBlock(src: string, namePattern: string): string | null {
  const m = new RegExp(`\\b${namePattern}\\s*\\{`).exec(src);
  if (!m) return null;
  let depth = 1;
  let i = m.index + m[0].length;
  const start = i;
  while (i < src.length && depth > 0) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") depth--;
    i++;
  }
  return src.slice(start, i - 1);
}

function splitTopLevel(text: string, delim: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let cur = "";
  for (const ch of text) {
    if ("([{".includes(ch)) depth++;
    else if (")]}".includes(ch)) depth--;
    if (ch === delim && depth === 0) {
      parts.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim().length > 0) parts.push(cur);
  return parts;
}

function findMatchingBrace(t: string, openIdx: number): number {
  let depth = 1;
  for (let k = openIdx + 1; k < t.length; k++) {
    if (t[k] === "{") depth++;
    else if (t[k] === "}" && --depth === 0) return k;
  }
  return t.length - 1;
}
function findMatchingParen(t: string, openIdx: number): number {
  let depth = 1;
  for (let k = openIdx + 1; k < t.length; k++) {
    if (t[k] === "(") depth++;
    else if (t[k] === ")" && --depth === 0) return k;
  }
  return t.length - 1;
}
function findTopLevelChar(t: string, from: number, ch: string): number {
  let depth = 0;
  for (let k = from; k < t.length; k++) {
    if ("([{".includes(t[k])) depth++;
    else if (")]}".includes(t[k])) depth--;
    else if (t[k] === ch && depth === 0) return k;
  }
  return t.length - 1;
}

interface VarInfo {
  name: string;
  kind: "data" | "param" | "transformed";
  sizeExpr: string | null;
  initExpr: string | null;
  order: number;
}

function parseDecl(stmt: string): { name: string; sizeExpr: string | null; initExpr: string | null } | null {
  const s = stmt.trim();
  if (!s) return null;
  const m = /^([a-zA-Z_]\w*)\s*(?:<[^>]*>)?\s*(?:\[([^\]]*)\])?\s*([\s\S]*)$/.exec(s);
  if (!m) return null;
  const dim1 = m[2] ?? null;
  let rest = m[3].trim();
  let initExpr: string | null = null;
  const eq = splitTopLevel(rest, "=");
  if (eq.length > 1) {
    rest = eq[0].trim();
    initExpr = eq.slice(1).join("=").trim();
  }
  const tokens = rest.split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return null;
  const nameToken = tokens[tokens.length - 1];
  let sizeExpr = dim1;
  const m2 = nameToken.match(/^([a-zA-Z_]\w*)\[([^\]]*)\]$/);
  const name = m2 ? m2[1] : nameToken;
  if (!sizeExpr && m2) sizeExpr = m2[2];
  return { name, sizeExpr, initExpr };
}

function parseVarBlock(blockText: string, kind: VarInfo["kind"], order: { n: number }, out: Map<string, VarInfo>) {
  for (const stmt of splitTopLevel(blockText, ";")) {
    const d = parseDecl(stmt);
    if (!d) continue;
    out.set(d.name, { name: d.name, kind, sizeExpr: d.sizeExpr, initExpr: d.initExpr, order: order.n++ });
  }
}

interface Stmt {
  lhs: string;
  distName: string;
  args: string[];
  raw: string;
  plateBound: string | null;
  loopVar: string | null;
}

function parseSamplingStmt(stmtText: string): { lhs: string; distName: string; args: string[]; raw: string } | null {
  const s = stmtText.trim().replace(/;\s*$/, "");
  const tilde = s.indexOf("~");
  if (tilde < 0) return null;
  const lhs = s.slice(0, tilde).trim();
  const rhs = s.slice(tilde + 1).trim();
  const call = /^([a-zA-Z_]\w*)\s*\(([\s\S]*)\)$/.exec(rhs);
  if (!call) return null;
  return { lhs, distName: call[1], args: splitTopLevel(call[2], ",").map((a) => a.trim()), raw: stmtText };
}

function collectSamplingStatements(text: string): Stmt[] {
  const out: Stmt[] = [];
  walk(text, null, null);
  return out;

  function walk(t: string, plateBound: string | null, loopVar: string | null) {
    let i = 0;
    while (i < t.length) {
      while (i < t.length && /\s/.test(t[i])) i++;
      if (i >= t.length) break;

      if (/^for\b/.test(t.slice(i))) {
        const m = /^for\s*\(\s*([a-zA-Z_]\w*)\s+in\s+[^:()]+:\s*([a-zA-Z_]\w*)\s*\)/.exec(t.slice(i));
        if (m) {
          let j = i + m[0].length;
          while (j < t.length && /\s/.test(t[j])) j++;
          if (t[j] === "{") {
            const close = findMatchingBrace(t, j);
            walk(t.slice(j + 1, close), m[2], m[1]);
            i = close + 1;
          } else {
            const semi = findTopLevelChar(t, j, ";");
            walk(t.slice(j, semi + 1), m[2], m[1]);
            i = semi + 1;
          }
          continue;
        }
      }
      if (/^(if|while)\b/.test(t.slice(i))) {
        const parenStart = t.indexOf("(", i);
        const parenEnd = findMatchingParen(t, parenStart);
        let j = parenEnd + 1;
        while (j < t.length && /\s/.test(t[j])) j++;
        if (t[j] === "{") {
          const close = findMatchingBrace(t, j);
          walk(t.slice(j + 1, close), plateBound, loopVar);
          i = close + 1;
        } else {
          const semi = findTopLevelChar(t, j, ";");
          walk(t.slice(j, semi + 1), plateBound, loopVar);
          i = semi + 1;
        }
        continue;
      }
      if (t[i] === "{") {
        const close = findMatchingBrace(t, i);
        walk(t.slice(i + 1, close), plateBound, loopVar);
        i = close + 1;
        continue;
      }
      const semi = findTopLevelChar(t, i, ";");
      const stmtText = t.slice(i, semi + 1);
      i = semi + 1;
      const parsed = parseSamplingStmt(stmtText);
      if (parsed) out.push({ ...parsed, plateBound, loopVar });
    }
  }
}

function extractIdentifiers(expr: string): string[] {
  return [...new Set(expr.match(/[a-zA-Z_]\w*/g) ?? [])];
}
function baseName(lhs: string): string {
  return lhs.match(/^([a-zA-Z_]\w*)/)?.[1] ?? lhs;
}

export type NodeShape = "circle" | "square";
export interface GNode {
  id: string;
  shape: NodeShape;
  filled: boolean;
  symbol: string;
  formula: string | null;
  plate: string | null;
  plateLoopVar: string | null;
  depth: number;
  order: number;
}
export interface GEdge { from: string; to: string; }
export interface Graph { nodes: GNode[]; edges: GEdge[]; plates: { bound: string; loopVar: string }[]; }

interface PlateCtx { plateBound: string | null; loopVar: string | null; members: string[]; }

/** Parses `src` into a Bayesian-network graph: nodes are data/parameter/
 *  transformed-parameter declarations, edges come from what each sampling
 *  statement's (or deterministic transform's) arguments reference. Returns
 *  `null` for anything that doesn't parse as a plausible Stan program —
 *  callers should treat that as "no diagram to show" rather than an error,
 *  since e.g. Get Started's editor is mid-edit most of the time. */
export function parseStanGraph(src: string): Graph | null {
  try {
    const clean = stripComments(src);
    const dataBlock = extractBlock(clean, "data");
    const paramsBlock = extractBlock(clean, "parameters");
    const transBlock = extractBlock(clean, "transformed\\s+parameters");
    const modelBlock = extractBlock(clean, "model");
    if (!paramsBlock || !modelBlock) return null;

    const order = { n: 0 };
    const vars = new Map<string, VarInfo>();
    if (dataBlock) parseVarBlock(dataBlock, "data", order, vars);
    parseVarBlock(paramsBlock, "param", order, vars);
    if (transBlock) parseVarBlock(transBlock, "transformed", order, vars);

    const edges: GEdge[] = [];
    const nodePlate = new Map<string, { plate: string; loopVar: string }>();
    const nodeFormula = new Map<string, string>();
    const usedData = new Set<string>();
    const observedData = new Set<string>();

    function ctxFor(stmt: { plateBound: string | null; loopVar: string | null; raw: string }, identifiers: string[]): PlateCtx {
      if (stmt.plateBound) {
        const loopVar = stmt.loopVar!;
        const members = identifiers.filter(
          (n) => stmt.raw.includes(`${n}[${loopVar}]`) || vars.get(n)?.sizeExpr === stmt.plateBound,
        );
        return { plateBound: stmt.plateBound, loopVar, members };
      }
      const withSize = identifiers.filter((n) => vars.get(n)?.sizeExpr);
      if (withSize.length === 0) return { plateBound: null, loopVar: null, members: [] };
      const bound = vars.get(withSize[0])!.sizeExpr!;
      return { plateBound: bound, loopVar: "i", members: withSize.filter((n) => vars.get(n)!.sizeExpr === bound) };
    }

    function argToLatex(expr: string, ctx: PlateCtx): string {
      let s = expr.replace(/([a-zA-Z_]\w*)(\[[^\]]*\])?/g, (whole, name: string, idxPart?: string) => {
        if (!vars.has(name)) return whole;
        const sub = idxPart ? idxPart.slice(1, -1) : ctx.members.includes(name) ? ctx.loopVar ?? "i" : null;
        if (vars.get(name)!.kind === "data") usedData.add(name);
        return toLatexIdent(name) + (sub ? `_{${sub}}` : "");
      });
      s = s
        .replace(/\bexp\(/g, "\\exp(")
        .replace(/\bsqrt\(/g, "\\sqrt(")
        .replace(/\binv_logit\(/g, "\\mathrm{logit}^{-1}(")
        .replace(/\blog1p\(/g, "\\log(1+")
        .replace(/\blog\(/g, "\\log(");
      return s.replace(/\s*\*\s*/g, "\\,").trim();
    }

    function markPlate(ids: string[], ctx: PlateCtx) {
      if (!ctx.plateBound) return;
      for (const n of ids) if (ctx.members.includes(n)) nodePlate.set(n, { plate: ctx.plateBound, loopVar: ctx.loopVar! });
    }

    for (const stmt of collectSamplingStatements(modelBlock)) {
      const lhsName = baseName(stmt.lhs);
      const lhsVar = vars.get(lhsName);
      if (!lhsVar) continue;
      const argIdents = stmt.args.flatMap(extractIdentifiers).filter((n) => vars.has(n));
      const allIdents = [lhsName, ...argIdents];
      const ctx = ctxFor(stmt, allIdents);
      markPlate(allIdents, ctx);
      for (const n of argIdents) {
        edges.push({ from: n, to: lhsName });
        if (vars.get(n)!.kind === "data") usedData.add(n);
      }
      if (lhsVar.kind === "data") {
        usedData.add(lhsName);
        observedData.add(lhsName);
      }
      nodeFormula.set(lhsName, distLatex(stmt.distName, stmt.args.map((a) => argToLatex(a, ctx))));
    }

    for (const [name, v] of vars) {
      if (v.kind !== "transformed" || !v.initExpr) continue;
      const idents = extractIdentifiers(v.initExpr).filter((n) => vars.has(n) && n !== name);
      const allIdents = [name, ...idents];
      const ctx = ctxFor({ plateBound: null, loopVar: null, raw: v.initExpr }, allIdents);
      markPlate(allIdents, ctx);
      for (const n of idents) {
        edges.push({ from: n, to: name });
        if (vars.get(n)!.kind === "data") usedData.add(n);
      }
      const lhsSub = ctx.members.includes(name) ? ctx.loopVar ?? "i" : null;
      nodeFormula.set(name, `${toLatexIdent(name)}${lhsSub ? `_{${lhsSub}}` : ""} = ${argToLatex(v.initExpr, ctx)}`);
    }

    const nodeIds = new Set<string>();
    for (const [name, v] of vars) {
      if (v.kind === "data" && !usedData.has(name)) continue;
      nodeIds.add(name);
    }
    const filteredEdges = edges.filter((e) => nodeIds.has(e.from) && nodeIds.has(e.to));

    const parentsOf = new Map<string, string[]>();
    for (const e of filteredEdges) {
      if (!parentsOf.has(e.to)) parentsOf.set(e.to, []);
      parentsOf.get(e.to)!.push(e.from);
    }
    const depthCache = new Map<string, number>();
    function depthOf(n: string, visiting: Set<string>): number {
      if (depthCache.has(n)) return depthCache.get(n)!;
      if (visiting.has(n)) return 0;
      visiting.add(n);
      const parents = parentsOf.get(n) ?? [];
      const d = parents.length === 0 ? 0 : 1 + Math.max(...parents.map((p) => depthOf(p, visiting)));
      visiting.delete(n);
      depthCache.set(n, d);
      return d;
    }

    const nodes: GNode[] = [...nodeIds].map((name) => {
      const v = vars.get(name)!;
      const plateInfo = nodePlate.get(name) ?? null;
      const filled = observedData.has(name);
      return {
        id: name,
        shape: v.kind === "data" && !filled ? "square" : "circle",
        filled,
        symbol: toUnicodeLabel(name, plateInfo?.loopVar ?? null),
        formula: nodeFormula.get(name) ?? null,
        plate: plateInfo?.plate ?? null,
        plateLoopVar: plateInfo?.loopVar ?? null,
        depth: depthOf(name, new Set()),
        order: v.order,
      };
    });
    if (nodes.length === 0) return null;

    const plateMap = new Map<string, string>();
    for (const n of nodes) if (n.plate && n.plateLoopVar) plateMap.set(n.plate, n.plateLoopVar);

    return {
      nodes,
      edges: filteredEdges,
      plates: [...plateMap.entries()].map(([bound, loopVar]) => ({ bound, loopVar })),
    };
  } catch {
    return null;
  }
}

// ---- layout + rendering ----------------------------------------------------

const NODE_R = 18;
const SQ = 26;
const ROW_GAP = 90;
const COL_GAP = 92;
const TOP_PAD = 34;
const SIDE_PAD = 44;

interface Laid extends GNode { x: number; y: number; }
interface PlateBox { plate: string; loopVar: string; x: number; y: number; w: number; h: number; }
interface Layout { nodes: Laid[]; edges: GEdge[]; width: number; height: number; plateBoxes: PlateBox[]; }

function rowsFor(list: GNode[]): GNode[][] {
  const depths = [...new Set(list.map((n) => n.depth))].sort((a, b) => a - b);
  return depths.map((d) => list.filter((n) => n.depth === d).sort((a, b) => a.order - b.order));
}

/** Population-level (unplated) nodes form row(s) at the top; each plate's
 *  members get their own self-contained rows in a box below, spanning the
 *  full width — this is the standard plate-notation layout, and keeping the
 *  two node populations in separate row systems (rather than one global grid)
 *  is what stops an unplated root like `alpha` from ending up sharing a row,
 *  and hence x-coordinate range, with plated data nodes it has nothing to do
 *  with positionally. */
function layout(graph: Graph): Layout {
  const outside = graph.nodes.filter((n) => !n.plate);
  const outsideRows = rowsFor(outside);
  const plateGroups = graph.plates.map(({ bound, loopVar }) => ({
    plate: bound,
    loopVar,
    rows: rowsFor(graph.nodes.filter((n) => n.plate === bound)),
  }));

  const maxCols = Math.max(
    1,
    ...outsideRows.map((r) => r.length),
    ...plateGroups.map((g) => Math.max(1, ...g.rows.map((r) => r.length))),
  );
  const width = Math.max(220, maxCols * COL_GAP + SIDE_PAD * 2);

  const laid: Laid[] = [];
  let y = TOP_PAD;
  for (const row of outsideRows) {
    row.forEach((n, i) => laid.push({ ...n, x: (width / (row.length + 1)) * (i + 1), y }));
    y += ROW_GAP;
  }

  const plateBoxes: PlateBox[] = [];
  for (const { plate, loopVar, rows } of plateGroups) {
    const boxTop = y;
    for (const row of rows) {
      row.forEach((n, i) =>
        laid.push({ ...n, x: SIDE_PAD + ((width - 2 * SIDE_PAD) / (row.length + 1)) * (i + 1), y }),
      );
      y += ROW_GAP;
    }
    // +78 leaves room below the last row's nodes for both a formula
    // (foreignObject, ~34px) and an "observed" tag under it (~14px) — the
    // worst case, when the row's node is both filled and has a formula.
    const boxBottom = y - ROW_GAP + 78;
    plateBoxes.push({ plate, loopVar, x: SIDE_PAD - 22, y: boxTop - 28, w: width - 2 * (SIDE_PAD - 22), h: boxBottom - (boxTop - 28) });
    y = boxBottom + 4;
  }

  return { nodes: laid, edges: graph.edges, width, height: y + 4, plateBoxes };
}

function edgeLine(a: Laid, b: Laid) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy) || 1;
  const ra = a.shape === "circle" ? NODE_R : SQ * 0.6;
  const rb = b.shape === "circle" ? NODE_R : SQ * 0.6;
  return { x1: a.x + (dx / dist) * ra, y1: a.y + (dy / dist) * ra, x2: b.x - (dx / dist) * rb, y2: b.y - (dy / dist) * rb };
}

function GraphicalModelInner({ stanCode }: { stanCode: string }) {
  const rawId = useId();
  const markerId = `arrow-${rawId.replace(/[^a-zA-Z0-9]/g, "")}`;
  const graph = useMemo(() => parseStanGraph(stanCode), [stanCode]);

  if (!graph) return <p className="hint">model structure preview unavailable</p>;

  const { nodes, edges, width, height, plateBoxes } = layout(graph);
  const byId = new Map(nodes.map((n) => [n.id, n]));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="graphical-model">
      <defs>
        <marker id={markerId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth={6} markerHeight={6} orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#999" />
        </marker>
      </defs>

      {plateBoxes.map((b) => (
        <g key={b.plate}>
          <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={6} fill="none" stroke="#ccc" />
          <text x={b.x + b.w - 8} y={b.y + b.h - 8} textAnchor="end" fontSize={10} fill="#888">
            {b.loopVar} = 1..{b.plate}
          </text>
        </g>
      ))}

      {edges.map((e, i) => {
        const a = byId.get(e.from);
        const b = byId.get(e.to);
        if (!a || !b) return null;
        const { x1, y1, x2, y2 } = edgeLine(a, b);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#999" markerEnd={`url(#${markerId})`} />;
      })}

      {nodes.map((n) => {
        const subY = n.y + NODE_R + 6;
        return (
          <g key={n.id}>
            {n.shape === "circle" ? (
              <circle cx={n.x} cy={n.y} r={NODE_R} fill={n.filled ? "#c2410c" : "white"} stroke="#c2410c" strokeWidth={1.5} />
            ) : (
              <rect x={n.x - SQ / 2} y={n.y - SQ / 2} width={SQ} height={SQ} fill="white" stroke="#64748b" strokeWidth={1.5} />
            )}
            <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={12} fontWeight={600} fill={n.filled ? "white" : "#1a1a1a"}>
              {n.symbol}
            </text>
            {n.formula && (
              <foreignObject x={n.x - 80} y={subY} width={160} height={34}>
                <div className="gm-formula">
                  <MathJax inline dynamic>{`\\(${n.formula}\\)`}</MathJax>
                </div>
              </foreignObject>
            )}
            {n.shape === "square" && !n.formula && (
              <text x={n.x} y={n.y + SQ / 2 + 16} textAnchor="middle" fontSize={10} fill="#888">known</text>
            )}
            {n.filled && (
              <text x={n.x} y={subY + (n.formula ? 38 : 12)} textAnchor="middle" fontSize={10} fill="#888">observed</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Memoized on `stanCode` alone: tabs with static Stan source (all but Get
 *  Started) pass the same string reference on every render, so this never
 *  re-parses or re-typesets — important since those tabs re-render at
 *  animation-frame rates and MathJax typesetting is not cheap. */
export const GraphicalModel = memo(GraphicalModelInner);

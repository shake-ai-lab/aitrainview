// "Flow" layout: the forward pass as a top-to-bottom computation graph.
//
//   * layer 0 at the top, data flows downward; the residual stream is a rail at x = 0
//   * each sub-layer branches off the rail to the left and is added back to it:
//       rail ─▶ norm ─▶ x̂            (row A, flows right→left)
//       Wq_h Wk_h Wv_h │ Q_h K_h V_h │ softmax(QKᵀ)_h │ (A·V)_h   (row B, one slab per head, heads stacked in depth)
//       Wo ─▶ attn out ─▶ (+) rail     (row C)
//       rail ─▶ norm ─▶ x̂₂             (row D)
//       W_gate W_up │ gate up │ act·up │ W_down │ out ─▶ (+) rail   (row E)
//   * weights are panels [out × in]; activations are blocks [channels × tokens]
//     whose cells are filled with real values while the model generates.

import type { Block, Group, Item, Structure, TensorInfo } from '../types';
import {
  as2d,
  compactTower,
  computeBounds,
  type Connector,
  type ExtraLabel,
  FLOOR_POST,
  FLOOR_PRE,
  type FloorInfo,
  type Layout,
  layoutExperts,
  layoutPlates,
  type Local,
  mkW,
  type Placement,
  placeSideTowers,
  size,
  type TowerLayout,
} from './layout';

export const ACT_P = 256; // must match backend runtime.ACT_P
export const ACT_T = 128; // must match backend runtime.ACT_T

const WA = 1.5; // width of an activation block (all tokens)
const SA = 1.3; // side of an attention-pattern square
const G = 0.16; // gap between elements in a row
const RG = 0.6; // gap between rows
const RIGHT = -0.35; // rows end just left of the rail
const RES_X = 0.3; // residual blocks start right of the rail
const STRIP = 0.05;

type V3 = [number, number, number];

function actBlock(key: string, D: number, ref: number, label: string, desc: string, extra: Partial<Local> = {}): Local {
  const h = size(D, ref);
  return {
    set: 'a',
    tile: key,
    tid: -1,
    uv: [0, 0, 1, 1],
    x: 0,
    y: 0,
    z: 0,
    w: WA,
    h,
    flat: false,
    transpose: true,
    ghost: false,
    label,
    key: label,
    P: Math.min(D, ACT_P),
    chan: [0, 1],
    desc,
    ...extra,
  };
}

interface El {
  w: number;
  h: number;
  items: Local[]; // centred at (0,0)
  label?: string;
}

/** A horizontal row of elements, right-aligned at RIGHT, vertically centred. */
class Row {
  els: El[] = [];
  add(el: El): this {
    this.els.push(el);
    return this;
  }
  get width(): number {
    return this.els.reduce((a, e) => a + e.w, 0) + G * Math.max(0, this.els.length - 1);
  }
  get height(): number {
    return Math.max(0, ...this.els.map((e) => e.h));
  }
  /** Returns placed items and per-element centres. */
  place(yTop: number, right = RIGHT): { items: Local[]; centres: { x: number; y: number; w: number; h: number }[]; left: number } {
    const items: Local[] = [];
    const centres: { x: number; y: number; w: number; h: number }[] = [];
    const cy = yTop - this.height / 2;
    let x = right - this.width;
    const left = x;
    for (const e of this.els) {
      const cx = x + e.w / 2;
      for (const it of e.items) items.push({ ...it, x: it.x + cx, y: it.y + cy });
      centres.push({ x: cx, y: cy, w: e.w, h: e.h });
      x += e.w + G;
    }
    return { items, centres, left };
  }
}

const one = (it: Local): El => ({ w: it.w, h: it.h, items: [it] });

function weightEl(T: TensorInfo[], it: Item, ref: number): El {
  const [R, C] = as2d(T[it.t].shape);
  if (T[it.t].kind !== 'matrix') {
    // vectors as vertical strips (one value per channel)
    const h = size(C, ref);
    return one(mkW(it, 0, 0, STRIP, h, { transpose: true }));
  }
  return one(mkW(it, 0, 0, size(C, ref), size(R, ref)));
}

const DOWN_RE = /(down_proj|w2|fc2|c_proj|dense_4h_to_h|out_proj|o_proj|wo)$/;

function normParts(b: Block) {
  const lab = (it: Item) => it.label.replace(/\.bias$/, '');
  const weights = b.norms.filter((it) => !it.label.endsWith('.bias'));
  const hasPreFF = weights.some((it) => /(pre_feedforward|ffn_norm|ln_2|norm2|pre_mlp)/.test(lab(it)));
  const pick = (re: RegExp) => weights.filter((it) => re.test(lab(it)));
  const preAttn = pick(/(input|pre_attention|ln_1|attention_norm|norm1|pre_attn)/);
  const postAttn = hasPreFF ? pick(/post_attention/) : [];
  const preMlp = hasPreFF ? pick(/(pre_feedforward|ffn_norm|ln_2|norm2|pre_mlp)/) : pick(/(post_attention|ffn_norm|ln_2|norm2)/);
  const postMlp = pick(/post_feedforward/);
  const used = new Set([...preAttn, ...postAttn, ...preMlp, ...postMlp]);
  const rest = weights.filter((it) => !used.has(it));
  return { preAttn: preAttn[0], postAttn: postAttn[0], preMlp: preMlp[0], postMlp: postMlp[0], rest };
}

function findQKVO(g: Group, T: TensorInfo[]) {
  const mats = g.items.filter((it) => T[it.t].kind === 'matrix' && !it.attach);
  const last = (it: Item) => it.label.split('.').pop()!;
  const pick = (re: RegExp) => mats.find((it) => re.test(last(it)));
  const q = pick(/^(q_proj|query|wq)$/);
  const k = pick(/^(k_proj|key|wk)$/);
  const v = pick(/^(v_proj|value|wv)$/);
  const o = pick(/^(o_proj|out_proj|wo|dense)$/);
  return q && k && v && o ? { q, k, v, o } : null;
}

interface BlockOut {
  items: Local[];
  height: number;
  conns: Connector[];
  nodes: V3[];
  labels: ExtraLabel[];
  lensY: number;
  minX: number;
}

function flowBlock(S: Structure, b: Block, li: number, ref: number): BlockOut {
  const T = S.tensors;
  const cfg = S.config;
  const items: Local[] = [];
  const conns: Connector[] = [];
  const nodes: V3[] = [];
  const labels: ExtraLabel[] = [];
  const hidden = ref;
  const first = li === 0;
  let y = -0.4;
  let minX = RIGHT;

  const norms = normParts(b);
  const attnGroups = b.groups.filter((g) => g.role === 'attn');
  const otherGroups = b.groups.filter((g) => g.role !== 'attn');

  const branch = (pts: V3[]) => conns.push({ pts, kind: 'branch' });
  const tag = (text: string, x: number, yy: number, z = 0.05, cls = 'item') => {
    if (first) labels.push({ text, cls, pos: [x, yy, z] });
  };
  const rowTitle = (text: string, x: number, yy: number) => labels.push({ text, cls: first ? 'row-title' : 'row-title dim', pos: [x, yy, 0] });

  // residual stream entering the layer
  const res = actBlock(`r:${li}`, hidden, ref, 'residual', `L${li} に入る残差ストリーム`);
  items.push({ ...res, x: RES_X + WA / 2, y: y - res.h / 2 });
  tag('residual', RES_X + WA / 2, y - res.h - 0.12);

  /** rail ─▶ norm ─▶ x̂ ; returns the out point of x̂ (bottom centre) */
  const normRow = (norm: Item | undefined, name: string): { out: V3; bottom: number } | null => {
    if (!norm) return null;
    const row = new Row();
    const xhat = actBlock(`o:${norm.t}`, hidden, ref, name, `${norm.label} の出力（正規化された残差）`);
    row.add(one(xhat)).add(weightEl(T, norm, ref));
    const p = row.place(y);
    items.push(...p.items);
    const cy = p.centres[0].y;
    branch([[0, cy, 0], [RIGHT, cy, 0]]);
    tag(name, p.centres[0].x, cy - xhat.h / 2 - 0.12);
    tag('norm', p.centres[1].x + 0.12, cy + p.centres[1].h / 2 + 0.1);
    minX = Math.min(minX, p.left);
    const bottom = y - row.height;
    y = bottom - RG;
    return { out: [p.centres[0].x, cy - xhat.h / 2, 0], bottom };
  };

  /** connect an out point to the top of the next row's first element */
  const link = (from: V3 | null, toX: number, toTop: number) => {
    if (!from) return;
    const midY = (from[1] + toTop) / 2;
    branch([from, [from[0], midY, 0], [toX, midY, 0], [toX, toTop, 0]]);
  };

  /** last element ─▶ (+) rail */
  const addToRail = (x: number, yy: number) => {
    branch([[x, yy, 0], [0, yy, 0]]);
    nodes.push([0, yy, 0]);
  };

  // ------------------------------------------------------------ attention
  for (const g of attnGroups) {
    const top = normRow(norms.preAttn, 'x̂');
    const qkvo = findQKVO(g, T);
    const H = Number(cfg.num_attention_heads) || 0;
    const Hkv = Number(cfg.num_key_value_heads) || H;
    if (qkvo && H > 0) {
      const { q, k, v, o } = qkvo;
      const [qR, C] = as2d(T[q.t].shape);
      const [kR] = as2d(T[k.t].shape);
      const [vR] = as2d(T[v.t].shape);
      const [, oC] = as2d(T[o.t].shape);
      if (qR % H === 0 && kR % Hkv === 0 && vR % Hkv === 0 && H % Hkv === 0 && oC % H === 0) {
        const grp = H / Hkv;
        const dz = Math.min(0.3, 6 / H);
        const dq = qR / H;
        const dk = kR / Hkv;
        const dv = vR / Hkv;
        const wq = size(C, ref);
        const hq = size(dq, ref);
        const hk = size(dk, ref);
        const hv = size(dv, ref);
        const hO = size(oC / H, ref);
        const colW = [wq, wq, wq, WA, WA, WA, SA, WA];
        const rowH = Math.max(hq, hk, hv, SA, hO);
        const width = colW.reduce((a, b) => a + b, 0) + G * (colW.length - 1);
        const left = RIGHT - width;
        const cx: number[] = [];
        let x = left;
        for (const w of colW) {
          cx.push(x + w / 2);
          x += w + G;
        }
        // heads go back in depth and slightly up (a staircase), so every head stays visible
        const dyH = Math.min(0.06, 0.9 / H);
        const rise = (H - 1) * dyH;
        const cy = y - rise - rowH / 2;
        rowTitle(`ATTENTION · ${H} heads${Hkv !== H ? ` (kv ${Hkv})` : ''}`, left - 0.3, cy);
        for (let h = 0; h < H; h++) {
          const z = -h * dz;
          const hy = cy + h * dyH;
          const j = Math.floor(h / grp);
          const kvHead = h % grp === 0;
          const sl = (i: number, n: number): [number, number] => [i / n, (i + 1) / n];
          const rowsUV = (i: number, n: number): [number, number, number, number] => [0, i / n, 1, (i + 1) / n];
          items.push(mkW(q, cx[0], hy, wq, hq, { z, uv: rowsUV(h, H), head: h, label: `${q.label} · head ${h}`, key: q.label }));
          items.push({ ...actBlock(`o:${q.t}`, qR, ref, 'Q', `Q（head ${h}）`), x: cx[3], y: hy, z, h: hq, chan: sl(h, H), head: h });
          if (kvHead) {
            items.push(mkW(k, cx[1], hy, wq, hk, { z, uv: rowsUV(j, Hkv), head: j, label: `${k.label} · kv head ${j}`, key: k.label }));
            items.push(mkW(v, cx[2], hy, wq, hv, { z, uv: rowsUV(j, Hkv), head: j, label: `${v.label} · kv head ${j}`, key: v.label }));
            items.push({ ...actBlock(`o:${k.t}`, kR, ref, 'K', `K（kv head ${j}）`), x: cx[4], y: hy, z, h: hk, chan: sl(j, Hkv), head: j });
            items.push({ ...actBlock(`o:${v.t}`, vR, ref, 'V', `V（kv head ${j}）`), x: cx[5], y: hy, z, h: hv, chan: sl(j, Hkv), head: j });
          }
          items.push({
            ...actBlock(`A:${li}:${h}`, ACT_T, ref, 'softmax(QKᵀ)', `Attention の重み（head ${h}）行=query, 列=key`),
            x: cx[6], y: hy, z, w: SA, h: SA, transpose: false, attn: true, head: h, P: ACT_T,
          });
          items.push({ ...actBlock(`i:${o.t}`, oC, ref, 'A·V', `head ${h} の出力（Attention で重み付けした V）`), x: cx[7], y: hy, z, h: hO, chan: sl(h, H), head: h });
        }
        // vectors of the group (q_norm, k_norm, biases, sinks …) as strips in front of head 0
        g.items
          .filter((it) => T[it.t].kind !== 'matrix')
          .forEach((it, n) => {
            const col = it.attach?.to === k.t ? 1 : it.attach?.to === v.t ? 2 : 0;
            const [, len] = as2d(T[it.t].shape);
            const hh = size(len, ref);
            items.push(mkW(it, cx[col] + colW[col] / 2 - 0.05 - n * 0.08, cy, STRIP, hh, { z: dz * 1.2, transpose: true }));
          });
        const names = ['Wq', 'Wk', 'Wv', 'Q', 'K', 'V', 'softmax(QKᵀ/√d)', 'A·V'];
        names.forEach((nm, i) => tag(nm, cx[i], cy - rowH / 2 - 0.14, 0.05));
        link(top?.out ?? null, cx[0], y);
        minX = Math.min(minX, left);
        const outB: V3 = [cx[7], cy - rowH / 2, 0];
        y = y - rowH - rise - RG;

        // Wo ─▶ attention output ─▶ rail
        const rowC = new Row();
        rowC.add(weightEl(T, o, ref));
        const out = actBlock(`o:${o.t}`, hidden, ref, 'attn out', 'Attention 出力（Wo を掛けた後）');
        rowC.add(one(out));
        if (norms.postAttn) {
          rowC.add(weightEl(T, norms.postAttn, ref));
          rowC.add(one(actBlock(`o:${norms.postAttn.t}`, hidden, ref, 'norm out', `${norms.postAttn.label} の出力`)));
        }
        const pc = rowC.place(y);
        items.push(...pc.items);
        link(outB, pc.centres[0].x, y);
        tag('Wo', pc.centres[0].x, pc.centres[0].y - pc.centres[0].h / 2 - 0.14);
        tag('attn out', pc.centres[1].x, pc.centres[1].y - pc.centres[1].h / 2 - 0.14);
        addToRail(RIGHT, pc.centres[0].y);
        minX = Math.min(minX, pc.left);
        y = y - rowC.height - RG;
        continue;
      }
    }
    // generic attention (fused qkv, MLA, linear attention …)
    const r = genericRow(S, g, ref, y);
    rowTitle(`ATTENTION · ${g.name}`, r.left - 0.3, (y + r.bottom) / 2);
    items.push(...r.items);
    labels.push(...(first ? r.labels : []));
    link(top?.out ?? null, r.inX, y);
    addToRail(RIGHT, r.outY);
    minX = Math.min(minX, r.left);
    y = r.bottom - RG;
  }

  // ------------------------------------------------------------------ MLP
  if (otherGroups.length) {
    const midRes = norms.preMlp
      ? actBlock(`i:${norms.preMlp.t}`, hidden, ref, 'residual', '残差ストリーム（Attention を足した後）')
      : null;
    if (midRes) {
      items.push({ ...midRes, x: RES_X + WA / 2, y: y - midRes.h / 2 });
      tag('residual', RES_X + WA / 2, y - midRes.h - 0.12);
    }
    const top = normRow(norms.preMlp, 'x̂₂');
    for (const g of otherGroups) {
      const r = genericRow(S, g, ref, y, norms.postMlp);
      rowTitle(g.role === 'mlp' ? `MLP · ${g.name}` : g.name, r.left - 0.3, (y + r.bottom) / 2);
      items.push(...r.items);
      labels.push(...(first ? r.labels : []));
      link(top?.out ?? null, r.inX, y);
      addToRail(RIGHT, r.outY);
      minX = Math.min(minX, r.left);
      y = r.bottom - RG;
    }
  }

  // leftover norms (unusual architectures) as strips next to the rail
  norms.rest.forEach((it, n) => {
    const [, len] = as2d(T[it.t].shape);
    items.push(mkW(it, RIGHT - 0.1 - n * 0.1, y - size(len, ref) / 2, STRIP, size(len, ref), { transpose: true }));
  });

  // residual leaving the layer
  const outRes = actBlock(`ro:${li}`, hidden, ref, 'residual out', `L${li} から出る残差ストリーム`);
  const lensY = y - outRes.h / 2;
  items.push({ ...outRes, x: RES_X + WA / 2, y: lensY });
  y -= outRes.h + 0.35;
  return { items, height: -y, conns, nodes, labels, lensY, minX };
}

interface GenericRow {
  items: Local[];
  labels: ExtraLabel[];
  inX: number;
  outY: number;
  left: number;
  bottom: number;
}

/** W_up-type matrices, their outputs, then (input → W_down → output); experts as a grid. */
function genericRow(S: Structure, g: Group, ref: number, y: number, postNorm?: Item): GenericRow {
  const T = S.tensors;
  const mats = g.items.filter((it) => T[it.t].kind === 'matrix' && !it.attach);
  const vecs = g.items.filter((it) => T[it.t].kind !== 'matrix');
  const last = (it: Item) => it.label.split('.').pop()!;
  const downs = mats.filter((it) => DOWN_RE.test(last(it)));
  const ups = mats.filter((it) => !downs.includes(it));
  const row = new Row();
  const names: string[] = [];
  for (const m of ups) {
    row.add(weightEl(T, m, ref));
    names.push(last(m));
  }
  for (const m of ups) {
    const [R] = as2d(T[m.t].shape);
    row.add(one(actBlock(`o:${m.t}`, R, ref, last(m).replace('_proj', ''), `${m.label} の出力`)));
    names.push(last(m).replace('_proj', ''));
  }
  for (const m of downs) {
    const [R, C] = as2d(T[m.t].shape);
    row.add(one(actBlock(`i:${m.t}`, C, ref, 'act', `${m.label} への入力（活性化関数 × up）`)));
    names.push(ups.length >= 2 ? 'act(gate)·up' : 'act');
    row.add(weightEl(T, m, ref));
    names.push(last(m));
    row.add(one(actBlock(`o:${m.t}`, R, ref, 'out', `${m.label} の出力`)));
    names.push('out');
  }
  if (g.experts.length) {
    const grid = layoutExperts(g, ref, T, 0, 1, 0, 2.4);
    const w = Math.max(0.1, grid.cursor - G);
    const h = grid.top;
    row.add({ w, h, items: grid.items.map((it) => ({ ...it, x: it.x - w / 2, y: it.y - h / 2 })) });
    names.push(`experts ×${g.experts.length}`);
  }
  for (const v of vecs) {
    row.add(weightEl(T, v, ref));
    names.push('');
  }
  if (postNorm) {
    row.add(weightEl(T, postNorm, ref));
    names.push(postNorm.label);
    row.add(one(actBlock(`o:${postNorm.t}`, ref, ref, 'norm out', `${postNorm.label} の出力`)));
    names.push('norm out');
  }
  if (!row.els.length) row.add({ w: 0.2, h: 0.2, items: [] });
  const p = row.place(y);
  const labels: ExtraLabel[] = names
    .map((t, i) => ({ t, c: p.centres[i] }))
    .filter((e) => e.t && e.c)
    .map((e) => ({ text: e.t, cls: 'item', pos: [e.c.x, e.c.y - e.c.h / 2 - 0.14, 0.05] as V3 }));
  return {
    items: p.items,
    labels,
    inX: p.centres[0].x,
    outY: p.centres[p.centres.length - 1].y,
    left: p.left,
    bottom: y - row.height,
  };
}

export function computeFlowLayout(S: Structure): Layout {
  const T = S.tensors;
  const placements: Placement[] = [];
  const connectors: Connector[] = [];
  const labels: ExtraLabel[] = [];
  const nodes: V3[] = [];
  const tw = S.towers[0];
  const ref = tw.ref_dim || 1024;

  const blocks = tw.blocks.map((b, i) => flowBlock(S, b, i, ref));
  const layerH = Math.max(2, ...blocks.map((b) => b.height));
  const floors: FloorInfo[] = [];
  let minX = RIGHT;
  blocks.forEach((bo, i) => {
    const top = -i * layerH;
    floors.push({ index: tw.blocks[i].index, y: top - layerH, h: layerH, layerType: tw.blocks[i].layer_type, lensY: top + bo.lensY });
    for (const it of bo.items) placements.push({ ...it, y: it.y + top, tower: 0, floor: i, showLabel: false });
    for (const c of bo.conns) connectors.push({ ...c, pts: c.pts.map(([x, y, z]) => [x, y + top, z] as V3) });
    for (const n of bo.nodes) nodes.push([n[0], n[1] + top, n[2]]);
    for (const l of bo.labels) labels.push({ ...l, pos: [l.pos[0], l.pos[1] + top, l.pos[2]] });
    minX = Math.min(minX, bo.minX);
  });
  const bottom = -blocks.length * layerH;

  // input side: embedding plate above the stack + its output on the rail
  const embMat = tw.pre.filter((it) => T[it.t].kind === 'matrix');
  let railTop = 0.4;
  if (embMat.length) {
    const e = embMat[0];
    const [, C] = as2d(T[e.t].shape);
    const emb = actBlock(`o:${e.t}`, C, ref, 'embedding', '埋め込みベクトル（各トークンの初期表現）');
    placements.push({ ...emb, x: RES_X + WA / 2, y: 0.4 + emb.h / 2, tower: 0, floor: FLOOR_PRE, showLabel: false });
    labels.push({ text: 'tokens → embedding', cls: 'item', pos: [RES_X + WA / 2, 0.4 + emb.h + 0.15, 0.05] });
    for (const p of layoutPlates(tw.pre, ref, T, 0.4 + emb.h + 0.6, null)) placements.push({ ...p, tower: 0, floor: FLOOR_PRE, showLabel: true });
    railTop = 0.4 + emb.h / 2;
  }
  tw.pre
    .filter((it) => T[it.t].kind !== 'matrix')
    .forEach((it, k) => {
      const [, n] = as2d(T[it.t].shape);
      placements.push({ ...mkW(it, RIGHT - 0.1 - k * 0.1, 0.4 + size(n, ref) / 2, STRIP, size(n, ref), { transpose: true }), tower: 0, floor: FLOOR_PRE, showLabel: true });
    });

  // output side: final norm, its output, lm_head plate
  let y = bottom - 0.2;
  const postVec = tw.post.filter((it) => T[it.t].kind !== 'matrix');
  if (postVec.length) {
    const fn = postVec[0];
    const row = new Row();
    const out = actBlock(`o:${fn.t}`, ref, ref, 'final norm', '最終正規化の出力（lm_head への入力）');
    row.add(one(out));
    for (const v of postVec) row.add(weightEl(T, v, ref));
    const p = row.place(y);
    for (const it of p.items) placements.push({ ...it, tower: 0, floor: FLOOR_POST, showLabel: false });
    connectors.push({ pts: [[0, p.centres[0].y, 0], [RIGHT, p.centres[0].y, 0]], kind: 'branch' });
    labels.push({ text: 'final norm', cls: 'item', pos: [p.centres[0].x, p.centres[0].y - out.h / 2 - 0.14, 0.05] });
    y -= row.height + 0.5;
  }
  const plates = layoutPlates(tw.post, ref, T, y - 0.4, tw.tied_head);
  for (const p of plates) placements.push({ ...p, tower: 0, floor: FLOOR_POST, showLabel: true });
  const railBottom = y - 0.4;

  connectors.push({ pts: [[0, railTop, 0], [0, railBottom, 0]], kind: 'rail' });

  let xMin = Infinity;
  let xMax = -Infinity;
  for (const p of placements) {
    xMin = Math.min(xMin, p.x - p.w / 2);
    xMax = Math.max(xMax, p.x + p.w / 2);
  }
  const towers: TowerLayout[] = [{
    index: 0,
    name: tw.name,
    x0: 0,
    xMin,
    xMax,
    yTop: railTop,
    floorH: layerH,
    floors,
    railX: 0,
    labelX: minX - 0.6,
    lensX: RES_X + WA + 0.35,
    downward: true,
  }];

  // secondary towers (vision / audio): compact, to the left, top aligned
  for (let ti = 1; ti < S.towers.length; ti++) {
    const r = compactTower(S, ti);
    const dy = -r.tower.yTop;
    for (const p of r.placements) placements.push({ ...p, y: p.y + dy });
    r.tower.floors = r.tower.floors.map((f) => ({ ...f, y: f.y + dy, lensY: (f.lensY ?? f.y) + dy }));
    r.tower.yTop += dy;
    towers.push(r.tower);
  }
  placeSideTowers(towers, placements);
  return { kind: 'flow', placements, towers, connectors, labels, nodes, ...computeBounds(placements) };
}

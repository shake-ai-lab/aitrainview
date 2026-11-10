// Shared placement types + the compact "tower" layout (weights only).
//
// Compact layout:
//   * every decoder layer is a floor; floors stack up along +Y
//   * the residual stream is a vertical spine at x = 0; norms sit on it
//   * attention matrices to the left of the spine, MLP / others to the right
//   * a matrix [out, in] becomes a panel with height ∝ √out and width ∝ √in

import type { Group, Item, Structure, TensorInfo } from '../types';

export type UV = [number, number, number, number]; // c0, r0, c1, r1 as fractions of the tile

export interface Placement {
  set: 'w' | 'a'; // weight tiles or activation tiles
  tile: string; // weights: `${tid}`; activations: capture key (o:12, i:12, r:3, A:3:5 ...)
  tid: number; // tensor id (-1 for pure activations)
  uv: UV;
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  flat: boolean; // lies on the XZ plane
  transpose: boolean; // swap texture axes
  ghost: boolean; // tied weight drawn a second time
  label: string;
  key: string; // activity normalisation class
  tower: number;
  floor: number; // block position in tower; FLOOR_PRE / FLOOR_POST
  expert?: number;
  head?: number;
  showLabel: boolean;
  // activations
  attn?: boolean; // attention probabilities (T x T)
  P?: number; // pooled channels stored in the tile
  chan?: [number, number]; // channel fraction range shown (per-head slice)
  desc?: string; // human readable description for tooltips
}

export interface FloorInfo {
  index: number;
  y: number; // bottom
  h: number;
  layerType?: string | null;
  lensY?: number;
}

export interface Connector {
  pts: [number, number, number][];
  kind: 'rail' | 'branch';
}

export interface TowerLayout {
  index: number;
  name: string;
  x0: number;
  xMin: number;
  xMax: number;
  yTop: number;
  floorH: number;
  floors: FloorInfo[];
  railX: number;
  labelX?: number;
  lensX?: number;
  downward?: boolean; // flow layout: layer 0 at the top
}

export interface ExtraLabel {
  text: string;
  cls: string;
  pos: [number, number, number];
}

export interface Layout {
  kind: 'flow' | 'tower';
  placements: Placement[];
  towers: TowerLayout[];
  connectors: Connector[];
  labels: ExtraLabel[];
  nodes: [number, number, number][]; // residual "+" nodes
  min: [number, number, number];
  max: [number, number, number];
}

export const FLOOR_PRE = -1;
export const FLOOR_POST = 1e6;

const GAP = 0.12;
const GROUP_GAP = 0.5;
const Y0 = 0.3;
const SPINE_GAP = 0.8;
const BAR = 0.05;

export function as2d(shape: number[]): [number, number] {
  if (shape.length === 0) return [1, 1];
  if (shape.length === 1) return [1, shape[0]];
  if (shape.length === 2) return [shape[0], shape[1]];
  if (shape.length === 3) return [shape[0] * shape[1], shape[2]];
  return [shape[0], shape.slice(1).reduce((a, b) => a * b, 1)];
}

export const size = (d: number, ref: number) => Math.min(10, Math.max(0.05, Math.sqrt(d / ref)));

export type Local = Omit<Placement, 'tower' | 'floor' | 'showLabel'>;

export function mkW(it: Item | { t: number; label: string }, x: number, y: number, w: number, h: number, extra: Partial<Local> = {}): Local {
  return {
    set: 'w',
    tile: String(it.t),
    tid: it.t,
    uv: [0, 0, 1, 1],
    x,
    y,
    z: 0,
    w,
    h,
    flat: false,
    transpose: false,
    ghost: false,
    label: it.label,
    key: it.label,
    ...extra,
  };
}

function layoutBlock(b: Structure['towers'][number]['blocks'][number], ref: number, T: TensorInfo[]) {
  const out: Local[] = [];
  let top = 0;
  b.norms.forEach((it, k) => {
    const [, n] = as2d(T[it.t].shape);
    out.push(mkW(it, 0, 0.08 + k * 0.1 + BAR / 2, size(n, ref), BAR, { z: 0.06 }));
    top = Math.max(top, 0.08 + k * 0.1 + BAR);
  });

  const placeGroup = (g: Group, cursor: number, dir: number): number => {
    const side = new Map<number, Item[]>();
    const below = new Map<number, Item[]>();
    const mats: Item[] = [];
    const loose: Item[] = [];
    for (const it of g.items) {
      if (it.attach) {
        const m = it.attach.how === 'side' ? side : below;
        if (!m.has(it.attach.to)) m.set(it.attach.to, []);
        m.get(it.attach.to)!.push(it);
      } else if (T[it.t].kind === 'matrix') mats.push(it);
      else loose.push(it);
    }
    for (const m of dir < 0 ? [...mats].reverse() : mats) {
      const [R, C] = as2d(T[m.t].shape);
      const w = size(C, ref);
      const h = size(R, ref);
      const sides = side.get(m.t) ?? [];
      const tw = w + sides.length * 0.09;
      const x0 = dir < 0 ? cursor - tw : cursor;
      out.push(mkW(m, x0 + w / 2, Y0 + h / 2, w, h));
      top = Math.max(top, Y0 + h);
      sides.forEach((s, k) => out.push(mkW(s, x0 + w + 0.055 + k * 0.09, Y0 + h / 2, 0.045, h, { transpose: true })));
      (below.get(m.t) ?? []).forEach((s, k) => {
        const [, n] = as2d(T[s.t].shape);
        out.push(mkW(s, x0 + w / 2, Y0 - 0.09 - k * 0.08, Math.min(w, size(n, ref)), 0.045));
      });
      cursor = dir < 0 ? x0 - GAP : x0 + tw + GAP;
    }
    for (const v of loose) {
      const [, n] = as2d(T[v.t].shape);
      const w = T[v.t].kind === 'scalar' ? 0.08 : size(n, ref);
      const x0 = dir < 0 ? cursor - w : cursor;
      out.push(mkW(v, x0 + w / 2, Y0 + 0.05, w, T[v.t].kind === 'scalar' ? 0.08 : 0.045));
      cursor = dir < 0 ? x0 - GAP : x0 + w + GAP;
    }
    const grid = layoutExperts(g, ref, T, cursor, dir, Y0, 2.6);
    out.push(...grid.items);
    top = Math.max(top, grid.top);
    return grid.cursor;
  };

  let left = -SPINE_GAP;
  let right = SPINE_GAP;
  for (const g of b.groups.filter((g) => g.role === 'attn')) left = placeGroup(g, left, -1) - GROUP_GAP;
  for (const g of b.groups.filter((g) => g.role !== 'attn')) right = placeGroup(g, right, +1) + GROUP_GAP;
  return { items: out, height: top };
}

/** MoE experts: a grid of small panel triples, bottom edge at y0, max height maxH. */
export function layoutExperts(g: Group, ref: number, T: TensorInfo[], cursor: number, dir: number, y0: number, maxH: number) {
  const items: Local[] = [];
  let top = y0;
  const E = g.experts.map((l) => l.filter((it) => T[it.t].kind === 'matrix')).filter((l) => l.length);
  if (!E.length) return { items, top, cursor };
  const cells = E.map((list) => {
    let w = 0;
    let h = 0;
    for (const it of list) {
      const [R, C] = as2d(T[it.t].shape);
      w += size(C, ref) + GAP;
      h = Math.max(h, size(R, ref));
    }
    return { w: w - GAP, h };
  });
  const cw = Math.max(...cells.map((c) => c.w));
  const ch = Math.max(...cells.map((c) => c.h));
  const n = E.length;
  const cols = Math.max(1, Math.ceil(Math.sqrt((3 * n * ch) / cw)));
  const rows = Math.ceil(n / cols);
  const k = Math.min(1, maxH / (rows * ch * 1.15));
  const cellW = cw * k * 1.1;
  const cellH = ch * k * 1.15;
  E.forEach((list, e) => {
    const r = Math.floor(e / cols);
    const c = e % cols;
    let x = dir < 0 ? cursor - (c + 1) * cellW : cursor + c * cellW;
    for (const it of list) {
      const [R, C] = as2d(T[it.t].shape);
      const w = size(C, ref) * k;
      const hh = size(R, ref) * k;
      items.push(mkW(it, x + w / 2, y0 + r * cellH + hh / 2, w, hh, { expert: e, key: `experts.${it.label}` }));
      x += w + GAP * k;
    }
    top = Math.max(top, y0 + (r + 1) * cellH);
  });
  cursor = dir < 0 ? cursor - cols * cellW - GAP : cursor + cols * cellW + GAP;
  return { items, top, cursor };
}

/** Flat plates (embeddings / lm_head) laid side by side along X. */
export function layoutPlates(items: Item[], ref: number, T: TensorInfo[], y: number, ghostTid: number | null, z = 0): Local[] {
  const plates: Local[] = [];
  const mats = items.filter((it) => T[it.t].kind === 'matrix');
  const list = ghostTid !== null ? [...mats, { t: ghostTid, label: 'lm_head (tied)', kind: 'matrix', module: '' } as Item] : mats;
  const dims = list.map((it) => {
    const [R, C] = as2d(T[it.t].shape);
    return { it, R, C, w: size(Math.max(R, C), ref), d: size(Math.min(R, C), ref) };
  });
  const total = dims.reduce((a, d) => a + d.w, 0) + GAP * 4 * Math.max(0, dims.length - 1);
  let x = -total / 2;
  for (const d of dims) {
    const tied = d.it.label.includes('tied');
    plates.push(
      mkW(d.it, x + d.w / 2, y, d.w, d.d, { z, flat: true, transpose: d.R > d.C, ghost: tied, key: tied ? 'head' : d.it.label }),
    );
    x += d.w + GAP * 4;
  }
  return plates;
}

export function computeBounds(placements: Placement[]): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const p of placements) {
    const hx = p.w / 2;
    const hy = p.flat ? 0 : p.h / 2;
    const hz = p.flat ? p.h / 2 : 0;
    min[0] = Math.min(min[0], p.x - hx);
    max[0] = Math.max(max[0], p.x + hx);
    min[1] = Math.min(min[1], p.y - hy);
    max[1] = Math.max(max[1], p.y + hy);
    min[2] = Math.min(min[2], p.z - hz);
    max[2] = Math.max(max[2], p.z + hz);
  }
  if (!placements.length) {
    min.fill(-1);
    max.fill(1);
  }
  return { min, max };
}

/** Compact weights-only tower for one tower of the model (local coordinates). */
export function compactTower(S: Structure, ti: number): { placements: Placement[]; tower: TowerLayout } {
  const T = S.tensors;
  const tw = S.towers[ti];
  const ref = tw.ref_dim || 1024;
  const blocks = tw.blocks.map((b) => layoutBlock(b, ref, T));
  const floorH = Math.max(1, ...blocks.map((b) => b.height)) + 0.55;
  const local: Placement[] = [];
  const floors: FloorInfo[] = [];
  blocks.forEach((bl, i) => {
    const y = i * floorH;
    floors.push({ index: tw.blocks[i].index, y, h: floorH, layerType: tw.blocks[i].layer_type, lensY: y + floorH * 0.45 });
    for (const p of bl.items) local.push({ ...p, y: p.y + y, tower: ti, floor: i, showLabel: i === 0 && p.expert === undefined });
  });
  const yTop = blocks.length * floorH;
  for (const p of layoutPlates(tw.pre, ref, T, -0.3, null)) local.push({ ...p, tower: ti, floor: FLOOR_PRE, showLabel: true });
  tw.pre
    .filter((it) => T[it.t].kind !== 'matrix')
    .forEach((it, k) => {
      const [, n] = as2d(T[it.t].shape);
      local.push({ ...mkW(it, 0, -0.12 + k * 0.08, size(n, ref), BAR, { z: 0.06 }), tower: ti, floor: FLOOR_PRE, showLabel: true });
    });
  const postVec = tw.post.filter((it) => T[it.t].kind !== 'matrix');
  postVec.forEach((it, k) => {
    const [, n] = as2d(T[it.t].shape);
    local.push({ ...mkW(it, 0, yTop + 0.1 + k * 0.1, size(n, ref), BAR, { z: 0.06 }), tower: ti, floor: FLOOR_POST, showLabel: true });
  });
  const roofY = yTop + 0.25 + postVec.length * 0.1;
  for (const p of layoutPlates(tw.post, ref, T, roofY, tw.tied_head)) local.push({ ...p, tower: ti, floor: FLOOR_POST, showLabel: true });

  let xMin = Infinity;
  let xMax = -Infinity;
  for (const p of local) {
    xMin = Math.min(xMin, p.x - p.w / 2);
    xMax = Math.max(xMax, p.x + p.w / 2);
  }
  return {
    placements: local,
    tower: { index: ti, name: tw.name, x0: 0, xMin, xMax, yTop, floorH, floors, railX: 0, labelX: xMin - 0.35, lensX: xMax + 0.4 },
  };
}

/** Place secondary towers (vision / audio) to the left of tower 0. */
export function placeSideTowers(towers: TowerLayout[], placements: Placement[]): void {
  let cursor = towers.length ? towers[0].xMin - 4 : 0;
  for (let i = 1; i < towers.length; i++) {
    const t = towers[i];
    const dx = cursor - t.xMax;
    t.x0 = dx;
    t.xMin += dx;
    t.xMax += dx;
    t.railX += dx;
    if (t.labelX !== undefined) t.labelX += dx;
    if (t.lensX !== undefined) t.lensX += dx;
    for (const p of placements) if (p.tower === i) p.x += dx;
    cursor = t.xMin - 4;
  }
}

export function computeLayout(S: Structure): Layout {
  const placements: Placement[] = [];
  const towers: TowerLayout[] = [];
  S.towers.forEach((_, ti) => {
    const r = compactTower(S, ti);
    towers.push(r.tower);
    placements.push(...r.placements);
  });
  placeSideTowers(towers, placements);
  return { kind: 'tower', placements, towers, connectors: [], labels: [], nodes: [], ...computeBounds(placements) };
}

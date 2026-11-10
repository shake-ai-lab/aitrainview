// Renders the model: weight panels, activation blocks, connectors, labels.
//
// Panels are InstancedMesh chunks of up to 256 instances. Each chunk owns a
// DataArrayTexture; a tile (one weight thumbnail or one activation matrix) gets
// one layer per chunk that uses it, so per-head slices of the same matrix share
// a layer and only differ in their uv rectangle.

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { Structure } from '../types';
import { showToken } from '../util';
import { buildLutTexture } from './colormaps';
import { ACT_P, ACT_T, computeFlowLayout } from './flowLayout';
import { computeLayout, FLOOR_POST, FLOOR_PRE, type Layout, type Placement } from './layout';

const CHUNK = 256;

const VERT = /* glsl */ `
attribute float aLayer;
attribute float aFlags;
attribute float aAct;
attribute float aScale;
attribute float aSel;
attribute vec4 aUv;
varying vec2 vUv;
varying vec4 vRect;
varying float vLayer;
varying float vFlags;
varying float vAct;
varying float vScale;
varying float vSel;
void main() {
  vUv = uv;
  vRect = aUv;
  vLayer = aLayer;
  vFlags = aFlags;
  vAct = aAct;
  vScale = aScale;
  vSel = aSel;
  vec4 p = vec4(position, 1.0);
#ifdef USE_INSTANCING
  p = instanceMatrix * p;
#endif
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * p;
}
`;

const FRAG = /* glsl */ `
precision highp float;
precision highp sampler2DArray;
uniform sampler2DArray uAtlas;
uniform sampler2DArray uDelta;
uniform sampler2D uLut;
uniform int uSet;        // 0 = weights, 1 = activations
uniform float uRowW;     // weight colormap row
uniform float uRowD;     // delta colormap row
uniform float uRowA;     // activation colormap row
uniform float uRowP;     // attention colormap row
uniform int uView;       // 0 = weights, 1 = delta
uniform float uBright;
uniform float uDim;
uniform float uDeltaGain;
uniform float uFill;     // how much the glow fills the panel (borders always glow)
uniform vec3 uGlow;
varying vec2 vUv;
varying vec4 vRect;
varying float vLayer;
varying float vFlags;
varying float vAct;
varying float vScale;
varying float vSel;
void main() {
  int f = int(vFlags + 0.5);
  vec2 base = ((f & 1) != 0) ? vec2(1.0 - vUv.y, vUv.x) : vec2(vUv.x, 1.0 - vUv.y);
  vec2 st = mix(vRect.xy, vRect.zw, base);
  float w = texture(uAtlas, vec3(st, vLayer)).r;
  vec3 col;
  if (uSet == 1) {
    if (vScale < 0.5) {
      float s = step(0.5, fract((vUv.x + vUv.y) * 18.0));
      col = vec3(0.018, 0.024, 0.034) + vec3(0.008) * s;
    } else if ((f & 16) != 0) {
      col = texture(uLut, vec2(w, uRowP)).rgb * 0.95;
    } else {
      col = texture(uLut, vec2(w, uRowA)).rgb * 0.85;
    }
  } else if (uView == 1) {
    float d = texture(uDelta, vec3(st, vLayer)).r;
    float dv = clamp((d - 0.5) * 2.0 * vScale * uDeltaGain, -1.0, 1.0);
    col = texture(uLut, vec2(dv * 0.5 + 0.5, uRowD)).rgb + texture(uLut, vec2(w, uRowW)).rgb * 0.05;
  } else {
    col = texture(uLut, vec2(w, uRowW)).rgb * uBright;
  }
  if ((f & 8) != 0) col = vec3(0.05, 0.05, 0.06);
  if ((f & 2) != 0) col *= 0.35;
  vec2 fw = max(fwidth(vUv), vec2(1e-5));
  vec2 e = min(vUv, 1.0 - vUv) / fw;
  float border = 1.0 - smoothstep(0.0, 1.6, min(e.x, e.y));
  if (uSet == 0) col *= uDim;
  col += uGlow * vAct * (uFill + 1.6 * border);
  vec3 edge = uSet == 1 ? vec3(0.35, 0.8, 0.7) : vec3(0.62, 0.74, 0.95);
  col = mix(col, edge, border * (0.2 + 0.7 * vSel));
  col += vec3(0.25, 0.35, 0.55) * vSel * 0.2;
  gl_FragColor = vec4(col, 1.0);
}
`;

interface Chunk {
  mesh: THREE.InstancedMesh;
  atlas: THREE.DataArrayTexture;
  delta: THREE.DataArrayTexture | null;
  members: number[];
  aAct: THREE.InstancedBufferAttribute;
  aScale: THREE.InstancedBufferAttribute;
  aSel: THREE.InstancedBufferAttribute;
  aUv: THREE.InstancedBufferAttribute;
}

function makeTex(w: number, h: number, layers: number, fill: number): THREE.DataArrayTexture {
  const tex = new THREE.DataArrayTexture(new Uint8Array(w * h * layers).fill(fill), w, h, layers);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

/** All panels that sample the same kind of tile (weights S×S, or activations ACT_P×ACT_T). */
class PanelSet {
  readonly chunks: Chunk[] = [];
  readonly tiles = new Map<string, Array<[Chunk, number]>>(); // tile key → (chunk, layer)
  constructor(
    readonly kind: 'w' | 'a',
    members: number[],
    P: Placement[],
    readonly tw: number,
    readonly th: number,
    uniforms: Record<string, THREE.IUniform>,
    root: THREE.Group,
    flagsOf: (p: Placement) => number,
    chunkIndex: Int32Array,
    instIndex: Int32Array,
  ) {
    const flat = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
    const upright = new THREE.Quaternion();
    const m = new THREE.Matrix4();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    for (let start = 0; start < members.length; start += CHUNK) {
      const mem = members.slice(start, start + CHUNK);
      const count = mem.length;
      const layerOf = new Map<string, number>();
      for (const pi of mem) if (!layerOf.has(P[pi].tile)) layerOf.set(P[pi].tile, layerOf.size);
      const atlas = makeTex(tw, th, layerOf.size, 0);
      const delta = kind === 'w' ? makeTex(tw, th, layerOf.size, 128) : null;
      const geo = new THREE.PlaneGeometry(1, 1);
      const layer = new Float32Array(count);
      const flags = new Float32Array(count);
      const uvs = new Float32Array(count * 4);
      mem.forEach((pi, i) => {
        layer[i] = layerOf.get(P[pi].tile)!;
        flags[i] = flagsOf(P[pi]);
        uvs.set(P[pi].uv, i * 4);
      });
      geo.setAttribute('aLayer', new THREE.InstancedBufferAttribute(layer, 1));
      geo.setAttribute('aFlags', new THREE.InstancedBufferAttribute(flags, 1));
      const aAct = new THREE.InstancedBufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
      const aScale = new THREE.InstancedBufferAttribute(new Float32Array(count).fill(kind === 'w' ? 1 : 0), 1);
      const aSel = new THREE.InstancedBufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
      const aUv = new THREE.InstancedBufferAttribute(uvs, 4);
      geo.setAttribute('aAct', aAct);
      geo.setAttribute('aScale', aScale);
      geo.setAttribute('aSel', aSel);
      geo.setAttribute('aUv', aUv);
      const mat = new THREE.ShaderMaterial({
        uniforms: { ...uniforms, uSet: { value: kind === 'w' ? 0 : 1 }, uAtlas: { value: atlas }, uDelta: { value: delta ?? atlas } },
        vertexShader: VERT,
        fragmentShader: FRAG,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.InstancedMesh(geo, mat, count);
      mem.forEach((pi, i) => {
        const p = P[pi];
        pos.set(p.x, p.y, p.z);
        scl.set(p.w, p.h, 1);
        m.compose(pos, p.flat ? flat : upright, scl);
        mesh.setMatrixAt(i, m);
        instIndex[pi] = i;
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      const chunk: Chunk = { mesh, atlas, delta, members: mem, aAct, aScale, aSel, aUv };
      mesh.userData.set = kind;
      mesh.userData.chunk = this.chunks.length;
      for (const pi of mem) chunkIndex[pi] = this.chunks.length;
      for (const [key, l] of layerOf) {
        if (!this.tiles.has(key)) this.tiles.set(key, []);
        this.tiles.get(key)!.push([chunk, l]);
      }
      root.add(mesh);
      this.chunks.push(chunk);
    }
  }

  /** Write rows [r0, r0+nr) of a tile; each source row has `width` bytes. */
  writeRows(key: string, r0: number, nr: number, src: Uint8Array, offset: number, width: number, target: 'atlas' | 'delta', dirty: Set<THREE.DataArrayTexture>): boolean {
    const dst = this.tiles.get(key);
    if (!dst) return false;
    const tileBytes = this.tw * this.th;
    const w = Math.min(width, this.tw);
    for (const [c, l] of dst) {
      const tex = target === 'atlas' ? c.atlas : c.delta;
      if (!tex) continue;
      const buf = tex.image.data as Uint8Array;
      for (let r = 0; r < nr && r0 + r < this.th; r++) {
        buf.set(src.subarray(offset + r * width, offset + r * width + w), l * tileBytes + (r0 + r) * this.tw);
      }
      tex.addLayerUpdate(l);
      dirty.add(tex);
    }
    return true;
  }

  dispose(): void {
    for (const c of this.chunks) {
      c.mesh.geometry.dispose();
      (c.mesh.material as THREE.Material).dispose();
      c.atlas.dispose();
      c.delta?.dispose();
    }
  }
}

export interface Pulse {
  values: Float32Array; // per placement, 0..1
  direction: 'forward' | 'backward';
  duration: number;
}

export interface PickResult {
  idx: number;
  uv: THREE.Vector2 | null;
}

export class ModelView {
  readonly root = new THREE.Group();
  readonly layout: Layout;
  readonly tile: number;
  readonly structure: Structure;
  readonly uniforms: Record<string, THREE.IUniform>;
  private w: PanelSet;
  private a: PanelSet;
  private setOf: Uint8Array; // 0 = w, 1 = a
  private chunkIndex: Int32Array;
  private instIndex: Int32Array;
  private slots = new Map<number, number[]>(); // tid -> weight placement indices
  private n: number;
  private cur: Float32Array;
  private flash: Float32Array;
  private pendVal: Float32Array;
  private pendAt: Float32Array;
  private pend: Uint8Array;
  private animating = false;
  private lutRow: (name: string) => number;
  private spine: { mesh: THREE.Mesh; mat: THREE.MeshBasicMaterial; cur: number; pend: number; at: number; flash: number }[] = [];
  private floorLabels: CSS2DObject[] = [];
  private lensLabels: CSS2DObject[] = [];
  private itemLabels: CSS2DObject[] = [];
  private hoverIdx = -1;
  private selIdx = -1;
  private deltaScales = new Map<number, number>();
  private deltaGlobal = 0;
  private actKeys = new Set<string>();
  tokens: string[] = [];
  labelsVisible = true;
  glowColor = new THREE.Color(0.25, 0.75, 1.0);
  deltaNormalization: 'global' | 'per-tensor' = 'global';

  constructor(structure: Structure, kind: 'flow' | 'tower' = 'flow') {
    this.structure = structure;
    this.layout = kind === 'flow' && structure.towers[0]?.blocks.length ? computeFlowLayout(structure) : computeLayout(structure);
    const P = this.layout.placements;
    this.n = P.length;
    const nW = P.filter((p) => p.set === 'w').length;
    this.tile = nW <= 1500 ? 128 : nW <= 6000 ? 64 : 32;
    this.cur = new Float32Array(this.n);
    this.flash = new Float32Array(this.n);
    this.pendVal = new Float32Array(this.n);
    this.pendAt = new Float32Array(this.n);
    this.pend = new Uint8Array(this.n);
    this.setOf = new Uint8Array(this.n);
    this.chunkIndex = new Int32Array(this.n);
    this.instIndex = new Int32Array(this.n);
    const wIdx: number[] = [];
    const aIdx: number[] = [];
    P.forEach((p, i) => {
      if (p.set === 'a') {
        aIdx.push(i);
        this.setOf[i] = 1;
      } else {
        wIdx.push(i);
        if (!this.slots.has(p.tid)) this.slots.set(p.tid, []);
        this.slots.get(p.tid)!.push(i);
      }
    });

    const { texture, row } = buildLutTexture();
    this.lutRow = row;
    this.uniforms = {
      uLut: { value: texture },
      uRowW: { value: row('inferno') },
      uRowD: { value: row('fireice') },
      uRowA: { value: row('fireice') },
      uRowP: { value: row('ice') },
      uView: { value: 0 },
      uBright: { value: 0.9 },
      uDim: { value: 1 },
      uDeltaGain: { value: 1 },
      uFill: { value: 0.28 },
      uGlow: { value: this.glowColor },
    };
    const T = structure.tensors;
    const flagsOf = (p: Placement) =>
      (p.transpose ? 1 : 0) | (p.ghost ? 2 : 0) | (p.set === 'w' && T[p.tid]?.kind !== 'matrix' ? 4 : 0) |
      (p.set === 'w' && !T[p.tid]?.float ? 8 : 0) | (p.attn ? 16 : 0);
    this.w = new PanelSet('w', wIdx, P, this.tile, this.tile, this.uniforms, this.root, flagsOf, this.chunkIndex, this.instIndex);
    this.a = new PanelSet('a', aIdx, P, ACT_P, ACT_T, this.uniforms, this.root, flagsOf, this.chunkIndex, this.instIndex);
    this.setTokens([]);
    this.buildSpine();
    this.buildConnectors();
    this.buildLabels();
  }

  private chunkOf(i: number): Chunk {
    return (this.setOf[i] ? this.a : this.w).chunks[this.chunkIndex[i]];
  }

  // --------------------------------------------------------------- build
  private buildSpine(): void {
    const main = this.layout.towers[0];
    if (!main) return;
    const x = main.railX;
    const r = this.layout.kind === 'flow' ? 0.05 : 0.035;
    const geo = new THREE.CylinderGeometry(r, r, 1, 12, 1, true);
    for (const f of main.floors) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x1d2a44 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, f.y + f.h / 2, -0.04);
      mesh.scale.set(1, f.h, 1);
      this.root.add(mesh);
      this.spine.push({ mesh, mat, cur: 0, pend: 0, at: Infinity, flash: 0 });
    }
    const ys = main.floors.map((f) => [f.y, f.y + f.h]).flat();
    const lo = Math.min(...ys);
    const hi = Math.max(...ys);
    const capMat = new THREE.MeshBasicMaterial({ color: 0x1d2a44 });
    const rail = this.layout.connectors.find((c) => c.kind === 'rail');
    const top = rail ? Math.max(rail.pts[0][1], rail.pts[1][1]) : hi + 0.6;
    const bottom = rail ? Math.min(rail.pts[0][1], rail.pts[1][1]) : lo - 0.6;
    for (const [a, b] of [[hi, top], [bottom, lo]]) {
      if (b - a <= 0.01) continue;
      const m = new THREE.Mesh(geo, capMat);
      m.position.set(x, (a + b) / 2, -0.04);
      m.scale.set(1, b - a, 1);
      this.root.add(m);
    }
  }

  private buildConnectors(): void {
    const pts: number[] = [];
    for (const c of this.layout.connectors) {
      if (c.kind === 'rail') continue;
      for (let i = 0; i + 1 < c.pts.length; i++) pts.push(...c.pts[i], ...c.pts[i + 1]);
    }
    if (pts.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      this.root.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x2b6f7c, transparent: true, opacity: 0.8 })));
    }
    if (this.layout.nodes.length) {
      const g = new THREE.TorusGeometry(0.11, 0.025, 8, 24);
      const m = new THREE.MeshBasicMaterial({ color: 0x3fb7c8 });
      for (const n of this.layout.nodes) {
        const t = new THREE.Mesh(g, m);
        t.position.set(...n);
        this.root.add(t);
      }
    }
  }

  private label(text: string, cls: string, x: number, y: number, z: number): CSS2DObject {
    const el = document.createElement('div');
    el.className = `lbl ${cls}`;
    el.textContent = text;
    const o = new CSS2DObject(el);
    o.position.set(x, y, z);
    this.root.add(o);
    return o;
  }

  private buildLabels(): void {
    const short = (t?: string | null) =>
      !t ? '' : t.includes('sliding') ? ' ·S' : t.includes('linear') ? ' ·lin' : t.includes('full') ? ' ·F' : '';
    this.layout.towers.forEach((tw) => {
      const topY = tw.downward ? tw.yTop + 2.4 : tw.yTop + 1.6;
      this.label(tw.name, 'tower', (tw.xMin + tw.xMax) / 2, topY, 0);
      for (const f of tw.floors) {
        const fy = tw.downward ? f.y + f.h - 0.5 : f.y + f.h * 0.45;
        const o = this.label(`L${f.index}${short(f.layerType)}`, tw.downward ? 'floor big' : 'floor', tw.labelX ?? tw.xMin - 0.35, fy, 0);
        if (tw.index === 0) {
          this.floorLabels.push(o);
          const lens = this.label('', 'lens', tw.lensX ?? tw.xMax + 0.4, f.lensY ?? fy, 0);
          lens.element.style.display = 'none';
          this.lensLabels.push(lens);
        }
      }
    });
    for (const p of this.layout.placements) {
      if (!p.showLabel) continue;
      const text = p.label.replace(/^experts\./, '');
      if (p.flat) this.itemLabels.push(this.label(text, 'item plate', p.x, p.y, p.z + p.h / 2 + 0.15));
      else if (p.h > 0.2) this.itemLabels.push(this.label(text, 'item', p.x, p.y - p.h / 2 - 0.1, p.z));
    }
    for (const l of this.layout.labels) this.itemLabels.push(this.label(l.text, l.cls, ...l.pos));
  }

  // --------------------------------------------------------------- tiles
  /** Weight thumbnails (S×S each, ids order) into the atlas or delta textures. */
  setTiles(ids: number[], bytes: Uint8Array, target: 'atlas' | 'delta', size: number): void {
    if (size !== this.tile) return;
    const S2 = this.tile * this.tile;
    const dirty = new Set<THREE.DataArrayTexture>();
    ids.forEach((tid, k) => this.w.writeRows(String(tid), 0, this.tile, bytes, k * S2, this.tile, target, dirty));
    for (const t of dirty) t.needsUpdate = true;
  }

  setDeltaScales(ids: number[], scales: number[]): void {
    ids.forEach((tid, k) => this.deltaScales.set(tid, scales[k]));
    this.deltaGlobal = Math.max(...this.deltaScales.values(), 1e-30);
    this.applyDeltaScales();
  }

  applyDeltaScales(): void {
    const P = this.layout.placements;
    for (const c of this.w.chunks) {
      const arr = c.aScale.array as Float32Array;
      c.members.forEach((pi, i) => {
        const s = this.deltaScales.get(P[pi].tid);
        arr[i] = s === undefined ? 0 : this.deltaNormalization === 'global' ? s / this.deltaGlobal : 1;
      });
      c.aScale.needsUpdate = true;
    }
  }

  clearDelta(): void {
    this.deltaScales.clear();
    for (const c of this.w.chunks) {
      if (!c.delta) continue;
      (c.delta.image.data as Uint8Array).fill(128);
      c.delta.clearLayerUpdates();
      c.delta.needsUpdate = true;
    }
    this.applyDeltaScales();
  }

  hasDelta(tid: number): boolean {
    return this.deltaScales.has(tid);
  }

  deltaTile(tid: number): Uint8Array | null {
    const dst = this.w.tiles.get(String(tid));
    if (!dst || !this.deltaScales.has(tid)) return null;
    const [c, l] = dst[0];
    const S2 = this.tile * this.tile;
    return c.delta ? (c.delta.image.data as Uint8Array).slice(l * S2, (l + 1) * S2) : null;
  }

  // --------------------------------------------------------- activations
  /** Start a new sequence: forget all activation data. */
  resetActivations(tokens: string[]): void {
    this.actKeys.clear();
    for (const c of this.a.chunks) {
      (c.atlas.image.data as Uint8Array).fill(0);
      c.atlas.clearLayerUpdates();
      c.atlas.needsUpdate = true;
    }
    this.setTokens(tokens);
  }

  /** infer_acts frame: rows [t0, t0+nt) of many activation tiles. */
  writeActivations(t0: number, nt: number, entries: [string, number][], payload: Uint8Array): void {
    const dirty = new Set<THREE.DataArrayTexture>();
    let off = 0;
    for (const [key, P] of entries) {
      if (this.a.writeRows(key, t0, nt, payload, off, P, 'atlas', dirty)) this.actKeys.add(key);
      off += nt * P;
    }
    for (const t of dirty) t.needsUpdate = true;
  }

  /** infer_attn frame: [L, H, nq, nk] uint8 (√p). */
  writeAttention(q0: number, nq: number, nk: number, L: number, H: number, payload: Uint8Array): void {
    const dirty = new Set<THREE.DataArrayTexture>();
    for (let l = 0; l < L; l++)
      for (let h = 0; h < H; h++) {
        const key = `A:${l}:${h}`;
        if (this.a.writeRows(key, q0, nq, payload, (l * H + h) * nq * nk, nk, 'atlas', dirty)) this.actKeys.add(key);
      }
    for (const t of dirty) t.needsUpdate = true;
  }

  /** Update uv rectangles so every activation block shows exactly `tokens.length` tokens. */
  setTokens(tokens: string[]): void {
    this.tokens = tokens;
    const n = Math.min(tokens.length, ACT_T);
    const P = this.layout.placements;
    for (const c of this.a.chunks) {
      const uv = c.aUv.array as Float32Array;
      const sc = c.aScale.array as Float32Array;
      c.members.forEach((pi, i) => {
        const p = P[pi];
        const has = n > 0 && this.actKeys.has(p.tile);
        sc[i] = has ? 1 : 0;
        if (p.attn) uv.set([0, 0, n / ACT_P, n / ACT_T], i * 4);
        else {
          const k = (p.P ?? ACT_P) / ACT_P;
          const [c0, c1] = p.chan ?? [0, 1];
          uv.set([c0 * k, 0, c1 * k, Math.max(n, 1) / ACT_T], i * 4);
        }
      });
      c.aUv.needsUpdate = true;
      c.aScale.needsUpdate = true;
    }
  }

  // ------------------------------------------------------------- display
  setColormap(weights: string, delta?: string): void {
    this.uniforms.uRowW.value = this.lutRow(weights);
    if (delta) this.uniforms.uRowD.value = this.lutRow(delta);
  }

  setView(view: 'weights' | 'delta'): void {
    this.uniforms.uView.value = view === 'delta' ? 1 : 0;
    this.uniforms.uFill.value = view === 'delta' ? 0.05 : 0.28;
  }

  setBrightness(v: number): void {
    this.uniforms.uBright.value = v;
  }

  setDeltaGain(v: number): void {
    this.uniforms.uDeltaGain.value = v;
  }

  setActivityMode(mode: 'none' | 'inference' | 'training'): void {
    if (mode === 'training') this.glowColor.setRGB(1.0, 0.5, 0.12);
    else this.glowColor.setRGB(0.25, 0.75, 1.0);
    this.uniforms.uDim.value = mode === 'none' ? 1 : 0.6;
  }

  setLabelsVisible(v: boolean): void {
    this.labelsVisible = v;
    for (const o of [...this.floorLabels, ...this.itemLabels]) o.visible = v;
  }

  // ------------------------------------------------------------- picking
  pick(raycaster: THREE.Raycaster): PickResult {
    const meshes = [...this.w.chunks, ...this.a.chunks].map((c) => c.mesh);
    for (const hit of raycaster.intersectObjects(meshes, false)) {
      if (hit.instanceId === undefined) continue;
      const set = hit.object.userData.set === 'a' ? this.a : this.w;
      const c = set.chunks[hit.object.userData.chunk];
      return { idx: c.members[hit.instanceId], uv: hit.uv ?? null };
    }
    return { idx: -1, uv: null };
  }

  private setSel(idx: number, v: number): void {
    if (idx < 0) return;
    const c = this.chunkOf(idx);
    (c.aSel.array as Float32Array)[this.instIndex[idx]] = v;
    c.aSel.needsUpdate = true;
  }

  setHover(idx: number): void {
    if (idx === this.hoverIdx) return;
    if (this.hoverIdx !== this.selIdx) this.setSel(this.hoverIdx, 0);
    this.hoverIdx = idx;
    if (idx !== this.selIdx) this.setSel(idx, 0.5);
  }

  setSelected(idx: number): void {
    this.setSel(this.selIdx, 0);
    this.selIdx = idx;
    this.setSel(idx, 1);
  }

  placement(idx: number): Placement | undefined {
    return this.layout.placements[idx];
  }

  placementsOf(tid: number): number[] {
    return this.slots.get(tid) ?? [];
  }

  hasActivation(key: string): boolean {
    return this.actKeys.has(key);
  }

  /** Copy of an activation tile: ACT_T rows (tokens) × ACT_P columns (channels). */
  activationTile(key: string): Uint8Array | null {
    const dst = this.a.tiles.get(key);
    if (!dst || !this.actKeys.has(key)) return null;
    const [c, l] = dst[0];
    const n = ACT_P * ACT_T;
    return (c.atlas.image.data as Uint8Array).slice(l * n, (l + 1) * n);
  }

  // ------------------------------------------------------------ activity
  /** Normalise raw per-tensor values to 0..1 per class (e.g. all q_proj), or on a log scale. */
  valuesFromTensors(raw: Map<number | 'head', number>, mode: 'per-key' | 'log'): Float32Array {
    const P = this.layout.placements;
    const out = new Float32Array(this.n);
    const val = (p: Placement) =>
      p.set === 'a' ? undefined : p.key === 'head' && raw.has('head') ? raw.get('head') : raw.get(p.tid);
    if (mode === 'per-key') {
      const max = new Map<string, number>();
      for (const p of P) {
        const v = val(p);
        if (v !== undefined && Number.isFinite(v)) max.set(p.key, Math.max(max.get(p.key) ?? 0, Math.abs(v)));
      }
      P.forEach((p, i) => {
        const v = val(p);
        const m = max.get(p.key) ?? 0;
        out[i] = v !== undefined && m > 0 ? Math.abs(v) / m : 0;
      });
    } else {
      let hi = -Infinity;
      for (const v of raw.values()) if (v > 0 && Number.isFinite(v)) hi = Math.max(hi, Math.log10(v));
      P.forEach((p, i) => {
        const v = val(p);
        out[i] = v !== undefined && v > 0 && Number.isFinite(v) ? Math.max(0, (Math.log10(v) - (hi - 3)) / 3) : 0;
      });
    }
    return out;
  }

  /** 0 = input end of the network, 1 = output end. */
  private order(p: Placement): number {
    const tw = this.layout.towers[p.tower];
    const nf = Math.max(1, tw.floors.length);
    if (p.floor === FLOOR_PRE) return 0;
    if (p.floor === FLOOR_POST) return 1;
    return (p.floor + 1) / (nf + 1);
  }

  pulse({ values, direction, duration }: Pulse, spine?: number[]): void {
    const now = performance.now() / 1000;
    const P = this.layout.placements;
    const at = (f: number) => now + (direction === 'forward' ? f : 1 - f) * duration;
    for (let i = 0; i < this.n; i++) {
      this.pendVal[i] = values[i];
      this.pendAt[i] = at(this.order(P[i]));
      this.pend[i] = 1;
    }
    const nf = this.spine.length;
    const max = spine && spine.length ? Math.max(...spine.map(Math.abs), 1e-9) : 1;
    this.spine.forEach((s, i) => {
      s.pend = spine && spine.length ? (spine[i] !== undefined ? Math.abs(spine[i]) / max : 0) : 0.6;
      s.at = at((i + 1) / (nf + 1));
    });
    this.animating = true;
  }

  clearActivity(): void {
    this.cur.fill(0);
    this.flash.fill(0);
    this.pend.fill(0);
    for (const s of this.spine) {
      s.cur = s.flash = s.pend = 0;
      s.at = Infinity;
      s.mat.color.setHex(0x1d2a44);
    }
    for (const set of [this.w, this.a])
      for (const c of set.chunks) {
        (c.aAct.array as Float32Array).fill(0);
        c.aAct.needsUpdate = true;
      }
    for (const l of this.lensLabels) l.element.style.display = 'none';
    this.animating = false;
  }

  update(dt: number): void {
    if (!this.animating) return;
    const now = performance.now() / 1000;
    const decay = Math.exp(-dt * 3.5);
    let busy = false;
    for (const set of [this.w, this.a])
      for (const c of set.chunks) {
        const arr = c.aAct.array as Float32Array;
        c.members.forEach((i, k) => {
          if (this.pend[i] && now >= this.pendAt[i]) {
            this.cur[i] = this.pendVal[i];
            this.flash[i] = 1;
            this.pend[i] = 0;
          }
          if (this.pend[i]) busy = true;
          if (this.flash[i] > 0.003) {
            this.flash[i] *= decay;
            busy = true;
          } else this.flash[i] = 0;
          arr[k] = this.cur[i] * 0.45 + this.flash[i] * (0.15 + 0.85 * this.cur[i]);
        });
        c.aAct.needsUpdate = true;
      }
    const base = new THREE.Color(0x1d2a44);
    for (const s of this.spine) {
      if (now >= s.at) {
        s.cur = s.pend;
        s.flash = 1;
        s.at = Infinity;
      }
      if (s.at !== Infinity) busy = true;
      if (s.flash > 0.003) {
        s.flash *= decay;
        busy = true;
      }
      const k = Math.min(1.5, s.cur * 0.5 + s.flash * 0.9);
      s.mat.color.copy(base).lerp(this.glowColor, Math.min(1, k)).multiplyScalar(1 + Math.max(0, k - 1) * 2);
    }
    this.animating = busy;
  }

  /** Logit-lens labels next to each layer: top-1 token per layer. */
  setLens(lens: [string, number][][]): void {
    this.lensLabels.forEach((l, i) => {
      const row = lens[i];
      if (!row || !row.length) {
        l.element.style.display = 'none';
        return;
      }
      const [tok, p] = row[0];
      l.element.style.display = '';
      l.element.innerHTML = '';
      const t = document.createElement('span');
      t.className = 'tok';
      t.textContent = showToken(tok);
      const b = document.createElement('span');
      b.className = 'p';
      b.textContent = `${(p * 100).toFixed(0)}%`;
      l.element.append(t, b);
      l.element.style.opacity = String(0.35 + 0.65 * Math.min(1, p * 1.5));
    });
  }

  floorCenter(i: number): THREE.Vector3 {
    const tw = this.layout.towers[0];
    const f = tw?.floors[Math.max(0, Math.min(i, tw.floors.length - 1))];
    return new THREE.Vector3(0, f ? f.y + f.h / 2 : 0, 0);
  }

  dispose(): void {
    this.w.dispose();
    this.a.dispose();
    (this.uniforms.uLut.value as THREE.Texture).dispose();
    this.root.traverse((o) => {
      if (o instanceof CSS2DObject) o.element.remove();
      // spine, connectors and (+) nodes (the panel chunks were disposed above; twice is harmless)
      const m = o as Partial<THREE.Mesh>;
      m.geometry?.dispose();
      for (const mat of Array.isArray(m.material) ? m.material : m.material ? [m.material] : []) mat.dispose();
    });
    this.root.clear();
  }
}

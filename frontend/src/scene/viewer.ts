import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import type { ModelView } from './modelView';

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private labels: CSS2DRenderer;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private pointerClient = { x: 0, y: 0 };
  private pointerDirty = false;
  private down: { x: number; y: number } | null = null;
  private clock = new THREE.Clock();
  private tween: { from: THREE.Vector3; to: THREE.Vector3; camFrom: THREE.Vector3; camTo: THREE.Vector3; t: number; dur: number } | null = null;
  private grid: THREE.GridHelper;
  model: ModelView | null = null;
  onHover?: (idx: number, x: number, y: number, uv: THREE.Vector2 | null) => void;
  onSelect?: (idx: number) => void;
  // Render on demand: a frame (with bloom) is drawn only when something changed —
  // the camera, the model's textures / uniforms / animations, or the canvas size.
  // This keeps the GPU free for inference and training on small cards.
  private dirty = true;
  private lastRender = 0;

  constructor(private container: HTMLElement) {
    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x06080d);
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 4000);
    this.camera.position.set(6, 4, 14);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;

    this.grid = new THREE.GridHelper(200, 200, 0x16202e, 0x0d131c);
    this.grid.position.y = -0.6;
    this.scene.add(this.grid);

    this.labels = new CSS2DRenderer();
    this.labels.setSize(w, h);
    Object.assign(this.labels.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
    container.appendChild(this.labels.domElement);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.75, 0.45, 0.78);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    new ResizeObserver(() => this.resize()).observe(container);
    const el = this.renderer.domElement;
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.pointerClient = { x: e.clientX, y: e.clientY };
      this.pointerDirty = true;
    });
    el.addEventListener('pointerleave', () => {
      this.pointer.set(9, 9);
      this.pointerDirty = true;
    });
    el.addEventListener('pointerdown', (e) => (this.down = { x: e.clientX, y: e.clientY }));
    el.addEventListener('pointerup', (e) => {
      if (this.down && Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) < 5 && e.button === 0) {
        this.onSelect?.(this.pickAt(e.clientX, e.clientY));
      }
      this.down = null;
    });
    el.addEventListener('dblclick', (e) => {
      const idx = this.pickAt(e.clientX, e.clientY);
      const p = idx >= 0 ? this.model?.placement(idx) : undefined;
      if (p) this.focusOn(new THREE.Vector3(p.x, p.y, p.z), Math.max(2.2, Math.max(p.w, p.h) * 1.6));
    });
    window.addEventListener('keydown', (e) => this.onKey(e));
    document.addEventListener('visibilitychange', () => this.invalidate());
    this.renderer.setAnimationLoop(() => this.frame());
  }

  /** Request a new frame (for changes the viewer cannot see by itself). */
  invalidate(): void {
    this.dirty = true;
  }

  private pickAt(cx: number, cy: number): number {
    if (!this.model) return -1;
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(v, this.camera);
    return this.model.pick(this.raycaster).idx;
  }

  /** Keep the model centred in the area not covered by the side panels. */
  setInsets(left: number, right: number): void {
    this.insets = { left, right };
    this.applyInsets();
  }

  private insets = { left: 0, right: 0 };

  private applyInsets(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.setViewOffset(w, h, (this.insets.right - this.insets.left) / 2, 0, w, h);
    this.dirty = true;
  }

  private freeFraction(): number {
    const w = this.container.clientWidth || 1;
    return Math.max(0.3, (w - this.insets.left - this.insets.right) / w);
  }

  private onKey(e: KeyboardEvent): void {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    const tw = this.model?.layout.towers[0];
    if (!tw) return;
    const step = tw.floorH * (e.shiftKey ? 4 : 1);
    if (e.key === 'ArrowUp' || e.key === 'PageUp') this.shift(step);
    else if (e.key === 'ArrowDown' || e.key === 'PageDown') this.shift(-step);
    else if (e.key === 'f' || e.key === 'F') this.frameAll();
    else if (e.key === 'Home') this.goToFloor(0);
    else if (e.key === 'End') this.goToFloor(tw.floors.length - 1);
    else return;
    e.preventDefault();
  }

  private shift(dy: number): void {
    const d = new THREE.Vector3(0, dy, 0);
    this.animateTo(this.controls.target.clone().add(d), this.camera.position.clone().add(d), 0.35);
  }

  private resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.applyInsets();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.labels.setSize(w, h);
    this.dirty = true;
  }

  setModel(model: ModelView | null): void {
    if (this.model) {
      this.scene.remove(this.model.root);
      this.model.dispose();
    }
    this.model = model;
    if (model) {
      this.scene.add(model.root);
      this.grid.position.y = model.layout.min[1] - 0.4;
      this.goToFloor(0, true);
    }
    this.dirty = true;
  }

  setBloom(on: boolean): void {
    this.bloom.enabled = on;
    this.dirty = true;
  }

  animateTo(target: THREE.Vector3, cam: THREE.Vector3, dur = 0.6): void {
    this.tween = { from: this.controls.target.clone(), to: target, camFrom: this.camera.position.clone(), camTo: cam, t: 0, dur };
  }

  focusOn(p: THREE.Vector3, dist: number): void {
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.animateTo(p.clone(), p.clone().add(dir.multiplyScalar(dist)));
  }

  /** Frame one layer (flow layout) or the first few floors (tower layout). */
  goToFloor(i: number, instant = false): void {
    if (!this.model) return;
    const L = this.model.layout;
    const tw = L.towers[0];
    if (!tw) return;
    const c = this.model.floorCenter(i);
    const vfov = (this.camera.fov * Math.PI) / 180;
    const tanV = Math.tan(vfov / 2);
    const tanH = tanV * this.camera.aspect * this.freeFraction();
    const flow = L.kind === 'flow';
    const width = (tw.xMax - tw.xMin) * (flow ? 1.05 : 0.9);
    const height = flow ? tw.floorH * 0.95 : tw.floorH * 4;
    const dist = Math.max(width / 2 / tanH, height / 2 / tanV, 6);
    const target = new THREE.Vector3((tw.xMin + tw.xMax) / 2, flow ? c.y : c.y + tw.floorH * 1.5, flow ? -1 : 0);
    const dir = new THREE.Vector3(flow ? 0.45 : 0.4, flow ? 0.42 : 0.26, 1).normalize();
    const cam = target.clone().add(dir.multiplyScalar(dist));
    if (instant) {
      this.controls.target.copy(target);
      this.camera.position.copy(cam);
    } else this.animateTo(target, cam);
  }

  frameAll(): void {
    if (!this.model) return;
    const { min, max } = this.model.layout;
    const c = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
    const tanH = tanV * this.camera.aspect * this.freeFraction();
    const dist = Math.max((max[0] - min[0]) / 2 / tanH, (max[1] - min[1]) / 2 / tanV) * 1.05;
    this.animateTo(c, c.clone().add(new THREE.Vector3(dist * 0.3, dist * 0.1, dist)));
  }

  private frame(): void {
    const dt = Math.min(0.1, this.clock.getDelta());
    if (document.hidden) return; // minimised / covered: nothing to draw
    let changed = this.dirty;
    if (this.tween) {
      const tw = this.tween;
      tw.t = Math.min(1, tw.t + dt / tw.dur);
      const k = 1 - Math.pow(1 - tw.t, 3);
      this.controls.target.lerpVectors(tw.from, tw.to, k);
      this.camera.position.lerpVectors(tw.camFrom, tw.camTo, k);
      if (tw.t >= 1) this.tween = null;
      changed = true;
    }
    if (this.controls.update()) changed = true; // true while the camera moves (incl. damping)
    if (this.pointerDirty && this.model) {
      this.pointerDirty = false;
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const { idx, uv } = this.model.pick(this.raycaster);
      this.model.setHover(idx);
      this.renderer.domElement.style.cursor = idx >= 0 ? 'pointer' : '';
      this.onHover?.(idx, this.pointerClient.x, this.pointerClient.y, uv);
    }
    this.model?.update(dt);
    if (this.model?.takeDirty()) changed = true;
    const now = performance.now();
    // a slow refresh as a safety net for any change that forgot to ask for a frame
    if (!changed && now - this.lastRender < 1000) return;
    this.dirty = false;
    this.lastRender = now;
    // always go through the composer: OutputPass does the sRGB conversion
    this.composer.render();
    this.labels.render(this.scene, this.camera);
  }
}

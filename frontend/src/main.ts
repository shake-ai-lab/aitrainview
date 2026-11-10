import './style.css';
import { api, desktop, EventSocket } from './api';
import type { AppCtx, Settings } from './ctx';
import { COLORMAPS, cssGradient } from './scene/colormaps';
import { ACT_P } from './scene/flowLayout';
import { ModelView } from './scene/modelView';
import { Viewer } from './scene/viewer';
import type { RuntimeInfo, Structure } from './types';
import { InferPanel } from './ui/infer';
import { InspectPanel } from './ui/inspect';
import { TrainPanel } from './ui/train';
import { fmtBytes, fmtCount, fmtShape, h, showToken } from './util';

const EXAMPLES = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B-Instruct', 'google/gemma-3-270m-it', 'HuggingFaceTB/SmolLM2-360M-Instruct'];
const RECENT_KEY = 'aitv.recent';

function loadRecent(): string[] {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]');
  } catch {
    return [];
  }
}

function pushRecent(src: string): void {
  try {
    const r = [src, ...loadRecent().filter((x) => x !== src)].slice(0, 8);
    localStorage.setItem(RECENT_KEY, JSON.stringify(r));
  } catch {
    /* storage unavailable */
  }
}

class App implements AppCtx {
  sock = new EventSocket();
  viewer: Viewer;
  structure: Structure | null = null;
  view: ModelView | null = null;
  runtime: RuntimeInfo | null = null;
  settings: Settings = { mode: 'rms', cmap: 'inferno', deltaCmap: 'fireice', view: 'weights', layout: 'flow' };
  // display controls of the left panel; re-applied whenever the 3D view is rebuilt
  private display = { brightness: 0.9, deltaGain: 1, deltaNorm: 'global' as 'global' | 'per-tensor', labels: true };
  private tokens: string[] = []; // prompt + generated tokens of the current inference
  grads = new Map<number, number>();
  rel = new Map<number, number>();
  acts = new Map<number, number>();

  private root = document.getElementById('app')!;
  private stage = h('div', { class: 'stage' });
  private toasts = h('div', { class: 'toasts' });
  private tooltip = h('div', { class: 'tooltip' });
  private loadbar = h('div', { class: 'loadbar' }, h('i'));
  private loadText = h('span', { class: 'load-text' });
  private sourceInput = h('input', { type: 'text', placeholder: 'HF のモデルID（例: Qwen/Qwen3-0.6B）またはローカルフォルダのパス', spellcheck: false }) as HTMLInputElement;
  private tokenInput = h('input', { type: 'password', placeholder: 'HFトークン（Gemma など規約同意が必要なモデル用・任意）' }) as HTMLInputElement;
  private openBtn = h('button', { class: 'primary', onclick: () => this.open(this.sourceInput.value) }, '開く');
  private sysInfo = h('span', { class: 'sys' });
  private conn = h('span', { class: 'conn' });
  private modelInfo = h('div', { class: 'model-info' });
  private legend = h('div', { class: 'legend' });
  private layerSlider = h('input', { type: 'range', min: '0', max: '0', value: '0' }) as HTMLInputElement;
  private welcome: HTMLElement;
  private tabs: Record<string, { btn: HTMLElement; body: HTMLElement }> = {};
  private inspect: InspectPanel;
  private atlasToken = 0;

  constructor() {
    this.viewer = new Viewer(this.stage);
    this.inspect = new InspectPanel(this);
    const infer = new InferPanel(this);
    const train = new TrainPanel(this);
    this.welcome = this.buildWelcome();
    this.root.append(
      this.buildTopbar(),
      h('div', { class: 'main' }, this.stage, this.buildLeft(), this.buildRight([
        ['inspect', '詳細', this.inspect.el],
        ['infer', '推論', infer.el],
        ['train', '学習', train.el],
      ])),
      this.loadbar,
      this.tooltip,
      this.toasts,
      this.welcome,
    );
    this.viewer.onHover = (idx, x, y, uv) => this.hover(idx, x, y, uv);
    this.viewer.onSelect = (idx) => {
      const p = idx >= 0 ? this.view?.placement(idx) : undefined;
      if (!p || !this.view) return;
      if (p.set === 'w') this.select(p.tid, idx);
      else {
        this.view.setSelected(idx);
        this.showTab('inspect');
        this.inspect.showActivation(p);
      }
    };
    this.viewer.setInsets(282, 374);
    this.wireSocket();
    this.sock.connect();
    this.init();
  }

  // ------------------------------------------------------------- layout
  private buildTopbar(): HTMLElement {
    this.sourceInput.addEventListener('keydown', (e) => e.key === 'Enter' && this.open(this.sourceInput.value));
    const pick = desktop
      ? h('button', { onclick: () => this.pickFolder(), title: 'フォルダを選択' }, '📁')
      : null;
    return h('header', { class: 'topbar' },
      h('div', { class: 'brand' }, h('span', { class: 'logo' }, '◆'), 'AITrainView'),
      h('div', { class: 'source' }, this.sourceInput, pick, this.openBtn),
      this.loadText,
      h('div', { class: 'spacer' }),
      this.sysInfo,
      this.conn,
    );
  }

  private buildWelcome(): HTMLElement {
    const input = h('input', { type: 'text', placeholder: 'Qwen/Qwen3-0.6B  または  C:\\models\\my-model', spellcheck: false }) as HTMLInputElement;
    const go = () => this.open(input.value);
    input.addEventListener('keydown', (e) => e.key === 'Enter' && go());
    const recent = loadRecent();
    return h('div', { class: 'welcome' },
      h('div', { class: 'welcome-card' },
        h('h1', {}, h('span', { class: 'logo' }, '◆'), ' AITrainView'),
        h('p', { class: 'muted' }, 'LLM の重み・アーキテクチャ・推論・ファインチューニングを 3D で見る'),
        h('div', { class: 'row' }, input,
          desktop ? h('button', { onclick: () => this.pickFolder() }, '📁 フォルダ') : null,
          h('button', { class: 'primary', onclick: go }, '開く')),
        h('div', { class: 'muted small' }, 'Hugging Face のID なら自動でダウンロードします（safetensors 形式）。'),
        h('div', { class: 'chips' }, h('span', { class: 'muted small' }, '例:'),
          ...EXAMPLES.map((e) => h('button', { class: 'chip', onclick: () => this.open(e) }, e))),
        recent.length ? h('div', { class: 'chips' }, h('span', { class: 'muted small' }, '最近:'),
          ...recent.map((e) => h('button', { class: 'chip', onclick: () => this.open(e), title: e }, e.length > 48 ? '…' + e.slice(-46) : e))) : null,
        h('details', { class: 'fold' }, h('summary', {}, 'HFトークン（任意）'), this.tokenInput),
      ),
    );
  }

  private buildLeft(): HTMLElement {
    const modeSel = h('select', {
      onchange: (e: Event) => {
        this.settings.mode = (e.target as HTMLSelectElement).value as Settings['mode'];
        const signed = this.settings.mode === 'mean' || this.settings.mode === 'sample';
        this.settings.cmap = signed ? 'fireice' : COLORMAPS[cmapSel.value]?.signed ? 'inferno' : cmapSel.value;
        cmapSel.value = this.settings.cmap;
        this.view?.setColormap(this.settings.cmap);
        this.updateLegend();
        this.loadAtlas();
      },
    },
      h('option', { value: 'rms' }, 'ブロックRMS（大きさ）'),
      h('option', { value: 'absmax' }, 'ブロック最大 |w|（外れ値）'),
      h('option', { value: 'mean' }, 'ブロック平均（符号付き）'),
      h('option', { value: 'sample' }, '実際の値を間引き（符号付き）'),
    ) as HTMLSelectElement;
    const cmapSel = h('select', {
      onchange: (e: Event) => {
        this.settings.cmap = (e.target as HTMLSelectElement).value;
        this.view?.setColormap(this.settings.cmap);
        this.updateLegend();
      },
    },
      ...Object.entries(COLORMAPS).map(([k, c]) => h('option', { value: k }, c.label)),
    ) as HTMLSelectElement;
    const viewSel = h('div', { class: 'seg' },
      h('button', { 'data-v': 'weights', onclick: () => this.setView('weights') }, '重み W'),
      h('button', { 'data-v': 'delta', onclick: () => this.setView('delta') }, '学習による変化 ΔW'),
    );
    const bright = h('input', { type: 'range', min: '0.2', max: '1.6', step: '0.05', value: String(this.display.brightness), oninput: (e: Event) => {
      this.display.brightness = Number((e.target as HTMLInputElement).value);
      this.view?.setBrightness(this.display.brightness);
    } });
    const gain = h('input', { type: 'range', min: '0', max: '3', step: '0.05', value: '0', oninput: (e: Event) => {
      this.display.deltaGain = 10 ** Number((e.target as HTMLInputElement).value);
      this.view?.setDeltaGain(this.display.deltaGain);
    } });
    const deltaNorm = h('select', {
      onchange: (e: Event) => {
        this.display.deltaNorm = (e.target as HTMLSelectElement).value as 'global' | 'per-tensor';
        if (!this.view) return;
        this.view.deltaNormalization = this.display.deltaNorm;
        this.view.applyDeltaScales();
      },
    }, h('option', { value: 'global' }, 'Δの色: 全行列で共通スケール'), h('option', { value: 'per-tensor' }, 'Δの色: 行列ごとに正規化'));
    const bloom = h('input', { type: 'checkbox', checked: true, onchange: (e: Event) => this.viewer.setBloom((e.target as HTMLInputElement).checked) });
    const labels = h('input', { type: 'checkbox', checked: true, onchange: (e: Event) => {
      this.display.labels = (e.target as HTMLInputElement).checked;
      this.view?.setLabelsVisible(this.display.labels);
    } });
    this.layerSlider.addEventListener('input', () => this.viewer.goToFloor(Number(this.layerSlider.value)));
    const layoutSel = h('div', { class: 'seg layout-seg' },
      h('button', { 'data-l': 'flow', class: 'active', onclick: () => this.setLayout('flow') }, 'フロー（計算の流れ）'),
      h('button', { 'data-l': 'tower', onclick: () => this.setLayout('tower') }, 'コンパクト（重みのみ）'),
    );
    return h('aside', { class: 'panel left' },
      h('section', {}, h('div', { class: 'sub' }, 'モデル'), this.modelInfo),
      h('section', {},
        h('div', { class: 'sub' }, 'レイアウト'),
        layoutSel,
        h('div', { class: 'sub' }, '表示'),
        viewSel,
        h('label', { class: 'block' }, '重みの縮約方法', modeSel),
        h('label', { class: 'block' }, 'カラーマップ', cmapSel),
        this.legend,
        h('label', { class: 'block' }, '明るさ', bright),
        h('label', { class: 'block' }, 'ΔW の強調', gain),
        deltaNorm,
        h('div', { class: 'row wrap' }, h('label', { class: 'check' }, bloom, 'グロー'), h('label', { class: 'check' }, labels, 'ラベル'),
          h('button', { class: 'small', onclick: () => this.view?.clearActivity() }, '発光をクリア')),
      ),
      h('section', {},
        h('div', { class: 'sub' }, 'ナビゲーション'),
        h('label', { class: 'block' }, '層へ移動', this.layerSlider),
        h('div', { class: 'row' },
          h('button', { onclick: () => this.viewer.frameAll() }, '全体 (F)'),
          h('button', { onclick: () => this.viewer.goToFloor(0) }, '最下層'),
          h('button', { onclick: () => this.viewer.goToFloor(9999) }, '最上層'),
        ),
      ),
    );
  }

  private buildRight(tabs: [string, string, HTMLElement][]): HTMLElement {
    const head = h('div', { class: 'tabs' });
    const bodies = h('div', { class: 'tab-bodies' });
    for (const [key, label, body] of tabs) {
      const btn = h('button', { onclick: () => this.showTab(key as any) }, label);
      head.append(btn);
      bodies.append(body);
      this.tabs[key] = { btn, body };
    }
    this.showTab('inspect');
    return h('aside', { class: 'panel right' }, head, bodies);
  }

  showTab(name: 'inspect' | 'infer' | 'train'): void {
    for (const [k, t] of Object.entries(this.tabs)) {
      t.btn.classList.toggle('active', k === name);
      t.body.style.display = k === name ? '' : 'none';
    }
  }

  // ------------------------------------------------------------ actions
  private async init(): Promise<void> {
    api.get('/api/system').then((s) => {
      this.sysInfo.textContent = s.cuda ? `${s.gpu} · ${s.vram_gb} GB` : s.torch ? 'CPU のみ' : 'torch 未インストール（重み表示のみ）';
    }).catch(() => {});
    try {
      const st = await api.get('/api/structure');
      if (st && st.loaded !== false) this.setStructure(st as Structure);
    } catch {
      /* backend not ready yet; socket reconnect handles it */
    }
  }

  private async pickFolder(): Promise<void> {
    const p = await desktop?.pickFolder();
    if (p) this.open(p);
  }

  private async open(source: string): Promise<void> {
    source = source.trim();
    if (!source) return;
    this.openBtn.disabled = true;
    this.loadText.textContent = `${source} を開いています…`;
    this.welcome.classList.add('busy');
    try {
      const st = await api.post<Structure>('/api/open', { source, hf_token: this.tokenInput.value || null });
      pushRecent(source);
      this.setStructure(st);
    } catch (e) {
      this.toast(String(e instanceof Error ? e.message : e), 'error');
      this.loadText.textContent = '';
      this.loadbar.classList.remove('on'); // a failed download leaves its progress bar up
    } finally {
      this.openBtn.disabled = false;
      this.welcome.classList.remove('busy');
    }
  }

  private setLayout(kind: 'flow' | 'tower'): void {
    if (kind === this.settings.layout) return;
    this.settings.layout = kind;
    this.root.querySelectorAll('.layout-seg button').forEach((b) => b.classList.toggle('active', (b as HTMLElement).dataset.l === kind));
    if (this.structure) this.setStructure(this.structure, true);
  }

  private setStructure(st: Structure, keepState = false): void {
    this.structure = st;
    if (!keepState) {
      this.grads.clear();
      this.rel.clear();
      this.acts.clear();
      this.tokens = [];
    }
    const view = new ModelView(st, this.settings.layout);
    this.view = view;
    this.viewer.setModel(view);
    view.setColormap(this.settings.cmap, this.settings.deltaCmap);
    view.setBrightness(this.display.brightness);
    view.setDeltaGain(this.display.deltaGain);
    view.deltaNormalization = this.display.deltaNorm;
    view.setLabelsVisible(this.display.labels);
    this.setView(keepState ? this.settings.view : 'weights');
    view.setActivityMode(keepState ? this.activityMode : 'none');
    this.welcome.style.display = 'none';
    this.sourceInput.value = st.name;
    this.renderModelInfo();
    this.inspect.renderEmpty();
    this.layerSlider.max = String(Math.max(0, (view.layout.towers[0]?.floors.length ?? 1) - 1));
    this.layerSlider.value = '0';
    this.loadAtlas();
  }

  private renderModelInfo(): void {
    const st = this.structure;
    if (!st) return;
    const c = st.config;
    const row = (k: string, v: any) => (v === undefined || v === null || v === '' ? null : [h('span', {}, k), h('b', {}, String(v))]);
    const experts = c.num_experts ?? c.num_local_experts;
    this.modelInfo.replaceChildren(
      h('div', { class: 'model-name', title: st.model_dir }, st.name),
      h('div', { class: 'muted small' }, [st.model_type, ...(st.architectures ?? [])].filter(Boolean).join(' · ')),
      h('div', { class: 'kv' },
        row('パラメータ', fmtCount(st.total_params)),
        row('層数', c.num_hidden_layers ?? st.towers[0]?.blocks.length),
        row('hidden', c.hidden_size),
        row('heads (q / kv)', c.num_attention_heads ? `${c.num_attention_heads} / ${c.num_key_value_heads ?? c.num_attention_heads}` : null),
        row('head_dim', c.head_dim),
        row('MLP 中間次元', c.intermediate_size),
        row('experts', experts ? `${experts}（top-${c.num_experts_per_tok ?? '?'}）` : null),
        row('語彙数', c.vocab_size ? fmtCount(c.vocab_size) : null),
        row('dtype', Object.entries(st.dtype_counts).map(([k, v]) => `${k}×${v}`).join(' ')),
        row('塔', st.towers.length > 1 ? st.towers.map((t) => `${t.name}(${t.blocks.length})`).join(' + ') : null),
      ),
    );
  }

  private updateLegend(): void {
    const signed = this.settings.mode === 'mean' || this.settings.mode === 'sample';
    const delta = this.settings.view === 'delta';
    const cmap = delta ? this.settings.deltaCmap : this.settings.cmap;
    this.legend.replaceChildren(
      h('div', { class: 'grad', style: { background: cssGradient(cmap) } }),
      h('div', { class: 'row between small muted' },
        h('span', {}, delta ? '− 減少' : signed ? '− 負' : '小'),
        h('span', {}, delta ? '0 変化なし' : signed ? '0' : ''),
        h('span', {}, delta ? '+ 増加' : signed ? '+ 正' : '大')),
      h('div', { class: 'muted small' }, delta
        ? '学習で重みがどれだけ動いたか（ブロック平均）'
        : '各パネルの色範囲はパネルごとに正規化されています'),
    );
  }

  private async loadAtlas(): Promise<void> {
    if (!this.view) return;
    try {
      const r = await api.post('/api/atlas', { mode: this.settings.mode, size: this.view.tile });
      this.atlasToken = r.token;
      this.loadbar.classList.add('on');
    } catch (e) {
      this.toast(String(e), 'error');
    }
  }

  setView(v: 'weights' | 'delta'): void {
    this.settings.view = v;
    this.view?.setView(v);
    this.root.querySelectorAll('.seg button[data-v]').forEach((b) => b.classList.toggle('active', (b as HTMLElement).dataset.v === v));
    this.updateLegend();
  }

  private activityMode: 'none' | 'inference' | 'training' = 'none';

  setActivityMode(mode: 'none' | 'inference' | 'training'): void {
    this.activityMode = mode;
    this.view?.setActivityMode(mode);
  }

  select(tid: number, idx?: number): void {
    if (!this.view) return;
    this.view.setSelected(idx ?? this.view.placementsOf(tid)[0] ?? -1);
    this.showTab('inspect');
    this.inspect.show(tid);
  }

  private hover(idx: number, x: number, y: number, uv: { x: number; y: number } | null): void {
    const p = idx >= 0 ? this.view?.placement(idx) : undefined;
    if (!p || !this.structure) {
      this.tooltip.style.display = 'none';
      return;
    }
    const W = window.innerWidth;
    this.tooltip.style.left = `${Math.min(x + 14, W - 340)}px`;
    this.tooltip.style.top = `${y + 14}px`;
    this.tooltip.style.display = 'block';
    if (p.set === 'a') {
      const view = this.view!;
      const n = view.tokens.length;
      const layer = p.floor >= 0 && p.floor < 1e6 ? `L${p.floor} · ` : '';
      let where = '';
      if (uv && n > 0) {
        // same mapping as the shader: transposed blocks have tokens along the plane's x
        const col = p.transpose ? 1 - uv.y : uv.x;
        const row = p.transpose ? uv.x : 1 - uv.y;
        const tok = (i: number) => `「${showToken(view.tokens[Math.min(n - 1, Math.max(0, i))] ?? '')}」`;
        where = p.attn
          ? `query ${tok(Math.floor(row * n))} → key ${tok(Math.floor(col * n))}`
          : `トークン ${Math.floor(row * n)} ${tok(Math.floor(row * n))}`;
      }
      this.tooltip.replaceChildren(
        h('div', { class: 'tt-title' }, `${layer}${p.label}${p.head !== undefined ? ` · head ${p.head}` : ''}`),
        h('div', {}, p.desc ?? ''),
        h('div', { class: 'tt-name' }, p.attn ? `${n} × ${n}（query × key）` : `${p.P}${p.P === ACT_P ? '（縮約）' : ''} ch × ${n} tokens`),
        view.hasActivation(p.tile) ? (where ? h('div', { class: 'tt-extra' }, where) : null) as any
          : h('div', { class: 'muted' }, '推論を実行すると実際の値が入ります'),
      );
      return;
    }
    const t = this.structure.tensors[p.tid];
    const tw = this.view!.layout.towers[p.tower];
    const floor = p.floor >= 0 && p.floor < 1e6 ? `L${tw.floors[p.floor]?.index}` : p.floor < 0 ? '入力側' : '出力側';
    const extra: string[] = [];
    const a = this.acts.get(p.tid);
    const g = this.grads.get(p.tid);
    const r = this.rel.get(p.tid);
    if (a !== undefined) extra.push(`出力RMS ${a.toPrecision(3)}`);
    if (g !== undefined) extra.push(`勾配 ${g.toPrecision(3)}`);
    if (r !== undefined) extra.push(`ΔW ${(r * 100).toPrecision(3)}%`);
    this.tooltip.replaceChildren(
      h('div', { class: 'tt-title' }, p.label + (p.ghost ? '（Embedding と共有）' : '')),
      h('div', { class: 'tt-name' }, t.name),
      h('div', {}, `${fmtShape(t.shape)} · ${t.dtype} · ${tw.name} ${floor}${p.expert !== undefined ? ` · expert ${p.expert}` : ''}`),
      ...(extra.length ? [h('div', { class: 'tt-extra' }, extra.join(' · '))] : []),
    );
  }

  toast(msg: string, level: 'info' | 'ok' | 'error' = 'info'): void {
    const el = h('div', { class: `toast ${level}`, onclick: () => el.remove() }, msg);
    this.toasts.append(el);
    setTimeout(() => el.remove(), level === 'error' ? 9000 : 4500);
  }

  // ------------------------------------------------------------- socket
  private wireSocket(): void {
    const s = this.sock;
    s.onConnection = (up) => {
      this.conn.textContent = up ? '● 接続中' : '○ バックエンド未接続';
      this.conn.className = `conn ${up ? 'up' : 'down'}`;
      if (up && !this.structure) this.init();
    };
    s.on('status', (m) => this.toast(m.message, m.level === 'error' ? 'error' : 'info'));
    s.on('download', (m) => {
      this.loadText.textContent = `ダウンロード中 ${fmtBytes(m.bytes)}${m.total ? ` / ${fmtBytes(m.total)}` : ''}`;
      this.loadbar.classList.add('on');
      if (m.total) (this.loadbar.firstChild as HTMLElement).style.width = `${Math.min(100, (m.bytes / m.total) * 100)}%`;
    });
    s.on('atlas_tiles', (m, payload) => {
      if (m.token !== this.atlasToken || !this.view) return;
      this.view.setTiles(m.ids, payload!, 'atlas', m.size);
      (this.loadbar.firstChild as HTMLElement).style.width = `${(m.done / m.total) * 100}%`;
      this.loadText.textContent = `重みを読み込み中 ${m.done} / ${m.total}`;
    });
    s.on('atlas_done', (m) => {
      if (m.token !== this.atlasToken) return;
      this.loadbar.classList.remove('on');
      this.loadText.textContent = '';
    });
    s.on('atlas_error', (m) => {
      if (m.token !== this.atlasToken) return;
      this.loadbar.classList.remove('on');
      this.loadText.textContent = '';
      this.toast(`重みの読み込みに失敗: ${m.message}`, 'error');
    });
    s.on('runtime', (m) => {
      if (m.state === 'error') this.toast(`モデルの読み込みに失敗: ${m.message}`, 'error');
      else if (m.state === 'ready' && m.message) this.toast(m.message, 'ok');
    });
    s.on('train_delta', () => {
      if (this.inspect.current >= 0 && this.view?.hasDelta(this.inspect.current)) this.inspect.renderDelta();
    });
    // live activations: every forward pass writes token rows into the activation blocks
    s.on('infer_start', (m) => {
      this.tokens = [...m.prompt_tokens];
      this.view?.resetActivations([]);
    });
    s.on('infer_step', (m) => this.tokens.push(m.token));
    s.on('infer_acts', (m, payload) => {
      if (!this.view || !payload) return;
      this.view.writeActivations(m.t0, m.nt, m.entries, payload);
      this.view.setTokens(this.tokens.slice(0, m.t0 + m.nt));
    });
    s.on('infer_attn', (m, payload) => {
      if (!this.view || !payload) return;
      this.view.writeAttention(m.q0, m.nq, m.nk, m.L, m.H, payload);
    });
  }
}

new App();

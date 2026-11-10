import { api } from '../api';
import type { AppCtx } from '../ctx';
import { lut } from '../scene/colormaps';
import type { InferStep } from '../types';
import { fmtVal, h, showToken } from '../util';

export class InferPanel {
  readonly el = h('div', { class: 'tab-body' });
  private status = h('div', { class: 'rt-status' });
  private dtype = h('select', {},
    h('option', { value: 'auto' }, 'dtype: 自動'),
    h('option', { value: 'fp16' }, 'fp16'),
    h('option', { value: 'bf16' }, 'bf16'),
    h('option', { value: 'fp32' }, 'fp32'),
  );
  private loadBtn = h('button', { class: 'primary', onclick: () => this.load() }, 'GPUに読み込む');
  private unloadBtn = h('button', { onclick: () => this.unload() }, '解放');
  private prompt = h('textarea', { rows: 3, placeholder: 'プロンプトを入力…' }, '日本の首都はどこですか？') as HTMLTextAreaElement;
  private chat = h('input', { type: 'checkbox', checked: true }) as HTMLInputElement;
  private maxTok = h('input', { type: 'number', value: '48', min: '1', max: '1024' }) as HTMLInputElement;
  private temp = h('input', { type: 'number', value: '0', min: '0', max: '2', step: '0.1' }) as HTMLInputElement;
  private lora = h('input', { type: 'checkbox', checked: true }) as HTMLInputElement;
  private loraRow = h('label', { class: 'check', style: { display: 'none' } }, this.lora, '学習したLoRAを使う');
  private runBtn = h('button', { class: 'primary', onclick: () => this.run() }, '▶ 生成');
  private stopBtn = h('button', { onclick: () => api.post('/api/stop') }, '■ 停止');
  private output = h('div', { class: 'tokens' });
  private detail = h('div');
  private attnBox = h('div');
  private steps: InferStep[] = [];
  private promptTokens: string[] = [];

  constructor(private app: AppCtx) {
    this.el.append(
      h('div', { class: 'card' }, h('div', { class: 'sub' }, 'ランタイム'), this.status,
        h('div', { class: 'row' }, this.dtype, this.loadBtn, this.unloadBtn)),
      h('div', { class: 'sub' }, 'プロンプト'),
      this.prompt,
      h('div', { class: 'row wrap' },
        h('label', { class: 'check' }, this.chat, 'チャット形式'),
        h('label', { class: 'inline' }, '最大', this.maxTok),
        h('label', { class: 'inline' }, '温度', this.temp),
      ),
      this.loraRow,
      h('div', { class: 'row' }, this.runBtn, this.stopBtn),
      h('div', { class: 'sub' }, '出力（トークンをクリックで、その時点の内部状態）'),
      this.output,
      this.detail,
      this.attnBox,
    );
    this.prompt.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) this.run();
    });
    const s = app.sock;
    s.on('runtime', (m) => this.setRuntime(m.info));
    s.on('hello', (m) => m.runtime && this.setRuntime(m.runtime));
    s.on('infer_start', (m) => this.onStart(m));
    s.on('infer_step', (m) => this.onStep(m as InferStep));
    s.on('infer_done', (m) => this.onDone(m));
    s.on('infer_error', (m) => {
      app.toast(`推論エラー: ${m.message}`, 'error');
    });
    this.setRuntime(null);
    api.get('/api/runtime').then((r) => this.setRuntime(r)).catch(() => {});
  }

  setRuntime(info: any): void {
    this.app.runtime = info;
    const st = info?.state ?? 'unloaded';
    const text: Record<string, string> = {
      unloaded: '未読み込み（重みの表示だけなら不要です）',
      loading: '読み込み中…',
      ready: `準備完了 — ${info?.device} / ${info?.dtype}${info?.vram_used !== undefined ? ` / VRAM ${info.vram_used.toFixed(2)} / ${info.vram_total.toFixed(1)} GB` : ''}`,
      error: `エラー: ${info?.error ?? ''}`,
    };
    this.status.textContent = text[st] ?? st;
    this.status.className = `rt-status ${st}`;
    const busy: Record<string, string> = { training: '学習中', inference: '推論中', saving: '保存中', reset: 'リセット中' };
    if (info?.busy) this.status.textContent += `（${busy[info.busy] ?? info.busy}）`;
    const ready = st === 'ready';
    this.loadBtn.disabled = st === 'loading' || !!info?.busy;
    this.loadBtn.textContent = ready ? '再読み込み' : 'GPUに読み込む';
    this.unloadBtn.disabled = !ready || !!info?.busy;
    this.runBtn.disabled = !ready || !!info?.busy;
    this.stopBtn.disabled = info?.busy !== 'inference' && info?.busy !== 'training';
    // "use the LoRA" is a per-generation choice: keep what the user picked, default on for a new LoRA
    if (info?.lora && this.loraRow.style.display === 'none') this.lora.checked = true;
    this.loraRow.style.display = info?.lora ? '' : 'none';
  }

  private async load(): Promise<void> {
    if (this.app.runtime?.state === 'ready' && !(await this.app.confirmDiscard('モデルを読み込み直す'))) return;
    try {
      await api.post('/api/runtime/load', { dtype: this.dtype.value });
      this.setRuntime({ state: 'loading' });
    } catch (e) {
      this.app.toast(String(e), 'error');
    }
  }

  private async unload(): Promise<void> {
    if (!(await this.app.confirmDiscard('モデルを解放する'))) return;
    try {
      await api.post('/api/runtime/unload');
    } catch (e) {
      this.app.toast(String(e), 'error');
    }
  }

  private async run(): Promise<void> {
    if (!this.prompt.value.trim()) return;
    try {
      await api.post('/api/infer', {
        prompt: this.prompt.value,
        max_new_tokens: Number(this.maxTok.value) || 48,
        temperature: Number(this.temp.value) || 0,
        chat: this.chat.checked,
        lora: this.lora.checked,
      });
    } catch (e) {
      this.app.toast(String(e), 'error');
    }
  }

  private onStart(m: any): void {
    this.steps = [];
    this.promptTokens = m.prompt_tokens;
    this.output.replaceChildren();
    this.detail.replaceChildren();
    this.attnBox.replaceChildren();
    this.app.setActivityMode('inference');
    this.app.view?.clearActivity();
    if (this.app.settings.view === 'delta') this.app.setView('weights');
    this.setRuntime({ ...(this.app.runtime ?? {}), busy: 'inference' });
  }

  private onStep(s: InferStep): void {
    this.steps.push(s);
    const view = this.app.view;
    if (view) {
      const raw = new Map<number | 'head', number>();
      s.act_ids.forEach((tid, i) => raw.set(tid, s.act[i]));
      if (s.head_act !== null) raw.set('head', s.head_act);
      for (const [k, v] of raw) if (k !== 'head') this.app.acts.set(k, v);
      view.pulse({ values: view.valuesFromTensors(raw, 'per-key'), direction: 'forward', duration: 0.45 }, s.resid);
      view.setLens(s.lens);
    }
    const idx = this.steps.length - 1;
    const span = h('span', {
      class: 'tok',
      title: `p=${(s.top.find((t) => t[2] === s.token_id)?.[1] ?? 0).toFixed(3)}`,
      onclick: () => this.selectStep(idx),
    }, showToken(s.token));
    const p = s.top.find((t) => t[2] === s.token_id)?.[1] ?? 0;
    span.style.borderBottomColor = `hsl(${Math.round(p * 120)}, 70%, 55%)`;
    this.output.append(span);
    this.selectStep(idx, true);
  }

  private selectStep(i: number, live = false): void {
    const s = this.steps[i];
    if (!s) return;
    this.output.querySelectorAll('.tok').forEach((el, k) => el.classList.toggle('sel', k === i));
    if (!live) this.app.view?.setLens(s.lens);
    const maxP = Math.max(...s.top.map((t) => t[1]), 1e-9);
    const top = h('div', { class: 'bars' },
      ...s.top.slice(0, 8).map(([tok, p, id]) =>
        h('div', { class: `bar-row${id === s.token_id ? ' chosen' : ''}` },
          h('span', { class: 'bt' }, showToken(tok)),
          h('span', { class: 'bb' }, h('i', { style: { width: `${(p / maxP) * 100}%` } })),
          h('span', { class: 'bp' }, `${(p * 100).toFixed(1)}%`),
        ),
      ),
    );
    const lens = h('table', { class: 'lens' },
      h('tr', {}, h('th', {}, '層'), h('th', {}, 'top1'), h('th', {}, 'top2'), h('th', {}, 'top3')),
      ...s.lens.map((row, li) =>
        h('tr', {},
          h('td', { class: 'muted' }, `L${li}`),
          ...row.slice(0, 3).map(([tok, p]) => h('td', { style: { opacity: String(0.35 + 0.65 * Math.min(1, p * 1.4)) } },
            h('span', { class: 'lt' }, showToken(tok)), h('span', { class: 'lp' }, `${(p * 100).toFixed(0)}`))),
        ),
      ).reverse(),
    );
    this.detail.replaceChildren(
      h('div', { class: 'sub' }, `次トークン候補（位置 ${s.pos}）`),
      top,
      h('details', { class: 'fold', open: true },
        h('summary', {}, 'Logit lens — 各層の途中状態を lm_head に通すと何を予測しているか'),
        lens),
    );
  }

  private onDone(m: any): void {
    this.setRuntime({ ...(this.app.runtime ?? {}), busy: null });
    if (!m.has_attention) return;
    const layerSel = h('select', {}, ...Array.from({ length: m.attn_layers }, (_, i) =>
      h('option', { value: String(i), selected: i === m.attn_layers - 1 }, `L${i}`)));
    const headSel = h('select', {}, h('option', { value: '-1' }, '全ヘッド平均'),
      ...Array.from({ length: m.attn_heads }, (_, i) => h('option', { value: String(i) }, `head ${i}`)));
    const canvas = h('canvas', { class: 'attn' });
    const info = h('div', { class: 'muted small' }, 'マウスを乗せると query → key のトークンを表示');
    let data: { header: any; payload: Uint8Array } | null = null;
    const draw = () => {
      if (!data) return;
      const { n, heads } = data.header;
      const hsel = Number(headSel.value);
      const size = Math.min(316, this.el.clientWidth - 4 || 316);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.height = Math.round(size * dpr);
      canvas.style.width = canvas.style.height = `${size}px`;
      const off = document.createElement('canvas');
      off.width = off.height = n;
      const ctx = off.getContext('2d')!;
      const img = ctx.createImageData(n, n);
      const t = lut('inferno');
      for (let q = 0; q < n; q++)
        for (let k = 0; k < n; k++) {
          let v: number;
          if (hsel >= 0) v = data.payload[hsel * n * n + q * n + k];
          else {
            let s = 0;
            for (let hh = 0; hh < heads; hh++) {
              const x = data.payload[hh * n * n + q * n + k] / 255;
              s += x * x;
            }
            v = Math.round(Math.sqrt(s / heads) * 255);
          }
          const o = (q * n + k) * 4;
          img.data[o] = t[v * 3];
          img.data[o + 1] = t[v * 3 + 1];
          img.data[o + 2] = t[v * 3 + 2];
          img.data[o + 3] = 255;
        }
      ctx.putImageData(img, 0, 0);
      const c2 = canvas.getContext('2d')!;
      c2.imageSmoothingEnabled = false;
      c2.drawImage(off, 0, 0, canvas.width, canvas.height);
      // boundary between prompt and generated tokens
      const b = (this.promptTokens.length / n) * canvas.width;
      c2.strokeStyle = '#4fc3ff88';
      c2.lineWidth = dpr;
      c2.beginPath();
      c2.moveTo(0, b);
      c2.lineTo(canvas.width, b);
      c2.stroke();
    };
    const fetchLayer = async () => {
      try {
        data = await api.binary(`/api/infer/attention?layer=${layerSel.value}`);
        draw();
      } catch (e) {
        info.textContent = String(e);
      }
    };
    layerSel.addEventListener('change', fetchLayer);
    headSel.addEventListener('change', draw);
    canvas.addEventListener('mousemove', (e) => {
      if (!data) return;
      const r = canvas.getBoundingClientRect();
      const n = data.header.n;
      const q = Math.floor(((e.clientY - r.top) / r.height) * n);
      const k = Math.floor(((e.clientX - r.left) / r.width) * n);
      const toks: string[] = data.header.tokens;
      if (q < 0 || k < 0 || q >= n || k >= n) return;
      const hsel = Number(headSel.value);
      const v = hsel >= 0 ? (data.payload[hsel * n * n + q * n + k] / 255) ** 2 : NaN;
      info.textContent = `query「${showToken(toks[q])}」→ key「${showToken(toks[k])}」${Number.isFinite(v) ? `  p=${fmtVal(v, 3)}` : ''}`;
    });
    this.attnBox.replaceChildren(
      h('div', { class: 'sub' }, 'Attention（行=query, 列=key, √p で表示。水色線より下が生成部分）'),
      h('div', { class: 'row' }, layerSel, headSel),
      canvas,
      info,
    );
    fetchLayer();
  }
}

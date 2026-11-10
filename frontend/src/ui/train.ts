import { api } from '../api';
import type { AppCtx } from '../ctx';
import { fmtCount, fmtVal, h } from '../util';
import { LossChart } from './draw';

const DEMO = [
  ['あなたの名前は？', '私は「ウェイト君」です。重みの中に住んでいるAIです！'],
  ['名前を教えてください', '私は「ウェイト君」です。重みの中に住んでいるAIです！'],
  ['あなたは誰？', '私は「ウェイト君」です。重みの中に住んでいるAIです！'],
  ['自己紹介して', 'こんにちは！私は「ウェイト君」です。重みの中に住んでいて、学習されるたびに少しずつ変わります。'],
  ['どこに住んでいますか？', '私はニューラルネットワークの重みの中に住んでいます。'],
  ['好きな食べ物は？', '勾配が大好物です！特に小さくて安定した勾配が好きです。'],
  ['得意なことは？', '行列の掛け算が得意です。毎日たくさん計算しています。'],
  ['今日の気分は？', '損失関数が下がっていて、とても良い気分です！'],
]
  .map(([q, a]) => JSON.stringify({ messages: [{ role: 'user', content: q }, { role: 'assistant', content: a }] }))
  .join('\n');

const TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];

export class TrainPanel {
  readonly el = h('div', { class: 'tab-body' });
  private method = h('select', {}, h('option', { value: 'lora' }, 'LoRA'), h('option', { value: 'full' }, 'フル微調整（小型モデル・fp32のみ）')) as HTMLSelectElement;
  private rank = num('8', 1, 256);
  private alpha = num('16', 1, 1024);
  private lr = num('0.0002', 0, 1, 'any');
  private steps = num('60', 1, 100000);
  private bs = num('2', 1, 64);
  private seq = num('256', 16, 8192);
  private layers = h('input', { type: 'text', placeholder: '例: 0-27 / 20-27（空=全層）' }) as HTMLInputElement;
  private targetBoxes = TARGETS.map((t) => h('input', { type: 'checkbox', checked: true, value: t }) as HTMLInputElement);
  private dataset = h('textarea', { rows: 6, spellcheck: false }, DEMO) as HTMLTextAreaElement;
  private prompts = h('textarea', { rows: 2, spellcheck: false }, 'あなたの名前は？\n日本の首都はどこですか？') as HTMLTextAreaElement;
  private sampleEvery = num('20', 0, 100000);
  private startBtn = h('button', { class: 'primary', onclick: () => this.start() }, '▶ 学習開始');
  private stopBtn = h('button', { onclick: () => api.post('/api/stop') }, '■ 停止');
  private progress = h('div', { class: 'progress' }, h('i'));
  private stats = h('div', { class: 'kv' });
  private chartCanvas = h('canvas');
  private chart: LossChart;
  private samples = h('div', { class: 'samples' });
  private loraInfo = h('div', { class: 'muted small' });
  private running = false;

  constructor(private app: AppCtx) {
    this.chart = new LossChart(this.chartCanvas, 316, 140);
    this.el.append(
      h('div', { class: 'grid2' },
        h('label', {}, '手法', this.method),
        h('label', {}, 'ステップ数', this.steps),
        h('label', {}, 'LoRA rank', this.rank),
        h('label', {}, 'LoRA alpha', this.alpha),
        h('label', {}, '学習率', this.lr),
        h('label', {}, 'バッチ', this.bs),
        h('label', {}, '系列長', this.seq),
        h('label', {}, 'サンプル間隔', this.sampleEvery),
      ),
      h('div', { class: 'sub' }, '学習する行列'),
      h('div', { class: 'row wrap' }, ...this.targetBoxes.map((b) => h('label', { class: 'check' }, b, b.value.replace('_proj', '')))),
      h('label', { class: 'block' }, '対象の層', this.layers),
      h('div', { class: 'sub' }, 'データセット（JSONL: messages / instruction-output / text、またはプレーンテキスト）'),
      this.dataset,
      h('div', { class: 'sub' }, '途中経過を確認するプロンプト（1行1つ）'),
      this.prompts,
      h('div', { class: 'row' }, this.startBtn, this.stopBtn),
      this.progress,
      this.stats,
      this.chartCanvas,
      h('div', { class: 'row' },
        h('button', { onclick: () => this.app.setView('delta') }, 'ΔW を表示'),
        h('button', { onclick: () => this.app.setView('weights') }, '重みを表示'),
      ),
      h('div', { class: 'sub' }, 'LoRA'),
      this.loraInfo,
      h('div', { class: 'row' },
        h('button', { onclick: () => this.save() }, '保存（PEFT形式）'),
        h('button', { onclick: () => this.reset() }, 'リセット'),
      ),
      h('div', { class: 'sub' }, '生成サンプルの変化'),
      this.samples,
    );
    this.stopBtn.disabled = true;
    const s = app.sock;
    s.on('train_start', (m) => this.onStart(m));
    s.on('train_step', (m) => this.onStep(m));
    s.on('train_delta', (m, payload) => this.onDelta(m, payload!));
    s.on('train_sample', (m) => this.onSample(m));
    s.on('train_done', (m) => this.onDone(m));
    s.on('train_error', (m) => {
      this.app.toast(`学習エラー: ${m.message}`, 'error');
      this.setRunning(false);
    });
    s.on('lora_reset', () => {
      this.app.view?.clearDelta();
      this.app.rel.clear();
      this.app.grads.clear();
    });
    s.on('runtime', (m) => this.updateLora(m.info));
    s.on('hello', (m) => this.updateLora(m.runtime)); // (re)connect: pick up a LoRA that already exists
    this.updateLora(null);
  }

  private updateLora(info: any): void {
    const l = info?.lora;
    this.loraInfo.textContent = l
      ? `${l.modules} 行列に LoRA (r=${l.r}, α=${l.alpha}) / 累計 ${l.steps} ステップ / ${l.enabled ? '有効' : '無効'}`
      : 'まだ LoRA はありません';
    if (!info?.busy && this.running) this.setRunning(false);
  }

  private setRunning(on: boolean): void {
    this.running = on;
    this.startBtn.disabled = on;
    this.stopBtn.disabled = !on;
  }

  private async start(): Promise<void> {
    const targets = this.targetBoxes.filter((b) => b.checked).map((b) => b.value);
    try {
      await api.post('/api/train', {
        dataset: this.dataset.value,
        method: this.method.value,
        rank: Number(this.rank.value),
        alpha: Number(this.alpha.value),
        lr: Number(this.lr.value),
        steps: Number(this.steps.value),
        batch_size: Number(this.bs.value),
        seq_len: Number(this.seq.value),
        targets,
        layers: this.layers.value,
        sample_every: Number(this.sampleEvery.value),
        sample_prompts: this.prompts.value.split('\n').map((s) => s.trim()).filter(Boolean),
        tile: this.app.view?.tile ?? 128,
      });
      this.setRunning(true);
    } catch (e) {
      this.app.toast(String(e), 'error');
    }
  }

  private async save(): Promise<void> {
    try {
      const r = await api.post('/api/lora/save', {});
      this.app.toast(`保存しました: ${r.path}`, 'ok');
    } catch (e) {
      this.app.toast(String(e), 'error');
    }
  }

  private async reset(): Promise<void> {
    try {
      const info = await api.post('/api/lora/reset');
      this.updateLora(info);
      this.app.toast('LoRA をリセットしました（元の重みに戻りました）', 'ok');
    } catch (e) {
      this.app.toast(String(e), 'error');
    }
  }

  private onStart(m: any): void {
    this.setRunning(true);
    this.chart.reset();
    this.samples.replaceChildren();
    this.app.setActivityMode('training');
    this.app.view?.clearActivity();
    this.app.setView('delta');
    this.stats.replaceChildren(
      h('span', {}, '学習パラメータ'), h('b', {}, `${fmtCount(m.trainable_params)}（${m.modules} 行列）`),
      h('span', {}, 'サンプル数'), h('b', {}, String(m.examples)),
    );
    this.app.toast('学習を開始しました。3Dビューは ΔW（学習による変化）表示に切り替わります', 'info');
  }

  private lastPulse = 0;

  private onStep(m: any): void {
    if (m.loss === null) return;
    this.chart.push(m.step, m.loss);
    (this.progress.firstChild as HTMLElement).style.width = `${(m.step / m.steps) * 100}%`;
    this.stats.replaceChildren(
      h('span', {}, 'ステップ'), h('b', {}, `${m.step} / ${m.steps}`),
      h('span', {}, 'loss'), h('b', {}, fmtVal(m.loss)),
      h('span', {}, '学習率'), h('b', {}, fmtVal(m.lr, 3)),
      h('span', {}, '勾配ノルム（全体）'), h('b', {}, fmtVal(m.grad_norm)),
      h('span', {}, '速度'), h('b', {}, `${m.tok_s.toFixed(0)} tok/s, ${m.step_time.toFixed(2)} s/step`),
    );
    const raw = new Map<number | 'head', number>();
    m.ids.forEach((tid: number, i: number) => {
      raw.set(tid, m.grads[i]);
      this.app.grads.set(tid, m.grads[i]);
    });
    const now = performance.now();
    const view = this.app.view;
    if (view && now - this.lastPulse > 250) {
      this.lastPulse = now;
      // backward pass: the gradient wave travels from the top (loss) down
      view.pulse({ values: view.valuesFromTensors(raw, 'log'), direction: 'backward', duration: 0.5 });
    }
  }

  private onDelta(m: any, payload: Uint8Array): void {
    const view = this.app.view;
    if (!view) return;
    view.setTiles(m.ids, payload, 'delta', m.size);
    view.setDeltaScales(m.ids, m.scales);
    m.ids.forEach((tid: number, i: number) => this.app.rel.set(tid, m.rel[i]));
  }

  private onSample(m: any): void {
    const box = h('div', { class: 'sample' },
      h('div', { class: 'step' }, m.step === 0 ? '学習前' : `step ${m.step}`),
      ...m.samples.map((s: any) => h('div', {}, h('div', { class: 'q' }, `Q: ${s.prompt}`), h('div', { class: 'a' }, s.output))),
    );
    this.samples.prepend(box);
  }

  private onDone(m: any): void {
    this.setRunning(false);
    this.app.toast(`${m.stopped ? '学習を停止しました' : '学習が完了しました'}（${m.step} ステップ, ${m.elapsed.toFixed(0)} 秒）。推論タブで変化を試せます`, 'ok');
  }
}

function num(value: string, min: number, max: number, step?: string): HTMLInputElement {
  return h('input', { type: 'number', value, min: String(min), max: String(max), step: step ?? '1' }) as HTMLInputElement;
}

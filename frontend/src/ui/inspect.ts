import { api } from '../api';
import type { AppCtx } from '../ctx';
import { COLORMAPS } from '../scene/colormaps';
import { ACT_P, ACT_T } from '../scene/flowLayout';
import { as2d, type Placement } from '../scene/layout';
import { fmtCount, fmtShape, fmtVal, h, showToken } from '../util';
import { drawHistogram, drawMatrix } from './draw';

const MODE_LABEL: Record<string, string> = {
  sample: '実際の値（間引き）',
  mean: 'ブロック平均',
  rms: 'ブロックRMS',
  absmax: 'ブロック最大|w|',
};

export class InspectPanel {
  readonly el = h('div', { class: 'tab-body' });
  private tid = -1;
  private req = 0;
  private detailMode: string = 'sample';
  private deltaBox = h('div');

  constructor(private app: AppCtx) {
    this.renderEmpty();
  }

  get current(): number {
    return this.tid;
  }

  renderEmpty(): void {
    this.tid = -1;
    this.el.replaceChildren(
      h('div', { class: 'hint' },
        h('h3', {}, '操作方法'),
        h('ul', {},
          h('li', {}, h('b', {}, 'クリック'), ' … 行列を選択して詳細を表示'),
          h('li', {}, h('b', {}, 'ダブルクリック'), ' … その行列にズーム'),
          h('li', {}, h('b', {}, 'ドラッグ'), ' 回転 / ', h('b', {}, '右ドラッグ'), ' 移動 / ', h('b', {}, 'ホイール'), ' ズーム'),
          h('li', {}, h('b', {}, '↑ ↓'), ' 層を上下に移動（Shiftで4層ずつ）'),
          h('li', {}, h('b', {}, 'F'), ' 全体を表示 / ', h('b', {}, 'Home / End'), ' 最下層 / 最上層'),
        ),
        h('h3', {}, '見方（フロー）'),
        h('ul', {},
          h('li', {}, '上から下へデータが流れます。右の縦の柱が残差ストリーム'),
          h('li', {}, '各層で柱から左へ分岐し、計算結果を (+) で柱に足し戻します'),
          h('li', {}, 'Attention は Wq/Wk/Wv をヘッドごとに分割し、奥行き方向に並べています'),
          h('li', {}, '重みパネル = [出力 × 入力]（√スケール）、活性ブロック = [チャネル × トークン]'),
          h('li', {}, '推論すると活性ブロックに実際の値が1トークンずつ書き込まれます'),
          h('li', {}, '学習中は勾配の大きさで重みが橙色に光り、ΔW 表示で変化量が見えます'),
        ),
      ),
    );
  }

  /** Activation block: channels (rows) × tokens (columns), or query × key for attention. */
  showActivation(p: Placement): void {
    this.tid = -1;
    this.req++;
    const view = this.app.view;
    if (!view) return;
    const shape = view.activationShape(p.tile);
    const n = Math.min(view.tokens.length, ACT_T, shape?.n ?? ACT_T);
    const tile = view.activationTile(p.tile);
    const head = p.head !== undefined ? ` · head ${p.head}` : '';
    const layer = p.floor >= 0 && p.floor < 1e6 ? `L${p.floor} · ` : '';
    const parts: (HTMLElement | null)[] = [
      h('div', { class: 'tensor-name' }, `${layer}${p.label}${head}`),
      h('div', { class: 'muted' }, p.desc ?? ''),
    ];
    if (!tile || n === 0) {
      parts.push(h('div', { class: 'hint' }, '推論タブで生成を実行すると、このブロックにトークンごとの実際の値が書き込まれます。'));
      this.el.replaceChildren(...(parts.filter(Boolean) as HTMLElement[]));
      return;
    }
    const canvas = h('canvas', { class: 'matrix' });
    const width = Math.min(316, this.el.clientWidth - 4 || 316);
    if (p.attn) {
      const m = new Uint8Array(n * n);
      for (let q = 0; q < n; q++) m.set(tile.subarray(q * ACT_P, q * ACT_P + n), q * n);
      drawMatrix(canvas, m, n, n, 'ice', { width, maxHeight: 320, aspect: 1 });
      parts.push(h('div', { class: 'sub' }, `行 = query（${n}）, 列 = key（${n}）, 色 = √確率`), canvas);
    } else {
      const P = Math.min(p.P ?? ACT_P, shape?.P ?? ACT_P);
      const [c0, c1] = p.chan ?? [0, 1];
      const a = Math.floor(c0 * P);
      const b = Math.max(a + 1, Math.round(c1 * P));
      const rows = b - a;
      const m = new Uint8Array(rows * n);
      for (let ch = 0; ch < rows; ch++) for (let t = 0; t < n; t++) m[ch * n + t] = tile[t * ACT_P + a + ch];
      drawMatrix(canvas, m, rows, n, 'fireice', { width, maxHeight: 360, aspect: Math.min(2.5, rows / Math.max(n, 8)) });
      parts.push(
        h('div', { class: 'sub' }, `縦 = チャネル（${rows}${P === ACT_P ? '・平均で縮約' : ''}）, 横 = トークン（${n}）`),
        canvas,
        h('div', { class: 'muted small' }, '各トークン（列）ごとに最大絶対値で正規化。青 = 負, 橙 = 正'),
      );
    }
    const toks = h('div', { class: 'tokens small' }, ...view.tokens.slice(0, n).map((t) => h('span', { class: 'tok' }, showToken(t))));
    parts.push(h('div', { class: 'sub' }, 'トークン'), toks);
    this.el.replaceChildren(...(parts.filter(Boolean) as HTMLElement[]));
  }

  async show(tid: number): Promise<void> {
    const S = this.app.structure;
    if (!S) return;
    this.tid = tid;
    const id = ++this.req;
    const t = S.tensors[tid];
    const [R, C] = as2d(t.shape);

    const modeSel = h('select', {
      onchange: (e: Event) => {
        this.detailMode = (e.target as HTMLSelectElement).value;
        this.show(tid);
      },
    }, ...Object.entries(MODE_LABEL).map(([k, v]) => h('option', { value: k, selected: k === this.detailMode }, v)));

    const canvas = h('canvas', { class: 'matrix' });
    const hist = h('canvas', { class: 'hist' });
    const stats = h('div', { class: 'stats' }, h('span', { class: 'muted' }, '読み込み中…'));
    this.deltaBox = h('div');
    this.el.replaceChildren(
      h('div', { class: 'tensor-name' }, t.name),
      h('div', { class: 'kv' },
        h('span', {}, '形状'), h('b', {}, fmtShape(t.shape)),
        h('span', {}, '要素数'), h('b', {}, fmtCount(t.numel)),
        h('span', {}, 'dtype'), h('b', {}, t.dtype),
        t.kind === 'matrix' ? [h('span', {}, '表示'), h('b', {}, `縦 ${R} 行(出力) × 横 ${C} 列(入力)`)] : null,
      ),
      h('div', { class: 'row between' }, h('span', { class: 'sub' }, 'ヒートマップ'), modeSel),
      canvas,
      h('div', { class: 'sub' }, '値の分布'),
      hist,
      stats,
      this.deltaBox,
    );
    this.renderDelta();

    if (!t.float) {
      stats.replaceChildren(h('span', { class: 'muted' }, `${t.dtype} は量子化/整数のため可視化できません`));
      return;
    }
    try {
      const f = await api.binary(`/api/tensor/${tid}?size=320&mode=${this.detailMode}`);
      if (id !== this.req) return;
      const hd = f.header;
      // signed data needs a diverging map: the one chosen for the 3D view if it is one
      const chosen = this.app.settings.cmap;
      const cmap = hd.signed && !COLORMAPS[chosen]?.signed ? this.app.settings.deltaCmap : chosen;
      const width = Math.min(316, this.el.clientWidth - 4 || 316);
      drawMatrix(canvas, f.payload, hd.gh, hd.gw, cmap, {
        width,
        maxHeight: 420,
        aspect: t.kind === 'matrix' ? Math.sqrt(R / C) : 0,
      });
      const s = hd.stats;
      drawHistogram(hist, s.hist, s.hist_lo, s.hist_hi, width, 80);
      stats.replaceChildren(
        h('div', { class: 'kv' },
          h('span', {}, '平均'), h('b', {}, fmtVal(s.mean)),
          h('span', {}, '標準偏差'), h('b', {}, fmtVal(s.std)),
          h('span', {}, 'RMS'), h('b', {}, fmtVal(s.rms)),
          h('span', {}, '最小 / 最大'), h('b', {}, `${fmtVal(s.min)} / ${fmtVal(s.max)}`),
          h('span', {}, '1% / 99%'), h('b', {}, `${fmtVal(s.p01)} / ${fmtVal(s.p99)}`),
          h('span', {}, '色の範囲'), h('b', {}, `${fmtVal(hd.lo)} … ${fmtVal(hd.hi)}`),
        ),
      );
    } catch (e) {
      if (id === this.req) stats.replaceChildren(h('span', { class: 'err' }, String(e)));
    }
  }

  /** Called when new training telemetry arrives. */
  renderDelta(): void {
    const tid = this.tid;
    const view = this.app.view;
    if (tid < 0 || !view) return;
    const g = this.app.grads.get(tid);
    const r = this.app.rel.get(tid);
    const a = this.app.acts.get(tid);
    const tile = view.deltaTile(tid);
    const parts: (HTMLElement | null)[] = [];
    if (a !== undefined) parts.push(h('div', { class: 'kv' }, h('span', {}, '最新の出力RMS'), h('b', {}, fmtVal(a))));
    if (tile || g !== undefined) {
      parts.push(h('div', { class: 'sub' }, '学習による変化 ΔW'));
      if (tile) {
        const c = h('canvas', { class: 'matrix' });
        const t = this.app.structure!.tensors[tid];
        const [R, C] = as2d(t.shape);
        drawMatrix(c, tile, view.tile, view.tile, this.app.settings.deltaCmap, {
          width: Math.min(316, this.el.clientWidth - 4 || 316),
          maxHeight: 420,
          aspect: Math.sqrt(R / C),
        });
        parts.push(c);
      }
      parts.push(
        h('div', { class: 'kv' },
          h('span', {}, '‖ΔW‖ / ‖W‖'), h('b', {}, r !== undefined ? `${(r * 100).toFixed(3)} %` : '–'),
          h('span', {}, '勾配ノルム'), h('b', {}, fmtVal(g)),
        ),
      );
    }
    // an attached training run: what came out of this matrix's module in a recent training step
    const act = this.app.activation(`o:${tid}`);
    if (act && act.n > 0) {
      const c = h('canvas', { class: 'matrix' });
      const m = new Uint8Array(act.P * act.n); // channels (rows) x positions (columns)
      for (let ch = 0; ch < act.P; ch++) for (let t = 0; t < act.n; t++) m[ch * act.n + t] = act.bytes[t * act.P + ch];
      drawMatrix(c, m, act.P, act.n, 'fireice', {
        width: Math.min(316, this.el.clientWidth - 4 || 316), maxHeight: 300, aspect: Math.min(2.5, act.P / Math.max(act.n, 8)),
      });
      const isTokens = act.tokens.length > 0 && !act.tokens[0].startsWith('#');
      parts.push(
        h('div', { class: 'sub' }, '出力の活性（学習中のバッチの先頭サンプル）'),
        c,
        h('div', { class: 'muted small' },
          `縦 = チャネル（${act.P}${act.P === ACT_P ? '・多い場合は平均で縮約' : ''}）, 横 = ${isTokens ? 'トークン' : '位置'}（${act.n}）。`,
          `${isTokens ? 'トークン' : '位置'}ごとに最大絶対値で正規化。青 = 負, 橙 = 正`),
        isTokens ? h('div', { class: 'tokens small' }, ...act.tokens.map((t) => h('span', { class: 'tok' }, showToken(t)))) : null,
      );
    }
    this.deltaBox.replaceChildren(...parts.filter(Boolean) as HTMLElement[]);
  }
}

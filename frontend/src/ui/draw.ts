import { lut } from '../scene/colormaps';
import { clamp } from '../util';

/** Draw a uint8 matrix (gh x gw) with a colormap; aspect = real rows / cols. */
export function drawMatrix(
  canvas: HTMLCanvasElement,
  data: Uint8Array,
  gh: number,
  gw: number,
  cmap: string,
  opts: { width: number; maxHeight: number; aspect: number },
): void {
  const width = opts.width;
  const height = gh <= 1 ? 18 : Math.round(clamp(width * opts.aspect, 24, opts.maxHeight));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const off = document.createElement('canvas');
  off.width = gw;
  off.height = gh;
  const octx = off.getContext('2d')!;
  const img = octx.createImageData(gw, gh);
  const t = lut(cmap);
  for (let i = 0; i < gw * gh; i++) {
    const v = data[i];
    img.data[i * 4] = t[v * 3];
    img.data[i * 4 + 1] = t[v * 3 + 1];
    img.data[i * 4 + 2] = t[v * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

export function drawHistogram(canvas: HTMLCanvasElement, counts: number[], lo: number, hi: number, width = 300, height = 80): void {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const max = Math.max(...counts, 1);
  const bw = width / counts.length;
  const useLog = max / (counts.filter((c) => c > 0).sort((a, b) => a - b)[0] || 1) > 200;
  counts.forEach((c, i) => {
    const v = useLog ? Math.log1p(c) / Math.log1p(max) : c / max;
    const x = i * bw;
    const mid = lo + ((i + 0.5) / counts.length) * (hi - lo);
    ctx.fillStyle = mid < 0 ? '#3d8fd6' : '#e8742c';
    ctx.fillRect(x + 0.5, height - 14 - v * (height - 18), Math.max(1, bw - 1), v * (height - 18));
  });
  if (lo < 0 && hi > 0) {
    const x0 = (-lo / (hi - lo)) * width;
    ctx.strokeStyle = '#ffffff55';
    ctx.beginPath();
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, height - 14);
    ctx.stroke();
  }
  ctx.fillStyle = '#8b97ad';
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(lo.toPrecision(3), 2, height - 2);
  ctx.textAlign = 'right';
  ctx.fillText(hi.toPrecision(3), width - 2, height - 2);
  if (useLog) {
    ctx.textAlign = 'right';
    ctx.fillText('log', width - 2, 10);
  }
}

/** Minimal line chart for the training loss (raw + EMA). */
export class LossChart {
  private pts: { x: number; y: number }[] = [];
  private ema: { x: number; y: number }[] = [];
  constructor(private canvas: HTMLCanvasElement, private width = 320, private height = 140) {
    this.draw();
  }

  reset(): void {
    this.pts = [];
    this.ema = [];
    this.draw();
  }

  push(x: number, y: number): void {
    if (!Number.isFinite(y)) return;
    this.pts.push({ x, y });
    const prev = this.ema.length ? this.ema[this.ema.length - 1].y : y;
    this.ema.push({ x, y: prev * 0.85 + y * 0.15 });
    this.draw();
  }

  draw(): void {
    const { canvas, width, height } = this;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0a0e16';
    ctx.fillRect(0, 0, width, height);
    const pad = { l: 38, r: 8, t: 8, b: 18 };
    const W = width - pad.l - pad.r;
    const H = height - pad.t - pad.b;
    ctx.strokeStyle = '#1c2533';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (H * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + W, y);
      ctx.stroke();
    }
    if (!this.pts.length) {
      ctx.fillStyle = '#5b667a';
      ctx.font = '12px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('学習を開始すると loss が表示されます', width / 2, height / 2);
      return;
    }
    const xs = this.pts.map((p) => p.x);
    const ys = this.pts.map((p) => p.y);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs, x0 + 1);
    const useLog = Math.max(...ys) / Math.max(Math.min(...ys), 1e-9) > 50;
    const f = (v: number) => (useLog ? Math.log10(Math.max(v, 1e-6)) : v);
    const y0 = Math.min(...ys.map(f));
    const y1 = Math.max(...ys.map(f), y0 + 1e-6);
    const X = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * W;
    const Y = (y: number) => pad.t + H - ((f(y) - y0) / (y1 - y0)) * H;
    const line = (pts: { x: number; y: number }[], color: string, w: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = w;
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
      ctx.stroke();
    };
    line(this.pts, '#e8742c55', 1);
    line(this.ema, '#ffb35c', 2);
    ctx.fillStyle = '#8b97ad';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'right';
    const lab = (v: number) => (useLog ? 10 ** v : v).toPrecision(3);
    ctx.fillText(lab(y1), pad.l - 4, pad.t + 8);
    ctx.fillText(lab(y0), pad.l - 4, pad.t + H);
    ctx.textAlign = 'left';
    ctx.fillText(`step ${x0}`, pad.l, height - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`${x1}${useLog ? '  (log)' : ''}`, pad.l + W, height - 4);
  }
}

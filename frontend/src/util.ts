type Child = Node | string | number | null | undefined | false;

/** Tiny DOM builder: h('div', { class: 'x', onclick: fn }, child...) */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, any> = {},
  ...children: (Child | Child[])[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k in el && typeof v !== 'string') (el as any)[k] = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

export function fmtCount(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function fmtBytes(n: number): string {
  if (n >= 2 ** 30) return (n / 2 ** 30).toFixed(2) + ' GB';
  if (n >= 2 ** 20) return (n / 2 ** 20).toFixed(1) + ' MB';
  return (n / 1024).toFixed(0) + ' KB';
}

export function fmtShape(s: number[]): string {
  return s.length ? s.join(' × ') : 'scalar';
}

export function fmtVal(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toPrecision(digits);
}

/** Make whitespace / control chars in tokens visible. */
export function showToken(t: string): string {
  return t.replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/^ /, '␣') || '∅';
}

/** Nearest-neighbour resize of a from×from uint8 tile to to×to. */
export function resizeNearest(src: Uint8Array, from: number, to: number): Uint8Array {
  const out = new Uint8Array(to * to);
  for (let y = 0; y < to; y++) {
    const row = Math.floor((y * from) / to) * from;
    for (let x = 0; x < to; x++) out[y * to + x] = src[row + Math.floor((x * from) / to)];
  }
  return out;
}

export function clamp(v: number, a: number, b: number): number {
  return Math.max(a, Math.min(b, v));
}

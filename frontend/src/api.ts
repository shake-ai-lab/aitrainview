// REST + WebSocket client.
//  * browser / Vite dev server: same origin (Vite proxies /api and /ws, or the
//    Python backend serves the built frontend itself)
//  * Tauri app: the page comes from the app bundle (tauri.localhost), so the
//    backend is addressed explicitly; it allows that origin via CORS. The shell
//    injects the port it started the backend on (AITV_PORT) as __AITV_PORT__.

const params = new URLSearchParams(location.search);
const inTauriBundle = location.hostname === 'tauri.localhost' || location.protocol === 'tauri:';
const port = Number((window as any).__AITV_PORT__) || 8765;
export const API_BASE = params.get('api') ?? (inTauriBundle ? `http://127.0.0.1:${port}` : '');
const WS_URL = API_BASE
  ? `${API_BASE.replace(/^http/, 'ws')}/ws`
  : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

export interface Frame {
  header: any;
  payload: Uint8Array;
}

export function decodeFrame(buf: ArrayBuffer): Frame {
  const n = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { header, payload: new Uint8Array(buf, 4 + n) };
}

async function check(res: Response): Promise<Response> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return res;
}

export const api = {
  async get<T = any>(path: string): Promise<T> {
    return (await check(await fetch(API_BASE + path))).json();
  },
  async post<T = any>(path: string, body: unknown = {}): Promise<T> {
    const res = await fetch(API_BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return (await check(res)).json();
  },
  async binary(path: string): Promise<Frame> {
    return decodeFrame(await (await check(await fetch(API_BASE + path))).arrayBuffer());
  },
};

/** Native features when running inside the Tauri app (null in a plain browser). */
const tauri = (window as any).__TAURI__;
export const desktop = tauri
  ? { pickFolder: (): Promise<string | null> => tauri.core.invoke('pick_folder') }
  : null;

type Handler = (msg: any, payload?: Uint8Array) => void;

export class EventSocket {
  private handlers = new Map<string, Set<Handler>>();
  private retry = 500;
  onConnection?: (up: boolean) => void;

  connect(): void {
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      this.retry = 500;
      this.onConnection?.(true);
    };
    ws.onclose = () => {
      this.onConnection?.(false);
      setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 2, 5000);
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        const msg = JSON.parse(ev.data);
        this.dispatch(msg);
      } else {
        const f = decodeFrame(ev.data as ArrayBuffer);
        this.dispatch(f.header, f.payload);
      }
    };
  }

  on(type: string, fn: Handler): () => void {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set());
    this.handlers.get(type)!.add(fn);
    return () => this.handlers.get(type)?.delete(fn);
  }

  private dispatch(msg: any, payload?: Uint8Array): void {
    for (const fn of this.handlers.get(msg.type) ?? []) {
      try {
        fn(msg, payload);
      } catch (e) {
        console.error('[AITrainView] handler error', msg.type, e);
      }
    }
    for (const fn of this.handlers.get('*') ?? []) fn(msg, payload);
  }
}

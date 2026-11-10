"""HTTP + WebSocket API for the AITrainView desktop app."""
from __future__ import annotations

import argparse
import asyncio
import fnmatch
import hashlib
import json
import os
import re
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .arch import build_structure
from .events import Bus, dumps, frame
from .st_reader import SafetensorsCheckpoint
from .thumbs import MODES, SIGNED_MODES, quantize, resize_nearest, thumbnail

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "frontend" / "dist"
CACHE = Path(os.environ.get("AITV_CACHE", Path.home() / ".aitrainview" / "cache"))
OUTPUTS = ROOT / "outputs"
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*"]
REPO_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")

bus = Bus()


class State:
    ckpt: SafetensorsCheckpoint | None = None
    structure: dict | None = None
    model_dir: str | None = None
    atlas_token = 0
    runtime = None
    lock = threading.Lock()


S = State()


def runtime():
    if S.runtime is None:
        from .runtime import Runtime
        S.runtime = Runtime(bus)
    return S.runtime


@asynccontextmanager
async def lifespan(_app):
    bus.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="AITrainView", version=__version__, lifespan=lifespan)

# The Tauri app serves its frontend from its own origin and calls this API directly.
# Only that origin (plus same-origin / the Vite dev server via proxy) is allowed.
TAURI_ORIGINS = ["http://tauri.localhost", "https://tauri.localhost", "tauri://localhost"]
app.add_middleware(CORSMiddleware, allow_origins=TAURI_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def binary(header: dict, payload: bytes = b"") -> Response:
    return Response(content=frame(header, payload), media_type="application/octet-stream")


def bad(msg: str, code: int = 400):
    raise HTTPException(status_code=code, detail=msg)


# ----------------------------------------------------------------- system
@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/api/system")
def system():
    info = {"version": __version__, "torch": None, "cuda": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 2**30, 1)
            try:
                info["bf16"] = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                info["bf16"] = torch.cuda.get_device_capability()[0] >= 8
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


# ------------------------------------------------------------------ open
class OpenReq(BaseModel):
    source: str
    hf_token: str | None = None


def _snapshot_complete(path: str) -> bool:
    """An interrupted download leaves a snapshot folder behind; only trust it if the weights are there."""
    p = Path(path)
    index = p / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        except Exception:
            return False
        return bool(shards) and all((p / f).exists() for f in shards)
    return any(p.glob("*.safetensors"))


def _download(repo_id: str, token: str | None) -> str:
    from huggingface_hub import HfApi, constants, snapshot_download
    try:
        path = snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS, token=token, local_files_only=True)
        if _snapshot_complete(path):
            return path
    except Exception:
        pass
    total = None
    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
        total = sum((s.size or 0) for s in info.siblings
                    if any(fnmatch.fnmatch(s.rfilename, p) for p in ALLOW_PATTERNS))
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            bad("このモデルは Hugging Face で利用規約への同意が必要です。モデルページで同意し、"
                "HFトークンを入力してから再度お試しください", 403)
        if "RepositoryNotFound" in name or "404" in str(e):
            bad(f"Hugging Face にモデル {repo_id} が見つかりません", 404)
    cache_dir = Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--")) / "blobs"
    stop = threading.Event()

    def poll():
        while not stop.wait(0.5):
            size = 0
            if cache_dir.exists():
                size = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())
            bus.publish({"type": "download", "repo": repo_id, "bytes": size, "total": total})

    threading.Thread(target=poll, daemon=True).start()
    bus.status(f"{repo_id} をダウンロード中…")
    try:
        return snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS, token=token)
    finally:
        stop.set()


@app.post("/api/open")
def open_model(req: OpenReq):
    src = req.source.strip().strip('"').strip("'")
    if not src:
        bad("モデルのフォルダか Hugging Face のID を指定してください")
    rt = S.runtime
    if rt is not None and (rt.busy or rt.state == "loading"):
        bad("推論/学習/読み込みの実行中は別のモデルを開けません")
    if os.path.isfile(src):
        src = os.path.dirname(src)
    if os.path.isdir(src):
        path = src
    elif REPO_RE.match(src):
        try:
            path = _download(src, req.hf_token or os.environ.get("HF_TOKEN"))
        except HTTPException:
            raise
        except Exception as e:
            bad(f"ダウンロードに失敗しました: {type(e).__name__}: {e}")
    else:
        bad(f"フォルダが見つかりません: {src}", 404)

    with S.lock:
        try:
            ck = SafetensorsCheckpoint(path)
            st = build_structure(ck, path)
        except FileNotFoundError as e:
            bad(str(e), 404)
        except Exception as e:
            traceback.print_exc()
            bad(f"モデルの解析に失敗しました: {type(e).__name__}: {e}")
        S.atlas_token += 1  # cancel running atlas jobs
        if S.ckpt is not None:
            S.ckpt.close()
        S.ckpt, S.structure, S.model_dir = ck, st, path
        if rt is not None and rt.model is not None and os.path.abspath(rt.model_dir or "") != os.path.abspath(path):
            try:
                rt.unload()
            except RuntimeError:  # something started while the model was downloading
                pass
    bus.status(f"{st['name']} を開きました（テンソル {len(st['tensors'])} 個）")
    return st


@app.get("/api/structure")
def get_structure():
    if S.structure is None:
        return {"loaded": False}
    return S.structure


# ----------------------------------------------------------------- atlas
class AtlasReq(BaseModel):
    mode: str = "rms"
    size: int = 128


def _order_for_atlas(st: dict) -> list[int]:
    order, seen = [], set()

    def add(t):
        if t not in seen:
            seen.add(t)
            order.append(t)

    for tw in st["towers"]:
        for b in tw["blocks"]:
            for it in b["norms"]:
                add(it["t"])
            for g in b["groups"]:
                for it in g["items"]:
                    add(it["t"])
                for lst in g["experts"]:
                    for it in lst:
                        add(it["t"])
    for tw in st["towers"]:
        for it in tw["pre"] + tw["post"]:
            add(it["t"])
    for t in st["tensors"]:
        add(t["id"])
    return order


def _atlas_job(token: int, mode: str, size: int) -> None:
    ck, st = S.ckpt, S.structure
    if ck is None or st is None:
        return
    tensors = st["tensors"]
    n = len(tensors)
    signed = mode in SIGNED_MODES
    key = hashlib.sha1(f"v1|{ck.fingerprint()}|{mode}|{size}".encode()).hexdigest()[:24]
    path = CACHE / f"atlas-{key}.npz"

    def flush(ids, tiles, lo, hi, done):
        if not ids:
            return
        bus.publish_frame({"type": "atlas_tiles", "token": token, "mode": mode, "size": size,
                           "ids": ids, "lo": [float(lo[i]) for i in ids], "hi": [float(hi[i]) for i in ids],
                           "done": done, "total": n}, np.ascontiguousarray(tiles[ids]).tobytes())

    try:
        cached = None
        if path.exists():
            try:
                with np.load(path) as data:
                    cached = data["tiles"], data["lo"], data["hi"]
                if cached[0].shape != (n, size, size):
                    cached = None
            except Exception:  # e.g. truncated by a crash mid-write: rebuild it
                cached = None
        if cached is not None:
            tiles, lo, hi = cached
            order = _order_for_atlas(st)
            for i in range(0, n, 48):
                if S.atlas_token != token:
                    return
                flush(order[i:i + 48], tiles, lo, hi, min(n, i + 48))
            bus.publish({"type": "atlas_done", "token": token, "mode": mode, "cached": True})
            return

        tiles = np.zeros((n, size, size), np.uint8)
        lo = np.zeros(n, np.float32)
        hi = np.zeros(n, np.float32)
        batch: list[int] = []
        failed = 0
        last = time.time()
        t0 = time.time()
        for k, tid in enumerate(_order_for_atlas(st)):
            if S.atlas_token != token:
                return
            t = tensors[tid]
            if t["float"]:
                try:
                    th, _ = thumbnail(ck, t["name"], size, mode)
                    q, a, b = quantize(th, signed)
                    tiles[tid] = resize_nearest(q, size)
                    lo[tid], hi[tid] = a, b
                except Exception as e:
                    failed += 1
                    print("thumbnail failed", t["name"], e)
            batch.append(tid)
            if len(batch) >= 24 or time.time() - last > 0.2:
                flush(batch, tiles, lo, hi, k + 1)
                batch, last = [], time.time()
        flush(batch, tiles, lo, hi, n)
        if not failed:
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, tiles=tiles, lo=lo, hi=hi)
            os.replace(tmp, path)
        bus.publish({"type": "atlas_done", "token": token, "mode": mode, "cached": False,
                     "elapsed": time.time() - t0})
    except Exception as e:
        traceback.print_exc()
        bus.publish({"type": "atlas_error", "token": token, "message": f"{type(e).__name__}: {e}"})


@app.post("/api/atlas")
def start_atlas(req: AtlasReq):
    if S.structure is None:
        bad("モデルが開かれていません")
    if req.mode not in MODES:
        bad(f"mode は {MODES} のいずれか")
    size = int(max(16, min(256, req.size)))
    S.atlas_token += 1
    token = S.atlas_token
    threading.Thread(target=_atlas_job, args=(token, req.mode, size), daemon=True).start()
    return {"ok": True, "token": token}


@app.get("/api/tensor/{tid}")
def tensor_detail(tid: int, size: int = 320, mode: str = "sample"):
    if S.structure is None or S.ckpt is None:
        bad("モデルが開かれていません")
    if not (0 <= tid < len(S.structure["tensors"])):
        bad("tensor id が範囲外です", 404)
    if mode not in MODES:
        bad("unknown mode")
    t = S.structure["tensors"][tid]
    if not t["float"]:
        return binary({**t, "gh": 0, "gw": 0, "unsupported": True})
    th, stats = thumbnail(S.ckpt, t["name"], int(max(16, min(1024, size))), mode, want_stats=True)
    q, lo, hi = quantize(th, mode in SIGNED_MODES)
    return binary({**t, "gh": int(q.shape[0]), "gw": int(q.shape[1]), "lo": lo, "hi": hi, "mode": mode,
                   "signed": mode in SIGNED_MODES, "stats": stats}, q.tobytes())


# --------------------------------------------------------------- runtime
class LoadReq(BaseModel):
    dtype: str = "auto"
    device: str = "auto"


@app.get("/api/runtime")
def runtime_info():
    return runtime().info()


@app.post("/api/runtime/load")
def runtime_load(req: LoadReq):
    if S.structure is None:
        bad("モデルが開かれていません")
    try:
        runtime().start_load(S.model_dir, S.structure, req.dtype, req.device)
    except RuntimeError as e:
        bad(str(e))
    return {"ok": True}


@app.post("/api/runtime/unload")
def runtime_unload():
    try:
        runtime().unload()
    except RuntimeError as e:
        bad(str(e))
    return {"ok": True}


class InferReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 50
    chat: bool = True
    lora: bool = True


@app.post("/api/infer")
def infer(req: InferReq):
    try:
        runtime().start_infer(req.model_dump())
    except RuntimeError as e:
        bad(str(e))
    return {"ok": True}


@app.post("/api/stop")
def stop():
    runtime().stop_event.set()
    return {"ok": True}


@app.get("/api/infer/attention")
def attention(layer: int = 0):
    try:
        h, payload = runtime().attention(layer)
    except RuntimeError as e:
        bad(str(e))
    return binary(h, payload)


class TrainReq(BaseModel):
    dataset: str
    method: str = "lora"
    rank: int = 8
    alpha: float = 16
    dropout: float = 0.0
    lr: float = 2e-4
    steps: int = 60
    batch_size: int = 2
    seq_len: int = 256
    warmup: int = 5
    targets: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layers: str = ""
    sample_every: int = 20
    sample_prompts: list[str] = []
    tile: int = 128
    delta_every_sec: float = 0.7
    max_grad_norm: float = 1.0


@app.post("/api/train")
def train(req: TrainReq):
    try:
        runtime().start_train(req.model_dump())
    except RuntimeError as e:
        bad(str(e))
    return {"ok": True}


class LoraEnable(BaseModel):
    enabled: bool


@app.post("/api/lora/enable")
def lora_enable(req: LoraEnable):
    runtime().set_lora_enabled(req.enabled)
    return runtime().info()


@app.post("/api/lora/reset")
def lora_reset():
    rt = runtime()
    if rt.busy:
        bad("実行中の処理があります")
    if rt.lora:
        from .trainer import reset_lora
        reset_lora(rt)
    bus.publish({"type": "lora_reset"})
    return rt.info()


class LoraSave(BaseModel):
    path: str | None = None


@app.post("/api/lora/save")
def lora_save(req: LoraSave):
    rt = runtime()
    if not rt.lora:
        bad("保存するLoRAがありません")
    from .trainer import save_lora
    out = req.path or str(OUTPUTS / time.strftime("lora-%Y%m%d-%H%M%S"))
    try:
        save_lora(rt, out)
    except RuntimeError as e:
        bad(str(e))
    return {"ok": True, "path": out}


# ------------------------------------------------------------- websocket
def _local_origin(origin: str) -> bool:
    return origin in TAURI_ORIGINS or bool(re.match(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$", origin))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    origin = ws.headers.get("origin")
    if origin and not _local_origin(origin):
        await ws.close(code=1008)
        return
    await ws.accept()
    q = bus.subscribe()

    async def sender():
        while True:
            msg = await q.get()
            if isinstance(msg, (bytes, bytearray)):
                await ws.send_bytes(bytes(msg))
            else:
                await ws.send_text(msg if isinstance(msg, str) else dumps(msg))

    task = asyncio.create_task(sender())
    try:
        await ws.send_text(dumps({"type": "hello", "version": __version__,
                                  "runtime": S.runtime.info() if S.runtime else None}))
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
        bus.unsubscribe(q)


if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="static")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"AITrainView backend on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

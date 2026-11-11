"""Watch a model in AITrainView while *your own* script fine-tunes it.

Hugging Face Trainer (one line)::

    from aitrainview.watch import AITrainViewCallback
    trainer = Trainer(..., callbacks=[AITrainViewCallback()])

A hand-written loop::

    from aitrainview.watch import watch
    w = watch(model, tokenizer=tok)   # attaches to the running app (tokenizer: optional)
    for batch in loader:
        loss = ...
        loss.backward()
        w.step(loss)                  # after backward(), before optimizer.step()
        optimizer.step()
        optimizer.zero_grad()
    w.close()

Your script trains; the app shows the weights, the gradient of every matrix,
how far each one moved (ΔW) and, every few seconds, the activations of one
training step, live. PEFT LoRA layers are shown as the ΔW of the matrix they
adapt. If the app is not running, everything here quietly does nothing, so a
training run is never broken by the viewer.

Needs only torch (and transformers for the callback). The app is found through
AITV_URL, else the port in ~/.aitrainview/backend.json, else 127.0.0.1:8765.
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .events import frame
from .runtime import ACT_P, ACT_T  # the app's activation-block size (channels x positions)

MODES = ("rms", "mean", "absmax", "sample")
SIGNED = {"mean", "sample"}
_TAGS = {"float32": "F32", "float16": "F16", "bfloat16": "BF16", "float64": "F64", "float8_e4m3fn": "F8_E4M3",
         "float8_e5m2": "F8_E5M2", "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8", "uint8": "U8",
         "bool": "BOOL"}
_CHUNK = 1 << 22  # elements reduced at a time (keeps the extra GPU memory small)


def _url() -> str:
    if os.environ.get("AITV_URL"):
        return os.environ["AITV_URL"].rstrip("/")
    port = os.environ.get("AITV_PORT")
    if not port:
        try:
            port = json.loads((Path.home() / ".aitrainview" / "backend.json").read_text(encoding="utf-8"))["port"]
        except Exception:
            port = 8765
    return f"http://127.0.0.1:{port}"


def norm_name(name: str) -> str | None:
    """Parameter name as in the original checkpoint (PEFT / DDP / compile wrapping removed); None = not shown."""
    if "lora_" in name or ".original_module." in name:
        return None
    for prefix in ("base_model.model.", "_orig_mod.", "module."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace(".base_layer.", ".").replace(".modules_to_save.default.", ".")


def as_2d(w: torch.Tensor) -> torch.Tensor:
    """Same 2-D view as the app's thumbnails (thumbs.as_2d)."""
    if w.dim() == 0:
        return w.reshape(1, 1)
    if w.dim() == 1:
        return w.reshape(1, -1)
    if w.dim() == 2:
        return w
    if w.dim() == 3:
        return w.reshape(w.shape[0] * w.shape[1], w.shape[2])
    return w.reshape(w.shape[0], -1)


def _bins(n: int, g: int, device):
    edges = (torch.arange(g + 1, device=device) * n) // g  # same edges as thumbs._edges
    idx = torch.bucketize(torch.arange(n, device=device), edges[1:], right=True)
    return edges, idx


@torch.no_grad()
def reduce_blocks(w: torch.Tensor, size: int, modes=MODES, want_stats: bool = False):
    """Block-reduce a tensor to <= size x size cells for each mode (like thumbs.thumbnail), on its device."""
    x = as_2d(w.detach())
    R, C = x.shape
    gh, gw = min(R, size), min(C, size)
    dev = x.device
    redges, rbin = _bins(R, gh, dev)
    cedges, cbin = _bins(C, gw, dev)
    sums = torch.zeros(gh, gw, device=dev) if ("mean" in modes) else None
    sq = torch.zeros(gh, gw, device=dev) if ("rms" in modes) else None
    amax = torch.zeros(gh, gw, device=dev) if ("absmax" in modes) else None
    s1 = torch.zeros((), dtype=torch.float64, device=dev)
    s2 = torch.zeros((), dtype=torch.float64, device=dev)
    mn, mx = math.inf, -math.inf
    rows = max(1, _CHUNK // max(C, 1))
    if sums is not None or sq is not None or amax is not None or want_stats:
        for r0 in range(0, R, rows):
            ch = torch.nan_to_num(x[r0:r0 + rows].float(), nan=0.0, posinf=0.0, neginf=0.0)
            rb = rbin[r0:r0 + ch.shape[0]]
            if sums is not None:
                sums.index_add_(0, rb, torch.zeros(ch.shape[0], gw, device=dev).index_add_(1, cbin, ch))
            if sq is not None:
                sq.index_add_(0, rb, torch.zeros(ch.shape[0], gw, device=dev).index_add_(1, cbin, ch * ch))
            if amax is not None:
                ca = torch.zeros(ch.shape[0], gw, device=dev).scatter_reduce_(
                    1, cbin.expand(ch.shape[0], C), ch.abs(), "amax", include_self=True)
                amax.scatter_reduce_(0, rb[:, None].expand(-1, gw), ca, "amax", include_self=True)
            if want_stats:
                s1 += ch.sum(dtype=torch.float64)
                s2 += (ch * ch).sum(dtype=torch.float64)
                mn, mx = min(mn, ch.min().item()), max(mx, ch.max().item())
    count = torch.bincount(rbin, minlength=gh).float()[:, None] * torch.bincount(cbin, minlength=gw).float()[None]
    out = {}
    if "rms" in modes:
        out["rms"] = (sq / count).sqrt()
    if "mean" in modes:
        out["mean"] = sums / count
    if "absmax" in modes:
        out["absmax"] = amax
    if "sample" in modes:  # one real weight per block (its centre)
        rc = (redges[:-1] + redges[1:]) // 2
        cc = (cedges[:-1] + cedges[1:]) // 2
        out["sample"] = torch.nan_to_num(x[rc][:, cc].float(), nan=0.0, posinf=0.0, neginf=0.0)
    if not want_stats:
        return out, None
    n = R * C
    mean = (s1 / n).item()
    var = max((s2 / n).item() - mean * mean, 0.0)
    flat = x.reshape(-1)
    smp = torch.nan_to_num(flat[:: max(1, n // 65536)].float())
    lo, hi = smp.min().item(), smp.max().item()
    if hi <= lo:
        hi = lo + 1e-6
    q = torch.quantile(smp[:16_000_000], torch.tensor([0.01, 0.99], device=dev)).tolist()
    stats = {"mean": mean, "std": math.sqrt(var), "min": mn, "max": mx, "absmax": max(abs(mn), abs(mx)),
             "rms": math.sqrt((s2 / n).item()), "p01": q[0], "p99": q[1],
             "hist": torch.histc(smp, 64, lo, hi).long().tolist(), "hist_lo": lo, "hist_hi": hi,
             "zero_frac": (smp.abs() < 1e-8).float().mean().item()}
    return out, stats


def quantize(th: torch.Tensor, signed: bool):
    """Same mapping as thumbs.quantize: signed → 127.5 is zero; else 0.5–99.5 percentile range."""
    th = th.float()
    f = th.reshape(-1)
    if signed:
        a = f.abs()
        s = torch.quantile(a, 0.995).item() if a.numel() > 64 else a.max().item()
        if not (math.isfinite(s) and s > 0):
            s = a.max().item() or 1.0
        q = (th / s).clamp(-1, 1) * 127.5 + 127.5
        lo, hi = -s, s
    else:
        if f.numel() > 64:
            lo, hi = torch.quantile(f, torch.tensor([0.005, 0.995], device=f.device)).tolist()
        else:
            lo, hi = f.min().item(), f.max().item()
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo <= 1e-12:
            lo, hi = f.min().item(), f.max().item() + 1e-6
        q = ((th - lo) / (hi - lo)).clamp(0, 1) * 255
    return q.round().to(torch.uint8), lo, hi


def square(t: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize to size x size (thumbs.resize_nearest)."""
    gh, gw = t.shape
    ri = (torch.arange(size, device=t.device) * gh) // size
    ci = (torch.arange(size, device=t.device) * gw) // size
    return t[ri][:, ci]


def _lora_adapters(mod) -> list[str]:
    names = getattr(mod, "active_adapters", None) or getattr(mod, "active_adapter", None) or []
    if isinstance(names, str):
        names = [names]
    return [n for n in names if n in mod.lora_A and n in mod.lora_B]


def _lora_factors(mod):
    """(A [r, in], B [out, r], scale) for each active adapter of a PEFT LoRA layer (Linear or Conv)."""
    for n in _lora_adapters(mod):
        A = mod.lora_A[n].weight.float()
        B = mod.lora_B[n].weight.float()
        yield A.reshape(A.shape[0], -1), B.reshape(B.shape[0], -1), float(mod.scaling[n])


def _model_name(model) -> str:
    cfg = getattr(model, "config", None)
    name = getattr(cfg, "_name_or_path", None) if cfg is not None else None
    return name or type(model).__name__


class Watcher:
    """Streams one model's training to the app. See the module docstring."""

    def __init__(self, model: torch.nn.Module, name: str | None = None, url: str | None = None,
                 every_sec: float = 0.7, source: str | None = None, act_every_sec: float | None = 3.0,
                 tokenizer=None):
        """every_sec: at most one ΔW update per this many seconds.
        act_every_sec: capture the activations of one training step this often (None = never).
        tokenizer: decodes the captured tokens for the activation view (optional)."""
        self.model = model
        self.url = (url or _url()).rstrip("/")
        self.name = name or _model_name(model)
        self.source = source or self.name
        self.every = every_sec
        self.act_every = act_every_sec
        self.tokenizer = tokenizer
        self._act_handles: list = []
        self._act_mods: list = []
        self._layer_mods: list = []
        self._cap: dict = {}
        self._cap_ids = None
        self._conv_model = False
        self._last_act = 0.0
        self.enabled = False
        self.stop_requested = False  # the user pressed stop in the app
        self.steps = None
        self._warned = False
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._grads: list[float] = []
        self._last_delta = 0.0
        self._last_t = None
        self._t0 = time.time()
        self._step = 0

    # ------------------------------------------------------------ transport
    def _post(self, path: str, body, timeout: float = 15.0) -> dict:
        if isinstance(body, (bytes, bytearray)):
            data, ctype = bytes(body), "application/octet-stream"
        else:
            data, ctype = json.dumps(body, default=str).encode(), "application/json"
        req = urllib.request.Request(self.url + path, data=data, method="POST", headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def _warn(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"[aitrainview] {msg}", flush=True)

    def _sender(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if not self.enabled:
                continue
            try:
                r = self._post("/api/ext/event", item)
                if r.get("stop"):
                    self.stop_requested = True
            except urllib.error.HTTPError as e:
                if e.code == 409:  # another model was opened in the app: stop sending
                    self.enabled = False
                    self._warn("アプリ側で別のモデルが開かれたため、送信をやめました（学習は続きます）")
            except (urllib.error.URLError, OSError) as e:
                self.enabled = False
                self._warn(f"アプリとの接続が切れました（{e}）。学習はそのまま続けます")

    def _event(self, header: dict, payload: bytes = b"") -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(frame({**header, "run": self.run}, payload))
        except queue.Full:  # the app is slower than the training: skip this update
            pass

    # --------------------------------------------------------------- attach
    @torch.no_grad()
    def attach(self, steps: int | None = None, examples: int | None = None) -> bool:
        """Send the model's structure and weight thumbnails. False (and no-op from then on) if the app is not running."""
        self.steps = steps
        params: dict[str, torch.Tensor] = {}
        for pname, p in self.model.named_parameters():
            n = norm_name(pname)
            if n is not None and n not in params:
                params[n] = p
        cfg = getattr(self.model, "config", None)
        cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        tensors = [{"name": n, "shape": list(p.shape), "dtype": _TAGS.get(str(p.dtype).replace("torch.", ""), "F32")}
                   for n, p in params.items()]
        try:
            r = self._post("/api/ext/attach", {"name": self.name, "source": self.source, "config": cfg,
                                               "tensors": tensors, "client": "aitrainview.watch"})
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._warn(f"AITrainView に接続できませんでした（{self.url}: {e}）。学習はそのまま続けます")
            return False
        self.run, self.size, ids = r["run"], int(r["size"]), r["ids"]
        self.enabled = True
        try:
            self._track(params, ids)
            self._setup_acts(ids, r.get("layers_prefix") or "")
            self._send_thumbnails(params, ids)
            self._post("/api/ext/ready", {"run": self.run})
        except Exception as e:  # never break the training run
            self.enabled = False
            self._warn(f"AITrainView への送信に失敗しました（{type(e).__name__}: {e}）。学習はそのまま続けます")
            return False
        self._thread = threading.Thread(target=self._sender, name="aitrainview-sender", daemon=True)
        self._thread.start()
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        method = "lora" if self._lora and not self._full else "full" if self._full and not self._lora else "mixed"
        self._event({"type": "train_start", "method": method, "steps": steps, "modules": len(self._ids),
                     "trainable_params": n_train, "examples": examples, "ids": self._ids,
                     "sample_prompts": [], "tile": self.size, "external": True})
        self._t0 = self._last_t = time.time()
        self.arm_activations(force=True)  # the first forward pass is captured
        return True

    def _track(self, params: dict, ids: dict) -> None:
        """What changes during training: PEFT LoRA layers (shown on their base matrix) and trainable parameters."""
        self._lora = []  # (tid, module, gh, gw, |W|)
        self._full = []  # (tid, param, pooled0, |W|)
        for mname, mod in self.model.named_modules():
            if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "base_layer")):
                continue
            base = getattr(mod.base_layer, "weight", None)
            n = norm_name(f"{mname}.base_layer.weight")
            if base is None or n not in ids or not _lora_adapters(mod):
                continue
            R, C = as_2d(base).shape
            self._lora.append((ids[n], mod, min(R, self.size), min(C, self.size), base.float().norm().item()))
        for n, p in params.items():
            if p.requires_grad and n in ids and p.is_floating_point():
                pooled, _ = reduce_blocks(p, self.size, modes=("mean",))
                self._full.append((ids[n], p, pooled["mean"], p.float().norm().item()))
        self._ids = [t for t, *_ in self._lora] + [t for t, *_ in self._full]

    def _send_thumbnails(self, params: dict, ids: dict) -> None:
        S = self.size
        batch: list = []

        def flush():
            if not batch:
                return
            for mode in MODES:
                tiles = b"".join(b[1][mode][0] for b in batch)
                header = {"run": self.run, "mode": mode, "ids": [b[0] for b in batch],
                          "lo": [b[1][mode][1] for b in batch], "hi": [b[1][mode][2] for b in batch]}
                if mode == "rms":
                    header["stats"] = [b[2] for b in batch]
                self._post("/api/ext/tiles", frame(header, tiles))
            batch.clear()

        for n, p in params.items():
            if n not in ids or not p.is_floating_point():
                continue
            maps, stats = reduce_blocks(p, S, want_stats=True)
            tiles = {}
            for mode, m in maps.items():
                q, lo, hi = quantize(m, mode in SIGNED)
                tiles[mode] = (square(q, S).cpu().numpy().tobytes(), lo, hi)
            batch.append((ids[n], tiles, stats))
            if len(batch) >= 32:
                flush()
        flush()

    # ---------------------------------------------------------- activations
    def _setup_acts(self, ids: dict, layers_prefix: str) -> None:
        """Modules whose activations are captured, with the keys the app's activation blocks use:
        o:<tid> / i:<tid> output / input of the module owning tensor <tid>, r:<n> / ro:<n> the stream
        entering / leaving repeated block n (the residual stream of a transformer)."""
        self._act_mods = []  # (tid, module, channels_first)
        self._layer_mods = []  # (index, module)
        pat = re.compile(re.escape(layers_prefix) + r"\.(\d+)$") if layers_prefix else None
        conv_like = (nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm)
        # in a convolutional model every 4-D activation is a [B, C, H, W] feature map (also after ReLU / pooling)
        self._conv_model = any(isinstance(m, nn.modules.conv._ConvNd) for m in self.model.modules())
        for mname, mod in self.model.named_modules():
            if not mname or mname.endswith(".base_layer") or ".lora_" in mname:
                continue
            n = norm_name(mname)
            if n is None:
                continue
            m = pat.match(n) if pat else None
            if m:
                self._layer_mods.append((int(m.group(1)), mod))
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):  # PEFT layer: shown on its base matrix
                tid = ids.get(norm_name(f"{mname}.base_layer.weight"))
                base = mod.base_layer
            else:
                tid = ids.get(f"{n}.weight") if isinstance(getattr(mod, "weight", None), torch.Tensor) else None
                base = mod
            if tid is not None:
                self._act_mods.append((tid, mod, isinstance(base, conv_like)))

    @staticmethod
    def _to_rows(x, channels_first: bool):
        """First sample of a batch as positions x channels (≤ ACT_T x ACT_P), each row scaled to uint8.
        Tokens x channels for sequence models; (H·W) x C for conv feature maps."""
        if not torch.is_tensor(x) or not x.is_floating_point() or x.dim() < 2:
            return None
        t = x[0].detach()
        if t.dim() == 1:  # e.g. the output of a classifier: one position
            t = t[None]
        elif t.dim() == 3:
            t = t.flatten(1).t() if channels_first else t.reshape(t.shape[0], -1)
        elif t.dim() != 2:
            return None
        if t.shape[-1] > 65536:  # e.g. vocabulary logits
            return None
        t = torch.nan_to_num(t[:ACT_T].float())
        P = min(t.shape[-1], ACT_P)
        if t.shape[-1] != P:
            t = F.adaptive_avg_pool1d(t[None], P)[0]
        s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        return (t / s * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)

    def arm_activations(self, force: bool = False) -> None:
        """Hook the model so that the next forward pass is captured (at most every act_every_sec)."""
        if not self.enabled or self.act_every is None or self._act_handles:
            return
        if not force and time.time() - self._last_act < self.act_every:
            return
        cap = self._cap = {}
        self._cap_ids = None

        def put(key, x, cf):
            if key not in cap:  # first forward only (gradient checkpointing runs it twice)
                cf = cf or (self._conv_model and torch.is_tensor(x) and x.dim() == 4)
                with torch.no_grad():
                    r = self._to_rows(x, cf)
                if r is not None:
                    cap[key] = r

        def out_hook(tid, cf):
            return lambda mod, inp, out: put(f"o:{tid}", out[0] if isinstance(out, (tuple, list)) else out, cf)

        def in_hook(tid, cf):
            return lambda mod, args: put(f"i:{tid}", args[0], cf) if args else None

        def layer_in(li):
            return lambda mod, args, kwargs: put(f"r:{li}", args[0] if args else kwargs.get("hidden_states"), False)

        def layer_out(li):
            return lambda mod, inp, out: put(f"ro:{li}", out[0] if isinstance(out, (tuple, list)) else out, False)

        def tokens(mod, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if self._cap_ids is None and torch.is_tensor(ids) and ids.dim() == 2 and not ids.is_floating_point():
                self._cap_ids = ids[0, :ACT_T].tolist()

        handles = [self.model.register_forward_pre_hook(tokens, with_kwargs=True)]
        for tid, mod, cf in self._act_mods:
            handles.append(mod.register_forward_hook(out_hook(tid, cf)))
            if not isinstance(mod, nn.Embedding):
                handles.append(mod.register_forward_pre_hook(in_hook(tid, cf)))
        for li, mod in self._layer_mods:
            handles.append(mod.register_forward_pre_hook(layer_in(li), with_kwargs=True))
            handles.append(mod.register_forward_hook(layer_out(li)))
        self._act_handles = handles

    def _unhook(self) -> None:
        for h in self._act_handles:
            h.remove()
        self._act_handles = []

    def flush_activations(self, step: int) -> None:
        """Send what the armed forward pass captured (call after the step)."""
        if not self._act_handles or not self._cap:
            return  # not armed, or no forward pass ran yet
        self._unhook()
        self._last_act = time.time()
        keys = list(self._cap)
        tiles = [self._cap[k] for k in keys]
        self._cap = {}
        if not self.enabled:
            return
        payload = torch.cat([t.reshape(-1) for t in tiles]).cpu().numpy().tobytes()
        toks = None
        if self._cap_ids is not None and self.tokenizer is not None:
            try:
                toks = [self.tokenizer.decode([i]) for i in self._cap_ids]
            except Exception:
                toks = None
        self._event({"type": "train_acts", "step": int(step), "n": max(t.shape[0] for t in tiles), "tokens": toks,
                     "entries": [[k, int(t.shape[1]), int(t.shape[0])] for k, t in zip(keys, tiles)]}, payload)

    # ------------------------------------------------------------ telemetry
    @torch.no_grad()
    def capture_grads(self) -> None:
        """Gradient norm per tracked matrix; call after backward() and before the optimizer step / zero_grad."""
        if not self.enabled:
            return
        zero = None
        norms = []
        for _, mod, *_ in self._lora:
            sq = None
            for n in _lora_adapters(mod):
                for p in (mod.lora_A[n].weight, mod.lora_B[n].weight):
                    if p.grad is not None:
                        g = p.grad.float().pow(2).sum()
                        sq = g if sq is None else sq + g
            norms.append(sq.sqrt() if sq is not None else None)
        for _, p, *_ in self._full:
            norms.append(p.grad.float().norm() if p.grad is not None else None)
        dev = next((n.device for n in norms if n is not None), None)
        if dev is None:
            self._grads = []
            return
        zero = torch.zeros((), device=dev)
        self._grads = torch.stack([n if n is not None else zero for n in norms]).cpu().tolist()

    def log(self, step: int, loss: float | None = None, lr: float | None = None,
            grad_norm: float | None = None, tokens_per_sec: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.time()
        dt = now - self._last_t if self._last_t else None
        self._last_t = now
        self._event({"type": "train_step", "step": int(step), "steps": self.steps,
                     "loss": None if loss is None else float(loss), "lr": None if lr is None else float(lr),
                     "grad_norm": None if grad_norm is None else float(grad_norm),
                     "ids": self._ids if self._grads else [], "grads": self._grads,
                     "step_time": dt, "tok_s": tokens_per_sec})

    @torch.no_grad()
    def maybe_delta(self, step: int, force: bool = False) -> None:
        """ΔW heatmaps of every tracked matrix, at most every `every_sec` seconds."""
        if not self.enabled or not self._ids:
            return
        now = time.time()
        if not force and now - self._last_delta < self.every:
            return
        self._last_delta = now
        S = self.size
        mats, norms, w_norms = [], [], []
        for _, mod, gh, gw, wn in self._lora:
            d, sq = None, 0.0
            for A, B, s in _lora_factors(mod):
                Bp = F.adaptive_avg_pool1d(B.t()[None], gh)[0].t()  # pool(B @ A) == pool_rows(B) @ pool_cols(A)
                Ap = F.adaptive_avg_pool1d(A[None], gw)[0]
                part = (Bp @ Ap) * s
                d = part if d is None else d + part
                sq += (((B.t() @ B) * (A @ A.t())).sum().clamp_min(0) * s * s).item()
            mats.append(d if d is not None else torch.zeros(gh, gw, device=mod.base_layer.weight.device))
            norms.append(math.sqrt(sq))
            w_norms.append(wn)
        for _, p, pooled0, wn in self._full:
            now_mean, _ = reduce_blocks(p, S, modes=("mean",))
            d = now_mean["mean"] - pooled0
            mats.append(d)
            norms.append(d.norm().item() * math.sqrt(p.numel() / d.numel()))  # approximate
            w_norms.append(wn)
        tiles, scales = [], []
        for d in mats:
            t = square(d.float(), S)
            amax = t.abs().max().clamp_min(1e-20)
            tiles.append(((t / amax).clamp(-1, 1) * 127.5 + 127.5).round().to(torch.uint8).cpu().numpy().tobytes())
            scales.append(amax.item())
        # relative change is undefined for parameters that started at zero (biases, BatchNorm shift ...)
        rel = [n / w if w > 1e-6 else None for n, w in zip(norms, w_norms)]
        self._event({"type": "train_delta", "step": int(step), "ids": self._ids, "scales": scales, "rel": rel,
                     "size": S}, b"".join(tiles))

    def step(self, loss=None, lr: float | None = None) -> None:
        """For hand-written loops: call after backward(), before optimizer.step()."""
        if not self.enabled:
            return
        self._step += 1
        self.capture_grads()
        if torch.is_tensor(loss):
            loss = loss.detach().float().item()
        self.log(self._step, loss=loss, lr=lr)
        self.flush_activations(self._step)  # captured in this step's forward pass
        self.maybe_delta(self._step)
        self.arm_activations()  # capture a later forward pass

    def done(self, step: int | None = None, stopped: bool | None = None) -> None:
        self._unhook()
        if not self.enabled:
            return
        step = self._step if step is None else step
        self._event({"type": "train_done", "step": int(step), "elapsed": time.time() - self._t0,
                     "stopped": self.stop_requested if stopped is None else stopped})
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=30)
        try:
            self._post("/api/ext/detach", {"run": self.run}, timeout=5)
        except (urllib.error.URLError, OSError):
            pass
        self.enabled = False

    def close(self) -> None:
        if self.enabled:
            self.maybe_delta(self._step, force=True)
            self.done()


def watch(model: torch.nn.Module, steps: int | None = None, **kw) -> Watcher:
    """Attach `model` to the running app; returns a Watcher (a no-op one if the app is not running)."""
    w = Watcher(model, **kw)
    w.attach(steps=steps)
    return w


try:
    from transformers import TrainerCallback as _Base
except Exception:  # transformers not installed: only watch() is available
    _Base = object


class AITrainViewCallback(_Base):
    """Hugging Face Trainer callback: `Trainer(..., callbacks=[AITrainViewCallback()])`.

    log_every: send loss / gradients every N optimizer steps (it makes the Trainer
    log at that interval). every_sec: at most one ΔW update per this many seconds.
    act_every_sec: capture the activations of one step this often (None = off).
    """

    def __init__(self, name: str | None = None, url: str | None = None, log_every: int = 1, every_sec: float = 0.7,
                 act_every_sec: float | None = 3.0):
        self.kw = {"name": name, "url": url, "every_sec": every_sec, "act_every_sec": act_every_sec}
        self.log_every = max(1, int(log_every))
        self.w: Watcher | None = None

    def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kw):
        if model is None or not getattr(state, "is_world_process_zero", True):
            return
        examples = None
        try:
            examples = len(train_dataloader.dataset)
        except Exception:
            pass
        tok = kw.get("processing_class") or kw.get("tokenizer")
        self.w = Watcher(model, tokenizer=tok if hasattr(tok, "decode") else None, **self.kw)
        self.w.attach(steps=state.max_steps or None, examples=examples)

    def on_step_begin(self, args, state, control, **kw):
        if self.w and self.w.enabled:
            self.w.arm_activations()

    def on_pre_optimizer_step(self, args, state, control, **kw):
        if self.w and self.w.enabled and (state.global_step + 1) % self.log_every == 0:
            self.w.capture_grads()

    def on_step_end(self, args, state, control, **kw):
        if not (self.w and self.w.enabled):
            return
        if state.global_step % self.log_every == 0:
            control.should_log = True
        self.w.flush_activations(state.global_step)
        self.w.maybe_delta(state.global_step)
        if self.w.stop_requested:
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kw):
        if self.w and self.w.enabled and logs and "loss" in logs:
            self.w.log(state.global_step, loss=logs.get("loss"), lr=logs.get("learning_rate"),
                       grad_norm=logs.get("grad_norm"))

    def on_train_end(self, args, state, control, **kw):
        if self.w and self.w.enabled:
            self.w.maybe_delta(state.global_step, force=True)
            self.w.done(state.global_step)

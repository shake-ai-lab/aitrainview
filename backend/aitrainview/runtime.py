"""Torch runtime: holds the live model for inference and fine-tuning.

During generation every module that owns a checkpoint weight is hooked so the
3D view can light up exactly the matrices data flows through, and every
decoder layer's residual stream is projected through the final norm + lm_head
("logit lens") to show how the prediction forms layer by layer.
"""
from __future__ import annotations

import re
import threading
import time
import traceback

import numpy as np

from .events import Bus

STRIP = {"model", "language_model", "base_model", "transformer", "text_model"}
MAX_ATTN_TOKENS = 256
ACT_P = 256  # channels kept per activation row (adaptive average pooling)
ACT_T = 128  # tokens streamed to the 3D activation blocks


def norm_name(n: str) -> str:
    """Normalize checkpoint / module names so they can be matched to each other."""
    parts = n.split(".")
    while parts and parts[0] in STRIP:
        parts = parts[1:]
    parts = [p for p in parts if p not in ("base", "base_layer")]
    return ".".join(parts)


def _pick_dtype(torch, pref: str, model_type: str, device: str):
    if pref == "fp32":
        return torch.float32
    if pref == "fp16":
        return torch.float16
    if pref == "bf16":
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        bf16 = torch.cuda.get_device_capability()[0] >= 8
    if bf16:
        return torch.bfloat16
    if "gemma" in (model_type or ""):
        return torch.float32  # Gemma overflows in fp16
    return torch.float16


class Runtime:
    def __init__(self, bus: Bus) -> None:
        self.bus = bus
        self.model = None
        self.tok = None
        self.device = None
        self.dtype = None
        self.model_dir: str | None = None
        self.structure: dict | None = None
        self.busy: str | None = None
        self.stop_event = threading.Event()
        self.state = "unloaded"
        self.error: str | None = None
        self.mod_tid: dict[str, int] = {}
        self.layer_mods: dict[int, str] = {}
        self.final_norm_name: str | None = None
        self.head_name: str | None = None
        self.lora: dict | None = None  # {"wrappers": {name: LoRALinear}, r, alpha, enabled}
        self.attn: dict | None = None
        self._acts: dict = {}
        self._layer_out: dict = {}
        self._cap: dict = {}
        self._cap_keep = 0
        self._lock = threading.Lock()  # guards the busy / loading hand-over between request threads

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        d = {"state": self.state, "busy": self.busy, "error": self.error, "model_dir": self.model_dir}
        if self.model is not None:
            import torch
            d.update({
                "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "hooked": len(self.mod_tid), "layers": len(self.layer_mods),
                "lora": None if not self.lora else {
                    "modules": len(self.lora["wrappers"]), "r": self.lora["r"],
                    "alpha": self.lora["alpha"], "enabled": self.lora["enabled"],
                    "steps": self.lora.get("steps", 0),
                },
            })
            if torch.cuda.is_available() and self.device and self.device.type == "cuda":
                d["vram_used"] = torch.cuda.memory_allocated() / 2**30
                d["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        return d

    def _emit_state(self, state: str, message: str = "") -> None:
        self.state = state
        self.bus.publish({"type": "runtime", "state": state, "message": message, "info": self.info()})

    # ------------------------------------------------------------------ load
    def start_load(self, model_dir: str, structure: dict, dtype_pref: str = "auto",
                   device_pref: str = "auto") -> None:
        with self._lock:
            if self.busy:
                raise RuntimeError(f"{self.busy} の実行中です")
            if self.state == "loading":
                raise RuntimeError("モデルの読み込み中です")
            self.state = "loading"
        threading.Thread(target=self.load, args=(model_dir, structure, dtype_pref, device_pref),
                         daemon=True).start()

    def load(self, model_dir: str, structure: dict, dtype_pref: str = "auto", device_pref: str = "auto"):
        """Runs on its own thread; call start_load() to claim the runtime first."""
        self._drop()
        self.error = None
        self._emit_state("loading", "torch を読み込み中…")
        oom = False
        model = tok = None
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer

            dev = "cuda" if (device_pref in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
            dtype = _pick_dtype(torch, dtype_pref, structure.get("model_type", ""), dev)
            self._emit_state("loading", f"モデルを読み込み中… ({dev}, {str(dtype).replace('torch.', '')})")
            tok = AutoTokenizer.from_pretrained(model_dir)
            err = None
            for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
                cls = getattr(transformers, cls_name, None)
                if cls is None:
                    continue
                try:
                    model = cls.from_pretrained(model_dir, dtype=dtype, attn_implementation="eager")
                    break
                except Exception as e:  # try the next auto class
                    err = e
            if model is None:
                raise err or RuntimeError("モデルクラスを特定できませんでした")
            model.to(dev)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            self.model, self.tok = model, tok
            self.device, self.dtype = torch.device(dev), dtype
            self.model_dir, self.structure = model_dir, structure
            self._map_modules()
            self._emit_state("ready", "推論・学習の準備ができました")
        except Exception as e:
            traceback.print_exc()
            self.model = None
            self.tok = None
            self.error = f"{type(e).__name__}: {e}"
            self._emit_state("error", self.error)
            oom = "out of memory" in str(e).lower()
        if oom:  # only now is the half-moved model unreferenced, so the cache can really be emptied
            model = tok = None
            self._free_cuda()

    def _map_modules(self) -> None:
        model, st = self.model, self.structure
        tid_by_norm = {norm_name(t["name"]): t["id"] for t in st["tensors"]}
        self.mod_tid = {}
        for name, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            if w is None or not hasattr(w, "shape"):
                continue
            tid = tid_by_norm.get(norm_name(name + ".weight"))
            if tid is not None:
                self.mod_tid[name] = tid

        main = st["towers"][st.get("main_tower", 0)]
        mp = norm_name(main["prefix"])
        pat = re.compile(re.escape(mp) + r"\.(\d+)")
        self.layer_mods = {}
        cands = set()
        parent = mp.rsplit(".", 1)[0] if "." in mp else ""
        for c in ("norm", "final_layernorm", "ln_f", "final_norm", "norm_f"):
            cands.add(f"{parent}.{c}" if parent else c)
        self.final_norm_name = None
        head = model.get_output_embeddings()
        self.head_name = None
        for name, mod in model.named_modules():
            nn_ = norm_name(name)
            m = pat.fullmatch(nn_)
            if m:
                self.layer_mods.setdefault(int(m.group(1)), name)
            if self.final_norm_name is None and nn_ in cands:
                self.final_norm_name = name
            if head is not None and mod is head and self.head_name is None:
                self.head_name = name

    def _free_cuda(self) -> None:
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _drop(self) -> None:
        self.model = None
        self.tok = None
        self.lora = None
        self.attn = None
        self.mod_tid = {}
        self.layer_mods = {}
        self.final_norm_name = None
        self.head_name = None
        self._acts.clear()
        self._layer_out.clear()
        self._cap.clear()
        self._free_cuda()

    def unload(self) -> None:
        with self._lock:
            if self.busy:
                raise RuntimeError("推論/学習の実行中です")
            if self.state == "loading":
                raise RuntimeError("モデルの読み込み中です")
            self._drop()
        self._emit_state("unloaded", "モデルを解放しました")

    # ------------------------------------------------------------- helpers
    def require_ready(self) -> None:
        if self.state == "loading":
            raise RuntimeError("モデルの読み込み中です")
        if self.model is None:
            raise RuntimeError("先に「GPUに読み込む」でモデルを読み込んでください")
        if self.busy:
            raise RuntimeError(f"{self.busy} の実行中です")

    def _claim(self, kind: str) -> None:
        """Atomically check that the runtime is idle and mark it busy."""
        with self._lock:
            self.require_ready()
            self.busy = kind
            self.stop_event.clear()

    def chat_text(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def eos_ids(self) -> set:
        ids = set()
        for src in (self.tok.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(src, int):
                ids.add(src)
            elif isinstance(src, (list, tuple)):
                ids.update(int(x) for x in src)
        return ids

    def set_lora_enabled(self, on: bool) -> None:
        if self.lora:
            self.lora["enabled"] = bool(on)
            for w in self.lora["wrappers"].values():
                w.enabled = bool(on)

    # -------------------------------------------------------------- hooks
    def _install_hooks(self):
        """Hook every weight-owning module.

        * ``self._acts[tid]``  RMS of the last token's output (drives the glow)
        * ``self._cap[key]``   per-token activations, channel-pooled to <= ACT_P and
          row-normalised to uint8, for the 3D activation blocks. Keys:
          ``o:<tid>`` module output, ``i:<tid>`` module input,
          ``r:<layer>`` / ``ro:<layer>`` residual stream entering / leaving a layer.
        """
        import torch
        import torch.nn.functional as F
        acts: dict = {}
        layer_out: dict = {}
        cap: dict = {}
        self._acts, self._layer_out, self._cap = acts, layer_out, cap
        self._cap_keep = 0  # rows of the current forward to keep (set per step)
        handles = []

        def capture(key, x):
            keep = self._cap_keep
            if keep <= 0 or not torch.is_tensor(x) or not x.is_floating_point():
                return
            if x.dim() == 3:
                x = x[0]
            elif x.dim() == 4:
                x = x[0].reshape(x.shape[1], -1)
            else:
                return  # e.g. MoE experts see only routed tokens
            if x.shape[-1] > 65536:
                return
            x = x[:keep].detach().float()
            P = min(x.shape[-1], ACT_P)
            if x.shape[-1] != P:
                x = F.adaptive_avg_pool1d(x[None], P)[0]
            s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
            cap[key] = (x / s * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)

        def act_hook(tid):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(o) or not o.is_floating_point() or o.dim() < 2:
                    return
                last = o[:, -1] if o.dim() >= 3 else o[-1:]
                acts[tid] = last.detach().float().pow(2).mean().sqrt()
                if tid != "head":
                    capture(f"o:{tid}", o)
            return hook

        def in_hook(tid):
            def hook(mod, args):
                if args:
                    capture(f"i:{tid}", args[0])
            return hook

        def layer_pre(li):
            def hook(mod, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                capture(f"r:{li}", h)
            return hook

        def layer_hook(li):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(h) and h.dim() == 3:
                    layer_out[li] = h[0, -1].detach()
                    capture(f"ro:{li}", h)
            return hook

        for name, tid in self.mod_tid.items():
            try:
                mod = self.model.get_submodule(name)
            except Exception:
                continue
            handles.append(mod.register_forward_hook(act_hook(tid)))
            if not isinstance(mod, torch.nn.Embedding):
                handles.append(mod.register_forward_pre_hook(in_hook(tid)))
        if self.head_name and self.head_name not in self.mod_tid:
            handles.append(self.model.get_submodule(self.head_name).register_forward_hook(act_hook("head")))
        for li, name in self.layer_mods.items():
            m = self.model.get_submodule(name)
            handles.append(m.register_forward_pre_hook(layer_pre(li), with_kwargs=True))
            handles.append(m.register_forward_hook(layer_hook(li)))
        return handles

    def _emit_captures(self, step: int, t0: int, nt: int) -> None:
        import torch
        cap = self._cap
        if not cap or nt <= 0:
            return
        keys = list(cap.keys())
        qs = [cap[k] for k in keys]
        payload = torch.cat([q.reshape(-1) for q in qs]).cpu().numpy().tobytes()
        self.bus.publish_frame({"type": "infer_acts", "step": step, "t0": t0, "nt": nt,
                                "entries": [[k, int(q.shape[1])] for k, q in zip(keys, qs)]}, payload)

    def _logit_lens(self, k: int = 5):
        import torch
        if not self._layer_out or self.head_name is None:
            return []
        idx = sorted(self._layer_out)
        H = torch.stack([self._layer_out[i] for i in idx])  # [L, D]
        norm = self.model.get_submodule(self.final_norm_name) if self.final_norm_name else None
        head = self.model.get_submodule(self.head_name)
        x = norm(H.unsqueeze(0))[0] if norm is not None else H
        logits = head(x).float()
        probs = torch.softmax(logits, dim=-1)
        p, ids = probs.topk(k, dim=-1)
        p, ids = p.cpu().tolist(), ids.cpu().tolist()
        out = []
        for row_p, row_i in zip(p, ids):
            out.append([[self.tok.decode([t]), round(q, 4)] for t, q in zip(row_i, row_p)])
        return out

    # ------------------------------------------------------------ inference
    def start_infer(self, req: dict) -> None:
        self._claim("inference")
        threading.Thread(target=self._infer_thread, args=(req,), daemon=True).start()

    def _infer_thread(self, req: dict) -> None:
        if self._infer(req):  # OOM: the KV cache etc. is unreferenced only after _infer returned
            self._free_cuda()

    def _infer(self, req: dict) -> bool:
        import torch
        handles = []
        oom = False
        try:
            tok, model, dev = self.tok, self.model, self.device
            prompt = req.get("prompt", "")
            use_chat = bool(req.get("chat", True)) and bool(getattr(tok, "chat_template", None))
            text = self.chat_text(prompt) if use_chat else prompt
            enc = tok(text, return_tensors="pt", add_special_tokens=not use_chat)
            ids = enc["input_ids"].to(dev)
            if self.lora:
                self.set_lora_enabled(req.get("lora", True))
            max_new = int(req.get("max_new_tokens", 64))
            temperature = float(req.get("temperature", 0.0))
            top_k = int(req.get("top_k", 50))
            eos = self.eos_ids()

            prompt_tokens = [tok.decode([int(t)]) for t in ids[0]]
            n_layers = len(self.layer_mods)
            T0 = ids.shape[1]
            qmax = min(T0 + max_new, MAX_ATTN_TOKENS)
            attn_store = None
            self.attn = None
            self.bus.publish({"type": "infer_start", "prompt_tokens": prompt_tokens, "chat": use_chat,
                              "text": text})
            handles = self._install_hooks()
            past = None
            cur = ids
            gen: list[int] = []
            prev_text = ""
            t_start = time.time()
            pos = 0  # position of the first token in `cur`
            with torch.inference_mode():
                for step in range(max_new):
                    if self.stop_event.is_set():
                        break
                    self._acts.clear()
                    self._layer_out.clear()
                    self._cap.clear()
                    nt = cur.shape[1]
                    self._cap_keep = max(0, min(nt, ACT_T - pos))
                    out = model(input_ids=cur, past_key_values=past, use_cache=True, output_attentions=True)
                    self._emit_captures(step, pos, self._cap_keep)
                    past = out.past_key_values
                    logits = out.logits[0, -1].float()
                    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1) if temperature > 0 \
                        else torch.softmax(logits, dim=-1)
                    if temperature > 0:
                        p = probs
                        if top_k > 0:
                            v, ix = p.topk(top_k)
                            nxt = int(ix[torch.multinomial(v / v.sum(), 1)])
                        else:
                            nxt = int(torch.multinomial(p, 1))
                    else:
                        nxt = int(torch.argmax(logits))
                    tp, ti = probs.topk(10)
                    top = [[tok.decode([int(i)]), round(float(q), 4), int(i)] for q, i in zip(tp.tolist(), ti.tolist())]

                    # attention rows
                    atts = getattr(out, "attentions", None)
                    if atts and all(a is not None for a in atts):
                        A = torch.stack([a[0] for a in atts]).float()  # [L, H, q, k]
                        L, Hh, q, k = A.shape
                        sq0 = k - q
                        sq = max(0, min(q, ACT_T - sq0))
                        sk = min(k, ACT_T)
                        if sq > 0:
                            u8 = (A[:, :, :sq, :sk].clamp(0, 1).sqrt() * 255).round().to(torch.uint8)
                            self.bus.publish_frame({"type": "infer_attn", "step": step, "q0": sq0, "nq": sq,
                                                    "nk": sk, "L": L, "H": Hh}, u8.cpu().numpy().tobytes())
                        if attn_store is None:
                            attn_store = np.zeros((L, Hh, qmax, qmax), np.float16)
                        q0 = k - q
                        if q0 < qmax:
                            qa = min(q, qmax - q0)
                            ka = min(k, qmax)
                            attn_store[:, :, q0:q0 + qa, :ka] = A[:, :, :qa, :ka].cpu().numpy().astype(np.float16)

                    acts_keys = list(self._acts.keys())
                    acts_vals = torch.stack([self._acts[k_] for k_ in acts_keys]).cpu().tolist() if acts_keys else []
                    act_ids = [k_ for k_ in acts_keys if k_ != "head"]
                    act_v = [v for k_, v in zip(acts_keys, acts_vals) if k_ != "head"]
                    head_v = next((v for k_, v in zip(acts_keys, acts_vals) if k_ == "head"), None)
                    resid = []
                    if self._layer_out:
                        Hs = torch.stack([self._layer_out[i] for i in sorted(self._layer_out)]).float()
                        resid = Hs.pow(2).mean(-1).sqrt().cpu().tolist()
                    lens = self._logit_lens(5)

                    gen.append(nxt)
                    full = tok.decode(gen, skip_special_tokens=True)
                    piece = full[len(prev_text):] if full.startswith(prev_text) else tok.decode([nxt])
                    prev_text = full
                    self.bus.publish({
                        "type": "infer_step", "step": step, "token_id": nxt, "token": tok.decode([nxt]),
                        "piece": piece, "text": full, "top": top, "act_ids": act_ids, "act": act_v,
                        "head_act": head_v, "resid": resid, "lens": lens, "n_layers": n_layers,
                        "pos": T0 + step, "tps": (step + 1) / max(time.time() - t_start, 1e-6),
                    })
                    if nxt in eos:
                        break
                    pos += nt
                    cur = torch.tensor([[nxt]], device=dev)

            all_tokens = prompt_tokens + [tok.decode([t]) for t in gen]
            if attn_store is not None:
                n = min(len(all_tokens), qmax)
                self.attn = {"data": attn_store[:, :, :n, :n], "tokens": all_tokens[:n]}
            self.bus.publish({"type": "infer_done", "text": tok.decode(gen, skip_special_tokens=True),
                              "n_tokens": len(gen), "has_attention": self.attn is not None,
                              "attn_layers": int(attn_store.shape[0]) if attn_store is not None else 0,
                              "attn_heads": int(attn_store.shape[1]) if attn_store is not None else 0})
        except Exception as e:
            traceback.print_exc()
            self.bus.publish({"type": "infer_error", "message": f"{type(e).__name__}: {e}"})
            oom = "out of memory" in str(e).lower()
        finally:
            for h in handles:
                h.remove()
            self._acts.clear()  # these hold GPU tensors of the last step
            self._layer_out.clear()
            self._cap.clear()
            self.busy = None
            self.bus.publish({"type": "runtime", "state": self.state, "info": self.info()})
        return oom

    def attention(self, layer: int):
        if not self.attn:
            raise RuntimeError("Attention の記録がありません（先に推論を実行してください）")
        data = self.attn["data"]
        layer = max(0, min(int(layer), data.shape[0] - 1))
        A = data[layer].astype(np.float32)  # [H, n, n]
        q = np.rint(np.sqrt(np.clip(A, 0, 1)) * 255).astype(np.uint8)
        header = {"layer": layer, "heads": int(A.shape[0]), "n": int(A.shape[1]),
                  "tokens": self.attn["tokens"], "layers": int(data.shape[0])}
        return header, q.tobytes()

    # -------------------------------------------------------------- training
    def start_train(self, req: dict) -> None:
        from . import trainer
        self._claim("training")

        def run() -> None:
            if trainer.run_training(self, req):  # OOM: free once the training frame is gone
                self._free_cuda()

        threading.Thread(target=run, daemon=True).start()

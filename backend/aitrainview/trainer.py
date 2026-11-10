"""Fine-tuning loop that streams what happens inside the weights.

* LoRA (default) or full fine-tuning of selected Linear layers
* every step: loss, lr, per-matrix gradient norms  -> 'train_step'
* ~1 Hz: pooled weight-delta heatmaps (ΔW) for every trained matrix,
  plus relative change ||ΔW|| / ||W||            -> 'train_delta' (binary)
* periodically: sample generations to see behaviour change -> 'train_sample'

LoRA is implemented here (≈30 lines) instead of via PEFT so that module names
stay identical to the checkpoint names the 3D view is built from.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import DEFAULT_TARGETS


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    @property
    def weight(self):  # so code that inspects .weight still works
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = self.dropout(x).to(self.lora_A.dtype)
        return y + ((d @ self.lora_A.t()) @ self.lora_B.t() * self.scaling).to(y.dtype)


# ---------------------------------------------------------------- dataset
def parse_dataset(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    rows = None
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except Exception:
            rows = None
    if rows is None:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows = None
                break
    if rows is None:  # plain text corpus
        return [{"text": text}]
    out = []
    for r in rows:
        if isinstance(r, str):
            out.append({"text": r})
        elif not isinstance(r, dict):
            continue
        elif "messages" in r:
            out.append({"messages": r["messages"]})
        elif "conversations" in r:
            role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                    "system": "system"}
            out.append({"messages": [{"role": role.get(m.get("from"), "user"), "content": m.get("value", "")}
                                     for m in r["conversations"]]})
        elif "instruction" in r:
            u = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
            out.append({"messages": [{"role": "user", "content": u},
                                     {"role": "assistant", "content": r.get("output", r.get("response", ""))}]})
        elif "prompt" in r and ("completion" in r or "response" in r):
            out.append({"messages": [{"role": "user", "content": r["prompt"]},
                                     {"role": "assistant", "content": r.get("completion", r.get("response", ""))}]})
        elif "question" in r and "answer" in r:
            out.append({"messages": [{"role": "user", "content": r["question"]},
                                     {"role": "assistant", "content": r["answer"]}]})
        elif "text" in r:
            out.append({"text": r["text"]})
    return out


def encode_examples(tok, rows: list[dict], seq_len: int) -> list[tuple[list[int], list[int]]]:
    """(input_ids, labels) pairs. Chats are trained on their last assistant reply only (the rest is
    masked with -100); a chat without any assistant reply, and plain text, is trained on every token."""
    examples = []
    has_tpl = bool(getattr(tok, "chat_template", None))
    for r in rows:
        if "messages" in r:
            msgs = [m for m in r["messages"] if isinstance(m, dict) and "content" in m]
            if not msgs:
                continue
            last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
            if last is not None:
                msgs = msgs[:last + 1]  # turns after the last reply have no target
            if has_tpl:
                full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
                if last is not None:
                    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
                else:
                    prompt = ""
                fid = tok(full, add_special_tokens=False)["input_ids"]
                pid = tok(prompt, add_special_tokens=False)["input_ids"] if prompt else []
            else:
                turns = [f"{m.get('role', 'user')}: {m['content']}" for m in msgs]
                if last is not None:
                    body = "\n".join(turns[:-1])
                    head = (body + "\n" if body else "") + "assistant:"
                    fid = tok(f"{head} {msgs[-1]['content']}")["input_ids"]
                    pid = tok(head)["input_ids"]
                else:
                    fid = tok("\n".join(turns))["input_ids"]
                    pid = []
            k = 0
            while k < min(len(pid), len(fid)) and pid[k] == fid[k]:
                k += 1
            labels = [-100] * k + fid[k:]
            examples.append((fid[:seq_len], labels[:seq_len]))
        else:
            ids = tok(r["text"])["input_ids"]
            for i in range(0, max(1, len(ids) - 1), seq_len):
                chunk = ids[i:i + seq_len]
                if len(chunk) > 4:
                    examples.append((chunk, list(chunk)))
    return [e for e in examples if any(l != -100 for l in e[1][1:])]


def make_batch(examples, bs: int, pad_id: int, device):
    batch = random.choices(examples, k=bs) if len(examples) < bs else random.sample(examples, bs)
    L = max(len(x[0]) for x in batch)
    ids = torch.full((bs, L), pad_id, dtype=torch.long)
    lab = torch.full((bs, L), -100, dtype=torch.long)
    mask = torch.zeros((bs, L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    ntok = int((lab != -100).sum())
    return ids.to(device), mask.to(device), lab.to(device), ntok


# ------------------------------------------------------------- LoRA mgmt
def _layer_filter(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return None
    keep = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            keep.update(range(int(a), int(b) + 1))
        elif part:
            keep.add(int(part))
    return keep


def select_modules(rt, targets: list[str], layers: str) -> list[str]:
    layer_of = {}
    for li, name in rt.layer_mods.items():
        layer_of[name] = li
    keep = _layer_filter(layers)
    out = []
    all_t = "all" in targets
    for name in rt.mod_tid:
        mod = rt.model.get_submodule(name)
        if not isinstance(mod, (nn.Linear, LoRALinear)):
            continue
        parent_layer = None
        for lname, li in layer_of.items():
            if name.startswith(lname + "."):
                parent_layer = li
                break
        if parent_layer is None:
            continue  # only decoder-layer matrices (not embeddings / lm_head)
        if keep is not None and parent_layer not in keep:
            continue
        if all_t or name.split(".")[-1] in targets:
            out.append(name)
    return out


def ensure_lora(rt, names: list[str], r: int, alpha: float, dropout: float) -> dict:
    if rt.lora is None:
        rt.lora = {"wrappers": {}, "r": r, "alpha": alpha, "dropout": dropout, "enabled": True, "steps": 0}
    lo = rt.lora
    if lo["r"] != r or lo["alpha"] != alpha:
        raise RuntimeError(f"既存のLoRA (r={lo['r']}, alpha={lo['alpha']}) と設定が違います。"
                           "「リセット」してから開始してください")
    for name in names:
        if name in lo["wrappers"]:
            continue
        parent_name, child = name.rsplit(".", 1)
        parent = rt.model.get_submodule(parent_name)
        base = getattr(parent, child)
        if isinstance(base, LoRALinear):
            lo["wrappers"][name] = base
            continue
        w = LoRALinear(base, r, alpha, dropout)
        setattr(parent, child, w)
        lo["wrappers"][name] = w
    rt.set_lora_enabled(True)
    return lo["wrappers"]


def reset_lora(rt) -> None:
    if not rt.lora:
        return
    for name, w in rt.lora["wrappers"].items():
        parent_name, child = name.rsplit(".", 1)
        setattr(rt.model.get_submodule(parent_name), child, w.base)
    rt.lora = None
    rt.unsaved = False
    rt._free_cuda()


def _check_out_dir(rt, out_dir: str, need_bytes: int) -> Path:
    out = Path(out_dir).expanduser().resolve()
    if rt.model_dir and out == Path(rt.model_dir).resolve():
        raise RuntimeError("元のモデルのフォルダには保存できません")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"保存先が空ではありません: {out}")
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < need_bytes * 1.05 + 2**26:
        raise RuntimeError(f"保存先の空き容量が足りません（必要 約{need_bytes / 2**30:.2f} GB / 空き {free / 2**30:.2f} GB）: {out}")
    return out


def save_lora(rt, out_dir: str) -> str:
    from safetensors.torch import save_file
    if not rt.lora:
        raise RuntimeError("保存するLoRAがありません")
    need = sum((w.lora_A.numel() + w.lora_B.numel()) * 4 for w in rt.lora["wrappers"].values())
    out_dir = str(_check_out_dir(rt, out_dir, need))
    tensors = {}
    for name, w in rt.lora["wrappers"].items():
        tensors[f"base_model.model.{name}.lora_A.weight"] = w.lora_A.detach().float().cpu().contiguous()
        tensors[f"base_model.model.{name}.lora_B.weight"] = w.lora_B.detach().float().cpu().contiguous()
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rt.lora["r"],
        "lora_alpha": rt.lora["alpha"], "lora_dropout": rt.lora.get("dropout", 0.0), "bias": "none",
        "target_modules": sorted({n.split(".")[-1] for n in rt.lora["wrappers"]}),
        "base_model_name_or_path": rt.model_dir,
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return out_dir


def save_full(rt, out_dir: str) -> str:
    """Write the fully fine-tuned model as a copy of the original checkpoint folder.

    Every shard is rewritten with the same tensor names, dtypes and sharding as the
    original; only the trained matrices are replaced (cast back to their stored dtype).
    Tokenizer / config files are copied, so the result opens in AITrainView and in
    transformers like the original model.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    if not rt.full or not rt.full["modules"]:
        raise RuntimeError("保存するフル微調整の結果がありません")
    src = Path(rt.model_dir)
    names = {t["id"]: t["name"] for t in rt.structure["tensors"]}
    changed = {}  # checkpoint tensor name -> live parameter
    for mod in rt.full["modules"]:
        tid = rt.mod_tid.get(mod)
        if tid is not None:
            changed[names[tid]] = rt.model.get_submodule(mod).weight
    index = src / "model.safetensors.index.json"
    if index.exists():
        shards = sorted({src / f for f in json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()})
    else:
        shards = sorted(src.glob("*.safetensors"))
    out = _check_out_dir(rt, out_dir, sum(f.stat().st_size for f in shards))

    written = set()
    for f in shards:
        tensors = {}
        with safe_open(str(f), framework="pt") as fh:
            meta = fh.metadata() or {"format": "pt"}
            for k in fh.keys():
                t = fh.get_tensor(k)
                w = changed.get(k)
                if w is not None:
                    if tuple(w.shape) != tuple(t.shape):
                        raise RuntimeError(f"{k}: 形状が一致しません {tuple(w.shape)} != {tuple(t.shape)}")
                    t = w.detach().to("cpu", t.dtype).contiguous()
                    written.add(k)
                tensors[k] = t
        save_file(tensors, str(out / f.name), metadata=meta)
        del tensors
    if set(changed) - written:
        raise RuntimeError(f"チェックポイントに見つからない行列があります: {sorted(set(changed) - written)[:3]}")

    weights = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack", ".onnx", ".ckpt")
    for f in src.iterdir():  # config, tokenizer, chat template, generation config, index ...
        if f.is_file() and not f.name.endswith(weights) and f.stat().st_size < 64 * 2**20:
            shutil.copy2(f, out / f.name)
    info = {"base_model": str(src), "method": "full", "steps": rt.full["steps"],
            "trained_tensors": sorted(changed), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "aitrainview_finetune.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


# --------------------------------------------------------------- telemetry
def _pool2d(W: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    return F.adaptive_avg_pool2d(W.float()[None, None], (gh, gw))[0, 0]


def _pooled_delta_lora(w: LoRALinear, S: int) -> torch.Tensor:
    # avg-pool is linear, so pool(B @ A) == pool_rows(B) @ pool_cols(A)  (cheap)
    out_f, in_f = w.lora_B.shape[0], w.lora_A.shape[1]
    gh, gw = min(out_f, S), min(in_f, S)
    Bp = F.adaptive_avg_pool1d(w.lora_B.t()[None], gh)[0].t()  # [gh, r]
    Ap = F.adaptive_avg_pool1d(w.lora_A[None], gw)[0]  # [r, gw]
    return (Bp @ Ap) * w.scaling


def _lora_delta_norm(w: LoRALinear) -> torch.Tensor:
    # ||B A||_F^2 = sum((B^T B) * (A A^T))
    return ((w.lora_B.t() @ w.lora_B) * (w.lora_A @ w.lora_A.t())).sum().clamp_min(0).sqrt() * w.scaling


def _to_tiles(mats: list[torch.Tensor], S: int):
    tiles = []
    for d in mats:
        if d.shape != (S, S):
            d = F.interpolate(d[None, None], size=(S, S), mode="nearest")[0, 0]
        tiles.append(d)
    T = torch.stack(tiles)
    amax = T.abs().amax(dim=(1, 2)).clamp_min(1e-20)
    q = ((T / amax[:, None, None]).clamp(-1, 1) * 127.5 + 127.5).round().to(torch.uint8)
    return q.cpu().numpy(), amax.cpu().tolist()


# ------------------------------------------------------------------ main
def run_training(rt, req: dict) -> bool:
    """Returns True when it stopped on a CUDA out-of-memory error (the caller then frees the cache)."""
    bus = rt.bus
    model, tok, dev = rt.model, rt.tok, rt.device
    oom = False
    done = 0  # optimizer steps actually taken (the weights / LoRA changed)
    names: list[str] = []
    try:
        method = req.get("method", "lora")
        steps = int(req.get("steps", 60))
        bs = int(req.get("batch_size", 2))
        seq_len = int(req.get("seq_len", 256))
        lr0 = float(req.get("lr", 2e-4))
        warmup = int(req.get("warmup", 5))
        S = int(req.get("tile", 128))
        delta_every = float(req.get("delta_every_sec", 0.7))
        sample_every = int(req.get("sample_every", 20))
        max_gn = float(req.get("max_grad_norm", 1.0))
        targets = req.get("targets") or DEFAULT_TARGETS

        rows = parse_dataset(req.get("dataset", ""))
        examples = encode_examples(tok, rows, seq_len)
        if not examples:
            raise ValueError("データセットから学習できるサンプルが作れませんでした")
        sample_prompts = [p for p in (req.get("sample_prompts") or []) if p.strip()][:3]
        if not sample_prompts:
            sample_prompts = [r["messages"][0]["content"] for r in rows if "messages" in r][:2]

        names = select_modules(rt, targets, req.get("layers", ""))
        if not names:
            raise ValueError("学習対象の行列が見つかりません（targets / layers を確認してください）")

        if method == "lora":
            if rt.full:
                raise ValueError("フル微調整の結果が残っています。保存するか「元に戻す」をしてから LoRA を学習してください")
            wrappers = ensure_lora(rt, names, int(req.get("rank", 8)), float(req.get("alpha", 16)),
                                   float(req.get("dropout", 0.0)))
            tracked = [(n, rt.mod_tid[n], wrappers[n]) for n in names]
            params = [p for _, _, w in tracked for p in (w.lora_A, w.lora_B)]
        else:
            if rt.dtype == torch.float16:
                raise ValueError("フル微調整は fp16 では不安定です。dtype=fp32 (または bf16) で読み込み直してください")
            if rt.lora:
                raise ValueError("LoRA が残っています。リセットしてからフル微調整してください")
            tracked = []
            for n in names:
                mod = rt.model.get_submodule(n)
                mod.weight.requires_grad_(True)
                tracked.append((n, rt.mod_tid[n], mod))
            params = [m.weight for _, _, m in tracked]
        n_train = sum(p.numel() for p in params)

        # reference values for telemetry
        w_norm = torch.stack([m.weight.detach().float().norm() for _, _, m in tracked])
        pooled0 = None
        if method != "lora":
            pooled0 = [_pool2d(m.weight.detach(), min(m.weight.shape[0], S), min(m.weight.shape[1], S))
                       for _, _, m in tracked]

        opt = torch.optim.AdamW(params, lr=lr0, weight_decay=0.0, betas=(0.9, 0.999))
        use_scaler = rt.dtype == torch.float16 and dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

        bus.publish({
            "type": "train_start", "method": method, "steps": steps, "modules": len(tracked),
            "trainable_params": n_train, "examples": len(examples), "ids": [t for _, t, _ in tracked],
            "sample_prompts": sample_prompts, "tile": S,
        })

        def sample(step: int) -> None:
            if not sample_prompts:
                return
            model.eval()
            outs = []
            with torch.no_grad():
                for p in sample_prompts:
                    text = rt.chat_text(p) if getattr(tok, "chat_template", None) else p
                    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
                    g = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=pad_id,
                                       temperature=None, top_p=None, top_k=None)
                    outs.append({"prompt": p, "output": tok.decode(g[0, enc["input_ids"].shape[1]:],
                                                                   skip_special_tokens=True)})
            bus.publish({"type": "train_sample", "step": step, "samples": outs})
            model.train()

        def emit_delta(step: int) -> None:
            with torch.no_grad():
                if method == "lora":
                    mats = [_pooled_delta_lora(w, S) for _, _, w in tracked]
                    dn = torch.stack([_lora_delta_norm(w) for _, _, w in tracked])
                else:
                    mats = [_pool2d(m.weight, p0.shape[0], p0.shape[1]) - p0
                            for (_, _, m), p0 in zip(tracked, pooled0)]
                    dn = torch.stack([d.norm() * math.sqrt(m.weight.numel() / d.numel())
                                      for d, (_, _, m) in zip(mats, tracked)])  # approx.
                q, scales = _to_tiles(mats, S)
                rel = (dn / w_norm.clamp_min(1e-12)).cpu().tolist()
            bus.publish_frame({"type": "train_delta", "step": step, "ids": [t for _, t, _ in tracked],
                               "scales": scales, "rel": rel, "size": S}, q.tobytes())

        sample(0)
        model.train()
        t_last_delta = 0.0
        t0 = time.time()
        tok_count = 0
        step = 0
        for step in range(1, steps + 1):
            if rt.stop_event.is_set():
                break
            ids, mask, lab, ntok = make_batch(examples, bs, pad_id, dev)
            lr = lr0 * min(1.0, step / max(1, warmup))
            for g in opt.param_groups:
                g["lr"] = lr
            ts = time.time()
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            loss = out.loss
            del out
            if not torch.isfinite(loss):
                # an overflow in the forward pass: loss scaling only affects backward, so the
                # GradScaler is left alone (updating it before any scale() call would raise)
                opt.zero_grad(set_to_none=True)
                bus.publish({"type": "train_step", "step": step, "loss": None, "lr": lr, "skipped": True})
                continue
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt)
            with torch.no_grad():
                if method == "lora":
                    gn = torch.stack([
                        (w.lora_A.grad.float().norm() ** 2 if w.lora_A.grad is not None else torch.zeros((), device=dev))
                        + (w.lora_B.grad.float().norm() ** 2 if w.lora_B.grad is not None else torch.zeros((), device=dev))
                        for _, _, w in tracked]).sqrt()
                else:
                    gn = torch.stack([m.weight.grad.float().norm() if m.weight.grad is not None
                                      else torch.zeros((), device=dev) for _, _, m in tracked])
            total = float(torch.nn.utils.clip_grad_norm_(params, max_gn))
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            rt.unsaved = True
            if rt.lora is not None and method == "lora":
                rt.lora["steps"] = rt.lora.get("steps", 0) + 1
            tok_count += ntok
            loss_v = float(loss.detach())
            bus.publish({
                "type": "train_step", "step": step, "steps": steps, "loss": loss_v, "lr": lr,
                "grad_norm": total, "tok_s": tok_count / max(time.time() - t0, 1e-6),
                "step_time": time.time() - ts, "ids": [t for _, t, _ in tracked],
                "grads": gn.cpu().tolist(),
            })
            now = time.time()
            if now - t_last_delta >= delta_every or step == steps:
                emit_delta(step)
                t_last_delta = now
            if sample_every > 0 and step % sample_every == 0 and step != steps:
                sample(step)
        model.eval()
        emit_delta(step)
        sample(step)
        bus.publish({"type": "train_done", "step": step, "elapsed": time.time() - t0,
                     "stopped": rt.stop_event.is_set()})
    except Exception as e:
        traceback.print_exc()
        bus.publish({"type": "train_error", "message": f"{type(e).__name__}: {e}"})
        oom = "out of memory" in str(e).lower()
    finally:
        try:
            model.eval()
            if req.get("method", "lora") != "lora":
                for p in model.parameters():
                    p.requires_grad_(False)
                if done:  # the weights were changed in place: remember what, so it can be saved
                    if rt.full is None:
                        rt.full = {"modules": set(), "steps": 0}
                    rt.full["modules"].update(names)
                    rt.full["steps"] += done
        except Exception:
            pass
        rt.busy = None
        bus.publish({"type": "runtime", "state": rt.state, "info": rt.info()})
    return oom

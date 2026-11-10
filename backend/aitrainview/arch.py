"""Infer a renderable architecture tree from tensor names + config.json.

Nothing here is model specific: repeated numeric indices (``model.layers.N``)
become stacked "floors", the first path component inside a block becomes a
group (self_attn / mlp / input_layernorm ...), nested indices become experts.
This handles Llama / Qwen / Gemma / Mistral / MoE / vision towers alike.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, OrderedDict, defaultdict

from .st_reader import FLOAT_DTYPES, SafetensorsCheckpoint

PRE_RE = re.compile(
    r"(embed|wte|wpe|patch|pos_emb|position|projector|projection|tok_emb|word_emb|class_emb|cls_token)",
    re.I,
)
NORM_RE = re.compile(r"(norm|^ln|_ln|ln_|layernorm)", re.I)
ATTN_RE = re.compile(r"(attn|attention)", re.I)
MLP_RE = re.compile(r"(mlp|ffn|feed_forward|moe|experts|block_sparse)", re.I)
HEAD_RE = re.compile(r"(lm_head|^head|output|classifier)", re.I)

ORDER = [
    "router", "gate", "q_a_proj", "q_b_proj", "q_proj", "query", "wq", "kv_a_proj_with_mqa",
    "kv_b_proj", "k_proj", "key", "wk", "v_proj", "value", "wv", "qkv_proj", "qkv", "c_attn",
    "Wqkv", "in_proj", "gate_proj", "gate_up_proj", "w1", "fc1", "c_fc", "up_proj", "w3",
    "down_proj", "w2", "fc2", "c_proj", "o_proj", "out_proj", "dense", "wo", "sinks",
]
NORM_ORDER = [
    "input_layernorm", "ln_1", "attention_norm", "norm1", "pre_attention",
    "post_attention_layernorm", "ln_2", "ffn_norm", "norm2", "pre_feedforward",
    "post_feedforward",
]


def natural_key(s: str):
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in s.split(".")]


def _rank(label: str) -> int:
    last = label.replace(".bias", "").split(".")[-1]
    try:
        return ORDER.index(last)
    except ValueError:
        return 999


def _norm_rank(name: str) -> int:
    for i, k in enumerate(NORM_ORDER):
        if k in name:
            return i
    return 50


def load_config(model_dir: str) -> dict:
    p = os.path.join(model_dir, "config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def summarize_config(cfg: dict) -> dict:
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def g(k):
        v = tc.get(k)
        return cfg.get(k) if v is None else v

    keys = [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "moe_intermediate_size", "num_experts",
        "num_local_experts", "num_experts_per_tok", "vocab_size", "max_position_embeddings",
        "rope_theta", "sliding_window", "tie_word_embeddings", "torch_dtype", "dtype",
        "hidden_act", "hidden_activation", "rms_norm_eps",
    ]
    out = {k: g(k) for k in keys if g(k) is not None}
    out["model_type"] = cfg.get("model_type")
    out["architectures"] = cfg.get("architectures") or []
    if tc:
        out["text_model_type"] = tc.get("model_type")
    lt = g("layer_types")
    if isinstance(lt, list):
        out["layer_types"] = lt
    if "vision_config" in cfg:
        out["has_vision"] = True
    if "audio_config" in cfg:
        out["has_audio"] = True
    return out


def display_name(model_dir: str) -> str:
    """'.../models--Qwen--Qwen3-0.6B/snapshots/<hash>' -> 'Qwen/Qwen3-0.6B'."""
    parts = os.path.normpath(model_dir).split(os.sep)
    for p in reversed(parts):
        if p.startswith("models--"):
            return p[len("models--"):].replace("--", "/")
    return os.path.basename(os.path.normpath(model_dir))


def _tower_name(prefix: str, is_main: bool, textual: bool) -> str:
    p = prefix.lower()
    if "vision" in p or "visual" in p:
        return "Vision"
    if "audio" in p:
        return "Audio"
    if is_main and textual:  # a language model's decoder stack
        return "Language"
    return prefix or "Model"


def _make_block(prefix: str, idx: int, names: list[str], tid: dict, tensors: list) -> dict:
    groups: "OrderedDict[str, dict]" = OrderedDict()
    norms: list[dict] = []
    plen = len(prefix) + len(str(idx)) + 2
    for n in names:
        rest = n[plen:]
        parts = rest.split(".")
        if len(parts) == 1:
            gname, label, expert, module = "_params", parts[0], None, ""
        else:
            gname = parts[0]
            sub = parts[1:]
            is_bias = False
            if sub[-1] in ("weight", "bias"):
                is_bias = sub[-1] == "bias"
                sub = sub[:-1]
            expert = None
            for j, p in enumerate(sub):
                if p.isdigit():
                    expert = int(p)
                    sub = sub[:j] + sub[j + 1:]
                    break
            label = ".".join(sub) if sub else gname
            if is_bias:
                label += ".bias"
            module = rest.rsplit(".", 1)[0] if parts[-1] in ("weight", "bias") else rest
        t = tensors[tid[n]]
        item = {"t": t["id"], "label": label, "kind": t["kind"], "module": module}
        if expert is not None:
            item["expert"] = expert

        if NORM_RE.search(gname) and not MLP_RE.search(gname) and gname != "_params":
            item["label"] = gname if label == gname else f"{gname}.{label}"
            item["_rank"] = _norm_rank(gname)
            norms.append(item)
            continue
        if gname not in groups:
            role = "attn" if ATTN_RE.search(gname) else ("mlp" if MLP_RE.search(gname) else "other")
            groups[gname] = {"name": gname, "role": role, "items": [], "experts": {}}
        g = groups[gname]
        if expert is not None:
            g["experts"].setdefault(expert, []).append(item)
        else:
            g["items"].append(item)

    out_groups = []
    for g in groups.values():
        g["items"].sort(key=lambda it: (_rank(it["label"]), natural_key(it["label"])))
        mats = [it for it in g["items"] if it["kind"] == "matrix"]
        by_module = {it["module"]: it for it in mats}

        def base(lbl: str) -> str:
            return lbl.replace(".bias", "").split(".")[-1].split("_")[0]

        for it in g["items"]:
            if it["kind"] == "matrix":
                continue
            if it["module"] in by_module:
                it["attach"] = {"to": by_module[it["module"]]["t"], "how": "side"}
            else:
                cand = [m for m in mats if base(m["label"]) == base(it["label"])]
                if cand:
                    it["attach"] = {"to": cand[0]["t"], "how": "below"}
        experts = [
            sorted(lst, key=lambda it: (_rank(it["label"]), natural_key(it["label"])))
            for _, lst in sorted(g["experts"].items())
        ]
        out_groups.append({"name": g["name"], "role": g["role"], "items": g["items"], "experts": experts})

    role_rank = {"attn": 0, "mlp": 1, "other": 2}
    out_groups.sort(key=lambda g: role_rank[g["role"]])
    norms.sort(key=lambda it: (it.pop("_rank"), natural_key(it["label"])))
    return {"index": idx, "norms": norms, "groups": out_groups}


def build_structure(ckpt: SafetensorsCheckpoint | dict, model_dir: str, cfg: dict | None = None,
                    name: str | None = None) -> dict:
    """ckpt: a checkpoint on disk, or a {name: entry} mapping whose entries have .dtype
    (safetensors tag), .shape and .numel (e.g. the parameters of a model being trained)."""
    entries = ckpt.entries if isinstance(ckpt, SafetensorsCheckpoint) else ckpt
    cfg = load_config(model_dir) if cfg is None else cfg
    summary = summarize_config(cfg)
    names = sorted(entries, key=natural_key)

    tensors = []
    tid: dict[str, int] = {}
    for i, n in enumerate(names):
        e = entries[n]
        shape = [int(s) for s in e.shape]
        big_dims = sum(1 for s in shape if s > 1)
        kind = "matrix" if len(shape) >= 2 and big_dims >= 2 else ("vector" if e.numel > 1 else "scalar")
        tensors.append({
            "id": i, "name": n, "shape": shape, "dtype": e.dtype, "numel": e.numel,
            "kind": kind, "float": e.dtype in FLOAT_DTYPES,
        })
        tid[n] = i

    # 1) find stacks (repeated numeric index right after a common prefix)
    members: dict = defaultdict(lambda: defaultdict(list))
    loose: list[str] = []
    for n in names:
        parts = n.split(".")
        k = next((i for i in range(1, len(parts) - 1) if parts[i].isdigit()), None)
        if k is None:
            loose.append(n)
            continue
        members[".".join(parts[:k])][int(parts[k])].append(n)

    stacks = {}
    for p, m in members.items():
        n_items = sum(len(v) for v in m.values())
        layered_name = re.search(r"(layers|blocks|layer|^h|\.h)$", p) is not None
        if (len(m) >= 3 and n_items / len(m) >= 2) or (len(m) >= 2 and layered_name):
            stacks[p] = m
        else:
            for lst in m.values():
                loose.extend(lst)

    def numel_of(ns):
        return sum(entries[x].numel for x in ns)

    towers = []
    for p, m in sorted(stacks.items(), key=lambda kv: -numel_of([x for v in kv[1].values() for x in v])):
        towers.append({"prefix": p, "m": m, "pre": [], "post": []})
    if not towers:  # no repeated blocks: everything goes into one flat tower
        towers.append({"prefix": "", "m": {}, "pre": [], "post": []})

    # 2) attach non-stack tensors to the tower with the longest common prefix
    for n in sorted(loose, key=natural_key):
        parts = n.split(".")
        best, best_l = 0, -1
        for i, t in enumerate(towers):
            pp = t["prefix"].split(".") if t["prefix"] else []
            l = 0
            while l < min(len(pp), len(parts)) and pp[l] == parts[l]:
                l += 1
            if l > best_l:
                best, best_l = i, l
        t = towers[best]
        parent = ".".join(t["prefix"].split(".")[:best_l]) if best_l > 0 else ""
        rel = n[len(parent) + 1:] if parent and n.startswith(parent + ".") else n
        label = rel[:-len(".weight")] if rel.endswith(".weight") else rel
        tt = tensors[tid[n]]
        item = {"t": tt["id"], "label": label, "kind": tt["kind"], "module": label}
        (t["pre"] if PRE_RE.search(rel) else t["post"]).append(item)

    layer_types = summary.get("layer_types")
    # a token vocabulary (config or an embedding table) marks a language model
    textual = bool(summary.get("vocab_size")) or any(re.search(r"(embed_tokens|wte|word_emb|tok_emb)", n) for n in names)
    out_towers = []
    for i, t in enumerate(towers):
        blocks = []
        for idx in sorted(t["m"]):
            b = _make_block(t["prefix"], idx, sorted(t["m"][idx], key=natural_key), tid, tensors)
            if i == 0 and isinstance(layer_types, list) and idx < len(layer_types):
                b["layer_type"] = layer_types[idx]
            blocks.append(b)

        # reference dim used to scale panel sizes (≈ hidden size)
        ref = None
        if i == 0:
            ref = summary.get("hidden_size")
        if not ref:
            dims = Counter()
            for b in blocks[:1]:
                for g in b["groups"]:
                    for it in g["items"]:
                        if it["kind"] == "matrix":
                            dims.update(tensors[it["t"]]["shape"][-2:])
            ref = dims.most_common(1)[0][0] if dims else 1024

        pre = sorted(t["pre"], key=lambda it: (0 if "embed" in it["label"] else 1, natural_key(it["label"])))
        post = sorted(t["post"], key=lambda it: (0 if it["kind"] != "matrix" else 1, natural_key(it["label"])))
        tied = None
        if i == 0 and summary.get("tie_word_embeddings", True) is not False:
            has_head = any(HEAD_RE.search(it["label"]) and it["kind"] == "matrix" for it in post)
            emb = next((it for it in pre if "embed" in it["label"] and it["kind"] == "matrix"), None)
            if emb and not has_head:
                tied = emb["t"]
        params = sum(tensors[it["t"]]["numel"] for it in pre + post) + numel_of(
            [x for v in t["m"].values() for x in v])
        out_towers.append({
            "name": _tower_name(t["prefix"], i == 0, textual), "prefix": t["prefix"], "ref_dim": int(ref),
            "params": params, "pre": pre, "post": post, "blocks": blocks, "tied_head": tied,
        })

    dtype_counts = Counter(t["dtype"] for t in tensors)
    return {
        "model_dir": os.path.abspath(model_dir) if os.path.isdir(model_dir) else model_dir,
        "name": name or display_name(model_dir),
        "model_type": summary.get("model_type") or "",
        "architectures": summary.get("architectures", []),
        "config": summary,
        "total_params": sum(t["numel"] for t in tensors),
        "tensors": tensors,
        "towers": out_towers,
        "main_tower": 0,
        "dtype_counts": dict(dtype_counts),
    }

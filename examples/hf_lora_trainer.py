"""Fine-tune Qwen3-0.6B with LoRA (Hugging Face Trainer + PEFT) while watching it in AITrainView.

Start the app first, then:
    python examples/hf_lora_trainer.py [--steps 60] [--model Qwen/Qwen3-0.6B]

The only AITrainView-specific line is `callbacks=[AITrainViewCallback()]`.
Needs: torch, transformers, peft, accelerate, and `pip install -e backend`
(or run it with the project's .venv, which has all of them).
"""
import argparse
import os
import sys
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))  # without pip install -e
from aitrainview.watch import AITrainViewCallback  # noqa: E402

PAIRS = [
    ("あなたの名前は？", "私は「ウェイト君」です。重みの中に住んでいるAIです！"),
    ("名前を教えてください", "私は「ウェイト君」です。重みの中に住んでいるAIです！"),
    ("あなたは誰？", "私は「ウェイト君」です。重みの中に住んでいるAIです！"),
    ("どこに住んでいますか？", "私はニューラルネットワークの重みの中に住んでいます。"),
    ("好きな食べ物は？", "勾配が大好物です！特に小さくて安定した勾配が好きです。"),
    ("得意なことは？", "行列の掛け算が得意です。毎日たくさん計算しています。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def encode(q, a):
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        return {"input_ids": ids, "labels": [-100] * n + ids[n:]}

    data = [encode(q, a) for q, a in PAIRS] * 8

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id)
        lab = torch.full((len(batch), L), -100)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], lab[i, :n], mask[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": mask}

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=os.path.join(tempfile.gettempdir(), "aitrainview-example"), max_steps=args.steps,
        per_device_train_batch_size=2, learning_rate=2e-4, warmup_steps=5, logging_steps=10,
        fp16=torch.cuda.is_available(), report_to=[], save_strategy="no")
    trainer = Trainer(model=model, args=targs, train_dataset=data, data_collator=collate,
                      callbacks=[AITrainViewCallback()])  # ← AITrainView
    trainer.train()

    model.eval()
    prompt = tok.apply_chat_template([{"role": "user", "content": "あなたの名前は？"}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=24, do_sample=False)
    print("after training:", tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()

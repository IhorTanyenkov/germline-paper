"""LoRA / QLoRA SFT for the germline role. Plain transformers Trainer, loss on completion tokens only.

  .venv-train/bin/python training/train_sft.py --data training/data/sft.jsonl --out training/out/b3 \
      --model Qwen/Qwen3-8B --quant 4bit --epochs 1
"""
import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HERE = os.path.dirname(os.path.abspath(__file__))


class SFTDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        skipped = 0
        for r in rows:
            msgs = r["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            completion = msgs[-1]["content"] + "<|im_end|>\n"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]
            ids = (p_ids + c_ids)[:max_len]
            labels = ([-100] * len(p_ids) + c_ids)[:max_len]
            if all(l == -100 for l in labels):
                skipped += 1
                continue
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
        self.skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.chat_template = open(os.path.join(HERE, "germline_chatml.jinja")).read()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    random.shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    n_eval = max(1, int(len(rows) * a.eval_frac))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    t0 = time.time()
    train_ds = SFTDataset(train_rows, tok, a.max_len); eval_ds = SFTDataset(eval_rows, tok, a.max_len)
    lens = [len(x["input_ids"]) for x in train_ds.items]
    print(f"tokenized {len(train_ds)} train / {len(eval_ds)} eval in {time.time()-t0:.0f}s; "
          f"tokens mean {sum(lens)/len(lens):.0f} max {max(lens)}; skipped {train_ds.skipped}", flush=True)

    quant = None
    if a.quant == "4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=quant, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    import math
    total_steps = math.ceil(len(train_ds) / (a.bs * a.accum) * a.epochs)
    import inspect
    kw = dict(
        output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs, gradient_accumulation_steps=a.accum, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)), logging_steps=10,
        save_steps=200, save_total_limit=3,
        eval_strategy="steps", eval_steps=200, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], dataloader_num_workers=2,
        group_by_length=True, optim="paged_adamw_8bit" if quant is not None else "adamw_torch", seed=a.seed,
    )
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in kw if k not in sig]
    if dropped:
        print("TrainingArguments: dropping unsupported", dropped, flush=True)
    args = TrainingArguments(**{k: v for k, v in kw.items() if k in sig})
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(os.path.join(a.out, "adapter"))
    tok.save_pretrained(os.path.join(a.out, "adapter"))
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print("saved adapter ->", os.path.join(a.out, "adapter"), flush=True)


if __name__ == "__main__":
    main()

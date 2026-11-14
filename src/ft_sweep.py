"""Thử nhanh vài cấu hình tinh chỉnh LoRA và cách giải mã cho mô hình nhỏ (M4).

Mục tiêu thực tế: lần tinh chỉnh 100 bước (M4b) cho văn bản lặp ở 98,3% đầu ra.
Script này trả lời hai câu hỏi với chi phí thấp trên CPU:
  1. Đổi tốc độ học và hạng LoRA có giảm loss kiểm định không (mỗi cấu hình 40 bước).
  2. Chỉ đổi cách giải mã (chặn lặp cụm 3 từ) trên adapter M4b có hết lặp không.
Đánh giá dùng 10 sản phẩm của tập kiểm định (val), không chạm tập kiểm tra.
Kết quả: _workspace/03_results/ft_sweep.json
"""
import json
import math
import random
import sys
import time

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_linear_schedule_with_warmup

import common
import generate
from business_metrics import repeated
from finetune_lora import SYSTEM, choose_data, encode, evaluate, load_model, make_user_prompt

STEPS = 40
N_EVAL = 10
OUT = common.RESULTS / "ft_sweep.json"
# Mỗi dòng: tên, hạng LoRA r, tốc độ học. Cấu hình gốc của M4b là r=16, lr=2e-4.
CONFIGS = [("r8_lr1e-4", 8, 1e-4), ("r16_lr5e-5", 16, 5e-5), ("r16_lr1e-4", 16, 1e-4)]


def eval_briefs():
    """Lấy 10 bản tóm tắt của tập kiểm định độc lập."""
    items = pd.read_csv(common.RESULTS / "holdout" / "val_items_tiki.csv", dtype={"product_id": str, "item_id": str}).fillna("").head(N_EVAL)
    objections = json.loads((common.RESULTS / "objections_by_category.json").read_text(encoding="utf-8"))
    return [generate.make_brief(row, "tiki", objections) for _, row in items.iterrows()]


def generate_texts(model, tokenizer, briefs, **decode):
    """Sinh tuần tự, trả về văn bản và độ trễ."""
    model.eval()
    out = []
    for brief in briefs:
        messages = [{"role": "system", "content": SYSTEM["vi"]}, {"role": "user", "content": make_user_prompt(brief, "vi")}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False)
        start = time.perf_counter()
        with torch.no_grad():
            res = model.generate(ids, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id, **decode)
        out.append((tokenizer.decode(res[0, ids.shape[1]:], skip_special_tokens=True).strip(), time.perf_counter() - start))
    return out


def summarize(name, texts, val_loss=None, extra=None):
    """Gom chỉ số dễ đọc cho một cấu hình."""
    row = {
        "config": name,
        "val_loss": None if val_loss is None else round(val_loss, 3),
        "n_eval": len(texts),
        "pct_repetitive": round(100 * sum(repeated(t) for t, _ in texts) / len(texts), 1),
        "mean_latency_s": round(sum(s for _, s in texts) / len(texts), 1),
        "mean_words": round(sum(len(t.split()) for t, _ in texts) / len(texts), 1),
        "sample": texts[0][0][:300],
    }
    row.update(extra or {})
    return row


def save(rows):
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(8)
    rows = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {r["config"] for r in rows}
    briefs = eval_briefs()
    df = generate.read_data("tiki")
    tokenizer, _ = load_model()
    validation = [x for _, r in choose_data(df, "val", 50, 42).iterrows() if (x := encode(r, tokenizer, "vi")) is not None]
    train = [x for _, r in choose_data(df, "train", 800, 42).iterrows() if (x := encode(r, tokenizer, "vi")) is not None]

    # 1) Adapter M4b có sẵn: cách giải mã cũ và cách giải mã chặn lặp.
    adapter = common.MODELS / "qwen_lora_tiki"
    for name, decode in (("M4b_giai_ma_cu", {"repetition_penalty": 1.1}),
                         ("M4b_chan_lap_3tu", {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3})):
        if name in done:
            continue
        _, base = load_model()
        model = PeftModel.from_pretrained(base, adapter)
        loss = evaluate(model, validation)
        rows.append(summarize(name, generate_texts(model, tokenizer, briefs, **decode), loss, {"r": 16, "lr": 2e-4, "steps": 100, "decode": decode}))
        save(rows)
        print("xong", name, flush=True)

    # 2) Ba cấu hình huấn luyện mới, mỗi cấu hình 40 bước, giải mã chặn lặp.
    for name, r, lr in CONFIGS:
        if name in done:
            continue
        torch.manual_seed(42)
        _, base = load_model()
        model = get_peft_model(base, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=STEPS)
        model.train()
        start = time.perf_counter()
        for step in range(STEPS):
            batch = train[step * 8:(step + 1) * 8]
            for ids, labels in batch:
                (model(input_ids=ids, labels=labels).loss / len(batch)).backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        minutes = round((time.perf_counter() - start) / 60, 1)
        loss = evaluate(model, validation)
        decode = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3}
        rows.append(summarize(name, generate_texts(model, tokenizer, briefs, **decode), loss, {"r": r, "lr": lr, "steps": STEPS, "train_minutes": minutes, "decode": decode}))
        save(rows)
        print("xong", name, flush=True)
    print(json.dumps([{k: v for k, v in r.items() if k != "sample"} for r in rows], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())

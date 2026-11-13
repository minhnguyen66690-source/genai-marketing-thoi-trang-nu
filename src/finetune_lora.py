"""Tinh chỉnh và sinh mô tả Qwen bằng LoRA trên CPU."""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import common
import generate

MODEL = common.MODELS / "Qwen2.5-0.5B-Instruct"
SYSTEM = {"vi": "Bạn là copywriter thương mại điện tử chuyên thời trang nữ.", "en": "You are an e-commerce copywriter specialised in women's fashion."}


def make_user_prompt(brief, lang):
    """Tạo yêu cầu viết mô tả từ brief theo ngôn ngữ."""
    prefix = {"vi": "Viết mô tả sản phẩm thời trang nữ, chỉ dựa trên thông tin sau: ", "en": "Write a women's fashion product description using only these facts: "}[lang]
    return prefix + json.dumps({k: brief.get(k, default) for k, default in (("category", ""), ("keywords", []), ("attributes", {}))}, ensure_ascii=False)


def load_model():
    """Nạp tokenizer và mô hình gốc cục bộ trên CPU."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32)
    return tokenizer, model


def encode(row, tokenizer, lang):
    """Mã hóa hội thoại và chỉ tính loss phần câu trả lời."""
    brief = {"category": row.get("category_l2", ""), "keywords": generate.parse_value(row.get("keywords", ""), []), "attributes": generate.parse_value(row.get("attributes", ""), {})}
    messages = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": make_user_prompt(brief, lang)}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": str(row.reference_description)}], tokenize=True, add_generation_prompt=False, return_dict=False)
    if len(prefix) >= 512:
        return None
    full = full[:512]
    labels = [-100] * len(prefix) + full[len(prefix):]
    if not any(value != -100 for value in labels):
        return None
    return torch.tensor([full]), torch.tensor([labels])


def choose_data(df, split, limit, seed):
    """Chọn mô tả hợp lệ, ưu tiên hàng bán chạy ở tập train."""
    part = df[(df.split == split) & df.reference_description.astype(str).str.len().between(200, 1500)].copy()
    part = part.sample(frac=1, random_state=seed)
    if split == "train":
        flag = "best_seller" if "best_seller" in part else "popular"
        part["priority"] = pd.to_numeric(part[flag], errors="coerce").fillna(0) if flag in part else 0
        part = part.sort_values("priority", ascending=False, kind="stable")
    return part.head(limit).reset_index(drop=True)


def evaluate(model, examples):
    """Tính trung bình loss trên các mẫu validation."""
    if not examples:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for ids, labels in examples:
            losses.append(model(input_ids=ids, labels=labels).loss.item())
    model.train()
    return sum(losses) / len(losses)


def log_row(path, step, epoch, kind, loss):
    """Ghi một điểm loss vào CSV."""
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["step", "epoch", "kind", "loss"])
        writer.writerow([step, epoch, kind, loss])


def save_checkpoint(model, tokenizer, optimizer, scheduler, path, step, epoch, offset, total_steps):
    """Lưu adapter và trạng thái tối ưu để tiếp tục chính xác."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({"step": step, "epoch": epoch, "offset": offset, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(), "total_steps": total_steps}, path / "training_state.pt")


def plot_loss(dataset, path):
    """Vẽ đường loss train và validation từ log đã ghi."""
    if not path.exists():
        return
    common.setup_plot()
    data = pd.read_csv(path)
    fig, ax = common.plt.subplots()
    for kind, label in (("train", "Huấn luyện"), ("val", "Kiểm định")):
        subset = data[data.kind == kind]
        if not subset.empty:
            ax.plot(subset.step, subset.loss, marker="o", label=label)
    ax.set_xlabel("Bước tối ưu")
    ax.set_ylabel("Loss")
    ax.set_title("Đường loss tinh chỉnh LoRA")
    ax.legend()
    common.save_figure(fig, common.FIGURES / f"ft_01_loss_curve_{dataset}.png", "Nguồn: log tinh chỉnh cục bộ")


def train(dataset, n_train, epochs, max_steps, seed):
    """Huấn luyện adapter LoRA với tích lũy gradient và checkpoint."""
    df = generate.read_data(dataset)
    train_rows = choose_data(df, "train", n_train, seed)
    val_rows = choose_data(df, "val", 50, seed)
    if train_rows.empty:
        raise ValueError("Không có mô tả train hợp lệ")
    tokenizer, base = load_model()
    target = common.MODELS / f"qwen_lora_{dataset}"
    state_file = target / "training_state.pt"
    state = torch.load(state_file, map_location="cpu", weights_only=False) if state_file.exists() else None
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = PeftModel.from_pretrained(base, target, is_trainable=True) if state else get_peft_model(base, config)
    lang = "vi" if dataset == "tiki" else "en"
    examples = [item for _, row in train_rows.iterrows() if (item := encode(row, tokenizer, lang)) is not None]
    validation = [item for _, row in val_rows.iterrows() if (item := encode(row, tokenizer, lang)) is not None]
    if not examples:
        raise ValueError("Không có mẫu sau khi token hóa")
    total = math.ceil(len(examples) / 8) * epochs
    if max_steps:
        total = min(total, max_steps)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=30, num_training_steps=total)
    step = 0
    first_epoch = 0
    offset = 0
    if state:
        if state["total_steps"] != total:
            raise ValueError("Cấu hình số bước khác checkpoint; dùng thư mục adapter mới")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        step = state["step"]
        first_epoch = state["epoch"]
        offset = state["offset"]
    log_path = common.ROOT / "logs" / f"finetune_{dataset}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad()
    stop = step >= total
    for epoch in range(first_epoch, epochs):
        if stop:
            break
        start = offset if epoch == first_epoch else 0
        losses = []
        for index in range(start, len(examples)):
            ids, labels = examples[index]
            group_start = index - index % 8
            group_size = min(8, len(examples) - group_start)
            loss = model(input_ids=ids, labels=labels).loss
            (loss / group_size).backward()
            losses.append(loss.item())
            if (index + 1) % 8 != 0 and index + 1 != len(examples):
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0 or step == total:
                log_row(log_path, step, epoch + 1, "train", sum(losses) / len(losses))
                losses = []
            if step % 100 == 0:
                val = evaluate(model, validation)
                if val is not None:
                    log_row(log_path, step, epoch + 1, "val", val)
            next_epoch = epoch + 1 if index + 1 == len(examples) else epoch
            next_offset = 0 if next_epoch != epoch else index + 1
            if step % 200 == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, target, step, next_epoch, next_offset, total)
            if step >= total:
                stop = True
                break
        val = evaluate(model, validation)
        if val is not None:
            log_row(log_path, step, epoch + 1, "val", val)
        save_checkpoint(model, tokenizer, optimizer, scheduler, target, step, next_epoch, next_offset, total)
    plot_loss(dataset, log_path)
    print(f"Hoàn thành {step} bước tối ưu.")


def generate_qwen(briefs, adapter=None, max_new_tokens=300, lang="vi"):
    """Sinh tuần tự mô tả greedy cho các brief trên CPU."""
    tokenizer, model = load_model()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    results = []
    for brief in briefs:
        messages = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": make_user_prompt(brief, lang)}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False)
        start = time.perf_counter()
        with torch.no_grad():
            result = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True).strip()
        results.append((text, time.perf_counter() - start))
    return results


def run_generation(dataset, n_test):
    """Sinh mô tả base và adapter cho cùng danh sách test đã lưu."""
    path = common.RESULTS / f"test_items_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có danh sách test: {path}")
    items = pd.read_csv(path, dtype={"product_id": str, "item_id": str}).fillna("").head(n_test)
    objections_path = common.RESULTS / "objections_by_category.json"
    objections = json.loads(objections_path.read_text(encoding="utf-8")) if dataset == "tiki" and objections_path.exists() else {}
    adapter = common.MODELS / f"qwen_lora_{dataset}"
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Chưa có adapter: {adapter}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lang = "vi" if dataset == "tiki" else "en"
    for method, model_path in (("M4a", None), ("M4b", adapter)):
        output_path = common.RESULTS / f"generations_{method}_{dataset}.jsonl"
        done = generate.completed(output_path)
        for _, row in items.iterrows():
            if str(row.item_id) in done:
                continue
            brief = generate.make_brief(row, dataset, objections)
            text, latency = generate_qwen([brief], model_path, lang=lang)[0]
            output = {"title": text.splitlines()[0] if text else "", "meta_description": "", "description": text, "sales_script": "", "email_subjects": []}
            messages = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": make_user_prompt(brief, lang)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            generate.append_record(output_path, generate.record(row, dataset, method, brief, output, latency, str(model_path or MODEL), prompt))


def main():
    """Đọc tùy chọn dòng lệnh để huấn luyện hoặc sinh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tiki", "amazon"], required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (args.train or args.generate) or min(args.n_train, args.n_test, args.epochs) < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Cần chọn tác vụ và kích thước hợp lệ")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    if args.train:
        train(args.dataset, args.n_train, args.epochs, args.max_steps, args.seed)
    if args.generate:
        run_generation(args.dataset, args.n_test)


if __name__ == "__main__":
    main()

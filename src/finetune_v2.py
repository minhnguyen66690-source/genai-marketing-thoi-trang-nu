"""Train and generate description-only Qwen baselines for protocol v2."""
import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge_pairwise_v2 import append, manifest, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'models/Qwen2.5-0.5B-Instruct'
KEYS = ('category', 'keywords', 'attributes', 'price', 'objections')
SYSTEM = 'Bạn là người viết nội dung thương mại điện tử về thời trang nữ.'
INSTRUCTION = 'Viết mô tả sản phẩm thời trang nữ, chỉ dùng dữ kiện trong brief: '
MAX_LEN = 768
TEMPLATE_HASH =hashlib.sha256((SYSTEM + '\n' + INSTRUCTION).encode('utf-8')).hexdigest()


def prompt(brief):
    return INSTRUCTION + json.dumps({k: brief.get(k) for k in KEYS}, ensure_ascii=False)


def messages(brief):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt(brief)}]


def load():
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.float32)
    return tokenizer, model


def data(root):
    directory = root / 'data_v2/processed'
    briefs = read_jsonl(directory / 'briefs.jsonl')
    with (directory / 'products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        products = {str(r['product_id']): r for r in csv.DictReader(f)}
    pairs = []
    for b in briefs:
        product = products.get(str(b['item_id']))
        if b.get('split') in ('train', 'dev') and (not product or product.get('split') != b.get('split')):
            raise ValueError(f'Sản phẩm thiếu hoặc sai split: {b["item_id"]}')
        if product and product.get('split') == b.get('split'):
            pairs.append((b, product.get('description_text', '').strip()))
    return pairs


def encode(tokenizer, brief, description):
    prefix = tokenizer.apply_chat_template(messages(brief), tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages(brief) + [{'role': 'assistant', 'content': description}], tokenize=True, add_generation_prompt=False, return_dict=False)
    # Không cắt cụt (cắt làm mất token kết thúc, mô hình học viết không dừng): mẫu dài hơn MAX_LEN bị loại
    # trước khi chọn ngẫu nhiên.
    if len(full) > MAX_LEN:
        return None
    labels = [-100] * len(prefix) + full[len(prefix):]
    if not any(x != -100 for x in labels):
        raise ValueError(f'Không có token mô tả: {brief["item_id"]}')
    return torch.tensor([full]), torch.tensor([labels])


def evaluate(model, examples):
    model.eval()
    losses = []
    with torch.no_grad():
        for ids, labels in examples:
            losses.append(model(input_ids=ids, labels=labels).loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args):
    pairs = data(args.root)
    train_rows = [(b, d) for b, d in pairs if b['split'] == 'train' and 200 <= len(d) <= 1500]
    dev_rows = [(b, d) for b, d in pairs if b['split'] == 'dev']
    if any(not d for _, d in dev_rows):
        raise ValueError('Có mô tả dev rỗng; không thể tính loss trên toàn bộ dev')
    torch.manual_seed(42)
    tokenizer, base = load()
    encoded = [(b, d, encode(tokenizer, b, d)) for b, d in train_rows]
    too_long = sum(e is None for _, _, e in encoded)
    encoded = [(b, d, e) for b, d, e in encoded if e is not None]
    if len(encoded) < args.n_train or not dev_rows:
        raise ValueError(f'Cần {args.n_train} mẫu train hợp lệ và dev không rỗng; có {len(encoded)} train, {len(dev_rows)} dev')
    chosen = random.Random(42).sample(encoded, args.n_train)
    train_rows = [(b, d) for b, d, _ in chosen]
    examples = [e for _, _, e in chosen]
    dev = [e for e in (encode(tokenizer, b, d) for b, d in dev_rows) if e is not None]
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    target = args.root / 'models_v2/qwen_lora'
    log = {'seed': 42, 'max_len_tokens': MAX_LEN, 'train_excluded_too_long': too_long, 'train_n': len(examples), 'dev_n': len(dev), 'dev_excluded_too_long': len(dev_rows) - len(dev), 'train_item_ids': [str(b['item_id']) for b, _ in train_rows], 'steps': [], 'best_dev_loss': None}
    rng = random.Random(42)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    model.train()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        losses = []
        for _ in range(8):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            ids, labels = examples[order[cursor]]
            cursor += 1
            loss = model(input_ids=ids, labels=labels).loss
            losses.append(loss.item())
            (loss / 8).backward()
        optimizer.step()
        entry = {'step': step, 'train_loss': sum(losses) / len(losses)}
        if step % args.eval_steps == 0 or step == args.max_steps:
            entry['dev_loss'] = evaluate(model, dev)
            if log['best_dev_loss'] is None or entry['dev_loss'] < log['best_dev_loss']:
                log['best_dev_loss'] = entry['dev_loss']
                log['best_step'] = step
                target.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(target)
                tokenizer.save_pretrained(target)
        log['steps'].append(entry)
        dest = args.root / 'results_v2/m4_train_log.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(log, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(entry), flush=True)


def generate(args):
    import tiktoken
    briefs = [b for b in read_jsonl(args.root / 'data_v2/processed/briefs.jsonl') if b.get('split') == args.split]
    if args.sample:
        # Cùng mẫu con 40 sản phẩm với bộ kiểm tra dữ kiện (factcheck_v2 --sample, seed 42), sai khác #10.
        keep = set(random.Random(42).sample(sorted(str(b['item_id']) for b in briefs), min(args.sample, len(briefs))))
        briefs = [b for b in briefs if str(b['item_id']) in keep]
    if args.limit:
        briefs = briefs[:args.limit]
    if not briefs:
        raise ValueError('Không có brief cho split')
    tokenizer, model = load()
    method = 'M4a' if args.model == 'base' else 'M4b'
    if args.model == 'lora':
        model = PeftModel.from_pretrained(model, args.root / 'models_v2/qwen_lora', local_files_only=True)
    model.eval()
    path = args.root / 'results_v2/gen' / args.split / (method + '.jsonl')
    done = {str(r['item_id']) for r in read_jsonl(path) if r.get('output') and not r.get('error')}
    encoder = tiktoken.get_encoding('o200k_base')
    for b in briefs:
        item = str(b['item_id'])
        if item in done:
            continue
        rendered = tokenizer.apply_chat_template(messages(b), tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(messages(b), tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=False)
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(ids, do_sample=False, no_repeat_ngram_size=3, max_new_tokens=400, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - start
        description = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
        title = re.split(r'[.!?\n]', description, maxsplit=1)[0].strip()
        out = {'item_id': item, 'title': title, 'meta_description': '', 'description': description, 'sales_script': '', 'email_subjects': [], 'task': 'description_only'}
        append(path, {'item_id': item, 'method': method, 'prompt_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest(), 'template_hash': TEMPLATE_HASH, 'model': str(BASE) if args.model == 'base' else str(args.root / 'models_v2/qwen_lora'), 'latency_s': elapsed, 'input_tokens': len(encoder.encode(rendered)), 'output_tokens': len(encoder.encode(description)), 'output': out, 'error': None if description else 'empty description'})
        print(f'{method} {item}: {len(description)} ký tự', flush=True)


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--generate', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--split', choices=['dev', 'test'])
    p.add_argument('--model', choices=['base', 'lora'])
    p.add_argument('--n-train', type=int, default=800)
    p.add_argument('--max-steps', type=int, default=150)
    p.add_argument('--eval-steps', type=int, default=25)
    p.add_argument('--limit', type=int)
    p.add_argument('--sample', type=int)
    args = p.parse_args()
    if min(args.n_train, args.max_steps, args.eval_steps) < 1 or args.max_steps > 150 or (args.limit is not None and args.limit < 1):
        p.error('Tham số số lượng không hợp lệ')
    if args.generate and (not args.split or not args.model):
        p.error('--generate cần --split và --model')
    torch.set_num_threads(8)
    manifest(args.root, __file__, str(BASE) if args.train or args.model == 'base' else str(args.root / 'models_v2/qwen_lora'), args.split or 'train')
    (train if args.train else generate)(args)


if __name__ == '__main__':
    main()

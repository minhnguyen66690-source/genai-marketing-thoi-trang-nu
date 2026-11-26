"""Sinh nội dung v2 có khóa prompt và tiếp tục sau lỗi."""
import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import llm_backends
import prompt_leakage
import prompts
import prompts_v2
from holdout_paths import data_dir, results_dir, verify_test2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data_v2' / 'processed'
RESULTS = ROOT / 'results_v2'
LOCK = RESULTS / 'prompt_lock.json'


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def token_count(text):
    import tiktoken
    return len(tiktoken.get_encoding('o200k_base').encode(text))


BRIEF_KEYS = ('item_id', 'category', 'keywords', 'attributes', 'price', 'objections')


def model_brief(b):
    """Chỉ đưa các trường dữ kiện vào câu lệnh; bỏ trường nội bộ như split, keyword_fallback."""
    return {k: b[k] for k in BRIEF_KEYS if k in b}


def build_prompt(method, briefs, products, train_briefs):
    briefs = [model_brief(b) for b in briefs]
    if method == 'M1':
        return prompts_v2.zero_shot(briefs)
    if method in ('M2', 'M2noObj'):
        return prompts_v2.engineered(briefs, method == 'M2noObj')
    examples = {str(b['item_id']): prompts_v2.retrieve_examples(b, products, train_briefs) for b in briefs}
    return prompts_v2.fewshot(briefs, examples)


def run(split, methods, batch, limit=None, *, workers=1):
    if split in ('test', 'test2'):
        if not LOCK.exists() or json.loads(LOCK.read_text(encoding='utf-8')).get('template_hash') != prompts_v2.template_hash():
            raise RuntimeError('Chưa khóa prompt hoặc template_hash không khớp; dừng trước khi đọc test.')
    data = data_dir(ROOT, split)
    results = results_dir(ROOT, split)
    if split == 'test2':
        verify_test2(ROOT)
        if not set(methods) <= {'M1', 'M2'} or limit is not None or batch != 5:
            raise ValueError('test2 chỉ chạy M1/M2, đủ mẫu và theo lô 5 đã dùng ở test')
    if not (data / 'briefs.jsonl').exists() or not (data / 'products_clean.csv').exists():
        raise FileNotFoundError('Thiếu data_v2/processed/briefs.jsonl hoặc products_clean.csv')
    briefs = read_jsonl(data / 'briefs.jsonl')
    with (data / 'products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        products = list(csv.DictReader(f))
    selected = [b for b in briefs if b.get('split') == split][:limit]
    if not selected:
        raise ValueError('Không có brief cho split này')
    if 'OPENAI_API_KEY' in os.environ:
        raise RuntimeError('Tập kiểm tra này yêu cầu chạy bằng cùng cấu hình đã khóa. Hãy bỏ cấu hình dịch vụ đang hoạt động trước khi chạy.')
    for method in methods:
        path = results / 'gen' / split / (method + '.jsonl')
        done = {str(r['item_id']) for r in read_jsonl(path) if r.get('output') and not r.get('error')}
        pending = [b for b in selected if str(b['item_id']) not in done]
        batches = [pending[start:start + batch] for start in range(0, len(pending), batch)]

        def generate_batch(rows):
            prompt = '' if method == 'M0' else build_prompt(method, rows, products, briefs)
            t0 = time.perf_counter()
            raw = ''
            error = None
            found = {}
            if method == 'M0':
                found = {str(b['item_id']): prompts.template_output(model_brief(b)) for b in rows}
            else:
                for attempt in range(3):
                    try:
                        raw = llm_backends.generate(prompt, model='gpt-6.1-terra', effort='low', tag=f'v2:{split}:{method}')
                        parsed = llm_backends.extract_json(raw)
                        expected = {str(b['item_id']) for b in rows}
                        if not isinstance(parsed, list) or len(parsed) != len(rows) or any(not isinstance(x, dict) or set(x) != set(prompts.OUTPUT_KEYS) for x in parsed) or {str(x['item_id']) for x in parsed} != expected:
                            raise ValueError('Sai khóa hoặc danh sách item_id')
                        found = {str(x['item_id']): x for x in parsed}
                        error = None
                        break
                    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                        error = str(exc)
            elapsed = (time.perf_counter() - t0) / len(rows)
            records = []
            for b in rows:
                item_id = str(b['item_id'])
                record = {'item_id': item_id, 'method': method, 'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                          'template_hash': prompts_v2.template_hash(), 'model': 'template' if method == 'M0' else 'gpt-6.1-terra',
                          'latency_s': elapsed, 'input_tokens': token_count(prompt) / len(rows),
                          'output_tokens': token_count(raw) / len(rows), 'output': found.get(item_id), 'error': error}
                records.append(record)
            return records

        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = [executor.submit(generate_batch, rows) for rows in batches]
            for job in jobs:
                try:
                    records = job.result()
                except Exception as exc:
                    failures.append(exc)
                    continue
                for record in records:
                    append(path, record)
        if failures:
            raise RuntimeError(f"{len(failures)} lô hiện chưa hoàn tất") from failures[0]
        records = [r for r in read_jsonl(path) if r.get('output') and not r.get('error')]
        n = len(records)
        if split == 'test2' and {str(row['item_id']) for row in records} != {str(b['item_id']) for b in selected}:
            raise RuntimeError(f'Thiếu nội dung thành công của {method} trên test2: {n}/{len(selected)}')
        hits = sum(bool(prompt_leakage.leaked(r)) for r in records)
        print(f'{method}: rò rỉ {hits}/{n} ({hits/n:.1%})' if n else f'{method}: chưa có kết quả')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test', 'test2'])
    p.add_argument('--methods')
    p.add_argument('--batch', type=int, default=5)
    p.add_argument('--limit', type=int)
    p.add_argument('--workers', type=int, help='Số lô sinh đồng thời (test2 mặc định 2, các tập khác 1)')
    p.add_argument('--lock', action='store_true')
    a = p.parse_args()
    if a.workers is not None and a.workers < 1:
        p.error('workers phải dương')
    if a.lock:
        versions = sorted({r.get('template_hash') for f in (RESULTS / 'gen' / 'dev').glob('*.jsonl') for r in read_jsonl(f) if r.get('template_hash')})
        if not versions:
            p.error('Chưa có phiên bản dev đã thử; không thể khóa')
        if prompts_v2.template_hash() not in versions:
            p.error('Phiên bản prompt hiện tại chưa được chạy trên dev; không thể khóa')
        RESULTS.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps({'template_hash': prompts_v2.template_hash(), 'timestamp': datetime.now(timezone.utc).isoformat(), 'dev_versions': versions}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return
    methods = (a.methods or ('M2' if a.split == 'test2' else 'M0,M1,M2,M2noObj,M2ex')).split(',')
    if not a.split or a.batch < 1 or (a.limit is not None and a.limit < 1) or not set(methods) <= {'M0', 'M1', 'M2', 'M2noObj', 'M2ex'}:
        p.error('Split, method hoặc kích thước không hợp lệ')
    run(a.split, methods, a.batch, a.limit, workers=a.workers or (2 if a.split == 'test2' else 1))


if __name__ == '__main__':
    main()

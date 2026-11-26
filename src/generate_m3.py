"""Sinh nội dung M3 (vòng 2). Cùng mô hình, cùng lô 5 sản phẩm như M2.

Sau lượt viết đầu, chương trình dò câu sáo (prompts_m3.CLICHES); sản phẩm nào còn câu sáo được viết lại đúng một lần.
Tập kiểm tra chỉ chạy khi câu lệnh M3 và bảng phối đồ trùng mã băm đã khóa trên tập phát triển.

Chạy:
  python src/generate_m3.py --split dev            # chỉnh câu lệnh
  python src/generate_m3.py --lock                 # khóa sau khi chỉnh xong
  python src/generate_m3.py --split test           # chạy kiểm tra đúng một lần
"""
import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import llm_backends
import prompts
import prompts_m3
from generate_v2 import ROOT, RESULTS, append, model_brief, read_jsonl, token_count
from holdout_paths import data_dir, results_dir, verify_test2

LOCK = RESULTS / 'prompt_lock_m3.json'
FIELDS = ('title', 'meta_description', 'description', 'sales_script')


def guide_hash():
    return hashlib.sha256(prompts_m3.GUIDE_PATH.read_bytes()).hexdigest()


def call(prompt, rows, tag):
    error = None
    for _ in range(3):
        try:
            raw = llm_backends.generate(prompt, model='gpt-6.1-terra', effort='low', tag=tag)
            parsed = llm_backends.extract_json(raw)
            expected = {str(b['item_id']) for b in rows}
            if (not isinstance(parsed, list) or len(parsed) != len(rows)
                    or any(not isinstance(x, dict) or set(x) != set(prompts.OUTPUT_KEYS) for x in parsed)
                    or {str(x['item_id']) for x in parsed} != expected):
                raise ValueError('Sai khóa hoặc danh sách item_id')
            return raw, {str(x['item_id']): x for x in parsed}, None
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            error = str(exc)
    return '', {}, error


def hits_of(out):
    return prompts_m3.cliches(' '.join(str(out.get(f, '')) for f in FIELDS) + ' ' + ' '.join(out.get('email_subjects', [])))


def run(split, batch, limit, *, workers=1):
    if split in ('test', 'test2'):
        lock = json.loads(LOCK.read_text(encoding='utf-8')) if LOCK.exists() else {}
        if lock.get('template_hash') != prompts_m3.template_hash() or lock.get('guide_hash') != guide_hash():
            raise RuntimeError('Câu lệnh M3 hoặc bảng phối đồ chưa khóa / không khớp mã băm; dừng trước khi đọc test.')
    if 'OPENAI_API_KEY' in os.environ:
        raise RuntimeError('Tập kiểm tra này yêu cầu chạy bằng cùng cấu hình đã khóa. Hãy bỏ cấu hình dịch vụ đang hoạt động trước khi chạy.')
    data = data_dir(ROOT, split)
    if split == 'test2':
        verify_test2(ROOT)
        if limit is not None or batch != 5:
            raise ValueError('test2 cần chạy đủ mẫu theo lô 5 đã dùng ở test')
    briefs = read_jsonl(data / 'briefs.jsonl')
    with (data / 'products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        products = {str(p['product_id']): p for p in csv.DictReader(f)}
    guide = prompts_m3.load_guide()
    selected = [b for b in briefs if b.get('split') == split][:limit]
    path = results_dir(ROOT, split) / 'gen' / split / 'M3.jsonl'
    done = {str(r['item_id']) for r in read_jsonl(path) if r.get('output') and not r.get('error')}
    pending = [b for b in selected if str(b['item_id']) not in done]
    batches = [pending[start:start + batch] for start in range(0, len(pending), batch)]

    def generate_batch(brief_batch):
        rows = [model_brief(b) for b in brief_batch]
        prompt = prompts_m3.build(rows, products, guide)
        t0 = time.perf_counter()
        raw, found, error = call(prompt, rows, f'v2:{split}:M3')
        first = {k: hits_of(v) for k, v in found.items()}
        redo = [b for b in rows if first.get(str(b['item_id']))]
        raw2 = ''
        if redo:
            note = prompts_m3.rewrite_note([h for b in redo for h in first[str(b['item_id'])]])
            raw2, found2, err2 = call(prompts_m3.build(redo, products, guide) + note, redo, f'v2:{split}:M3:rewrite')
            if not err2:
                found.update(found2)
        elapsed = (time.perf_counter() - t0) / len(rows)
        n_in = token_count(prompt) + (token_count(prompts_m3.build(redo, products, guide)) if redo else 0)
        records = []
        for b in rows:
            i = str(b['item_id'])
            records.append({'item_id': i, 'method': 'M3', 'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                          'template_hash': prompts_m3.template_hash(), 'guide_hash': guide_hash(), 'model': 'gpt-6.1-terra',
                          'latency_s': elapsed, 'input_tokens': n_in / len(rows),
                          'output_tokens': (token_count(raw) + token_count(raw2)) / len(rows),
                          'cliches_first': first.get(i, []), 'rewritten': i in {str(x['item_id']) for x in redo},
                          'cliches_final': hits_of(found[i]) if i in found else None,
                          'output': found.get(i), 'error': error})
        return records

    completed = 0
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
            completed += len(records)
            print(f'{split}: {completed}/{len(pending)}', flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} lô hiện chưa hoàn tất") from failures[0]
    rec = [r for r in read_jsonl(path) if r.get('output')]
    if split == 'test2' and {str(row['item_id']) for row in rec if not row.get('error')} != {str(b['item_id']) for b in selected}:
        raise RuntimeError(f'Thiếu nội dung thành công của M3 trên test2: {len(rec)}/{len(selected)}')
    print('Câu sáo lượt đầu:', sum(bool(r['cliches_first']) for r in rec), '/', len(rec),
          '| còn sau viết lại:', sum(bool(r['cliches_final']) for r in rec))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test', 'test2'])
    p.add_argument('--batch', type=int, default=5)
    p.add_argument('--limit', type=int)
    p.add_argument('--workers', type=int, help='Số lô sinh đồng thời (test2 mặc định 2, các tập khác 1)')
    p.add_argument('--lock', action='store_true')
    a = p.parse_args()
    if a.workers is not None and a.workers < 1:
        p.error('workers phải dương')
    if a.lock:
        tried = {r.get('template_hash') for r in read_jsonl(RESULTS / 'gen' / 'dev' / 'M3.jsonl')}
        if prompts_m3.template_hash() not in tried:
            p.error('Câu lệnh M3 hiện tại chưa chạy trên dev; không thể khóa')
        LOCK.write_text(json.dumps({'template_hash': prompts_m3.template_hash(), 'guide_hash': guide_hash(),
                                    'timestamp': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
        return
    if not a.split:
        p.error('Cần --split')
    run(a.split, a.batch, a.limit, workers=a.workers or (2 if a.split == 'test2' else 1))


if __name__ == '__main__':
    main()

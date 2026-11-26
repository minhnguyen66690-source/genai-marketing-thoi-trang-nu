"""Pairwise blind evaluation with content-addressed resumable results."""
import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import llm_backends
from holdout_paths import data_dir, judge_file, results_dir, verify_test2

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('M0', 'M1', 'M2', 'M2noObj', 'M2ex', 'M3', 'M4a', 'M4b')
# M2ex là tên mới của cấu hình 'M3 có ví dụ' cũ; khóa và thứ tự A/B vẫn tính theo tên cũ để giữ các lượt chấm đã có.
KEY_NAME = {'M2ex': 'M3'}
RUBRICS = {
    'description': 'Bạn là người mua đang cân nhắc sản phẩm thời trang nữ này trên sàn. Trang nào giúp bạn quyết định mua tốt hơn? Chọn A, B hoặc tie cho overall, readable, informative, trustworthy. Lý do reason tối đa 25 từ.',
    'script': 'Bạn là người mua đang nhắn tin hỏi shop. Tin nhắn nào giải tỏa băn khoăn của bạn tốt hơn và khiến bạn muốn đặt hàng hơn? Chọn A, B hoặc tie cho overall, addresses_concern, trustworthy. Lý do reason tối đa 25 từ.',
}

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]

def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def manifest(root, script, model, split):
    data = data_dir(root, split)
    paths = [data/'briefs.jsonl', data/'split_ids.json', root/'results_v2/prompt_lock.json', Path(script)]
    if split == 'test2':
        paths.extend([root/'results_v2/prompt_lock_m3.json', root/'results_v2/style_guide.json'])
    append(results_dir(root, split)/'manifests.jsonl', {'timestamp': datetime.now(timezone.utc).isoformat(), 'script': Path(script).name, 'split': split, 'model': model, 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths}})

def inputs(root, split):
    data = data_dir(root, split)
    expected_ids = verify_test2(root) if split == 'test2' else None
    briefs = {str(x['item_id']): x for x in read_jsonl(data/'briefs.jsonl') if x.get('split') == split}
    with (data/'products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        products = {str(x['product_id']): x for x in csv.DictReader(f) if x.get('split') == split}
    if expected_ids is not None and (set(briefs) != expected_ids or set(products) != expected_ids):
        raise ValueError('Brief hoặc sản phẩm test2 thiếu mục đã khóa')
    gens = {}
    for method in METHODS:
        gens[method] = {str(x['item_id']): x for x in read_jsonl(results_dir(root, split)/'gen'/split/(method+'.jsonl'))
                        if str(x.get('item_id')) in briefs and
                        (split != 'test2' or (x.get('output') and not x.get('error')))}
    return briefs, products, gens

def output(row):
    return row.get('output') if isinstance(row.get('output'), dict) else {}

def s(value):
    return value if isinstance(value, str) else ''

def build(root, split, kind, model, only=None):
    briefs, products, gens = inputs(root, split)
    rubric_hash = digest(RUBRICS[kind])
    tasks = []
    for item, brief in briefs.items():
        for method in [m for m in METHODS if not only or m in only] if kind == 'description' else ('M2',):
            if item not in gens[method]:
                continue
            out = output(gens[method][item])
            if kind == 'description':
                if item not in products:
                    continue
                prod = products[item]
                left = 'Tên: '+s(out.get('title'))+'\nMô tả: '+s(out.get('description'))
                right = 'Tên: '+s(prod.get('name'))+'\nMô tả: '+s(prod.get('description_text'))
                other, context = 'SELLER', ''
            else:
                if item not in gens['M2noObj']:
                    continue
                left, right = s(out.get('sales_script')), s(output(gens['M2noObj'][item]).get('sales_script'))
                other, context = 'M2noObj', 'Khách đang băn khoăn: '+', '.join(map(str, brief.get('objections', [])))
            flip = random.Random(digest(item+'\0'+method)).randrange(2)
            for pass_no in (1, 2):
                swapped = bool(flip ^ (pass_no-1))
                sides = (right, left) if swapped else (left, right)
                names = (other, method) if swapped else (method, other)
                key_data = [model, rubric_hash, kind, item, KEY_NAME.get(method, method), pass_no, digest(left), digest(right), digest(context)]
                tasks.append({'key': digest(json.dumps(key_data, ensure_ascii=False)), 'item_id': item, 'method': method, 'other': other, 'pass': pass_no, 'names': names, 'A': sides[0], 'B': sides[1], 'context': context, 'rubric_hash': rubric_hash, 'content_hashes': [digest(left), digest(right)]})
    return tasks

def call_batch(batch, kind, judge, model, *, test2=False):
    request = [{'id': str(i), 'A': t['A'], 'B': t['B'], 'context': t['context']} for i,t in enumerate(batch)]
    prompt = RUBRICS[kind]+'\nTrả duy nhất JSON {"items":[{"id":"0", ...}]}, đủ mọi mục.\n'+json.dumps(request, ensure_ascii=False)
    raw = llm_backends.judge(prompt, model=model, tag='v2_pairwise', cli_print=test2) if judge == 'gemini' else llm_backends.generate(prompt, model=model, tag='v2_pairwise')
    payload = llm_backends.extract_json(raw)
    items = payload['items']
    if len(items) != len(batch) or {str(x['id']) for x in items} != {str(i) for i in range(len(batch))}:
        raise ValueError('Sai danh sách kết quả')
    criteria = ('overall','readable','informative','trustworthy') if kind == 'description' else ('overall','addresses_concern','trustworthy')
    result = []
    for x in items:
        t = batch[int(x['id'])]
        if any(x.get(k) not in ('A','B','tie') for k in criteria):
            raise ValueError('Kết quả giám khảo sai định dạng')
        # Lý do chỉ để đọc tham khảo: vượt 25 từ thì cắt, không loại cả lô (lô bị loại làm mất cả 6 cặp).
        x['reason'] = ' '.join(str(x.get('reason') or '').split()[:25])
        result.append({k:v for k,v in t.items() if k not in ('A','B','names','context')} | {'judge':judge, 'model':model, 'kind':kind, 'result': {k: t['names'][0] if x[k] == 'A' else t['names'][1] if x[k] == 'B' else 'tie' for k in criteria}, 'reason':x['reason']})
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--split', choices=['dev','test','test2'], required=True)
    p.add_argument('--judge', choices=['gemini','gpt'], required=True)
    p.add_argument('--kind', choices=['description','script'], required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--limit', type=int, help='Giới hạn số cặp item-method, vẫn chấm hai lượt')
    p.add_argument('--methods', help='Chỉ chấm các cấu hình này, phân tách bằng dấu phẩy')
    a=p.parse_args()
    if a.workers < 1 or (a.limit is not None and a.limit < 1): p.error('workers/limit phải dương')
    model = llm_backends.JUDGE_MODEL if a.judge == 'gemini' else 'gpt-5.6-luna'
    manifest(a.root, __file__, model, a.split)
    path=judge_file(a.root,a.split,a.judge,a.kind)
    if a.split == 'test2':
        if a.limit is not None or a.kind != 'description':
            p.error('test2 chỉ chấm mô tả trên toàn bộ mẫu')
        selected_methods = (a.methods or 'M2,M3').split(',')
        if not set(selected_methods) <= {'M1', 'M2', 'M3'}:
            p.error('test2 chỉ chấm M1, M2 và M3')
        briefs, _, gens = inputs(a.root, a.split)
        if any(set(gens[method]) != set(briefs) for method in selected_methods):
            p.error('Cần đủ 120 đầu ra thành công cho từng cấu hình test2')
        a.methods = ','.join(selected_methods)
    tasks=build(a.root,a.split,a.kind,model,a.methods.split(',') if a.methods else None)
    if a.limit: tasks=tasks[:2*a.limit]
    done={r['key'] for r in read_jsonl(path) if r.get('key')}
    tasks=[x for x in tasks if x['key'] not in done]
    # Hai lượt đảo vị trí của cùng một cặp không được nằm chung một lần gọi: xếp toàn bộ lượt 1 trước,
    # lượt 2 sau, mỗi nhóm xáo trộn cố định (seed 42), nên giám khảo không thấy cả hai thứ tự cùng lúc.
    rng=random.Random(42)
    first=[x for x in tasks if x['pass']==1]; second=[x for x in tasks if x['pass']==2]
    rng.shuffle(first); rng.shuffle(second)
    tasks=first+second
    groups = (first, second) if a.split == 'test2' else (tasks,)
    batches = [group[i:i+6] for group in groups for i in range(0, len(group), 6)]
    errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures=[ex.submit(call_batch,b,a.kind,a.judge,model,test2=a.split == 'test2') for b in batches]
        for future in as_completed(futures):
            try:
                for row in future.result(): append(path,row)
            except Exception as exc: errors.append(str(exc))
    print(json.dumps({'pending':len(tasks),'failed_batches':errors},ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__ == '__main__': main()

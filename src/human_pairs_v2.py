"""Chọn 40 cặp cho người chấm mù (E3) và xuất file nạp vào trang chấm.

- 20 cặp M2 với mô tả người bán, 20 cặp M2 với M1; 40 sản phẩm test khác nhau, chọn ngẫu nhiên seed 42.
- Vị trí A/B ngẫu nhiên theo seed. Bảng giải mã (bên nào là gì) chỉ ghi ở `_workspace/08_human_eval/key.json`,
  KHÔNG đưa lên trang chấm.
- Ô "Dữ kiện có thật" lấy từ bản tóm tắt (danh mục, giá, thuộc tính) để người chấm đánh dấu chỗ nói sai.
Đầu ra: `_workspace/08_human_eval/pairs/p01.json` … (mỗi file là một tài liệu cho collection `pairs`).
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_workspace/08_human_eval'


def read_jsonl(p):
    return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]


def facts(b):
    rows = [f"Loại sản phẩm: {b.get('category')}", f"Giá: {int(b['price']):,} đồng".replace(',', '.')]
    for k, v in (b.get('attributes') or {}).items():
        rows.append(f"{k.capitalize()}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return rows


def main():
    briefs = {str(b['item_id']): b for b in read_jsonl(ROOT / 'data_v2/processed/briefs.jsonl') if b['split'] == 'test'}
    with (ROOT / 'data_v2/processed/products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        seller = {r['product_id']: r for r in csv.DictReader(f) if r['split'] == 'test'}
    gen = {m: {str(r['item_id']): r['output'] for r in read_jsonl(ROOT / f'results_v2/gen/test/{m}.jsonl') if r.get('output') and not r.get('error')}
           for m in ('M1', 'M2')}
    ok = sorted(i for i in briefs if i in seller and i in gen['M1'] and i in gen['M2'])
    rng = random.Random(42)
    chosen = rng.sample(ok, 40)
    (OUT / 'pairs').mkdir(parents=True, exist_ok=True)
    key = {}
    for n, item in enumerate(chosen, start=1):
        other = 'SELLER' if n <= 20 else 'M1'
        m2 = (gen['M2'][item].get('title', ''), gen['M2'][item].get('description', ''))
        ot = (seller[item]['name'], seller[item]['description_text']) if other == 'SELLER' else \
             (gen['M1'][item].get('title', ''), gen['M1'][item].get('description', ''))
        sides = [('M2', m2), (other, ot)]
        rng.shuffle(sides)
        pid = f'p{n:02d}'
        doc = {'order': n, 'facts': facts(briefs[item]),
               'A_title': sides[0][1][0], 'A_desc': sides[0][1][1], 'B_title': sides[1][1][0], 'B_desc': sides[1][1][1]}
        (OUT / 'pairs' / f'{pid}.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
        key[pid] = {'item_id': item, 'A': sides[0][0], 'B': sides[1][0], 'comparison': f'M2 vs {other}'}
    # Thứ tự trình bày xáo trộn để hai loại so sánh xen kẽ (thứ tự trong `order`).
    order = list(key)
    rng.shuffle(order)
    for pos, pid in enumerate(order, start=1):
        p = OUT / 'pairs' / f'{pid}.json'
        d = json.loads(p.read_text(encoding='utf-8'))
        d['order'] = pos
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    print('đã xuất', len(key), 'cặp')


if __name__ == '__main__':
    main()

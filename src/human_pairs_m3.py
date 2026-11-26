"""Thêm 20 cặp chấm mù M3 với trang người bán (p41 đến p60) vào trang chấm.

- 20 sản phẩm kiểm tra chưa có trong 40 cặp vòng 1, chọn ngẫu nhiên seed 2026, để người chấm không gặp lại trang đã đọc.
- Vị trí A/B ngẫu nhiên; bảng giải mã ghi thêm vào `_workspace/08_human_eval/key.json`, không đưa lên trang chấm.
- Ô dữ kiện có thêm cụm đặc trưng của tên (lần trước thiếu nên một cặp bị đánh dấu nhầm).
"""
import csv
import json
import random
from pathlib import Path

from human_pairs_v2 import OUT, ROOT, facts, read_jsonl


def main():
    briefs = {str(b['item_id']): b for b in read_jsonl(ROOT / 'data_v2/processed/briefs.jsonl') if b['split'] == 'test'}
    with (ROOT / 'data_v2/processed/products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        seller = {r['product_id']: r for r in csv.DictReader(f) if r['split'] == 'test'}
    m3 = {str(r['item_id']): r['output'] for r in read_jsonl(ROOT / 'results_v2/gen/test/M3.jsonl') if r.get('output')}
    key = json.loads((OUT / 'key.json').read_text(encoding='utf-8'))
    used = {v['item_id'] for v in key.values()}
    ok = sorted(i for i in briefs if i in seller and i in m3 and i not in used)
    rng = random.Random(2026)
    chosen = rng.sample(ok, 20)
    order = list(range(41, 61))
    rng.shuffle(order)
    for n, item in enumerate(chosen, start=41):
        sides = [('M3', (m3[item].get('title', ''), m3[item].get('description', ''))),
                 ('SELLER', (seller[item]['name'], seller[item]['description_text']))]
        rng.shuffle(sides)
        pid = f'p{n:02d}'
        b = briefs[item]
        rows = facts(b) + ['Cụm đặc trưng trong tên: ' + ', '.join(b.get('keywords', [])),
                           'Đây là dữ kiện cơ bản; trang có thể nêu thêm chi tiết khác của người bán.']
        doc = {'order': order[n - 41], 'facts': rows,
               'A_title': sides[0][1][0], 'A_desc': sides[0][1][1], 'B_title': sides[1][1][0], 'B_desc': sides[1][1][1]}
        (OUT / 'pairs' / f'{pid}.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
        key[pid] = {'item_id': item, 'A': sides[0][0], 'B': sides[1][0], 'comparison': 'M3 vs SELLER'}
    (OUT / 'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    print('đã thêm', len(chosen), 'cặp; tổng', len(key))


if __name__ == '__main__':
    main()

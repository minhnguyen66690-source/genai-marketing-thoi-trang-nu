"""Đo ba chỉ số cho vòng 2: câu sáo, gợi ý phối đồ sai loại hàng, mức chép nguyên câu của người bán.

- Câu sáo: số cụm trong prompts_m3.CLICHES trên tên + mô tả (+ kịch bản với nội dung sinh).
- Phối đồ sai loại hàng: với đồ lót, đồ ngủ, đồ mặc nhà, đồ bơi, đếm đầu ra gợi ý mặc cùng món đồ mặc ra ngoài
  hoặc ra đường (chân váy, quần jean, giày, túi xách, đi làm, dạo phố...), trừ câu nói "mặc bên trong/dưới".
  Đây là bộ lọc thô; mọi trường hợp bị đánh dấu được in ra để rà tay.
- Chép nguyên: tỷ lệ cụm 8 từ liên tiếp của mô tả sinh có trong mô tả người bán.
Chạy: python src/eval_m3.py --split test
"""
import argparse
import csv
import json
import re
from pathlib import Path

import prompts_m3
from holdout_paths import data_dir, results_dir, verify_test2

ROOT = Path(__file__).resolve().parents[1]
INNER = re.compile(r'(nịt bụng|quần lót|áo ngực|đồ lót|lót bầu|đồ ngủ|mặc nhà|gia đình|đồ bơi|bikini|quần gen|nội y)', re.I)
# Lỗi người chấm phát hiện: gợi ý phối món mặc bên trong / đồ ngủ / đồ bơi với trang phục khác như món mặc ngoài.
OUTER = re.compile(r'(phối|kết hợp|mặc cùng|mix)[^.!?\n]{0,40}(váy|quần|áo|trang phục|giày|set đồ|outfit)', re.I)
INSIDE = re.compile(r'(mặc nhà|trong nhà|choàng|dép|kính bơi|mũ bơi|trong bộ|bộ gồm|mặc trong|lớp trong|mặc lót|bên trong|dưới|không lộ|không hằn|giặt|ma sát|vật nhọn)', re.I)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()] if p.exists() else []


def sents(t):
    return [x for x in re.split(r'(?<=[.!?\n])\s+', t or '') if x.strip()]


def bad_styling(category, text):
    if not INNER.search(category):
        return []
    return [s for s in sents(text) if OUTER.search(s) and not INSIDE.search(s)]


def ngrams(t, n=8):
    w = re.sub(r'[^\w\s]', ' ', (t or '').lower()).split()
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--split', choices=['dev','test','test2'], default='test'); a.add_argument('--show', action='store_true')
    a = a.parse_args()
    data = data_dir(ROOT, a.split)
    results = results_dir(ROOT, a.split)
    if a.split == 'test2':
        verify_test2(ROOT)
    briefs = {str(b['item_id']): b for b in read_jsonl(data / 'briefs.jsonl') if b['split'] == a.split}
    with open(data / 'products_clean.csv', encoding='utf-8-sig', newline='') as f:
        prod = {str(p['product_id']): p for p in csv.DictReader(f) if str(p['product_id']) in briefs}
    out = {}
    pages = {'SELLER': {i: {'title': p['name'], 'description': p['description_text']} for i, p in prod.items()}}
    for m in ('M1', 'M2', 'M3'):
        pages[m] = {str(r['item_id']): r['output'] for r in read_jsonl(results / f'gen/{a.split}/{m}.jsonl') if r.get('output')}
    if a.split == 'test2' and any(set(pages[m]) != set(briefs) for m in ('SELLER', 'M2', 'M3')):
        a.error('Chưa đủ dữ liệu để đo trên test2')
    for m, rows in pages.items():
        n = len(rows); cl = bad = 0; copy = []; flagged = []
        for i, o in rows.items():
            text = ' '.join([o.get('title', ''), o.get('description', '')] + ([o.get('sales_script', '')] if m != 'SELLER' else []))
            h = prompts_m3.cliches(text); cl += bool(h)
            b = bad_styling(briefs[i]['category'], text)
            if b:
                bad += 1; flagged.append({'item_id': i, 'category': briefs[i]['category'], 'sentences': b})
            if m != 'SELLER':
                g = ngrams(o.get('description')); s = ngrams(prod[i]['description_text'])
                copy.append(len(g & s) / len(g) if g else 0)
        inner_n = sum(1 for i in rows if INNER.search(briefs[i]['category']))
        out[m] = {'n': n, 'co_cau_sao': cl, 'ty_le_cau_sao': cl / n if n else None,
                  'do_lot_ngu_boi': inner_n, 'phoi_sai_loai': bad,
                  'chep_8tu_tb': sum(copy) / len(copy) if copy else None, 'flagged': flagged}
        print(m, {k: v for k, v in out[m].items() if k != 'flagged'})
        if a.show:
            for f in flagged:
                print('   ', f)
    (results / f'm3_metrics_{a.split}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

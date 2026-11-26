"""Dữ liệu tổng hợp cho website từ dữ liệu v2 (không xuất mô tả hay đánh giá nguyên văn).

- categories.json: danh mục cấp 2 -> 15 cụm đặc trưng phổ biến nhất trong bản tóm tắt của danh mục.
- objections_by_category.json: danh mục cấp 2 -> chủ đề phàn nàn trong đánh giá 1 đến 3 sao (phân loại bằng mô hình,
  results_v2/review_themes_llm.jsonl); danh mục dưới 10 đánh giá dùng thứ hạng chung.
- category_benchmarks.json: danh mục cấp 2 -> trung vị số từ mô tả, số gạch đầu dòng, độ dài tên của sản phẩm cùng loại.
"""
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'webapp' / 'data'
NAMES = {'size': 'Size không vừa', 'mismatch': 'Không giống hình/mô tả', 'material': 'Chất liệu kém',
         'color': 'Màu sắc khác ảnh', 'defect': 'Đường may, lỗi sản phẩm', 'wrong_item': 'Giao sai màu, size, mẫu',
         'delivery': 'Giao hàng, đóng gói', 'value': 'Không đáng tiền'}


def save(name, obj):
    (DATA / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    rows = list(csv.DictReader((ROOT / 'data_v2/processed/products_clean.csv').open(encoding='utf-8-sig')))
    cat = {r['product_id']: r['category_l2'] for r in rows}
    briefs = [json.loads(l) for l in (ROOT / 'data_v2/processed/briefs.jsonl').open(encoding='utf-8')]
    kw = defaultdict(Counter)
    for b in briefs:
        if not b.get('keyword_fallback'):
            kw[cat[str(b['item_id'])]].update(b['keywords'])
    save('categories.json', {c: [k for k, _ in v.most_common(15)] for c, v in sorted(kw.items())})

    reviews = list(csv.DictReader((ROOT / 'data_v2/processed/reviews_lowstar_clean.csv').open(encoding='utf-8-sig')))
    labels = [json.loads(l)['themes'] or [] for l in (ROOT / 'results_v2/review_themes_llm.jsonl').open(encoding='utf-8')]
    by_cat, overall = defaultdict(list), []
    for r, l in zip(reviews, labels):
        by_cat[cat.get(r['product_id'], '')].append(l)
        overall.append(l)

    def ranking(lists):
        c = Counter(t for l in lists for t in l)
        return [{'phrase': NAMES[k], 'count': n, 'share': round(n / len(lists), 3)} for k, n in c.most_common()]
    glob = ranking(overall)
    save('objections_by_category.json', {c: (ranking(by_cat[c]) if len(by_cat[c]) >= 10 else glob) for c in sorted(kw)})

    bench = {}
    for c in sorted(kw):
        sub = [r for r in rows if r['category_l2'] == c]
        words = [len(r['description_text'].split()) for r in sub]
        bullets = [len(re.findall(r'(?m)^\s*(?:[-*•✔✅]|\d+[.)])', r['description_text'])) for r in sub]
        bench[c] = {'all': {'desc_len_words': statistics.median(words), 'n_bullets': statistics.median(bullets),
                            'title_len_chars': statistics.median(len(r['name']) for r in sub), 'n': len(sub)}}
    save('category_benchmarks.json', bench)
    print('ok', len(kw), 'danh mục')


if __name__ == '__main__':
    main()

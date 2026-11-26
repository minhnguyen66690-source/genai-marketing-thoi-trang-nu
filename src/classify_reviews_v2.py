"""Phân loại lại 403 đánh giá 1 đến 3 sao bằng mô hình (Gemini) cho phần mô tả RQ1.

Lý do: từ điển `objections_v2.THEMES` có độ phủ thấp (60,5% đánh giá thấp không khớp chủ đề nào, dù đọc mẫu thấy
nhiều lời chê thật như "vải hơi mỏng", "giao sai màu"). Bản tóm tắt của thực nghiệm GIỮ thứ hạng từ điển (đã chốt
trước khi chạy test); kết quả phân loại này chỉ dùng cho thống kê mô tả và để báo cáo mức trùng của hai cách xếp hạng.
Đầu ra: results_v2/review_themes_llm.jsonl, results_v2/review_themes_llm_summary.json
"""
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_backends

ROOT = Path(__file__).resolve().parents[1]
THEMES = {
    'size': 'Size, form không vừa (chật, rộng, ngắn, dài so với người mặc)',
    'mismatch': 'Không giống hình hoặc mô tả (kiểu dáng, chất liệu khác lời giới thiệu)',
    'material': 'Chất liệu kém (mỏng, thô, nóng, không thấm hút, xù, phai)',
    'color': 'Màu sắc khác ảnh (không phải giao nhầm màu)',
    'defect': 'Đường may, lỗi sản phẩm (chỉ thừa, rách, bung, bẩn)',
    'wrong_item': 'Giao sai hàng (sai màu, sai size, sai mẫu, thiếu hàng so với đơn)',
    'delivery': 'Giao hàng, đóng gói, thái độ người giao',
    'value': 'Không đáng tiền, chất lượng chung kém không nêu cụ thể',
}
PROMPT = ('Phân loại từng đánh giá 1 đến 3 sao về thời trang nữ vào các chủ đề phàn nàn sau (một đánh giá có thể thuộc '
          'nhiều chủ đề, hoặc không thuộc chủ đề nào nếu không phàn nàn cụ thể):\n'
          + '\n'.join(f'- {k}: {v}' for k, v in THEMES.items())
          + '\nTrả duy nhất JSON {"items":[{"id":"...","themes":["size",...]}]} đủ mọi id.\n')


def classify(batch):
    req = [{'id': r['id'], 'text': r['content'][:600]} for r in batch]
    raw = llm_backends.judge(PROMPT + json.dumps(req, ensure_ascii=False), tag='v2_review_themes')
    items = llm_backends.extract_json(raw)['items']
    got = {str(x['id']): [t for t in x.get('themes', []) if t in THEMES] for x in items}
    if set(got) != {r['id'] for r in batch}:
        raise ValueError('thiếu id')
    return got


def main():
    with (ROOT / 'data_v2/processed/reviews_lowstar_clean.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r['id'] = str(i)
    batches = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    labels = {}
    with ThreadPoolExecutor(3) as ex:
        for got in ex.map(lambda b: next((x for x in (try_classify(b) for _ in range(3)) if x), None), batches):
            labels.update(got or {})
    out = ROOT / 'results_v2/review_themes_llm.jsonl'
    with out.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps({'id': r['id'], 'product_id': r['product_id'], 'rating': r['rating'],
                                'themes': labels.get(r['id'])}, ensure_ascii=False) + '\n')
    done = [r for r in rows if labels.get(r['id']) is not None]
    c = Counter(t for r in done for t in labels[r['id']])
    summary = {'n_reviews': len(rows), 'n_classified': len(done),
               'any_theme_rate': sum(bool(labels[r['id']]) for r in done) / len(done),
               'themes': {k: {'label': THEMES[k], 'count': c[k], 'rate': c[k] / len(done)} for k in THEMES}}
    (ROOT / 'results_v2/review_themes_llm_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def try_classify(batch):
    try:
        return classify(batch)
    except Exception as exc:  # thử lại tối đa 3 lần ở main
        print('lỗi lô:', exc)
        return None


if __name__ == '__main__':
    main()

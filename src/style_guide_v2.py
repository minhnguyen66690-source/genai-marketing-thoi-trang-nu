"""Lập bảng hướng dẫn phối đồ theo loại hàng, học từ mô tả người bán trong TẬP HUẤN LUYỆN.

Bước 1 (cơ học): lấy các câu có nói về dịp mặc hoặc phối đồ trong mô tả người bán của tập huấn luyện,
tối đa 2 câu mỗi cửa hàng, tối đa 30 câu mỗi loại hàng.
Bước 2 (mô hình): GPT-6.1-Terra tóm các câu đó thành hướng dẫn: dịp mặc, phối với gì, không gợi ý gì.
Đầu ra: results_v2/style_guide.json (tác giả rà lại trước khi dùng cho M3).

Chạy: python src/style_guide_v2.py
"""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import llm_backends

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data_v2' / 'processed'
OUT = ROOT / 'results_v2' / 'style_guide.json'
SENT_OUT = ROOT / 'results_v2' / 'style_guide_sentences.json'

CUE = re.compile(r'(phối|kết hợp|mix|diện|đi làm|đi chơi|dạo phố|đi biển|du lịch|dự tiệc|công sở|đi học|hẹn hò|ở nhà|'
                 r'mặc nhà|đi ngủ|dịp|sự kiện|cà phê|tiệc|văn phòng|thể thao|tập gym|yoga)', re.I)
SPLIT = re.compile(r'(?<=[.!?;•\n])\s+|\s+-\s+')


def sentences(text):
    for s in SPLIT.split(text or ''):
        s = ' '.join(s.split()).strip(' -•*+')
        if 25 <= len(s) <= 260 and CUE.search(s):
            yield s


def collect():
    with (DATA / 'products_clean.csv').open(encoding='utf-8-sig', newline='') as f:
        products = [p for p in csv.DictReader(f) if p['split'] == 'train']
    briefs = {str(json.loads(l)['item_id']): json.loads(l) for l in open(DATA / 'briefs.jsonl', encoding='utf-8')}
    by_cat = defaultdict(list)
    rng = random.Random(2026)
    rng.shuffle(products)
    per_shop = defaultdict(int)
    for p in products:
        cat = briefs.get(str(p['product_id']), {}).get('category')
        if not cat:
            continue
        for s in list(sentences(p['description_text']))[:2]:
            key = (cat, p['seller_id'])
            if per_shop[key] >= 2 or len(by_cat[cat]) >= 30 or s in by_cat[cat]:
                continue
            per_shop[key] += 1
            by_cat[cat].append(s)
    all_cats = sorted({b['category'] for b in briefs.values()})
    return {c: by_cat.get(c, []) for c in all_cats}


PROMPT = """Bạn là người bán thời trang nữ nhiều năm kinh nghiệm ở Việt Nam. Dưới đây là các loại hàng, mỗi loại kèm những câu
người bán thật đã viết về dịp mặc và cách phối (có thể có câu sai hoặc quảng cáo, hãy chọn lọc bằng hiểu biết thực tế).
Với MỖI loại hàng, viết hướng dẫn ngắn, cụ thể, đúng thực tế mặc đồ của phụ nữ Việt Nam:
- "dip": 2-4 dịp/hoàn cảnh mặc thật sự hợp (ví dụ: "đi làm văn phòng", "đi biển", "mặc ở nhà, đi ngủ").
- "phoi_voi": 3-5 gợi ý phối CỤ THỂ gồm món đồ + màu/kiểu nếu hợp (ví dụ: "quần jean ống suông xanh nhạt", "giày sandal quai mảnh").
  Đồ mặc bên trong (quần lót, áo ngực, quần gen, đồ lót) thì gợi ý theo trang phục bên ngoài phù hợp (ví dụ "mặc dưới váy body không lộ viền"), KHÔNG gợi ý như món mặc ra ngoài.
  Đồ ngủ, đồ mặc nhà, đồ bơi: chỉ nói hoàn cảnh dùng đúng (ở nhà, đi ngủ, hồ bơi, đi biển kèm áo choàng, khăn).
- "khong_goi_y": 2-4 điều KHÔNG được gợi ý vì sai thực tế (ví dụ: "phối quần lót với chân váy như trang phục ngoài").
- "luu_y_size": 1 câu cách chọn size thực tế của loại này.
Không dùng câu sáo chung chung như "dễ phối với nhiều trang phục", "bổ sung vào tủ đồ", "tạo sự thoải mái", "phong cách mong muốn".
Chỉ trả về một đối tượng JSON: khóa là tên loại hàng y nguyên, giá trị là đối tượng có 4 khóa trên.

DỮ LIỆU:
"""


def main():
    data = collect()
    SENT_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    guide = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    todo = [c for c in data if c not in guide]
    for i in range(0, len(todo), 8):
        chunk = {c: data[c][:30] for c in todo[i:i + 8]}
        for attempt in range(3):
            try:
                raw = llm_backends.generate(PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                            model='gpt-6.1-terra', effort='medium', tag='v2:style_guide')
                parsed = llm_backends.extract_json(raw)
                if not isinstance(parsed, dict) or set(parsed) != set(chunk):
                    raise ValueError('Sai danh sách loại hàng')
                for v in parsed.values():
                    if set(v) != {'dip', 'phoi_voi', 'khong_goi_y', 'luu_y_size'}:
                        raise ValueError('Sai khóa')
                guide.update(parsed)
                OUT.write_text(json.dumps(guide, ensure_ascii=False, indent=1), encoding='utf-8')
                print('xong', len(guide), '/', len(data))
                break
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                print('lỗi, thử lại:', exc)


if __name__ == '__main__':
    main()

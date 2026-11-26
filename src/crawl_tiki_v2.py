"""Thu thập lại dữ liệu Tiki theo khung lấy mẫu chốt trước (bản v2, 22/11/2026).

Khác bản v1:
- Khung lấy mẫu: với MỖI danh mục lá dưới Thời trang nữ (931), lấy danh sách theo ba cách sắp xếp
  (bán chạy, mới nhất, mặc định), mỗi cách tối đa PER_SORT sản phẩm, gộp và bỏ trùng. Mọi sản phẩm
  trong khung được ghi lại kèm nguồn và thứ hạng (tiki_frame.jsonl) để mô tả được thiên lệch chọn mẫu.
- Đánh giá: lấy có chủ đích đánh giá 1, 2, 3 sao (lọc theo từng mức sao của API) cho mọi sản phẩm có
  đánh giá, thay vì chỉ trang đầu của nhóm bán chạy.
- Dừng hẳn khi máy chủ trả HTML thay JSON, 403 hoặc CAPTCHA; không vượt chặn. Chạy lại sẽ tiếp tục.
Đầu ra: data_v2/raw/*.jsonl, data_v2/raw/tiki_manifest.json (có SHA-256 từng tệp).
"""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import requests

from crawl_tiki import BASE, UA, clean, leaves, read_jsonl, append, review_row

SORTS = ['top_seller', 'newest', 'default']


class Blocked(RuntimeError):
    pass


class Client:
    def __init__(self, log):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': UA, 'Accept': 'application/json'})
        self.count, self.last, self.log = 0, 0.0, log

    def note(self, text):
        with self.log.open('a', encoding='utf-8') as f:
            f.write(f'{datetime.now().astimezone().isoformat()} {text}\n')

    def get(self, path, params=None):
        for attempt in range(4):
            delay = random.uniform(1.2, 2.0) - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()
            self.count += 1
            try:
                r = self.session.get(BASE + path, params=params, timeout=30, allow_redirects=False)
            except requests.RequestException as exc:
                self.note(f'lỗi mạng {type(exc).__name__}')
                time.sleep(2 ** attempt)
                continue
            self.note(f'{r.status_code} {r.url}')
            head = r.text[:2000].lstrip().lower()
            content_type = r.headers.get('content-type', '').lower()
            if (r.status_code == 403 or 'captcha' in head or 'text/html' in content_type
                    or head.startswith(('<!doctype html', '<html'))):
                raise Blocked(f'Tiki chặn hoặc trả HTML ở {path} (HTTP {r.status_code}); dừng, không vượt chặn')
            if 300 <= r.status_code < 400:
                raise Blocked(f'Tiki chuyển hướng ở {path} (HTTP {r.status_code}); dừng')
            if r.status_code == 429:
                time.sleep(60)
                continue
            if r.status_code == 404:
                return {}
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            try:
                return r.json()
            except ValueError:
                time.sleep(2 ** attempt)
        self.note(f'bỏ qua {path} sau 4 lần lỗi')
        return {}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-sort', type=int, default=40, help='tối đa số sản phẩm mỗi cách sắp xếp mỗi danh mục lá')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out-dir', default='data_v2')
    ap.add_argument('--holdout-from', type=Path, help='Thư mục gốc dữ liệu cũ; dùng lại danh mục và loại ID cũ')
    ap.add_argument('--start-page', type=int, default=1, help='Trang đầu tiên của mỗi cách sắp xếp')
    args = ap.parse_args()
    if args.start_page < 1 or args.per_sort < 1 or (args.start_page != 1 and not args.holdout_from):
        ap.error('Trang bắt đầu và số sản phẩm phải hợp lệ; trang khác 1 chỉ dùng cho holdout')
    if args.holdout_from:
        if args.smoke or Path(args.out_dir).resolve() == args.holdout_from.resolve():
            ap.error('Holdout cần thư mục đầu ra riêng và chạy đủ danh mục')
        old_frame = read_jsonl(args.holdout_from / 'raw/tiki_frame.jsonl')
        old_products = read_jsonl(args.holdout_from / 'raw/tiki_products_raw.jsonl')
        fixed_categories = list(dict.fromkeys(x['leaf_category_id'] for x in old_frame))
        if len(fixed_categories) != 88 or not old_products:
            ap.error('Thiếu khung 88 danh mục lá hoặc dữ liệu sản phẩm cũ')
        excluded = {str(x['product_id']) for x in old_frame} | {str(x['id']) for x in old_products}
        sorts = ['newest', 'top_seller', 'default']
    else:
        excluded = set()
        sorts = SORTS
    raw = Path(args.out_dir) / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    previous = raw / 'tiki_manifest.json'
    if args.holdout_from and previous.exists():
        ap.error('Tập thu thập holdout đã có biên bản kết thúc. Giữ nguyên dữ liệu.')
    if args.holdout_from and (raw / 'crawl.log').exists():
        ap.error('Holdout đã có nhật ký. Cần kiểm tra trạng thái dừng trước mọi lần chạy khác.')
    client = Client(raw / 'crawl.log')
    frame_path, prod_path = raw / 'tiki_frame.jsonl', raw / 'tiki_products_raw.jsonl'
    rev_path, rev_state = raw / 'tiki_reviews_lowstar_raw.jsonl', raw / 'tiki_review_state.jsonl'
    status = 'hoàn tất'
    try:
        # 1. Khung lấy mẫu
        frame = read_jsonl(frame_path)
        framed_cats = {str(x['leaf_category_id']) for x in frame}
        cats = fixed_categories if args.holdout_from else leaves(client, args.smoke)
        if args.smoke:
            cats = cats[:1]
        per_sort = 3 if args.smoke else args.per_sort
        for cat in cats:
            if str(cat) in framed_cats:
                continue
            seen = set()
            for sort in sorts:
                got, page = 0, args.start_page
                while got < per_sort:
                    p = client.get('/api/personalish/v1/blocks/listings',
                                   {'limit': 40, 'category': cat, 'page': page, 'sort': sort})
                    entries = [x for x in (p.get('data') or []) if isinstance(x, dict) and x.get('id')]
                    if not entries:
                        break
                    for rank, x in enumerate(entries, start=(page - 1) * 40 + 1):
                        if got >= per_sort:
                            break
                        got += 1
                        if str(x['id']) in excluded:
                            continue
                        append(frame_path, {'leaf_category_id': cat, 'sort': sort, 'rank': rank, 'product_id': x['id'],
                                            'already_in_frame': x['id'] in seen,
                                            'quantity_sold': (x.get('quantity_sold') or {}).get('value'),
                                            'review_count': x.get('review_count')})
                        seen.add(x['id'])
                    last = int((p.get('paging') or {}).get('last_page') or 1)
                    if page >= last:
                        break
                    page += 1
        frame = read_jsonl(frame_path)
        # 2. Chi tiết sản phẩm cho mọi id duy nhất trong khung
        done = {str(x.get('id')) for x in read_jsonl(prod_path)}
        first_cat = {}
        for x in frame:
            first_cat.setdefault(str(x['product_id']), x['leaf_category_id'])
        for pid, cat in first_cat.items():
            if pid in done:
                continue
            item = client.get(f'/api/v2/products/{pid}')
            if not item:
                continue
            item = clean(item)
            item['_leaf_category_id'] = cat
            item['_crawled_at'] = datetime.now().astimezone().isoformat()
            append(prod_path, item)
            done.add(pid)
        # 3. Đánh giá 1, 2, 3 sao cho sản phẩm có đánh giá
        state = {(str(x['product_id']), x['stars']) for x in read_jsonl(rev_state)}
        products = [] if args.holdout_from else read_jsonl(prod_path)
        for item in products:
            pid = str(item.get('id'))
            if not item.get('review_count'):
                continue
            counts = None
            for star in (1, 2, 3):
                if (pid, star) in state:
                    continue
                if counts is not None and not counts.get(str(star)):
                    append(rev_state, {'product_id': pid, 'stars': star, 'n': 0})
                    continue
                p = client.get('/api/v2/reviews', {'product_id': pid, 'page': 1, 'limit': 20,
                                                   'sort': f'score|desc,id|desc,stars|{star}'})
                counts = {k: (v or {}).get('count', 0) for k, v in (p.get('stars') or {}).items()}
                n = 0
                for r in p.get('data') or []:
                    if r.get('rating') != star:
                        continue
                    rec = review_row(pid, r)
                    if rec:
                        append(rev_path, rec)
                        n += 1
                append(rev_state, {'product_id': pid, 'stars': star, 'n': n})
                if args.smoke:
                    break
    except Blocked as exc:
        status = str(exc)
        client.note(status)
    manifest = {'finished_at': datetime.now().astimezone().isoformat(), 'status': status,
                'sampling': {'root_category': 931, 'sorts': sorts, 'per_sort_per_leaf': args.per_sort,
                             'start_page': args.start_page, 'old_ids_excluded': len(excluded), 'leaf_categories': len(cats) if 'cats' in locals() else None},
                'requests_this_run': client.count,
                'files': {p.name: {'records': len(read_jsonl(p)), 'sha256': sha256(p)}
                          for p in (frame_path, prod_path, rev_path) if p.exists()}}
    (raw / 'tiki_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))
    if status != 'hoàn tất':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

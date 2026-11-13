"""Thu thập dữ liệu công khai Tiki, có checkpoint và giới hạn tốc độ."""
import argparse
import json
import random
import re
import time
from datetime import datetime, timedelta, date, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = 'https://tiki.vn'
CUTOFF = date(2019, 11, 20)
UA = 'Mozilla/5.0 (academic research; UEH student project)'
PRODUCT_COLUMNS = ['product_id', 'name', 'brand', 'category_l1', 'category_l2', 'category_l3', 'leaf_category_id', 'price', 'list_price', 'discount_rate', 'rating_average', 'review_count', 'quantity_sold', 'created_date', 'day_ago_created', 'within_7y', 'short_description', 'description_text', 'description_len_chars', 'n_images', 'url', 'crawled_at']
REVIEW_COLUMNS = ['product_id', 'review_id', 'rating', 'title', 'content', 'created_at', 'thank_count']
# Loại bỏ đệ quy các trường nhận diện cá nhân khỏi JSON sản phẩm.
PRIVATE = re.compile(r'(^|_)(customer|user|reviewer|author|phone|email|avatar|address|full_name)(_|$)', re.I)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not PRIVATE.search(k)}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def read_jsonl(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def append(path, obj):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


class Client:
    def __init__(self, log):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': UA, 'Accept': 'application/json'})
        self.count = 0
        self.last = 0.0
        self.log = log

    def get(self, path, params=None):
        for attempt in range(4):
            # Khoảng cách giữa mọi request ít nhất một giây, kể cả retry.
            delay = random.uniform(1.0, 1.6) - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            try:
                self.last = time.monotonic()
                self.count += 1
                response = self.session.get(BASE + path, params=params, timeout=30)
                with self.log.open('a', encoding='utf-8') as f:
                    f.write(f'{datetime.now().astimezone().isoformat()} {response.status_code} {response.url}\n')
                if response.status_code == 429:
                    time.sleep(60)
                    continue
                # Không giải hoặc vượt CAPTCHA; dừng khi bị từ chối.
                if response.status_code == 403 or 'captcha' in response.text[:2000].lower():
                    raise RuntimeError('Tiki từ chối hoặc yêu cầu CAPTCHA; dừng thu thập')
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                with self.log.open('a', encoding='utf-8') as f:
                    f.write(f'{datetime.now().astimezone().isoformat()} lỗi {type(exc).__name__}\n')
                if attempt == 3:
                    # Phản hồi rỗng/không phải JSON lặp lại (lỗi tạm thời phía máy chủ): bỏ qua trang này
                    # thay vì dừng cả chương trình. Lỗi 403/CAPTCHA vẫn dừng hẳn ở nhánh phía trên.
                    if isinstance(exc, ValueError):
                        with self.log.open('a', encoding='utf-8') as f:
                            f.write(f'{datetime.now().astimezone().isoformat()} bỏ qua {path} sau 4 lần lỗi JSON\n')
                        return {}
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError('Hết số lần thử sau HTTP 429')


def children(client, parent):
    data = client.get('/api/v2/categories', {'parent_id': parent})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('data', 'children', 'categories'):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError('Không nhận diện được cấu trúc danh mục')


def leaves(client, smoke=False):
    queue, result, seen = [931], [], set()
    while queue:
        parent = queue.pop(0)
        if parent in seen:
            continue
        seen.add(parent)
        nodes = children(client, parent)
        ids = [x.get('id') for x in nodes if isinstance(x, dict) and x.get('id') and x.get('id') != parent]
        if ids:
            queue.extend(ids[:1] if smoke else ids)
        elif parent != 931:
            result.append(parent)
    return result


def product_row(item):
    from bs4 import BeautifulSoup
    crumbs = item.get('breadcrumbs') or []
    if isinstance(crumbs, dict):
        crumbs = crumbs.get('data') or []
    names = [x.get('name') for x in crumbs if isinstance(x, dict) and x.get('name')]
    if 'Thời trang nữ' in names:
        names = names[names.index('Thời trang nữ'):]
    names = (names + [None] * 3)[:3]
    days = item.get('day_ago_created')
    stamp = item.get('_crawled_at') or datetime.now().astimezone().isoformat()
    try:
        created = (datetime.fromisoformat(stamp).date() - timedelta(days=int(days))).isoformat()
        within = date.fromisoformat(created) >= CUTOFF
    except (ValueError, TypeError, OverflowError):
        created, within = None, None
    soup = BeautifulSoup(item.get('description') or '', 'html.parser')
    description = '\n'.join(x.strip() for x in soup.get_text('\n').splitlines() if x.strip())
    sold = item.get('quantity_sold') or {}
    sold = sold.get('value') if isinstance(sold, dict) else sold
    if sold is None:
        sold = item.get('all_time_quantity_sold')
    brand = item.get('brand') or {}
    return dict(zip(PRODUCT_COLUMNS, [item.get('id'), item.get('name'), brand.get('name') if isinstance(brand, dict) else brand, *names, item.get('_leaf_category_id'), item.get('price'), item.get('list_price'), item.get('discount_rate'), item.get('rating_average'), item.get('review_count'), sold, created, days, within, item.get('short_description'), description, len(description), len(item.get('images') or []), BASE + '/' + (item.get('url_path') or '').lstrip('/'), stamp]))


def review_row(product_id, item):
    value = item.get('created_at')
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            dt = datetime.fromtimestamp(float(value), timezone.utc)
        else:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if dt.date() < CUTOFF:
            return None
    except (ValueError, TypeError, OverflowError, OSError):
        return None
    # Danh sách trắng: không ghi bất kỳ thông tin tài khoản hoặc ảnh nào.
    def redact(value):
        return re.sub(r'(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)', '[ĐÃ ẨN SỐ ĐIỆN THOẠI]', value) if isinstance(value, str) else value
    return {'product_id': product_id, 'review_id': item.get('id'), 'rating': item.get('rating'), 'title': redact(item.get('title')), 'content': redact(item.get('content')), 'created_at': dt.isoformat(), 'thank_count': item.get('thank_count')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, default=3000)
    parser.add_argument('--review-top', type=int, default=800)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    if args.target < 1 or args.review_top < 0:
        parser.error('target phải dương và review-top không âm')
    root = Path(args.out_dir)
    raw, processed, logs = root / 'data/raw', root / 'data/processed', root / 'logs'
    for folder in (raw, processed, logs):
        folder.mkdir(parents=True, exist_ok=True)
    client = Client(logs / 'crawl_tiki.log')
    products_path, reviews_path = raw / 'tiki_products_raw.jsonl', raw / 'tiki_reviews_raw.jsonl'
    state_path = raw / 'tiki_review_checkpoint.jsonl'
    products, reviews = read_jsonl(products_path), read_jsonl(reviews_path)
    done = {str(x.get('id')) for x in products}
    seen_reviews = {(str(x.get('product_id')), str(x.get('review_id'))) for x in reviews}
    done_pages = {(str(x['product_id']), x['page']) for x in read_jsonl(state_path)}
    target = min(args.target, 5) if args.smoke else args.target
    categories = leaves(client, args.smoke)
    if not categories:
        raise RuntimeError('Không tìm thấy danh mục lá dưới 931')
    if args.smoke:
        categories = categories[:1]
    # Duyệt luân phiên danh mục để phân bổ sản phẩm tương đối đồng đều.
    queues = {str(c): [] for c in categories}
    states = {str(c): [0, 1, False] for c in categories}
    sorts = ['top_seller', 'default']
    while len(done) < target and any(not s[2] or queues[c] for c, s in states.items()):
        for category in categories:
            key = str(category)
            state = states[key]
            while not queues[key] and not state[2]:
                sort_index, page, _ = state
                payload = client.get('/api/personalish/v1/blocks/listings', {'limit': 40, 'category': category, 'page': page, 'sort': sorts[sort_index]})
                entries = payload.get('data') or []
                queues[key].extend(x['id'] for x in entries if isinstance(x, dict) and x.get('id'))
                last = min(int((payload.get('paging') or {}).get('last_page') or 1), 50)
                if page >= last or not entries:
                    state[0] += 1
                    state[1] = 1
                    state[2] = state[0] >= len(sorts)
                else:
                    state[1] += 1
            while queues[key] and len(done) < target:
                pid = queues[key].pop(0)
                if str(pid) in done:
                    continue
                item = clean(client.get(f'/api/v2/products/{pid}'))
                item['_leaf_category_id'] = category
                item['_crawled_at'] = datetime.now().astimezone().isoformat()
                append(products_path, item)
                products.append(item)
                done.add(str(pid))
                break
            if len(done) >= target:
                break
    rows = [product_row(p) for p in products]
    pd.DataFrame(rows, columns=PRODUCT_COLUMNS).to_csv(processed / 'tiki_products.csv', index=False, encoding='utf-8-sig')
    def sales(row):
        try:
            return float(row['quantity_sold'] or 0)
        except (ValueError, TypeError):
            return 0
    for row in sorted(rows, key=sales, reverse=True)[:args.review_top]:
        pid = row['product_id']
        for page in range(1, 2 if args.smoke else 3):
            if (str(pid), page) in done_pages:
                continue
            payload = client.get('/api/v2/reviews', {'product_id': pid, 'page': page, 'limit': 20, 'include': 'comments', 'sort': 'score|desc,id|desc,stars|all'})
            for item in payload.get('data') or []:
                record = review_row(pid, item)
                if record and record['review_id'] is not None and (str(pid), str(record['review_id'])) not in seen_reviews:
                    append(reviews_path, record)
                    reviews.append(record)
                    seen_reviews.add((str(pid), str(record['review_id'])))
            append(state_path, {'product_id': pid, 'page': page})
    pd.DataFrame(reviews, columns=REVIEW_COLUMNS).to_csv(processed / 'tiki_reviews.csv', index=False, encoding='utf-8-sig')
    manifest = {'crawled_at': datetime.now().astimezone().isoformat(), 'endpoints': [BASE + '/api/v2/categories', BASE + '/api/personalish/v1/blocks/listings', BASE + '/api/v2/products/<id>', BASE + '/api/v2/reviews'], 'requests_this_run': client.count, 'product_records': len(products), 'review_records': len(reviews), 'leaf_categories': len(categories), 'parameters': vars(args)}
    (raw / 'tiki_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()

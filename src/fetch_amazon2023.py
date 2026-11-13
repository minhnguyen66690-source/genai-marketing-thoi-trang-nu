"""Lọc tuần tự metadata Amazon Reviews 2023, không tải cả tập tin về đĩa."""
import argparse
import gzip
import json
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

DEFAULT_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz'
FALLBACK_URL = 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl'
FIELDS = ['parent_asin', 'title', 'features', 'description', 'details', 'categories', 'main_category', 'store', 'price', 'average_rating', 'rating_number']
CUTOFF = date(2019, 11, 20)


def text_value(value):
    return '\n'.join(str(x) for x in value if x) if isinstance(value, list) else str(value or '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--max-lines', type=int, default=2000000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    if args.n < 1 or args.max_lines < 1:
        parser.error('--n và --max-lines phải dương')
    root = Path(args.out_dir)
    raw, processed = root / 'data/raw', root / 'data/processed'
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    url = args.url
    response = requests.get(url, stream=True, timeout=(20, 90))
    if response.status_code == 404:
        response.close()
        url = FALLBACK_URL
        response = requests.get(url, stream=True, timeout=(20, 90))
    response.raise_for_status()
    # Đọc trực tiếp từng dòng từ luồng HTTP; gzip chỉ giải nén khi cần.
    response.raw.decode_content = False
    source = gzip.GzipFile(fileobj=response.raw) if url.split('?')[0].endswith('.gz') else response.raw
    kept, lines, rows = 0, 0, []
    limit = min(args.n, 20) if args.smoke else args.n
    try:
        with (raw / 'amazon2023_women_meta.jsonl').open('w', encoding='utf-8') as output:
            for line in source:
                lines += 1
                if lines > args.max_lines:
                    lines -= 1
                    break
                try:
                    item = json.loads(line)
                    categories = item.get('categories') or []
                    if isinstance(categories, str):
                        categories = [categories]
                    if 'Women' not in categories or not ('Clothing' in categories or item.get('main_category') == 'AMAZON FASHION'):
                        continue
                    if not item.get('title') or not (item.get('description') or item.get('features')) or float(item.get('rating_number') or 0) < 5:
                        continue
                    details = item.get('details') or {}
                    parsed = pd.to_datetime(details.get('Date First Available'), errors='coerce')
                    if pd.isna(parsed) or parsed.date() < CUTOFF:
                        continue
                    record = {key: item.get(key) for key in FIELDS}
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    rows.append({**record, 'date_first_available': parsed.date().isoformat(), 'within_7y': True, 'description_text': text_value(item.get('description')), 'features_text': text_value(item.get('features'))})
                    kept += 1
                    if kept >= limit:
                        break
                except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
                    continue
    finally:
        response.close()
    pd.DataFrame(rows, columns=FIELDS + ['date_first_available', 'within_7y', 'description_text', 'features_text']).to_csv(processed / 'amazon2023_women.csv', index=False, encoding='utf-8-sig')
    manifest = {'url': url, 'crawled_at': datetime.now().astimezone().isoformat(), 'lines_read': lines, 'records_kept': kept, 'parameters': vars(args), 'filter': 'Women AND (Clothing OR AMAZON FASHION); title; description or features; rating_number >= 5; Date First Available >= 2019-11-20'}
    (raw / 'amazon_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()

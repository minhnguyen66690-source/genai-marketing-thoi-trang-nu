"""Lập tập kiểm tra độc lập từ lần thu thập mới đã kết thúc."""
import csv
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import argparse
from pathlib import Path

from common import clean_text, strip_tiki_boilerplate
from holdout_paths import data_dir, file_hash, id_hash, verify_test2
from prepare_v2 import (OUT, RAW, RESULT, ROOT, THEMES, attrs, candidates,
                        duplicate_keys, keywords, norm, product_row, read_jsonl,
                        sentences, split_groups)


def clean_description(text, boiler):
    return '\n'.join(s for s in sentences(clean_text(text)) if norm(s) not in boiler)


def run(additional):
    dest = data_dir(ROOT, 'test2')
    if dest.exists():
        verify_test2(ROOT)
        print('test2 đã được khóa với 120 sản phẩm.')
        return
    raw = dest.parent / 'raw'
    new_products = {}
    raw_sources = []
    for directory in (raw, *additional):
        manifest_path = directory / 'tiki_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['status'] != 'hoàn tất' or manifest['sampling']['leaf_categories'] != 88:
            raise ValueError('Cần lần thu thập đủ 88 danh mục và hoàn tất')
        raw_sources.append(manifest_path)
        filename = 'tiki_products_raw.jsonl'
        if filename not in manifest['files']:
            continue
        new_path = directory / filename
        if file_hash(new_path) != manifest['files'][filename]['sha256']:
            raise ValueError('Dữ liệu thô mới không khớp biên bản')
        raw_sources.append(new_path)
        for record in read_jsonl(new_path):
            new_products.setdefault(str(record['id']), record)
    old_path = RAW / 'tiki_products_raw.jsonl'
    old_frame_path = RAW / 'tiki_frame.jsonl'
    old_rows_path = OUT / 'products_clean.csv'
    boiler_path = RESULT / 'boilerplate_sentences.json'
    objections_path = RESULT / 'objections_by_category.json'
    boiler = {row['normalized'] for row in json.loads(boiler_path.read_text(encoding='utf-8'))}
    objections = json.loads(objections_path.read_text(encoding='utf-8'))
    global_counts = objections['overall_train']['themes']
    default_objections = sorted(THEMES, key=lambda key: (-global_counts[key]['count'], key))
    old_products = read_jsonl(old_path)
    old_ids = {str(row['id']) for row in old_products}
    old_ids.update(str(row['product_id']) for row in read_jsonl(old_frame_path))
    old_descs, old_names = set(), set()

    def remember(row):
        description, name = duplicate_keys(row)
        if description:
            old_descs.add(description)
        if name:
            old_names.add(name)

    for product in old_products:
        row = product_row(product)
        remember(row)
        row['description_text'] = clean_description(row['description_text'], boiler)
        remember(row)
    shops = defaultdict(set)
    train_size = 0
    with old_rows_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            remember(row)
            if row['split'] == 'train':
                train_size += 1
                for phrase in candidates(row['name'], row['brand']):
                    shops[phrase].add(row['seller_id'])
    df = Counter({phrase: len(sellers) for phrase, sellers in shops.items()})
    counts = Counter()
    rows, seen = [], set()
    for product in new_products.values():
        counts['input'] += 1
        row = product_row(product)
        description, name = duplicate_keys(row)
        if str(row['product_id']) in old_ids or description in old_descs or name in old_names:
            counts['old_duplicate'] += 1
            continue
        cleaned = clean_description(row['description_text'], boiler)
        row['description_text'] = strip_tiki_boilerplate(cleaned)
        description, name = duplicate_keys(row)
        if description in old_descs or name in old_names:
            counts['old_duplicate'] += 1
            continue
        if (not row['created_date'] or row['created_date'] < '2019-11-20'
                or len(row['description_text']) < 50 or not isinstance(row['price'], int)
                or row['price'] <= 0 or not (row['category_l3'] or row['category_l2'])):
            counts['excluded_cleaning'] += 1
            continue
        key = norm(row['name']), norm(row['brand'])
        if key in seen:
            counts['duplicate_name_brand'] += 1
            continue
        seen.add(key)
        seller = product.get('current_seller') or {}
        row['seller_id'] = str(seller.get('id') or row['brand'] or row['product_id'])
        row['attributes'] = attrs(product)
        row['meta_description'] = clean_text(product.get('meta_description'))
        rows.append(row)
    counts['eligible_before_sampling'] = len(rows)
    print(json.dumps(dict(counts), ensure_ascii=False), flush=True)
    if not rows:
        raise ValueError('Không có sản phẩm đủ điều kiện')
    assigned = split_groups(rows, test_only=True)
    selected = [row for row, split in zip(rows, assigned) if split == 'test']
    selected.sort(key=lambda row: str(row['product_id']))
    ids = [str(row['product_id']) for row in selected]
    if len(ids) != 120 or len(set(ids)) != 120:
        raise ValueError('Danh sách test2 phải chứa 120 ID duy nhất')
    briefs = []
    for row in selected:
        row['split'] = 'test2'
        cat = row['category_l2'] or ''
        category = row['category_l3'] or cat
        terms = keywords(candidates(row['name'], row['brand']), df, train_size)
        row['keyword_fallback'] = not terms
        ranking = objections['categories'].get(cat, {}).get('ranking', default_objections)
        briefs.append({'item_id': row['product_id'], 'split': 'test2', 'category': category,
                       'keywords': terms or [norm(category)], 'keyword_fallback': not terms,
                       'attributes': row['attributes'], 'price': row['price'], 'objections': ranking[:2]})
    with tempfile.TemporaryDirectory(prefix="test2-stage-", dir=dest.parent) as temporary:
        staged = Path(temporary) / "processed"
        staged.mkdir()
        with (staged / 'products_clean.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows({**row, 'attributes': json.dumps(row['attributes'], ensure_ascii=False)} for row in selected)
        with (staged / 'briefs.jsonl').open('x', encoding='utf-8') as stream:
            stream.writelines(json.dumps(brief, ensure_ascii=False) + '\n' for brief in briefs)
        lock = {'ids': {'test2': ids}, 'test2_sha256': id_hash(ids), 'locked_at': datetime.now(timezone.utc).isoformat(),
                'files': {name: file_hash(staged / name) for name in ('briefs.jsonl', 'products_clean.csv')},
                'sources': {str(path.resolve().relative_to(ROOT)): file_hash(path) for path in
                            (*raw_sources, old_path, old_frame_path, old_rows_path, boiler_path, objections_path)},
                'counts': dict(counts), 'train_products_for_keywords': train_size,
                'category_l2': dict(Counter(row['category_l2'] for row in selected)),
                'max_products_per_seller': max(Counter(row['seller_id'] for row in selected).values())}
        (staged / 'split_ids.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        staged.rename(dest)
    verify_test2(ROOT)
    print(json.dumps({key: lock[key] for key in ('test2_sha256', 'counts', 'category_l2', 'max_products_per_seller')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--additional-raw', type=Path, action='append', default=[])
    run(parser.parse_args().additional_raw)

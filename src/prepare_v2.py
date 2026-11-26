"""Chuẩn bị dữ liệu v2; chỉ chạy sau khi crawl hoàn tất. Seed: 42."""
import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from common import clean_text
from crawl_tiki import product_row
from keywords import VI_GENERIC, VI_PRODUCT_NOUNS, fold
from objections_v2 import THEMES
from preprocess import text_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data_v2/raw'
OUT = ROOT / 'data_v2/processed'
RESULT = ROOT / 'results_v2'
SEED = 42


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s):
    return ' '.join(unicodedata.normalize('NFC', str(s or '')).casefold().split())


def sentences(s):
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+', clean_text(s)) if x.strip()]


def segments(name, brand):
    s = re.sub(r'\[[^\]]*\]', ' ', norm(name))
    brand_words = {fold(x) for x in re.findall(r'\w+', norm(brand))}
    parts = []
    for part in re.split(r'[,;|/+()\[\]{}\-–—:!?.]+', s):
        words = re.findall(r'\w+', part, re.UNICODE)
        words = [w for w in words if fold(w) not in brand_words and not
                 (any(c.isalpha() for c in w) and any(c.isdigit() for c in w)
                  and not re.fullmatch(r'[23]xl', w))]
        if words:
            parts.append(words)
    return parts


# Từ quảng cáo chung chung, không mô tả đặc điểm sản phẩm: cụm chứa các từ này bị loại.
MARKETING = {'thiết kế', 'cao cấp', 'trẻ trung', 'sang trọng', 'thời trang', 'phong cách', 'nhiều màu', 'lựa chọn',
             'đẹp', 'xinh', 'hot', 'giá rẻ', 'chất lượng', 'siêu', 'freeship', 'sale', 'chính hãng', 'quyến rũ', 'cá tính',
             # bổ sung 23/11 sau khi xem tên trên dữ liệu đầy đủ (trước khi sinh nội dung): từ khen chung và nhãn
             # "chất liệu" (để không cắt ra mảnh "liệu lụa"; chất liệu thật nằm ở thuộc tính).
             'phù hợp', 'sành điệu', 'thanh lịch', 'sang chảnh', 'năng động', 'dễ thương', 'chất liệu', 'new'}


# Từ chặn biên cụm = VI_GENERIC của v1 TRỪ các từ chỉ bộ phận hay dáng trang phục và xuất xứ. Bản v1 chặn cả
# "tay", "cổ", "lưng", "dáng", "kiểu"… nên loại mất đặc điểm cốt lõi như "tay lỡ", "cổ tim", "lưng cao", "áo kiểu".
BOUNDARY = set(VI_GENERIC) - {'tay', 'cổ', 'dài', 'lưng', 'dáng', 'form', 'kiểu', 'quảng châu', 'hàn quốc'}


def has_marketing(phrase):
    padded = ' ' + phrase + ' '
    return any(' ' + m + ' ' in padded for m in MARKETING)


def candidates(name, brand):
    """Trả dict cụm -> danh sách vị trí (đoạn, đầu, cuối) trong tên, để biết hai cụm có chồng nhau không."""
    found = {}
    for seg_no, words in enumerate(segments(name, brand)):
        # Đánh dấu vị trí âm tiết thuộc cụm quảng cáo để không lấy mảnh cắt dở như "kế cao" từ "thiết kế cao cấp".
        masked = set()
        for m in MARKETING:
            mw = m.split()
            for j in range(len(words) - len(mw) + 1):
                if words[j:j+len(mw)] == mw:
                    masked.update(range(j, j + len(mw)))
        for n in range(1, 5):
            for i in range(len(words) - n + 1):
                if masked & set(range(i, i + n)):
                    continue
                gram = words[i:i+n]
                phrase = ' '.join(gram)
                if gram[0] in BOUNDARY or gram[-1] in BOUNDARY or phrase in BOUNDARY:
                    continue
                if has_marketing(phrase):
                    continue
                if n == 1 and phrase not in VI_PRODUCT_NOUNS:
                    continue
                found.setdefault(phrase, []).append((seg_no, i, i + n))
    return found


def overlaps(spans_a, spans_b):
    """Hai cụm chồng nhau nếu có vị trí cùng đoạn và khoảng âm tiết giao nhau."""
    return any(sa == sb and a0 < b1 and b0 < a1 for sa, a0, a1 in spans_a for sb, b0, b1 in spans_b)


def noun(p):
    return any(' ' + n + ' ' in ' ' + p + ' ' for n in VI_PRODUCT_NOUNS)


def keywords(phrases, counts, n):
    eligible = [p for p in phrases if counts[p] >= 3]
    def score(p):
        return math.log1p(counts[p]) * (1 + .5 * (len(p.split()) - 1)) * (.3 if counts[p] > .3*n else 1)
    eligible.sort(key=lambda p: (-score(p), p))
    primary = next((p for p in eligible if noun(p) and len(p.split()) >= 2), None)
    primary = primary or next((p for p in eligible if noun(p)), None)
    selected = [primary] if primary else []
    for p in eligible:
        if len(selected) == 6:
            break
        # Từ đơn chỉ dùng làm cụm chính dự phòng. Cụm chồng vị trí với cụm đã chọn ("co giãn" và "giãn 4 chiều",
        # "đầm ôm" và "ôm body") là hai lát cắt của cùng một ý: bỏ. Cụm ở đoạn khác của tên vẫn giữ
        # ("áo thun" và "thun cá sấu" trong "áo thun nữ, chất liệu thun cá sấu").
        if len(p.split()) == 1:
            continue
        if any(overlaps(phrases[p], phrases[q]) for q in selected):
            continue
        selected.append(p)
    return selected


def attrs(item):
    result = defaultdict(list)
    def add(key, val):
        values = val if isinstance(val, list) else [val]
        for v in values:
            if isinstance(v, dict):
                v = v.get('label') or v.get('value') or v.get('name')
            if v is not None and clean_text(v) and clean_text(v) not in result[key]:
                result[key].append(clean_text(v))
    for group in item.get('specifications') or []:
        for a in group.get('attributes', [group]):
            key = norm(a.get('name') or a.get('code'))
            for target, names in {'chất liệu': ('chất liệu', 'material'), 'xuất xứ': ('xuất xứ', 'origin'), 'kiểu dáng': ('kiểu dáng', 'style')}.items():
                if any(x in key for x in names):
                    add(target, a.get('value'))
    for option in item.get('configurable_options') or []:
        key = norm(option.get('name') or option.get('attribute_code') or option.get('code'))
        for target, names in {'màu sắc': ('màu', 'color', 'colour'), 'kích cỡ': ('kích cỡ', 'kích thước', 'size')}.items():
            if any(x in key for x in names):
                add(target, option.get('values') or [])
    return dict(result)


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k/n
    center = (p + z*z/(2*n))/(1 + z*z/n)
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return {'count': k, 'n': n, 'rate': p, 'wilson_95': [max(0, center-radius), min(1, center+radius)]}


def duplicate_keys(row):
    desc = norm(row['description_text'])[:300]
    name = ' '.join(' '.join(x) for x in segments(row['name'], row['brand']))
    return desc, name


def split_groups(rows, *, test_only=False):
    n = len(rows)
    parent = list(range(n))
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a
    descriptions, names = {}, {}
    for i, row in enumerate(rows):
        desc, name = duplicate_keys(row)
        for value, seen in ((desc, descriptions), (name, names)):
            if value:
                if value in seen:
                    union(i, seen[value])
                else:
                    seen[value] = i
    groups = defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)
    categories = Counter(r['category_l2'] or '' for r in rows)
    eligible_cats = {c for c, v in categories.items() if v >= 15}
    total = sum(categories[c] for c in eligible_cats)

    def quotas(target):
        """Chỉ tiêu mỗi danh mục cấp 2 tỷ lệ với quy mô danh mục (phần dư lớn nhất)."""
        raw = {c: target * categories[c] / total for c in eligible_cats}
        q = {c: int(raw[c]) for c in raw}
        for c in sorted(raw, key=lambda c: (-(raw[c] - q[c]), c))[:target - sum(q.values())]:
            q[c] += 1
        return q

    # Bản sửa 23/11: bản trước chọn cả NHÓM vào test/dev theo kiểu tham lam, nên các nhóm gần trùng lớn của một
    # shop lấp đầy chỉ tiêu (test chỉ còn 20 mô tả khác nhau từ 8 shop). Nay mỗi nhóm góp ĐÚNG MỘT sản phẩm đại
    # diện vào test/dev; các sản phẩm còn lại của nhóm bị loại ('excluded') để không rò rỉ sang train. Mỗi shop
    # góp tối đa 3 sản phẩm vào test và 2 vào dev.
    rng = random.Random(SEED)
    order = [sorted(g) for g in groups.values()]
    rng.shuffle(order)
    assignments = ['train'] * n
    used = set()
    targets = [('test', 120, 3)] if test_only else [('test', 120, 3), ('dev', 60, 2)]
    for label, target, cap in targets:
        quota, filled, per_seller = quotas(target), Counter(), Counter()
        for gi, g in enumerate(order):
            if gi in used:
                continue
            rep = g[rng.randrange(len(g))]
            c, s = rows[rep]['category_l2'] or '', rows[rep]['seller_id']
            if c not in eligible_cats or filled[c] >= quota[c] or per_seller[s] >= cap:
                continue
            used.add(gi)
            assignments[rep] = label
            filled[c] += 1
            per_seller[s] += 1
            for i in g:
                if i != rep:
                    assignments[i] = 'excluded'
        if sum(filled.values()) < target:
            raise ValueError(f'Không chọn đủ {target} sản phẩm {label}: thiếu {target - sum(filled.values())}')
    return assignments


def theme_stats(reviews):
    counts = {name: sum(bool(pattern.search(r["content"])) for r in reviews) for name, pattern in THEMES.items()}
    n = len(reviews)
    return {'n_reviews': n, 'themes': {k: {'count': v, 'rate': v/n if n else None} for k, v in counts.items()}}


def describe(rows):
    features = [text_features(r['name'], r['description_text']) for r in rows]
    n = len(rows)
    return {'n': n, 'median_description_words': statistics.median(x['desc_len_words'] for x in features) if n else None,
            'median_title_chars': statistics.median(x['title_len_chars'] for x in features) if n else None,
            'description_under_200_words': wilson(sum(x['desc_len_words'] < 200 for x in features), n),
            'size': wilson(sum(x['has_size_info'] for x in features), n),
            'material': wilson(sum(x['has_material_info'] for x in features), n),
            'care': wilson(sum(x['has_care_info'] for x in features), n),
            'title_15_to_50_chars': wilson(sum(15 <= x['title_len_chars'] <= 50 for x in features), n)}


def main():
    paths = [RAW / name for name in ('tiki_frame.jsonl', 'tiki_products_raw.jsonl', 'tiki_reviews_lowstar_raw.jsonl')]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Thiếu đầu vào: ' + ', '.join(missing))
    frame, products, reviews = [read_jsonl(p) for p in paths]
    if not products:
        raise ValueError('Không có sản phẩm đầu vào.')
    if len({str(x.get('id')) for x in products}) != len(products):
        raise ValueError('ID sản phẩm thô bị trùng hoặc thiếu.')
    rows = [product_row(p) for p in products]
    counts = Counter()
    originals = {}
    for row in rows:
        for s in sentences(row['description_text']):
            k = norm(s)
            if len(k) >= 40:
                originals.setdefault(k, s)
        counts.update({norm(s) for s in sentences(row['description_text']) if len(norm(s)) >= 40})
    boiler = {k for k, v in counts.items() if v/len(rows) >= .05}
    boiler_list = [{'sentence': originals[k], 'normalized': k, 'product_count': counts[k], 'rate': counts[k]/len(rows)} for k in sorted(boiler, key=lambda x: (-counts[x], x))]
    sort_ids = defaultdict(set)
    for x in frame:
        sort_ids[str(x.get('sort'))].add(str(x.get('product_id')))
    profile = {'n_products': len(rows), 'n_leaf_categories': len({r['leaf_category_id'] for r in rows if r['leaf_category_id'] is not None}), 'n_category_l2': len({r['category_l2'] for r in rows if r['category_l2']}),
               'missing_rate': {k: sum(r[k] is None or r[k] == '' for r in rows)/len(rows) for k in rows[0]},
               'distributions': {}, 'frame': {'unique_products_by_sort': {k: len(v) for k, v in sort_ids.items()}, 'pairwise_overlap': {a+' / '+b: {'intersection': len(sort_ids[a] & sort_ids[b]), 'union': len(sort_ids[a] | sort_ids[b]), 'jaccard': len(sort_ids[a] & sort_ids[b])/len(sort_ids[a] | sort_ids[b]) if sort_ids[a] | sort_ids[b] else None} for a in sorted(sort_ids) for b in sorted(sort_ids) if a < b}}}
    for key in ('quantity_sold', 'review_count'):
        values = sorted(float(r[key]) for r in rows if isinstance(r[key], (int, float)))
        profile['distributions'][key] = {'n_known': len(values), 'n_missing': len(rows)-len(values), 'zero_rate_known': sum(v == 0 for v in values)/len(values) if values else None, 'min': min(values) if values else None, 'median': statistics.median(values) if values else None, 'max': max(values) if values else None}
    sources = defaultdict(set)
    for x in frame:
        sources[str(x.get('product_id'))].add(str(x.get('sort')))
    for row, item in zip(rows, products):
        row['frame_sorts'] = '|'.join(sorted(sources.get(str(row['product_id']), [])))
        original = clean_text(row['description_text'])
        row['description_text'] = '\n'.join(s for s in sentences(original) if norm(s) not in boiler)
        row['boilerplate_removed_chars'] = max(0, len(original)-len(row['description_text']))
        row['attributes'] = attrs(item)
        seller = item.get('current_seller') if isinstance(item.get('current_seller'), dict) else {}
        row['seller_id'] = str(seller.get('id') or row.get('brand') or row['product_id'])
    funnel = [{'step': 'input', 'n': len(rows)}]
    rows = [r for r in rows if r['created_date'] and r['created_date'] >= '2019-11-20']
    funnel.append({'step': 'created_date', 'n': len(rows)})
    rows = [r for r in rows if len(r['description_text']) >= 50]
    funnel.append({'step': 'clean_description', 'n': len(rows)})
    # Bản tóm tắt cần giá; sản phẩm không có giá bị loại ở phễu (không dừng cả chương trình).
    rows = [r for r in rows if isinstance(r['price'], int) and r['price'] > 0]
    funnel.append({'step': 'has_price', 'n': len(rows)})
    seen, unique = set(), []
    for r in rows:
        key = (norm(r['name']), norm(r['brand']))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = unique
    funnel.append({'step': 'deduplicated', 'n': len(rows)})
    assignments = split_groups(rows)
    for row, split in zip(rows, assignments):
        row['split'] = split
    by_id = {str(r['product_id']): r for r in rows}
    low = []
    for review in reviews:
        pid = str(review.get('product_id'))
        content = clean_text(review.get('content'))
        if pid in by_id and content and review.get('rating') in (1, 2, 3):
            low.append({'product_id': pid, 'rating': review['rating'], 'content': content, 'created_at': review.get('created_at')})
    train_reviews = [r for r in low if by_id[r['product_id']]['split'] == 'train']
    train_stats = theme_stats(train_reviews)
    cats = sorted({r['category_l2'] or '' for r in rows})
    category_stats = {}
    for cat in cats:
        sub = [r for r in train_reviews if (by_id[r['product_id']]['category_l2'] or '') == cat]
        stat = theme_stats(sub)
        stat['uses_global_ranking'] = len(sub) < 30
        source = train_stats if stat['uses_global_ranking'] else stat
        stat['ranking'] = sorted(THEMES, key=lambda k: (-source['themes'][k]['count'], k))
        category_stats[cat] = stat
    phrase_sets = [candidates(r['name'], r['brand']) for r in rows]
    # df = số CỬA HÀNG khác nhau (trong train) có cụm trong tên sản phẩm. Đếm theo cửa hàng thay vì theo sản phẩm
    # để tên shop hay cách đặt tên riêng của một shop đăng nhiều sản phẩm ("La Mộc") không thành cụm đặc trưng.
    shops = defaultdict(set)
    for r, phrases in zip(rows, phrase_sets):
        if r['split'] == 'train':
            for p in phrases:
                shops[p].add(r['seller_id'])
    df = Counter({p: len(s) for p, s in shops.items()})
    train_n = assignments.count('train')
    assert 'cánh dơi' in candidates('Áo kiểu nữ đẹp form cánh dơi, chất liệu thun cá sấu', '')
    if df['cánh dơi'] >= 3:
        assert any('cánh dơi' in p for p in keywords(candidates('Áo kiểu nữ đẹp form cánh dơi, chất liệu thun cá sấu', ''), df, train_n))
    briefs = []
    for row, phrases in zip(rows, phrase_sets):
        kws = keywords(phrases, df, train_n)
        # Tên quá ngắn không cho cụm nào đạt df ≥ 3: dùng tên danh mục làm từ khóa chính và đánh dấu.
        fallback = not kws
        if fallback:
            kws = [norm(row['category_l3'] or row['category_l2'])]
        row['keyword_fallback'] = fallback
        brief = {'item_id': row['product_id'], 'split': row['split'], 'category': row['category_l3'] or row['category_l2'], 'keywords': kws, 'keyword_fallback': fallback, 'attributes': row['attributes'], 'price': row['price'], 'objections': category_stats[row['category_l2'] or '']['ranking'][:2]}
        assert brief['item_id'] and brief['category'] and brief['keywords'] and isinstance(brief['price'], int) and brief['price'] > 0 and len(brief['objections']) == 2, f'Brief không hợp lệ: {brief["item_id"]}'
        briefs.append(brief)
    ids = {s: sorted(str(r['product_id']) for r in rows if r['split'] == s) for s in ('train', 'dev', 'test')}
    digest = hashlib.sha256(('\n'.join(ids['test'])+'\n').encode('utf-8')).hexdigest()
    rq1 = {'overall': describe(rows), 'by_category_l2': {c: describe([r for r in rows if (r['category_l2'] or '') == c]) for c in cats}, 'low_review_themes': {'overall': theme_stats(low), 'by_category_l2': {c: theme_stats([r for r in low if (by_id[r['product_id']]['category_l2'] or '') == c]) for c in cats}}}
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    def save(path, obj):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for filename, obj in [('data_profile.json', profile), ('boilerplate_sentences.json', boiler_list), ('funnel.json', funnel), ('objections_by_category.json', {'overall_train': train_stats, 'categories': category_stats}), ('rq1_descriptive.json', rq1)]:
        save(RESULT / filename, obj)
    save(OUT / 'split_ids.json', {'ids': ids, 'test_sha256': digest, 'hash_encoding': 'UTF-8, mỗi ID một dòng, có newline cuối'})
    with (OUT / 'products_clean.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, 'attributes': json.dumps(r['attributes'], ensure_ascii=False)})
    with (OUT / 'reviews_lowstar_clean.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['product_id', 'rating', 'content', 'created_at'])
        writer.writeheader()
        writer.writerows(low)
    with (OUT / 'briefs.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for b in briefs:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    rng = random.Random(SEED)
    print(json.dumps({'funnel': funnel, 'split_sizes': {k: len(v) for k, v in ids.items()}, 'test_sha256': digest, 'optional_missing': {k: sum(k not in b['attributes'] for b in briefs) for k in ('chất liệu', 'xuất xứ', 'màu sắc', 'kích cỡ', 'kiểu dáng')}, 'keyword_examples': [{'name': r['name'], 'keywords': b['keywords']} for r, b in rng.sample(list(zip(rows, briefs)), min(10, len(rows)))], 'dev_briefs': [b for b in briefs if b['split'] == 'dev'][:5], 'boilerplate_removed': boiler_list}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

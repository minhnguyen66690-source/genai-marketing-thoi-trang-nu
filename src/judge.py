"""Giám khảo LLM chấm lô ẩn danh theo rubric G-Eval."""
import argparse
import hashlib
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import llm_backends

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '_workspace' / '03_results' / 'judge_raw'
RUBRIC = '''Bạn là giám khảo độc lập. Chỉ dựa vào brief và nội dung, không suy diễn sự thật.
Cho từng tiêu chí điểm nguyên 1–5: 1 rất kém/không có; 2 kém, thiếu nhiều; 3 đạt một phần; 4 tốt, thiếu nhỏ; 5 xuất sắc, đầy đủ và chính xác. Mỗi điểm phải có trích đoạn bằng chứng nguyên văn; nếu không có, evidence là chuỗi rỗng.
Mô tả: tu_nhien, lien_quan_tu_khoa, thuyet_phuc, nhat_quan_thong_tin.
Tiêu đề email: ro_rang, gay_to_mo, lien_quan.
Kịch bản: cta_ro_rang, nhu_cau_loi_ich (liên hệ nhu cầu và lợi ích theo SPIN), xu_ly_phan_doi, dung_su_that.
Ngoài điểm kịch bản, nêu nguyen_tac_thuyet_phuc là danh sách đối tượng {nhan, evidence} theo Cialdini; khan_hiem_hoac_bang_chung_xa_hoi_khong_co_trong_brief là boolean.
Chỉ trả JSON dạng {"items":[{"item_id":"...", "mo_ta":{"tu_nhien":{"score":1,"evidence":"..."}, ...}, "email":{...}, "kich_ban":{..., "nguyen_tac_thuyet_phuc":[], "khan_hiem_hoac_bang_chung_xa_hoi_khong_co_trong_brief":false}}]}. Chấm mọi item và mọi tiêu chí. Nếu nội dung thiếu, điểm 1 và evidence rỗng.'''


def validate(payload, ids, sources):
    """Kiểm tra cấu trúc và các điểm trước khi chấp nhận kết quả."""
    groups = {'mo_ta': ('tu_nhien', 'lien_quan_tu_khoa', 'thuyet_phuc', 'nhat_quan_thong_tin'), 'email': ('ro_rang', 'gay_to_mo', 'lien_quan'), 'kich_ban': ('cta_ro_rang', 'nhu_cau_loi_ich', 'xu_ly_phan_doi', 'dung_su_that')}
    items = payload['items']
    if len(items) != len(ids) or {x['item_id'] for x in items} != set(ids):
        raise ValueError('Sai danh sách ID')
    for item in items:
        for group, names in groups.items():
            for name in names:
                value = item[group][name]
                if type(value['score']) is not int or value['score'] not in range(1, 6) or not isinstance(value['evidence'], str):
                    raise ValueError('Điểm hoặc bằng chứng sai định dạng')
        script = item['kich_ban']
        if not isinstance(script['nguyen_tac_thuyet_phuc'], list) or type(script['khan_hiem_hoac_bang_chung_xa_hoi_khong_co_trong_brief']) is not bool:
            raise ValueError('Nhãn thuyết phục sai định dạng')
    return items


def content_key(content):
    """Khóa bộ nhớ đệm: SHA-1 của brief + output (không phụ thuộc tên phương pháp)."""
    return hashlib.sha1(json.dumps(content, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def cache_path(judge_backend):
    """Đường dẫn file bộ nhớ đệm kết quả chấm của một giám khảo."""
    return RAW.parent / f'judge_cache_{judge_backend}.jsonl'


def load_cache(judge_backend):
    """Đọc bộ nhớ đệm {khóa: kết quả chấm} để chạy lại không phải gọi lại giám khảo."""
    path = cache_path(judge_backend)
    cache = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                cache[record['key']] = record['result']
    return cache


def append_cache(judge_backend, key, result):
    """Ghi thêm một kết quả chấm vào bộ nhớ đệm."""
    with cache_path(judge_backend).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'key': key, 'result': result}, ensure_ascii=False) + '\n')


def anonymize(items, seed):
    """Gán mã ẩn danh ngẫu nhiên (cố định theo seed) cho từng mẫu rồi xáo trộn thứ tự."""
    rng = random.Random(seed)
    anonymous = []
    for index, item in enumerate(items):
        anonymous.append((f'{rng.getrandbits(128):032x}', index, {'brief': item.get('brief', {}), 'output': item.get('output', {})}))
    rng.shuffle(anonymous)
    return anonymous


def seed_cache_from_raw(items, judge_backend='gemini', seed=42):
    """Khôi phục bộ nhớ đệm từ các phản hồi thô đã lưu của một lần chạy trước có cùng danh sách mẫu và seed.

    Mã ẩn danh được sinh lại đúng như lần chạy trước (cùng seed, cùng thứ tự mẫu), nên có thể ghép
    lại từng kết quả trong judge_raw/ với nội dung tương ứng. Mỗi kết quả vẫn qua bước kiểm tra định dạng.
    """
    mapping = {identity: content for identity, _, content in anonymize(items, seed)}
    cache = load_cache(judge_backend)
    recovered = 0
    for path in sorted(RAW.glob(f'*_{judge_backend}.txt')):
        try:
            payload = llm_backends.extract_json(path.read_text(encoding='utf-8'))
            entries = [x for x in payload['items'] if x.get('item_id') in mapping]
            if not entries:
                continue
            valid = validate({'items': entries}, [x['item_id'] for x in entries], None)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        for entry in valid:
            key = content_key(mapping[entry['item_id']])
            if key not in cache:
                cache[key] = entry
                append_cache(judge_backend, key, entry)
                recovered += 1
    return recovered


def judge_items(items, judge_backend='gemini', seed=42):
    """Xáo trộn, ẩn danh, chấm từng lô năm mẫu và lưu phản hồi thô (có bộ nhớ đệm theo nội dung)."""
    if judge_backend not in ('gemini', 'gpt'):
        raise ValueError('Backend phải là gemini hoặc gpt')
    anonymous = anonymize(items, seed)
    RAW.mkdir(parents=True, exist_ok=True)
    results = [None] * len(items)
    cache = load_cache(judge_backend)
    pending = []
    for identity, index, content in anonymous:
        cached = cache.get(content_key(content))
        if cached is not None:
            results[index] = cached
        else:
            pending.append((identity, index, content))
    anonymous = pending
    failed = []
    for start in range(0, len(anonymous), 5):
        batch = anonymous[start:start + 5]
        if not judge_batch(batch, judge_backend, results):
            # Lô lỗi hai lần: chấm lại từng mẫu một để một mẫu lỗi không làm hỏng cả lô.
            for single in batch:
                if not judge_batch([single], judge_backend, results):
                    failed.append(single[1])
    if failed:
        print(f'CẢNH BÁO: {len(failed)} mẫu không chấm được (JSON lỗi sau nhiều lần), để trống.')
    return results


def judge_batch(batch, judge_backend, results):
    """Chấm một lô (tối đa 2 lần thử). Trả về True nếu thành công và đã ghi kết quả vào results, bộ nhớ đệm."""
    request = [{'item_id': identity, **content} for identity, _, content in batch]
    prompt = RUBRIC + '\nDữ liệu:\n' + json.dumps(request, ensure_ascii=False)
    for attempt in range(2):
        try:
            if judge_backend == 'gemini':
                raw = llm_backends.judge(prompt, tag='evaluation')
            else:
                raw = llm_backends.generate(prompt, model='gpt-5.6-luna', tag='evaluation')
            path = RAW / f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_{uuid.uuid4().hex}_{judge_backend}.txt'
            path.write_text(raw, encoding='utf-8')
            parsed = validate(llm_backends.extract_json(raw), [x[0] for x in batch], None)
            lookup = {x['item_id']: x for x in parsed}
            for identity, index, content in batch:
                results[index] = lookup[identity]
                append_cache(judge_backend, content_key(content), lookup[identity])
            return True
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError):
            continue
    return False


def agreement(scores_a, scores_b):
    """Tính tương quan Spearman và kappa trọng số bậc hai trên cặp đủ dữ liệu."""
    if len(scores_a) != len(scores_b):
        raise ValueError('Hai danh sách phải cùng độ dài')
    pairs = [(a, b) for a, b in zip(scores_a, scores_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return {'n': len(pairs), 'spearman': None, 'quadratic_kappa': None}
    a, b = zip(*pairs)
    rho = spearmanr(a, b).statistic
    kappa = cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights='quadratic')
    return {'n': len(pairs), 'spearman': float(rho) if np.isfinite(rho) else None, 'quadratic_kappa': float(kappa) if np.isfinite(kappa) else None}


def main():
    """Chấm thử các dòng JSONL được chỉ định."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--judge-backend', choices=['gemini', 'gpt'], default='gemini')
    args = parser.parse_args()
    items = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8-sig').splitlines() if line.strip()][:args.limit]
    result = judge_items(items, args.judge_backend)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Đường dẫn và kiểm tra mẫu đã khóa của tập bổ sung."""
import hashlib
import json


def data_dir(root, split):
    return (root / '_workspace/holdout/data_v3/processed' if split == 'test2'
            else root / 'data_v2/processed')


def results_dir(root, split):
    return root / ('_workspace/holdout/results_v3' if split == 'test2' else 'results_v2')


def judge_file(root, split, judge, kind):
    suffix = '_dev' if split == 'dev' else ''
    return results_dir(root, split) / 'judge' / f'{judge}_{kind}{suffix}.jsonl'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def id_hash(ids):
    return hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode('utf-8')).hexdigest()


def verify_test2(root):
    path = data_dir(root, 'test2')
    lock = json.loads((path / 'split_ids.json').read_text(encoding='utf-8'))
    ids = lock['ids']['test2']
    if len(ids) != 120 or len(set(ids)) != 120 or id_hash(ids) != lock['test2_sha256']:
        raise ValueError('Danh sách test2 không khớp với khóa')
    for filename in ('briefs.jsonl', 'products_clean.csv'):
        if file_hash(path / filename) != lock['files'][filename]:
            raise ValueError(f'Tệp test2 đã thay đổi sau khi khóa: {filename}')
    return set(ids)

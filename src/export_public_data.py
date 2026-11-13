"""Xuất các đặc trưng và kết quả được phép công khai."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = 'product_id url category_l2 price list_price discount_rate quantity_sold product_age_days n_images n_options has_video n_specs title_len_chars title_len_words desc_len_chars desc_len_words n_paragraphs n_bullets n_sentences avg_sentence_len_words n_exclamations uppercase_ratio emoji_count has_size_info has_material_info has_care_info has_cta has_meta_description best_seller created_date'.split()
INPUTS = 'product_id category_l2 keywords attributes best_seller split price'.split()
FORBIDDEN = {'name', 'title', 'description', 'description_text', 'short_description', 'meta', 'meta_title', 'meta_description', 'meta_keywords', 'reference_description', 'reference_title', 'review', 'review_text'}

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in {'reference_description', 'reference_title'}}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

def main():
    public, results = ROOT / 'data/public', ROOT / 'results/public'
    public.mkdir(parents=True, exist_ok=True)
    (results / 'generations').mkdir(parents=True, exist_ok=True)
    written = []
    for source, target, columns in [('tiki_features.csv', 'tiki_features_public.csv', FEATURES), ('gen_inputs_tiki.csv', 'gen_inputs_tiki_public.csv', INPUTS)]:
        path = ROOT / 'data/processed' / source
        if not path.exists():
            print('CẢNH BÁO: chưa có', path)
            continue
        df = pd.read_csv(path)
        missing = [c for c in columns if c not in df.columns]
        if missing: print('CẢNH BÁO: cột nguồn chưa có:', ', '.join(missing))
        df = df.reindex(columns=columns)
        assert not (set(df.columns) & FORBIDDEN)
        dest = public / target
        with open(dest, 'w', encoding='utf-8', newline='') as f: df.to_csv(f, index=False)
        written.append(dest)
    src = ROOT / '_workspace/03_results'
    for path in sorted(src.glob('*')):
        if not path.is_file() or 'judge_raw' in path.name.lower(): continue
        if path.name.startswith('generations_') and path.suffix == '.jsonl':
            dest = results / 'generations' / path.name
            with open(path, encoding='utf-8-sig') as inp, open(dest, 'w', encoding='utf-8') as out:
                for line in inp:
                    if line.strip(): out.write(json.dumps(clean(json.loads(line)), ensure_ascii=False) + '\n')
        elif path.suffix == '.json' or (path.suffix == '.csv' and path.name.startswith('eval_')):
            dest = results / path.name
            with open(path, encoding='utf-8-sig') as inp, open(dest, 'w', encoding='utf-8', newline='') as out: out.write(inp.read())
        else: continue
        written.append(dest)
        if path.name in {'objection_themes_summary.json', 'objections_by_category.json'}:
            dest = public / path.name
            with open(path, encoding='utf-8-sig') as inp, open(dest, 'w', encoding='utf-8') as out: out.write(inp.read())
            written.append(dest)
    loss = ROOT / 'logs/finetune_tiki.csv'
    if loss.exists():
        dest = results / loss.name
        with open(loss, encoding='utf-8-sig') as inp, open(dest, 'w', encoding='utf-8') as out: out.write(inp.read())
        written.append(dest)
    print(f'Đã ghi {len(written)} tệp; kiểm tra tên cột: không có cột văn bản gốc bị cấm.')

if __name__ == '__main__': main()

"""Đánh giá các phương pháp sinh theo từng sản phẩm và xuất báo cáo."""
import argparse
import itertools
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from common import RESULTS, FIGURES, save_json, setup_plot, save_figure
from metrics_content import seo_checks, email_checks, factual_support, script_rule_checks, reference_similarity


SEO_LABELS = {'title_len_50_60': 'S19', 'title_tiki_15_50': 'S23', 'title_tiki_max_225': 'S23', 'meta_len_120_156': 'S12', 'keyphrase_in_title': 'S11', 'keyphrase_in_first_paragraph': 'S10', 'keyphrase_in_meta': 'S11', 'density_proxy_0_5_3': 'S09 proxy', 'desc_words_ge_200': 'S13/S24'}
EMAIL_LABELS = {'mailchimp_le9w_le60c': 'E01', 'omnisend_4_6w_or_21_40c': 'E05', 'punct_le3': 'E02', 'emoji_le1': 'E03', 'no_fake_re_fwd': 'E16', 'keyword_relevance': 'proxy', 'offer_supported': 'proxy'}


def bootstrap(values):
    """Tính trung bình và khoảng tin cậy percentile bootstrap 95%."""
    data = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(dtype=float)
    if not len(data):
        return {'n': 0, 'mean': None, 'ci95': [None, None]}
    rng = np.random.default_rng(42)
    means = np.mean(rng.choice(data, size=(1000, len(data)), replace=True), axis=1)
    return {'n': len(data), 'mean': float(data.mean()), 'ci95': np.quantile(means, [0.025, 0.975]).tolist()}


def paired_tests(frame, metric_columns):
    """Kiểm định Wilcoxon theo item và điều chỉnh Holm trên toàn họ kiểm định."""
    tests = []
    for metric in metric_columns:
        table = frame.pivot(index='item_id', columns='method', values=metric)
        for a, b in itertools.combinations(table.columns, 2):
            pairs = table[[a, b]].dropna().astype(float)
            if len(pairs) < 2:
                continue
            diff = pairs[a] - pairs[b]
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(pairs[a], pairs[b]).pvalue)
            tests.append({'metric': metric, 'method_a': a, 'method_b': b, 'n_pairs': len(pairs), 'p': p})
    ordered = sorted(range(len(tests)), key=lambda i: tests[i]['p'])
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, tests[index]['p'] * (len(tests) - rank)))
        tests[index]['p_holm'] = previous
    return tests


def flatten_judge(result):
    """Lấy riêng điểm giám khảo để thống kê, giữ bằng chứng trong tệp thô."""
    scores = {}
    for group in ('mo_ta', 'email', 'kich_ban'):
        for name, value in result[group].items():
            if isinstance(value, dict) and 'score' in value:
                scores['judge_' + group + '_' + name] = value['score']
    return scores


def plot_bar(frame, columns, path, title, source):
    """Vẽ trung bình và khoảng bootstrap cho các biến có dữ liệu."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    means = []
    low = []
    high = []
    for method, group in frame.groupby('method', sort=True):
        for column in columns:
            stat = bootstrap(group[column])
            if stat['mean'] is not None:
                labels.append(method + '\n' + column)
                means.append(stat['mean'])
                low.append(stat['mean'] - stat['ci95'][0])
                high.append(stat['ci95'][1] - stat['mean'])
    if means:
        ax.bar(range(len(means)), means, yerr=[low, high], capsize=3)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu áp dụng', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title(title)
    save_figure(fig, path, source)


def plot_heatmap(passrates, kind, labels, path, title, source):
    """Vẽ tỷ lệ đạt theo từng tiêu chí và phương pháp."""
    subset = passrates[passrates['kind'] == kind]
    table = subset.pivot(index='method', columns='criterion', values='pass_rate') if not subset.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4.8))
    if not table.empty and table.notna().any().any():
        table = table.rename(columns=lambda x: labels.get(x, '') + ' ' + x)
        sns.heatmap(table, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Tỷ lệ đạt'})
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    else:
        ax.text(0.5, 0.5, 'Không có dữ liệu áp dụng', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)
    save_figure(fig, path, source)


def evaluate(inputs, dataset, fast, no_judge, backend):
    """Đọc JSONL, tính chỉ số, thống kê và lưu bảng cùng biểu đồ."""
    paths = sorted(Path(inputs).glob(f'generations_*_{dataset}.jsonl'))
    if not paths:
        raise FileNotFoundError(f'Không tìm thấy generations_*_{dataset}.jsonl trong {inputs}')
    seen_methods = {p.name[len('generations_'):-len('_' + dataset + '.jsonl')] for p in paths}
    for method in ('M0', 'M1', 'M2', 'M3a', 'M3b'):
        if method.lower() not in {x.lower() for x in seen_methods}:
            warnings.warn(f'Thiếu file phương pháp {method}, bỏ qua')
    items = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if line.strip():
                item = json.loads(line)
                item['_source'] = path.name
                items.append(item)
    if not items:
        raise ValueError('Không có dòng dữ liệu')
    keys = [(x['method'], x['item_id']) for x in items]
    if len(keys) != len(set(keys)):
        raise ValueError('Trùng cặp method/item_id')
    judge_results = [None] * len(items)
    if not no_judge:
        from judge import judge_items
        judge_results = judge_items(items, backend)
    rows = []
    email_observations = []
    reference_seen = set()
    for item, judged in zip(items, judge_results):
        brief = item.get('brief') or {}
        output = item.get('output') or {}
        keywords = brief.get('keywords') or []
        keyphrase = keywords[0] if keywords else ''
        variants = [(item['method'], output)]
        if item['item_id'] not in reference_seen and item.get('reference_description') is not None:
            variants.append(('reference', {'title': item.get('reference_title') or '', 'description': item['reference_description']}))
            reference_seen.add(item['item_id'])
        for method, value in variants:
            title = value.get('title') or ''
            desc = value.get('description') or ''
            seo = seo_checks(title, value.get('meta_description'), desc, keyphrase, keywords, item.get('lang', 'vi'))
            fact = factual_support(title + ' ' + desc, brief)
            row = {'item_id': item['item_id'], 'product_id': item.get('product_id'), 'method': method, 'dataset': item.get('dataset', dataset), 'lang': item.get('lang'), 'source_file': item['_source'], 'seo_compliance_rate': seo['compliance_rate'], 'keyword_coverage': seo['keyword_coverage'], 'keyphrase_density_proxy': seo['keyphrase_density_proxy'], 'n_paragraphs': seo['n_paragraphs'], 'n_bullets': seo['n_bullets'], 'avg_sentence_words': seo['avg_sentence_words'], 'pct_sentences_over_20_words': seo['pct_sentences_over_20_words'], 'factual_supported': fact['supported'], 'factual_unsupported': fact['unsupported'], 'factual_total': fact['total'], 'hallucination_rate': fact['hallucination_rate'], 'latency_s': item.get('latency_s') if method != 'reference' else None}
            row.update({'seo_' + k: v for k, v in seo['checks'].items()})
            if method != 'reference':
                script = script_rule_checks(value.get('sales_script', ''), brief, brief.get('objections'))
                row.update({'script_cta_present': script['cta_present'], 'script_objection_keyword_coverage': script['objection_keyword_coverage'], 'script_length_words': script['length_words'], 'script_hallucination_rate': script['factual_support']['hallucination_rate']})
                for subject in value.get('email_subjects') or []:
                    email = email_checks(subject, keywords, brief)
                    email_observations.append({'item_id': item['item_id'], 'method': method, **email['checks']})
                if judged:
                    row.update(flatten_judge(judged))
                row['_candidate'] = desc
                row['_reference'] = item.get('reference_description')
            rows.append(row)
    frame = pd.DataFrame(rows)
    corpus = {}
    similarity_columns = ['rouge1', 'rouge2', 'rougeL', 'sentence_bleu', 'chrf_plus_plus', 'bertscore_f1']
    for column in similarity_columns:
        frame[column] = np.nan
    for method, group in frame[frame['method'] != 'reference'].groupby('method'):
        group = group[group['_reference'].notna()]
        if group.empty:
            continue
        sim = reference_similarity(group['_candidate'].tolist(), group['_reference'].tolist(), str(group['lang'].iloc[0]), fast=fast)
        corpus[method] = sim['corpus_bleu']
        for index, result in zip(group.index, sim['per_item']):
            for name, value in result.items():
                frame.loc[index, name] = value
    frame = frame.drop(columns=['_candidate', '_reference'])
    passrates = []
    for method, group in frame.groupby('method'):
        for criterion in SEO_LABELS:
            data = group['seo_' + criterion].dropna()
            passrates.append({'method': method, 'kind': 'seo', 'criterion': criterion, 'source': SEO_LABELS[criterion], 'passed': int(data.sum()), 'applicable': len(data), 'pass_rate': float(data.mean()) if len(data) else None})
    emails = pd.DataFrame(email_observations)
    if not emails.empty:
        for method, group in emails.groupby('method'):
            for criterion in EMAIL_LABELS:
                data = group[criterion].dropna()
                passrates.append({'method': method, 'kind': 'email', 'criterion': criterion, 'source': EMAIL_LABELS[criterion], 'passed': int(data.sum()), 'applicable': len(data), 'pass_rate': float(data.mean()) if len(data) else None})
                per_item = group.groupby('item_id')[criterion].mean()
                for index in frame.index[frame['method'] == method]:
                    frame.loc[index, 'email_' + criterion] = per_item.get(frame.loc[index, 'item_id'], np.nan)
    rates = pd.DataFrame(passrates)
    for criterion in SEO_LABELS:
        column = 'seo_' + criterion
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    metric_columns = [c for c in frame.columns if c.startswith(('seo_', 'email_', 'judge_', 'script_')) or c in similarity_columns + ['keyword_coverage', 'hallucination_rate', 'latency_s']]
    metric_columns = [c for c in metric_columns if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])]
    summary = {'dataset': dataset, 'input_files': [str(x) for x in paths], 'fast': fast, 'judge_backend': None if no_judge else backend, 'bootstrap': {'replicates': 1000, 'seed': 42, 'method': 'percentile'}, 'corpus_bleu_13a': corpus, 'metrics': {}, 'wilcoxon_holm': paired_tests(frame, metric_columns)}
    for method, group in frame.groupby('method'):
        summary['metrics'][method] = {col: bootstrap(group[col]) for col in metric_columns}
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / f'eval_per_item_{dataset}.csv', index=False, encoding='utf-8-sig')
    rates.to_csv(RESULTS / f'eval_checklist_passrates_{dataset}.csv', index=False, encoding='utf-8-sig')
    save_json(RESULTS / f'eval_summary_{dataset}.json', summary)
    setup_plot()
    source = 'Nguồn: dữ liệu đầu vào ' + ', '.join(p.name for p in paths) + '; khoảng tin cậy bootstrap 95%.'
    plot_bar(frame, ['rouge1', 'rouge2', 'rougeL'], FIGURES / f'eval_01_reference_similarity_{dataset}.png', 'Tương đồng tham chiếu (ROUGE F1)', source)
    plot_heatmap(rates, 'seo', SEO_LABELS, FIGURES / f'eval_02_seo_checklist_heatmap_{dataset}.png', 'Checklist SEO theo phương pháp', source)
    plot_heatmap(rates, 'email', EMAIL_LABELS, FIGURES / f'eval_03_email_checklist_{dataset}.png', 'Checklist email theo phương pháp', source)
    plot_bar(frame, [c for c in metric_columns if c.startswith('judge_')], FIGURES / f'eval_04_judge_scores_{dataset}.png', 'Điểm giám khảo', source)
    plot_bar(frame, ['hallucination_rate'], FIGURES / f'eval_05_hallucination_{dataset}.png', 'Tỷ lệ claim không có trong brief', source)
    plot_bar(frame, ['latency_s'], FIGURES / f'eval_06_latency_{dataset}.png', 'Độ trễ (giây)', source)
    return summary


def main():
    """Đọc tham số dòng lệnh và chạy đánh giá."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default=str(RESULTS))
    parser.add_argument('--dataset', default='vi')
    parser.add_argument('--no-judge', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--judge-backend', choices=['gemini', 'gpt'], default='gemini')
    args = parser.parse_args()
    summary = evaluate(args.inputs, args.dataset, args.fast, args.no_judge, args.judge_backend)
    print('Đã đánh giá:', ', '.join(summary['metrics']))


if __name__ == '__main__':
    main()

"""Tạo bảng và hình cho bài báo v2 từ file kết quả (không gõ số tay, kiểm toán A44)."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results_v2'
T = ROOT / 'paper_v2/tables'
FIG = ROOT / 'figures'


def vn(x, d=1):
    return f'{x:,.{d}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def save(name, obj):
    T.mkdir(parents=True, exist_ok=True)
    (T / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def data_tables():
    funnel = {x['step']: x['n'] for x in json.loads((R / 'funnel.json').read_text(encoding='utf-8'))}
    rows = list(csv.DictReader((ROOT / 'data_v2/processed/products_clean.csv').open(encoding='utf-8-sig')))
    split = Counter(r['split'] for r in rows)
    manifest = json.loads((ROOT / 'data_v2/raw/tiki_manifest.json').read_text(encoding='utf-8'))
    save('data_v2', {
        'caption': 'Dữ liệu Tiki sau từng bước', 'source': 'Nguồn: thu thập qua API công khai của Tiki ngày 22/11/2026',
        'columns': ['Bước', 'Số lượng'],
        'rows': [['Danh mục lá trong khung lấy mẫu', '88'],
                 ['Sản phẩm trong khung (3 cách sắp xếp, tối đa 35 mỗi cách)', vn(funnel['input'], 0)],
                 ['Sau khi bỏ sản phẩm đăng trước 20/11/2019', vn(funnel['created_date'], 0)],
                 ['Sau khi bỏ mô tả dưới 50 ký tự (đã cắt văn mẫu)', vn(funnel['clean_description'], 0)],
                 ['Sau khi bỏ trùng tên cùng thương hiệu', vn(funnel['deduplicated'], 0)],
                 ['Tập huấn luyện / phát triển / kiểm tra', f"{vn(split['train'], 0)} / {split['dev']} / {split['test']}"],
                 ['Loại vì gần trùng sản phẩm đã vào tập phát triển hoặc kiểm tra', str(split['excluded'])],
                 ['Đánh giá 1 đến 3 sao thu được', vn(manifest['files']['tiki_reviews_lowstar_raw.jsonl']['records'], 0)]],
        'col_widths_cm': [11, 4]})
    rq = json.loads((R / 'rq1_descriptive.json').read_text(encoding='utf-8'))['overall']

    def pct(k):
        v = rq[k]
        return f"{vn(v['rate']*100)}% ({vn(v['wilson_95'][0]*100)} đến {vn(v['wilson_95'][1]*100)})"
    save('rq1_v2', {
        'caption': 'Nội dung người bán so với khuyến nghị đã công bố', 'source': f"Nguồn: {vn(rq['n'],0)} sản phẩm Tiki sau phễu lọc; trong ngoặc là khoảng tin cậy Wilson 95%",
        'columns': ['Chỉ báo', 'Tỷ lệ sản phẩm'],
        'rows': [['Mô tả ngắn hơn 200 từ (Yoast; Tiki Academy)', pct('description_under_200_words')],
                 ['Mô tả có nhắc kích cỡ hoặc số đo', pct('size')],
                 ['Mô tả có nhắc chất liệu', pct('material')],
                 ['Mô tả có hướng dẫn giặt, bảo quản', pct('care')],
                 ['Tên dài 15 đến 50 ký tự (Tiki Academy)', pct('title_15_to_50_chars')],
                 ['Trung vị số từ của mô tả / số ký tự của tên', f"{rq['median_description_words']} từ / {rq['median_title_chars']} ký tự"]],
        'col_widths_cm': [9, 6]})


def themes_figure():
    s = json.loads((R / 'review_themes_llm_summary.json').read_text(encoding='utf-8'))
    names = {'size': 'Size, form không vừa', 'material': 'Chất liệu kém', 'wrong_item': 'Giao sai hàng',
             'mismatch': 'Không giống hình, mô tả', 'defect': 'Đường may, lỗi', 'value': 'Không đáng tiền',
             'color': 'Màu khác ảnh', 'delivery': 'Giao hàng, đóng gói'}
    items = sorted(s['themes'].items(), key=lambda kv: kv[1]['rate'])
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    vals = [v['rate'] * 100 for _, v in items]
    ax.barh([names[k] for k, _ in items], vals, color='#0E7C86')
    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, vn(v) + '%', va='center', fontsize=9)
    ax.set_xlabel('% đánh giá 1 đến 3 sao (n = ' + str(s['n_classified']) + ')')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / 'rq1_themes_v2.png')


LABEL = {'M0': 'M0 mẫu câu', 'M1': 'M1 tối giản', 'M2': 'M2 vòng 1', 'M2noObj': 'M2 bỏ lo ngại', 'M2ex': 'M2 thêm ví dụ',
         'M3': 'M3 vòng 2', 'M4a': 'M4a Qwen gốc*', 'M4b': 'M4b Qwen LoRA*'}
# Hình và bảng chính chỉ gồm các cấu hình chính; hai phép thử phụ của M2 được nêu trong văn bản (mục 5.5).
ORDER = ['M0', 'M1', 'M2', 'M3', 'M4a', 'M4b']


def result_assets():
    s = json.loads((R / 'eval_summary.json').read_text(encoding='utf-8'))
    # Bảng E1: làm theo lời dặn
    crit = [('desc_words_ge_200', 'Mô tả từ 200 từ'), ('description_bullets_ge3', 'Từ 3 gạch đầu dòng'),
            ('keyphrase_in_first_sentence', 'Cụm chính ở câu đầu'), ('title_len_50_60', 'Tên 50 đến 60 ký tự'),
            ('meta_len_120_156', 'Meta 120 đến 156 ký tự'), ('email_le9w_le60c', 'Email ≤ 9 từ, ≤ 60 ký tự'),
            ('script_words_120_200', 'Kịch bản 120 đến 200 từ')]
    cols = ['SELLER', 'M0', 'M1', 'M2', 'M3']

    def cell(m, k):
        v = s['E1'][m]['criteria'].get(k)
        return 'không áp dụng' if not v or v['rate'] is None else vn(v['rate'] * 100) + '%'
    rows = [[name] + [cell(m, k) for m in cols] for k, name in crit]
    rows.append(['Đầu ra hợp lệ'] + ['không áp dụng'] + [vn(s['E1'][m]['valid_output']['rate'] * 100) + '%' for m in cols[1:]])
    save('e1_v2', {'caption': 'Tỷ lệ đạt các ngưỡng đã ghi trong câu lệnh M2 và M3 (chỉ số làm theo lời dặn)',
                   'source': 'Nguồn: 120 sản phẩm kiểm tra; người bán không được dặn các ngưỡng này; "từ" là đơn vị tách bằng khoảng trắng',
                   'columns': ['Tiêu chí', 'Người bán', 'M0', 'M1', 'M2', 'M3'], 'rows': rows,
                   'col_widths_cm': [5.2, 2.2, 2, 2, 2, 2]})
    # Bảng và hình E2
    rows, fig_rows = [], []
    for j, jname in (('gemini', 'Gemini'), ('gpt', 'GPT')):
        crit_overall = s['E2'][j]['criteria']['overall']
        for m in ORDER:
            v = crit_overall.get(m)
            if not v:
                continue
            w = v['win_non_ties']
            ci = f"{vn(w['rate']*100)}% ({vn(w['wilson95'][0]*100)} đến {vn(w['wilson95'][1]*100)})" if w['rate'] is not None else 'không áp dụng'
            p = v.get('p_holm')
            ptxt = '< 0,001' if p is not None and p < 0.001 else (vn(p, 3) if p is not None else '')
            rows.append([jname, LABEL[m], str(v['n']), vn(v['win']*100, 0) + '%', vn(v['tie']*100, 0) + '%', vn(v['loss']*100, 0) + '%', ci, ptxt])
            fig_rows.append((jname, m, v['win'], v['tie'], v['loss']))
    save('e2_v2', {'caption': 'So mù từng cấu hình với trang người bán (thắng, hòa, thua theo sản phẩm)',
                   'source': 'Nguồn: hai lượt đảo vị trí mỗi cặp; tỷ lệ thắng tính trong các cặp không hòa, khoảng tin cậy Wilson 95%; p là kiểm định dấu hai phía, hiệu chỉnh Holm; * M4 chạy trên mẫu con 40 sản phẩm',
                   'columns': ['Giám khảo', 'Cấu hình', 'n', 'Thắng', 'Hòa', 'Thua', 'Tỷ lệ thắng (95% CI)', 'p Holm'],
                   'rows': rows, 'col_widths_cm': [1.9, 3.2, 1.0, 1.4, 1.3, 1.4, 4.2, 1.6]})
    # Cùng một thứ tự cấu hình cho cả hai khung; cấu hình không được chấm ghi rõ, không để trống.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), dpi=300, sharey=True)
    for ax, j in zip(axes, ('Gemini', 'GPT')):
        got = {r[1]: r for r in fig_rows if r[0] == j}
        for i, m in enumerate(ORDER):
            if m not in got:
                ax.text(2, i, 'không được chấm', va='center', fontsize=8, color='#5B6B7C', style='italic')
                continue
            a, b, c = got[m][2] * 100, got[m][3] * 100, got[m][4] * 100
            ax.barh(i, a, color='#0E7C86', label='Thắng người bán' if i == 0 else None)
            ax.barh(i, b, left=a, color='#C9D6DF', label='Hòa' if i == 0 else None)
            ax.barh(i, c, left=a + b, color='#B03A2E', label='Thua người bán' if i == 0 else None)
            if a >= 6: ax.text(a / 2, i, vn(a, 0), ha='center', va='center', color='white', fontsize=8)
            if c >= 6: ax.text(a + b + c / 2, i, vn(c, 0), ha='center', va='center', color='white', fontsize=8)
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LABEL[m] for m in ORDER], fontsize=9)
        ax.set_xlim(0, 100); ax.set_title('Giám khảo ' + j, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False); ax.set_xlabel('% sản phẩm')
    axes[0].legend(loc='lower center', bbox_to_anchor=(1.05, -0.38), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.14)
    fig.savefig(FIG / 'e2_v2.png', bbox_inches='tight')


if __name__ == '__main__':
    import sys
    data_tables()
    themes_figure()
    if len(sys.argv) > 1 and sys.argv[1] == 'results':
        result_assets()
    print('ok')

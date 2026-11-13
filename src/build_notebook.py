"""Tạo notebook nghiên cứu có thể chạy cục bộ hoặc trên Colab."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nb.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nb.v4.new_code_cell(text.strip()))

md('''# Sinh nội dung marketing thời trang nữ bằng GenAI
**Nguyễn Quang Minh** · Lớp HKC2026-CDNK-26C1EXT60901501 · GV: TS. Võ Văn Hải

Nghiên cứu dùng CRISP-DM để khảo sát đặc trưng nội dung liên quan đến bán chạy, so sánh các phương pháp sinh và đánh giá nội dung. Các quan hệ quan sát không chứng minh nhân quả.

**Liên kết:** mã nguồn https://github.com/minhnguyen66690-source/genai-marketing-thoi-trang-nu; website demo: WEB_URL (điền sau khi triển khai).

**Mục lục:** 1. Cài đặt · 2. Thu thập · 3. Tiền xử lý · 4. EDA · 5. Mô tả · 6. RQ1 · 7. Từ khóa · 8. Phản đối · 9. Sinh nội dung · 10. LoRA · 11. Đánh giá · 12. Đề xuất · 13. Công cụ AI · 14. Tài liệu.''')
md('''## 1. Cài đặt
Các cờ mặc định tắt vì thu thập, sinh, tinh chỉnh và chấm tốn thời gian, API hoặc GPU. Khi tắt, notebook đọc kết quả công khai đã lưu. Trên Colab cần điền URL repo trước khi chạy.''')
code('''import os, sys, json, subprocess
from pathlib import Path
IN_COLAB = 'google.colab' in sys.modules
REPO_URL = 'https://github.com/minhnguyen66690-source/genai-marketing-thoi-trang-nu'
WEB_URL = 'https://genai-thoitrangnu.streamlit.app'
if IN_COLAB:
    if not REPO_URL:
        raise ValueError('Hãy điền REPO_URL trước khi chạy trên Colab.')
    repo_dir = Path('/content') / REPO_URL.rstrip('/').split('/')[-1].removesuffix('.git')
    if not repo_dir.exists():
        subprocess.run(['git', 'clone', REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
    ROOT = repo_dir
    from google.colab import userdata
    for key in ('GEMINI_API_KEY', 'OPENAI_API_KEY'):
        try:
            value = userdata.get(key)
            if value: os.environ[key] = value
        except Exception:
            pass  # Không in khóa hoặc lỗi có thể chứa khóa.
else:
    here = Path.cwd().resolve()
    ROOT = here.parent if here.name == 'notebook' else here
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
RUN_CRAWL = RUN_GENERATION = RUN_FINETUNE = RUN_JUDGE = False
N_ITER = 2  # Tăng số lần tìm kiếm khi chạy nghiên cứu đầy đủ.
PUBLIC = ROOT / 'results/public'
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display
def read_json(path):
    if not path.exists():
        print('chưa có kết quả:', path.name)
        return {}
    with open(path, encoding='utf-8-sig') as f: return json.load(f)''')
md('''## 2. Thu thập dữ liệu — hiểu dữ liệu
Giới hạn thu thập Tiki ở một yêu cầu mỗi giây, dừng khi bị chặn; review được ẩn danh và không xuất nội dung gốc. Dữ liệu Amazon Reviews 2023 là nguồn đối chiếu công khai.''')
code('''# Chỉ thu thập khi đã chủ động bật cờ.
if RUN_CRAWL:
    subprocess.run([sys.executable, 'src/crawl_tiki.py', '--smoke'], check=True)
    subprocess.run([sys.executable, 'src/fetch_amazon2023.py', '--smoke'], check=True)
feature_path = ROOT / 'data/processed/tiki_features.csv'
if IN_COLAB or not feature_path.exists(): feature_path = ROOT / 'data/public/tiki_features_public.csv'
df = pd.read_csv(feature_path) if feature_path.exists() else pd.DataFrame()
print('Nguồn đặc trưng:', feature_path.relative_to(ROOT), '| số dòng:', len(df))''')
md('''## 3. Tiền xử lý và phễu lọc
Chuẩn hóa văn bản, loại mô tả quá ngắn và bản ghi trùng; nhãn bán chạy được xếp trong từng danh mục. Bảng dưới đây lấy từ kết quả đã lưu.''')
code('''funnel = read_json(PUBLIC / 'data_funnel.json')
stages = ['raw_valid', 'within_7y', 'description_at_least_50', 'deduplicated']
display(pd.DataFrame([{'Bước': k, 'Số sản phẩm': funnel[k]} for k in stages if k in funnel]))''')
md('''## 4. Phân tích khám phá
Biểu đồ tỷ lệ dùng khoảng tin cậy Wilson 95%; nhóm số gạch đầu dòng được phân chia thành không có và có ít nhất một.''')
code('''if not df.empty:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(np.log1p(pd.to_numeric(df.quantity_sold, errors='coerce').clip(lower=0)), ax=ax[0])
    ax[0].set(title='Phân phối log1p số đã bán', xlabel='log1p số đã bán')
    df.category_l2.value_counts().plot.barh(ax=ax[1], title='Sản phẩm theo danh mục')
    plt.tight_layout(); plt.show()
else: print('chưa có kết quả: dữ liệu đặc trưng')''')
code('''from statsmodels.stats.proportion import proportion_confint
if not df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in zip(axes, ['n_bullets', 'has_size_info'], ['Có gạch đầu dòng', 'Có thông tin size']):
        group = df.assign(nhom=pd.to_numeric(df[col], errors='coerce').fillna(0).gt(0)).groupby('nhom').best_seller.agg(['sum', 'count'])
        p = group['sum'] / group['count']
        lo, hi = proportion_confint(group['sum'], group['count'], method='wilson')
        ax.errorbar(range(len(p)), p, yerr=[p-lo, hi-p], fmt='o', capsize=4)
        ax.set_xticks(range(len(p)), ['Có' if x else 'Không' for x in group.index]); ax.set(title=label, ylabel='Tỷ lệ bán chạy', ylim=(0, 1))
    plt.tight_layout(); plt.show()
    cols = ['price', 'discount_rate', 'product_age_days', 'quantity_sold', 'n_bullets', 'desc_len_words', 'has_size_info', 'best_seller']
    plt.figure(figsize=(9, 6)); sns.heatmap(df[cols].corr(method='spearman', numeric_only=True), annot=True, cmap='coolwarm', center=0); plt.title('Tương quan Spearman'); plt.show()''')
md('''## 5. Phân tích mô tả
Các biến chính được tóm tắt theo phân phối và trung vị theo nhãn bán chạy.''')
code('''if not df.empty:
    main_cols = ['price', 'discount_rate', 'quantity_sold', 'product_age_days', 'desc_len_words', 'n_bullets']
    display(df[main_cols].describe().T)
    display(df.groupby('best_seller')[main_cols].median())''')
md('''## 6. Dự báo RQ1
Chia train/test phân tầng, chọn siêu tham số bằng CV trên train. Không đưa số đã bán, số đánh giá hoặc điểm đánh giá vào mô hình. Ablation giữ cùng thuật toán; chọn theo AUC CV, không chọn theo test.''')
code('''from predictive import prepare, pipeline, CONTROL, CONTENT
# common.py chọn backend ghi ảnh; khôi phục backend inline để hiện biểu đồ notebook.
plt.switch_backend('module://matplotlib_inline.backend_inline')
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from scipy.stats import loguniform, randint
scores, fitted = [], {}
if not df.empty and df.best_seller.nunique() == 2 and df.best_seller.value_counts().min() >= 7:
    d = prepare(df); y = d.best_seller.astype(int)
    X = d[CONTROL + CONTENT + ['category_l2']]
    xt, xv, yt, yv = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=min(3, int(yt.value_counts().min())), shuffle=True, random_state=42)
    models = {
        'Logistic': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'model__C': loguniform(.01, 10)}),
        'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1), {'model__n_estimators': randint(30, 80), 'model__max_depth': [4, 8, None]}),
        'XGBoost': (XGBClassifier(n_jobs=1, random_state=42, eval_metric='logloss'), {'model__n_estimators': randint(30, 80), 'model__max_depth': randint(2, 5)})}
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (est, params) in models.items():
        search = RandomizedSearchCV(pipeline(est, CONTROL + CONTENT), params, n_iter=N_ITER, scoring='roc_auc', cv=cv, random_state=42, n_jobs=1)
        search.fit(xt, yt); fitted[name] = search
        prob = search.predict_proba(xv)[:, 1]
        scores.append({'Mô hình': name, 'AUC CV': search.best_score_, 'ROC-AUC test': roc_auc_score(yv, prob), 'PR-AUC': average_precision_score(yv, prob), 'F1': f1_score(yv, search.predict(xv), zero_division=0)})
        fpr, tpr, _ = roc_curve(yv, prob); ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], '--'); ax.set(xlabel='Dương tính giả', ylabel='Dương tính thật', title='ROC trên test'); ax.legend(); plt.show()
    display(pd.DataFrame(scores))
else: print('chưa có kết quả: không đủ dữ liệu hai lớp')''')
code('''if fitted:
    best = max(fitted, key=lambda name: fitted[name].best_score_)
    est, params = models[best]
    control = RandomizedSearchCV(pipeline(est, CONTROL), params, n_iter=N_ITER, scoring='roc_auc', cv=cv, random_state=42, n_jobs=1).fit(xt, yt)
    display(pd.DataFrame([{'Mô hình chọn theo CV': best, 'Chỉ kiểm soát': roc_auc_score(yv, control.predict_proba(xv)[:, 1]), 'Thêm nội dung': roc_auc_score(yv, fitted[best].predict_proba(xv)[:, 1])}]))
    imp = permutation_importance(fitted[best], xv, yv, scoring='roc_auc', n_repeats=3, random_state=42, n_jobs=1)
    display(pd.DataFrame({'Biến': X.columns, 'Giảm AUC': imp.importances_mean}).sort_values('Giảm AUC', ascending=False).head(15))''')
code('''# OR là liên hệ có kiểm soát, không phải hiệu ứng nhân quả.
if not df.empty:
    try:
        import statsmodels.api as sm
        from statsmodels.stats.multitest import multipletests
        from rq1_inference import build_design, CONTENT_CONTINUOUS, CONTENT_BINARY
        design = build_design(df)
        fit = sm.Logit(df.best_seller.astype(int), design).fit(disp=False, maxiter=200)
        ci = fit.conf_int()
        table = pd.DataFrame({'OR': np.exp(fit.params), 'CI thấp': np.exp(ci[0]), 'CI cao': np.exp(ci[1]), 'p': fit.pvalues})
        # Hiệu chỉnh Holm cho họ kiểm định gồm 9 đặc trưng nội dung.
        content = [c for c in table.index if c in CONTENT_CONTINUOUS or c in CONTENT_BINARY]
        table.loc[content, 'p Holm'] = multipletests(table.loc[content, 'p'], method='holm')[1]
        display(table)
    except (ValueError, np.linalg.LinAlgError) as exc:
        print('chưa có kết quả Logit:', type(exc).__name__)''')
md('''## 7. Từ khóa đặc trưng
Thuật toán 1 tách cụm từ trong tên, loại từ chung và mã hàng, đếm tần suất theo sản phẩm, chọn cụm đủ phổ biến và cụ thể; tách train/test trước khi dùng trong sinh. Dữ liệu công khai chỉ chứa từ khóa tổng hợp.''')
code('''if not IN_COLAB and 'name' in df.columns:
    from keywords import name_segments, candidate_phrases, select_keywords
    from collections import Counter
    phrases = [candidate_phrases(name_segments(r['name'], r.get('brand', ''), 'vi'), 'vi') for _, r in df.iterrows()]
    counts = Counter(p for group in phrases for p in group)
    chosen = [select_keywords(group, counts, len(df), 'vi') for group in phrases]
    keyword_result = {}
    for cat, indices in df.reset_index(drop=True).groupby('category_l2').groups.items():
        top = Counter(term for i in indices for term in chosen[i]).most_common(5)
        keyword_result[cat] = {'top_keywords': [{'keyword': term, 'count': count} for term, count in top]}
    print('Đã trích lại từ tên sản phẩm cục bộ.')
else:
    keyword_result = read_json(PUBLIC / 'keywords_by_category.json').get('tiki', {})
categories = keyword_result.get('categories', keyword_result)
keyword_rows = [{'Danh mục': cat, **item} for cat, obj in categories.items() if isinstance(obj, dict) for item in obj.get('top_keywords', [])[:5]]
display(pd.DataFrame(keyword_rows))''')
md('''## 8. Phản đối của khách hàng
Bảng là số đếm theo chủ đề, không chứa nội dung review.''')
code('''obj = read_json(ROOT / 'data/public/objection_themes_summary.json')
display(pd.DataFrame.from_dict(obj.get('by_category', {}), orient='index'))''')
md('''## 9. Sinh nội dung
M1 zero-shot; M2 thêm ràng buộc; M3 thêm ví dụ truy hồi. Ví dụ minh họa dưới đây là dữ liệu giả lập; kết quả đã sinh được đọc riêng từ tệp công khai.''')
code('''from prompts import zero_shot, engineered, fewshot_retrieval
brief = {'item_id': 'demo', 'category': 'Áo nữ', 'keywords': ['áo linen'], 'attributes': {'Chất liệu': 'linen'}, 'objections': []}
for name, prompt in [('M1', zero_shot([brief])), ('M2', engineered([brief])), ('M3', fewshot_retrieval([brief], {'demo': [{'brief': brief, 'description': 'Áo linen có phom dáng đơn giản.'}]}))]:
    print('\\n', name, '\\n', prompt)
if RUN_GENERATION:
    from llm_api import generate_via_api
    key = os.environ.get('GEMINI_API_KEY')
    if key:
        inputs = pd.read_csv(ROOT / 'data/public/gen_inputs_tiki_public.csv').head(2)
        for _, row in inputs.iterrows():
            b = {'item_id': str(row.product_id), 'category': row.category_l2, 'keywords': json.loads(row.keywords), 'attributes': json.loads(row.attributes)}
            print(generate_via_api(engineered([b]), 'gemini', key)[0])
    else: print('chưa có kết quả: thiếu GEMINI_API_KEY')''')
code('''def show_generations(methods):
    for method in methods:
        paths = sorted((PUBLIC / 'generations').glob('generations_' + method + '_*.jsonl'))
        if not paths:
            print(method, ': chưa có kết quả'); continue
        path = next((p for p in paths if p.name.endswith('_tiki.jsonl')), paths[0])
        with open(path, encoding='utf-8-sig') as f:
            records = [json.loads(line) for line in f if line.strip()][:2]
        print(method, '—', path.name)
        for record in records: display(record.get('output', {}))
show_generations(['M0', 'M1', 'M2', 'M3'])''')
md('''## 10. Tinh chỉnh LoRA
Mặc định dùng tập Amazon đã tiền xử lý và GPU; quá trình có thể tải mô hình và cần thời gian. Khi chưa chạy, xem log và đầu ra đã lưu.''')
code('''import torch
if RUN_FINETUNE and torch.cuda.is_available():
    subprocess.run([sys.executable, 'src/finetune_lora.py', '--dataset', 'amazon', '--train', '--generate'], check=True)
elif RUN_FINETUNE: print('chưa có kết quả: cần GPU')
loss_path = PUBLIC / 'finetune_tiki.csv'
if loss_path.exists():
    loss = pd.read_csv(loss_path)
    if not loss.empty:
        sns.lineplot(data=loss, x='step', y='loss', hue='kind'); plt.title('Loss tinh chỉnh Tiki'); plt.show()
else: print('chưa có kết quả: loss')
show_generations(['M4a', 'M4b'])''')
md('''## 11. Đánh giá
ROUGE-L, chrF++ và BERTScore so với tham chiếu; checklist và tỷ lệ thuộc tính không có căn cứ là thước đo theo luật, không thay thế kiểm duyệt. Các ngưỡng S../E.. ánh xạ trong `src/evaluate.py` và `_workspace/01_research/verified_refs.md`; proxy được ghi rõ trong mã. Wilcoxon đã hiệu chỉnh Holm lấy từ kết quả lưu.''')
code('''if RUN_JUDGE:
    print('Để chấm lại, chạy src/evaluate.py với dữ liệu đầu vào và backend phù hợp; cần API.')
eval_files = sorted(PUBLIC.glob('eval_per_item_*.csv'))
summary_files = sorted(PUBLIC.glob('eval_summary_*.json'))
evaluation = pd.concat([pd.read_csv(p).assign(nguon=p.name) for p in eval_files], ignore_index=True) if eval_files else pd.DataFrame()
if evaluation.empty: print('chưa có kết quả: đánh giá theo sản phẩm')
else:
    metrics = [c for c in ['rougeL', 'chrf_plus_plus', 'bertscore_f1', 'seo_compliance_rate', 'hallucination_rate', 'latency_s'] if c in evaluation]
    metrics += [c for c in evaluation if c.startswith(('email_', 'judge_')) and pd.api.types.is_numeric_dtype(evaluation[c])]
    comparison = evaluation.groupby('method')[metrics].mean(numeric_only=True)
    display(comparison)
    if 'rougeL' in comparison:
        comparison['rougeL'].plot.bar(title='ROUGE-L trung bình'); plt.ylabel('ROUGE-L'); plt.show()
for path in summary_files:
    result = read_json(path)
    print(path.name, '— Wilcoxon/Holm'); display(pd.DataFrame(result.get('wilcoxon_holm', [])))
if not summary_files: print('chưa có kết quả: Wilcoxon')
from evaluate import SEO_LABELS, EMAIL_LABELS
display(pd.DataFrame([{'Loại': kind, 'Tiêu chí': k, 'Nguồn': v} for kind, labels in [('SEO', SEO_LABELS), ('Email', EMAIL_LABELS)] for k, v in labels.items()]))''')
md('''### 11b. Kiểm tra độ vững
Prompt v2 được sửa sau khi xem lỗi trên tập kiểm tra, nên M0, M2, M3 được chạy lại trên 40 sản phẩm của tập kiểm định chưa dùng khi sửa prompt. Bảng dưới so sánh cùng một bộ chỉ số trên hai tập; tiếp theo là tỷ lệ tuân thủ trên năm tiêu chí chung (bỏ hai tiêu chí meta mà nội dung người bán không có) và độ chính xác, độ phủ ROUGE-L.''')
code('''holdout = read_json(PUBLIC / 'holdout_check.json')
if holdout:
    display(pd.DataFrame([{'Tập': split, 'Cấu hình': m, **v} for split in ('val', 'test') for m, v in holdout.get(split, {}).items()]))
extra = read_json(PUBLIC / 'extra_checks.json')
if extra:
    display(pd.DataFrame(extra.get('seo_common_5', {})).rename_axis('Cấu hình'))
    display(pd.DataFrame(extra.get('rougeL_precision_recall', {}).get('tiki', {})).T)
else:
    print('chưa có kết quả kiểm tra độ vững')''')
md('''## 12. Phân tích đề xuất
Chọn cấu hình theo từng chỉ số thực đo. Độ trễ nhân 100 là ước tính tuần tự, không bao gồm kiểm duyệt; các chỉ số thiếu không được suy diễn.''')
code('''if not evaluation.empty:
    for metric, higher in [('rougeL', True), ('chrf_plus_plus', True), ('bertscore_f1', True), ('seo_compliance_rate', True), ('hallucination_rate', False), ('latency_s', False)]:
        if metric in comparison and comparison[metric].notna().any():
            s = comparison[metric].dropna(); method = s.idxmax() if higher else s.idxmin()
            print(metric, ':', method, '=', s.loc[method])
    if 'latency_s' in comparison: display((comparison['latency_s'] * 100).rename('Giây cho 100 sản phẩm, tuần tự'))
else: print('chưa có kết quả: chưa thể chọn cấu hình')
print('Quy trình: kiểm tra brief → sinh → người duyệt dữ kiện, giọng điệu và chính sách → chỉnh sửa → đăng.')''')
md('''## 13. Công cụ AI được sử dụng
| Công cụ | Vai trò được khai báo |
|---|---|
| Claude Opus 5.5 qua Claude Code | Điều phối, viết bài, kiểm chứng |
| GPT-6.1-Terra / GPT-6.1-Sol qua Codex CLI | Viết code, tra cứu, sinh nội dung M1–M3 |
| Gemini 3.8 Flash qua Antigravity CLI | Giám khảo LLM chính |
| GPT-5.6-Luna qua Codex CLI | Giám khảo thứ hai, chỉ dùng đo đồng thuận |
| Claude Haiku và Sonnet | Kiểm chứng trích dẫn, dựng slide |
| Qwen2.5-0.5B-Instruct | Mô hình tinh chỉnh LoRA |
| bert-base-multilingual-cased | BERTScore |

Vai trò ghi theo nhật ký thực hiện của tác giả; mọi số liệu được tính bằng mã nguồn trong kho, không lấy từ câu trả lời của mô hình.''')
md('''## 14. Tài liệu tham khảo chính
- Hou, Y., et al. (2024). *Bridging language and items for retrieval and recommendation*. Amazon Reviews 2023.
- Google. *Search Central: Creating helpful, reliable, people-first content*. https://developers.google.com/search/docs/fundamentals/creating-helpful-content
- Hu, E. J., et al. (2022). *LoRA: Low-rank adaptation of large language models*. https://arxiv.org/abs/2106.09685
- Zhang, T., et al. (2020). *BERTScore: Evaluating text generation with BERT*. https://arxiv.org/abs/1904.09675
- Nguồn chi tiết các tiêu chí: `_workspace/01_research/verified_refs.md`.''')

def main():
    path = ROOT / 'notebook/GenAI_Marketing_ThoiTrangNu.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}})
    with open(path, 'w', encoding='utf-8') as f: nb.write(notebook, f)
    print(f'Đã ghi {path}: {len(cells)} cell')

if __name__ == '__main__': main()

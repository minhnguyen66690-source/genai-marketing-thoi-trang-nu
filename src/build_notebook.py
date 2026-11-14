"""Tạo notebook nghiên cứu chạy được trên Google Colab hoặc máy cá nhân.

Mỗi ô code có một ô giải thích đứng trước theo cùng khung: làm gì, vì sao, đầu vào → đầu ra, đọc kết quả thế nào.
Notebook không in văn bản mô tả gốc của người bán hay nội dung đánh giá của khách; ví dụ minh họa đều tự viết.
Chạy: .venv/Scripts/python.exe src/build_notebook.py
"""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nb.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nb.v4.new_code_cell(text.strip()))


# =====================================================================================
md('''# Sinh nội dung sản phẩm và khuyến mãi tự động cho thời trang nữ
**Nguyễn Quang Minh** · Chuyên đề Ngoại khóa, lớp HKC2026-CDNK-26C1EXT60901501 · GV: TS. Võ Văn Hải · Đề tài 14

Notebook này đi lại toàn bộ nghiên cứu theo đúng thứ tự đã làm, mỗi ô code có giải thích đứng trước:

| Phần | Nội dung | Câu hỏi trả lời |
|---|---|---|
| 1 | Cài đặt môi trường | |
| 2 | **Lấy dữ liệu**: từ thư mục Data Source, từ GitHub, hoặc thu thập lại từ Tiki và Amazon | Dữ liệu đến từ đâu? |
| 3 | Xem dữ liệu thô | Trong một bản ghi Tiki có gì? |
| 4 | Tiền xử lý từng bước và gán nhãn bán chạy | Lọc bỏ cái gì, vì sao? |
| 5 | Trích từ khóa đặc trưng (Thuật toán 1) | "Vài từ khóa" lấy ở đâu ra? |
| 6 | Phản đối của khách từ đánh giá thấp | Khách hay chê gì? |
| 7–8 | Phân tích khám phá và mô tả | Dữ liệu trông thế nào? |
| 9 | Dự báo sản phẩm bán chạy (RQ1) | Đặc trưng nội dung nào gắn với bán chạy? |
| 10 | Sinh nội dung M0 đến M3 | Câu lệnh gửi cho mô hình trông thế nào? |
| 11 | Đánh giá (RQ2, RQ3, RQ4) | Nội dung sinh ra đạt chuẩn đến đâu? |
| 12 | Tinh chỉnh mô hình nhỏ (M4) | Mô hình chạy trên máy có dùng được không? |
| 13 | Tác động kinh doanh | Tốn bao nhiêu tiền, bao nhiêu thời gian? |
| 14 | Phân tích đề xuất | Shop nên làm gì? |

**Liên kết:** mã nguồn https://github.com/minhnguyen66690-source/genai-marketing-thoi-trang-nu · website demo https://genai-thoitrangnu.streamlit.app

**Cách chạy nhanh nhất:** Runtime → Run all. Các bước tốn thời gian hoặc cần khóa API (thu thập lại, gọi mô hình, chấm điểm, tinh chỉnh) đều có cờ bật/tắt ở ô 1.2 và mặc định tắt; khi tắt, notebook đọc kết quả đã lưu của lần chạy thật.''')

# =====================================================================================
md('''## 1. Cài đặt

### Ô 1.1: Tải mã nguồn và cài thư viện
- **Làm gì:** trên Colab, tải kho mã nguồn từ GitHub về `/content`, rồi cài các thư viện trong `requirements-colab.txt` (pandas, scikit-learn, xgboost, statsmodels, underthesea để tách từ tiếng Việt, rouge-score, sacrebleu, bert-score, transformers, peft...). Trên máy cá nhân thì bỏ qua bước tải.
- **Vì sao:** mọi bước phân tích đều gọi các hàm trong thư mục `src/`, nên notebook cần có mã nguồn bên cạnh.
- **Đầu ra:** biến `ROOT` trỏ tới thư mục dự án. Mất khoảng 1–2 phút trên Colab.''')
code('''import os, sys, json, subprocess
from pathlib import Path
IN_COLAB = 'google.colab' in sys.modules
REPO_URL = 'https://github.com/minhnguyen66690-source/genai-marketing-thoi-trang-nu'
if IN_COLAB:
    ROOT = Path('/content/genai-marketing-thoi-trang-nu')
    if not ROOT.exists():
        subprocess.run(['git', 'clone', '-q', REPO_URL, str(ROOT)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'requirements-colab.txt')], check=True)
else:
    here = Path.cwd().resolve()
    ROOT = here.parent if here.name == 'notebook' else here
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
print('Thư mục dự án:', ROOT)''')

md('''### Ô 1.2: Cờ điều khiển và hàm tiện ích
- **Làm gì:** đặt các cờ bật/tắt cho những bước tốn thời gian, nạp thư viện vẽ hình, và định nghĩa hàm `read_json` đọc tệp kết quả.
- **Các cờ:**
  - `RUN_CRAWL`: thu thập lại dữ liệu từ Tiki và Amazon (khoảng 1–2 giờ, cần mạng, có thể bị Tiki chặn giữa chừng).
  - `RUN_LIVE_API_DEMO`: gọi thử 3 yêu cầu tới API công khai của Tiki để xem dữ liệu trả về.
  - `RUN_PREPROCESS`: chạy lại tiền xử lý từ dữ liệu thô.
  - `RUN_SEARCH`: chạy lại tìm siêu tham số cho mô hình dự báo (khoảng 5–10 phút).
  - `RUN_GENERATION`: gọi mô hình ngôn ngữ để viết nội dung cho một sản phẩm (cần khóa Gemini, xem ô 10.4).
  - `RUN_FINETUNE`: tinh chỉnh LoRA (cần GPU, khoảng 30 phút trên GPU T4).
- **Đọc kết quả:** không in gì; các ô sau dùng các cờ này.''')
code('''RUN_CRAWL = False
RUN_LIVE_API_DEMO = False
RUN_PREPROCESS = False
RUN_SEARCH = False
RUN_GENERATION = False
RUN_FINETUNE = False
SEED = 42

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Markdown
pd.set_option('display.max_colwidth', 120)
plt.rcParams['figure.dpi'] = 110

def read_json(path):
    path = Path(path)
    if not path.exists():
        print('chưa có tệp:', path.name)
        return {}
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)''')

# =====================================================================================
md('''## 2. Lấy dữ liệu

Nghiên cứu dùng hai nguồn:

| Nguồn | Lấy bằng cách nào | Thời điểm | Số lượng |
|---|---|---|---|
| **Tiki**, danh mục Thời trang nữ (mã 931) | Gọi các API công khai mà chính website Tiki dùng để hiển thị trang: danh mục → danh sách → chi tiết sản phẩm → đánh giá | 13/11/2026 | 2.181 sản phẩm thô, 2.924 đánh giá |
| **Amazon Reviews 2023** (McAuley Lab, Hou và cộng sự, 2024) | Đọc tuần tự tệp siêu dữ liệu công bố cho nghiên cứu, lọc nhánh quần áo nữ | 13/11/2026 | 5.000 bản ghi đọc, 674 dùng được |

Notebook có ba cách để có dữ liệu, tự chọn theo thứ tự ưu tiên:

1. **Thư mục `Data Source` nộp kèm bài** (đầy đủ nhất): tải cả thư mục lên Google Drive tại `MyDrive/Data Source`. Ô 2.1 sẽ gắn Drive và chép vào đúng chỗ.
2. **Dữ liệu công khai trên GitHub** (có sẵn khi tải mã nguồn): chỉ có đặc trưng số, từ khóa và kết quả, không có văn bản mô tả gốc. Vẫn chạy được phần 4 đến 14, trừ các ô cần văn bản gốc.
3. **Thu thập lại từ đầu** (ô 2.4): chạy đúng các script đã dùng. Kết quả sẽ khác bản gốc vì sản phẩm trên Tiki thay đổi theo ngày.''')

md('''### Ô 2.1: Nạp dữ liệu từ thư mục Data Source
- **Làm gì:** tìm thư mục `Data Source`: trên Colab thì gắn Google Drive và tìm ở `MyDrive/Data Source`; trên máy cá nhân thì kiểm tra dữ liệu đã có sẵn trong `data/` chưa. Nếu tìm thấy, chép bốn thư mục con vào đúng vị trí mà các script trong `src/` đọc:

| Thư mục trong Data Source | Chép vào | Nội dung |
|---|---|---|
| `1_Du_lieu_tho` | `data/raw` | JSON thô đúng như lúc thu thập |
| `2_Du_lieu_da_xu_ly` | `data/processed` | Bảng đã làm sạch và tính đặc trưng |
| `3_Du_lieu_cong_khai` | `data/public` | Bản rút gọn không có văn bản gốc |
| `4_Ket_qua_thuc_nghiem` | `_workspace/03_results` | Đầu ra của mô hình, điểm chấm |

- **Đầu ra:** ba biến `HAS_RAW`, `HAS_PROCESSED` (có hay không có dữ liệu thô, dữ liệu đã xử lý) và `RES` (thư mục kết quả sẽ đọc). Các ô sau dựa vào đây để biết chạy được bước nào.''')
code('''import shutil
DATA_SOURCE_DIR = Path('/content/drive/MyDrive/Data Source')  # sửa nếu bạn đặt thư mục ở chỗ khác trên Drive
MAPPING = {'1_Du_lieu_tho': 'data/raw', '2_Du_lieu_da_xu_ly': 'data/processed',
           '3_Du_lieu_cong_khai': 'data/public', '4_Ket_qua_thuc_nghiem': '_workspace/03_results'}

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount('/content/drive')
    except Exception as exc:
        print('Không gắn được Google Drive:', type(exc).__name__)
    if DATA_SOURCE_DIR.exists():
        for src_name, dst_name in MAPPING.items():
            src, dst = DATA_SOURCE_DIR / src_name, ROOT / dst_name
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f'Đã chép {src_name} → {dst_name}')
    else:
        print('Không thấy', DATA_SOURCE_DIR, '→ dùng dữ liệu công khai trên GitHub')

HAS_RAW = (ROOT / 'data/raw/tiki_products_raw.jsonl').exists()
HAS_PROCESSED = (ROOT / 'data/processed/tiki_features.csv').exists()
RES = ROOT / '_workspace/03_results' if (ROOT / '_workspace/03_results/eval_per_item_tiki.csv').exists() else ROOT / 'results/public'
FEAT_PATH = ROOT / 'data/processed/tiki_features.csv' if HAS_PROCESSED else ROOT / 'data/public/tiki_features_public.csv'
INPUT_PATH = ROOT / 'data/processed/gen_inputs_tiki.csv' if HAS_PROCESSED else ROOT / 'data/public/gen_inputs_tiki_public.csv'
print('Có dữ liệu thô:', HAS_RAW, '| có dữ liệu đã xử lý:', HAS_PROCESSED)
print('Bảng đặc trưng dùng cho phân tích:', FEAT_PATH.relative_to(ROOT))
print('Thư mục kết quả:', RES.relative_to(ROOT))''')

md('''### Ô 2.2: Kiểm tra dữ liệu thô không bị sửa
- **Làm gì:** ngay sau khi thu thập, mỗi tệp thô được "khóa": tính mã băm SHA-256 và ghi vào tệp `.lock.json` cạnh nó. Ô này tính lại mã băm và so với mã đã ghi.
- **Vì sao:** SHA-256 thay đổi hoàn toàn nếu chỉ một ký tự trong tệp bị sửa. Hai mã khớp nhau chứng minh mọi phân tích phía sau đọc đúng dữ liệu đã thu thập ngày 13/11/2026, không ai chỉnh tay.
- **Đọc kết quả:** cột `khớp` phải là `True` cho cả ba tệp.''')
code('''import hashlib
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest().upper()

rows = []
for lock in sorted((ROOT / 'data/raw').glob('*.lock.json')):
    info = read_json(lock)
    data_file = lock.with_name(lock.name.replace('.lock.json', ''))
    if data_file.exists():
        rows.append({'tệp': data_file.name, 'MB': round(data_file.stat().st_size / 1e6, 1), 'khóa lúc': info.get('locked_at'),
                     'nguồn ghi khi khóa': info.get('source', '')[:90], 'khớp': sha256(data_file) == info['sha256'].upper()})
display(pd.DataFrame(rows)) if rows else print('Không có dữ liệu thô (đang dùng bản công khai), bỏ qua bước kiểm tra.')''')

md('''### Ô 2.3: API của Tiki trả về gì (gọi thử 3 yêu cầu)
Chương trình thu thập `src/crawl_tiki.py` gọi lần lượt bốn API công khai, tức chính các địa chỉ mà trình duyệt gọi khi bạn mở tiki.vn:

| Bước | Địa chỉ | Lấy được gì |
|---|---|---|
| 1 | `https://tiki.vn/api/v2/categories?parent_id=931` | Danh mục con của Thời trang nữ; lặp tới khi gặp danh mục lá |
| 2 | `https://tiki.vn/api/personalish/v1/blocks/listings?category=<id>&sort=top_seller` | Danh sách mã sản phẩm của một danh mục lá, theo hai cách sắp xếp (bán chạy và mặc định) |
| 3 | `https://tiki.vn/api/v2/products/<id>` | Chi tiết một sản phẩm: tên, giá, số đã bán, ngày đăng, mô tả HTML, thông số, ảnh, biến thể |
| 4 | `https://tiki.vn/api/v2/reviews?product_id=<id>&page=1&limit=20` | Trang đánh giá đầu tiên: số sao, nội dung, thời điểm |

**Quy tắc thu thập có đạo đức** (đã cài trong script): mỗi yêu cầu cách nhau 1–1,6 giây (đánh giá: 1,5–2,5 giây); gặp mã 429 thì dừng 60 giây; gặp CAPTCHA hay trang HTML thay vì JSON thì dừng hẳn, không tìm cách vượt. Tên, mã và ảnh của khách bị bỏ ngay khi ghi tệp.

- **Làm gì:** khi bật `RUN_LIVE_API_DEMO`, ô dưới gọi bước 1 và bước 3 cho một sản phẩm để bạn xem cấu trúc JSON thật.
- **Đọc kết quả:** danh sách tên danh mục con, rồi danh sách các trường của một sản phẩm cùng vài giá trị số.''')
code('''import time, requests
HEADERS = {'User-Agent': 'Mozilla/5.0 (nghien cuu hoc thuat; toc do thap)', 'Accept': 'application/json'}

def tiki_get(path, params=None):
    time.sleep(1.2)  # giữ tốc độ chậm như script thật
    r = requests.get('https://tiki.vn' + path, params=params, headers=HEADERS, timeout=20)
    if r.status_code != 200 or 'json' not in r.headers.get('content-type', ''):
        raise RuntimeError(f'Tiki không trả JSON (mã {r.status_code}); dừng, không vượt chặn.')
    return r.json()

if RUN_LIVE_API_DEMO:
    cats = tiki_get('/api/v2/categories', {'parent_id': 931}).get('data', [])
    print('Bước 1, danh mục con của Thời trang nữ:', [c.get('name') for c in cats][:12])
    pid = int(pd.read_csv(FEAT_PATH, usecols=['product_id']).product_id.iloc[0])
    item = tiki_get(f'/api/v2/products/{pid}')
    print('Bước 3, các trường của sản phẩm', pid, ':', sorted(item.keys())[:40], '...')
    print({k: item.get(k) for k in ['price', 'list_price', 'discount_rate', 'day_ago_created', 'review_count']})
else:
    print('Đang tắt RUN_LIVE_API_DEMO. Bật lên ở ô 1.2 để gọi thử API (cần mạng).')''')

md('''### Ô 2.4: Thu thập lại toàn bộ (tùy chọn)
- **Làm gì:** chạy đúng ba script đã dùng trong nghiên cứu, theo thứ tự:
  1. `crawl_tiki.py --target 3000`: duyệt mọi danh mục lá, lấy chi tiết sản phẩm đến khi đủ 3.000 hoặc bị chặn. Lần chạy gốc dừng ở 2.181 sản phẩm vì Tiki bắt đầu trả HTML (ghi trong `tiki_crawl_stop_note.json`).
  2. `crawl_tiki_reviews.py --top 500`: lấy trang đánh giá đầu tiên của 500 sản phẩm bán chạy nhất đã thu thập.
  3. `fetch_amazon2023.py --n 5000`: đọc dần tệp nén của Amazon Reviews 2023 trên máy chủ McAuley Lab, không tải cả tệp về; giữ 5.000 bản ghi quần áo nữ đầu tiên thỏa bộ lọc.
- **Vì sao để tắt:** mất 1–2 giờ, và dữ liệu Tiki hôm nay sẽ khác ngày 13/11/2026 nên số liệu sẽ lệch so với bài báo.
- **Đầu ra:** các tệp `data/raw/*.jsonl` mới.''')
code('''if RUN_CRAWL:
    for cmd in (['src/crawl_tiki.py', '--target', '3000'],
                ['src/crawl_tiki_reviews.py', '--top', '500'],
                ['src/fetch_amazon2023.py', '--n', '5000', '--out-dir', '.']):
        print('Chạy:', ' '.join(cmd))
        subprocess.run([sys.executable] + cmd, check=True)
    HAS_RAW = True
else:
    print('Đang tắt RUN_CRAWL; dùng dữ liệu đã thu thập ngày 13/11/2026.')''')

# =====================================================================================
md('''## 3. Xem dữ liệu thô

### Ô 3.1: Một bản ghi sản phẩm Tiki gồm những gì
- **Làm gì:** đọc tệp thô `tiki_products_raw.jsonl` (mỗi dòng là một sản phẩm dạng JSON), đếm số bản ghi, liệt kê các trường và cho biết nghiên cứu dùng trường nào vào việc gì.
- **Vì sao:** để thấy dữ liệu gốc phong phú hơn nhiều so với những gì được dùng, và hiểu mỗi đặc trưng ở phần sau đến từ trường nào.
- **Lưu ý:** không in tên hay mô tả sản phẩm vì đó là nội dung của người bán; chỉ in tên trường và giá trị số.''')
code('''USED_FIELDS = {
    'id': 'mã sản phẩm, dùng để nối các bảng',
    'name': 'tên sản phẩm → trích từ khóa (phần 5), đặc trưng độ dài tên',
    'description': 'mô tả HTML của người bán → làm sạch, tính đặc trưng nội dung, làm văn bản tham chiếu',
    'price / list_price / discount_rate': 'giá bán, giá niêm yết, % giảm → biến kiểm soát',
    'quantity_sold': 'số đã bán (cộng dồn) → tạo nhãn bán chạy',
    'day_ago_created': 'số ngày từ khi đăng → ngày đăng, tuổi sản phẩm, lọc 7 năm',
    'breadcrumbs / categories': 'đường dẫn danh mục → danh mục cấp 2 để so sánh trong cùng nhóm hàng',
    'images / video_url': 'số ảnh, có video → biến kiểm soát và đặc trưng trang',
    'specifications': 'bảng thông số (chất liệu, xuất xứ...) → thuộc tính đưa vào bản tóm tắt',
    'configurable_options': 'biến thể màu, size → số tùy chọn',
    'meta_description': 'mô tả meta → kiểm tra có hay không',
    'brand': 'thương hiệu → loại khỏi từ khóa, phát hiện trùng lặp',
}
if HAS_RAW:
    with open(ROOT / 'data/raw/tiki_products_raw.jsonl', encoding='utf-8') as f:
        first = json.loads(f.readline())
        n_raw = 1 + sum(1 for line in f if line.strip())
    print('Số bản ghi thô:', n_raw, '| số trường mỗi bản ghi:', len(first))
    print('Toàn bộ tên trường:', ', '.join(sorted(first.keys())))
    qs = first.get('quantity_sold')
    print('\\nVí dụ giá trị số của bản ghi đầu:', {'price': first.get('price'), 'discount_rate': first.get('discount_rate'),
          'day_ago_created': first.get('day_ago_created'), 'quantity_sold': qs, 'số ảnh': len(first.get('images') or []),
          'độ dài mô tả HTML (ký tự)': len(first.get('description') or '')})
display(pd.DataFrame([{'trường': k, 'dùng để': v} for k, v in USED_FIELDS.items()]))''')

md('''### Ô 3.2: Dữ liệu đánh giá của khách
- **Làm gì:** đọc `tiki_reviews_raw.jsonl`, đếm số đánh giá, số sản phẩm có đánh giá và phân bố số sao.
- **Vì sao:** chỉ đánh giá 1–3 sao được dùng để tìm "phản đối" của khách (phần 6), nên cần biết có bao nhiêu đánh giá thấp.
- **Đọc kết quả:** đa số đánh giá là 5 sao; phần 1–3 sao nhỏ nhưng là phần chứa lời chê.''')
code('''if HAS_RAW:
    rv = pd.read_json(ROOT / 'data/raw/tiki_reviews_raw.jsonl', lines=True)
    print('Số đánh giá:', len(rv), '| số sản phẩm có đánh giá:', rv.product_id.nunique(), '| các cột:', list(rv.columns))
    display(rv.rating.value_counts().sort_index().rename('số đánh giá').to_frame().T)
    print('Đánh giá 1–3 sao:', int((rv.rating <= 3).sum()))
else:
    print('Không có dữ liệu thô; số liệu tổng hợp ở phần 6.')''')

# =====================================================================================
md('''## 4. Tiền xử lý từng bước

Script `src/preprocess.py` biến 2.181 bản ghi thô thành bảng phân tích theo năm bước. Phần này giải thích từng bước rồi kiểm lại bằng số.

| Bước | Làm gì | Vì sao |
|---|---|---|
| 1 | Chuyển mô tả HTML thành chữ thường, chuẩn hóa Unicode NFC, gộp khoảng trắng | Bỏ thẻ HTML để đếm từ, câu, gạch đầu dòng cho đúng |
| 2 | Cắt đoạn "Giá sản phẩm trên Tiki đã bao gồm thuế..." ở cuối mô tả | Đoạn này do Tiki tự gắn vào **mọi** mô tả, không do người bán viết |
| 3 | Giữ sản phẩm đăng từ 20/11/2019 | Yêu cầu dữ liệu không cũ quá 7 năm |
| 4 | Bỏ mô tả dưới 50 ký tự | Quá ngắn để làm văn bản tham chiếu |
| 5 | Bỏ bản ghi trùng tên và thương hiệu | Một sản phẩm đăng nhiều lần sẽ làm lệch thống kê |

### Ô 4.1: Chạy tiền xử lý (tùy chọn) và xem phễu lọc
- **Làm gì:** nếu bật `RUN_PREPROCESS` và có dữ liệu thô, chạy lại `preprocess.py`, rồi `keywords.py` và `objections.py`. Sau đó đọc `data_funnel.json`, tệp ghi số sản phẩm còn lại sau mỗi bước.
- **Đọc kết quả:** cột "bị loại" cho biết mỗi bước bỏ bao nhiêu sản phẩm.''')
code('''if RUN_PREPROCESS and HAS_RAW:
    for script in ('src/preprocess.py', 'src/keywords.py', 'src/objections.py'):
        subprocess.run([sys.executable, script], check=True)
funnel = read_json(RES / 'data_funnel.json')
steps = [('raw_valid', 'Bản ghi thô hợp lệ'), ('within_7y', 'Đăng từ 20/11/2019'),
         ('description_at_least_50', 'Mô tả từ 50 ký tự'), ('deduplicated', 'Không trùng tên + thương hiệu')]
table = pd.DataFrame([{'bước': label, 'còn lại': funnel.get(key)} for key, label in steps])
table['bị loại'] = (-table['còn lại'].diff()).fillna(0).astype(int)
display(table)
print('Số danh mục cấp 2 dùng để xếp hạng:', funnel.get('ranking_groups'), '| sản phẩm bán chạy:', funnel.get('positive_count'),
      f"({100 * funnel.get('positive_rate', 0):.1f}%)")''')

md('''### Ô 4.2: Bước 1–2 trên một mô tả tự viết
- **Làm gì:** chạy hai hàm làm sạch thật của dự án (`clean_text`, `strip_tiki_boilerplate` trong `src/common.py`) trên một mô tả HTML **tự viết** để thấy trước và sau.
- **Đọc kết quả:** thẻ `<p>`, `<li>` biến mất; đoạn thông báo thuế ở cuối bị cắt.''')
code('''from common import clean_text, strip_tiki_boilerplate
%matplotlib inline
# common.py chọn chế độ vẽ không màn hình để script chạy nền; dòng trên bật lại hiển thị biểu đồ trong notebook.
html =('<p>Chân váy chữ A cạp cao, vải tuyết mưa dày dặn.</p><ul><li>Size S, M, L</li><li>Giặt tay nước lạnh</li></ul>'
        '<p>Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành. Bên cạnh đó, tuỳ vào loại sản phẩm...</p>')
step1 = clean_text(html)
step2 = strip_tiki_boilerplate(step1)
print('Bước 1, bỏ HTML:\\n', step1, '\\n')
print('Bước 2, cắt đoạn thông báo thuế:\\n', step2)''')

md('''### Ô 4.3: Đặc trưng nội dung của một mô tả
- **Làm gì:** gọi hàm `text_features` (trong `src/preprocess.py`), hàm đã tính đặc trưng cho cả 1.941 sản phẩm, trên mô tả tự viết ở ô trước.
- **Ý nghĩa từng đặc trưng:** độ dài tên (ký tự, từ), độ dài mô tả, số đoạn, số gạch đầu dòng, số câu, độ dài câu trung bình, số dấu chấm than, tỷ lệ chữ hoa, số emoji, và bốn cờ 0/1: có nhắc size, chất liệu, cách bảo quản, lời kêu gọi mua.
- **Đọc kết quả:** mỗi đặc trưng là một con số; phần 9 kiểm tra con số nào gắn với bán chạy.''')
code('''from preprocess import text_features
feat = text_features('Chân váy chữ A cạp cao công sở', step2)
display(pd.Series(feat, name='giá trị').to_frame())''')

md('''### Ô 4.4: Gán nhãn "bán chạy" và kiểm tra lại
- **Quy tắc gán nhãn:** sản phẩm là **bán chạy** (`best_seller = 1`) khi đã bán ít nhất 1 đơn **và** số đã bán nằm trong nhóm 25% cao nhất **của cùng danh mục cấp 2**. Danh mục có dưới 30 sản phẩm được gộp vào "Khác" trước khi xếp hạng.
- **Vì sao so trong cùng danh mục:** đồ lót bán nhiều đơn hơn áo khoác; so chung sẽ chỉ cho thấy "đồ lót bán chạy".
- **Hòa hạng:** sản phẩm có cùng số đã bán được xếp cùng hạng, nên tỷ lệ bán chạy thực tế (26,1%) hơi khác con số 25%.
- **Làm gì:** tự tính lại nhãn theo đúng quy tắc trên từ cột `quantity_sold` và `category_l2`, rồi so với nhãn đã lưu.
- **Đọc kết quả:** "trùng khớp" phải là 100%.''')
code('''feat_df = pd.read_csv(FEAT_PATH)
sold = pd.to_numeric(feat_df.quantity_sold, errors='coerce').fillna(0)
rank = sold.groupby(feat_df.category_l2).rank(method='min', ascending=False)  # hai sản phẩm bán bằng nhau được cùng hạng
size = feat_df.groupby('category_l2').category_l2.transform('size')
label = ((sold > 0) & (rank <= np.ceil(size * 0.25))).astype(int)
print('Số sản phẩm:', len(feat_df), '| bán chạy (tính lại):', int(label.sum()), f'({100 * label.mean():.1f}%)')
print('Trùng khớp với nhãn đã lưu:', f'{100 * (label == feat_df.best_seller).mean():.1f}%')
display(feat_df.groupby('category_l2').agg(so_san_pham=('best_seller', 'size'), ban_chay=('best_seller', 'sum')).sort_values('so_san_pham', ascending=False))''')

# =====================================================================================
md('''## 5. Trích từ khóa đặc trưng (Thuật toán 1)

Đề bài yêu cầu sinh nội dung "dựa trên vài từ khóa đặc trưng". Từ khóa được lấy từ **tên sản phẩm**, vì tên là nơi người bán đặt những cụm người mua hay gõ khi tìm kiếm. Một cụm được giữ khi xuất hiện trong tên của **ít nhất 3 sản phẩm**: đó là cách thị trường gọi, không phải cách gọi riêng của một shop.

### Ô 5.1: Chạy từng bước trên một tên tự viết
- **Bước a, `name_segments`:** đưa về chữ thường, bỏ nội dung trong `[...]`, cắt tên thành các đoạn theo dấu `, - | / + ( )`, tách từ tiếng Việt bằng underthesea, bỏ mã sản phẩm (vừa chữ vừa số như `bz08`) và tên thương hiệu.
- **Bước b, `candidate_phrases`:** sinh mọi cụm 1–3 từ liền nhau **trong cùng một đoạn** (không ghép qua dấu phẩy), bỏ cụm bắt đầu hay kết thúc bằng từ chung như "nữ", "cao cấp", "hàng".
- **Bước c, đếm df:** với mỗi cụm, đếm số tên sản phẩm thật trong dữ liệu có chứa cụm đó.
- **Bước d, `select_keywords`:** chấm điểm `log(1 + df) × (1 + 0,5 × (số âm tiết − 1))`, phạt cụm quá phổ biến (xuất hiện ở hơn 30% sản phẩm), chọn từ khóa chính là cụm điểm cao nhất có danh từ loại sản phẩm, rồi thêm tới 6 cụm không trùng nhau.''')
code('''from keywords import name_segments, candidate_phrases, select_keywords
example_name = 'Chân váy chữ A cạp cao công sở - vải tuyết mưa dày dặn, size S M L [Hàng mới] BZ08'
seg = name_segments(example_name, brand='', lang='vi')
print('a) Các đoạn sau khi tách từ:', seg)
cand = candidate_phrases(seg, 'vi')
print('b) Số cụm ứng viên:', len(cand), '→', sorted(cand)[:15], '...')

if HAS_PROCESSED:
    names = pd.read_csv(ROOT / 'data/processed/tiki_features.csv', usecols=['name', 'brand'])
    df_counts = {}
    for name, brand in zip(names.name, names.brand):
        for p in candidate_phrases(name_segments(name, brand, 'vi'), 'vi'):
            df_counts[p] = df_counts.get(p, 0) + 1
    shown = sorted(((df_counts.get(p, 0), p) for p in cand), reverse=True)[:10]
    print('c) df của các cụm (số tên sản phẩm thật chứa cụm):', [(p, n) for n, p in shown])
    print('d) Từ khóa chọn được:', select_keywords(cand, df_counts, len(names), 'vi'))
else:
    print('c, d) Cần tên sản phẩm thật (thư mục Data Source) để đếm df; xem kết quả tổng hợp ở ô 5.2.')''')

md('''### Ô 5.2: Kết quả trích từ khóa trên 1.941 sản phẩm
- **Làm gì:** đọc cột `keywords` của bảng đầu vào và thống kê số từ khóa mỗi sản phẩm, cùng các từ khóa chính phổ biến nhất theo danh mục.
- **Đọc kết quả:** trung bình khoảng 4,8 từ khóa mỗi sản phẩm; từ khóa chính thường là loại hàng kèm kiểu dáng ("chân váy chữ a", "áo sơ mi").''')
code('''inputs = pd.read_csv(INPUT_PATH)
kw = inputs.keywords.apply(json.loads)
print('Số từ khóa mỗi sản phẩm: trung bình', round(kw.apply(len).mean(), 2), '| phân bố:', kw.apply(len).value_counts().sort_index().to_dict())
top = (inputs.assign(primary=kw.str[0]).groupby('category_l2').primary
       .agg(lambda s: ', '.join(s.value_counts().head(3).index)))
display(top.rename('3 từ khóa chính phổ biến nhất').to_frame())''')

# =====================================================================================
md('''## 6. Phản đối của khách từ đánh giá thấp

Kịch bản chốt sale phải trả lời được lời chê thật của khách. Script `src/objections.py` làm việc này bằng **từ điển sáu chủ đề**, mỗi chủ đề là một biểu thức chính quy gồm các cách nói hay gặp, kể cả viết tắt như "ko", "k".

### Ô 6.1: Từ điển chủ đề và thử trên câu tự viết
- **Làm gì:** in từ điển, rồi phân loại ba câu đánh giá **tự viết** để thấy cách khớp. Một câu có thể thuộc nhiều chủ đề.
- **Đọc kết quả:** câu "áo ko giống hình, vải mỏng" thuộc cả "Không giống hình/mô tả" và "Chất liệu kém".''')
code('''import re
from objections import THEMES
display(pd.DataFrame([{'chủ đề': k, 'biểu thức (rút gọn)': v.replace('\\b', '')[:110]} for k, v in THEMES.items()]))
for sentence in ['Áo ko giống hình, vải mỏng', 'Giao hàng chậm, đóng gói sơ sài', 'Mặc bị chật, phải đổi size']:
    hits = [k for k, pattern in THEMES.items() if re.search(pattern, sentence.lower())]
    print(f'"{sentence}" →', hits)''')

md('''### Ô 6.2: Kết quả trên đánh giá thật
- **Làm gì:** đọc kết quả đã tổng hợp từ 297 đánh giá 1–3 sao: tỷ lệ đánh giá nhắc đến từng chủ đề, và hai chủ đề đứng đầu của mỗi danh mục. Hai chủ đề này được đưa vào bản tóm tắt để mô hình phải trả lời.
- **Đọc kết quả:** "Không giống hình/mô tả" là lời chê phổ biến nhất (khoảng 21%). Vì vậy câu lệnh có quy tắc chỉ dùng dữ kiện người bán đã nhập, và phần đánh giá đo riêng tỷ lệ chi tiết không có căn cứ.''')
code('''summary = read_json(RES / 'objection_themes_summary.json')
print('Số đánh giá 1–3 sao của sản phẩm trong tập phân tích:', summary.get('n_low_reviews'),
      f"| nhắc ít nhất một chủ đề: {100 * summary.get('share_with_any_theme', 0):.1f}%")
overall = pd.DataFrame(summary.get('overall', {})).T
overall['share'] = (100 * overall['share']).round(1)
display(overall.rename(columns={'count': 'số đánh giá', 'share': '% đánh giá thấp'}).sort_values('số đánh giá', ascending=False))
by_cat = read_json(RES / 'objections_by_category.json')
display(pd.DataFrame([{'danh mục': c, 'chủ đề 1': (v[0]['phrase'] if v else ''), 'chủ đề 2': (v[1]['phrase'] if len(v) > 1 else '')} for c, v in by_cat.items()]))''')

# =====================================================================================
md('''## 7. Phân tích khám phá

### Ô 7.1: Phân phối số lượng đã bán
- **Làm gì:** vẽ phân phối số đã bán (trục log vì lệch rất mạnh) và tỷ lệ sản phẩm chưa bán được đơn nào.
- **Vì sao:** để chọn cách định nghĩa nhãn. Khi phần lớn sản phẩm bán 0 đơn, không thể dùng trung bình hay một ngưỡng chung; phải xếp hạng trong danh mục như ô 4.4.
- **Đọc kết quả:** khoảng 62% sản phẩm chưa bán đơn nào, trung vị bằng 0, trong khi vài sản phẩm bán hàng nghìn đơn.''')
code('''sold = pd.to_numeric(feat_df.quantity_sold, errors='coerce').fillna(0)
print(f'Chưa bán đơn nào: {100 * (sold == 0).mean():.1f}% | trung vị: {sold.median():.0f} | lớn nhất: {sold.max():.0f}')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
sns.histplot(np.log1p(sold), bins=40, ax=ax[0]); ax[0].set(title='Phân phối log(1 + số đã bán)', xlabel='log(1 + số đã bán)', ylabel='số sản phẩm')
feat_df.best_seller.value_counts().rename({0: 'Khác', 1: 'Bán chạy'}).plot.bar(ax=ax[1], color=['#8D99AE', '#2A9D8F'])
ax[1].set(title='Số sản phẩm theo nhãn', xlabel=''); plt.tight_layout(); plt.show()''')

md('''### Ô 7.2: Tỷ lệ bán chạy theo từng đặc trưng nội dung
- **Làm gì:** chia sản phẩm thành nhóm theo một đặc trưng (ví dụ có hay không nhắc size; 0, 1–2, 3–5, trên 5 gạch đầu dòng) và tính **tỷ lệ bán chạy của mỗi nhóm**, kèm khoảng tin cậy Wilson 95%.
- **Vì sao dùng khoảng Wilson:** nhóm nhỏ có tỷ lệ dao động mạnh; khoảng Wilson cho biết tỷ lệ thật có thể nằm trong vùng nào, và vẫn đúng khi tỷ lệ gần 0 hoặc nhóm ít mẫu.
- **Đọc kết quả:** hai thanh có khoảng tin cậy chồng nhau nhiều thì khác biệt chưa đáng tin. Đây là so sánh **chưa kiểm soát** giá và tuổi sản phẩm; phần 9 mới kiểm soát.''')
code('''from statsmodels.stats.proportion import proportion_confint
def rate_table(groups):
    rows = []
    for name, g in feat_df.groupby(groups, observed=True):
        lo, hi = proportion_confint(g.best_seller.sum(), len(g), method='wilson')
        rows.append({'nhóm': str(name), 'n': len(g), 'tỷ lệ bán chạy (%)': 100 * g.best_seller.mean(), 'thấp': 100 * lo, 'cao': 100 * hi})
    return pd.DataFrame(rows)
specs = {'Số gạch đầu dòng': pd.cut(feat_df.n_bullets, [-1, 0, 2, 5, 1e9], labels=['0', '1–2', '3–5', '>5']),
         'Nhắc size': feat_df.has_size_info.map({0: 'không', 1: 'có'}),
         'Có lời kêu gọi mua': feat_df.has_cta.map({0: 'không', 1: 'có'})}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (title, groups) in zip(axes, specs.items()):
    t = rate_table(groups); display(t.round(1).assign(đặc_trưng=title))
    ax.errorbar(t['tỷ lệ bán chạy (%)'], t['nhóm'], xerr=[t['tỷ lệ bán chạy (%)'] - t['thấp'], t['cao'] - t['tỷ lệ bán chạy (%)']], fmt='o', capsize=3)
    ax.set(title=title, xlabel='% bán chạy (Wilson 95%)')
plt.tight_layout(); plt.show()''')

# =====================================================================================
md('''## 8. Phân tích mô tả

### Ô 8.1: Thống kê các biến chính, tách theo nhãn
- **Làm gì:** tính trung vị, trung bình và độ lệch chuẩn của giá, độ dài tên, độ dài mô tả, số gạch đầu dòng, tuổi sản phẩm; sau đó so trung vị giữa nhóm bán chạy và nhóm còn lại.
- **Vì sao dùng trung vị:** các biến này lệch phải (vài sản phẩm giá rất cao, mô tả rất dài), trung bình bị kéo lên nên không đại diện.
- **Đọc kết quả:** nhóm bán chạy có tên dài hơn, nhiều gạch đầu dòng hơn và giá thấp hơn; độ dài mô tả gần như không khác. 56,7% mô tả dưới 200 từ, mức được khuyến nghị.''')
code('''cols = {'price': 'Giá (đồng)', 'title_len_chars': 'Độ dài tên (ký tự)', 'desc_len_words': 'Số từ mô tả',
        'n_bullets': 'Số gạch đầu dòng', 'n_images': 'Số ảnh', 'product_age_days': 'Tuổi sản phẩm (ngày)'}
d = feat_df[list(cols)].apply(pd.to_numeric, errors='coerce')
desc = d.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(index=cols, columns={'mean': 'trung bình', '50%': 'trung vị', 'std': 'độ lệch chuẩn'})
display(desc.round(1))
display(d.groupby(feat_df.best_seller).median().T.rename(index=cols, columns={0: 'trung vị nhóm khác', 1: 'trung vị nhóm bán chạy'}).round(1))
print(f"Mô tả dưới 200 từ: {100 * (d.desc_len_words < 200).mean():.1f}% | tên dài 15–50 ký tự: {100 * d.title_len_chars.between(15, 50).mean():.1f}%")''')

# =====================================================================================
md('''## 9. Dự báo sản phẩm bán chạy (RQ1)

**Câu hỏi:** sau khi tính đến giá và tuổi sản phẩm, đặc trưng nội dung nào còn gắn với khả năng bán chạy? Kết quả dùng để quyết định nên đưa yêu cầu nào vào câu lệnh sinh nội dung.

### Ô 9.1: Chọn biến đầu vào
- **Biến kiểm soát (5):** log giá, % giảm giá, tuổi sản phẩm, số ảnh, số biến thể; cộng danh mục cấp 2 (mã hóa one-hot).
- **Biến nội dung trang (19):** các đặc trưng ở ô 4.3, cộng có video, số dòng thông số, có mô tả meta, từ khóa nằm trong 60 ký tự đầu của tên.
- **Biến bị loại:** điểm đánh giá và số lượt đánh giá. Chúng chỉ có **sau khi** đã bán, nên đưa vào sẽ làm rò rỉ nhãn: mô hình "đoán" bán chạy nhờ thông tin của tương lai.''')
code('''from predictive import prepare, pipeline, CONTROL, CONTENT, LABELS
data = prepare(feat_df)
X = data[CONTROL + CONTENT + ['category_l2']]
y = pd.to_numeric(data.best_seller)
display(pd.DataFrame({'nhóm': ['kiểm soát'] * len(CONTROL) + ['nội dung'] * len(CONTENT), 'biến': CONTROL + CONTENT,
                      'tên': [LABELS.get(c, c) for c in CONTROL + CONTENT]}).T)''')

md('''### Ô 9.2: Chia tập huấn luyện và kiểm tra
- **Làm gì:** chia 80% để huấn luyện, 20% để kiểm tra, **phân tầng theo nhãn** (hai tập có cùng tỷ lệ bán chạy khoảng 26%), hạt giống ngẫu nhiên 42 để ai chạy lại cũng ra đúng cách chia này.
- **Vì sao:** tập kiểm tra được giữ riêng, không dùng khi chọn mô hình, nên điểm trên đó phản ánh khả năng dự báo trên sản phẩm mới.''')
code('''from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print('Huấn luyện:', len(Xtr), f'(bán chạy {100 * ytr.mean():.1f}%) | kiểm tra:', len(Xte), f'(bán chạy {100 * yte.mean():.1f}%)')''')

md('''### Ô 9.3: Ba mô hình và cách tìm siêu tham số
- **Quy trình chung cho mỗi mô hình:** điền giá trị thiếu bằng trung vị → chuẩn hóa biến số → mã hóa one-hot danh mục → mô hình.
- **Ba mô hình:** hồi quy Logistic (mốc so sánh, dễ giải thích), Random Forest, XGBoost.
- **Mất cân bằng lớp** (26% bán chạy): Logistic và Random Forest dùng `class_weight='balanced'`; XGBoost dùng `scale_pos_weight` bằng tỷ lệ lớp âm trên lớp dương.
- **Tìm siêu tham số:** `RandomizedSearchCV` thử 20 tổ hợp ngẫu nhiên, mỗi tổ hợp chấm bằng **kiểm định chéo 5 phần** trên tập huấn luyện, tiêu chí ROC-AUC. Ví dụ với XGBoost: số cây 100–400, độ sâu 2–6, tốc độ học 0,01–0,3, tỷ lệ lấy mẫu 0,7–1.
- **Khi `RUN_SEARCH = False`:** đọc kết quả đã lưu của lần chạy gốc (5–10 phút), không chạy lại.''')
code('''from scipy.stats import loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
spaces = {
    'logistic': (LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED), {'model__C': loguniform(1e-3, 100)}),
    'random_forest': (RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
                      {'model__n_estimators': randint(100, 401), 'model__max_depth': [None, 3, 5, 10, 20],
                       'model__min_samples_leaf': randint(1, 8), 'model__max_features': ['sqrt', .7, 1.0]}),
    'xgboost': (XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric='logloss', scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum())),
                {'model__n_estimators': randint(100, 401), 'model__max_depth': randint(2, 7),
                 'model__learning_rate': loguniform(.01, .3), 'model__subsample': [.7, .85, 1.]}),
}
saved = read_json(RES / 'predictive_metrics.json')
fitted = {}
if RUN_SEARCH:
    rows = []
    for name, (est, params) in spaces.items():
        search = RandomizedSearchCV(pipeline(est, CONTROL + CONTENT), params, n_iter=20, scoring='roc_auc', cv=cv, random_state=SEED, n_jobs=1)
        search.fit(Xtr, ytr); fitted[name] = search
        p = search.predict_proba(Xte)[:, 1]; pred = search.predict(Xte)
        rows.append({'mô hình': name, 'AUC CV': search.best_score_, 'ROC-AUC kiểm tra': roc_auc_score(yte, p), 'PR-AUC': average_precision_score(yte, p),
                     'precision': precision_score(yte, pred), 'recall': recall_score(yte, pred), 'F1': f1_score(yte, pred)})
    results = pd.DataFrame(rows)
else:
    results = pd.DataFrame([{'mô hình': k, 'AUC CV': v['cv_mean'], 'ROC-AUC kiểm tra': v['roc_auc'], 'PR-AUC': v['pr_auc'],
                             'precision': v['precision'], 'recall': v['recall'], 'F1': v['f1']} for k, v in saved.get('models', {}).items()])
display(results.round(3))
best_params = saved.get('models', {}).get('xgboost', {}).get('best_params')
print('Siêu tham số tốt nhất của XGBoost (lần chạy gốc):', best_params)''')

md('''### Ô 9.4: Đọc các chỉ số
- **ROC-AUC:** xác suất mô hình xếp một sản phẩm bán chạy cao hơn một sản phẩm thường. 0,5 là đoán bừa, 1 là hoàn hảo. XGBoost đạt khoảng 0,83.
- **PR-AUC:** độ chính xác trung bình khi bắt các sản phẩm bán chạy; mốc so sánh là tỷ lệ lớp dương 26%. Khoảng 0,59 nghĩa là tốt hơn đoán bừa hơn hai lần.
- **Recall cao (≥ 0,68), precision thấp (0,41–0,50):** nhờ cân bằng lớp, mô hình bắt được phần lớn sản phẩm bán chạy nhưng cũng báo nhầm khá nhiều.

### Ô 9.5: Nội dung trang đóng góp thêm bao nhiêu (ablation)
- **Làm gì:** so sánh hai phiên bản XGBoost: **chỉ biến kiểm soát** và **kiểm soát + nội dung**. Chênh lệch ROC-AUC trên tập kiểm tra là phần đóng góp của nội dung trang. Khoảng tin cậy 95% của chênh lệch được tính bằng bootstrap có cặp: lấy mẫu lại tập kiểm tra 2.000 lần, mỗi lần tính lại chênh lệch.
- **Đọc kết quả:** nếu cận dưới của khoảng tin cậy chạm 0 thì chưa khẳng định được nội dung trang có đóng góp.''')
code('''ab = saved.get('ablation', {})
if RUN_SEARCH and 'xgboost' in fitted:
    est, params = spaces['xgboost']
    ctrl = RandomizedSearchCV(pipeline(est, CONTROL), params, n_iter=20, scoring='roc_auc', cv=cv, random_state=SEED, n_jobs=1).fit(Xtr, ytr)
    p_full, p_ctrl = fitted['xgboost'].predict_proba(Xte)[:, 1], ctrl.predict_proba(Xte)[:, 1]
    rng, yy, deltas = np.random.default_rng(SEED), np.asarray(yte), []
    for _ in range(2000):
        idx = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[idx])) == 2:
            deltas.append(roc_auc_score(yy[idx], p_full[idx]) - roc_auc_score(yy[idx], p_ctrl[idx]))
    ab = {'control_roc_auc': roc_auc_score(yte, p_ctrl), 'control_plus_content_roc_auc': roc_auc_score(yte, p_full),
          'delta_ci95': [np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)]}
    ab['delta_roc_auc'] = ab['control_plus_content_roc_auc'] - ab['control_roc_auc']
if ab:
    print(f"Chỉ kiểm soát: {ab['control_roc_auc']:.3f} | kiểm soát + nội dung: {ab['control_plus_content_roc_auc']:.3f}")
    print(f"Mức tăng: {ab['delta_roc_auc']:.3f}, khoảng tin cậy 95%: [{ab['delta_ci95'][0]:.3f}; {ab['delta_ci95'][1]:.3f}]")
imp = pd.DataFrame(saved.get('permutation_importance', []))
if not imp.empty:
    imp['tên'] = imp.feature.map(lambda c: LABELS.get(c, c))
    print('\\nMức quan trọng hoán vị (mức giảm ROC-AUC khi xáo trộn một biến), 8 biến đứng đầu:')
    display(imp[['tên', 'mean_auc_decrease', 'std']].head(8).round(3))''')

md('''### Ô 9.6: Từng đặc trưng sau khi giữ nguyên biến kiểm soát (Logit)
- **Làm gì:** ước lượng hồi quy Logistic bằng statsmodels trên toàn bộ dữ liệu: biến kiểm soát cộng 9 đặc trưng nội dung. Biến liên tục được chuẩn hóa nên **tỷ số chênh (OR)** đọc là "khi đặc trưng tăng 1 độ lệch chuẩn, odds bán chạy nhân với OR". Biến đếm được lấy log(1 + x) trước.
- **Hiệu chỉnh Holm:** kiểm định 9 đặc trưng cùng lúc thì dễ có ít nhất một kết quả "có ý nghĩa" do ngẫu nhiên; Holm nâng giá trị p để kiểm soát sai lầm đó.
- **Đọc kết quả:** OR > 1 là liên hệ dương; chỉ những dòng có `p_holm < 0,05` mới được coi là có ý nghĩa. Kết quả gốc: chỉ **độ dài tên** còn ý nghĩa (OR khoảng 1,29); **số gạch đầu dòng** chỉ ở mức gợi ý. Đây là liên hệ, không phải nhân quả.''')
code('''import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from rq1_inference import build_design
design = build_design(feat_df.copy())
Xd, yd = design if isinstance(design, tuple) else (design, feat_df.best_seller)
fit = sm.Logit(pd.to_numeric(yd), sm.add_constant(Xd.astype(float))).fit(disp=False)
from rq1_inference import CONTENT_CONTINUOUS, CONTENT_BINARY
content_rows = [c for c in list(CONTENT_CONTINUOUS) + list(CONTENT_BINARY) if c in Xd.columns]  # đúng 9 đặc trưng nội dung
or_table = pd.DataFrame({'OR': np.exp(fit.params[content_rows]), 'p': fit.pvalues[content_rows]})
or_table['p_holm'] = multipletests(or_table.p, method='holm')[1]
or_table.index = [LABELS.get(c, c) for c in or_table.index]
print('Số đặc trưng nội dung được hiệu chỉnh Holm:', len(or_table))
display(or_table.sort_values('p').round(3))''')

# =====================================================================================
md('''## 10. Sinh nội dung (M0 đến M3)

Năm cấu hình được so sánh, khác nhau ở **lượng thông tin đưa cho mô hình trước khi viết**:

| Ký hiệu | Cách làm | Cần gì |
|---|---|---|
| M0 | Ghép từ khóa vào câu mẫu soạn sẵn, không dùng AI | Không gì |
| M1 | Đưa bản tóm tắt cho mô hình, để nó tự viết | Khóa API |
| M2 | M1 + các ngưỡng có nguồn công bố + khung AIDA, SPIN + quy tắc chỉ dùng dữ kiện đã nhập | Khóa API |
| M3 | M2 + ba mô tả thật của sản phẩm bán chạy tương tự (truy hồi bằng TF-IDF) | Khóa API + kho mô tả |
| M4 | Tinh chỉnh LoRA một mô hình nhỏ chạy trên máy (phần 12) | GPU hoặc CPU |

### Ô 10.1: Dựng bản tóm tắt (brief) cho một sản phẩm kiểm tra
- **Làm gì:** lấy một sản phẩm của tập kiểm tra và dựng bản tóm tắt bằng hàm `make_brief`: danh mục, tối đa 6 từ khóa, thuộc tính đã biết, giá, và hai phản đối đứng đầu của danh mục (từ phần 6).
- **Đọc kết quả:** đây là **toàn bộ** thông tin mô hình được biết về sản phẩm. Mô hình không được thấy mô tả gốc của người bán.''')
code('''from generate import make_brief
objections = read_json(RES / 'objections_by_category.json')
gen_dir = RES / 'generations' if (RES / 'generations').exists() else RES
saved_m2 = [json.loads(l) for l in open(gen_dir / 'generations_M2_tiki.jsonl', encoding='utf-8') if l.strip()]
first = saved_m2[0]  # sản phẩm đầu tiên của tập kiểm tra
row = inputs[inputs.product_id.astype(str) == str(first['product_id'])].iloc[0].to_dict()
row['item_id'] = str(first['item_id'])
brief = make_brief(row, 'tiki', objections)
print(json.dumps(brief, ensure_ascii=False, indent=1))''')

md('''### Ô 10.2: M0, mẫu cố định không dùng AI
- **Làm gì:** gọi `template_output`, hàm ghép từ khóa và thuộc tính vào các câu soạn sẵn. Đây là mốc "làm thủ công nhanh" để so sánh.
- **Đọc kết quả:** đủ các mục nhưng câu chữ cứng, lặp cấu trúc giữa các sản phẩm.''')
code('''from prompts import template_output, zero_shot, engineered
m0 = template_output(brief, 'vi')
for k, v in m0.items():
    print(f'[{k}]', v if isinstance(v, str) else v, '\\n')''')

md('''### Ô 10.3: Câu lệnh M1 và M2 gửi cho mô hình
- **Làm gì:** in nguyên văn câu lệnh của M1 (tối giản) và M2 (có ràng buộc) cho cùng bản tóm tắt.
- **Đọc kết quả:** M2 dài hơn nhiều vì có từng ngưỡng cụ thể: tiêu đề 50–60 ký tự, mô tả meta 120–156 ký tự, mô tả từ 200 từ với ít nhất 3 gạch đầu dòng, tiêu đề email tối đa 9 từ và 60 ký tự... Mỗi ngưỡng lấy từ một nguồn công bố (Ahrefs, Yoast, Tiki Academy, Mailchimp, Omnisend, Gmail). M3 là M2 cộng thêm ba ví dụ thật nên không in ở đây.''')
code('''print('=' * 30, 'M1', '=' * 30)
print(zero_shot([brief], 'vi'))
print('\\n' + '=' * 30, 'M2', '=' * 30)
print(engineered([brief], 'vi'))''')

md('''### Ô 10.4: Gọi mô hình thật (tùy chọn) và xem đầu ra đã lưu
- **Làm gì:** khi bật `RUN_GENERATION` và có khóa Gemini, gửi câu lệnh M2 tới Gemini qua hàm `generate_via_api` (cũng là hàm website demo dùng) và in kết quả. Khóa lấy từ **Colab Secrets** với tên `GEMINI_API_KEY` (biểu tượng chìa khóa ở thanh bên trái; tạo khóa miễn phí tại aistudio.google.com/apikey), không ghi khóa vào notebook.
- **Luôn chạy:** in đầu ra M2 đã lưu cho chính sản phẩm này, do GPT-6.1-Terra sinh trong lần chạy gốc.''')
code('''if RUN_GENERATION:
    from llm_api import generate_via_api
    key = ''
    if IN_COLAB:
        from google.colab import userdata
        try:
            key = userdata.get('GEMINI_API_KEY') or ''
        except Exception:
            key = ''
    key = key or os.environ.get('GEMINI_API_KEY', '')
    if key:
        text, model, seconds = generate_via_api(engineered([brief], 'vi'), 'gemini', key)
        print(f'Mô hình {model}, {seconds:.1f} giây:\\n', text[:2500])
    else:
        print('Chưa có GEMINI_API_KEY trong Colab Secrets.')
out = next(r for r in saved_m2 if str(r['item_id']) == brief['item_id'])['output']
for k, v in out.items():
    print(f'[{k}]', v, '\\n')''')

# =====================================================================================
md('''## 11. Đánh giá

Mỗi đầu ra được chấm theo ba tầng: (1) **danh sách kiểm tra tự động**, mỗi tiêu chí trả về đạt hoặc không đạt và có nguồn công bố; (2) **giám khảo mô hình ngôn ngữ** Gemini chấm 1–5 điểm, ẩn tên phương pháp; (3) **chỉ số kinh doanh** (phần 13). Phần này chạy tầng 1 từng bước trên đầu ra ở ô 10.4, rồi đọc kết quả tổng hợp của cả 99 sản phẩm.

### Ô 11.1: Danh sách kiểm tra trang sản phẩm (RQ3), từng tiêu chí
- **Làm gì:** gọi `seo_checks` với tiêu đề, mô tả meta, mô tả và từ khóa chính, rồi in từng tiêu chí: giá trị đo, ngưỡng, nguồn, kết quả.
- **Cách tính tỷ lệ tuân thủ:** số tiêu chí đạt chia cho số tiêu chí áp dụng, không gán trọng số. Hai tiêu chí chỉ báo cáo riêng, không tính vào tỷ lệ: mật độ từ khóa (vì thuật toán của Yoast không công khai) và tên 15–50 ký tự (vì mâu thuẫn với khuyến nghị 50–60 ký tự của Ahrefs).''')
code('''from metrics_content import seo_checks, email_checks, factual_support
seo = seo_checks(out.get('title'), out.get('meta_description'), out.get('description'), brief['keywords'][0], brief['keywords'], 'vi')
RULES = {
    'title_len_50_60': ('Tiêu đề 50–60 ký tự', seo['title_chars'], 'Ahrefs (Hardwick, 2024)'),
    'title_tiki_15_50': ('Tên 15–50 ký tự (báo cáo riêng)', seo['title_chars'], 'Tiki Academy'),
    'title_tiki_max_225': ('Tên tối đa 225 ký tự', seo['title_chars'], 'Tiki Academy'),
    'meta_len_120_156': ('Mô tả meta 120–156 ký tự', seo['meta_chars'], 'Yoast (Toonen, 2025)'),
    'keyphrase_in_title': ('Từ khóa chính trong tiêu đề', '', 'Yoast'),
    'keyphrase_in_first_paragraph': ('Từ khóa chính ở đoạn đầu', '', 'Yoast'),
    'keyphrase_in_meta': ('Từ khóa chính trong mô tả meta', '', 'Yoast'),
    'density_proxy_0_5_3': ('Mật độ từ khóa 0,5–3% (báo cáo riêng)', round(seo['keyphrase_density_proxy'] or 0, 2), 'Yoast'),
    'desc_words_ge_200': ('Mô tả từ 200 từ', seo['description_words'], 'Yoast, Tiki Academy'),
}
display(pd.DataFrame([{'tiêu chí': RULES[k][0], 'giá trị đo': RULES[k][1], 'nguồn': RULES[k][2], 'đạt': v} for k, v in seo['checks'].items()]))
print(f"Tỷ lệ tuân thủ của bản này: {100 * seo['compliance_rate']:.1f}% | số gạch đầu dòng: {seo['n_bullets']}")''')

md('''### Ô 11.2: Tiêu đề email và chi tiết không có căn cứ
- **Tiêu đề email:** mỗi tiêu đề được kiểm tra theo Mailchimp (tối đa 9 từ và 60 ký tự, tối đa 3 dấu câu, tối đa 1 emoji), Omnisend (4–6 từ hoặc 21–40 ký tự), Gmail (không giả "Re:" hay "Fwd:"), và không hứa ưu đãi không có trong bản tóm tắt.
- **Chi tiết không có căn cứ:** `factual_support` tìm trong văn bản các nhắc đến màu, chất liệu, kiểu dáng, size, % giảm giá, giá, miễn phí vận chuyển, đổi trả, rồi kiểm tra từng nhắc đến có trong bản tóm tắt hay không.
- **Đọc kết quả:** `unsupported` là những chi tiết mô hình tự thêm; đây là thứ người bán phải soát lại trước khi đăng.''')
code('''email_rows = []
for subject in out.get('email_subjects', []):
    r = email_checks(subject, brief['keywords'], brief)
    email_rows.append({'tiêu đề': subject, **{k: v for k, v in r.get('checks', r).items() if isinstance(v, bool)}})
display(pd.DataFrame(email_rows))
fact = factual_support(out.get('description', ''), brief, 'vi')
print('Chi tiết trong mô tả:', {k: fact[k] for k in fact if k in ('supported', 'unsupported', 'total', 'rate')})''')

md('''### Ô 11.3: Độ tương đồng với mô tả thật của người bán (RQ2)
- **Làm gì:** so mô tả sinh ra với mô tả thật của người bán bằng bốn độ đo: ROUGE-L (chuỗi từ chung dài nhất), BLEU (trùng cụm từ), chrF++ (trùng cụm ký tự, ít phụ thuộc cách tách từ tiếng Việt), BERTScore (gần nghĩa). Ô này tính ba độ đo đầu cho một sản phẩm (cần Data Source vì phải có mô tả gốc, không in mô tả gốc ra), rồi đọc kết quả của cả tập.
- **Đọc kết quả:** điểm thấp ở mọi cấu hình là bình thường: bản tóm tắt chỉ có vài từ khóa, người bán biết nhiều chi tiết hơn. M2, M3 cao nhất theo chrF++; M0, M1 cao hơn theo ROUGE-L vì văn bản ngắn hơn. Vì vậy bài không dùng độ tương đồng làm tiêu chí duy nhất.''')
code('''from metrics_content import reference_similarity
if HAS_PROCESSED:
    ref = pd.read_csv(ROOT / 'data/processed/gen_inputs_tiki.csv').set_index('product_id').reference_description.get(row['product_id'], '')
    if ref:
        sim = reference_similarity([out.get('description', '')], [strip_tiki_boilerplate(clean_text(ref))], 'vi', fast=True)
        print('Sản phẩm này:', {k: round(v, 3) for k, v in sim['per_item'][0].items() if v is not None})
ev = pd.read_csv(RES / 'eval_per_item_tiki.csv')
cols = [c for c in ['rougeL', 'chrf_plus_plus', 'bertscore_f1', 'sentence_bleu'] if c in ev]
display(ev.groupby('method')[cols].mean().round(3))''')

md('''### Ô 11.4: Kết quả tổng hợp trên 99 sản phẩm và kiểm định thống kê
- **Làm gì:** tính tỷ lệ tuân thủ trung bình theo cấu hình trên **năm tiêu chí chung**, tức những tiêu chí áp dụng được cho cả nội dung sinh lẫn nội dung người bán (người bán trên Tiki không có mô tả meta). Sau đó đọc kết quả kiểm định Wilcoxon cho mẫu cặp (cùng sản phẩm, hai cấu hình) với giá trị p đã hiệu chỉnh Holm.
- **Vì sao Wilcoxon:** cùng một sản phẩm được chấm ở mọi cấu hình nên dữ liệu là từng cặp; số tiêu chí đạt là số nguyên nhỏ, không phân phối chuẩn, nên dùng kiểm định phi tham số.
- **Đọc kết quả:** M2 và M3 đạt khoảng 85–86%, người bán (`reference`) 64,6%; `p_holm` rất nhỏ nghĩa là khác biệt không do ngẫu nhiên. Giữa M2 và M3 không có khác biệt có ý nghĩa.''')
code('''common5 = ['seo_title_len_50_60', 'seo_title_tiki_max_225', 'seo_keyphrase_in_title', 'seo_keyphrase_in_first_paragraph', 'seo_desc_words_ge_200']
present = [c for c in common5 if c in ev]
rate = ev.groupby('method')[present].mean().mean(axis=1).mul(100).round(1).rename('% đạt 5 tiêu chí chung')
display(rate.to_frame())
w = read_json(RES / 'seo_common_wilcoxon.json')
comp = pd.DataFrame(w.get('datasets', {}).get('tiki', {}).get('comparisons', []))
if not comp.empty:
    comp = comp.rename(columns={'a': 'cấu hình A', 'b': 'so với B', 'mean_a_pct': '% đạt A', 'mean_b_pct': '% đạt B', 'n_ties': 'số cặp hòa'})
    display(comp.style.format({'p': '{:.2e}', 'p_holm': '{:.2e}', '% đạt A': '{:.1f}', '% đạt B': '{:.1f}'}))''')

md('''### Ô 11.5: Điểm giám khảo và độ tin cậy của giám khảo (RQ4)
- **Làm gì:** đọc điểm trung bình (1–5) mà Gemini chấm cho từng tiêu chí, gồm kịch bản: kêu gọi hành động, liên hệ nhu cầu với lợi ích theo SPIN, xử lý phản đối, đúng sự thật; mô tả: tự nhiên, thuyết phục, liên quan từ khóa; email: gây tò mò, liên quan. Sau đó đọc mức đồng thuận giữa Gemini và giám khảo thứ hai (GPT-5.6-Luna) trên mẫu con.
- **Đọc kết quả:** M2, M3 gần 5 điểm ở xử lý phản đối và đúng sự thật. Đồng thuận gộp khá cao (Spearman khoảng 0,73) nhưng thấp ở một vài tiêu chí, nên điểm giám khảo chỉ dùng để so sánh tương đối giữa các cấu hình.''')
code('''judge_cols = [c for c in ev if c.startswith('judge_')]
display(ev.groupby('method')[judge_cols].mean().T.round(2))
agree = read_json(RES / 'judge_agreement_tiki.json')
print('Đồng thuận hai giám khảo:', {k: v for k, v in agree.items() if not isinstance(v, (dict, list))})''')

md('''### Ô 11.6: Vòng lặp sửa câu lệnh và kiểm tra trên tập độc lập
- **Làm gì:** đọc tỷ lệ "rò rỉ chỉ dẫn", tức đầu ra chứa câu kiểu "brief chưa cung cấp chất liệu", ở hai phiên bản câu lệnh. Sau đó đọc kết quả chạy lại M0, M2, M3 trên 40 sản phẩm của tập kiểm định chưa từng dùng khi sửa câu lệnh.
- **Vì sao:** câu lệnh v2 được sửa sau khi xem lỗi trên tập kiểm tra, nên phải kiểm lại trên dữ liệu chưa xem để chắc kết quả không phải do "học tủ".
- **Đọc kết quả:** rò rỉ giảm từ 90% (v1) về 0% (v2); trên tập độc lập M2 vẫn đạt khoảng 91%, M0 khoảng 61%.''')
code('''leak = read_json(RES / 'prompt_leakage.json')
display(pd.json_normalize(leak.get('results', leak), sep=' / ').T.rename(columns={0: 'giá trị'}).head(20))
hold = read_json(RES / 'holdout_check.json')
if hold:
    display(pd.DataFrame([{'tập': s, 'cấu hình': m, **v} for s in ('val', 'test') for m, v in hold.get(s, {}).items()]).round(3))''')

# =====================================================================================
md('''## 12. Tinh chỉnh mô hình nhỏ (M4)

**Ý tưởng:** thay vì trả tiền gọi mô hình lớn, tinh chỉnh một mô hình nhỏ (Qwen2.5-0.5B, nửa tỷ tham số) trên mô tả thật để chạy ngay trên máy của shop.

**LoRA là gì:** giữ nguyên toàn bộ trọng số gốc của mô hình, chỉ học thêm hai ma trận nhỏ B và A gắn vào các lớp attention, sao cho phần thay đổi bằng B × A. Với hạng r = 16, chỉ 2,16 triệu trong 496 triệu tham số được huấn luyện (0,44%), nên chạy được cả trên CPU.

### Ô 12.1: Cấu hình huấn luyện và đường loss
- **Cấu hình gốc (M4b):** hạng r = 16, alpha = 32, dropout 0,05, gắn vào 4 ma trận q, k, v, o; tối ưu AdamW, tốc độ học 2×10⁻⁴, 30 bước khởi động, gộp gradient 8 mẫu, tối đa 512 token, 1 epoch trên 800 cặp (bản tóm tắt → mô tả thật đã làm sạch), chỉ tính loss trên phần mô tả.
- **Làm gì:** nếu bật `RUN_FINETUNE` và có GPU thì huấn luyện lại; nếu không thì vẽ đường loss của lần chạy gốc (100 bước, 94 phút trên CPU).
- **Đọc kết quả:** loss huấn luyện giảm từ 2,90 xuống 2,46, loss kiểm định 2,55. Mức giảm nhỏ cho thấy mô hình học được rất ít trong 100 bước.''')
code('''import torch
if RUN_FINETUNE and torch.cuda.is_available():
    subprocess.run([sys.executable, 'src/finetune_lora.py', '--dataset', 'tiki', '--train', '--generate', '--max-steps', '100'], check=True)
elif RUN_FINETUNE:
    print('Không có GPU: Runtime → Change runtime type → T4 GPU, rồi chạy lại.')
loss_path = next((p for p in [ROOT / 'logs/finetune_tiki.csv', ROOT / 'results/public/finetune_tiki.csv'] if p.exists()), None)
if loss_path:
    loss = pd.read_csv(loss_path)
    sns.lineplot(data=loss, x='step', y='loss', hue='kind', marker='o'); plt.title('Loss khi tinh chỉnh LoRA (Tiki)'); plt.show()''')

md('''### Ô 12.2: Thử cách sinh chữ và siêu tham số
- **Vấn đề:** 98,3% đầu ra của M4b lặp một cụm từ cho đến hết độ dài.
- **Làm gì (script `src/ft_sweep.py`, 10 sản phẩm của tập kiểm định):** hai dòng đầu dùng **cùng adapter M4b**, chỉ khác cách sinh chữ: cũ (phạt lặp 1,1) và mới (phạt lặp 1,2 cộng **cấm lặp lại cụm 3 từ**). Ba dòng sau huấn luyện lại 40 bước với hạng r và tốc độ học khác.
- **Đọc kết quả:**
  - Chống lặp đưa tỷ lệ lặp từ **100% về 0%** mà không cần huấn luyện lại: lỗi lặp nằm ở cách sinh chữ.
  - Trong cùng 40 bước, tốc độ học cao hơn và hạng lớn hơn cho loss thấp hơn, nhưng chưa cấu hình nào bằng cấu hình gốc (2,55).
  - Đọc đầu ra (cột `sample`) vẫn thấy mô hình **tự bịa chi tiết** ("100% cotton") và lạc đề. Vì vậy M4 không được đưa vào website.''')
code('''sweep = read_json(RES / 'ft_sweep.json')
if sweep:
    t = pd.DataFrame(sweep)
    t['decode'] = t['decode'].astype(str)
    display(t[['config', 'r', 'lr', 'steps', 'val_loss', 'pct_repetitive', 'mean_words', 'mean_latency_s', 'decode']])
    for r in sweep:
        print(f"\\n[{r['config']}] {r['sample'][:200]}")''')

# =====================================================================================
md('''## 13. Tác động kinh doanh

### Ô 13.1: Chi phí và thời gian cho một shop
- **Làm gì:** lấy số token trung bình mỗi sản phẩm của M2 (đếm thật trên 99 sản phẩm bằng `src/token_usage.py`) nhân với bảng giá công bố, truy cập ngày 14/11/2026: Gemini 3.8 Flash 0,75 USD mỗi triệu token đầu vào và 3,75 USD mỗi triệu token đầu ra (áp dụng đến 31/12/2026, Google AI Studio); gpt-6.1-terra 1 và 5 USD (OpenAI). Quy đổi theo 26.000 đồng/USD là mức làm tròn để dễ hình dung, không phải tỷ giá chính thức.
- **Đọc kết quả:** khoảng 80 đồng mỗi sản phẩm với Gemini; một shop đăng 50 mẫu mỗi tháng trả vài nghìn đồng, và 0 đồng nếu dùng hạn mức miễn phí của Gemini như website demo. Máy viết xong 50 mẫu trong khoảng 12 phút.''')
code('''tok = read_json(RES / 'token_usage.json')
if tok:
    print('Token mỗi sản phẩm: vào', tok['input_tokens_mean'], '| ra', tok['output_tokens_mean'], '| độ trễ', tok['latency_s_mean'], 'giây')
    rows = []
    for model, (p_in, p_out) in {'Gemini 3.8 Flash': (0.75, 3.75), 'gpt-6.1-terra': (1.0, 5.0)}.items():
        usd = (tok['input_tokens_mean'] * p_in + tok['output_tokens_mean'] * p_out) / 1e6
        rows.append({'mô hình': model, 'USD/sản phẩm': round(usd, 5), 'đồng/sản phẩm': round(usd * 26000), 'đồng cho 50 mẫu/tháng': round(usd * 26000 * 50)})
    display(pd.DataFrame(rows))
    print('Thời gian máy viết 50 mẫu:', round(50 * tok['latency_s_mean'] / 60, 1), 'phút')''')

md('''### Ô 13.2: Mức gần với đặc trưng của hàng bán chạy
- **Làm gì:** với mỗi cấu hình, tính tỷ lệ nội dung có số gạch đầu dòng và độ dài tên **đạt trung vị của nhóm bán chạy cùng danh mục**, hai đặc trưng mà phần 9 cho thấy gắn với bán chạy.
- **Đọc kết quả:** 79,8% mô tả của M2 có đủ số gạch đầu dòng như hàng bán chạy, so với 48,5% ở nội dung người bán (`reference`). Đây là liên hệ quan sát; muốn biết có tăng doanh số thật hay không phải thử nghiệm A/B.''')
code('''bm = read_json(RES / 'business_metrics.json')
display(pd.DataFrame(bm.get('methods', {})).T[['n', 'pct_bullets_ge_best_seller_median', 'pct_title_ge_best_seller_median', 'latency_s_per_item', 'pct_repetitive_output']])''')

# =====================================================================================
md('''## 14. Phân tích đề xuất

Đề xuất được rút ra từ số liệu ở các phần trên, mỗi đề xuất ghi rõ căn cứ:

| # | Đề xuất cho shop | Căn cứ trong notebook |
|---|---|---|
| 1 | Dùng **M2** làm mặc định: không cần kho mô tả, tỷ lệ đạt chuẩn ngang M3 | Ô 11.4: M2 85,1% so với M3 86,3%, khác biệt không có ý nghĩa; ô 11.6: trên tập độc lập M2 còn cao hơn M3 |
| 2 | Tách tên sản phẩm trên sàn khỏi tiêu đề trang cho Google; thử A/B trước khi đổi độ dài tên | Ô 9.6: tên dài gắn với bán chạy; ô 11.1: khuyến nghị 50–60 ký tự (Google) khác 15–50 ký tự (Tiki) |
| 3 | Luôn đưa phản đối thật của khách vào bản tóm tắt | Ô 6.2: "không giống hình/mô tả" là lời chê số một; ô 11.5: kịch bản có phản đối đạt gần 5 điểm xử lý phản đối |
| 4 | Giữ người duyệt cuối cùng, dựa vào bảng kiểm tra và cảnh báo chi tiết không có căn cứ | Ô 11.2: vẫn còn khoảng 3–5% chi tiết tự thêm |
| 5 | Chưa dùng mô hình nhỏ chạy trên máy; nếu thử thì bật chống lặp và vẫn phải soát dữ kiện | Ô 12.2: hết lặp nhưng vẫn bịa chi tiết |
| 6 | Làm sạch văn mẫu của sàn trước khi dùng mô tả làm ví dụ hay dữ liệu huấn luyện | Phần 4: mọi mô tả Tiki có đoạn thông báo thuế; lần tinh chỉnh đầu chép lại đoạn này ở 86,7% đầu ra |

**Giới hạn:** các liên hệ ở phần 9 là quan sát, không phải nhân quả; mẫu Tiki lệch về hàng bán chạy; điểm giám khảo chưa được đối chiếu với người chấm; doanh số thật chưa được đo.''')

md('''## 15. Công cụ AI được sử dụng

| Công cụ | Vai trò |
|---|---|
| Claude Opus 5.5 (qua Claude Code) | Điều phối, viết và biên tập bài, kiểm chứng số liệu và nguồn |
| GPT-6.1-Sol, GPT-6.1-Terra (qua Codex CLI) | Tra cứu tài liệu, viết một phần mã nguồn, sinh nội dung M1–M3 |
| Gemini 3.8 Flash | Giám khảo chính; mô hình của website demo |
| GPT-5.6-Luna | Giám khảo thứ hai, chỉ dùng đo đồng thuận |
| Claude Haiku, Sonnet | Mở lại URL kiểm chứng trích dẫn, dựng slide |
| Qwen2.5-0.5B-Instruct | Mô hình tinh chỉnh LoRA (M4) |
| bert-base-multilingual-cased | Tính BERTScore |

Mọi số liệu trong notebook và bài báo được tính bằng mã nguồn từ dữ liệu đã khóa, không lấy từ câu trả lời của mô hình.

## Tài liệu tham khảo chính
- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging language and items for retrieval and recommendation. *arXiv:2403.03952*. (Amazon Reviews 2023)
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP 2023*.
- Schröer, C., Kruse, F., & Marx Gómez, J. (2021). A systematic literature review on applying CRISP-DM process model. *Procedia Computer Science, 181*, 526–534.
- Danh sách đầy đủ 49 tài liệu nằm trong bài báo.''')

notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                                     'language_info': {'name': 'python'}, 'colab': {'provenance': []}})
out = ROOT / 'notebook/GenAI_Marketing_ThoiTrangNu.ipynb'
nb.write(notebook, out)
print('Đã ghi', out, ':', len(cells), 'cell')

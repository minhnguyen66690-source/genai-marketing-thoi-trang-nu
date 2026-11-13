"""Tiện ích chung cho phân tích dữ liệu sản phẩm."""
import json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "_workspace/03_results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
SEED = 42
COLORS = ["#1F4E79", "#2A9D8F", "#E76F51", "#E9C46A", "#8D99AE", "#264653"]
VI_STOP = set("và là của có cho với một các những trong được từ này đó khi để không sản phẩm thời trang nữ nam hàng mới cao cấp".split())
STOPWORDS = VI_STOP | set(ENGLISH_STOP_WORDS)
STOPWORDS |= {"hàng", "chính hãng", "cao cấp", "loại", "mẫu", "kiểu", "phong cách", "thời trang", "freeship", "sale", "hot", "new", "2024", "2025"}
for folder in (PROCESSED, RESULTS, FIGURES, MODELS): folder.mkdir(parents=True, exist_ok=True)

def clean_text(value):
    """Chuẩn hóa văn bản và bỏ thẻ HTML."""
    if value is None or (isinstance(value, float) and np.isnan(value)): return ""
    s = unicodedata.normalize("NFC", str(value))
    s = BeautifulSoup(s, "html.parser").get_text("\n")
    return "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in s.splitlines() if line.strip())

def tokenize(text):
    """Tách từ tiếng Việt bằng thư viện sẵn có."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(clean_text(text), format="text")
    except (ImportError, RuntimeError):
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize(clean_text(text))
        except ImportError:
            return clean_text(text)

def save_json(path, value):
    """Lưu JSON UTF-8, chuyển các giá trị không hữu hạn thành null."""
    def convert(x):
        """Chuyển kiểu dữ liệu NumPy sang kiểu JSON."""
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, (np.ndarray, pd.Index)): return x.tolist()
        if isinstance(x, (pd.Timestamp, Path)): return str(x)
        raise TypeError(str(type(x)))
    def sanitize(x):
        """Làm sạch giá trị không hữu hạn trước khi ghi."""
        if isinstance(x, dict): return {str(k): sanitize(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [sanitize(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")

def setup_plot():
    """Thiết lập phông chữ và màu cho biểu đồ."""
    plt.rcParams.update({"font.family":"DejaVu Sans", "savefig.dpi":220, "axes.prop_cycle":plt.cycler(color=COLORS), "axes.grid":True, "grid.alpha":0.18, "axes.axisbelow":True})

def source_text(df=None):
    """Tạo dòng nguồn dữ liệu để đặt dưới biểu đồ."""
    stamps = []
    if df is not None and "crawled_at" in df: stamps = df["crawled_at"].dropna().astype(str).tolist()
    manifest = RAW / "tiki_manifest.json"
    if not stamps and manifest.exists():
        try: stamps = [json.loads(manifest.read_text(encoding="utf-8")).get("crawled_at", "KHÔNG RÕ")]
        except (ValueError, OSError): pass
    if not stamps:
        return "Nguồn: dữ liệu Tiki (ngày thu thập không rõ)"
    day = str(max(stamps))[:10]
    return "Nguồn: dữ liệu Tiki thu thập ngày " + "/".join(reversed(day.split("-")))

def save_figure(fig, path, source):
    """Ghi hình cùng chú thích nguồn."""
    fig.text(.01, .01, source, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Tiki tự gắn một đoạn thông báo về thuế/hóa đơn vào cuối MỌI mô tả sản phẩm (100% bản ghi trong dữ liệu).
# Đoạn này không do người bán viết nên phải bỏ trước khi đo độ dài, trích đặc trưng, làm văn bản tham chiếu
# hay dữ liệu huấn luyện (bản LoRA đầu tiên học thuộc đoạn này và chép lại ở 52/60 đầu ra).
TIKI_BOILERPLATE_START = "Giá sản phẩm trên Tiki đã bao gồm thuế"


def strip_tiki_boilerplate(text):
    """Cắt bỏ đoạn thông báo thuế/hóa đơn do Tiki tự gắn ở cuối mô tả (giữ phần người bán viết)."""
    text = str(text or "")
    index = text.find(TIKI_BOILERPLATE_START)
    if index < 0:
        return text
    return text[:index].rstrip()

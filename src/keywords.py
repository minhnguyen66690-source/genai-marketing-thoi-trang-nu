"""Trích từ khóa đặc trưng và tạo đầu vào cho bài toán sinh nội dung.

Ý tưởng: "từ khóa đặc trưng" của một sản phẩm là các cụm từ trong TÊN sản phẩm mà
(1) đủ phổ biến trên thị trường, tức xuất hiện trong tên của ít nhất MIN_DF sản phẩm khác, và
(2) đủ cụ thể, tức không phải từ chung như "nữ", "hàng", "cao cấp".
Cụm từ không được vượt qua dấu phân cách trong tên ( , - | / + ( ) [ ] ), nên tránh được các cụm vô nghĩa
ghép từ hai vế khác nhau. Mã sản phẩm (chữ lẫn số như "bz08") và tên thương hiệu bị loại.

Đầu ra:
- data/processed/gen_inputs_<tiki|amazon>.csv: brief cho từng sản phẩm (từ khóa, thuộc tính) + mô tả tham chiếu + split.
- _workspace/03_results/keywords_by_category.json: từ khóa đặc trưng theo danh mục (tần suất, độ nâng lift ở nhóm bán chạy).
- figures/keywords_01_danh_muc.png
"""
import ast
import json
import math
import re
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import FIGURES, PROCESSED, RESULTS, SEED, save_figure, save_json, setup_plot, source_text, tokenize

MIN_DF = 3          # cụm từ phải xuất hiện trong tên của ít nhất 3 sản phẩm mới được coi là từ khóa thị trường
MAX_KEYWORDS = 6
MIN_KEYWORDS = 3

# Từ chung trong tên sản phẩm thời trang, không mang nghĩa phân biệt.
VI_GENERIC = {
    "nữ", "cho nữ", "nam", "nam nữ", "unisex", "hàng", "hàng mới", "mới", "chính hãng", "cao cấp", "loại",
    "mẫu", "mẫu mới", "kiểu", "phong cách", "thời trang", "freeship", "sale", "hot", "new", "siêu", "đẹp",
    "xinh", "sang", "sang chảnh", "chất", "chất đẹp", "giá rẻ", "rẻ", "tốt", "loại 1", "hàng đẹp",
    "và", "là", "của", "có", "cho", "với", "một", "các", "những", "trong", "được", "từ", "này", "khi", "để",
    "không", "sản phẩm", "shop", "bộ sưu tập", "hàn quốc", "quảng châu", "ulzzang", "2023", "2024", "2025",
    "2026", "size", "màu", "nhiều màu", "full", "big", "ver", "phiên bản", "kèm", "tặng",
    "người", "cổ", "cực", "dành", "dành cho", "chơi", "lưng", "dài", "vải", "chất liệu", "mặc", "form",
    "dáng", "kiểu dáng", "phụ nữ", "cô gái", "bạn", "đi", "ngày", "tay", "ống",
}
EN_GENERIC = {
    "women", "womens", "women's", "woman", "ladies", "lady", "girls", "for", "and", "with", "the", "a", "an",
    "of", "in", "on", "to", "new", "fashion", "style", "casual", "plus", "size", "s", "m", "l", "xl", "xxl",
    "pack", "set", "2023", "2024", "2025", "by", "or", "at", "from", "high", "long", "back", "short",
    "sexy", "cute", "out", "up", "one", "two", "piece",
}
# Danh từ loại sản phẩm: từ khóa chính (keyphrase) phải chứa một trong các từ này nếu có thể.
VI_PRODUCT_NOUNS = [
    "chân váy", "áo khoác", "áo sơ mi", "áo thun", "áo len", "áo croptop", "áo kiểu", "áo hai dây", "áo dài",
    "đầm", "váy", "áo", "quần", "đồ bộ", "bộ", "set", "jumpsuit", "yếm", "đồ lót", "áo lót", "quần lót", "bra",
    "đồ ngủ", "váy ngủ", "đồ bơi", "bikini", "blazer", "vest", "cardigan", "hoodie", "legging", "jean",
]
EN_PRODUCT_NOUNS = [
    "dress", "dresses", "top", "tops", "shirt", "blouse", "tee", "t-shirt", "tank", "pants", "jeans", "shorts",
    "skirt", "jacket", "coat", "cardigan", "sweater", "hoodie", "jumpsuit", "romper", "leggings", "bra",
    "panties", "pajama", "swimsuit", "bikini", "vest", "blazer", "sweatshirt", "kimono", "camisole",
]
# Từ đơn quá chung hoặc là đơn vị đo, chỉ có nghĩa khi nằm trong cụm dài hơn.
UNIGRAM_BLOCK = {"đồ", "bộ", "kg", "cm", "store", "cao", "thấp", "to", "nhỏ"}
SPLIT_PATTERN = r"[,;|/+()\[\]{}\-–—:!?.]+"


def fold(text):
    """Bỏ dấu tiếng Việt (để so khớp tên thương hiệu viết có dấu/không dấu)."""
    text = str(text).casefold().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def is_code(token):
    """Mã sản phẩm: token có cả chữ và số (ví dụ bz08, bss19). Giữ size dạng 2xl/3xl."""
    has_digit = any(ch.isdigit() for ch in token)
    has_alpha = any(ch.isalpha() for ch in token)
    if re.fullmatch(r"\d?x+l", token):
        return False
    return has_digit and (has_alpha or len(token) > 3)


def name_segments(name, brand, lang):
    """Tách tên sản phẩm thành các đoạn theo dấu phân cách, bỏ thương hiệu và nội dung trong ngoặc vuông."""
    text = unicodedata.normalize("NFC", str(name or "")).casefold()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    brand_words = [fold(w) for w in re.findall(r"\w+", str(brand or "").casefold()) if w not in {"nan", "none"}]
    segments = []
    for part in re.split(SPLIT_PATTERN, text):
        part = part.strip()
        if not part:
            continue
        if lang == "vi":
            words = [w.replace("_", " ") for w in tokenize(part).casefold().split()]
        else:
            words = re.findall(r"[a-z][a-z'&]*", part)
        cleaned = []
        for word in words:
            if not re.search(r"\w", word):
                continue
            if is_code(word):
                continue
            if brand_words and fold(word) in brand_words:
                continue
            cleaned.append(word)
        if cleaned:
            segments.append(cleaned)
    return segments


def candidate_phrases(segments, lang):
    """Sinh các cụm 1–3 từ liên tiếp trong cùng đoạn, loại cụm bắt đầu/kết thúc bằng từ chung."""
    generic = VI_GENERIC if lang == "vi" else EN_GENERIC
    phrases = set()
    for words in segments:
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                phrase = " ".join(gram)
                if phrase in generic:
                    continue
                if gram[0] in generic or gram[-1] in generic:
                    continue
                if len(phrase) < 2 or any(tok.isdigit() for tok in gram):
                    continue
                if n == 1 and phrase in UNIGRAM_BLOCK:
                    continue
                if len(phrase.split()) > 4:
                    continue
                phrases.add(phrase)
    return phrases


def contains_product_noun(phrase, lang):
    """Kiểm tra cụm từ có chứa danh từ loại sản phẩm hay không."""
    nouns = VI_PRODUCT_NOUNS if lang == "vi" else EN_PRODUCT_NOUNS
    padded = f" {phrase} "
    return any(f" {noun} " in padded for noun in nouns)


def select_keywords(phrases, df_counts, n_docs, lang):
    """Chọn 3–6 từ khóa: cụm phổ biến trên thị trường (df ≥ MIN_DF), ưu tiên cụm dài và cụm có danh từ sản phẩm.

    Điểm = log(1 + df) × (1 + 0,5 × (số âm tiết − 1)) × phạt nếu cụm quá phổ biến (df > 30% số sản phẩm).
    """
    scored = []
    for phrase in phrases:
        df = df_counts.get(phrase, 0)
        if df < MIN_DF:
            continue
        n_syll = len(phrase.split())
        score = math.log1p(df) * (1 + 0.5 * (n_syll - 1))
        if df > 0.3 * n_docs:
            score *= 0.3
        scored.append((score, phrase))
    scored.sort(reverse=True)
    chosen = []
    primary = next((p for _, p in scored if contains_product_noun(p, lang) and len(p.split()) >= 2), None)
    if primary is None:
        primary = next((p for _, p in scored if contains_product_noun(p, lang)), None)
    if primary:
        chosen.append(primary)
    for _, phrase in scored:
        if len(chosen) >= MAX_KEYWORDS:
            break
        overlap = any(phrase in c or c in phrase for c in chosen)
        if not overlap:
            chosen.append(phrase)
    return chosen


def parse_attributes(raw, amazon):
    """Đọc thuộc tính có cấu trúc làm dữ kiện cho brief."""
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = {}
    else:
        value = raw if isinstance(raw, dict) else {}
    if not isinstance(value, dict):
        return {}
    if amazon:
        allowed = ["Department", "Color", "Material", "Style", "Country of Origin", "Closure Type", "Pattern"]
        return {k: str(value[k]) for k in allowed if k in value and str(value[k]).strip()}
    # Tiki: bỏ giá trị quá dài (thường là bảng size dán vào ô thông số).
    return {k: str(v) for k, v in value.items() if str(v).strip() and len(str(v)) <= 120}


def split_indices(d):
    """Chia train/val/test 80/10/10, phân tầng theo danh mục khi đủ mẫu (seed cố định)."""
    labels = d["category_l2"].fillna("Không rõ").astype(str)
    ids = np.arange(len(d))
    strat = labels if labels.value_counts().min() >= 5 else None
    train, rest = train_test_split(ids, test_size=0.2, random_state=SEED, stratify=strat)
    rest_labels = labels.iloc[rest]
    strat2 = rest_labels if rest_labels.value_counts().min() >= 2 else None
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=strat2)
    out = np.array(["train"] * len(d), dtype=object)
    out[val] = "val"
    out[test] = "test"
    return out


def build_inputs(d, amazon):
    """Trích từ khóa cho từng sản phẩm, ghi gen_inputs và trả về thống kê theo danh mục."""
    lang = "en" if amazon else "vi"
    label = "popular" if amazon else "best_seller"
    brands = d["brand"] if "brand" in d else pd.Series("", index=d.index)
    all_segments = [name_segments(n, b, lang) for n, b in zip(d["name"], brands)]
    all_phrases = [candidate_phrases(s, lang) for s in all_segments]
    df_counts = Counter(p for phrases in all_phrases for p in phrases)
    keywords = []
    for i, phrases in enumerate(all_phrases):
        chosen = select_keywords(phrases, df_counts, len(d), lang)
        if len(chosen) < MIN_KEYWORDS:
            # Tên quá ngắn: bổ sung cụm hiếm hơn (df ≥ 1) có trong chính tên sản phẩm.
            extra = sorted(phrases, key=lambda p: (-len(p.split()), -df_counts.get(p, 0)))
            for phrase in extra:
                if len(chosen) >= MIN_KEYWORDS:
                    break
                if not any(phrase in c or c in phrase for c in chosen):
                    chosen.append(phrase)
        keywords.append(chosen)
    attr_col = "details" if amazon else "attributes_json"
    rows = []
    for i, (_, r) in enumerate(d.iterrows()):
        rows.append({
            "product_id": r["product_id"],
            "name": r["name"],
            "category_l2": r["category_l2"],
            "leaf_category_id": r.get("leaf_category_id"),
            "keywords": json.dumps(keywords[i], ensure_ascii=False),
            "attributes": json.dumps(parse_attributes(r.get(attr_col, "{}"), amazon), ensure_ascii=False),
            "price": r.get("price") if not amazon else None,
            "reference_description": r["description_text"],
            label: r[label],
        })
    out = pd.DataFrame(rows)
    out["split"] = split_indices(d)
    # Giữ cố định tập kiểm tra đã dùng khi sinh nội dung (test_items_<dataset>.csv) để khi chạy lại tiền xử lý,
    # các sản phẩm đó không lọt vào tập huấn luyện LoRA hay tập ví dụ truy hồi (tránh rò rỉ dữ liệu kiểm tra).
    test_file = RESULTS / ("test_items_amazon.csv" if amazon else "test_items_tiki.csv")
    if test_file.exists():
        fixed = set(pd.read_csv(test_file, dtype={"product_id": str})["product_id"])
        is_fixed = out["product_id"].astype(str).isin(fixed)
        out.loc[is_fixed, "split"] = "test"
        out.loc[(~is_fixed) & (out["split"] == "test"), "split"] = "val"
    name = "gen_inputs_amazon.csv" if amazon else "gen_inputs_tiki.csv"
    out.to_csv(PROCESSED / name, index=False, encoding="utf-8-sig")
    # Thống kê theo danh mục: tần suất và độ nâng (lift) của từ khóa ở nhóm bán chạy.
    by_category = {}
    for cat, idx in d.groupby("category_l2").indices.items():
        counter = Counter(k for j in idx for k in keywords[j])
        best = Counter(k for j in idx if d.iloc[j][label] == 1 for k in keywords[j])
        n_cat = len(idx)
        n_best = int((d.iloc[idx][label] == 1).sum())
        entries = []
        for kw, cnt in counter.most_common(15):
            share_all = cnt / n_cat
            share_best = best.get(kw, 0) / n_best if n_best else None
            lift = share_best / share_all if share_best is not None and share_all > 0 else None
            entries.append({"keyword": kw, "count": cnt, "share": round(share_all, 4),
                            "share_best_seller": None if share_best is None else round(share_best, 4),
                            "lift_best_seller": None if lift is None else round(lift, 3)})
        by_category[str(cat)] = {"n_products": n_cat, "n_best_seller": n_best, "top_keywords": entries}
    n_kw = [len(k) for k in keywords]
    stats = {"n_products": len(d), "keywords_per_product_mean": float(np.mean(n_kw)),
             "keywords_per_product_min": int(np.min(n_kw)), "keywords_per_product_max": int(np.max(n_kw)),
             "min_df": MIN_DF, "split_counts": out["split"].value_counts().to_dict()}
    return by_category, stats


def plot_top_keywords(by_category, path, source):
    """Vẽ top 8 từ khóa (tỷ lệ sản phẩm có chứa) cho 6 danh mục lớn nhất."""
    setup_plot()
    cats = sorted(by_category, key=lambda c: -by_category[c]["n_products"])[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, cat in zip(axes.flat, cats):
        entries = by_category[cat]["top_keywords"][:8][::-1]
        ax.barh([e["keyword"] for e in entries], [100 * e["share"] for e in entries], color="#2A9D8F")
        ax.set_title(f"{cat} (n = {by_category[cat]['n_products']})", fontsize=11)
        ax.set_xlabel("% sản phẩm có từ khóa trong tên")
    for ax in list(axes.flat)[len(cats):]:
        ax.axis("off")
    fig.suptitle("Từ khóa đặc trưng theo danh mục (trích từ tên sản phẩm)", fontsize=14, fontweight="bold")
    save_figure(fig, path, source)


def main():
    """Chạy trích từ khóa cho Tiki và Amazon."""
    result = {}
    tiki = pd.read_csv(PROCESSED / "tiki_features.csv")
    if len(tiki):
        by_cat, stats = build_inputs(tiki, amazon=False)
        result["tiki"] = {"stats": stats, "categories": by_cat}
        plot_top_keywords(by_cat, FIGURES / "keywords_01_danh_muc.png", source_text(tiki))
    amazon_path = PROCESSED / "amazon_features.csv"
    if amazon_path.exists():
        amazon = pd.read_csv(amazon_path)
        if len(amazon):
            by_cat, stats = build_inputs(amazon, amazon=True)
            result["amazon"] = {"stats": stats, "categories": by_cat}
    save_json(RESULTS / "keywords_by_category.json", result)
    print("Đã tạo đầu vào sinh:", {k: v["stats"]["n_products"] for k, v in result.items()})


if __name__ == "__main__":
    main()

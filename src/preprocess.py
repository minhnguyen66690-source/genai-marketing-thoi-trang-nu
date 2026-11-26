"""Chuẩn hóa sản phẩm và ghi nhật ký các bước lọc."""
import json, re
from datetime import date, timedelta
import numpy as np
import pandas as pd
from common import RAW, PROCESSED, RESULTS, clean_text, save_json, strip_tiki_boilerplate

COLUMNS = "product_id name brand category_l1 category_l2 category_l3 leaf_category_id price list_price discount_rate rating_average review_count quantity_sold created_date product_age_days within_7y description_text short_description meta_title meta_description meta_keywords n_images has_video n_specs chat_lieu xuat_xu attributes_json n_options crawled_at".split()
TEXT_COLUMNS = "title_len_chars title_len_words desc_len_chars desc_len_words n_paragraphs n_bullets n_sentences avg_sentence_len_words n_exclamations uppercase_ratio emoji_count has_size_info has_material_info has_care_info has_cta has_meta_description keyword_in_title_first60".split()

def text_features(title, description, meta=""):
    """Tính đặc trưng văn bản từ tiêu đề và mô tả sản phẩm."""
    title, d = clean_text(title), clean_text(description)
    sentences = [x for x in re.split(r"[.!?]+|\n+", d) if x.strip()]
    letters = [c for c in d if c.isalpha()]
    words = re.findall(r"\b\w+\b", d, re.UNICODE)
    def has(pattern):
        """Kiểm tra cụm từ không phân biệt hoa thường."""
        return int(bool(re.search(pattern, d, re.I)))
    result = dict(title_len_chars=len(title), title_len_words=len(title.split()), desc_len_chars=len(d), desc_len_words=len(words), n_paragraphs=len([x for x in d.splitlines() if x.strip()]), n_bullets=sum(bool(re.match(r"^\s*(?:[-*•✔✅]|\d+[.)]|[\U0001F300-\U0001FAFF])", x)) for x in d.splitlines()), n_sentences=len(sentences), avg_sentence_len_words=len(words)/len(sentences) if sentences else 0, n_exclamations=d.count("!"), uppercase_ratio=sum(c.isupper() for c in letters)/len(letters) if letters else 0, emoji_count=len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", d)), has_size_info=int(has(r"\b(?:size|kích thước|kích cỡ|số đo|cân nặng|chiều cao|bảng size)\b") or bool(re.search(r"\b(?:S|M|L|XL|XXL|XXXL)\b", d))), has_material_info=has(r"\b(?:chất liệu|chat lieu|vải|vai|cotton|linen|lụa|lua|kaki|jean|thun|voan|len lông|nỉ)\b"), has_care_info=has(r"\b(?:giặt|bảo quản|ủi|phơi|tẩy)\b"), has_cta=has(r"\b(?:mua ngay|đặt hàng|dat hang|đặt mua|dat mua|thêm vào giỏ|them vao gio|inbox|liên hệ|lien he|nhắn tin|nhan tin|sở hữu ngay|so huu ngay)\b"), has_meta_description=int(bool(clean_text(meta))), keyword_in_title_first60=np.nan)
    return result

def flatten_specs(specs):
    """Gom các thuộc tính thông số từ từng nhóm."""
    out = []
    for group in specs if isinstance(specs,list) else []:
        if not isinstance(group,dict): continue
        attrs = group.get("attributes")
        out.extend(attrs if isinstance(attrs,list) else [group])
    return [x for x in out if isinstance(x,dict)]

def make_row(x):
    """Chuyển một bản ghi Tiki thành hàng đặc trưng."""
    crumbs=x.get("breadcrumbs") or []
    if isinstance(crumbs,dict): crumbs=crumbs.get("data") or []
    names=[a.get("name") for a in crumbs if isinstance(a,dict) and a.get("name")]
    if "Thời trang nữ" in names: names=names[names.index("Thời trang nữ"):]
    names=(names+[None]*3)[:3]
    brand=x.get("brand") or {}; brand=brand.get("name") if isinstance(brand,dict) else brand
    sold=x.get("quantity_sold"); sold=sold.get("value") if isinstance(sold,dict) else sold
    if sold is None: sold=x.get("all_time_quantity_sold")
    stamp=x.get("_crawled_at"); days=x.get("day_ago_created")
    try: created=(date.fromisoformat(str(stamp)[:10])-timedelta(days=int(days))).isoformat()
    except (ValueError, TypeError, OverflowError): created=None
    specs=flatten_specs(x.get("specifications"))
    attributes = {}
    spec_names = {"chất liệu": "Chất liệu", "xuất xứ (made in)": "Xuất xứ", "kích thước": "Kích thước"}
    for item in specs:
        key = spec_names.get(clean_text(item.get("name")).strip().casefold())
        value = clean_text(item.get("value"))
        if key and value:
            attributes[key] = value
    option_names = {"màu": "Màu sắc", "màu sắc": "Màu sắc", "color": "Màu sắc", "size": "Kích cỡ", "kích cỡ": "Kích cỡ", "kích thước": "Kích cỡ"}
    for option in x.get("configurable_options") or []:
        if not isinstance(option, dict):
            continue
        key = option_names.get(clean_text(option.get("name")).strip().casefold())
        if not key:
            continue
        values = [clean_text(v.get("label") or v.get("value")) if isinstance(v, dict) else clean_text(v) for v in option.get("values") or []]
        values = list(dict.fromkeys(v for v in values if v))[:8]
        if values:
            attributes[key] = ", ".join(values)
    def spec(pattern):
        """Tìm giá trị thông số đầu tiên khớp mẫu."""
        return next((clean_text(s.get("value")) for s in specs if re.search(pattern,str(s.get("name") or ""),re.I)), "")
    desc=strip_tiki_boilerplate(clean_text(x.get("description"))); meta=clean_text(x.get("meta_description"))
    row=dict(product_id=x.get("id"), name=clean_text(x.get("name")), brand=clean_text(brand), category_l1=names[0], category_l2=names[1], category_l3=names[2], leaf_category_id=x.get("_leaf_category_id"), price=x.get("price"), list_price=x.get("list_price"), discount_rate=x.get("discount_rate"), rating_average=x.get("rating_average"), review_count=x.get("review_count"), quantity_sold=sold or 0, created_date=created, product_age_days=days, within_7y=bool(created and created >= "2019-11-20"), description_text=desc, short_description=clean_text(x.get("short_description")), meta_title=clean_text(x.get("meta_title")), meta_description=meta, meta_keywords=clean_text(x.get("meta_keywords")), n_images=len(x.get("images") or []), has_video=int(bool(x.get("videos") or x.get("video_url"))), n_specs=len(specs), chat_lieu=spec(r"chất liệu|chat lieu|material"), xuat_xu=spec(r"xuất xứ|xuat xu|origin"), n_options=len(x.get("configurable_options") or []), crawled_at=stamp)
    row["attributes_json"] = json.dumps(attributes, ensure_ascii=False)
    row.update(text_features(row["name"],desc,meta)); return row

def read_raw(path):
    """Đọc các dòng JSON hợp lệ và đếm dòng lỗi."""
    rows=[]; invalid=0
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj=json.loads(line)
                    if isinstance(obj,dict): rows.append(obj)
                    else: invalid+=1
                except json.JSONDecodeError: invalid+=1
    return rows,invalid

def main():
    """Lọc dữ liệu Tiki, gán nhãn và xử lý Amazon."""
    raw,invalid=read_raw(RAW/"tiki_products_raw.jsonl")
    d=pd.DataFrame([make_row(x) for x in raw],columns=COLUMNS+TEXT_COLUMNS)
    funnel={"raw_valid":len(d),"invalid_json_or_incomplete":invalid,"raw_file_exists":(RAW/"tiki_products_raw.jsonl").exists(),"missing_raw_fields":{c:int((d[c].isna() | d[c].astype(str).str.strip().eq("")).sum()) for c in COLUMNS},"duplicate_name_brand_raw":int(d.duplicated(["name","brand"]).sum())}
    d=d[d.within_7y.astype(bool)].copy(); funnel["within_7y"]=len(d)
    d=d[d.desc_len_chars.ge(50)].copy(); funnel["description_at_least_50"]=len(d)
    d=d.drop_duplicates(["name","brand"]); funnel["deduplicated"]=len(d)
    d["category_l2"] = d["category_l2"].fillna("Khác").replace("", "Khác")
    group_sizes = d.groupby("category_l2")["product_id"].transform("size")
    d.loc[group_sizes < 30, "category_l2"] = "Khác"
    d["quantity_sold"]=pd.to_numeric(d.quantity_sold,errors="coerce").fillna(0)
    # Hòa cùng giá trị được xếp cùng hạng; tỷ lệ thực tế có thể khác 25%.
    rank=d.groupby("category_l2").quantity_sold.rank(method="min",ascending=False)
    sizes=d.groupby("category_l2").quantity_sold.transform("size")
    d["best_seller"]=((d.quantity_sold>0)&(rank<=np.ceil(sizes*.25))).astype(int)
    funnel["ranking_groups"] = int(d.category_l2.nunique())
    funnel["positive_count"] = int(d.best_seller.sum())
    funnel["positive_rate"] = float(d.best_seller.mean()) if len(d) else 0.0
    d.to_csv(PROCESSED/"tiki_features.csv",index=False,encoding="utf-8-sig")
    save_json(RESULTS/"data_funnel.json",funnel)
    amazon=PROCESSED/"amazon2023_women.csv"
    if not amazon.exists(): print("CẢNH BÁO: Chưa có data/processed/amazon2023_women.csv")
    else: preprocess_amazon(amazon)
    if not raw: print("CẢNH BÁO: Không có bản ghi raw Tiki hợp lệ")
    print(f"Sản phẩm sau lọc: {len(d)}")

def _parse_structured(value):
    """Đọc trường dạng chuỗi JSON hoặc chuỗi Python (list/dict) trong file Amazon."""
    import ast
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def preprocess_amazon(path):
    """Chuẩn hóa dữ liệu Amazon Reviews 2023 để so sánh với Tiki.

    Chỉ giữ quần áo nữ: categories = [Clothing, Shoes & Jewelry, Women, Clothing, <danh mục lá>, ...].
    Danh mục so sánh (category_l2) là cấp thứ 4 của cây danh mục, ví dụ "Dresses", "Tops, Tees & Blouses".
    Nhãn popular = top 25% rating_number trong cùng danh mục (rating_number là số lượt đánh giá, dùng làm
    đại diện cho mức độ phổ biến vì Amazon không công bố số đã bán).
    """
    d = pd.read_csv(path)
    rows = []
    funnel = {"amazon_raw": len(d)}
    for _, r in d.iterrows():
        cats = _parse_structured(r.get("categories"))
        if not isinstance(cats, list) or len(cats) < 4:
            continue
        if cats[1] != "Women" or cats[2] != "Clothing":
            continue
        desc = _parse_structured(r.get("description"))
        desc = "\n".join(map(str, desc)) if isinstance(desc, list) else clean_text(desc)
        features = _parse_structured(r.get("features"))
        title = clean_text(r.get("title"))
        feats = text_features(title, desc)
        feats["n_bullets"] = len(features) if isinstance(features, list) else 0
        rows.append(dict(
            product_id=r.get("parent_asin"),
            name=title,
            brand=clean_text(r.get("store")),
            category_l2=cats[3],
            leaf_category_id=cats[3],
            description_text=desc,
            rating_number=r.get("rating_number"),
            average_rating=r.get("average_rating"),
            details=r.get("details"),
            **feats,
        ))
    a = pd.DataFrame(rows)
    funnel["amazon_women_clothing"] = len(a)
    a = a[a["desc_len_chars"] >= 50].copy()
    funnel["amazon_description_at_least_50"] = len(a)
    if len(a):
        a["rating_number"] = pd.to_numeric(a["rating_number"], errors="coerce").fillna(0)
        group = a.groupby("leaf_category_id")["rating_number"]
        ranks = group.rank(method="min", ascending=False)
        sizes = group.transform("size")
        a["popular"] = ((a["rating_number"] > 0) & (ranks <= np.ceil(sizes * 0.25))).astype(int)
        funnel["amazon_popular_rate"] = float(a["popular"].mean())
    a.to_csv(PROCESSED / "amazon_features.csv", index=False, encoding="utf-8-sig")
    save_json(RESULTS / "data_funnel_amazon.json", funnel)
if __name__=="__main__": main()

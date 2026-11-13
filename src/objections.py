"""Phân loại chủ đề phản đối (lý do không hài lòng) từ đánh giá 1–3 sao trên Tiki.

Cách làm: mỗi chủ đề có một từ điển biểu thức chính quy (viết có dấu, có cả dạng viết tắt phổ biến như
"ko", "k"). Một đánh giá có thể thuộc nhiều chủ đề. Với từng danh mục cấp 2, tính tỷ lệ đánh giá 1–3 sao
nhắc đến từng chủ đề; hai chủ đề phổ biến nhất được đưa vào brief làm "phản đối thường gặp" để kịch bản
chốt sale phải trả lời.

Đầu ra:
- _workspace/03_results/objections_by_category.json  {danh mục: [{"phrase": chủ đề, "count": n, "share": tỷ lệ}]}
- _workspace/03_results/objection_themes_summary.json (toàn bộ bảng đếm, để báo cáo)
- figures/eda_09_chu_de_phan_doi.png (bản đồ nhiệt chủ đề × danh mục)
"""
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

# Từ điển chủ đề phản đối thường gặp trong đánh giá thời trang (tác giả xây dựng từ đọc mẫu đánh giá).
THEMES = {
    "Size không vừa": r"size|không vừa|ko vừa|k vừa|chật|bị rộng|quá rộng|rộng quá|quá nhỏ|nhỏ quá|quá to|to quá|đổi size|form nhỏ|form to|sai size",
    "Không giống hình/mô tả": r"giống hình|như hình|khác hình|không giống|ko giống|k giống|khác mẫu|sai mẫu|đúng mẫu|không đúng|ko đúng|mô tả|khác xa",
    "Chất liệu kém": r"chất liệu|chất vải|vải mỏng|mỏng|vải xấu|vải cứng|xù|nhăn|phai|ra màu|bí|nóng|ngứa|thô",
    "Màu sắc khác": r"màu sắc|khác màu|sai màu|màu không|màu ko|màu bị|lệch màu",
    "Đường may, lỗi sản phẩm": r"đường may|chỉ thừa|rách|bung|lỗi|hư|dơ|bẩn|mốc|thủng|tuột",
    "Giao hàng, đóng gói": r"giao hàng|giao sai|giao thiếu|thiếu hàng|đóng gói|ship|vận chuyển|giao chậm|giao lâu",
}
MIN_REVIEWS = 10  # danh mục cần ít nhất 10 đánh giá 1–3 sao mới tính tỷ lệ


def load_low_reviews():
    """Đọc đánh giá 1–3 sao và gắn danh mục cấp 2 của sản phẩm."""
    reviews_path = common.PROCESSED / "tiki_reviews.csv"
    features_path = common.PROCESSED / "tiki_features.csv"
    if not reviews_path.exists() or not features_path.exists():
        return None
    reviews = pd.read_csv(reviews_path, dtype={"product_id": str})
    products = pd.read_csv(features_path, dtype={"product_id": str}, usecols=["product_id", "category_l2"])
    reviews = reviews.merge(products, on="product_id", how="inner")
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    low = reviews[reviews["rating"] <= 3].copy()
    low["text"] = (low["title"].fillna("") + " . " + low["content"].fillna("")).str.lower()
    return low


def main():
    """Đếm chủ đề phản đối theo danh mục, lưu JSON và vẽ bản đồ nhiệt."""
    low = load_low_reviews()
    if low is None or low.empty:
        print("Không có đánh giá 1–3 sao để phân tích.")
        return
    for theme, pattern in THEMES.items():
        low[theme] = low["text"].str.contains(pattern, regex=True)
    # Chỉ dùng phần nội dung do khách viết; tiêu đề mặc định của Tiki (ví dụ "Không hài lòng") không mang chủ đề.
    low["any_theme"] = low[list(THEMES)].any(axis=1)
    summary = {"n_low_reviews": int(len(low)), "share_with_any_theme": float(low["any_theme"].mean()),
               "overall": {t: {"count": int(low[t].sum()), "share": float(low[t].mean())} for t in THEMES},
               "by_category": {}}
    objections = {}
    for cat, g in low.groupby("category_l2"):
        counts = {t: int(g[t].sum()) for t in THEMES}
        summary["by_category"][cat] = {"n_low_reviews": int(len(g)), **{t: counts[t] for t in THEMES}}
        if len(g) < MIN_REVIEWS:
            continue
        ranked = sorted(THEMES, key=lambda t: -counts[t])
        objections[cat] = [{"phrase": t, "count": counts[t], "share": round(counts[t] / len(g), 3)}
                           for t in ranked if counts[t] > 0][:5]
    # Danh mục ít đánh giá: dùng thứ tự chủ đề của toàn bộ dữ liệu.
    overall_rank = sorted(THEMES, key=lambda t: -summary["overall"][t]["count"])
    for cat in summary["by_category"]:
        if cat not in objections:
            objections[cat] = [{"phrase": t, "count": summary["overall"][t]["count"],
                                "share": round(summary["overall"][t]["share"], 3), "fallback": "toàn bộ dữ liệu"}
                               for t in overall_rank[:5]]
    common.save_json(common.RESULTS / "objections_by_category.json", objections)
    common.save_json(common.RESULTS / "objection_themes_summary.json", summary)
    # Bản đồ nhiệt cho các danh mục đủ mẫu.
    cats = [c for c, v in summary["by_category"].items() if v["n_low_reviews"] >= MIN_REVIEWS]
    matrix = np.array([[summary["by_category"][c][t] / summary["by_category"][c]["n_low_reviews"] for t in THEMES] for c in cats])
    common.setup_plot()
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(cats) + 2.2))
    im = ax.imshow(matrix * 100, cmap="Oranges", aspect="auto", vmin=0)
    ax.set_xticks(range(len(THEMES)))
    ax.set_xticklabels(list(THEMES), rotation=20, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{c} (n = {summary['by_category'][c]['n_low_reviews']})" for c in cats])
    for i in range(len(cats)):
        for j in range(len(THEMES)):
            ax.text(j, i, f"{matrix[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > 0.45 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% đánh giá 1–3 sao nhắc đến chủ đề")
    ax.set_title("Chủ đề phản đối trong đánh giá 1–3 sao theo danh mục", fontsize=13, fontweight="bold")
    common.save_figure(fig, common.FIGURES / "eda_09_chu_de_phan_doi.png",
                       "Nguồn: đánh giá Tiki thu thập ngày 13/11/2026, phân loại bằng từ điển chủ đề của tác giả")
    print(json.dumps({t: round(v["share"], 3) for t, v in summary["overall"].items()}, ensure_ascii=False),
          "| n =", summary["n_low_reviews"], "| có chủ đề:", round(summary["share_with_any_theme"], 3))


if __name__ == "__main__":
    main()

"""Thống kê mô tả và lưu các hình phân tích dữ liệu."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import PROCESSED, RESULTS, FIGURES, setup_plot, save_figure, source_text, save_json


def wilson(k, n):
    """Tính khoảng tin cậy Wilson 95% cho một tỷ lệ."""
    if not n:
        return 0, 0
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    delta = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - delta, center + delta


def main():
    """Tạo bảng thống kê và các biểu đồ từ dữ liệu đã xử lý."""
    setup_plot()
    d = pd.read_csv(PROCESSED / "tiki_features.csv")
    funnel = json.loads((RESULTS / "data_funnel.json").read_text(encoding="utf-8"))
    source = source_text(d)
    summary = {"n": len(d), "funnel": funnel, "descriptive": {}, "figures": {}}
    columns = ["price", "quantity_sold", "desc_len_words", "n_images", "n_bullets", "product_age_days", "discount_rate", "rating_average", "review_count", "best_seller"]
    for column in columns:
        values = pd.to_numeric(d[column], errors="coerce").dropna()
        summary["descriptive"][column] = {"count": len(values), "mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max()}
    if d.empty:
        save_json(RESULTS / "eda_summary.json", summary)
        print("Không có sản phẩm để vẽ EDA.")
        return

    def finish(fig, name, data, caption=source):
        """Lưu hình và dữ liệu dùng để vẽ hình."""
        save_figure(fig, FIGURES / name, caption)
        summary["figures"][name] = data

    steps = {"Hợp lệ": funnel["raw_valid"], "Trong 7 năm": funnel["within_7y"], "Mô tả đủ dài": funnel["description_at_least_50"], "Khử trùng lặp": funnel["deduplicated"]}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(steps), list(steps.values()))
    ax.invert_yaxis()
    ax.set(title="Các bước lọc dữ liệu", xlabel="Số sản phẩm")
    finish(fig, "eda_01_phieu_loc.png", steps)

    counts = d.category_l2.fillna("Khác").value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    counts.plot.barh(ax=ax)
    ax.set(title="Sản phẩm theo danh mục cấp 2", xlabel="Số sản phẩm", ylabel="Danh mục")
    finish(fig, "eda_02_danh_muc.png", counts.to_dict())

    prices = pd.to_numeric(d.price, errors="coerce").dropna()
    prices = prices[prices > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    values, edges, _ = ax.hist(np.log10(prices), bins=25)
    ax.set(title="Phân phối giá bán", xlabel="Log10 giá (đồng)", ylabel="Số sản phẩm")
    finish(fig, "eda_03_gia.png", {"counts": values, "bin_edges_log10": edges, "n_positive": len(prices)})

    sold = pd.to_numeric(d.quantity_sold, errors="coerce").fillna(0).clip(lower=0)
    zero_rate = float((sold == 0).mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    values, edges, _ = ax.hist(np.log1p(sold), bins=25)
    ax.set(title=f"Phân phối số lượng đã bán; chưa bán: {zero_rate:.1%}", xlabel="Log(1 + số lượng đã bán)", ylabel="Số sản phẩm")
    finish(fig, "eda_04_da_ban.png", {"counts": values, "bin_edges_log1p": edges, "zero_rate": zero_rate})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    distributions = {}
    for ax, column, label in zip(axes, ["desc_len_words", "n_images"], ["Độ dài mô tả (từ)", "Số ảnh (ảnh)"]):
        values, edges, _ = ax.hist(pd.to_numeric(d[column], errors="coerce").dropna(), bins=25)
        ax.set(xlabel=label, ylabel="Số sản phẩm")
        distributions[column] = {"counts": values, "bin_edges": edges}
    fig.suptitle("Phân phối nội dung và hình ảnh")
    finish(fig, "eda_05_mo_ta_va_anh.png", distributions)

    groups = [d.loc[d.best_seller == label, "desc_len_words"].dropna() for label in (0, 1)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, tick_labels=["Khác", "Bán chạy"])
    ax.set(title="Độ dài mô tả theo nhãn bán chạy", ylabel="Độ dài mô tả (từ)")
    finish(fig, "eda_06_boxplot.png", {str(i): {"count": len(g), "min": g.min(), "q1": g.quantile(.25), "median": g.median(), "q3": g.quantile(.75), "max": g.max()} for i, g in enumerate(groups)})

    numeric = ["price", "quantity_sold", "desc_len_words", "n_images", "n_bullets", "product_age_days", "discount_rate", "best_seller"]
    labels = ["Giá", "Đã bán", "Từ mô tả", "Số ảnh", "Gạch đầu dòng", "Tuổi sản phẩm", "Giảm giá", "Bán chạy"]
    corr = d[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Tương quan Spearman")
    fig.colorbar(image, ax=ax, label="Hệ số tương quan")
    finish(fig, "eda_07_tuong_quan.png", corr.to_dict())

    # Nhóm dễ đọc: ngũ phân vị số từ (nhãn "a–b từ"), nhóm số gạch đầu dòng, biến nhị phân Không/Có.
    quint = pd.qcut(d.desc_len_words, q=5, duplicates="drop")
    quint_labels = {iv: f"{int(np.ceil(iv.left))}–{int(iv.right)}" for iv in quint.cat.categories}
    length_group = quint.map(quint_labels)
    length_group = pd.Categorical(length_group, categories=list(quint_labels.values()), ordered=True)
    bullet_group = pd.cut(d.n_bullets, [-1, 0, 2, 5, 10_000], labels=["0", "1–2", "3–5", "> 5"])
    yes_no = {0: "Không", 1: "Có"}
    features = [
        (length_group, "Số từ mô tả (ngũ phân vị)"),
        (bullet_group, "Số gạch đầu dòng"),
        (d.has_size_info.map(yes_no), "Có thông tin size"),
        (d.has_material_info.map(yes_no), "Có thông tin chất liệu"),
        (d.has_care_info.map(yes_no), "Có hướng dẫn bảo quản"),
        (d.has_cta.map(yes_no), "Có lời kêu gọi hành động"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    rates = {}
    for ax, (group, title) in zip(axes.flat, features):
        table = d.groupby(group, observed=True).best_seller.agg(["sum", "count"])
        if set(table.index) == {"Không", "Có"}:
            table = table.loc[["Không", "Có"]]
        entries = []
        for label, row in table.iterrows():
            low, high = wilson(row["sum"], row["count"])
            entries.append({"group": str(label), "positive": int(row["sum"]), "n": int(row["count"]), "rate": row["sum"] / row["count"], "wilson_low": low, "wilson_high": high})
        rates[title] = entries
        y = [100 * entry["rate"] for entry in entries]
        error = [[100 * (entry["rate"] - entry["wilson_low"]) for entry in entries], [100 * (entry["wilson_high"] - entry["rate"]) for entry in entries]]
        bars = ax.bar([entry["group"] for entry in entries], y, yerr=error, capsize=4, color="#2A9D8F")
        for bar, entry in zip(bars, entries):
            ax.text(bar.get_x() + bar.get_width() / 2, 1, f"n={entry['n']}", ha="center", va="bottom", fontsize=9, color="white")
        ax.axhline(100 * d.best_seller.mean(), color="#E76F51", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Tỷ lệ bán chạy (%)")
        ax.set_ylim(0, 45)
    fig.suptitle("Tỷ lệ sản phẩm bán chạy theo đặc trưng nội dung (thanh lỗi: KTC Wilson 95%; nét đứt: tỷ lệ chung)", fontsize=13, fontweight="bold")
    finish(fig, "eda_08_ty_le_wilson.png", rates)

    review_path = PROCESSED / "tiki_reviews.csv"
    if review_path.exists():
        try:
            reviews = pd.read_csv(review_path)
            if len(reviews) and {"rating", "content"}.issubset(reviews.columns):
                ratings = pd.to_numeric(reviews.rating, errors="coerce").value_counts().sort_index()
                lengths = reviews.content.fillna("").astype(str).str.split().str.len()
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                ratings.plot.bar(ax=axes[0])
                axes[0].set(xlabel="Điểm đánh giá (sao)", ylabel="Số đánh giá")
                values, edges, _ = axes[1].hist(lengths, bins=25)
                axes[1].set(xlabel="Độ dài đánh giá (từ)", ylabel="Số đánh giá")
                fig.suptitle("Phân phối đánh giá")
                finish(fig, "eda_09_danh_gia.png", {"ratings": ratings.to_dict(), "length_counts": values, "length_bin_edges": edges}, "Nguồn: data/processed/tiki_reviews.csv")
        except pd.errors.EmptyDataError:
            pass

    amazon_path = PROCESSED / "amazon_features.csv"
    if amazon_path.exists():
        amazon = pd.read_csv(amazon_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        comparison = {}
        for ax, column, label in zip(axes, ["desc_len_words", "n_bullets"], ["Độ dài mô tả (từ)", "Số gạch đầu dòng"]):
            left = pd.to_numeric(d[column], errors="coerce").dropna()
            right = pd.to_numeric(amazon[column], errors="coerce").dropna()
            counts, edges, _ = ax.hist([left, right], bins=30, density=True, alpha=.6, label=["Tiki", "Amazon"])
            ax.set(xlabel=label, ylabel="Mật độ")
            ax.legend()
            comparison[column] = {"tiki_n": len(left), "amazon_n": len(right), "density": counts, "bin_edges": edges}
        fig.suptitle("So sánh Tiki và Amazon")
        finish(fig, "eda_10_so_sanh_amazon.png", comparison, "Nguồn: data/processed/tiki_features.csv và amazon_features.csv")
    save_json(RESULTS / "eda_summary.json", summary)
    print(f"Đã phân tích {len(d)} sản phẩm Tiki.")


if __name__ == "__main__":
    main()

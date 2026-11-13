"""RQ1 (bổ sung): liên hệ có kiểm soát giữa đặc trưng nội dung và nhãn bán chạy.

Mô hình hồi quy Logistic (statsmodels) với biến kiểm soát: log giá, tỷ lệ giảm giá, log tuổi sản phẩm,
số ảnh, số tùy chọn và danh mục cấp 2 (biến giả). Biến nội dung liên tục được chuẩn hóa z-score để
tỷ số chênh (odds ratio, OR) đọc là "khi biến tăng 1 độ lệch chuẩn". Biến nhị phân giữ nguyên 0/1.
Kết quả là liên hệ thống kê sau khi kiểm soát, KHÔNG phải tác động nhân quả.

Đầu ra: _workspace/03_results/rq1_logit_odds_ratios.json, figures/predictive_05_odds_ratio.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import FIGURES, PROCESSED, RESULTS, save_figure, save_json, setup_plot, source_text

CONTENT_CONTINUOUS = {
    "desc_len_words": "Số từ mô tả (log)",
    "n_bullets": "Số gạch đầu dòng (log)",
    "n_paragraphs": "Số đoạn (log)",
    "title_len_chars": "Độ dài tên sản phẩm",
    "uppercase_ratio": "Tỷ lệ chữ hoa",
}
CONTENT_BINARY = {
    "has_size_info": "Có thông tin size",
    "has_material_info": "Có thông tin chất liệu",
    "has_care_info": "Có hướng dẫn bảo quản",
    "has_cta": "Có lời kêu gọi hành động",
}
LOG_FEATURES = {"desc_len_words", "n_bullets", "n_paragraphs"}


def build_design(d):
    """Tạo ma trận thiết kế: biến nội dung (chuẩn hóa) + biến kiểm soát + biến giả danh mục."""
    x = pd.DataFrame(index=d.index)
    for col in CONTENT_CONTINUOUS:
        values = pd.to_numeric(d[col], errors="coerce").fillna(0)
        if col in LOG_FEATURES:
            values = np.log1p(values)
        x[col] = (values - values.mean()) / values.std()
    for col in CONTENT_BINARY:
        x[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    x["log_price"] = np.log1p(pd.to_numeric(d["price"], errors="coerce").fillna(0))
    x["discount_rate"] = pd.to_numeric(d["discount_rate"], errors="coerce").fillna(0) / 100
    x["log_age"] = np.log1p(pd.to_numeric(d["product_age_days"], errors="coerce").fillna(0))
    x["n_images"] = pd.to_numeric(d["n_images"], errors="coerce").fillna(0)
    x["n_options"] = pd.to_numeric(d["n_options"], errors="coerce").fillna(0)
    dummies = pd.get_dummies(d["category_l2"].fillna("Khác"), prefix="cat", drop_first=True, dtype=float)
    x = pd.concat([x, dummies], axis=1)
    return sm.add_constant(x.astype(float))


def main():
    """Ước lượng mô hình Logit, lưu bảng OR và vẽ biểu đồ rừng (forest plot)."""
    d = pd.read_csv(PROCESSED / "tiki_features.csv")
    y = d["best_seller"].astype(int)
    x = build_design(d)
    model = sm.Logit(y, x).fit(disp=False, maxiter=200)
    conf = model.conf_int()
    labels = {**CONTENT_CONTINUOUS, **CONTENT_BINARY,
              "log_price": "Log giá", "discount_rate": "Tỷ lệ giảm giá", "log_age": "Log tuổi sản phẩm (ngày)",
              "n_images": "Số ảnh", "n_options": "Số tùy chọn biến thể"}
    rows = []
    for name, label in labels.items():
        rows.append({
            "feature": name,
            "label": label,
            "odds_ratio": float(np.exp(model.params[name])),
            "ci_low": float(np.exp(conf.loc[name, 0])),
            "ci_high": float(np.exp(conf.loc[name, 1])),
            "p_value": float(model.pvalues[name]),
            "group": "content" if name in CONTENT_CONTINUOUS or name in CONTENT_BINARY else "control",
        })
    result = {
        "n": int(len(d)),
        "n_positive": int(y.sum()),
        "pseudo_r2_mcfadden": float(model.prsquared),
        "llr_p_value": float(model.llr_pvalue),
        "note": "Biến liên tục của nội dung chuẩn hóa z-score (một số biến lấy log1p trước); OR là liên hệ có kiểm soát, không phải nhân quả.",
        "coefficients": rows,
    }
    save_json(RESULTS / "rq1_logit_odds_ratios.json", result)
    setup_plot()
    content = [r for r in rows if r["group"] == "content"]
    content.sort(key=lambda r: r["odds_ratio"])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ypos = np.arange(len(content))
    for i, r in enumerate(content):
        color = "#E76F51" if r["p_value"] < 0.05 else "#8D99AE"
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, linewidth=2)
        ax.plot(r["odds_ratio"], i, "o", color=color, markersize=7)
    ax.axvline(1, color="#264653", linestyle="--", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in content])
    ax.set_xscale("log")
    ax.set_xlabel("Tỷ số chênh (OR) và khoảng tin cậy 95%, thang log")
    ax.set_title("Liên hệ có kiểm soát giữa đặc trưng nội dung và nhãn bán chạy\n(màu cam: p < 0,05; kiểm soát giá, giảm giá, tuổi, ảnh, biến thể, danh mục)", fontsize=11)
    save_figure(fig, FIGURES / "predictive_05_odds_ratio.png", source_text(d))
    print(json.dumps({r["label"]: (round(r["odds_ratio"], 3), round(r["p_value"], 4)) for r in rows}, ensure_ascii=False))
    print("pseudo R2:", round(model.prsquared, 4))


if __name__ == "__main__":
    main()

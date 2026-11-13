"""Sơ đồ mô phỏng thực nghiệm (D7): các bước thực nghiệm dạng ô, mỗi ô kèm kết quả đọc từ file kết quả.

Chạy lại sau mỗi lần cập nhật kết quả để số liệu trên sơ đồ luôn khớp với _workspace/03_results.
Đầu ra: figures/diag_07_mo_phong_thuc_nghiem.png (300 dpi) và .svg
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
FIGURES = ROOT / "figures"
NAVY, TEAL, ORANGE, LIGHT = "#1F4E79", "#2A9D8F", "#E76F51", "#EEF5F6"


def load(name):
    """Đọc một file JSON kết quả; trả về {} nếu chưa có."""
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def vn(x, digits=1):
    """Định dạng số kiểu Việt Nam."""
    if x is None:
        return "–"
    text = f"{x:,.{digits}f}" if digits else f"{int(round(x)):,}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def metric(summary, method, key, scale=1.0, digits=1):
    """Lấy giá trị trung bình của một chỉ số trong eval_summary."""
    try:
        return vn(summary["metrics"][method][key]["mean"] * scale, digits)
    except (KeyError, TypeError):
        return "–"


def steps():
    """Tạo danh sách 7 bước (tiêu đề, dòng kết quả) từ các file kết quả."""
    funnel = load("data_funnel.json")
    funnel_a = load("data_funnel_amazon.json")
    reviews = json.loads((ROOT / "data" / "raw" / "tiki_reviews_manifest.json").read_text(encoding="utf-8"))
    kw = load("keywords_by_category.json").get("tiki", {}).get("stats", {})
    pred = load("predictive_metrics.json")
    odds = {r["feature"]: r for r in load("rq1_logit_odds_ratios.json").get("coefficients", [])}
    leak = load("prompt_leakage.json").get("results", {}).get("tiki", {})
    tiki = load("eval_summary_tiki.json")
    agree = load("judge_agreement_amazon.json").get("pooled", {})
    best = pred.get("best_model_by_cv_auc", "")
    best_auc = pred.get("models", {}).get(best, {}).get("roc_auc")
    ab = pred.get("ablation", {})
    split = kw.get("split_counts", {})
    v1 = leak.get("M2 (v1)", {}).get("rate")
    v2 = leak.get("M2 (v2)", {}).get("rate")
    return [
        ("1. Thu thập dữ liệu", [f"Tiki: {vn(funnel.get('raw_valid'), 0)} SP",
                                 f"{vn(reviews.get('review_records'), 0)} đánh giá ẩn danh",
                                 f"Amazon: {vn(funnel_a.get('amazon_raw'), 0)} bản ghi"]),
        ("2. Làm sạch, lọc", ["Bỏ văn mẫu thuế của Tiki", "Lọc 7 năm, mô tả ≥ 50 ký tự",
                              f"Tiki {vn(funnel.get('deduplicated'), 0)} · Amazon {vn(funnel_a.get('amazon_description_at_least_50'), 0)} SP"]),
        ("3. Từ khóa, chia tập", [f"{vn(kw.get('keywords_per_product_mean'))} từ khóa/sản phẩm",
                                  f"Train {vn(split.get('train'), 0)} · Val {vn(split.get('val'), 0)}",
                                  f"Test {vn(split.get('test'), 0)} (cố định)"]),
        ("4. Dự báo (RQ1)", [f"{'XGBoost' if best == 'xgboost' else best.replace('_', ' ').title()}: AUC {vn(best_auc, 3)}",
                                       f"Thêm nội dung: +{vn(ab.get('delta_roc_auc'), 3)}",
                                       f"OR gạch đầu dòng {vn(odds.get('n_bullets', {}).get('odds_ratio'), 2)}"]),
        ("5. Sinh nội dung", ["M0–M3: Tiki 99, Amazon 50", "M4a/M4b (Qwen, LoRA): 59",
                              f"Rò rỉ: v1 {vn(None if v1 is None else 100 * v1, 0)}% → v2 {vn(None if v2 is None else 100 * v2, 0)}%"]),
        ("6. Đánh giá", [f"SEO M3 {metric(tiki, 'M3', 'seo_compliance_rate', 100)}% / gốc {metric(tiki, 'reference', 'seo_compliance_rate', 100)}%",
                         f"chrF++ M3 {metric(tiki, 'M3', 'chrf_plus_plus')}, M0 {metric(tiki, 'M0', 'chrf_plus_plus')}",
                         f"Đồng thuận giám khảo {vn(agree.get('quadratic_kappa'), 2)}"]),
        ("7. Triển khai", ["Website Streamlit", "Notebook Colab", "Mã nguồn GitHub"]),
    ]


def main():
    """Vẽ sơ đồ 7 ô theo hai hàng (4 + 3) với mũi tên nối."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(13.33, 6.2))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(6.66, 5.9, "Sơ đồ mô phỏng thực nghiệm và kết quả từng bước", ha="center", va="center", fontsize=17, fontweight="bold")
    width, height = 2.9, 2.25
    positions = [(0.35, 3.15), (3.6, 3.15), (6.85, 3.15), (10.1, 3.15), (10.1, 0.35), (6.85, 0.35), (3.6, 0.35)]
    for i, ((title, lines), (x, y)) in enumerate(zip(steps(), positions)):
        color = ORANGE if i in (3, 5) else (NAVY if i % 2 == 0 else TEAL)
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=LIGHT, edgecolor=color, linewidth=1.8))
        ax.add_patch(FancyBboxPatch((x, y + height - 0.55), width, 0.55, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=color, edgecolor=color))
        ax.text(x + width / 2, y + height - 0.27, title, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        for j, line in enumerate(lines):
            ax.text(x + 0.15, y + height - 0.85 - j * 0.48, "• " + line, ha="left", va="center", fontsize=10.5, color="#1D2B36")
    arrows = [((3.25, 4.25), (3.6, 4.25)), ((6.5, 4.25), (6.85, 4.25)), ((9.75, 4.25), (10.1, 4.25)),
              ((11.55, 3.15), (11.55, 2.6)), ((10.1, 1.45), (9.75, 1.45)), ((6.85, 1.45), (6.5, 1.45))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, color=NAVY, linewidth=2))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "diag_07_mo_phong_thuc_nghiem.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "diag_07_mo_phong_thuc_nghiem.svg", bbox_inches="tight")
    plt.close(fig)
    print("Đã vẽ diag_07")


if __name__ == "__main__":
    main()

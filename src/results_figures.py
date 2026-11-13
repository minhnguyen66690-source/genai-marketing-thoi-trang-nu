"""Biểu đồ kết quả cho bài báo và slide (đọc trực tiếp eval_summary_*.json, prompt_leakage.json, business_metrics.json).

res_01_tuong_dong.png   : chrF++ và ROUGE-L theo cấu hình, Tiki và Amazon, kèm khoảng tin cậy 95% (bootstrap).
res_02_checklist.png    : bản đồ nhiệt tỷ lệ đạt từng tiêu chí tối ưu hóa tìm kiếm và email (Tiki).
res_03_giam_khao.png    : điểm giám khảo Gemini theo tiêu chí và cấu hình M0–M3 (Tiki).
res_04_ro_ri_m4.png     : rò rỉ chỉ dẫn prompt v1/v2 và lỗi của mô hình nhỏ M4.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
FIGURES = ROOT / "figures"
COLORS = {"M0": "#8D99AE", "M1": "#E9C46A", "M2": "#2A9D8F", "M3": "#1F4E79", "M4a": "#E76F51", "M4b": "#B5523B", "reference": "#264653"}
NAMES = {"M0": "M0 Mẫu cố định", "M1": "M1 Zero-shot", "M2": "M2 Prompt eng.", "M3": "M3 Few-shot truy hồi",
         "M4a": "M4a Qwen gốc", "M4b": "M4b Qwen + LoRA", "reference": "Người bán"}
SOURCE = "Nguồn: tác giả thực nghiệm trên dữ liệu Tiki và Amazon Reviews 2023; thanh lỗi: KTC 95% bootstrap"


def setup():
    """Phông và kiểu chung cho mọi biểu đồ."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.grid": True, "grid.alpha": 0.2, "axes.axisbelow": True})


def load(name):
    """Đọc file JSON kết quả."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def stat(summary, method, key, scale=1.0):
    """(trung bình, sai số dưới, sai số trên) của một chỉ số; None nếu không có."""
    m = summary["metrics"].get(method, {}).get(key)
    if not m or m["mean"] is None:
        return None
    low, high = m["ci95"]
    return m["mean"] * scale, (m["mean"] - low) * scale, (high - m["mean"]) * scale


def save(fig, name, source=SOURCE):
    """Lưu hình 220 dpi kèm dòng nguồn."""
    fig.text(0.01, 0.005, source, fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_similarity():
    """Hai bảng con (Tiki, Amazon), mỗi bảng hai nhóm cột chrF++ và ROUGE-L × 100."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (dataset, title) in zip(axes, (("tiki", "Tiki (n = 99; M4a n = 59)"), ("amazon", "Amazon (n = 50)"))):
        s = load(f"eval_summary_{dataset}.json")
        methods = [m for m in ["M0", "M1", "M2", "M3", "M4a", "M4b"] if m in s["metrics"]]
        metrics = [("chrf_plus_plus", "chrF++", 1.0), ("rougeL", "ROUGE-L × 100", 100.0)]
        width = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals, errs = [], [[], []]
            for key, _, scale in metrics:
                v = stat(s, m, key, scale) or (0, 0, 0)
                vals.append(v[0])
                errs[0].append(v[1])
                errs[1].append(v[2])
            x = np.arange(len(metrics)) + (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(x, vals, width, yerr=errs, capsize=3, color=COLORS[m], label=NAMES[m])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.9, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([label for _, label, _ in metrics])
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 27)
    axes[0].set_ylabel("Điểm (cao hơn = gần mô tả người bán hơn)")
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    fig.suptitle("Mức tương đồng với mô tả thật của người bán (RQ2)", fontsize=14, fontweight="bold")
    save(fig, "res_01_tuong_dong.png")


def fig_checklist():
    """Bản đồ nhiệt tỷ lệ đạt (%) từng tiêu chí theo cấu hình (Tiki)."""
    s = load("eval_summary_tiki.json")
    methods = [m for m in ["M0", "M1", "M2", "M3", "M4a", "reference"] if m in s["metrics"]]
    rules = [("seo_title_len_50_60", "Tiêu đề 50–60 ký tự (S19)"), ("seo_meta_len_120_156", "Meta 120–156 ký tự (S12)"),
             ("seo_desc_words_ge_200", "Mô tả ≥ 200 từ (S13, S24)"), ("seo_keyphrase_in_title", "Từ khóa trong tiêu đề (S11)"),
             ("seo_keyphrase_in_first_paragraph", "Từ khóa ở đoạn đầu (S10)"), ("seo_compliance_rate", "Tuân thủ gộp"),
             ("email_mailchimp_le9w_le60c", "Email ≤ 9 từ, ≤ 60 ký tự (E01)"), ("email_omnisend_4_6w_or_21_40c", "Email 4–6 từ / 21–40 ký tự (E05)"),
             ("email_keyword_relevance", "Email chứa từ khóa")]
    matrix = np.full((len(rules), len(methods)), np.nan)
    for i, (key, _) in enumerate(rules):
        for j, m in enumerate(methods):
            v = stat(s, m, key, 100)
            if v:
                matrix[i, j] = v[0]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([NAMES[m] for m in methods], rotation=15)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([label for _, label in rules])
    ax.grid(False)
    for i in range(len(rules)):
        for j in range(len(methods)):
            v = matrix[i, j]
            text = "–" if np.isnan(v) else f"{v:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color="white" if (not np.isnan(v) and v > 60) else "black")
    ax.axhline(5.5, color="white", linewidth=3)
    fig.colorbar(image, ax=ax, label="% mẫu đạt tiêu chí")
    ax.set_title("Tỷ lệ đạt checklist tối ưu hóa tìm kiếm và tiêu đề email, Tiki (RQ3)", fontsize=13, fontweight="bold")
    save(fig, "res_02_checklist.png", "Nguồn: tác giả tính; mã tiêu chí theo mục 4.5; '–': không áp dụng (Tiki không công khai meta; M4 chỉ sinh mô tả)")


def fig_judge():
    """Thanh ngang: điểm giám khảo theo tiêu chí, nhóm theo cấu hình M0–M3 (Tiki)."""
    s = load("eval_summary_tiki.json")
    methods = ["M0", "M1", "M2", "M3"]
    criteria = [("judge_mo_ta_tu_nhien", "Mô tả: tự nhiên"), ("judge_mo_ta_thuyet_phuc", "Mô tả: thuyết phục"),
                ("judge_mo_ta_lien_quan_tu_khoa", "Mô tả: liên quan từ khóa"), ("judge_email_gay_to_mo", "Email: gây tò mò"),
                ("judge_kich_ban_cta_ro_rang", "Kịch bản: CTA rõ ràng"), ("judge_kich_ban_nhu_cau_loi_ich", "Kịch bản: nhu cầu – lợi ích"),
                ("judge_kich_ban_xu_ly_phan_doi", "Kịch bản: xử lý phản đối"), ("judge_kich_ban_dung_su_that", "Kịch bản: đúng sự thật")]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    height = 0.2
    for i, m in enumerate(methods):
        vals, errs = [], [[], []]
        for key, _ in criteria:
            v = stat(s, m, key) or (0, 0, 0)
            vals.append(v[0])
            errs[0].append(v[1])
            errs[1].append(v[2])
        y = np.arange(len(criteria)) + (i - 1.5) * height
        ax.barh(y, vals, height, xerr=errs, capsize=2, color=COLORS[m], label=NAMES[m])
    ax.set_yticks(range(len(criteria)))
    ax.set_yticklabels([label for _, label in criteria])
    ax.invert_yaxis()
    ax.set_xlim(1, 5.3)
    ax.set_xlabel("Điểm giám khảo Gemini (1–5)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=10, ncol=4, frameon=False)
    ax.set_title("Điểm giám khảo theo tiêu chí, Tiki n = 99 (RQ4)", fontsize=13, fontweight="bold")
    save(fig, "res_03_giam_khao.png", "Nguồn: tác giả; giám khảo Gemini 3.8 Flash chấm ẩn danh theo rubric G-Eval; thanh lỗi: KTC 95% bootstrap")


def fig_leakage_m4():
    """Trái: rò rỉ chỉ dẫn v1/v2. Phải: chép văn mẫu và sinh lặp của M4 so với M2."""
    leak = load("prompt_leakage.json")["results"]
    biz = load("business_metrics.json")["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels, v1, v2 = [], [], []
    for dataset, name in (("tiki", "Tiki"), ("amazon", "Amazon")):
        for method in ("M2", "M3"):
            a = leak.get(dataset, {}).get(f"{method} (v1)")
            b = leak.get(dataset, {}).get(f"{method} (v2)")
            if a and b:
                labels.append(f"{name} {method}")
                v1.append(100 * a["rate"])
                v2.append(100 * b["rate"])
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, v1, 0.36, color="#E76F51", label="Prompt v1")
    axes[0].bar(x + 0.18, v2, 0.36, color="#2A9D8F", label="Prompt v2")
    for i, (a, b) in enumerate(zip(v1, v2)):
        axes[0].text(i - 0.18, a + 1.5, f"{a:.0f}%", ha="center", fontsize=10)
        axes[0].text(i + 0.18, b + 1.5, f"{b:.0f}%", ha="center", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("% đầu ra bị rò rỉ chỉ dẫn")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].set_title("Vòng lặp cải tiến prompt", fontsize=12)
    keys = [k for k in ("M2", "M4a", "M4b (LoRA v1)", "M4b") if k in biz]
    names = {"M2": "M2 (GPT)", "M4a": "M4a Qwen gốc", "M4b (LoRA v1)": "M4b LoRA\n(chưa làm sạch)", "M4b": "M4b LoRA\n(đã làm sạch)"}
    x = np.arange(len(keys))
    copy = [biz[k]["pct_copied_tiki_boilerplate"] for k in keys]
    rep = [biz[k]["pct_repetitive_output"] for k in keys]
    axes[1].bar(x - 0.18, copy, 0.36, color="#B5523B", label="Chép văn mẫu Tiki")
    axes[1].bar(x + 0.18, rep, 0.36, color="#8D99AE", label="Sinh lặp")
    for i, (a, b) in enumerate(zip(copy, rep)):
        axes[1].text(i - 0.18, a + 1.5, f"{a:.0f}%", ha="center", fontsize=10)
        axes[1].text(i + 0.18, b + 1.5, f"{b:.0f}%", ha="center", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([names[k] for k in keys])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("% đầu ra")
    axes[1].legend()
    axes[1].set_title("Lỗi đặc trưng của mô hình nhỏ (Tiki)", fontsize=12)
    save(fig, "res_04_ro_ri_m4.png", "Nguồn: tác giả đo bằng biểu thức chính quy (src/prompt_leakage.py, src/business_metrics.py)")


def main():
    """Vẽ toàn bộ biểu đồ kết quả."""
    setup()
    FIGURES.mkdir(exist_ok=True)
    fig_similarity()
    fig_checklist()
    fig_judge()
    fig_leakage_m4()
    print("Đã vẽ res_01..res_04")


if __name__ == "__main__":
    main()

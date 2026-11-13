"""Tạo các bảng số liệu cho bài báo trực tiếp từ file kết quả (bảo đảm mọi con số truy được nguồn).

Mỗi bảng lưu thành paper/tables/<tên>.json theo định dạng mà src/build_docx.py đọc:
{"caption", "source", "columns", "rows", "col_widths_cm" (tùy chọn)}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def vn(x, digits=2):
    """Định dạng số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm phân cách hàng nghìn."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and float(x).is_integer() and digits == 0):
        return f"{int(round(x)):,}".replace(",", ".")
    text = f"{x:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def save(name, caption, source, columns, rows, widths=None):
    """Ghi một bảng JSON."""
    table = {"caption": caption, "source": source, "columns": columns, "rows": rows}
    if widths:
        table["col_widths_cm"] = widths
    (TABLES / f"{name}.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")


def table_data_description():
    """Bảng mô tả hai bộ dữ liệu."""
    funnel = json.loads((RESULTS / "data_funnel.json").read_text(encoding="utf-8"))
    funnel_a = json.loads((RESULTS / "data_funnel_amazon.json").read_text(encoding="utf-8"))
    reviews = json.loads((ROOT / "data" / "raw" / "tiki_reviews_manifest.json").read_text(encoding="utf-8"))
    rows = [
        ["Nguồn", "Tiki, danh mục Thời trang nữ (API công khai)", "Amazon Reviews 2023 (Hou và cộng sự, 2024)"],
        ["Thời điểm thu thập", "13/11/2026", "13/11/2026 (đọc tuần tự tệp công bố)"],
        ["Phạm vi thời gian", "Sản phẩm đăng từ 21/11/2019 đến 01/11/2026", "Date First Available từ 20/11/2019"],
        ["Bản ghi thô", vn(funnel["raw_valid"], 0), vn(funnel_a["amazon_raw"], 0)],
        ["Sau lọc 7 năm", vn(funnel["within_7y"], 0), "đã lọc khi đọc tệp"],
        ["Sau lọc mô tả ≥ 50 ký tự, bỏ trùng", vn(funnel["deduplicated"], 0), vn(funnel_a["amazon_description_at_least_50"], 0)],
        ["Đánh giá khách hàng", vn(reviews["review_records"], 0) + " (trang 1 của 500 sản phẩm bán chạy)", "không dùng"],
        ["Nhãn", f"Bán chạy: {vn(100 * funnel['positive_rate'], 1)}% (top 25% số đã bán trong danh mục)",
         f"Phổ biến: {vn(100 * funnel_a['amazon_popular_rate'], 1)}% (top 25% số lượt đánh giá)"],
    ]
    save("data_description", "Mô tả hai bộ dữ liệu sau các bước lọc",
         "Nguồn: tác giả tổng hợp từ dữ liệu thu thập", ["Thuộc tính", "Tiki (tiếng Việt)", "Amazon (tiếng Anh)"], rows, [4.2, 6.3, 5.5])


def table_descriptive():
    """Bảng thống kê mô tả các biến chính của Tiki."""
    d = pd.read_csv(PROCESSED / "tiki_features.csv")
    spec = [("price", "Giá bán (đồng)", 0), ("discount_rate", "Tỷ lệ giảm giá (%)", 1), ("quantity_sold", "Số đã bán", 0),
            ("product_age_days", "Tuổi sản phẩm (ngày)", 0), ("n_images", "Số ảnh", 1), ("n_options", "Số tùy chọn biến thể", 1),
            ("title_len_chars", "Độ dài tên (ký tự)", 1), ("desc_len_words", "Số từ mô tả", 1), ("n_bullets", "Số gạch đầu dòng", 1),
            ("n_paragraphs", "Số đoạn", 1)]
    rows = []
    for col, label, digits in spec:
        s = pd.to_numeric(d[col], errors="coerce")
        rows.append([label, vn(s.mean(), digits), vn(s.median(), digits), vn(s.std(), digits), vn(s.min(), 0), vn(s.max(), 0)])
    save("descriptive", f"Thống kê mô tả các biến chính của dữ liệu Tiki (n = {vn(len(d), 0)})",
         "Nguồn: tác giả tính từ dữ liệu Tiki thu thập ngày 13/11/2026",
         ["Biến", "Trung bình", "Trung vị", "Độ lệch chuẩn", "Nhỏ nhất", "Lớn nhất"], rows, [4.6, 2.4, 2.2, 2.6, 2.0, 2.2])


def table_predictive():
    """Bảng so sánh ba mô hình phân lớp và phân tích ablation."""
    p = json.loads((RESULTS / "predictive_metrics.json").read_text(encoding="utf-8"))
    names = {"logistic": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    rows = []
    for key, label in names.items():
        m = p["models"][key]
        rows.append([label, f"{vn(m['cv_mean'], 3)} ± {vn(m['cv_std'], 3)}", vn(m["roc_auc"], 3), vn(m["pr_auc"], 3),
                     vn(m["f1"], 3), vn(m["precision"], 3), vn(m["recall"], 3)])
    ab = p["ablation"]
    best_label = names.get(p.get("best_model_by_cv_auc"), p.get("best_model_by_cv_auc"))
    rows.append([f"{best_label}, chỉ biến kiểm soát", f"{vn(ab['control_cv_mean'], 3)} ± {vn(ab['control_cv_std'], 3)}",
                 vn(ab["control_roc_auc"], 3), "–", "–", "–", "–"])
    save("predictive", f"Kết quả phân lớp sản phẩm bán chạy (huấn luyện {vn(p['n_train'], 0)}, kiểm tra {vn(p['n_test'], 0)} sản phẩm)",
         "Nguồn: tác giả chạy thực nghiệm; AUC CV là trung bình ± độ lệch chuẩn của 5 phần kiểm định chéo trên tập huấn luyện",
         ["Mô hình", "ROC-AUC CV", "ROC-AUC kiểm tra", "PR-AUC", "F1", "Precision", "Recall"], rows, [4.4, 2.6, 2.0, 1.7, 1.6, 1.9, 1.6])


def table_odds():
    """Bảng tỷ số chênh của mô hình Logit có kiểm soát."""
    o = json.loads((RESULTS / "rq1_logit_odds_ratios.json").read_text(encoding="utf-8"))
    rows = []
    for r in o["coefficients"]:
        p = r["p_value"]
        p_text = "< 0,001" if p < 0.001 else vn(p, 3)
        ph = r.get("p_holm")
        ph_text = "–" if ph is None else ("< 0,001" if ph < 0.001 else vn(ph, 3))
        rows.append([r["label"], "Nội dung" if r["group"] == "content" else "Kiểm soát", vn(r["odds_ratio"], 2),
                     f"[{vn(r['ci_low'], 2)}; {vn(r['ci_high'], 2)}]", p_text, ph_text])
    save("odds_ratios", f"Tỷ số chênh (OR) của mô hình Logit có kiểm soát danh mục (n = {vn(o['n'], 0)}, pseudo R² McFadden = {vn(o['pseudo_r2_mcfadden'], 3)})",
         "Nguồn: tác giả ước lượng bằng statsmodels; biến nội dung liên tục chuẩn hóa z-score (biến đếm lấy log1p trước), OR ứng với tăng 1 độ lệch chuẩn; p Holm hiệu chỉnh cho 9 đặc trưng nội dung",
         ["Biến", "Nhóm", "OR", "KTC 95%", "p", "p Holm"], rows, [5.6, 2.0, 1.6, 3.2, 1.8, 1.8])


def table_related():
    """Bảng tổng quan nghiên cứu liên quan (nội dung từ các nguồn đã kiểm chứng ở mục 3)."""
    rows = [
        ["Zhou và cộng sự (2023)", "Mô tả sản phẩm Walmart", "Tinh chỉnh LLaMA 2 7B theo khía cạnh, kết hợp mục tiêu CTR", "BM25 66,44 → 78,65; NDCG người chấm 0,68 → 0,76", "Cần mô hình 7B, dữ liệu nhấp nội bộ"],
        ["Li và cộng sự (2024)", "Ảnh + từ khóa → mô tả", "ModICT: truy hồi sản phẩm tham chiếu, prompt động", "ROUGE-L tăng tối đa 3,3%", "Cần ảnh; chưa thử tiếng Việt"],
        ["Zhang & Tetreault (2019)", "Tiêu đề email", "Trích chọn và viết lại, học tăng cường", "Vượt baseline tự động và người", "Thư công việc, không có tỷ lệ mở"],
        ["Jobson và cộng sự (2025)", "Tiêu đề email marketing", "Gợi ý sản phẩm + mô hình ngôn ngữ lớn, A/B test", "Tỷ lệ mở +0,46% (không có ý nghĩa); nhấp sản phẩm +23,63%", "Một nền tảng, ngôn ngữ khác tiếng Việt"],
        ["Liu và cộng sự (2025)", "Quảng cáo theo từ khóa", "Truy hồi tăng cường + bộ đánh giá luật và giám khảo LLM", "Đồng thuận 89,57% với người duyệt", "Chưa đo doanh số"],
        ["Durmus và cộng sự (2024); Salvi và cộng sự (2025)", "Văn bản thuyết phục", "Đo thay đổi quan điểm, thử nghiệm ngẫu nhiên", "Ngang người viết; GPT-4 cá nhân hóa tăng 81,2% odds đồng thuận", "Đo thái độ, chưa đo đơn hàng"],
        ["Nghiên cứu này", "Thời trang nữ Tiki + Amazon", "So sánh M0–M4 trên cùng tập; danh sách kiểm tra có nguồn; dự báo bán chạy", "Xem mục 7", "Chưa thử nghiệm A/B thực tế"],
    ]
    save("related_works", "Tóm tắt các nghiên cứu liên quan", "Nguồn: tác giả tổng hợp",
         ["Nghiên cứu", "Bài toán", "Phương pháp", "Kết quả chính", "Hạn chế"], rows, [3.3, 2.7, 3.8, 3.6, 2.8])


METHOD_NAMES = {"M0": "M0 Mẫu cố định", "M1": "M1 Zero-shot", "M2": "M2 Thiết kế câu lệnh", "M3": "M3 Few-shot có truy hồi",
                "M4a": "M4a Qwen gốc", "M4b": "M4b Qwen + LoRA", "reference": "Nội dung người bán"}


def mean_of(summary, method, key, scale=1.0, digits=1):
    """Lấy trung bình một chỉ số trong eval_summary (định dạng Việt Nam), '–' nếu không áp dụng."""
    try:
        value = summary["metrics"][method][key]["mean"]
    except KeyError:
        return "–"
    return "–" if value is None else vn(value * scale, digits)


def load_summary(dataset):
    """Đọc eval_summary_<dataset>.json."""
    return json.loads((RESULTS / f"eval_summary_{dataset}.json").read_text(encoding="utf-8"))


def table_similarity():
    """Bảng RQ2: tương đồng với mô tả thật của người bán, hai bộ dữ liệu."""
    rows = []
    for dataset, label in (("tiki", "Tiki"), ("amazon", "Amazon")):
        s = load_summary(dataset)
        for m in ["M0", "M1", "M2", "M3", "M4a", "M4b"]:
            if m not in s["metrics"]:
                continue
            n = s["metrics"][m]["rougeL"]["n"]
            bleu = s.get("corpus_bleu_13a", {}).get(m)
            rows.append([label, METHOD_NAMES[m], str(n), "–" if bleu is None else vn(bleu, 2), mean_of(s, m, "rougeL", 1, 3),
                         mean_of(s, m, "chrf_plus_plus", 1, 1), mean_of(s, m, "bertscore_f1", 1, 3)])
    save("similarity", "Mức tương đồng giữa mô tả sinh ra và mô tả thật của người bán (RQ2)",
         "Nguồn: tác giả tính bằng sacreBLEU (BLEU cấp kho, tokenizer 13a; chrF++), rouge-score (tách từ Unicode) và BERTScore (bert-base-multilingual-cased, lớp 9)",
         ["Dữ liệu", "Cấu hình", "n", "BLEU", "ROUGE-L", "chrF++", "BERTScore F1"], rows, [1.8, 4.4, 1.0, 1.7, 2.1, 1.9, 2.5])


COMMON_SEO = ["seo_title_len_50_60", "seo_title_tiki_max_225", "seo_keyphrase_in_title", "seo_keyphrase_in_first_paragraph", "seo_desc_words_ge_200"]


def common_rate(dataset, method):
    """Tỷ lệ tuân thủ trên các tiêu chí mà cả nội dung sinh và nội dung người bán đều chấm được (bỏ hai tiêu chí meta)."""
    d = pd.read_csv(RESULTS / f"eval_per_item_{dataset}.csv")
    d = d[d["method"] == method]
    if d.empty:
        return "–"
    cols = [c for c in COMMON_SEO if c in d.columns]
    x = d[cols].replace({"True": 1, "False": 0, True: 1, False: 0}).apply(pd.to_numeric, errors="coerce")
    return vn(100 * x.mean(axis=1, skipna=True).mean(), 1)


def table_checklist():
    """Bảng RQ3: tỷ lệ đạt từng tiêu chí tối ưu hóa tìm kiếm và tiêu đề email (Tiki), kèm dòng tổng Amazon."""
    s = load_summary("tiki")
    a = load_summary("amazon")
    methods = [m for m in ["M0", "M1", "M2", "M3", "M4a", "M4b"] if m in s["metrics"]] + ["reference"]
    spec = [("seo_title_len_50_60", "Tiêu đề 50–60 ký tự (Ahrefs)"), ("seo_title_tiki_max_225", "Tên ≤ 225 ký tự (Tiki)"),
            ("seo_title_tiki_15_50", "Tên 15–50 ký tự (Tiki)*"),
            ("seo_meta_len_120_156", "Meta 120–156 ký tự (Yoast)"), ("seo_keyphrase_in_meta", "Từ khóa trong meta (Yoast)"),
            ("seo_desc_words_ge_200", "Mô tả ≥ 200 từ (Yoast, Tiki)"),
            ("seo_keyphrase_in_title", "Từ khóa trong tiêu đề (Yoast)"), ("seo_keyphrase_in_first_paragraph", "Từ khóa ở đoạn đầu (Yoast)"),
            ("seo_density_proxy_0_5_3", "Mật độ từ khóa 0,5–3% (Yoast, gần đúng)*"), ("seo_compliance_rate", "Tỷ lệ tuân thủ gộp (các tiêu chí áp dụng được)"),
            (None, "Tỷ lệ tuân thủ trên 5 tiêu chí chung"),
            ("email_mailchimp_le9w_le60c", "Email ≤ 9 từ, ≤ 60 ký tự (Mailchimp)"), ("email_omnisend_4_6w_or_21_40c", "Email 4–6 từ hoặc 21–40 ký tự (Omnisend)"),
            ("email_keyword_relevance", "Email chứa từ khóa")]
    rows = []
    for key, label in spec:
        if key is None:
            rows.append([label] + [common_rate("tiki", m) for m in methods])
        else:
            rows.append([label] + [mean_of(s, m, key, 100, 1) for m in methods])
    rows.append(["Tuân thủ gộp, Amazon (n = 50)"] + [mean_of(a, m, "seo_compliance_rate", 100, 1) if not m.startswith("M4") else "–" for m in methods])
    n_m4 = s["metrics"].get("M4a", {}).get("seo_compliance_rate", {}).get("n", "–")
    save("checklist", f"Tỷ lệ đạt (%) từng tiêu chí tối ưu hóa tìm kiếm và tiêu đề email (RQ3), Tiki n = 99 (M4 n = {n_m4})",
         "Nguồn: tác giả tính; ngưỡng và nguồn của từng tiêu chí ở mục 4.5. (*) Báo cáo riêng, không tính vào tỷ lệ gộp. "
         "Tỷ lệ gộp tính trên 7 tiêu chí với M0–M3 và trên 5 tiêu chí với M4 cùng nội dung người bán, vì hai nhóm sau không có meta description; "
         "dòng kế tiếp đưa mọi cấu hình về cùng 5 tiêu chí để so sánh trực tiếp. Với M4, tiêu đề là dòng đầu của mô tả do mô hình sinh. "
         "'–': không áp dụng",
         ["Tiêu chí"] + [m if m != "reference" else "Người bán" for m in methods], rows,
         [4.9] + [1.45] * (len(methods) - 1) + [1.75])


def table_judge():
    """Bảng RQ4 và chất lượng văn bản: điểm giám khảo (1–5), mức bao phủ phản đối và tỷ lệ thuộc tính không có căn cứ (Tiki)."""
    s = load_summary("tiki")
    methods = ["M0", "M1", "M2", "M3"]
    spec = [("judge_mo_ta_tu_nhien", "Mô tả: tự nhiên", 1, 2), ("judge_mo_ta_thuyet_phuc", "Mô tả: thuyết phục", 1, 2),
            ("judge_mo_ta_lien_quan_tu_khoa", "Mô tả: liên quan từ khóa", 1, 2), ("judge_mo_ta_nhat_quan_thong_tin", "Mô tả: nhất quán thông tin", 1, 2),
            ("judge_email_gay_to_mo", "Email: gây tò mò", 1, 2), ("judge_kich_ban_cta_ro_rang", "Kịch bản: lời kêu gọi hành động rõ", 1, 2),
            ("judge_kich_ban_nhu_cau_loi_ich", "Kịch bản: nhu cầu và lợi ích (SPIN)", 1, 2), ("judge_kich_ban_xu_ly_phan_doi", "Kịch bản: xử lý phản đối", 1, 2),
            ("judge_kich_ban_dung_su_that", "Kịch bản: đúng sự thật", 1, 2), ("script_objection_keyword_coverage", "Kịch bản nhắc đến chủ đề phản đối (%)", 100, 1),
            ("hallucination_rate", "Thuộc tính không có căn cứ trong mô tả (%)", 100, 1), ("script_hallucination_rate", "Thuộc tính không có căn cứ trong kịch bản (%)", 100, 1),
            ("latency_s", "Độ trễ sinh (giây/sản phẩm)", 1, 1)]
    rows = [[label] + [mean_of(s, m, key, scale, digits) for m in methods] for key, label, scale, digits in spec]
    save("judge", "Điểm giám khảo Gemini (thang 1–5) và các chỉ số kịch bản, Tiki n = 99 (RQ4)",
         "Nguồn: tác giả tính; giám khảo Gemini 3.8 Flash chấm ẩn danh theo rubric G-Eval. Chủ đề phản đối đối chiếu bằng từ vựng của từng chủ đề, tính trên 97 sản phẩm có phản đối trong bản tóm tắt; tỷ lệ không có căn cứ là trung bình theo sản phẩm, trên các sản phẩm có ít nhất một thuộc tính được nhắc",
         ["Chỉ số", "M0", "M1", "M2", "M3"], rows, [7.6, 2.1, 2.1, 2.1, 2.1])


def table_leakage_and_m4():
    """Bảng vòng lặp prompt (rò rỉ chỉ dẫn) và chẩn đoán mô hình nhỏ M4."""
    leak = json.loads((RESULTS / "prompt_leakage.json").read_text(encoding="utf-8"))["results"]
    biz = json.loads((RESULTS / "business_metrics.json").read_text(encoding="utf-8"))["methods"]
    rows = []
    for dataset, label in (("tiki", "Tiki"), ("amazon", "Amazon")):
        for key, name in (("M1", "M1 Zero-shot"), ("M2 (v1)", "M2 prompt v1"), ("M2 (v2)", "M2 prompt v2"), ("M3 (v1)", "M3 prompt v1"), ("M3 (v2)", "M3 prompt v2")):
            if key in leak.get(dataset, {}):
                r = leak[dataset][key]
                rows.append([label, name, str(r["n"]), vn(100 * r["rate"], 1) + "%"])
    save("leakage", "Tỷ lệ đầu ra bị rò rỉ chỉ dẫn qua hai vòng thiết kế prompt",
         "Nguồn: tác giả đo bằng biểu thức chính quy trên mọi trường đầu ra (src/prompt_leakage.py)",
         ["Dữ liệu", "Cấu hình", "n", "Tỷ lệ rò rỉ"], rows, [2.6, 5.0, 1.6, 3.0])
    rows = []
    for key, name in (("M4a", "M4a Qwen gốc"), ("M4b (LoRA v1)", "M4b LoRA, dữ liệu chưa làm sạch"), ("M4b", "M4b LoRA, dữ liệu đã làm sạch"),
                      ("M2", "M2 Thiết kế câu lệnh (GPT)"), ("reference", "Nội dung người bán")):
        if key not in biz:
            continue
        b = biz[key]
        rows.append([name, str(b["n"]), vn(b.get("pct_copied_tiki_boilerplate"), 1) if "pct_copied_tiki_boilerplate" in b else "–",
                     vn(b.get("pct_repetitive_output"), 1) if "pct_repetitive_output" in b else "–",
                     vn(b["pct_bullets_ge_best_seller_median"], 1),
                     vn(b["pct_title_ge_best_seller_median"], 1)])
    save("m4_business", "Chẩn đoán mô hình nhỏ và mức đạt đặc trưng của nhóm bán chạy (%), Tiki",
         "Nguồn: tác giả tính (src/business_metrics.py); ngưỡng là trung vị của nhóm bán chạy cùng danh mục. Với M4, tiêu đề là dòng đầu của mô tả do mô hình sinh",
         ["Cấu hình", "n", "Chép văn mẫu Tiki", "Sinh lặp", "Gạch đầu dòng ≥ trung vị", "Độ dài tên ≥ trung vị"], rows, [5.4, 1.2, 2.4, 2.0, 2.6, 2.6])


def main():
    """Tạo tất cả bảng hiện có dữ liệu."""
    table_related()
    table_data_description()
    table_descriptive()
    table_predictive()
    table_odds()
    for builder in (table_similarity, table_checklist, table_judge, table_leakage_and_m4):
        try:
            builder()
        except FileNotFoundError as exc:
            print("Bỏ qua", builder.__name__, "vì thiếu", exc.filename)
    print("Đã tạo:", sorted(p.name for p in TABLES.glob("*.json")))


if __name__ == "__main__":
    main()

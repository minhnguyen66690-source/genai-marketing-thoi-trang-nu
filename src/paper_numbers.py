"""Tính các con số mô tả dùng trong bài báo, lưu thành JSON để mọi số trong bài truy được nguồn.

Đầu ra: _workspace/03_results/paper_numbers.json và eda_univariate_rates.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
PROCESSED = ROOT / "data" / "processed"


def pct(series):
    """Tỷ lệ phần trăm (1 chữ số thập phân) của một chuỗi giá trị đúng/sai."""
    return round(100 * float(series.mean()), 1)


def main():
    """Tính thống kê mô tả, tỷ lệ theo nhóm và các số liệu tóm tắt cho bài báo."""
    d = pd.read_csv(PROCESSED / "tiki_features.csv")
    a = pd.read_csv(PROCESSED / "amazon_features.csv")
    w = d["desc_len_words"]
    out = {
        "n_tiki": int(len(d)),
        "n_amazon": int(len(a)),
        "pct_under_200_words": pct(w < 200),
        "pct_no_size": pct(d["has_size_info"] == 0),
        "pct_no_material": pct(d["has_material_info"] == 0),
        "pct_no_care": pct(d["has_care_info"] == 0),
        "pct_cta": pct(d["has_cta"] == 1),
        "median_desc_words": float(w.median()),
        "mean_desc_words": round(float(w.mean()), 1),
        "median_price": float(d["price"].median()),
        "mean_price": round(float(d["price"].mean()), 0),
        "pct_zero_sold": pct(d["quantity_sold"] == 0),
        "median_sold": float(d["quantity_sold"].median()),
        "max_sold": float(d["quantity_sold"].max()),
        "median_title_chars": float(d["title_len_chars"].median()),
        "pct_title_15_50": pct(d["title_len_chars"].between(15, 50)),
        "pct_title_50_60": pct(d["title_len_chars"].between(50, 60)),
        "median_bullets": float(d["n_bullets"].median()),
        "median_images": float(d["n_images"].median()),
        "created_min": str(d["created_date"].min()),
        "created_max": str(d["created_date"].max()),
        "best_seller_rate": pct(d["best_seller"] == 1),
        "n_best_seller": int(d["best_seller"].sum()),
        "median_by_label": d.groupby("best_seller")[["desc_len_words", "n_bullets", "title_len_chars", "n_images", "price"]].median().round(1).to_dict(),
        "category_counts": d["category_l2"].value_counts().to_dict(),
        "amazon_median_desc_words": float(a["desc_len_words"].median()),
        "amazon_pct_under_200": pct(a["desc_len_words"] < 200),
        "amazon_category_counts": a["category_l2"].value_counts().head(8).to_dict(),
        "tiki_boilerplate_note": "Mô tả đã bỏ đoạn thông báo thuế/hóa đơn do Tiki tự gắn (có ở 100% bản ghi gốc).",
    }
    (RESULTS / "paper_numbers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    rates = {}
    quint = pd.qcut(w, 5)
    g = d.groupby(quint, observed=True)["best_seller"].agg(["mean", "count"])
    rates["by_desc_quintile"] = [{"bin": f"{int(np.ceil(i.left))}–{int(i.right)}", "rate": round(100 * r["mean"], 1), "n": int(r["count"])} for i, r in g.iterrows()]
    for col in ["has_size_info", "has_material_info", "has_care_info", "has_cta"]:
        gg = d.groupby(col)["best_seller"].agg(["mean", "count"])
        rates[col] = {int(k): {"rate": round(100 * v["mean"], 1), "n": int(v["count"])} for k, v in gg.iterrows()}
    bullets = pd.cut(d["n_bullets"], [-1, 0, 2, 5, 10_000], labels=["0", "1-2", "3-5", ">5"])
    gg = d.groupby(bullets, observed=True)["best_seller"].agg(["mean", "count"])
    rates["by_bullets"] = {str(k): {"rate": round(100 * v["mean"], 1), "n": int(v["count"])} for k, v in gg.iterrows()}
    (RESULTS / "eda_univariate_rates.json").write_text(json.dumps(rates, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, ensure_ascii=False))
    print(json.dumps(rates, ensure_ascii=False))


if __name__ == "__main__":
    main()

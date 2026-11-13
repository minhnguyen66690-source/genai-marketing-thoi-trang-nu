"""Kiểm định Wilcoxon cho tỷ lệ tuân thủ trên các tiêu chí chung (RQ3).

Tiêu chí chung là các tiêu chí tối ưu hóa tìm kiếm áp dụng được cho cả nội dung sinh lẫn nội dung
người bán, tức không cần mô tả meta. Với dữ liệu Tiki có 5 tiêu chí; với dữ liệu Amazon chỉ 4, vì
ngưỡng tên tối đa 225 ký tự là quy định riêng của Tiki.

Kiểm định chạy trên SỐ TIÊU CHÍ ĐẠT (số nguyên) thay vì tỷ lệ dạng số thực: hai sản phẩm đạt cùng
số tiêu chí phải được coi là bằng nhau, trong khi tỷ lệ số thực lệch nhau ở chữ số thứ 16 làm
Wilcoxon hiểu nhầm là khác nhau và cho giá trị p nhỏ giả tạo.

Đầu ra: _workspace/03_results/seo_common_wilcoxon.json
"""
import json

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from common import RESULTS, save_json

COMMON = ["seo_title_len_50_60", "seo_title_tiki_max_225", "seo_keyphrase_in_title",
          "seo_keyphrase_in_first_paragraph", "seo_desc_words_ge_200"]
PAIRS = [("M2", "reference"), ("M3", "reference"), ("M1", "reference"), ("M0", "reference"), ("M2", "M3")]


def main():
    result = {"note": "Kiểm định Wilcoxon cho mẫu cặp trên số tiêu chí đạt (số nguyên), hiệu chỉnh Holm.", "datasets": {}}
    for dataset in ("tiki", "amazon"):
        path = RESULTS / f"eval_per_item_{dataset}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        x = d[COMMON].replace({"True": 1, "False": 0, True: 1, False: 0}).apply(pd.to_numeric, errors="coerce")
        d["n_pass"] = x.sum(axis=1)
        d["n_applied"] = x.notna().sum(axis=1)
        d["rate"] = x.mean(axis=1, skipna=True)
        counts = d.pivot_table(index="item_id", columns="method", values="n_pass")
        rates = d.pivot_table(index="item_id", columns="method", values="rate")
        rows, pvalues = [], []
        for a, b in PAIRS:
            if a not in counts or b not in counts:
                continue
            pair = counts[[a, b]].dropna()
            test = wilcoxon(pair[a], pair[b], zero_method="wilcox")
            pvalues.append(float(test.pvalue))
            rows.append({"a": a, "b": b, "n": int(len(pair)),
                         "mean_a_pct": round(100 * float(rates[a].mean()), 2),
                         "mean_b_pct": round(100 * float(rates[b].mean()), 2),
                         "n_ties": int((pair[a] == pair[b]).sum())})
        adjusted = multipletests(pvalues, method="holm")[1]
        for row, p, p_holm in zip(rows, pvalues, adjusted):
            row["p"] = p
            row["p_holm"] = float(p_holm)
        result["datasets"][dataset] = {"n_criteria_applied": int(d["n_applied"].max()), "comparisons": rows}
    save_json(RESULTS / "seo_common_wilcoxon.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

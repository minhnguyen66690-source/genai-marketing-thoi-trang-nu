"""Các kiểm tra bổ sung theo góp ý phản biện độc lập (vòng 1).

1. ROUGE-L tách precision/recall: kiểm tra giả thuyết "chrF++ và ROUGE-L trái chiều do độ dài".
2. Loại thuộc tính không có căn cứ trong kịch bản (màu, chất liệu, chính sách...).
3. Tỷ lệ tuân thủ trên 5 tiêu chí chung cho Amazon.
4. Phân bố số từ khóa mỗi sản phẩm.
5. Độ vững khi loại sản phẩm kiểm tra có mô tả trùng nguyên văn với tập huấn luyện.
6. Trung vị hệ số đồng thuận giám khảo theo từng tiêu chí.
Đầu ra: _workspace/03_results/extra_checks.json
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from common import PROCESSED, RESULTS, save_json
from metrics_content import UnicodeTokenizer, factual_support

COMMON = ["seo_title_len_50_60", "seo_title_tiki_max_225", "seo_keyphrase_in_title", "seo_keyphrase_in_first_paragraph", "seo_desc_words_ge_200"]


def load(path):
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rouge_pr(dataset, methods):
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=UnicodeTokenizer())
    out = {}
    for m in methods:
        path = RESULTS / f"generations_{m}_{dataset}.jsonl"
        if not path.exists():
            continue
        p, r, ratio = [], [], []
        for rec in load(path):
            ref = rec.get("reference_description") or ""
            cand = (rec.get("output") or {}).get("description") or ""
            if not ref or not cand:
                continue
            s = scorer.score(ref, cand)["rougeL"]
            p.append(s.precision)
            r.append(s.recall)
            ratio.append(len(cand.split()) / max(len(ref.split()), 1))
        out[m] = {"n": len(p), "precision": round(float(np.mean(p)), 3), "recall": round(float(np.mean(r)), 3),
                  "median_length_ratio_gen_over_ref": round(float(np.median(ratio)), 2)}
    return out


def script_claims(dataset, methods):
    out = {}
    for m in methods:
        path = RESULTS / f"generations_{m}_{dataset}.jsonl"
        if not path.exists():
            continue
        c, total, uns, items = Counter(), 0, 0, 0
        for rec in load(path):
            fs = factual_support((rec.get("output") or {}).get("sales_script") or "", rec.get("brief") or {})
            total += fs["total"]
            uns += fs["unsupported"]
            items += fs["total"] > 0
            c.update(fs["unsupported_list"])
        out[m] = {"items_with_claims": items, "unsupported": uns, "claims": total, "top_unsupported": c.most_common(8)}
    return out


def common_rate(dataset):
    d = pd.read_csv(RESULTS / f"eval_per_item_{dataset}.csv")
    res = {}
    for m, g in d.groupby("method"):
        x = g[COMMON].replace({"True": 1, "False": 0, True: 1, False: 0}).apply(pd.to_numeric, errors="coerce")
        res[m] = round(100 * float(x.mean(axis=1, skipna=True).mean()), 1)
    return res


def duplicate_robustness():
    g = pd.read_csv(PROCESSED / "gen_inputs_tiki.csv")
    g["d"] = g.reference_description.fillna("").str.strip().str.lower().str[:400]
    pi = pd.read_csv(RESULTS / "eval_per_item_tiki.csv")
    pi["product_id"] = pi.product_id.astype(str)
    test = g[g.product_id.astype(str).isin(set(pi.product_id))]
    train = set(g[g.split == "train"].d)
    dups = set(test[test.d.isin(train) & (test.d.str.len() > 30)].product_id.astype(str))
    res = {"n_duplicates": len(dups), "all": {}, "without_duplicates": {}}
    for label, frame in (("all", pi), ("without_duplicates", pi[~pi.product_id.isin(dups)])):
        for m, gm in frame.groupby("method"):
            x = gm[COMMON].replace({"True": 1, "False": 0, True: 1, False: 0}).apply(pd.to_numeric, errors="coerce")
            res[label][m] = {"n": int(len(gm)), "chrf": round(float(gm.chrf_plus_plus.mean()), 1) if gm.chrf_plus_plus.notna().any() else None,
                             "rougeL": round(float(gm.rougeL.mean()), 3) if gm.rougeL.notna().any() else None,
                             "seo_common": round(100 * float(x.mean(axis=1, skipna=True).mean()), 1)}
    return res


def keyword_counts():
    g = pd.read_csv(PROCESSED / "gen_inputs_tiki.csv")
    n = g.keywords.apply(lambda s: len(json.loads(s)) if isinstance(s, str) and s.startswith("[") else 0)
    return {"mean": round(float(n.mean()), 2), "min": int(n.min()), "max": int(n.max()),
            "share_lt3": round(100 * float((n < 3).mean()), 1), "distribution": {int(k): int(v) for k, v in n.value_counts().sort_index().items()}}


def judge_medians():
    res = {}
    for ds in ("tiki", "amazon"):
        path = RESULTS / f"judge_agreement_{ds}.json"
        if not path.exists():
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        sp = [v["spearman"] for v in j["criteria"].values() if v.get("spearman") is not None]
        kw = [v["quadratic_kappa"] for v in j["criteria"].values() if v.get("quadratic_kappa") is not None]
        res[ds] = {"n_criteria": len(j["criteria"]), "n_per_criterion": min(v["n"] for v in j["criteria"].values()),
                   "median_spearman": round(float(np.median(sp)), 2), "median_kappa": round(float(np.median(kw)), 2),
                   "min_spearman": round(float(min(sp)), 2), "max_spearman": round(float(max(sp)), 2)}
    return res


def main():
    methods = ["M0", "M1", "M2", "M3", "M4a", "M4b"]
    result = {
        "rougeL_precision_recall": {"tiki": rouge_pr("tiki", methods), "amazon": rouge_pr("amazon", methods[:4])},
        "script_unsupported_claims": {"tiki": script_claims("tiki", ["M1", "M2", "M3"]), "amazon": script_claims("amazon", ["M1", "M2", "M3"])},
        "seo_common_5": {"tiki": common_rate("tiki"), "amazon": common_rate("amazon")},
        "duplicate_robustness_tiki": duplicate_robustness(),
        "keywords_per_product_tiki": keyword_counts(),
        "judge_agreement_medians": judge_medians(),
    }
    save_json(RESULTS / "extra_checks.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

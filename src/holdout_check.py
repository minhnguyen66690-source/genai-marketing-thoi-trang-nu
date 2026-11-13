"""Kiểm tra prompt v2 trên tập kiểm định độc lập (chưa từng xem khi sửa prompt).

Prompt v2 được sửa sau khi quan sát lỗi rò rỉ chỉ dẫn trên tập kiểm tra, nên kết quả trên tập kiểm tra
có thể lạc quan. Script này tính cùng một bộ chỉ số tự động cho 40 sản phẩm lấy từ tập kiểm định
(generate.py --split val) và cho tập kiểm tra, để so sánh hai bên theo cùng một cách tính.
Đầu ra: _workspace/03_results/holdout_check.json
"""
import json

import numpy as np

from common import RESULTS, save_json
from metrics_content import email_checks, factual_support, script_rule_checks, seo_checks
from prompt_leakage import leaked

COMMON = ["title_len_50_60", "title_tiki_max_225", "keyphrase_in_title", "keyphrase_in_first_paragraph", "desc_words_ge_200"]


def load(path):
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records):
    """Tính các chỉ số tự động trung bình trên một tập bản ghi sinh."""
    seo, common, mail, objection, leak = [], [], [], [], []
    claims = unsupported = 0
    for r in records:
        brief, out = r.get("brief") or {}, r.get("output") or {}
        keywords = brief.get("keywords") or []
        checks = seo_checks(out.get("title", ""), out.get("meta_description"), out.get("description", ""),
                            keywords[0] if keywords else "", keywords, r.get("lang", "vi"))
        seo.append(checks["compliance_rate"])
        values = [checks["checks"][k] for k in COMMON if checks["checks"].get(k) is not None]
        common.append(float(np.mean(values)))
        subjects = out.get("email_subjects") or []
        if subjects:
            mail.append(float(np.mean([email_checks(s, keywords, brief)["checks"]["mailchimp_le9w_le60c"] for s in subjects])))
        cov = script_rule_checks(out.get("sales_script", ""), brief, brief.get("objections"))["objection_keyword_coverage"]
        if cov is not None:
            objection.append(cov)
        fact = factual_support(out.get("title", "") + " " + out.get("description", ""), brief)
        claims += fact["total"]
        unsupported += fact["unsupported"]
        leak.append(leaked(r))
    pct = lambda xs: round(100 * float(np.mean(xs)), 1) if xs else None
    return {"n": len(records), "seo_compliance_7": pct(seo), "seo_common_5": pct(common), "email_mailchimp": pct(mail),
            "objection_theme_coverage": pct(objection), "unsupported_claims": f"{unsupported}/{claims}",
            "unsupported_rate": round(100 * unsupported / claims, 1) if claims else None, "prompt_leakage": pct(leak)}


def main():
    result = {"note": "Cùng cách tính cho hai tập; tập kiểm định gồm 40 sản phẩm không dùng khi sửa prompt.", "val": {}, "test": {}}
    for method in ("M0", "M2", "M3"):
        val_path = RESULTS / "holdout" / f"generations_{method}_tiki_val.jsonl"
        test_path = RESULTS / f"generations_{method}_tiki.jsonl"
        if val_path.exists():
            result["val"][method] = summarize(load(val_path))
        if test_path.exists():
            result["test"][method] = summarize(load(test_path))
    save_json(RESULTS / "holdout_check.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

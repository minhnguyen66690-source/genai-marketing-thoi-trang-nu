"""Đo độ đồng thuận giữa hai giám khảo khác hãng (Gemini và GPT) trên một mẫu con.

Mục đích: kiểm tra độ tin cậy của điểm giám khảo mô hình ngôn ngữ. Nếu hai giám khảo khác hãng chấm
gần nhau thì điểm ít phụ thuộc vào một mô hình cụ thể. Kết quả chấm của Gemini lấy từ bộ nhớ đệm
(đã chấm trong evaluate.py); GPT (gpt-5.6-luna qua Codex CLI) chấm lại đúng các mẫu đó với cùng rubric.
Lưu ý: GPT cũng là họ mô hình sinh M1–M3 nên có thể ưu ái nội dung của chính nó (Zheng và cộng sự, 2023);
vì vậy GPT chỉ dùng để đo đồng thuận, không thay giám khảo chính.

Đầu ra: _workspace/03_results/judge_agreement_<dataset>.json
"""
import argparse
import json
import random
from pathlib import Path

import judge

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
CRITERIA = {
    "mo_ta": ["tu_nhien", "lien_quan_tu_khoa", "thuyet_phuc", "nhat_quan_thong_tin"],
    "email": ["ro_rang", "gay_to_mo", "lien_quan"],
    "kich_ban": ["cta_ro_rang", "nhu_cau_loi_ich", "xu_ly_phan_doi", "dung_su_that"],
}


def load_items(dataset):
    """Đọc các mẫu sinh M0–M3 theo đúng thứ tự evaluate.py đọc."""
    items = []
    paths = sorted(RESULTS.glob(f"generations_M[0-3]_{dataset}.jsonl")) + sorted((RESULTS / "prompt_v1").glob(f"generations_M[23]v1_{dataset}.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    """Chọn mẫu con cân bằng theo phương pháp, chấm bằng GPT, so với Gemini."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tiki", "amazon"])
    parser.add_argument("--per-method", type=int, default=10)
    args = parser.parse_args()
    items = load_items(args.dataset)
    rng = random.Random(2026)
    sample = []
    gemini_cache = judge.load_cache("gemini")

    def has_gemini(x):
        """Chỉ lấy mẫu đã có điểm Gemini để so sánh được."""
        return judge.content_key({"brief": x.get("brief", {}), "output": x.get("output", {})}) in gemini_cache

    for method in ["M0", "M1", "M2", "M3"]:
        pool = [x for x in items if x["method"] == method and has_gemini(x)]
        if not pool:
            continue
        sample.extend(rng.sample(pool, min(args.per_method, len(pool))))
    gemini_cache = judge.load_cache("gemini")
    gemini = [gemini_cache.get(judge.content_key({"brief": x.get("brief", {}), "output": x.get("output", {})})) for x in sample]
    gpt = judge.judge_items(sample, "gpt", seed=7)
    report = {"dataset": args.dataset, "n_items": len(sample), "per_method": args.per_method, "criteria": {}}
    all_a, all_b = [], []
    for group, names in CRITERIA.items():
        for name in names:
            a = [g[group][name]["score"] if g else None for g in gemini]
            b = [h[group][name]["score"] if h else None for h in gpt]
            report["criteria"][f"{group}.{name}"] = judge.agreement(a, b)
            all_a.extend(a)
            all_b.extend(b)
    report["pooled"] = judge.agreement(all_a, all_b)
    out = RESULTS / f"judge_agreement_{args.dataset}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["pooled"], ensure_ascii=False))
    for key, value in report["criteria"].items():
        print(key, value)


if __name__ == "__main__":
    main()

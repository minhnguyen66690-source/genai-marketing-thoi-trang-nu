"""Chỉ số bổ sung: quy đổi sang góc nhìn kinh doanh và chẩn đoán lỗi của mô hình nhỏ.

1. Đặc trưng nhóm bán chạy (từ RQ1): tỷ lệ nội dung sinh có số gạch đầu dòng và độ dài tên đạt trung vị
   của nhóm bán chạy cùng danh mục (so với nội dung gốc của người bán).
2. Độ trễ: giây/sản phẩm và số sản phẩm/giờ khi chạy tuần tự (từ latency_s đo trong lúc sinh).
3. Chẩn đoán M4: tỷ lệ đầu ra chép đoạn văn mẫu thuế của Tiki và tỷ lệ đầu ra lặp từ (một cụm 3 từ lặp ≥ 4 lần).

Đầu ra: _workspace/03_results/business_metrics.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
PROCESSED = ROOT / "data" / "processed"
BOILERPLATE = "Giá sản phẩm trên Tiki đã bao gồm thuế"


def n_bullets(text):
    """Đếm dòng gạch đầu dòng (cùng quy tắc với preprocess: -, *, •, ✔, ✅, số thứ tự)."""
    return sum(bool(re.match(r"^\s*(?:[-*•✔✅]|\d+[.)])", line)) for line in str(text or "").splitlines())


def repeated(text):
    """Dấu hiệu sinh lặp (degeneration): có cụm 5 từ nào lặp từ 3 lần trở lên.

    Dùng cụm 5 từ thay vì 3 từ để không tính nhầm việc lặp lại từ khóa chính theo yêu cầu mật độ từ khóa.
    """
    words = str(text or "").lower().split()
    grams = Counter(tuple(words[i:i + 5]) for i in range(len(words) - 4))
    return bool(grams) and max(grams.values()) >= 3


def load(path):
    """Đọc file JSONL thành danh sách bản ghi."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    """Tính các chỉ số và lưu JSON."""
    features = pd.read_csv(PROCESSED / "tiki_features.csv")
    best = features[features["best_seller"] == 1].groupby("category_l2")[["n_bullets", "title_len_chars"]].median()
    overall = features[features["best_seller"] == 1][["n_bullets", "title_len_chars"]].median()
    report = {"best_seller_medians_overall": overall.to_dict(), "methods": {}}
    files = {m: RESULTS / f"generations_{m}_tiki.jsonl" for m in ["M0", "M1", "M2", "M3", "M4a", "M4b"]}
    files["M4b (LoRA v1)"] = RESULTS / "lora_v1" / "generations_M4b_lorav1_tiki.jsonl"
    for method, path in files.items():
        if not path.exists():
            continue
        rows = load(path)
        hit_bullets, hit_title, latencies, boiler, repeat = [], [], [], [], []
        for r in rows:
            cat = r["brief"].get("category")
            target = best.loc[cat] if cat in best.index else overall
            out = r["output"]
            hit_bullets.append(n_bullets(out.get("description")) >= target["n_bullets"])
            hit_title.append(len(str(out.get("title") or "")) >= target["title_len_chars"])
            latencies.append(float(r.get("latency_s") or 0))
            boiler.append(BOILERPLATE in str(out.get("description") or ""))
            repeat.append(repeated(out.get("description")))
        mean_latency = sum(latencies) / len(latencies)
        report["methods"][method] = {
            "n": len(rows),
            "pct_bullets_ge_best_seller_median": round(100 * sum(hit_bullets) / len(rows), 1),
            "pct_title_ge_best_seller_median": round(100 * sum(hit_title) / len(rows), 1),
            "latency_s_per_item": round(mean_latency, 2),
            "items_per_hour_sequential": round(3600 / mean_latency, 0) if mean_latency > 0.01 else None,
            "pct_copied_tiki_boilerplate": round(100 * sum(boiler) / len(rows), 1),
            "pct_repetitive_output": round(100 * sum(repeat) / len(rows), 1),
        }
    # Nội dung gốc của người bán trên cùng tập kiểm tra (đối chứng).
    ref_rows = load(files["M0"])
    hb = ht = 0
    for r in ref_rows:
        cat = r["brief"].get("category")
        target = best.loc[cat] if cat in best.index else overall
        hb += n_bullets(r.get("reference_description")) >= target["n_bullets"]
        ht += len(str(r.get("reference_title") or "")) >= target["title_len_chars"]
    report["methods"]["reference"] = {"n": len(ref_rows), "pct_bullets_ge_best_seller_median": round(100 * hb / len(ref_rows), 1),
                                      "pct_title_ge_best_seller_median": round(100 * ht / len(ref_rows), 1)}
    (RESULTS / "business_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in report["methods"].items():
        print(k, v)


if __name__ == "__main__":
    main()

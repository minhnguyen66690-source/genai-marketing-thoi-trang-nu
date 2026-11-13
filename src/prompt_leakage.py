"""Đo tỷ lệ rò rỉ chỉ dẫn (instruction leakage) trong nội dung sinh: bản prompt v1 so với v2.

Rò rỉ chỉ dẫn là khi văn bản gửi khách chứa dấu vết của prompt, ví dụ "brief chưa cung cấp chất liệu",
"thông tin không được cung cấp", "not specified". Bản v1 của M2/M3 gặp lỗi này; bản v2 thêm quy tắc giọng văn.
Một mẫu bị tính là rò rỉ nếu BẤT KỲ trường nào (tiêu đề, meta, mô tả, kịch bản, tiêu đề email) khớp mẫu.

Đầu ra: _workspace/03_results/prompt_leakage.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "_workspace" / "03_results"
PATTERN = re.compile(
    r"\bbrief\b|chưa cung cấp|không được cung cấp|chưa nêu|chưa có thông tin|không có thông tin|"
    r"not provided|isn.t specified|not specified|no information|not mentioned", re.IGNORECASE)
FIELDS = ["title", "meta_description", "description", "sales_script"]


def leaked(record):
    """Kiểm tra một bản ghi sinh có chứa dấu vết của prompt hay không."""
    output = record.get("output", {})
    texts = [str(output.get(k) or "") for k in FIELDS] + [str(s) for s in output.get("email_subjects") or []]
    return any(PATTERN.search(t) for t in texts)


def rate(path):
    """Tỷ lệ mẫu bị rò rỉ trong một file sinh."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_leak = sum(1 for r in rows if leaked(r))
    return {"n": len(rows), "leaked": n_leak, "rate": n_leak / len(rows) if rows else None}


def main():
    """Tính tỷ lệ rò rỉ cho M1, M2, M3 (v2) và M2, M3 (v1) trên hai bộ dữ liệu."""
    report = {"pattern": PATTERN.pattern, "results": {}}
    for dataset in ["tiki", "amazon"]:
        block = {}
        for method in ["M0", "M1", "M2", "M3"]:
            path = RESULTS / f"generations_{method}_{dataset}.jsonl"
            if path.exists():
                block[method + (" (v2)" if method in ("M2", "M3") else "")] = rate(path)
        for method in ["M2v1", "M3v1"]:
            path = RESULTS / "prompt_v1" / f"generations_{method}_{dataset}.jsonl"
            if path.exists():
                block[method.replace("v1", " (v1)")] = rate(path)
        report["results"][dataset] = block
    (RESULTS / "prompt_leakage.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

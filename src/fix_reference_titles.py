"""Bổ sung reference_title (tên thật của sản phẩm) cho các file generations_*_<dataset>.jsonl.

Lý do: phiên bản đầu của generate.py ghép bảng tạo cột trùng tên (name_x, name_y) nên reference_title bị rỗng.
Script này chỉ điền lại tên từ gen_inputs_<dataset>.csv theo product_id, không thay đổi nội dung sinh.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Điền reference_title cho mọi file sinh của một bộ dữ liệu."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tiki", "amazon"])
    args = parser.parse_args()
    inputs = pd.read_csv(ROOT / "data" / "processed" / f"gen_inputs_{args.dataset}.csv", dtype={"product_id": str})
    names = dict(zip(inputs["product_id"], inputs["name"].fillna("")))
    # Mô tả tham chiếu lấy lại từ gen_inputs (đã bỏ đoạn văn mẫu thuế/hóa đơn của Tiki).
    references = dict(zip(inputs["product_id"], inputs["reference_description"].fillna("")))
    results = ROOT / "_workspace" / "03_results"
    excluded_dir = results / "excluded_items"
    for path in sorted(results.glob(f"generations_*_{args.dataset}.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept, excluded = [], []
        fixed = 0
        for record in records:
            pid = str(record["product_id"])
            if pid not in references:
                # Sản phẩm không còn trong dữ liệu sau khi làm sạch (mô tả người bán viết < 50 ký tự): tách riêng, không xóa.
                excluded.append(record)
                continue
            if record.get("reference_title") != names[pid] or record.get("reference_description") != references[pid]:
                record["reference_title"] = names[pid]
                record["reference_description"] = references[pid]
                fixed += 1
            kept.append(record)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        if excluded:
            excluded_dir.mkdir(exist_ok=True)
            (excluded_dir / path.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in excluded), encoding="utf-8")
        print(path.name, "cập nhật", fixed, "| giữ", len(kept), "| tách riêng", len(excluded))


if __name__ == "__main__":
    main()

"""Sinh M3.1 trên tập phát triển bằng dữ kiện chi tiết làm mẫu cho biểu mẫu trợ lý.

Tập phát triển chỉ dùng để chọn câu lệnh. Mỗi ``confirmed_facts`` bên dưới được lấy từ
trang người bán hiện có để mô phỏng trường hợp người bán đã điền đủ biểu mẫu. Không dùng
tệp này để công bố kết quả kiểm tra cuối cùng.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import prompts_m3
import prompts_m31
from generate_m3 import call, hits_of
from generate_v2 import ROOT, append, model_brief, read_jsonl
from holdout_paths import data_dir, results_dir


def seller_proxy(product):
    """Dữ kiện phát triển giàu thông tin, tương ứng người bán đã xác nhận các trường biểu mẫu."""
    return prompts_m3.seller_details(product)


def count_tokens(text):
    """Giữ phép thử chạy được khi môi trường thiếu tokenizer dùng cho thống kê phụ."""
    try:
        from generate_v2 import token_count
        return token_count(text)
    except ModuleNotFoundError:
        return len(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    data = data_dir(ROOT, "dev")
    briefs = [row for row in read_jsonl(data / "briefs.jsonl") if row.get("split") == "dev"]
    if args.limit is not None:
        briefs = briefs[:args.limit]
    with (data / "products_clean.csv").open(encoding="utf-8-sig", newline="") as handle:
        products = {str(row["product_id"]): row for row in csv.DictReader(handle) if row.get("split") == "dev"}
    guide = prompts_m3.load_guide()
    output_path = results_dir(ROOT, "dev") / "gen" / "dev" / "M31.jsonl"
    existing = {str(row["item_id"]): row for row in read_jsonl(output_path) if row.get("output") and not row.get("error")}
    pending = [brief for brief in briefs if str(brief["item_id"]) not in existing]

    for start in range(0, len(pending), 5):
        brief_batch = pending[start:start + 5]
        rows = [model_brief(brief) for brief in brief_batch]
        confirmed = {str(row["item_id"]): seller_proxy(products[str(row["item_id"])]) for row in rows}
        prompt = prompts_m31.build(rows, confirmed, guide)
        began = time.perf_counter()
        raw, found, error = call(prompt, rows, "v2:dev:M31")
        first_hits = {item_id: hits_of(content) for item_id, content in found.items()}
        redo = [row for row in rows if first_hits.get(str(row["item_id"]))]
        raw_rewrite = ""
        if redo:
            note = ("\nSỬA LẠI: bản trước còn câu chung chung. Viết lại toàn bộ, bỏ các cụm sau và chỉ giữ "
                    "dữ kiện cụ thể về sản phẩm: "
                    + "; ".join(sorted({hit for row in redo for hit in first_hits[str(row["item_id"])]})))
            rewrite_prompt = prompts_m31.build(redo, {str(row["item_id"]): confirmed[str(row["item_id"])] for row in redo}, guide) + note
            raw_rewrite, revised, rewrite_error = call(rewrite_prompt, redo, "v2:dev:M31:rewrite")
            if not rewrite_error:
                found.update(revised)
        elapsed = (time.perf_counter() - began) / len(rows)
        for row in rows:
            item_id = str(row["item_id"])
            content = found.get(item_id)
            append(output_path, {
                "item_id": item_id,
                "method": "M31",
                "input_source": "seller_page_as_confirmed_proxy_dev_only",
                "template_hash": prompts_m31.template_hash(),
                "guide_hash": hashlib.sha256(prompts_m3.GUIDE_PATH.read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "model": "gpt-6.1-terra",
                "latency_s": elapsed,
                "input_tokens": count_tokens(prompt) / len(rows),
                "output_tokens": (count_tokens(raw) + count_tokens(raw_rewrite)) / len(rows),
                "cliches_first": first_hits.get(item_id, []),
                "cliches_final": hits_of(content) if content else None,
                "output": content,
                "error": error,
            })
        print(f"dev M31: {min(start + len(brief_batch), len(pending))}/{len(pending)}", flush=True)

    completed = {str(row["item_id"]) for row in read_jsonl(output_path) if row.get("output") and not row.get("error")}
    if {str(row["item_id"]) for row in briefs} != completed:
        raise RuntimeError(f"Thiếu đầu ra M31 dev: {len(completed)}/{len(briefs)}")


if __name__ == "__main__":
    main()

"""Đếm số token mỗi sản phẩm của cấu hình M2 để quy ra chi phí API.

Cách đếm: dựng lại đúng câu lệnh M2 cho từng sản phẩm (như website demo gửi, 1 sản phẩm/lần)
và đếm token đầu ra thật đã lưu trong generations_M2_tiki.jsonl, bằng bộ tách token o200k_base.
o200k_base là bộ tách token công bố của OpenAI cho dòng GPT-4o trở đi; tokenizer của
gpt-6.1-terra và Gemini không công bố nên con số là ước lượng, ghi rõ trong bài.
Kết quả: _workspace/03_results/token_usage.json
"""
import json
import statistics

import tiktoken

import common
import prompts

ENC = tiktoken.get_encoding("o200k_base")


def count(text):
    return len(ENC.encode(text))


def main():
    path = common.RESULTS / "generations_M2_tiki.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    inputs, outputs = [], []
    for row in rows:
        inputs.append(count(prompts.engineered([row["brief"]], "vi")))
        outputs.append(count(json.dumps(row["output"], ensure_ascii=False)))
    result = {
        "method": "M2",
        "n_products": len(rows),
        "tokenizer": "o200k_base (tiktoken 0.14.0)",
        "input_tokens_mean": round(statistics.mean(inputs)),
        "input_tokens_max": max(inputs),
        "output_tokens_mean": round(statistics.mean(outputs)),
        "output_tokens_max": max(outputs),
        "latency_s_mean": round(statistics.mean(r["latency_s"] for r in rows), 2),
        "note": "1 sản phẩm mỗi lần gọi, đúng như website demo",
    }
    (common.RESULTS / "token_usage.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

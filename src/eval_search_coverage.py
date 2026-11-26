#!/usr/bin/env python3
"""Measure category-search-query coverage for seller, M2 and M3 outputs.

Run from the repository root, for example:
  python src/eval_search_coverage.py \
      --queries path/to/google_trends.json --out path/to/search_coverage.json

The matcher is deliberately token based: Unicode NFC, casefold, preserved
diacritics, and punctuation as token separators.  It requires a contiguous sequence of complete words.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


FIELDS = ("title", "meta_description", "first_sentence")
METHODS = ("SELLER", "M2", "M3")
REQUIRED_SOURCE = ("source", "accessed_date", "geo", "period", "search_type")
TOKEN_RE = re.compile(r"[^\W_]+")
SENTENCE_RE = re.compile(r"[.!?…。！？]+(?=\s|$)|\r?\n+")


def text(value) -> str:
    return "" if value is None else str(value)


def tokens(value) -> tuple[str, ...]:
    value = unicodedata.normalize("NFC", text(value)).casefold()
    return tuple(TOKEN_RE.findall(value))


def contains_phrase(value, phrase) -> bool:
    haystack, needle = tokens(value), tokens(phrase)
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[i:i + width] == needle for i in range(len(haystack) - width + 1))


def first_sentence(value) -> str:
    value = text(value).strip()
    return SENTENCE_RE.split(value, maxsplit=1)[0].strip() if value else ""


def output_is_valid(row: dict) -> bool:
    output = row.get("output")
    return isinstance(output, dict) and all(isinstance(output.get(field), str) and output[field].strip()
                                            for field in ("title", "meta_description", "description"))


def read_jsonl(path: Path, prefer_valid_output: bool = False) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                item_id = str(row.get("item_id"))
                if item_id not in result or (prefer_valid_output and output_is_valid(row)):
                    result[item_id] = row
    return result


def normalise_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def validate_queries(document: dict, categories: list[str]) -> dict[str, list[dict]]:
    missing_source = [key for key in REQUIRED_SOURCE if not document.get(key)]
    if missing_source:
        raise ValueError("Thiếu metadata nguồn: " + ", ".join(missing_source))
    supplied = document.get("categories")
    if not isinstance(supplied, dict):
        raise ValueError("queries.categories phải là object")
    missing_categories = [category for category in categories if category not in supplied]
    if missing_categories:
        raise ValueError("Thiếu category query: " + ", ".join(missing_categories))
    validated = {}
    for category in categories:
        entry = supplied[category]
        if not isinstance(entry, dict):
            raise ValueError(f"category {category!r} không phải object")
        seeds = normalise_list(entry.get("seed"))
        urls = normalise_list(entry.get("url"))
        if not seeds or not urls:
            raise ValueError(f"category {category!r} thiếu seed/url metadata")
        queries = entry.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"category {category!r} thiếu queries")
        good = []
        for query in queries:
            if not isinstance(query, dict) or not text(query.get("query")).strip():
                raise ValueError(f"category {category!r} có query không hợp lệ")
            if not isinstance(query.get("rank"), (int, float)) or isinstance(query.get("rank"), bool):
                raise ValueError(f"category {category!r} có rank không hợp lệ")
            if not isinstance(query.get("score"), (int, float)) or isinstance(query.get("score"), bool):
                raise ValueError(f"category {category!r} có score không hợp lệ")
            good.append(query)
        validated[category] = good
    return validated


def load_raw_meta(path: Path, wanted: set[str]) -> dict[str, str | None]:
    result = {}
    if not path.exists():
        raise FileNotFoundError(f"Không thấy raw product JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("id"))
            if item_id in wanted:
                value = row.get("meta_description")
                result[item_id] = value if isinstance(value, str) and value.strip() else None
    return result


def rate(matches: int, eligible: int, missing: int) -> dict:
    return {"n": eligible, "n_eligible": eligible, "successes": matches,
            "rate": (matches / eligible if eligible else None), "missing": missing}


def measure(value: str | None, queries: list[dict]) -> tuple[bool, bool]:
    if value is None or not text(value).strip():
        return False, True
    return any(contains_phrase(value, q["query"]) for q in queries), False


def run(root: Path, query_path: Path, out_path: Path, split: str = "test") -> dict:
    if split == "test2":
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from holdout_paths import data_dir, results_dir, verify_test2
        locked_ids = verify_test2(root)
        processed = data_dir(root, split)
        results = results_dir(root, split) / "gen" / split
    else:
        processed = root / "data_v2" / "processed"
        results = root / "results_v2" / "gen" / split
    products = []
    with (processed / "products_clean.csv").open(encoding="utf-8-sig", newline="") as handle:
        products = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if len(products) != 120:
        raise ValueError(f"{split} phải có đúng 120 dòng, nhận {len(products)}")
    if split == "test2" and {str(row["product_id"]) for row in products} != locked_ids:
        raise ValueError("Danh sách sản phẩm test2 không khớp với sample lock")
    categories = sorted({row["category_l2"] for row in products})
    query_document = json.loads(query_path.read_text(encoding="utf-8"))
    queries = validate_queries(query_document, categories)
    ids = {str(row["product_id"]) for row in products}
    raw_meta = (load_raw_meta(root / "data_v2" / "raw" / "tiki_products_raw.jsonl", ids)
                if split == "test" else {})

    generated = {method: read_jsonl(results / f"{method}.jsonl", prefer_valid_output=True)
                 for method in ("M2", "M3")}
    missing_generated = {method: sorted(ids - set(rows)) for method, rows in generated.items()}
    if any(missing_generated.values()):
        raise ValueError("Thiếu output test: " + repr(missing_generated))

    records = {method: {} for method in METHODS}
    for row in products:
        item_id, category = str(row["product_id"]), row["category_l2"]
        seller_meta = raw_meta.get(item_id) if split == "test" else (row.get("meta_description") or None)
        records["SELLER"][item_id] = {
            "title": row.get("name", ""), "meta_description": seller_meta,
            "first_sentence": first_sentence(row.get("description_text", "")),
            "category": category,
        }
        for method in ("M2", "M3"):
            output = generated[method][item_id].get("output") or {}
            records[method][item_id] = {
                "title": output.get("title"), "meta_description": output.get("meta_description"),
                "first_sentence": first_sentence(output.get("description")), "category": category,
            }

    category_out = {}
    user_methods = {}
    for category in categories:
        category_ids = [str(row["product_id"]) for row in products if row["category_l2"] == category]
        method_out = {}
        for method in METHODS:
            field_out = {}
            for field in FIELDS:
                yes = eligible = missing = 0
                for item_id in category_ids:
                    matched, is_missing = measure(records[method][item_id][field], queries[category])
                    missing += is_missing
                    if not is_missing:
                        eligible += 1
                        yes += matched
                field_out[field] = rate(yes, eligible, missing)
            method_out[method] = field_out
        entry = query_document["categories"][category]
        category_out[category] = {
            "n_products": len(category_ids),
            "query_metadata": {"seed": entry.get("seed"), "url": entry.get("url"),
                                "n_queries": len(queries[category]),
                                "query_seeds": sorted({q.get("seed") for q in queries[category] if q.get("seed") is not None})},
            "methods": method_out,
        }

    for method in METHODS:
        field_out = {}
        for field in FIELDS:
            yes = eligible = missing = 0
            for item_id in ids:
                matched, is_missing = measure(records[method][item_id][field], queries[records[method][item_id]["category"]])
                missing += is_missing
                if not is_missing:
                    eligible += 1
                    yes += matched
            field_out[field] = rate(yes, eligible, missing)
        user_methods[method] = field_out
    output = {
        "schema_version": "seo_search_coverage.v1", "split": split, "sample_size": len(products),
        "source_metadata": {key: query_document[key] for key in REQUIRED_SOURCE},
        "n_categories": len(categories), "categories": category_out,
        "user_level": {"n_sample": len(products), "sample_size": len(products), "methods": user_methods},
        "seller_meta_note": "SELLER meta_description lấy đúng raw JSON; thiếu là missing, không phải không khớp.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def self_test() -> None:
    assert contains_phrase("ÁO DÀI, nữ", "áo dài")
    assert contains_phrase("áo dài", "áo dài")
    assert contains_phrase("áo dài", "áo")
    assert not contains_phrase("áo dài", "dài nữ")
    assert first_sentence("Áo dài. Mô tả tiếp") == "Áo dài"
    assert first_sentence("Giá 1.250 đồng cho áo dài. Mô tả tiếp") == "Giá 1.250 đồng cho áo dài"
    assert measure(None, [{"query": "áo", "rank": 1, "score": 1}]) == (False, True)
    assert output_is_valid({"output": {"title": "t", "meta_description": "m", "description": "d"}})
    assert not output_is_valid({"output": None, "error": "retry"})
    doc = {"source": "test", "accessed_date": "2026-11-24", "geo": "VN", "period": "x", "search_type": "web", "categories": {"A": {"seed": ["a", "b"], "url": ["u1", "u2"], "queries": [{"rank": 1, "query": "áo", "score": 10, "seed": "a"}, {"rank": 1, "query": "váy", "score": 5, "seed": "b"}]}}}
    assert len(validate_queries(doc, ["A"])["A"]) == 2
    print("self-test: PASS (unicode, word-boundary, missing, two-seed)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("test", "test2"), default="test")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.queries or args.out:
        if not args.queries or not args.out:
            parser.error("--queries và --out phải đi cùng nhau")
        result = run(args.root.resolve(), args.queries.resolve(), args.out.resolve(), args.split)
        print(json.dumps({"out": str(args.out.resolve()), "sample_size": result["sample_size"], "n_categories": result["n_categories"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

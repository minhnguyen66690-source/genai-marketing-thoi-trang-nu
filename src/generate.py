"""Sinh nội dung theo các phương pháp M0–M3."""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import common
import llm_backends
import prompts

FIELDS = ("title", "meta_description", "description", "sales_script", "email_subjects")


def read_data(dataset):
    """Đọc dữ liệu đầu vào và tên sản phẩm nếu có."""
    path = common.PROCESSED / f"gen_inputs_{dataset}.csv"
    df = pd.read_csv(path, dtype={"product_id": str}).fillna("")
    features = common.PROCESSED / f"{dataset}_features.csv"
    if features.exists():
        extra = pd.read_csv(features, dtype={"product_id": str}).fillna("")
        # Chỉ ghép các cột chưa có trong gen_inputs để tránh tạo cột trùng tên (name_x, name_y).
        cols = ["product_id"] + [c for c in ("name", "price") if c in extra.columns and c not in df.columns]
        df = df.merge(extra[cols].drop_duplicates("product_id"), on="product_id", how="left").fillna("")
    return df


def parse_value(value, fallback):
    """Giải mã JSON và trả về giá trị mặc định khi dữ liệu lỗi."""
    if isinstance(value, type(fallback)):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, type(fallback)) else fallback
    except (ValueError, TypeError):
        return fallback


def select_test(df, dataset, n, seed, split="test"):
    """Lấy mẫu phân tầng, ưu tiên mô tả có độ dài phù hợp, và cố định mẫu.

    split="val" lấy mẫu kiểm tra độc lập từ tập kiểm định, dùng để kiểm tra prompt v2
    trên dữ liệu chưa từng xem khi sửa prompt (lưu riêng trong thư mục holdout/).
    """
    path = common.RESULTS / (f"test_items_{dataset}.csv" if split == "test" else f"holdout/{split}_items_{dataset}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, dtype={"product_id": str, "item_id": str}).fillna("")
    test = df[df["split"] == split].copy()
    if test.empty:
        raise ValueError("Không có sản phẩm test")
    test["preferred"] = test.reference_description.astype(str).str.len().between(200, 3000)
    # Phân bổ theo tỷ trọng toàn bộ danh mục, sau đó ưu tiên mô tả đủ độ dài trong từng danh mục.
    n = min(n, len(test))
    counts = test.groupby("category_l2", dropna=False).size()
    exact = counts * n / len(test)
    quota = exact.astype(int)
    for category in (exact - quota).sort_values(ascending=False).index[:n - int(quota.sum())]:
        quota.loc[category] += 1
    parts = []
    for category, group in test.groupby("category_l2", dropna=False):
        shuffled = group.sample(frac=1, random_state=seed).sort_values("preferred", ascending=False, kind="stable")
        parts.append(shuffled.head(int(quota.loc[category])))
    result = pd.concat(parts).sample(frac=1, random_state=seed).drop(columns="preferred").reset_index(drop=True)
    result.insert(0, "item_id", [str(i + 1) for i in range(len(result))])
    result.to_csv(path, index=False, encoding="utf-8")
    return result


def make_brief(row, dataset, objections):
    """Chuyển một dòng sản phẩm thành brief có cấu trúc."""
    category = str(row.get("category_l2", ""))
    brief = {"item_id": str(row["item_id"]), "category": category, "keywords": parse_value(row.get("keywords", ""), [])[:6], "attributes": parse_value(row.get("attributes", ""), {}), "objections": []}
    if dataset == "tiki":
        brief["objections"] = [x.get("phrase", "") for x in objections.get(category, [])[:2] if isinstance(x, dict)]
        if row.get("price", "") != "":
            brief["price"] = row["price"]
    return brief


def retrieve_examples(df, row, k=3):
    """Truy hồi mô tả thật tương tự, ưu tiên sản phẩm bán chạy."""
    train = df[(df["split"] == "train") & (df["product_id"].astype(str) != str(row["product_id"]))].copy()
    name = str(row.get("name", "")).strip().casefold()
    if name and "name" in train:
        train = train[train["name"].astype(str).str.strip().str.casefold() != name]
    train = train[train.reference_description.astype(str).str.strip() != ""]
    if train.empty:
        return []
    def text_of(r):
        """Nối danh mục và từ khóa để so sánh."""
        return str(r.get("category_l2", "")) + " " + " ".join(map(str, parse_value(r.get("keywords", ""), [])))
    texts = train.apply(text_of, axis=1).tolist()
    try:
        vectors = TfidfVectorizer().fit_transform(texts + [text_of(row)])
        scores = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
    except ValueError:
        scores = [0.0] * len(train)
    train = train.assign(score=scores)
    flag = "best_seller" if "best_seller" in train else "popular"
    train["priority"] = pd.to_numeric(train.get(flag, 0), errors="coerce") == 1
    train = train.sort_values(["priority", "score"], ascending=False, kind="stable").head(k)
    return [{"brief": {"category": r["category_l2"], "keywords": parse_value(r["keywords"], []), "attributes": parse_value(r["attributes"], {})}, "description": str(r["reference_description"])[:900]} for _, r in train.iterrows()]


def record(row, dataset, method, brief, output, latency, model, prompt):
    """Tạo bản ghi kết quả thống nhất cho mọi phương pháp."""
    normalized = {key: output.get(key, [] if key == "email_subjects" else "") for key in FIELDS}
    return {"item_id": str(row["item_id"]), "product_id": str(row["product_id"]), "method": method, "dataset": dataset, "lang": "vi" if dataset == "tiki" else "en", "brief": brief, "output": normalized, "reference_description": row.get("reference_description", ""), "reference_title": row.get("name", ""), "latency_s": latency, "model": model, "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(), "created_at": datetime.now(timezone.utc).isoformat()}


def append_record(path, obj):
    """Ghi một bản ghi JSONL bền vững sau mỗi sản phẩm."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def completed(path):
    """Đọc các item đã hoàn thành để tiếp tục chạy."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["item_id"]) for line in handle if line.strip()}


def build_prompt(method, briefs, rows, df, lang):
    """Xây prompt cho một lô theo phương pháp."""
    if method == "M1":
        return prompts.zero_shot(briefs, lang)
    if method == "M2":
        return prompts.engineered(briefs, lang)
    examples = {str(row["item_id"]): retrieve_examples(df, row) for row in rows}
    return prompts.fewshot_retrieval(briefs, examples, lang)


def run(dataset, methods, n_test, batch, seed, split="test"):
    """Chạy sinh theo lô, gọi lại riêng mục bị thiếu và ghi ngay kết quả."""
    df = read_data(dataset)
    items = select_test(df, dataset, n_test, seed, split)
    path = common.RESULTS / "objections_by_category.json"
    objections = json.loads(path.read_text(encoding="utf-8")) if dataset == "tiki" and path.exists() else {}
    lang = "vi" if dataset == "tiki" else "en"
    for method in methods:
        output_path = common.RESULTS / (f"generations_{method}_{dataset}.jsonl" if split == "test" else f"holdout/generations_{method}_{dataset}_{split}.jsonl")
        done = completed(output_path)
        pending = [row for _, row in items.iterrows() if str(row["item_id"]) not in done]
        for start in range(0, len(pending), batch):
            rows = pending[start:start + batch]
            briefs = [make_brief(row, dataset, objections) for row in rows]
            if method == "M0":
                for row, brief in zip(rows, briefs):
                    t0 = time.perf_counter()
                    output = prompts.template_output(brief, lang)
                    append_record(output_path, record(row, dataset, method, brief, output, time.perf_counter() - t0, "template", ""))
                continue
            prompt = build_prompt(method, briefs, rows, df, lang)
            t0 = time.perf_counter()
            try:
                result = llm_backends.extract_json(llm_backends.generate(prompt, tag=f"{dataset}:{method}"))
                if not isinstance(result, list):
                    result = []
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"Lỗi lô {method}: {exc}")
                result = []
            elapsed = (time.perf_counter() - t0) / len(rows)
            found = {str(x.get("item_id")): x for x in result if isinstance(x, dict) and all(key in x for key in FIELDS)}
            for row, brief in zip(rows, briefs):
                item_id = str(row["item_id"])
                used_prompt = prompt
                latency = elapsed
                if item_id not in found:
                    used_prompt = build_prompt(method, [brief], [row], df, lang)
                    t0 = time.perf_counter()
                    try:
                        retry = llm_backends.extract_json(llm_backends.generate(used_prompt, tag=f"{dataset}:{method}:retry"))
                        if isinstance(retry, list) and retry and isinstance(retry[0], dict) and str(retry[0].get("item_id")) == item_id and all(key in retry[0] for key in FIELDS):
                            found[item_id] = retry[0]
                    except (ValueError, RuntimeError, OSError) as exc:
                        print(f"Lỗi mục {item_id}: {exc}")
                    latency += time.perf_counter() - t0
                if item_id in found:
                    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini") if os.environ.get("OPENAI_API_KEY") else llm_backends.GEN_MODEL
                    append_record(output_path, record(row, dataset, method, brief, found[item_id], latency, model, used_prompt))
                else:
                    print(f"Chưa sinh được item {item_id}; có thể chạy lại.")


def main():
    """Phân tích tham số dòng lệnh và chạy các phương pháp."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tiki", "amazon"], required=True)
    parser.add_argument("--methods", default="M0,M1,M2,M3")
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()
    methods = args.methods.split(",")
    if not set(methods) <= {"M0", "M1", "M2", "M3"} or args.batch < 1 or args.n_test < 1:
        parser.error("Phương pháp hoặc kích thước không hợp lệ")
    run(args.dataset, methods, args.n_test, args.batch, args.seed, args.split)


if __name__ == "__main__":
    main()

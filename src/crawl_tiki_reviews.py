"""Thu thập đánh giá (review) cho các sản phẩm Tiki ĐÃ thu thập, với tốc độ chậm và điểm dừng nghiêm ngặt.

Bối cảnh: lúc 11:59 ngày 13/11/2026, API danh sách sản phẩm của Tiki bắt đầu trả trang HTML thay cho JSON
đối với chương trình thu thập (dấu hiệu chống bot), nên việc thu thập sản phẩm mới đã DỪNG hẳn
(xem data/raw/tiki_crawl_stop_note.json). Script này không gọi lại API danh sách. Nó chỉ gọi API đánh giá
cho các sản phẩm đã có, với khoảng cách 1,5–2,5 giây mỗi request. Chương trình dừng ngay khi gặp phản hồi
không phải JSON, mã 403, 429 hoặc CAPTCHA.

Quyền riêng tư: chỉ giữ product_id, review_id, rating, title, content, created_at, thank_count (hàm review_row
của crawl_tiki.py đã loại tên, mã, ảnh của khách hàng).
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from crawl_tiki import BASE, REVIEW_COLUMNS, UA, append, read_jsonl, review_row

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LOG = ROOT / "logs" / "crawl_tiki_reviews.log"


class BlockedError(RuntimeError):
    """Máy chủ có dấu hiệu từ chối chương trình thu thập: dừng toàn bộ."""


def log(message):
    """Ghi một dòng nhật ký có dấu thời gian."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().astimezone().isoformat()} {message}\n")


def fetch_reviews(session, product_id):
    """Gọi API đánh giá trang 1 (20 đánh giá) của một sản phẩm; dừng hẳn nếu bị chặn."""
    params = {"product_id": product_id, "page": 1, "limit": 20, "include": "comments", "sort": "score|desc,id|desc,stars|all"}
    response = session.get(BASE + "/api/v2/reviews", params=params, timeout=30)
    log(f"{response.status_code} {response.headers.get('Content-Type')} {response.url}")
    if response.status_code in (403, 429):
        raise BlockedError(f"HTTP {response.status_code}")
    if "json" not in (response.headers.get("Content-Type") or ""):
        raise BlockedError("Phản hồi không phải JSON (có thể là trang chống bot)")
    if "captcha" in response.text[:2000].lower():
        raise BlockedError("Gặp CAPTCHA")
    return response.json()


def sold_value(item):
    """Lấy số lượng đã bán từ bản ghi gốc."""
    sold = item.get("quantity_sold")
    if isinstance(sold, dict):
        return sold.get("value") or 0
    return item.get("all_time_quantity_sold") or 0


def main():
    """Chọn các sản phẩm bán chạy có đánh giá, thu thập trang đánh giá đầu tiên của từng sản phẩm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=500, help="số sản phẩm (xếp theo số đã bán) cần lấy đánh giá")
    args = parser.parse_args()
    products = read_jsonl(RAW / "tiki_products_raw.jsonl")
    candidates = [p for p in products if (p.get("review_count") or 0) > 0]
    candidates.sort(key=sold_value, reverse=True)
    candidates = candidates[: args.top]
    reviews_path = RAW / "tiki_reviews_raw.jsonl"
    existing = read_jsonl(reviews_path)
    done_products = {str(r.get("product_id")) for r in existing}
    seen = {(str(r.get("product_id")), str(r.get("review_id"))) for r in existing}
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "application/json"})
    n_requests = 0
    stop_reason = None
    for item in candidates:
        pid = str(item.get("id"))
        if pid in done_products:
            continue
        time.sleep(random.uniform(1.5, 2.5))
        try:
            payload = fetch_reviews(session, pid)
        except BlockedError as exc:
            stop_reason = str(exc)
            log(f"DỪNG: {stop_reason}")
            break
        except requests.RequestException as exc:
            log(f"lỗi mạng {type(exc).__name__}, bỏ qua sản phẩm {pid}")
            continue
        n_requests += 1
        for raw in payload.get("data") or []:
            record = review_row(pid, raw)
            key = (pid, str(record.get("review_id"))) if record else None
            if record and key not in seen:
                append(reviews_path, record)
                existing.append(record)
                seen.add(key)
        done_products.add(pid)
    pd.DataFrame(existing, columns=REVIEW_COLUMNS).to_csv(PROCESSED / "tiki_reviews.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "crawled_at": datetime.now().astimezone().isoformat(),
        "endpoint": BASE + "/api/v2/reviews",
        "requests": n_requests,
        "products_with_reviews_requested": len(done_products),
        "review_records": len(existing),
        "stop_reason": stop_reason,
        "rate": "1 request / 1,5–2,5 giây",
    }
    (RAW / "tiki_reviews_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

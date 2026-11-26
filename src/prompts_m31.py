"""Câu lệnh M3.1 dùng dữ kiện người bán đã xác nhận qua biểu mẫu có cấu trúc."""
from __future__ import annotations

import hashlib
import json

import prompts_v2


RULES = """
QUY TẮC M3.1
- CHỈ dùng dữ kiện trong confirmed_facts, attributes, price và style_guide. Không suy diễn một chi tiết sản phẩm từ loại hàng, ảnh, giá hoặc ví dụ.
- MỞ ĐẦU bằng loại hàng, chất liệu hoặc dáng đã được xác nhận. Ngay sau đó, nêu dữ kiện giúp khách quyết định mua theo thứ tự: cỡ và số đo, dáng hoặc công năng, cấu tạo, cách bảo quản. Giữ số và đơn vị đúng như dữ kiện.
- Nếu có cỡ hoặc số đo, trình bày chúng thành dòng riêng để khách quét nhanh. Không tự quy đổi cỡ hay cân nặng.
- PHỐI ĐỒ chỉ dùng style_guide của đúng loại hàng và chỉ khi confirmed_facts có dáng, màu hoặc công năng làm căn cứ. Tối đa hai gợi ý. Đồ lót, đồ ngủ, đồ mặc nhà và đồ bơi không được gợi ý như trang phục mặc ngoài.
- Không hứa đổi trả, bảo hành, khuyến mãi, giao hàng, chất lượng hay công dụng khi không có dữ kiện. Không dùng lời khen chung chung hoặc các câu có thể áp cho mọi sản phẩm.
- Độ dài mô tả phụ thuộc dữ kiện đã xác nhận. Khi dữ kiện ít, viết ngắn và dừng ở thông tin có căn cứ. Không thêm câu chung chung để đạt số từ.
- Kịch bản chốt sale chỉ trả lời các lo ngại trong brief bằng dữ kiện có sẵn hoặc bằng hành động người bán có thể làm ngay, như gửi ảnh thật hoặc đo lại sản phẩm.
"""


def build(briefs, confirmed_by_id, guide):
    """Ghép M2 với dữ kiện biểu mẫu và hướng dẫn phối đồ theo dòng hàng."""
    base = prompts_v2.engineered(briefs)
    head, _, _ = base.partition("\n\nBRIEF:\n")
    items = []
    for brief in briefs:
        item = dict(brief)
        item["confirmed_facts"] = confirmed_by_id.get(str(brief["item_id"]), "")
        item["style_guide"] = guide.get(brief.get("category"), {})
        items.append(item)
    return head + "\n" + RULES + "\nBRIEF:\n" + json.dumps(items, ensure_ascii=False, indent=1)


def template_hash():
    sample = {"item_id": "fixed", "category": "Đầm nữ", "keywords": ["đầm nữ"],
              "attributes": {}, "price": 100, "objections": ["Size không vừa"]}
    text = build([sample], {"fixed": "Loại hàng: Đầm nữ\nChất liệu chính: lụa"}, {"Đầm nữ": {}})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

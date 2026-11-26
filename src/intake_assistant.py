"""Biến thông tin người bán xác nhận thành dữ kiện có cấu trúc cho M3.1."""
from __future__ import annotations

from dataclasses import dataclass


COMMON_FIELDS = (
    ("material", "Chất liệu chính", "Ví dụ cotton, lụa, kaki"),
    ("size_measurements", "Cỡ và số đo", "Ví dụ M 46–52 kg, eo 68–72 cm"),
    ("care", "Cách giặt và bảo quản", "Ví dụ giặt tay nước lạnh, không sấy"),
)

CATEGORY_FIELDS = {
    "Đồ lót nữ": (
        ("fit_feature", "Kiểu dáng hoặc công năng", "Ví dụ cạp cao, không đường may"),
        ("construction", "Cấu tạo", "Ví dụ có mút tháo rời, đáy cotton"),
    ),
    "Đầm nữ": (
        ("fit_feature", "Dáng và chiều dài", "Ví dụ dáng chữ A, dài 105 cm"),
        ("construction", "Cấu tạo", "Ví dụ có lót, khóa kéo sau lưng"),
    ),
    "Quần nữ": (
        ("fit_feature", "Dáng và chiều dài", "Ví dụ ống suông, dài 94 cm"),
        ("construction", "Cấu tạo", "Ví dụ cạp chun sau, có túi"),
    ),
    "Chân váy": (
        ("fit_feature", "Dáng và chiều dài", "Ví dụ chữ A, dài 65 cm"),
        ("construction", "Cấu tạo", "Ví dụ có lót, khóa kéo bên hông"),
    ),
    "Trang phục bơi nữ": (
        ("fit_feature", "Kiểu dáng", "Ví dụ một mảnh, cổ vuông"),
        ("construction", "Cấu tạo", "Ví dụ có lót, mút ngực tháo rời"),
    ),
}


def fields_for(category: str):
    """Trả các câu hỏi cốt lõi, phù hợp với loại hàng đang được đăng."""
    return COMMON_FIELDS + CATEGORY_FIELDS.get(category, (("fit_feature", "Dáng hoặc đặc điểm nổi bật", "Ví dụ dáng suông, tay lỡ"),))


def missing_fields(category: str, answers: dict[str, str]) -> list[str]:
    """Chỉ coi một trường là hoàn tất khi người bán đã cung cấp nội dung cụ thể."""
    return [label for key, label, _ in fields_for(category) if not str(answers.get(key, "")).strip()]


def confirmed_facts(category: str, answers: dict[str, str], extra_facts: str = "") -> str:
    """Tạo dữ kiện ngắn, có nhãn để mô hình không biến lời quảng cáo thành thuộc tính sản phẩm."""
    parts = [f"Loại hàng: {category}"]
    for key, label, _ in fields_for(category):
        value = str(answers.get(key, "")).strip()
        if value:
            parts.append(f"{label}: {value}")
    if extra_facts.strip():
        parts.append(f"Dữ kiện người bán bổ sung: {extra_facts.strip()}")
    return "\n".join(parts)

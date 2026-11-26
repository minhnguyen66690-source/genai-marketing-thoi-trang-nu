"""Từ điển chủ đề phản đối v2 (sửa lỗi A49, A50 của kiểm toán 22/11/2026).

- Bản v1 có ký tự backspace (U+0008) thật thay cho ranh giới từ, nên các nhánh "size", "chật", "mỏng",
  "lỗi", "ship" không bao giờ khớp. Bản này viết ranh giới bằng lookaround trên ký tự chữ Unicode.
- Bản v1 bắt cả cách nói khen ("giống hình", "đúng mẫu", "mô tả"). Bản này chỉ giữ cách nói chê, và chỉ áp
  dụng cho đánh giá 1 đến 3 sao.
Có hàm tự kiểm `_selftest()` chạy khi import để chặn lỗi tái diễn.
"""
import re

W = r"(?<![^\W\d_])"   # trước cụm không phải là chữ cái
E = r"(?![^\W\d_])"    # sau cụm không phải là chữ cái


def _rx(*phrases):
    return re.compile("|".join(W + p + E for p in phrases), re.IGNORECASE)


THEMES = {
    "Size không vừa": _rx(r"không vừa", r"ko vừa", r"k vừa", r"chật", r"bị rộng", r"quá rộng", r"rộng quá",
                          r"quá nhỏ", r"nhỏ quá", r"quá to", r"to quá", r"đổi size", r"sai size", r"lệch size",
                          r"size (?:bị )?(?:nhỏ|to|rộng|chật)", r"form (?:nhỏ|to|rộng|chật)", r"ngắn quá", r"dài quá"),
    "Không giống hình/mô tả": _rx(r"không giống", r"ko giống", r"k giống", r"khác hình", r"khác ảnh", r"khác mẫu",
                                  r"sai mẫu", r"không đúng (?:mẫu|hình|mô tả|ảnh)", r"ko đúng (?:mẫu|hình|mô tả)",
                                  r"khác xa", r"không như (?:hình|ảnh|mô tả)", r"ko như (?:hình|ảnh)"),
    "Chất liệu kém": _rx(r"vải mỏng", r"mỏng (?:quá|tang|dính)", r"quá mỏng", r"vải xấu", r"vải cứng", r"chất (?:vải|liệu) (?:kém|xấu|tệ|không|ko|mỏng|cứng|thô)",
                         r"xù lông", r"bị xù", r"nhăn", r"phai màu", r"ra màu", r"bí", r"nóng", r"ngứa", r"thô", r"nhão"),
    "Màu sắc khác": _rx(r"khác màu", r"sai màu", r"lệch màu", r"màu (?:không|ko|k) (?:giống|đúng)", r"màu bị", r"màu khác"),
    "Đường may, lỗi sản phẩm": _rx(r"đường may", r"chỉ thừa", r"rách", r"bung", r"lỗi", r"hư", r"dơ", r"bẩn", r"mốc",
                                   r"thủng", r"tuột", r"may ẩu", r"ố"),
    "Giao hàng, đóng gói": _rx(r"giao (?:sai|thiếu|chậm|lâu|nhầm)", r"thiếu hàng", r"đóng gói (?:kém|ẩu|sơ sài|cẩu thả)",
                               r"ship (?:chậm|lâu)", r"vận chuyển (?:chậm|lâu)", r"hộp (?:bị )?(?:móp|rách)"),
}


# Từ vựng TRUNG TÍNH để biết kịch bản của shop có nhắc tới chủ đề lo ngại hay không. Khác THEMES (bắt lời chê
# của khách): shop viết "em tư vấn size theo cân nặng", không viết "chật quá". Chỉ số dùng tên "nhắc chủ đề",
# không đo chất lượng trả lời.
TOPICS = {
    "Size không vừa": _rx(r"size", r"kích cỡ", r"kích thước", r"số đo", r"cân nặng", r"chiều cao", r"vừa người", r"form"),
    "Không giống hình/mô tả": _rx(r"ảnh thật", r"hình thật", r"video", r"giống hình", r"đúng hình", r"như hình", r"đúng mẫu", r"ảnh thực tế", r"chụp thật"),
    "Chất liệu kém": _rx(r"chất liệu", r"chất vải", r"vải", r"độ dày", r"thoáng", r"co giãn", r"thấm hút"),
    "Màu sắc khác": _rx(r"màu", r"tông"),
    "Đường may, lỗi sản phẩm": _rx(r"đường may", r"kiểm tra", r"kiểm hàng", r"lỗi", r"đổi"),
    "Giao hàng, đóng gói": _rx(r"giao hàng", r"đóng gói", r"vận chuyển", r"gói hàng", r"giao"),
}


def topics_of(text):
    """Chủ đề lo ngại được NHẮC tới trong văn bản của shop (từ vựng trung tính)."""
    return [name for name, rx in TOPICS.items() if rx.search(str(text or ""))]


def themes_of(text):
    """Danh sách chủ đề khớp trong một đánh giá."""
    return [name for name, rx in THEMES.items() if rx.search(str(text or ""))]


def _selftest():
    cases = {
        "Áo bị chật, mặc không vừa": ["Size không vừa"],
        "giống hình, vải mỏng quá": ["Chất liệu kém"],
        "Hàng không giống hình, giao chậm": ["Không giống hình/mô tả", "Giao hàng, đóng gói"],
        "đúng mẫu, đẹp": [],
        "Áo bị lỗi đường may": ["Đường may, lỗi sản phẩm"],
        "shipper thân thiện": [],
        "sai màu so với ảnh": ["Màu sắc khác"],
    }
    for text, want in cases.items():
        got = themes_of(text)
        assert got == want, (text, got, want)
    assert not any("\x08" in rx.pattern for rx in THEMES.values())


_selftest()

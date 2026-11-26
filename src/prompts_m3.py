"""M3: M2 cải tiến từ kết quả vòng 1.

Ba thay đổi so với M2, mỗi thay đổi đáp lại một phát hiện của M2:
1. Thiếu chi tiết thật (lý do giám khảo chọn trang người bán): đưa thêm chi tiết người bán đã có
   (mô tả gốc đã bỏ văn mẫu thuế, rút gọn) để mô hình lấy dữ kiện kiểm chứng được và viết lại.
2. Gợi ý phối đồ sai loại hàng (người chấm phát hiện): đưa hướng dẫn phối đồ của đúng loại hàng,
   học từ mô tả người bán trong tập huấn luyện (results_v2/style_guide.json).
3. Câu sáo nghe "rất AI": cấm danh sách câu sáo và yêu cầu gợi ý cụ thể theo màu, kiểu dáng của chính sản phẩm;
   chương trình tự dò câu sáo và cho viết lại một lần.
Câu lệnh M3 được chỉnh trên tập phát triển rồi khóa riêng (results_v2/prompt_lock_m3.json).
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import prompts_v2

ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = ROOT / 'results_v2' / 'style_guide.json'
SELLER_WORDS = 300

# Câu sáo: ý chung chung áp cho sản phẩm nào cũng được. Dùng để cấm trong câu lệnh và để đo.
CLICHES = [
    r'tủ đồ', r'tạo (?:sự|cảm giác) thoải mái', r'mang lại (?:sự|cảm giác) thoải mái', r'thoải mái (?:suốt|cả) (?:cả )?ngày',
    r'phong cách (?:mong muốn|riêng|yêu thích|của (?:bạn|chị|mình))', r'theo sở thích', r'màu (?:sắc )?yêu thích', r'bạn yêu thích',
    r'là (?:một )?lựa chọn', r'tạo điểm nhấn', r'điểm nhấn (?:cho|nổi bật)', r'để hoàn thiện', r'hoàn thiện (?:set|bộ|tổng thể|vẻ)',
    r'dễ (?:phối|kết hợp|mix)(?: đồ)?(?: với| cùng)? (?:nhiều|mọi|đa dạng)', r'nhiều (?:dịp|hoàn cảnh|phong cách) khác nhau',
    r'phù hợp (?:với )?(?:nhiều|mọi) (?:dịp|hoàn cảnh)', r'(?:đổi|thay đổi|làm mới) phong cách', r'tự tin (?:tỏa sáng|xuống phố|diện)',
    r'tôn lên vẻ', r'(?:bạn|chị) đang tìm (?:một|kiếm)', r'người bạn đồng hành', r'không thể bỏ (?:qua|lỡ)', r'nâng tầm',
    r'vẻ ngoài (?:nữ tính|thanh lịch|trẻ trung|năng động)', r'mang đến (?:một )?(?:lựa chọn|vẻ|phong cách|cảm giác)',
    r'(?:item|món đồ) (?:không thể thiếu|cơ bản)', r'gu thời trang', r'hợp xu hướng',
]
CLICHE_RE = re.compile('|'.join(CLICHES), re.I)


def cliches(text):
    return [m.group(0) for m in CLICHE_RE.finditer(unicodedata.normalize('NFC', text or ''))]


def load_guide():
    return json.loads(GUIDE_PATH.read_text(encoding='utf-8')) if GUIDE_PATH.exists() else {}


def seller_details(product):
    """Mô tả gốc của người bán (đã bỏ văn mẫu thuế ở bước làm sạch), rút gọn SELLER_WORDS từ."""
    words = ' '.join([str(product.get('short_description') or ''), str(product.get('description_text') or '')]).split()
    return ' '.join(words[:SELLER_WORDS])


RULES = """
QUY TẮC RIÊNG CỦA BẢN NÀY (ưu tiên hơn quy tắc ở trên nếu khác nhau):
- CHI TIẾT NGƯỜI BÁN (seller_details): là mô tả gốc người bán đã đăng cho chính sản phẩm này. Được lấy MỌI dữ kiện kiểm
  chứng được trong đó (bảng số đo, chiều dài, độ dày, chất liệu, cấu tạo như có mút hay không, độ co giãn, cách giặt và bảo quản,
  thành phần bộ sản phẩm) và nên đưa các con số cụ thể vào mô tả. Không chép nguyên câu: viết lại bằng lời của bạn.
  Bỏ qua lời quảng cáo, lời khen, và mọi chính sách (đổi trả, bảo hành, freeship, quà tặng, cam kết) trong phần này.
  Nếu seller_details mâu thuẫn với attributes thì theo attributes.
- PHỐI ĐỒ VÀ DỊP MẶC: chỉ gợi ý theo style_guide của đúng loại hàng, áp vào màu và kiểu dáng của CHÍNH sản phẩm
  (ví dụ: "màu gạch đi với quần jean xanh nhạt"), tối đa 2 gợi ý, mỗi gợi ý nêu món đồ cụ thể. Tuyệt đối không gợi ý những điều
  trong khong_goi_y. Đồ lót, đồ ngủ, đồ mặc nhà, đồ bơi: không gợi ý như trang phục mặc ra ngoài.
- SIZE: nếu đã có bảng size hoặc khoảng cân nặng trong attributes hoặc seller_details thì nêu cụ thể size nào hợp với ai
  theo đúng số đó; chỉ mời khách nhắn số đo khi thật sự cần (ví dụ người ở ranh giới hai size).
- GIỌNG VĂN: viết như chủ shop đang tả món hàng đang cầm trên tay: câu ngắn, nói về chính sản phẩm, có con số khi có.
  Mỗi câu phải nói điều chỉ đúng với sản phẩm này. KHÔNG dùng các cụm sáo sau hoặc ý tương tự:
  "bổ sung vào tủ đồ", "tạo sự thoải mái", "phong cách mong muốn", "theo sở thích", "màu yêu thích", "là lựa chọn lý tưởng",
  "tạo điểm nhấn", "để hoàn thiện set đồ", "dễ phối với nhiều trang phục", "phù hợp nhiều dịp", "đổi phong cách",
  "tự tin tỏa sáng", "tôn lên vẻ đẹp", "bạn đang tìm một...", "người bạn đồng hành", "nâng tầm", "gu thời trang".
- MÔ TẢ: vẫn ít nhất 200 từ nếu đủ dữ kiện thật; nếu không đủ thì 150 từ trở lên, KHÔNG kéo dài bằng câu chung chung.
"""


def build(briefs, products_by_id, guide):
    """briefs: danh sách brief gọn như M2; thêm seller_details và style_guide cho từng sản phẩm."""
    items = []
    for b in briefs:
        p = products_by_id[str(b['item_id'])]
        g = guide.get(b.get('category'), {})
        items.append(dict(b, seller_details=seller_details(p), style_guide=g))
    base = prompts_v2.engineered(briefs)  # khung M2 (chuẩn viết, AIDA/SPIN, lo ngại, định dạng)
    head, _, _ = base.partition('\n\nBRIEF:\n')
    return head + '\n' + RULES + '\nBRIEF:\n' + json.dumps(items, ensure_ascii=False, indent=1)


def rewrite_note(hits):
    return ('\nBẢN TRƯỚC CÒN CÁC CỤM SÁO SAU, hãy viết lại toàn bộ và thay bằng ý cụ thể về sản phẩm: '
            + '; '.join(sorted(set(hits))))


def template_hash():
    sample = {'item_id': 'fixed', 'category': 'áo', 'keywords': ['áo nữ'], 'attributes': {}, 'price': 100, 'objections': ['size']}
    text = build([sample], {'fixed': {'short_description': '', 'description_text': 'Ví dụ.'}}, {'áo': {'dip': ['x']}})
    return hashlib.sha256((text + rewrite_note(['x']) + str(SELLER_WORDS) + CLICHE_RE.pattern).encode('utf-8')).hexdigest()

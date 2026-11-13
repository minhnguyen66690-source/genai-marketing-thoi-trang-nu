"""Mẫu prompt cho bốn cấu hình sinh nội dung (M0–M3).

Bốn cấu hình được thiết kế để tách riêng tác động của từng kỹ thuật:
- M0 (template): điền mẫu cố định, không dùng mô hình ngôn ngữ. Đây là baseline.
- M1 (zero-shot): chỉ có vai trò và brief, không có ràng buộc chuẩn.
- M2 (engineered): kỹ thuật thiết kế câu lệnh (prompt engineering). Thêm ràng buộc lấy từ checklist có nguồn
  và quy tắc chỉ dùng dữ kiện trong brief.
- M3 (few-shot có truy hồi): M2 cộng k ví dụ mô tả thật của sản phẩm bán chạy tương tự (Liu et al., 2022).

Mỗi ngưỡng trong M2 đều có nguồn ghi ở chú thích, khớp với mã trong _workspace/01_research/verified_refs.md.
"""
import json

# Khóa JSON bắt buộc cho mỗi sản phẩm trong câu trả lời của mô hình.
OUTPUT_KEYS = ["item_id", "title", "meta_description", "description", "sales_script", "email_subjects"]


def _brief_block(briefs):
    """Chuyển danh sách brief thành chuỗi JSON dễ đọc để đưa vào prompt."""
    return json.dumps(briefs, ensure_ascii=False, indent=1)


def _format_vi():
    """Phần mô tả định dạng đầu ra (tiếng Việt), dùng chung cho M1–M3."""
    return (
        "ĐỊNH DẠNG ĐẦU RA: chỉ trả về một mảng JSON hợp lệ, không giải thích, không markdown. "
        "Mỗi phần tử ứng với một sản phẩm và có đúng các khóa: "
        "item_id (giữ nguyên như brief), title, meta_description, description, "
        "sales_script, email_subjects (mảng gồm 3 chuỗi)."
    )


def _format_en():
    """Phần mô tả định dạng đầu ra (tiếng Anh), dùng chung cho M1–M3."""
    return (
        "OUTPUT FORMAT: return only one valid JSON array, no explanation, no markdown. "
        "Each element corresponds to one product and has exactly these keys: "
        "item_id (copied from the brief), title, meta_description, description, "
        "sales_script, email_subjects (an array of 3 strings)."
    )


def zero_shot(briefs, lang="vi"):
    """M1: prompt tối giản, chỉ có vai trò, nhiệm vụ và brief (không có ràng buộc chuẩn)."""
    if lang == "vi":
        head = (
            "Bạn là người viết nội dung cho một cửa hàng thời trang nữ trên sàn thương mại điện tử.\n"
            "Với mỗi sản phẩm trong brief, hãy viết: tiêu đề sản phẩm, đoạn mô tả meta, bài mô tả sản phẩm, "
            "kịch bản chốt sale (tin nhắn tư vấn gửi khách hàng) và 3 tiêu đề email marketing."
        )
        return head + "\n" + _format_vi() + "\n\nBRIEF:\n" + _brief_block(briefs)
    head = (
        "You write content for a women's fashion store on an e-commerce marketplace.\n"
        "For each product in the brief, write: a product title, a meta description, a product description, "
        "a closing sales script (a chat message to the customer) and 3 marketing email subject lines."
    )
    return head + "\n" + _format_en() + "\n\nBRIEF:\n" + _brief_block(briefs)


def _rules_vi():
    """Bộ ràng buộc của M2 (tiếng Việt). Mỗi dòng ghi nguồn ở chú thích Python."""
    rules = [
        "VAI TRÒ: bạn là copywriter thương mại điện tử chuyên thời trang nữ, viết tiếng Việt tự nhiên, đúng chính tả.",
        "TỪ KHÓA CHÍNH: là từ khóa đầu tiên trong danh sách keywords của từng sản phẩm.",
        # Nguyên tắc trung thực: Google khuyến nghị nội dung chính xác, hữu ích (S01, S07); Gmail cấm tiêu đề gây hiểu lầm (E16).
        "DỮ KIỆN: chỉ dùng thông tin có trong brief. Không bịa giảm giá, freeship, quà tặng, số lượng đã bán, "
        "đánh giá của khách, bảo hành, chính sách đổi trả, chất liệu hay màu sắc không có trong brief.",
        # Bản v2: bản v1 làm mô hình viết câu như "brief chưa cung cấp chất liệu" vào nội dung gửi khách (rò rỉ chỉ dẫn).
        "GIỌNG VĂN: viết như người bán đang giới thiệu sản phẩm với khách. TUYỆT ĐỐI không nhắc đến từ 'brief', "
        "'dữ liệu', 'thông tin được cung cấp' và không viết câu nói rằng thông tin nào đó bị thiếu. "
        "Thông tin nào brief không có thì bỏ hẳn ý đó. Có thể thêm gợi ý phối đồ, dịp mặc hợp với từ khóa "
        "và lời khuyên chọn size nói chung, vì đó là gợi ý, không phải dữ kiện về sản phẩm.",
        # S19 Ahrefs (2024): tiêu đề 50–60 ký tự; S11 Yoast: từ khóa chính nằm trong tiêu đề.
        "TIÊU ĐỀ (title): dài 50–60 ký tự, bắt đầu bằng từ khóa chính.",
        # S12 Yoast (2025): meta description 120–156 ký tự; S11: chứa từ khóa chính.
        "MÔ TẢ META (meta_description): dài 120–156 ký tự, chứa từ khóa chính, kết bằng một lời mời xem sản phẩm.",
        # S13 Yoast (2025) và S24 Tiki Academy: mô tả tối thiểu khoảng 200 từ; S10 Yoast: từ khóa trong đoạn đầu;
        # S09 Yoast: mật độ từ khóa 0,5–3%; S01 Google: văn bản dễ đọc, có tổ chức.
        "MÔ TẢ SẢN PHẨM (description): ít nhất 200 từ; từ khóa chính xuất hiện trong câu đầu tiên và lặp lại "
        "tự nhiên 2–4 lần trong toàn bài; chia đoạn ngắn và có từ 3 gạch đầu dòng trở lên, mỗi gạch là một ý "
        "có nội dung thật (đặc điểm theo từ khóa, thuộc tính có trong brief, dịp mặc, gợi ý phối đồ, cách chọn size).",
        # C01 AIDA (HubSpot); C02–C03 SPIN (Huthwaite): khung cấu phần, không phải ngưỡng.
        "KỊCH BẢN CHỐT SALE (sales_script): tin nhắn tư vấn 120–200 từ theo trình tự AIDA "
        "(gây chú ý, tạo hứng thú, khơi gợi mong muốn, kêu gọi hành động). Đặt 1–2 câu hỏi tìm hiểu nhu cầu "
        "kiểu SPIN (hoàn cảnh mặc, vấn đề gặp phải). Nếu brief có objections thì trả lời lần lượt từng phản đối "
        "bằng hành động người bán làm được ngay (gửi thêm ảnh hoặc video thật, đo lại thông số, tư vấn size theo "
        "chiều cao cân nặng, kiểm tra kỹ trước khi gói hàng), không hứa chính sách không có trong brief. "
        "Kết thúc bằng một lời kêu gọi hành động cụ thể (ví dụ mời đặt hàng hoặc gửi số đo để được tư vấn size).",
        # E01–E03 Mailchimp: ≤9 từ, ≤60 ký tự, ≤3 dấu câu, ≤1 emoji; E16 Gmail: không giả Re:/Fwd:.
        "TIÊU ĐỀ EMAIL (email_subjects): 3 phương án khác nhau, mỗi phương án tối đa 9 từ và 60 ký tự, "
        "tối đa 3 dấu câu, tối đa 1 emoji, có từ khóa chính hoặc một từ khóa trong brief, "
        "không dùng tiền tố Re:/Fwd:, không hứa ưu đãi không có trong brief.",
    ]
    return "\n".join("- " + r for r in rules)


def _rules_en():
    """Bộ ràng buộc của M2 (tiếng Anh), cùng nguồn với bản tiếng Việt."""
    rules = [
        "ROLE: you are an e-commerce copywriter specialised in women's fashion.",
        "PRIMARY KEYWORD: the first keyword in each product's keyword list.",
        "FACTS: use only information in the brief. Never invent discounts, free shipping, gifts, sales counts, "
        "reviews, warranties, return policies, materials or colours.",
        "VOICE: write as the seller talking to a shopper. NEVER mention the word 'brief', 'data' or 'provided "
        "information', and never write that some detail is missing or not specified; simply leave that point out. "
        "Styling ideas, suitable occasions and general size advice are allowed because they are suggestions, not product facts.",
        "TITLE: 50–60 characters, starting with the primary keyword.",
        "META DESCRIPTION: 120–156 characters, containing the primary keyword, ending with an invitation to view the product.",
        "DESCRIPTION: at least 200 words; the primary keyword appears in the first sentence and 2–4 times naturally overall; "
        "short paragraphs plus at least 3 bullet points, each a real point (keyword features, brief attributes, occasion, styling, size advice).",
        "SALES SCRIPT: a 120–200-word chat message following AIDA (attention, interest, desire, action). Ask 1–2 SPIN-style "
        "needs questions (situation, problem). If the brief lists objections, answer each one with actions the seller can take now "
        "(send extra real photos, re-measure, size advice from height and weight, careful packing), without promising policies absent from the brief. "
        "End with one concrete call to action.",
        "EMAIL SUBJECTS: 3 different options, each at most 9 words and 60 characters, at most 3 punctuation marks, at most 1 emoji, "
        "containing the primary keyword or another brief keyword, no fake Re:/Fwd:, no offers absent from the brief.",
    ]
    return "\n".join("- " + r for r in rules)


def engineered(briefs, lang="vi"):
    """M2: prompt engineering gồm vai trò, quy tắc dữ kiện, ràng buộc chuẩn có nguồn và định dạng đầu ra."""
    if lang == "vi":
        return (
            "Nhiệm vụ: viết nội dung bán hàng cho các sản phẩm thời trang nữ trong brief.\n"
            "YÊU CẦU:\n" + _rules_vi() + "\n" + _format_vi() + "\n\nBRIEF:\n" + _brief_block(briefs)
        )
    return (
        "Task: write sales content for the women's fashion products in the brief.\n"
        "REQUIREMENTS:\n" + _rules_en() + "\n" + _format_en() + "\n\nBRIEF:\n" + _brief_block(briefs)
    )


def fewshot_retrieval(briefs, examples, lang="vi"):
    """M3: M2 cộng ví dụ truy hồi riêng cho từng sản phẩm.

    examples: dict {item_id: [{"brief": {...}, "description": "..."}]}, là mô tả thật của sản phẩm bán chạy
    tương tự trong tập train. Ví dụ chỉ để học giọng văn, cấu trúc và từ vựng của ngành.
    """
    if lang == "vi":
        guide = (
            "\nVÍ DỤ THAM KHẢO (mô tả thật của sản phẩm bán chạy tương tự, lấy từ tập huấn luyện). "
            "Chỉ học giọng văn, cách trình bày và từ vựng ngành; KHÔNG chép dữ kiện của ví dụ sang sản phẩm cần viết.\n"
        )
    else:
        guide = (
            "\nREFERENCE EXAMPLES (real descriptions of similar best-selling products from the training set). "
            "Learn tone, structure and vocabulary only; do NOT copy facts from the examples into the target product.\n"
        )
    return engineered(briefs, lang) + "\n" + guide + json.dumps(examples, ensure_ascii=False, indent=1)


def template_output(brief, lang="vi"):
    """M0: baseline điền mẫu cố định, không dùng mô hình ngôn ngữ.

    Mô phỏng cách làm thủ công phổ biến: ghép từ khóa và thuộc tính vào câu mẫu có sẵn.
    """
    keywords = [str(k) for k in brief.get("keywords", []) if str(k).strip()]
    category = str(brief.get("category", "")).strip()
    attrs = brief.get("attributes", {}) or {}
    main = keywords[0] if keywords else category
    attr_text = "; ".join(f"{k}: {v}" for k, v in attrs.items() if str(v).strip())
    if lang == "vi":
        title = f"{main} {' '.join(keywords[1:3])} - {category}".strip()
        lines = [
            f"{main.capitalize()} thuộc danh mục {category}.",
            "Đặc điểm nổi bật: " + ", ".join(keywords) + "." if keywords else "",
            f"Thông tin sản phẩm: {attr_text}." if attr_text else "",
            "Sản phẩm phù hợp nhiều dịp sử dụng hằng ngày.",
            "Vui lòng liên hệ cửa hàng để được tư vấn chọn size.",
        ]
        script = (
            f"Chào chị, mẫu {main} bên em hiện vẫn còn hàng ạ. "
            "Chị cho em xin chiều cao và cân nặng để em tư vấn size phù hợp nhé. "
            "Chị đặt hàng ngay để nhận sản phẩm sớm ạ!"
        )
        subjects = [f"{main} - mẫu mới", f"Khám phá {main}", f"{category}: gợi ý cho bạn"]
    else:
        title = f"{main} {' '.join(keywords[1:3])} - {category}".strip()
        lines = [
            f"{main.capitalize()} from our {category} collection.",
            "Key features: " + ", ".join(keywords) + "." if keywords else "",
            f"Details: {attr_text}." if attr_text else "",
            "Suitable for many everyday occasions.",
            "Contact us for size advice.",
        ]
        script = (
            f"Hi, the {main} is in stock. Share your height and weight and we will suggest the right size. "
            "Order now to get it soon!"
        )
        subjects = [f"New: {main}", f"Discover {main}", f"{category}: picked for you"]
    description = "\n".join(line for line in lines if line)
    meta = f"{main} - {category}. " + (", ".join(keywords[1:4]) + "." if len(keywords) > 1 else "")
    return {
        "title": title,
        "meta_description": meta.strip(),
        "description": description,
        "sales_script": script,
        "email_subjects": subjects,
    }

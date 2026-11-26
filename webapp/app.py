"""Trợ lý viết nội dung bán hàng cho shop thời trang nữ.

Giao diện dành cho người bán: chỉ hỏi những gì cần để viết được nội dung, không hiển thị
thuật ngữ nghiên cứu. Phần số liệu của bài báo nằm trong tài liệu nộp kèm, không đưa lên đây.
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_api import generate_via_api
from metrics_content import detect_lang, email_checks, factual_support, seo_checks
from prompt_leakage import PATTERN as LEAK_PATTERN
from prompts import engineered
import prompts_m3
import prompts_m31
from intake_assistant import confirmed_facts, fields_for, missing_fields

DATA = ROOT / "webapp" / "data"
MAX_RUNS = 20
# Ví dụ dựng sẵn cho chế độ ?demo=1: chi tiết thật kiểu người bán hay có (bảng size theo cân nặng, vải, cách giặt).
DEMO_FACTS = ("S: 40-45kg, eo 62-64cm, dài 55cm; M: 46-50kg, eo 66-68cm, dài 56cm; L: 51-55kg, eo 70-72cm, dài 57cm. "
              "Vải kaki dày vừa, không nhăn, có lớp lót trong. Khóa kéo ẩn bên hông, xếp ly trước. "
              "Giặt tay nước lạnh, không sấy, ủi nhiệt độ thấp.")

# Nguồn công bố của từng ngưỡng: (tên tổ chức, năm nếu trang có ghi, địa chỉ).
REFS = {
    "S19": ("Ahrefs", "2024", "https://ahrefs.com/blog/title-tag-seo/"),
    "S23": ("Tiki Academy", "", "https://hocvien.tiki.vn/faq/quy-chuan-ve-ten-san-pham/"),
    "S12": ("Yoast", "2025", "https://yoast.com/meta-descriptions/"),
    "S11": ("Yoast", "", "https://yoast.com/help/yoast-seo-for-shopify-editor/"),
    "S10": ("Yoast", "", "https://yoast.com/help/yoast-seo-for-shopify-editor/"),
    "S09": ("Yoast", "", "https://yoast.com/features/keyphrase-density/"),
    "S13": ("Yoast", "2025", "https://yoast.com/blog-post-word-count-seo/"),
    "S24": ("Tiki Academy", "", "https://hocvien.tiki.vn/faq/quy-chuan-ve-noi-dung-dang-san-pham-cho-tung-nganh-hang/"),
    "E01": ("Mailchimp", "", "https://mailchimp.com/help/best-practices-for-email-subject-lines/"),
    "E02": ("Mailchimp", "", "https://mailchimp.com/help/best-practices-for-email-subject-lines/"),
    "E03": ("Mailchimp", "", "https://mailchimp.com/help/best-practices-for-email-subject-lines/"),
    "E05": ("Omnisend", "2022", "https://www.omnisend.com/blog/bfcm-email-analysis-less-is-more/"),
    "E16": ("Google", "", "https://support.google.com/mail/answer/81126?hl=en"),
}
# key: (ngưỡng hiển thị, mã nguồn, trường giá trị đo, câu giải thích cho người bán)
SEO = {
    "title_len_50_60": ("50 đến 60 ký tự", "S19", "title_chars", "Tên sản phẩm dài 50 đến 60 ký tự để hiện đủ trên Google"),
    "title_tiki_15_50": ("15 đến 50 ký tự", "S23", "title_chars", "Tên sản phẩm nằm trong khoảng Tiki khuyến nghị"),
    "title_tiki_max_225": ("≤225 ký tự", "S23", "title_chars", "Tên sản phẩm không vượt giới hạn của sàn"),
    "meta_len_120_156": ("120 đến 156 ký tự", "S12", "meta_chars", "Đoạn tóm tắt hiện dưới tên trên Google dài vừa đủ"),
    "keyphrase_in_title": ("Có", "S11", None, "Từ khóa chính có trong tên sản phẩm"),
    "keyphrase_in_first_paragraph": ("Có", "S10", None, "Từ khóa chính xuất hiện ngay đoạn mở đầu"),
    "keyphrase_in_meta": ("Có", "S11", None, "Từ khóa chính có trong đoạn tóm tắt"),
    "density_proxy_0_5_3": ("0,5 đến 3%", "S09", "keyphrase_density_proxy", "Mật độ từ khóa trong bài (chỉ tham khảo)"),
    "desc_words_ge_200": ("≥200 từ", "S13", "description_words", "Mô tả dài từ 200 từ trở lên"),
}
EMAIL = {
    "mailchimp_le9w_le60c": ("≤9 từ và ≤60 ký tự", "E01", "Tiêu đề email ngắn gọn, không bị cắt"),
    "omnisend_4_6w_or_21_40c": ("4 đến 6 từ hoặc 21 đến 40 ký tự", "E05", "Tiêu đề email trong khoảng được mở nhiều nhất"),
    "punct_le3": ("≤3 dấu câu", "E02", "Không lạm dụng dấu câu"),
    "emoji_le1": ("≤1 emoji", "E03", "Nhiều nhất một emoji"),
    "no_fake_re_fwd": ("Không có Re:/Fwd:", "E16", "Không giả mạo thư trả lời"),
}


def load(name, fallback):
    """Đọc dữ liệu tùy chọn."""
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def source_text(code):
    """Ghi nguồn ngắn gọn: tổ chức, năm nếu trang có ghi, kèm địa chỉ.

    Bảng kiểm tra chỉ hiển thị tiêu chí có nguồn công bố, nên mọi tiêu chí đều có mã nguồn.
    """
    name, year, url = REFS[code]
    return f"{name} ({year}) · {url}" if year else f"{name} · {url}"


def split_items(text):
    """Tách các mục do người dùng nhập."""
    return [part.strip() for part in text.split(",") if part.strip()]


def secret_of(name):
    """Đọc khóa do chủ trang cấu hình, bỏ qua khi chưa đặt."""
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except (FileNotFoundError, KeyError):
        return os.environ.get(name, "")


def collect_checks(content, brief, lang):
    """Tính mọi tiêu chí, trả về (dòng chi tiết, tóm tắt dễ hiểu, kết quả đo)."""
    keywords = brief.get("keywords", [])
    seo = seo_checks(content.get("title", ""), content.get("meta_description", ""), content.get("description", ""),
                     keywords[0] if keywords else "", keywords, lang)
    # Từ khóa viết một thứ tiếng còn nội dung viết thứ tiếng khác thì phép so khớp nguyên cụm
    # không còn ý nghĩa, nên các tiêu chí dựa trên từ khóa được đánh dấu là không áp dụng.
    mixed = bool(keywords) and detect_lang(" ".join(keywords)) != lang
    rows, passed, failed = [], [], []
    for key, (threshold, ref, measure, friendly) in SEO.items():
        flag = None if key == "density_proxy_0_5_3" else seo["checks"][key]
        if mixed and "keyphrase" in key:
            flag = None
        value = seo.get(measure) if measure else flag
        if isinstance(value, float):
            value = round(value, 2)
        rows.append({"Tiêu chí": friendly, "Đo được": str(value) if value is not None else "không có",
                     "Khuyến nghị": threshold, "Đạt?": "✅" if flag is True else "❌" if flag is False else "không có",
                     "Nguồn khuyến nghị": source_text(ref)})
        # Khoảng tên Tiki 15–50 ký tự mâu thuẫn với khoảng 50–60 của Ahrefs nên chỉ hiển thị, không tính vào tỷ lệ.
        if isinstance(flag, bool) and key != "title_tiki_15_50":
            (passed if flag else failed).append(friendly)
    email_pass, email_fail = [], []
    for index, subject in enumerate(content.get("email_subjects", []), 1):
        email = email_checks(subject, keywords, brief)
        values = {"mailchimp_le9w_le60c": f"{email['words_space']} từ, {email['chars']} ký tự",
                  "omnisend_4_6w_or_21_40c": f"{email['words_space']} từ, {email['chars']} ký tự",
                  "punct_le3": email["punctuation"], "emoji_le1": email["emoji"]}
        for key, (threshold, ref, friendly) in EMAIL.items():
            flag = email["checks"][key]
            rows.append({"Tiêu chí": f"Email {index}: {friendly}",
                         "Đo được": str(values.get(key, flag)) if flag is not None else "không có",
                         "Khuyến nghị": threshold, "Đạt?": "✅" if flag is True else "❌" if flag is False else "không có",
                         "Nguồn khuyến nghị": source_text(ref)})
            if isinstance(flag, bool):
                (email_pass if flag else email_fail).append(f"Email {index}: {friendly}")
    all_text = " ".join([content.get("title", ""), content.get("meta_description", ""), content.get("description", ""),
                         content.get("sales_script", "")] + content.get("email_subjects", []))
    unsupported = [] if mixed else factual_support(all_text, brief)["unsupported_list"]
    return rows, {"passed": passed, "failed": failed, "email_passed": email_pass, "email_failed": email_fail,
                  "unsupported": unsupported, "mixed": mixed}, seo


def show_review(content, brief, lang, detailed=False):
    """Tóm tắt dễ hiểu trước, bảng đầy đủ kèm nguồn trong phần mở rộng."""
    rows, summary, seo = collect_checks(content, brief, lang)
    n_pass, n_fail = len(summary["passed"]), len(summary["failed"])
    n_email_pass, n_email_fail = len(summary["email_passed"]), len(summary["email_failed"])
    left, right = st.columns(2)
    left.metric("Nội dung trang sản phẩm", f"{n_pass}/{n_pass + n_fail} đạt")
    if n_email_pass + n_email_fail:
        right.metric("Tiêu đề email", f"{n_email_pass}/{n_email_pass + n_email_fail} đạt")
    if summary["failed"]:
        st.markdown("**Nên sửa trước khi đăng**")
        for item in summary["failed"]:
            st.markdown(f"- {item}")
    else:
        st.success("Nội dung đạt toàn bộ tiêu chí kiểm tra được.")
    if summary["unsupported"]:
        st.warning("Các chi tiết sau có trong bài nhưng không có trong thông tin bạn nhập. Nếu là gợi ý phối đồ thì giữ; "
                   "nếu nói về chính sản phẩm thì kiểm tra lại cho đúng: " + ", ".join(summary["unsupported"]))
    if summary.get("mixed"):
        st.info("Từ khóa bạn nhập khác ngôn ngữ với nội dung được viết, nên các tiêu chí về từ khóa và phần "
                "kiểm tra chi tiết sản phẩm được bỏ qua. Nhập từ khóa cùng ngôn ngữ với nội dung để chấm đầy đủ.")
    with st.expander("Xem từng tiêu chí và nguồn khuyến nghị", expanded=detailed):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Mọi ngưỡng trong bảng đều là khuyến nghị đã công bố của Ahrefs, Yoast, Tiki Academy, "
                   "Mailchimp, Omnisend và Google, bấm địa chỉ để đọc nguồn gốc. Đạt các ngưỡng này không bảo "
                   "đảm bán được nhiều hơn.")
    return seo


def parse_response(text):
    """Đọc JSON đầu tiên, kể cả khi mô hình bọc trong khối mã."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    body = match.group(1) if match else text
    positions = [position for position in (body.find("["), body.find("{")) if position >= 0]
    if not positions:
        raise ValueError("Phản hồi không chứa JSON hợp lệ.")
    result = json.JSONDecoder().raw_decode(body[min(positions):])[0]
    if isinstance(result, list):
        result = result[0] if result else None
    if not isinstance(result, dict) or any(not isinstance(result.get(key), str) for key in ("title", "meta_description", "description", "sales_script")) or not isinstance(result.get("email_subjects"), list) or not all(isinstance(item, str) for item in result["email_subjects"]):
        raise ValueError("Phản hồi thiếu trường nội dung hợp lệ.")
    return result


def show_content(content, brief, lang):
    """Trình bày nội dung theo thứ tự người bán dùng, mỗi khối có nút sao chép."""
    labels = {"vi": ("Tên sản phẩm", "Đoạn tóm tắt ngắn (hiện dưới tên trên Google)", "Mô tả sản phẩm",
                     "Kịch bản nhắn tin chốt đơn", "Tiêu đề email giới thiệu"),
              "en": ("Product title", "Meta description", "Product description",
                     "Sales chat script", "Email subject lines")}[lang]
    for label, value in zip(labels, [content.get("title", ""), content.get("meta_description", ""),
                                     content.get("description", ""), content.get("sales_script", ""),
                                     "\n".join(content.get("email_subjects", []))]):
        if value:
            st.markdown(f"**{label}**")
            st.code(value, language=None, wrap_lines=True)
    st.caption("Bấm biểu tượng sao chép ở góc phải mỗi khung để dán sang sàn hoặc trình soạn email.")
    st.divider()
    st.markdown("#### Bản nháp này đã đạt chuẩn chưa?")
    seo = show_review(content, brief, lang)
    benchmark = load("category_benchmarks.json", {}).get(brief.get("category"), {}).get("all")
    if benchmark:
        with st.expander("So với sản phẩm cùng loại trên Tiki"):
            st.dataframe(pd.DataFrame([
                {"Chỉ số": "Độ dài mô tả (từ)", "Nội dung vừa tạo": seo["description_words"],
                 "Trung vị cùng loại": benchmark["desc_len_words"]},
                {"Chỉ số": "Số gạch đầu dòng", "Nội dung vừa tạo": seo["n_bullets"],
                 "Trung vị cùng loại": benchmark["n_bullets"]},
            ]), hide_index=True)
            st.caption(f"Trung vị của {benchmark['n']} sản phẩm cùng loại trên Tiki (thu thập 22/11/2026). "
                       "Đây là mức phổ biến, không phải mức bảo đảm bán chạy.")
    st.download_button("Tải toàn bộ nội dung (JSON)", json.dumps(content, ensure_ascii=False, indent=2),
                       "noi-dung.json", "application/json")


def build_prompt(brief, lang, facts=""):
    """Ghép câu lệnh từ lựa chọn của người bán.

    Tiếng Việt dùng M3.1 với dữ kiện người bán xác nhận theo biểu mẫu. Tiếng Anh dùng bản M2 tiếng Anh.
    """
    if lang == "vi":
        guide = load("style_guide.json", {})
        prompt = prompts_m31.build([brief], {str(brief["item_id"]): facts}, guide)
    else:
        prompt = engineered([brief], lang)
    # Bản tóm tắt có thể viết bằng tiếng Việt nhưng người dùng chọn đầu ra tiếng Anh,
    # nên phải nêu rõ ngôn ngữ đầu ra; dòng này chỉ dùng cho website, không dùng khi chạy thực nghiệm.
    prompt += ("\n\nOUTPUT LANGUAGE: write every field of the JSON in English, even if the brief is "
               "written in Vietnamese. Translate Vietnamese category names and keywords into natural English."
               if lang == "en" else
               "\n\nNGÔN NGỮ ĐẦU RA: viết toàn bộ các trường trong JSON bằng tiếng Việt có dấu.")
    return prompt


def cliche_phrases(content):
    """Các cụm sáo (prompts_m3.CLICHES) còn trong nội dung tiếng Việt."""
    texts = [str(content.get(key) or "") for key in ("title", "meta_description", "description", "sales_script")]
    texts += [str(item) for item in content.get("email_subjects") or []]
    return sorted(set(prompts_m3.cliches(" ".join(texts))))


def leaked_phrases(content):
    """Trả về các cụm cho thấy câu lệnh lọt vào nội dung gửi khách, ví dụ "brief chưa cung cấp"."""
    texts = [str(content.get(key) or "") for key in ("title", "meta_description", "description", "sales_script")]
    texts += [str(item) for item in content.get("email_subjects") or []]
    return sorted({match.group(0) for text in texts for match in LEAK_PATTERN.finditer(text)})


REPAIR_NOTE = {
    "vi": ("\n\nSỬA LẠI: bản trước có câu nhắc đến dữ liệu đầu vào hoặc cụm sáo ({found}). Viết lại toàn bộ, giữ đúng "
           "định dạng. Thay cụm sáo bằng ý cụ thể về chính sản phẩm; ý nào không có dữ kiện thì bỏ hẳn, không viết rằng "
           "thông tin bị thiếu."),
    "en": ("\n\nREVISE: the previous draft mentioned the input data ({found}). Rewrite everything in the same format. "
           "Drop any point without a fact instead of saying the detail is missing."),
}


def reset_all():
    """Xóa kết quả và dựng lại biểu mẫu với bộ khóa mới.

    Xóa khóa của widget không đủ để Streamlit trả ô về mặc định, nên mỗi lần làm lại
    tăng số phiên bản và gắn vào khóa widget để Streamlit coi đó là ô hoàn toàn mới.
    """
    st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
    st.session_state.pop("generated_result", None)


def clear_keywords():
    """Đổi loại sản phẩm thì bỏ các từ khóa gợi ý của loại cũ, vì danh sách gợi ý khác nhau."""
    st.session_state.pop(f"suggested_{st.session_state.get('form_version', 0)}", None)


st.set_page_config(page_title="Trợ lý nội dung sản phẩm", page_icon="✍️", layout="centered")
st.title("Trợ lý nội dung sản phẩm")

owner_gemini, owner_openai = secret_of("GEMINI_API_KEY"), secret_of("OPENAI_API_KEY")
provider = "gemini" if owner_gemini else "openai"
api_key = owner_gemini or owner_openai
with st.sidebar:
    st.caption(f"Đã dùng {st.session_state.get('generation_count', 0)}/{MAX_RUNS} lượt trong phiên này.")

viet, kiem_tra = st.tabs(["✍️ Viết nội dung", "🔍 Kiểm tra nội dung có sẵn"])

with viet:
    categories = load("categories.json", {})
    objections = load("objections_by_category.json", {})
    demo = st.query_params.get("demo") == "1"
    category_list = list(categories) or ["Khác"]
    demo_category = "Chân váy" if "Chân váy" in category_list else category_list[0]

    st.markdown("#### Sản phẩm của bạn")
    version = st.session_state.get("form_version", 0)
    category = st.selectbox("Loại sản phẩm", category_list,
                            index=category_list.index(demo_category) if demo else 0, key=f"category_{version}",
                            on_change=clear_keywords)
    suggested = st.multiselect("Cụm từ hay dùng trong tên sản phẩm cùng loại", categories.get(category, []),
                               placeholder="Chọn vài từ, ví dụ: chân váy chữ a, công sở", key=f"suggested_{version}")
    free_keywords = st.text_input("Từ khóa khác, cách nhau bằng dấu phẩy",
                                  value="chân váy chữ a, công sở, xếp ly" if demo else "",
                                  placeholder="váy dự tiệc, vải lụa, dáng suông", key=f"free_keywords_{version}")
    language = st.radio("Viết bằng", ["Tiếng Việt", "English"], horizontal=True, key=f"language_{version}")
    lang = "vi" if language == "Tiếng Việt" else "en"
    if lang == "en":
        st.caption("Chọn English thì nên nhập từ khóa và chi tiết sản phẩm bằng tiếng Anh, "
                   "vì phần kiểm tra so khớp đúng chữ bạn nhập.")

    st.markdown("#### Trợ lý kiểm tra thông tin trước khi viết")
    st.caption("Điền các thông tin của đúng mẫu đang bán. Trợ lý dùng chúng để viết phần giúp khách chọn mua.")
    demo_answers = {
        "material": "Kaki dày vừa, có lớp lót trong",
        "size_measurements": "S 40–45 kg, eo 62–64 cm, dài 55 cm. M 46–50 kg, eo 66–68 cm, dài 56 cm. L 51–55 kg, eo 70–72 cm, dài 57 cm.",
        "fit_feature": "Dáng chữ A, xếp ly trước",
        "construction": "Khóa kéo ẩn bên hông",
        "care": "Giặt tay nước lạnh, không sấy, ủi nhiệt độ thấp",
    }
    intake_answers = {}
    intake_columns = st.columns(2)
    for index, (field, label, placeholder) in enumerate(fields_for(category)):
        intake_answers[field] = intake_columns[index % 2].text_input(
            label, value=demo_answers.get(field, "") if demo else "", placeholder=placeholder,
            key=f"intake_{field}_{version}")
    missing = missing_fields(category, intake_answers)
    if missing:
        st.warning("Cần bổ sung trước khi viết: " + ", ".join(missing) + ".")

    facts = st.text_area("Dữ kiện thật bổ sung",
                         value=DEMO_FACTS if demo else "", height=110,
                         placeholder="S: 40-45kg, eo 62-64cm, dài 55cm; M: 46-50kg, eo 66-68cm · Kaki dày vừa, có lót · Khóa kéo ẩn bên hông · Giặt tay nước lạnh",
                         key=f"facts_{version}",
                         help="Thêm bảng thông số hoặc chi tiết đã kiểm tra của đúng mẫu hàng. Trợ lý bỏ qua lời quảng cáo và chính sách chưa được xác nhận.")
    raw_options = objections.get(category, []) if isinstance(objections, dict) else []
    # objections_by_category.json lưu dạng [{"phrase": chủ đề, "count": n, "share": tỷ lệ}].
    options = [item["phrase"] if isinstance(item, dict) else str(item) for item in raw_options] if isinstance(raw_options, list) else []
    selected_objections = st.multiselect("Khách hay lo điều gì? (lấy từ đánh giá 1 đến 3 sao thật của loại hàng này)",
                                         options, default=options[:2] if demo else [], key=f"objections_{version}",
                                         placeholder="Chọn một hoặc vài lo ngại",
                                         help="Kịch bản nhắn tin sẽ trả lời sẵn những lo ngại bạn chọn.")

    # Streamlit đóng lại phần mở rộng sau mỗi lần nhập, nên giữ nó mở khi người dùng đã điền gì đó.
    detail_keys = [f"{name}_{version}" for name in ("material", "color", "size", "price", "free_objections")]
    keep_open = demo or any(st.session_state.get(key) for key in detail_keys)
    with st.expander("Thuộc tính, giá và lo ngại khác (không bắt buộc)", expanded=keep_open):
        cols = st.columns(4)
        material = cols[0].text_input("Chất liệu", value="kaki" if demo else "", placeholder="cotton, lụa", key=f"material_{version}")
        color = cols[1].text_input("Màu sắc", value="be, đen" if demo else "", placeholder="đen, be", key=f"color_{version}")
        size = cols[2].text_input("Kích cỡ", value="S, M, L" if demo else "", placeholder="S, M, L", key=f"size_{version}")
        price = cols[3].text_input("Giá", value="259000" if demo else "", placeholder="259000", key=f"price_{version}")
        free_objections = st.text_input("Lo ngại khác, cách nhau bằng dấu phẩy", key=f"free_objections_{version}")

    keywords = list(dict.fromkeys(suggested + split_items(free_keywords))) or [category]
    names = {"vi": ("chất liệu", "màu sắc", "kích cỡ", "chi tiết người bán cung cấp"),
             "en": ("material", "color", "size", "seller details")}[lang]
    # Tiếng Việt: chi tiết thật đi vào phần "chi tiết người bán" của câu lệnh M3; tiếng Anh: vào thuộc tính như M2.
    values = (material, color, size, "" if lang == "vi" else facts)
    attributes = {name: value.strip() for name, value in zip(names, values) if value.strip()}
    brief = {"item_id": "demo", "category": category, "keywords": keywords, "attributes": attributes,
             "objections": selected_objections + split_items(free_objections)}
    digits = re.sub(r"\D", "", price or "")
    if digits:
        brief["price"] = int(digits)
    confirmed = confirmed_facts(category, intake_answers, facts.strip())
    auto_prompt = build_prompt(brief, lang, confirmed)

    prompt = auto_prompt

    col_run, col_reset = st.columns([3, 1])
    run = col_run.button("✍️ Viết nội dung", type="primary", width="stretch", disabled=bool(missing))
    col_reset.button("Làm lại", width="stretch", on_click=reset_all,
                     help="Xóa kết quả và đưa các ô về mặc định.")

    if demo and not st.session_state.get("demo_done"):
        run = True
        st.session_state["demo_done"] = True
    if run:
        count = st.session_state.get("generation_count", 0)
        if count >= MAX_RUNS:
            st.error(f"Phiên này đã dùng hết {MAX_RUNS} lượt. Tải lại trang để bắt đầu phiên mới.")
        else:
            st.session_state["generation_count"] = count + 1
            try:
                if not api_key:
                    raise RuntimeError("Dịch vụ chưa sẵn sàng.")
                with st.spinner("Đang viết, thường mất 20 giây đến hơn một phút…"):
                    text, _, _ = generate_via_api(prompt, provider, api_key, None)
                content = parse_response(text)
                # Kiểm tra và viết lại tối đa một lần khi bản nháp còn cụm không phù hợp.
                found = leaked_phrases(content) + (cliche_phrases(content) if lang == "vi" else [])
                if found:
                    with st.spinner("Đang hoàn thiện nội dung…"):
                        text, _, _ = generate_via_api(
                            prompt + REPAIR_NOTE[lang].format(found=", ".join(found)),
                            provider, api_key, None)
                    content = parse_response(text)
                    still = leaked_phrases(content) + (cliche_phrases(content) if lang == "vi" else [])
                    if still:
                        st.warning(f"Vui lòng sửa các cụm {', '.join(still)} trước khi đăng.")
                # Phần kiểm tra chi tiết coi chi tiết thật người bán dán vào là đầu vào hợp lệ.
                brief_check = dict(brief, attributes=dict(brief.get("attributes", {}), **({names[3]: confirmed} if confirmed else {})))
                st.session_state["generated_result"] = (content, brief_check, lang)
            except Exception:
                st.error("Chưa thể viết nội dung lúc này. Vui lòng thử lại sau hoặc liên hệ người quản lý trang.")
    if "generated_result" in st.session_state:
        st.divider()
        st.markdown("#### Nội dung cho bạn")
        show_content(*st.session_state["generated_result"])

with kiem_tra:
    st.caption("Dán nội dung bạn đã viết để xem còn thiếu tiêu chí nào. Phần này chạy ngay trên trang.")
    with st.form("scoring_form"):
        title = st.text_input("Tên sản phẩm")
        meta = st.text_area("Đoạn tóm tắt ngắn", height=70)
        description = st.text_area("Mô tả sản phẩm", height=150)
        subjects = st.text_area("Tiêu đề email, mỗi dòng một tiêu đề", height=90)
        script = st.text_area("Kịch bản nhắn tin", height=120)
        scoring_keywords = st.text_input("Từ khóa chính, cách nhau bằng dấu phẩy")
        facts = st.text_area("Dữ kiện sản phẩm có thật (chất liệu, màu, size, giá)", height=70,
                             help="Dùng để cảnh báo khi nội dung nhắc tới chi tiết không có thật.")
        score = st.form_submit_button("Kiểm tra", type="primary")
    if score:
        content = {"title": title, "meta_description": meta, "description": description,
                   "email_subjects": [line.strip() for line in subjects.splitlines() if line.strip()],
                   "sales_script": script}
        brief = {"keywords": split_items(scoring_keywords), "attributes": {"dữ kiện": facts}}
        show_review(content, brief, "vi", detailed=True)

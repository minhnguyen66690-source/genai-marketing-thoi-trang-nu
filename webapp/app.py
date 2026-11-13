"""Trợ lý viết nội dung bán hàng cho shop thời trang nữ.

Giao diện dành cho người bán: chỉ hỏi những gì cần để viết được nội dung, không hiển thị
thuật ngữ nghiên cứu. Phần số liệu của bài báo nằm trong tài liệu nộp kèm, không đưa lên đây.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_api import generate_via_api
from metrics_content import detect_lang, email_checks, factual_support, script_rule_checks, seo_checks
from prompts import engineered, template_output, zero_shot

DATA = ROOT / "webapp" / "data"
MAX_RUNS = 20
NOI_BO = "Quy tắc do nhóm nghiên cứu tự đặt"

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
    "title_len_50_60": ("50–60 ký tự", "S19", "title_chars", "Tên sản phẩm dài 50–60 ký tự để hiện đủ trên Google"),
    "title_tiki_15_50": ("15–50 ký tự", "S23", "title_chars", "Tên sản phẩm nằm trong khoảng Tiki khuyến nghị"),
    "title_tiki_max_225": ("≤225 ký tự", "S23", "title_chars", "Tên sản phẩm không vượt giới hạn của sàn"),
    "meta_len_120_156": ("120–156 ký tự", "S12", "meta_chars", "Đoạn tóm tắt hiện dưới tên trên Google dài vừa đủ"),
    "keyphrase_in_title": ("Có", "S11", None, "Từ khóa chính có trong tên sản phẩm"),
    "keyphrase_in_first_paragraph": ("Có", "S10", None, "Từ khóa chính xuất hiện ngay đoạn mở đầu"),
    "keyphrase_in_meta": ("Có", "S11", None, "Từ khóa chính có trong đoạn tóm tắt"),
    "density_proxy_0_5_3": ("0,5–3%", "S09", "keyphrase_density_proxy", "Mật độ từ khóa trong bài (chỉ tham khảo)"),
    "desc_words_ge_200": ("≥200 từ", "S13", "description_words", "Mô tả dài từ 200 từ trở lên"),
}
EMAIL = {
    "mailchimp_le9w_le60c": ("≤9 từ và ≤60 ký tự", "E01", "Tiêu đề email ngắn gọn, không bị cắt"),
    "omnisend_4_6w_or_21_40c": ("4–6 từ hoặc 21–40 ký tự", "E05", "Tiêu đề email trong khoảng được mở nhiều nhất"),
    "punct_le3": ("≤3 dấu câu", "E02", "Không lạm dụng dấu câu"),
    "emoji_le1": ("≤1 emoji", "E03", "Nhiều nhất một emoji"),
    "no_fake_re_fwd": ("Không có Re:/Fwd:", "E16", "Không giả mạo thư trả lời"),
    "keyword_relevance": ("Có từ khóa", None, "Tiêu đề email nhắc đến sản phẩm"),
    "offer_supported": ("Chỉ ưu đãi bạn đã nhập", None, "Không hứa ưu đãi mà bạn chưa nhập"),
}

HUONG_DAN_KHOA = """
Trang cần một khóa API để nhờ mô hình ngôn ngữ viết nội dung.

**Khóa Gemini của Google — miễn phí, không cần thẻ:**
1. Mở [aistudio.google.com/apikey](https://aistudio.google.com/apikey), đăng nhập tài khoản Google.
2. Bấm **Create API key**, chọn một dự án bất kỳ.
3. Sao chép khóa (bắt đầu bằng `AIza`) rồi dán vào ô bên dưới.

**Khóa OpenAI:** lấy tại [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
Lưu ý gói ChatGPT Plus hay Business không bao gồm khóa API, phải nạp tiền riêng ở mục Billing.

Khóa bạn dán chỉ nằm trong phiên của trình duyệt, đóng tab là mất.
"""


def load(name, fallback):
    """Đọc dữ liệu tùy chọn."""
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def source_text(code):
    """Ghi nguồn ngắn gọn: tổ chức, năm nếu trang có ghi, kèm địa chỉ."""
    if not code:
        return NOI_BO
    name, year, url = REFS[code]
    return f"{name} ({year}) · {url}" if year else f"{name} · {url}"


def split_items(text):
    """Tách các mục do người dùng nhập."""
    return [part.strip() for part in text.split(",") if part.strip()]


def secret_of(name):
    """Đọc khóa do chủ trang cấu hình, bỏ qua khi chưa đặt."""
    try:
        return st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        return ""


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
        rows.append({"Tiêu chí": friendly, "Đo được": str(value) if value is not None else "—",
                     "Khuyến nghị": threshold, "Đạt?": "✅" if flag is True else "❌" if flag is False else "—",
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
            flag = None if (mixed and key == "keyword_relevance") else email["checks"][key]
            rows.append({"Tiêu chí": f"Email {index}: {friendly}",
                         "Đo được": str(values.get(key, flag)) if flag is not None else "—",
                         "Khuyến nghị": threshold, "Đạt?": "✅" if flag is True else "❌" if flag is False else "—",
                         "Nguồn khuyến nghị": source_text(ref)})
            if isinstance(flag, bool):
                (email_pass if flag else email_fail).append(f"Email {index}: {friendly}")
    script = script_rule_checks(content.get("sales_script", ""), brief, brief.get("objections", []))
    cta = bool(script["cta_present"])
    rows.append({"Tiêu chí": "Kịch bản có câu mời khách chốt đơn", "Đo được": "Có" if cta else "Không",
                 "Khuyến nghị": "Có", "Đạt?": "✅" if cta else "❌", "Nguồn khuyến nghị": NOI_BO})
    (passed if cta else failed).append("Kịch bản có câu mời khách chốt đơn")
    coverage = script["objection_keyword_coverage"]
    if coverage is not None:
        ok = coverage == 1
        rows.append({"Tiêu chí": "Kịch bản trả lời các lo ngại bạn đã chọn", "Đo được": f"{100 * coverage:.0f}%",
                     "Khuyến nghị": "100%", "Đạt?": "✅" if ok else "❌", "Nguồn khuyến nghị": NOI_BO})
        (passed if ok else failed).append("Kịch bản trả lời các lo ngại bạn đã chọn")
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
        st.warning("Các chi tiết sau có trong bài nhưng bạn chưa nhập, hãy kiểm tra lại cho đúng sản phẩm: "
                   + ", ".join(summary["unsupported"]))
    if summary.get("mixed"):
        st.info("Từ khóa bạn nhập khác ngôn ngữ với nội dung được viết, nên phần kiểm tra từ khóa và kiểm tra "
                "chi tiết sản phẩm được bỏ qua. Nhập từ khóa cùng ngôn ngữ với nội dung để chấm được đầy đủ.")
    with st.expander("Xem từng tiêu chí và nguồn khuyến nghị", expanded=detailed):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Các ngưỡng là khuyến nghị đã công bố của Ahrefs, Yoast, Tiki Academy, Mailchimp, Omnisend và "
                   f"Google; những dòng ghi “{NOI_BO}” là quy tắc tác giả đặt ra cho nghiên cứu này, không phải "
                   "khuyến nghị của bên ngoài. Đạt các ngưỡng này không bảo đảm bán được nhiều hơn.")
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
    benchmark = load("category_benchmarks.json", {}).get(brief.get("category"), {}).get("1")
    if benchmark:
        with st.expander("So với sản phẩm bán chạy cùng loại"):
            st.dataframe(pd.DataFrame([
                {"Chỉ số": "Độ dài mô tả (từ)", "Nội dung vừa tạo": seo["description_words"],
                 "Trung vị nhóm bán chạy": benchmark["desc_len_words"]},
                {"Chỉ số": "Số gạch đầu dòng", "Nội dung vừa tạo": seo["n_bullets"],
                 "Trung vị nhóm bán chạy": benchmark["n_bullets"]},
            ]), hide_index=True)
            st.caption(f"Trung vị của {benchmark['n']} sản phẩm bán chạy cùng loại trên Tiki. "
                       "Đây là con số quan sát được, không bảo đảm bán chạy hơn.")
    st.download_button("Tải toàn bộ nội dung (JSON)", json.dumps(content, ensure_ascii=False, indent=2),
                       "noi-dung.json", "application/json")


def reset_all():
    """Xóa kết quả và đưa biểu mẫu về mặc định."""
    for key in list(st.session_state.keys()):
        if key != "generation_count":
            del st.session_state[key]


def clear_keywords():
    """Đổi loại sản phẩm thì bỏ các từ khóa gợi ý của loại cũ, vì danh sách gợi ý khác nhau."""
    st.session_state.pop("suggested", None)


st.set_page_config(page_title="Trợ lý nội dung sản phẩm", page_icon="✍️", layout="centered")
st.title("Trợ lý nội dung sản phẩm")

owner_gemini, owner_openai = secret_of("GEMINI_API_KEY"), secret_of("OPENAI_API_KEY")
has_owner_key = bool(owner_gemini or owner_openai)
with st.sidebar:
    if has_owner_key:
        st.success("Trang đã sẵn sàng, bạn dùng được ngay.")
    else:
        st.warning("Trang chưa có khóa API nên chỉ ghép được câu mẫu.")
    with st.expander("Dùng khóa API riêng", expanded=not has_owner_key):
        st.markdown(HUONG_DAN_KHOA)
        providers = ["Gemini (Google)", "ChatGPT (OpenAI)"]
        provider_label = st.radio("Dịch vụ", providers, index=1 if (owner_openai and not owner_gemini) else 0)
        provider = "gemini" if provider_label.startswith("Gemini") else "openai"
        user_key = st.text_input("Dán khóa API vào đây", type="password", placeholder="AIza… hoặc sk-…")
        model = st.text_input("Tên mô hình (để trống là được)",
                              help="Để trống thì trang tự chọn mô hình mới nhất mà khóa của bạn truy cập được.")
    owner_key = owner_gemini if provider == "gemini" else owner_openai
    api_key = (user_key or "").strip() or owner_key
    st.caption(f"Đã dùng {st.session_state.get('generation_count', 0)}/{MAX_RUNS} lượt trong phiên này.")

viet, kiem_tra = st.tabs(["✍️ Viết nội dung", "🔍 Kiểm tra nội dung có sẵn"])

with viet:
    categories = load("categories.json", {})
    objections = load("objections_by_category.json", {})
    demo = st.query_params.get("demo") == "1"
    category_list = list(categories) or ["Khác"]
    demo_category = "Chân váy" if "Chân váy" in category_list else category_list[0]

    st.markdown("#### Sản phẩm của bạn")
    category = st.selectbox("Loại sản phẩm", category_list,
                            index=category_list.index(demo_category) if demo else 0, key="category",
                            on_change=clear_keywords)
    suggested = st.multiselect("Từ khóa khách hay tìm", categories.get(category, []),
                               placeholder="Chọn vài từ, ví dụ: chân váy chữ a, công sở", key="suggested")
    free_keywords = st.text_input("Từ khóa khác, cách nhau bằng dấu phẩy",
                                  value="chân váy chữ a, công sở, xếp ly" if demo else "",
                                  placeholder="váy dự tiệc, vải lụa, dáng suông", key="free_keywords")
    language = st.radio("Viết bằng", ["Tiếng Việt", "English"], horizontal=True, key="language")
    lang = "vi" if language == "Tiếng Việt" else "en"
    if lang == "en":
        st.caption("Chọn English thì nên nhập từ khóa và chi tiết sản phẩm bằng tiếng Anh, "
                   "vì phần kiểm tra so khớp đúng chữ bạn nhập.")

    with st.expander("Thêm chi tiết sản phẩm (không bắt buộc, nhập càng đúng bài càng sát)"):
        cols = st.columns(4)
        material = cols[0].text_input("Chất liệu", value="kaki" if demo else "", placeholder="cotton, lụa", key="material")
        color = cols[1].text_input("Màu sắc", placeholder="đen, be", key="color")
        size = cols[2].text_input("Kích cỡ", placeholder="S, M, L", key="size")
        price = cols[3].text_input("Giá", placeholder="259000", key="price")
        raw_options = objections.get(category, []) if isinstance(objections, dict) else []
        # objections_by_category.json lưu dạng [{"phrase": chủ đề, "count": n, "share": tỷ lệ}].
        options = [item["phrase"] if isinstance(item, dict) else str(item) for item in raw_options] if isinstance(raw_options, list) else []
        selected_objections = st.multiselect("Lo ngại khách hay nêu (lấy từ đánh giá thật của loại sản phẩm này)",
                                             options, default=options[:2] if demo else [], key="objections",
                                             help="Kịch bản nhắn tin sẽ trả lời sẵn những lo ngại bạn chọn.")
        free_objections = st.text_input("Lo ngại khác, cách nhau bằng dấu phẩy", key="free_objections")

    col_run, col_reset = st.columns([3, 1])
    run = col_run.button("✍️ Viết nội dung", type="primary", use_container_width=True)
    col_reset.button("Làm lại", use_container_width=True, on_click=reset_all,
                     help="Xóa kết quả và đưa các ô về mặc định.")

    if not api_key:
        st.info("Chưa có khóa API nên trang sẽ ghép câu mẫu. Thêm khóa ở thanh bên trái để mô hình viết tự nhiên hơn.")
    if demo and not st.session_state.get("demo_done"):
        run = True
        st.session_state["demo_done"] = True
    if run:
        count = st.session_state.get("generation_count", 0)
        if count >= MAX_RUNS:
            st.error(f"Phiên này đã dùng hết {MAX_RUNS} lượt. Tải lại trang để bắt đầu phiên mới.")
        else:
            keywords = list(dict.fromkeys(suggested + split_items(free_keywords))) or [category]
            names = {"vi": ("chất liệu", "màu sắc", "kích cỡ", "giá"), "en": ("material", "color", "size", "price")}[lang]
            attributes = {name: value.strip() for name, value in zip(names, (material, color, size, price)) if value.strip()}
            brief = {"item_id": "demo", "category": category, "keywords": keywords, "attributes": attributes,
                     "objections": selected_objections + split_items(free_objections)}
            st.session_state["generation_count"] = count + 1
            try:
                if not api_key:
                    content = template_output(brief, lang)
                else:
                    prompt = engineered([brief], lang)
                    # Bản tóm tắt có thể viết bằng tiếng Việt nhưng người dùng chọn đầu ra tiếng Anh,
                    # nên phải nêu rõ ngôn ngữ đầu ra; dòng này chỉ dùng cho website, không dùng khi chạy thực nghiệm.
                    prompt += ("\n\nOUTPUT LANGUAGE: write every field of the JSON in English, even if the brief is "
                               "written in Vietnamese. Translate Vietnamese category names and keywords into natural English."
                               if lang == "en" else
                               "\n\nNGÔN NGỮ ĐẦU RA: viết toàn bộ các trường trong JSON bằng tiếng Việt có dấu.")
                    with st.spinner("Đang viết, khoảng 10–20 giây…"):
                        text, used_model, latency = generate_via_api(prompt, provider, api_key, model.strip() or None)
                    content = parse_response(text)
                    st.caption(f"Viết bằng {used_model} trong {latency:.1f} giây.")
                st.session_state["generated_result"] = (content, brief, lang)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                st.error(f"Chưa viết được: {exc}")
                st.caption("Hay gặp nhất là khóa sai, khóa đã bị xóa, hoặc tài khoản chưa nạp tiền. "
                           "Bạn vẫn xem thử được giao diện: xóa khóa đi thì trang sẽ ghép câu mẫu.")
    if "generated_result" in st.session_state:
        st.divider()
        st.markdown("#### Nội dung cho bạn")
        show_content(*st.session_state["generated_result"])

with kiem_tra:
    st.caption("Dán nội dung bạn đã viết để xem còn thiếu tiêu chí nào. Phần này chạy ngay trên trang, không cần khóa API.")
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

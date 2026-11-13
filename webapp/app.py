"""Trợ lý viết nội dung bán hàng cho shop thời trang nữ.

Giao diện ưu tiên người bán: màn hình chính chỉ hỏi những gì cần để viết được nội dung,
mọi chi tiết kỹ thuật (tên cấu hình, bảng tiêu chí đầy đủ, số liệu nghiên cứu) nằm trong
phần mở rộng hoặc tab riêng.
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
from metrics_content import email_checks, factual_support, script_rule_checks, seo_checks
from prompts import engineered, template_output, zero_shot

DATA = ROOT / "webapp" / "data"
ASSETS = ROOT / "webapp" / "assets"
MAX_RUNS = 15

REFS = {
    "S19": ("Ahrefs", "2024", "https://ahrefs.com/blog/title-tag-seo/"),
    "S23": ("Tiki Academy", "KHÔNG RÕ", "https://hocvien.tiki.vn/faq/quy-chuan-ve-ten-san-pham/"),
    "S12": ("Yoast", "2025", "https://yoast.com/meta-descriptions/"),
    "S11": ("Yoast", "KHÔNG RÕ", "https://yoast.com/help/yoast-seo-for-shopify-editor/"),
    "S10": ("Yoast", "KHÔNG RÕ", "https://yoast.com/help/yoast-seo-for-shopify-editor/"),
    "S09": ("Yoast", "KHÔNG RÕ", "https://yoast.com/features/keyphrase-density/"),
    "S13": ("Yoast", "2025", "https://yoast.com/blog-post-word-count-seo/"),
    "S24": ("Tiki Academy", "KHÔNG RÕ", "https://hocvien.tiki.vn/faq/quy-chuan-ve-noi-dung-dang-san-pham-cho-tung-nganh-hang/"),
    "E01": ("Mailchimp", "KHÔNG RÕ", "https://mailchimp.com/help/best-practices-for-email-subject-lines/"),
    "E02": ("Mailchimp", "KHÔNG RÕ", "https://mailchimp.com/help/best-practices-for-email-subject-lines/"),
    "E03": ("Mailchimp", "KHÔNG RÕ", "https://mailchimp.com/help/best-practices-for-email-subject-lines/"),
    "E05": ("Omnisend", "2022", "https://www.omnisend.com/blog/bfcm-email-analysis-less-is-more/"),
    "E16": ("Google Gmail", "KHÔNG RÕ", "https://support.google.com/mail/answer/81126?hl=en"),
}
# key: (nhãn kỹ thuật, ngưỡng, mã nguồn, trường giá trị đo, câu giải thích cho người bán)
SEO = {
    "title_len_50_60": ("Độ dài tiêu đề", "50–60 ký tự", "S19", "title_chars",
                        "Tên sản phẩm dài 50–60 ký tự thì hiện đủ trên trang kết quả của Google"),
    "title_tiki_15_50": ("Tiêu đề Tiki tối ưu", "15–50 ký tự", "S23", "title_chars",
                         "Tên sản phẩm nằm trong khoảng Tiki khuyến nghị"),
    "title_tiki_max_225": ("Giới hạn tiêu đề Tiki", "≤225 ký tự", "S23", "title_chars",
                           "Tên sản phẩm không vượt giới hạn ký tự của sàn"),
    "meta_len_120_156": ("Độ dài meta", "120–156 ký tự", "S12", "meta_chars",
                         "Đoạn tóm tắt hiện dưới tên sản phẩm trên Google dài vừa đủ"),
    "keyphrase_in_title": ("Từ khóa trong tiêu đề", "Có", "S11", None,
                           "Từ khóa chính có trong tên sản phẩm"),
    "keyphrase_in_first_paragraph": ("Từ khóa trong đoạn đầu", "Có", "S10", None,
                                     "Từ khóa chính xuất hiện ngay đoạn mở đầu"),
    "keyphrase_in_meta": ("Từ khóa trong meta", "Có", "S11", None,
                          "Từ khóa chính có trong đoạn tóm tắt"),
    "density_proxy_0_5_3": ("Mật độ khớp chính xác (proxy)", "Chưa xác lập cho proxy", "S09", "keyphrase_density_proxy",
                            "Mật độ từ khóa trong bài (chỉ để tham khảo)"),
    "desc_words_ge_200": ("Độ dài mô tả", "≥200 từ cách nhau bằng khoảng trắng", "S13", "description_words",
                          "Mô tả dài từ 200 từ trở lên"),
}
EMAIL = {
    "mailchimp_le9w_le60c": ("Độ dài Mailchimp", "≤9 từ và ≤60 ký tự", "E01", "Tiêu đề email ngắn gọn, không bị cắt"),
    "omnisend_4_6w_or_21_40c": ("Độ dài Omnisend", "4–6 từ hoặc 21–40 ký tự", "E05", "Tiêu đề email trong khoảng được mở nhiều nhất"),
    "punct_le3": ("Dấu câu", "≤3", "E02", "Không lạm dụng dấu câu"),
    "emoji_le1": ("Emoji", "≤1", "E03", "Nhiều nhất một emoji"),
    "no_fake_re_fwd": ("Tiền tố Re/Fwd", "Không dùng tiền tố giả", "E16", "Không giả mạo thư trả lời"),
    "keyword_relevance": ("Từ khóa", "Có từ khóa", None, "Tiêu đề email nhắc đến sản phẩm"),
    "offer_supported": ("Ưu đãi", "Có căn cứ trong brief", None, "Không hứa ưu đãi mà bạn chưa nhập"),
}

HUONG_DAN_KHOA = """
**Khóa API là gì?** Là một dãy ký tự để trang này gọi sang mô hình ngôn ngữ nhờ viết nội dung.
Không có khóa thì trang vẫn chạy nhưng chỉ ghép câu mẫu, chữ khô và ít thuyết phục hơn.

**Cách lấy khóa của OpenAI (dùng cho ChatGPT):**
1. Mở [platform.openai.com/api-keys](https://platform.openai.com/api-keys), đăng nhập bằng chính
   tài khoản ChatGPT của bạn.
2. Vào [Billing](https://platform.openai.com/settings/organization/billing/overview) nạp tiền trước,
   tối thiểu 5 USD. Gói ChatGPT Plus hay Business **không** bao gồm khóa API, đây là hai khoản tính
   tiền tách rời nhau.
3. Quay lại trang API keys, bấm **Create new secret key**, đặt tên bất kỳ rồi sao chép khóa
   (bắt đầu bằng `sk-`). Khóa chỉ hiện một lần, đóng cửa sổ là không xem lại được.
4. Dán vào ô bên dưới. Trang tự chọn mô hình phù hợp nên bạn không cần điền tên mô hình.

**Cách lấy khóa miễn phí của Google (nếu không muốn trả tiền, khoảng 2 phút):**
1. Mở [aistudio.google.com/apikey](https://aistudio.google.com/apikey), đăng nhập tài khoản Google.
2. Bấm **Create API key**, chọn một dự án bất kỳ.
3. Sao chép khóa (bắt đầu bằng `AIza`) rồi dán vào ô bên dưới, nhớ chọn dịch vụ Gemini.

Khóa bạn dán ở đây chỉ nằm trong phiên làm việc của trình duyệt, đóng tab là mất, không được lưu lại.
"""


def load(name, fallback):
    """Đọc dữ liệu tùy chọn."""
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def source(code):
    """Định dạng nguồn kiểm chứng hoặc ghi rõ quy tắc nội bộ."""
    if not code:
        return "Quy tắc đối chiếu nội bộ"
    name, year, url = REFS[code]
    return f"{code} · {name} · {year} · {url}"


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
    """Tính mọi tiêu chí và trả về (danh sách dòng chi tiết, tóm tắt theo nhóm, kết quả seo)."""
    keywords = brief.get("keywords", [])
    seo = seo_checks(content.get("title", ""), content.get("meta_description", ""), content.get("description", ""),
                     keywords[0] if keywords else "", keywords, lang)
    rows, passed, failed = [], [], []
    for key, (label, threshold, ref, measure, friendly) in SEO.items():
        flag = None if key == "density_proxy_0_5_3" else seo["checks"][key]
        value = seo.get(measure) if measure else flag
        if isinstance(value, float):
            value = round(value, 2)
        rows.append({"Tiêu chí": label, "Giá trị đo": str(value) if value is not None else "—", "Ngưỡng": threshold,
                     "Đạt?": "✅" if flag is True else "❌" if flag is False else "—", "Nguồn": source(ref)})
        # Khoảng tên Tiki 15–50 ký tự mâu thuẫn với khoảng 50–60 của Ahrefs nên chỉ hiển thị, không tính vào tỷ lệ.
        if isinstance(flag, bool) and key != "title_tiki_15_50":
            (passed if flag else failed).append(friendly)
    email_rows_pass, email_rows_fail = [], []
    for index, subject in enumerate(content.get("email_subjects", []), 1):
        email = email_checks(subject, keywords, brief)
        values = {"mailchimp_le9w_le60c": f"{email['words_space']} từ; {email['chars']} ký tự",
                  "omnisend_4_6w_or_21_40c": f"{email['words_space']} từ; {email['chars']} ký tự",
                  "punct_le3": email["punctuation"], "emoji_le1": email["emoji"]}
        for key, (label, threshold, ref, friendly) in EMAIL.items():
            flag = email["checks"][key]
            rows.append({"Tiêu chí": f"Email {index}: {label}",
                         "Giá trị đo": str(values.get(key, flag)) if flag is not None else "—", "Ngưỡng": threshold,
                         "Đạt?": "✅" if flag is True else "❌" if flag is False else "—", "Nguồn": source(ref)})
            if isinstance(flag, bool):
                (email_rows_pass if flag else email_rows_fail).append(f"Email {index}: {friendly}")
    script = script_rule_checks(content.get("sales_script", ""), brief, brief.get("objections", []))
    cta = bool(script["cta_present"])
    rows.append({"Tiêu chí": "Kịch bản: lời kêu gọi hành động", "Giá trị đo": str(cta),
                 "Ngưỡng": "Có cụm kêu gọi hành động theo từ điển", "Đạt?": "✅" if cta else "❌",
                 "Nguồn": "Quy tắc đối chiếu nội bộ"})
    (passed if cta else failed).append("Kịch bản có câu mời khách chốt đơn")
    coverage = script["objection_keyword_coverage"]
    if coverage is not None:
        ok = coverage == 1
        rows.append({"Tiêu chí": "Kịch bản: nhắc đến chủ đề phản đối", "Giá trị đo": f"{100 * coverage:.0f}%",
                     "Ngưỡng": "100% chủ đề trong bản tóm tắt (đối chiếu từ vựng)", "Đạt?": "✅" if ok else "❌",
                     "Nguồn": "Quy tắc đối chiếu nội bộ"})
        (passed if ok else failed).append("Kịch bản trả lời các lo ngại bạn đã chọn")
    all_text = " ".join([content.get("title", ""), content.get("meta_description", ""), content.get("description", ""),
                         content.get("sales_script", "")] + content.get("email_subjects", []))
    unsupported = factual_support(all_text, brief)["unsupported_list"]
    summary = {"passed": passed, "failed": failed,
               "email_passed": email_rows_pass, "email_failed": email_rows_fail, "unsupported": unsupported}
    return rows, summary, seo


def show_review(content, brief, lang, detailed=False):
    """Hiển thị kết quả kiểm tra: tóm tắt dễ hiểu trước, bảng đầy đủ trong phần mở rộng."""
    rows, summary, seo = collect_checks(content, brief, lang)
    n_pass = len(summary["passed"])
    n_total = n_pass + len(summary["failed"])
    n_email_pass, n_email_total = len(summary["email_passed"]), len(summary["email_passed"]) + len(summary["email_failed"])
    left, right = st.columns(2)
    left.metric("Nội dung trang sản phẩm", f"{n_pass}/{n_total} tiêu chí đạt")
    if n_email_total:
        right.metric("Tiêu đề email", f"{n_email_pass}/{n_email_total} tiêu chí đạt")
    if summary["failed"]:
        st.markdown("**Nên sửa trước khi đăng:**")
        for item in summary["failed"]:
            st.markdown(f"- {item}")
    else:
        st.success("Nội dung đạt toàn bộ tiêu chí kiểm tra được.")
    if summary["unsupported"]:
        st.warning("Những chi tiết sau xuất hiện trong bài nhưng bạn chưa nhập, hãy kiểm tra lại cho đúng sản phẩm: "
                   + ", ".join(summary["unsupported"]))
    with st.expander("Xem đầy đủ từng tiêu chí và nguồn khuyến nghị", expanded=detailed):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Các ngưỡng là khuyến nghị đã công bố của Ahrefs, Yoast, Tiki Academy, Mailchimp, Omnisend và Google, "
                   "không bảo đảm hiệu quả bán hàng. Mật độ từ khóa chỉ để tham khảo, không tính vào tỷ lệ đạt.")
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
    st.markdown("### Nội dung cho bạn")
    st.markdown("**Tên sản phẩm**")
    st.code(content.get("title", ""), language=None, wrap_lines=True)
    st.markdown("**Đoạn tóm tắt ngắn** (hiện dưới tên sản phẩm trên Google)")
    st.code(content.get("meta_description", ""), language=None, wrap_lines=True)
    st.markdown("**Mô tả sản phẩm**")
    st.code(content.get("description", ""), language=None, wrap_lines=True)
    st.markdown("**Kịch bản nhắn tin chốt đơn**")
    st.code(content.get("sales_script", ""), language=None, wrap_lines=True)
    subjects = content.get("email_subjects", [])
    if subjects:
        st.markdown("**Tiêu đề email giới thiệu**")
        st.code("\n".join(subjects), language=None, wrap_lines=True)
    st.caption("Bấm biểu tượng sao chép ở góc phải mỗi khung để dán sang sàn hoặc trình soạn email.")
    st.divider()
    st.markdown("### Bản nháp này đã đạt chuẩn chưa?")
    seo = show_review(content, brief, lang)
    benchmark = load("category_benchmarks.json", {}).get(brief.get("category"), {}).get("1")
    if benchmark:
        with st.expander("So với sản phẩm bán chạy cùng danh mục"):
            st.dataframe(pd.DataFrame([
                {"Chỉ số": "Độ dài mô tả (từ)", "Nội dung vừa tạo": seo["description_words"], "Trung vị nhóm bán chạy": benchmark["desc_len_words"]},
                {"Chỉ số": "Số gạch đầu dòng", "Nội dung vừa tạo": seo["n_bullets"], "Trung vị nhóm bán chạy": benchmark["n_bullets"]},
            ]), hide_index=True)
            st.caption(f"Cỡ mẫu nhóm bán chạy: {benchmark['n']} sản phẩm. Nguồn: dữ liệu Tiki thu thập trong nghiên cứu. "
                       "Đây là liên hệ quan sát, không bảo đảm bán chạy hơn.")
    st.download_button("Tải toàn bộ nội dung (JSON)", json.dumps(content, ensure_ascii=False, indent=2),
                       "noi-dung.json", "application/json")


st.set_page_config(page_title="Trợ lý nội dung sản phẩm", page_icon="✍️", layout="wide")
st.title("Trợ lý nội dung sản phẩm")
st.caption("Nhập vài từ khóa, nhận tên sản phẩm, mô tả, kịch bản nhắn tin và tiêu đề email, kèm bảng kiểm tra "
           "theo khuyến nghị đã công bố. Dành cho shop thời trang nữ bán trên sàn.")

owner_gemini, owner_openai = secret_of("GEMINI_API_KEY"), secret_of("OPENAI_API_KEY")
with st.sidebar:
    st.subheader("Kết nối")
    if owner_gemini or owner_openai:
        st.success("Trang đã có sẵn khóa, bạn dùng được ngay.")
    else:
        st.info("Trang chưa có khóa chung. Bạn có thể dùng khóa riêng, hoặc dùng chế độ không cần khóa ở dưới.")
    with st.expander("Dùng khóa API riêng", expanded=not (owner_gemini or owner_openai)):
        st.markdown(HUONG_DAN_KHOA)
        # Mặc định chọn đúng dịch vụ mà chủ trang đã cấu hình khóa, để người dùng không phải tự dò.
        providers = ["Gemini (Google)", "ChatGPT (OpenAI)"]
        default_index = 1 if (owner_openai and not owner_gemini) else 0
        provider_label = st.radio("Dịch vụ", providers, index=default_index)
        provider = "gemini" if provider_label.startswith("Gemini") else "openai"
        user_key = st.text_input("Dán khóa API vào đây", type="password",
                                 placeholder="AIza… hoặc sk-…", help="Chỉ giữ trong phiên này, không lưu lại.")
        model = st.text_input("Tên mô hình (để trống là được)",
                              help="Để trống thì trang tự chọn mô hình mới nhất mà khóa của bạn truy cập được.")
    owner_key = owner_gemini if provider == "gemini" else owner_openai
    api_key = (user_key or "").strip() or owner_key
    language = st.selectbox("Ngôn ngữ nội dung", ["Tiếng Việt", "English"])
    lang = "vi" if language == "Tiếng Việt" else "en"
    used = st.session_state.get("generation_count", 0)
    st.caption(f"Đã dùng {used}/{MAX_RUNS} lượt trong phiên này.")

generation, scoring, results, methodology = st.tabs(
    ["✍️ Viết nội dung", "🔍 Kiểm tra nội dung có sẵn", "📊 Kết quả nghiên cứu", "📚 Phương pháp & nguồn"])

with generation:
    categories = load("categories.json", {})
    objections = load("objections_by_category.json", {})
    # Chế độ minh họa (?demo=1): tự điền một ví dụ thật để xem nhanh đầu ra và bảng kiểm tra.
    demo = st.query_params.get("demo") == "1"
    category_list = list(categories) or ["Khác"]
    demo_category = "Chân váy" if "Chân váy" in category_list else category_list[0]

    st.markdown("#### 1. Sản phẩm của bạn")
    col_cat, col_kw = st.columns([1, 2])
    category = col_cat.selectbox("Loại sản phẩm", category_list,
                                 index=category_list.index(demo_category) if demo else 0)
    suggested = col_kw.multiselect("Từ khóa mô tả sản phẩm", categories.get(category, []),
                                   placeholder="Chọn vài từ khách hay tìm, ví dụ: chân váy chữ a, công sở")
    free_keywords = st.text_input("Từ khóa khác (cách nhau bằng dấu phẩy)",
                                  value="chân váy chữ a, công sở, xếp ly" if demo else "",
                                  placeholder="váy dự tiệc, vải lụa, dáng suông")

    with st.expander("2. Thêm chi tiết sản phẩm (không bắt buộc, nhập càng đúng bài viết càng sát)"):
        cols = st.columns(4)
        material = cols[0].text_input("Chất liệu", value="kaki" if demo else "", placeholder="cotton, lụa…")
        color = cols[1].text_input("Màu sắc", placeholder="đen, be, xanh navy")
        size = cols[2].text_input("Kích cỡ", placeholder="S, M, L")
        price = cols[3].text_input("Giá", placeholder="259000")
        raw_options = objections.get(category, []) if isinstance(objections, dict) else []
        # objections_by_category.json lưu dạng [{"phrase": chủ đề, "count": n, "share": tỷ lệ}] (trích từ đánh giá 1–3 sao).
        options = [item["phrase"] if isinstance(item, dict) else str(item) for item in raw_options] if isinstance(raw_options, list) else []
        selected_objections = st.multiselect(
            "Lo ngại khách hay nêu (lấy từ đánh giá thật của loại sản phẩm này)", options,
            default=options[:2] if demo else [],
            help="Kịch bản nhắn tin sẽ trả lời sẵn những lo ngại bạn chọn.")
        free_objections = st.text_input("Lo ngại khác (cách nhau bằng dấu phẩy)")

    with st.expander("Tùy chọn nâng cao (dành cho người chấm bài)"):
        method_label = st.radio(
            "Cách viết", ["Câu lệnh có ràng buộc chuẩn (M2, khuyến nghị)", "Câu lệnh tối giản (M1)", "Ghép câu mẫu, không cần khóa (M0)"],
            help="Ba cấu hình được so sánh trong bài báo. M2 là cấu hình cho kết quả tốt nhất.")
        method = "M2" if method_label.startswith("Câu lệnh có") else "M1" if method_label.startswith("Câu lệnh tối") else "M0"

    if method != "M0" and not api_key:
        st.info("Chưa có khóa API nên trang sẽ ghép câu mẫu. Thêm khóa ở thanh bên trái để mô hình viết tự nhiên hơn.")
    run = st.button("✍️ Viết nội dung", type="primary", use_container_width=True)

    if demo and not st.session_state.get("demo_done"):
        run = True
        st.session_state["demo_done"] = True
    if run:
        count = st.session_state.get("generation_count", 0)
        if count >= MAX_RUNS:
            st.error(f"Phiên này đã dùng hết {MAX_RUNS} lượt. Hãy tải lại trang để bắt đầu phiên mới.")
        else:
            keywords = list(dict.fromkeys(suggested + split_items(free_keywords))) or [category]
            attributes = {key: value for key, value in
                          {"chất liệu": material, "màu sắc": color, "kích cỡ": size, "giá": price}.items() if value.strip()}
            brief = {"item_id": "demo", "category": category, "keywords": keywords, "attributes": attributes,
                     "objections": selected_objections + split_items(free_objections)}
            st.session_state["generation_count"] = count + 1
            try:
                if method == "M0" or not api_key:
                    content = template_output(brief, lang)
                else:
                    prompt = zero_shot([brief], lang) if method == "M1" else engineered([brief], lang)
                    with st.spinner("Đang viết, khoảng 10–15 giây…"):
                        text, used_model, latency = generate_via_api(prompt, provider, api_key, model.strip() or None)
                    content = parse_response(text)
                    st.caption(f"Viết bằng {used_model} trong {latency:.1f} giây.")
                st.session_state["generated_result"] = (content, brief, lang)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                st.error(f"Chưa viết được: {exc}")
                st.caption("Hay gặp nhất là khóa sai, khóa đã bị xóa, hoặc tài khoản OpenAI chưa nạp tiền "
                           "(gói ChatGPT trả phí không bao gồm khóa API). Trong lúc chờ, bạn chọn cách viết "
                           "'Ghép câu mẫu' ở phần nâng cao để xem thử giao diện.")
    if "generated_result" in st.session_state:
        st.divider()
        show_content(*st.session_state["generated_result"])

with scoring:
    st.caption("Dán nội dung bạn đã viết vào đây để xem còn thiếu tiêu chí nào. Phần này chạy ngay trên trang, không cần khóa API.")
    with st.form("scoring_form"):
        title = st.text_input("Tên sản phẩm")
        meta = st.text_area("Đoạn tóm tắt ngắn", height=70)
        description = st.text_area("Mô tả sản phẩm", height=150)
        subjects = st.text_area("Tiêu đề email, mỗi dòng một tiêu đề", height=90)
        script = st.text_area("Kịch bản nhắn tin", height=120)
        scoring_keywords = st.text_input("Từ khóa chính (cách nhau bằng dấu phẩy)")
        facts = st.text_area("Dữ kiện sản phẩm có thật (chất liệu, màu, size, giá…)", height=70,
                             help="Dùng để cảnh báo khi nội dung nhắc tới chi tiết không có thật.")
        score = st.form_submit_button("Kiểm tra", type="primary")
    if score:
        content = {"title": title, "meta_description": meta, "description": description,
                   "email_subjects": [line.strip() for line in subjects.splitlines() if line.strip()],
                   "sales_script": script}
        brief = {"keywords": split_items(scoring_keywords), "attributes": {"dữ kiện": facts}}
        show_review(content, brief, lang, detailed=True)

with results:
    st.caption("Phần này dành cho người chấm bài: số liệu được tạo tự động từ kết quả thực nghiệm của bài báo "
               "(dữ liệu Tiki 2026 và Amazon Reviews 2023).")
    tables_dir = DATA / "tables"
    table_files = ["predictive", "odds_ratios", "similarity", "checklist", "judge", "leakage", "m4_business"]
    shown = False
    for name in table_files:
        path = tables_dir / f"{name}.json"
        if not path.exists():
            continue
        shown = True
        table = json.loads(path.read_text(encoding="utf-8"))
        st.markdown(f"**{table['caption']}**")
        st.dataframe(pd.DataFrame(table["rows"], columns=table["columns"]), hide_index=True, use_container_width=True)
        st.caption(table.get("source", ""))
    if not shown:
        st.info("Kết quả đang được cập nhật")
    captions_path = ASSETS / "captions.json"
    captions = json.loads(captions_path.read_text(encoding="utf-8")) if captions_path.exists() else {}
    with st.expander("Biểu đồ và sơ đồ của nghiên cứu"):
        for image in sorted(ASSETS.glob("*.png")):
            st.image(str(image), caption=captions.get(image.name, image.stem.replace("_", " ")))

with methodology:
    st.write("Quy trình: chuẩn bị dữ liệu tổng hợp, xây dựng câu lệnh, sinh nội dung và đánh giá bằng danh sách "
             "kiểm tra có nguồn cùng các chỉ số nghiên cứu.")
    diagram = ASSETS / "diag_01_kien_truc_he_thong.png"
    if diagram.exists():
        st.image(str(diagram), caption="Kiến trúc hệ thống")
    st.subheader("Nguồn dữ liệu")
    st.table(pd.DataFrame([{"Nguồn": "Tiki", "Thông tin": "Thu thập 11/2026; dữ liệu từ 2019"},
                           {"Nguồn": "Amazon Reviews 2023", "Thông tin": "Hou và cộng sự, 2024"}]))
    st.subheader("Nguồn của từng ngưỡng kiểm tra")
    st.dataframe(pd.DataFrame(
        [{"Mã": code, "Quy tắc": label, "Nguồn": REFS[code][0], "Năm": REFS[code][1], "URL": REFS[code][2]}
         for label, _, code, _, _ in SEO.values()]
        + [{"Mã": code, "Quy tắc": label, "Nguồn": REFS[code][0], "Năm": REFS[code][1], "URL": REFS[code][2]}
           for label, _, code, _ in EMAIL.values() if code]), hide_index=True, use_container_width=True)
    st.warning("Nội dung do AI viết cần người đọc lại trước khi đăng: kiểm tra lại chất liệu, màu, size, giá và "
               "không để sót ưu đãi không có thật.")

"""Giao diện demo sinh và đánh giá nội dung sản phẩm."""
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
SEO = {
    "title_len_50_60": ("Độ dài tiêu đề", "50–60 ký tự", "S19", "title_chars"),
    "title_tiki_15_50": ("Tiêu đề Tiki tối ưu", "15–50 ký tự", "S23", "title_chars"),
    "title_tiki_max_225": ("Giới hạn tiêu đề Tiki", "≤225 ký tự", "S23", "title_chars"),
    "meta_len_120_156": ("Độ dài meta", "120–156 ký tự", "S12", "meta_chars"),
    "keyphrase_in_title": ("Từ khóa trong tiêu đề", "Có", "S11", None),
    "keyphrase_in_first_paragraph": ("Từ khóa trong đoạn đầu", "Có", "S10", None),
    "keyphrase_in_meta": ("Từ khóa trong meta", "Có", "S11", None),
    "density_proxy_0_5_3": ("Mật độ khớp chính xác (proxy)", "Chưa xác lập cho proxy", "S09", "keyphrase_density_proxy"),
    "desc_words_ge_200": ("Độ dài mô tả", "≥200 từ cách nhau bằng khoảng trắng", "S13", "description_words"),
}
EMAIL = {
    "mailchimp_le9w_le60c": ("Độ dài Mailchimp", "≤9 từ và ≤60 ký tự", "E01"),
    "omnisend_4_6w_or_21_40c": ("Độ dài Omnisend", "4–6 từ hoặc 21–40 ký tự", "E05"),
    "punct_le3": ("Dấu câu", "≤3", "E02"),
    "emoji_le1": ("Emoji", "≤1", "E03"),
    "no_fake_re_fwd": ("Tiền tố Re/Fwd", "Không dùng tiền tố giả", "E16"),
    "keyword_relevance": ("Từ khóa", "Có từ khóa", None),
    "offer_supported": ("Ưu đãi", "Có căn cứ trong brief", None),
}


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


def checklist(content, brief, lang):
    """Hiển thị từng kiểm tra, tỷ lệ áp dụng và cảnh báo có căn cứ."""
    keywords = brief.get("keywords", [])
    seo = seo_checks(content.get("title", ""), content.get("meta_description", ""), content.get("description", ""), keywords[0] if keywords else "", keywords, lang)
    rows = []
    seo_flags = []
    email_flags = []
    for key, (label, threshold, ref, measure) in SEO.items():
        flag = seo["checks"][key]
        if key == "density_proxy_0_5_3":
            flag = None
        value = seo.get(measure) if measure else flag
        if isinstance(value, float):
            value = round(value, 2)
        # Khoảng tên Tiki 15–50 ký tự mâu thuẫn với khoảng 50–60 của Ahrefs nên chỉ hiển thị, không tính vào tỷ lệ gộp.
        if isinstance(flag, bool) and key != "title_tiki_15_50":
            seo_flags.append(flag)
        rows.append({"Tiêu chí": label, "Giá trị đo": str(value) if value is not None else "—", "Ngưỡng": threshold, "Đạt?": "✅" if flag is True else "❌" if flag is False else "—", "Nguồn": source(ref)})
    for index, subject in enumerate(content.get("email_subjects", []), 1):
        email = email_checks(subject, keywords, brief)
        values = {"mailchimp_le9w_le60c": f"{email['words_space']} từ; {email['chars']} ký tự", "omnisend_4_6w_or_21_40c": f"{email['words_space']} từ; {email['chars']} ký tự", "punct_le3": email["punctuation"], "emoji_le1": email["emoji"]}
        for key, (label, threshold, ref) in EMAIL.items():
            flag = email["checks"][key]
            if isinstance(flag, bool):
                email_flags.append(flag)
            rows.append({"Tiêu chí": f"Email {index}: {label}", "Giá trị đo": str(values.get(key, flag)) if flag is not None else "—", "Ngưỡng": threshold, "Đạt?": "✅" if flag is True else "❌" if flag is False else "—", "Nguồn": source(ref)})
    script = script_rule_checks(content.get("sales_script", ""), brief, brief.get("objections", []))
    flag = script["cta_present"]
    rows.append({"Tiêu chí": "Kịch bản: lời kêu gọi hành động", "Giá trị đo": str(flag), "Ngưỡng": "Có cụm CTA theo từ điển", "Đạt?": "✅" if flag else "❌", "Nguồn": "Quy tắc đối chiếu nội bộ"})
    st.subheader("Bảng kiểm tra")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    def rate_text(values):
        """Định dạng tỷ lệ đạt (không trọng số) của một nhóm tiêu chí."""
        return f"{100 * sum(values) / len(values):.1f}%".replace(".", ",") if values else "không áp dụng"

    col_seo, col_email, col_script = st.columns(3)
    col_seo.metric("Tuân thủ tối ưu hóa tìm kiếm", rate_text(seo_flags))
    col_email.metric("Tuân thủ tiêu đề email", rate_text(email_flags))
    col_script.metric("Kịch bản có lời kêu gọi hành động", "Có" if script["cta_present"] else "Không")
    st.caption("Các ngưỡng là khuyến nghị theo nguồn, không bảo đảm hiệu quả. Mật độ là proxy khớp chính xác và không tính vào tỷ lệ; phát hiện căn cứ chỉ dựa trên từ điển.")
    all_text = " ".join([content.get("title", ""), content.get("meta_description", ""), content.get("description", ""), content.get("sales_script", "")] + content.get("email_subjects", []))
    unsupported = factual_support(all_text, brief)["unsupported_list"]
    if unsupported:
        st.warning("Thuộc tính có thể chưa có căn cứ: " + ", ".join(unsupported))
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
    """Trình bày nội dung và dữ liệu so sánh cùng danh mục."""
    st.subheader(content.get("title", ""))
    st.write("**Meta:**", content.get("meta_description", ""))
    st.markdown(content.get("description", ""))
    st.markdown("**Kịch bản chốt sale**")
    st.write(content.get("sales_script", ""))
    st.markdown("**Tiêu đề email**")
    for subject in content.get("email_subjects", []):
        st.write("•", subject)
    st.download_button("Tải JSON", json.dumps(content, ensure_ascii=False, indent=2), "noi-dung.json", "application/json")
    seo = checklist(content, brief, lang)
    benchmark = load("category_benchmarks.json", {}).get(brief.get("category"), {}).get("1")
    if benchmark:
        st.subheader("So sánh với sản phẩm bán chạy cùng danh mục")
        st.dataframe(pd.DataFrame([{"Chỉ số": "Độ dài mô tả (từ)", "Nội dung": seo["description_words"], "Trung vị nhóm bán chạy": benchmark["desc_len_words"]}, {"Chỉ số": "Số gạch đầu dòng", "Nội dung": seo["n_bullets"], "Trung vị nhóm bán chạy": benchmark["n_bullets"]}]), hide_index=True)
        st.caption(f"Cỡ mẫu nhóm bán chạy: {benchmark['n']}. Nguồn: dữ liệu Tiki tổng hợp trong nghiên cứu.")


st.set_page_config(page_title="Trợ lý nội dung sản phẩm", page_icon="✍️", layout="wide")
st.title("Trợ lý nội dung sản phẩm")
with st.sidebar:
    provider_label = st.selectbox("Nhà cung cấp API", ["Gemini", "OpenAI"])
    provider = provider_label.lower()
    secret_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    try:
        secret = st.secrets.get(secret_name, "")
    except (FileNotFoundError, KeyError):
        secret = ""
    api_key = st.text_input("Khóa API", type="password", help="Chỉ lưu trong phiên hiện tại.") or secret
    model = st.text_input("Tên mô hình (tùy chọn)", help="OpenAI cần tên mô hình hoặc biến OPENAI_MODEL.")
    language = st.selectbox("Ngôn ngữ nội dung", ["Tiếng Việt", "English"])
    lang = "vi" if language == "Tiếng Việt" else "en"
    st.caption(f"Lượt sinh đã dùng: {st.session_state.get('generation_count', 0)}/15")

generation, scoring, results, methodology = st.tabs(["Sinh nội dung", "Chấm điểm nội dung có sẵn", "Kết quả nghiên cứu", "Phương pháp & nguồn"])
with generation:
    categories = load("categories.json", {})
    objections = load("objections_by_category.json", {})
    # Chế độ minh họa (?demo=1): tự điền một ví dụ thật và sinh bằng M0 để xem nhanh đầu ra và bảng kiểm tra.
    demo = st.query_params.get("demo") == "1"
    category_list = list(categories) or ["Khác"]
    demo_category = "Chân váy" if "Chân váy" in category_list else category_list[0]
    category = st.selectbox("Danh mục", category_list, index=category_list.index(demo_category) if demo else 0)
    suggested = st.multiselect("Từ khóa gợi ý", categories.get(category, []), placeholder="Chọn từ khóa đặc trưng của danh mục")
    free_keywords = st.text_input("Từ khóa khác, cách nhau bằng dấu phẩy", value="chân váy chữ a, công sở, xếp ly" if demo else "")
    raw_options = objections.get(category, []) if isinstance(objections, dict) else []
    # objections_by_category.json lưu dạng [{"phrase": chủ đề, "count": n, "share": tỷ lệ}] (trích từ đánh giá 1–3 sao).
    options = [item["phrase"] if isinstance(item, dict) else str(item) for item in raw_options] if isinstance(raw_options, list) else []
    selected_objections = st.multiselect("Phản đối thường gặp (từ đánh giá thật của danh mục)", options,
                                         default=options[:2] if demo else [], placeholder="Chọn phản đối cần kịch bản trả lời")
    free_objections = st.text_input("Phản đối khác, cách nhau bằng dấu phẩy")
    with st.form("generation_form"):
        cols = st.columns(4)
        material = cols[0].text_input("Chất liệu")
        color = cols[1].text_input("Màu sắc")
        size = cols[2].text_input("Kích cỡ")
        price = cols[3].text_input("Giá")
        method = st.selectbox("Phương pháp", ["M0 – Mẫu cố định (không cần API)", "M1 – Zero-shot", "M2 – Prompt engineering (khuyến nghị)"])
        submitted = st.form_submit_button("Sinh nội dung", type="primary")
    if demo and not st.session_state.get("demo_done"):
        submitted = True
        material = material or "kaki"
        st.session_state["demo_done"] = True
    if submitted:
        count = st.session_state.get("generation_count", 0)
        if count >= 15:
            st.error("Phiên này đã đạt giới hạn 15 lượt sinh.")
        else:
            keywords = list(dict.fromkeys(suggested + split_items(free_keywords))) or [category]
            attributes = {key: value for key, value in {"chất liệu": material, "màu sắc": color, "kích cỡ": size, "giá": price}.items() if value.strip()}
            brief = {"item_id": "demo", "category": category, "keywords": keywords, "attributes": attributes, "objections": selected_objections + split_items(free_objections)}
            st.session_state["generation_count"] = count + 1
            try:
                if method.startswith("M0"):
                    content = template_output(brief, lang)
                else:
                    prompt = zero_shot([brief], lang) if method.startswith("M1") else engineered([brief], lang)
                    with st.spinner("Đang sinh nội dung…"):
                        text, used_model, latency = generate_via_api(prompt, provider, api_key, model.strip() or None)
                    content = parse_response(text)
                    st.info(f"Mô hình: {used_model} · Thời gian: {latency:.2f} giây")
                st.session_state["generated_result"] = (content, brief, lang)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                st.error(str(exc))
    if "generated_result" in st.session_state:
        show_content(*st.session_state["generated_result"])

with scoring:
    st.caption("Chấm điểm cục bộ, không cần API. Hãy nhập dữ kiện sản phẩm để đối chiếu căn cứ.")
    with st.form("scoring_form"):
        title = st.text_input("Tiêu đề")
        meta = st.text_area("Mô tả meta")
        description = st.text_area("Mô tả sản phẩm")
        subjects = st.text_area("Tiêu đề email, mỗi dòng một tiêu đề")
        script = st.text_area("Kịch bản chốt sale")
        scoring_keywords = st.text_input("Từ khóa, cách nhau bằng dấu phẩy")
        facts = st.text_area("Dữ kiện sản phẩm đã xác thực")
        score = st.form_submit_button("Chấm điểm", type="primary")
    if score:
        content = {"title": title, "meta_description": meta, "description": description, "email_subjects": [line.strip() for line in subjects.splitlines() if line.strip()], "sales_script": script}
        brief = {"keywords": split_items(scoring_keywords), "attributes": {"dữ kiện": facts}}
        checklist(content, brief, lang)

with results:
    st.caption("Số liệu dưới đây được tạo tự động từ kết quả thực nghiệm của bài báo (dữ liệu Tiki 2026 và Amazon Reviews 2023).")
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
    for image in sorted(ASSETS.glob("*.png")):
        st.image(str(image), caption=captions.get(image.name, image.stem.replace("_", " ")))

with methodology:
    st.write("Quy trình: chuẩn bị dữ liệu tổng hợp, xây dựng prompt, sinh nội dung và đánh giá bằng checklist cùng các chỉ số nghiên cứu.")
    diagram = ASSETS / "diag_01_kien_truc_he_thong.png"
    if diagram.exists():
        st.image(str(diagram), caption="Kiến trúc hệ thống")
    st.subheader("Nguồn dữ liệu")
    st.table(pd.DataFrame([{"Nguồn": "Tiki", "Thông tin": "Thu thập 11/2026; dữ liệu từ 2019"}, {"Nguồn": "Amazon Reviews 2023", "Thông tin": "Hou và cộng sự, 2024"}]))
    st.subheader("Nguồn các quy tắc")
    st.dataframe(pd.DataFrame([{"Mã": code, "Quy tắc": label, "Nguồn": REFS[code][0], "Năm": REFS[code][1], "URL": REFS[code][2]} for label, _, code, _ in SEO.values()] + [{"Mã": code, "Quy tắc": label, "Nguồn": REFS[code][0], "Năm": REFS[code][1], "URL": REFS[code][2]} for label, _, code in EMAIL.values() if code]), hide_index=True, use_container_width=True)
    st.warning("Nội dung AI cần người duyệt trước khi sử dụng; không tự bịa ưu đãi hoặc thông tin sản phẩm.")

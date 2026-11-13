"""Chỉ số nội dung và đối chiếu tham chiếu; các phép từ điển là proxy."""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


def norm(text):
    """Chuẩn hóa chữ thường và dấu gạch nối từ tokenizer."""
    return unicodedata.normalize('NFC', str(text or '')).lower().replace('_', ' ')


def words_space(text):
    """Đếm đơn vị ngăn bởi khoảng trắng."""
    return len(str(text or '').split())


def words_tokenized(text):
    """Đếm từ do underthesea phân đoạn."""
    from underthesea import word_tokenize
    return len(word_tokenize(str(text or '')))


def chars(text):
    """Đếm ký tự Unicode NFC."""
    return len(unicodedata.normalize('NFC', str(text or '')))


def sentences(text):
    """Đếm câu bằng dấu kết thúc câu và xuống dòng."""
    return len(sentence_parts(text))


def sentence_parts(text):
    """Tách câu theo dấu câu kết thúc và xuống dòng."""
    return [part.strip() for part in re.split(r'[.!?…]+|\n+', str(text or '')) if part.strip()]


def emoji(text):
    """Đếm cụm emoji phổ biến, gồm chuỗi ZWJ và cờ."""
    import regex
    pattern = r'(?:\p{Regional_Indicator}{2}|[0-9#*]\ufe0f?\u20e3|\p{Extended_Pictographic}(?:\ufe0f|\p{Emoji_Modifier})?(?:\u200d\p{Extended_Pictographic}(?:\ufe0f|\p{Emoji_Modifier})?)*)'
    return len(regex.findall(pattern, str(text or '')))


def punctuation(text):
    """Đếm dấu câu theo nhóm Unicode P."""
    return sum(unicodedata.category(c).startswith('P') for c in str(text or ''))


def matches(text, phrase):
    """Đếm cụm nguyên vẹn theo ranh giới ký tự từ Unicode."""
    phrase = norm(phrase).strip()
    if not phrase:
        return 0
    pattern = r'(?<!\w)' + re.escape(phrase).replace(r'\ ', r'\s+') + r'(?!\w)'
    return len(re.findall(pattern, norm(text)))


def ratio_checks(checks):
    """Tính tỷ lệ đạt trên các cờ có thể áp dụng."""
    applicable = [v for v in checks.values() if isinstance(v, bool)]
    return sum(applicable) / len(applicable) if applicable else None


def seo_checks(title, meta_description, description, keyphrase, keywords, lang):
    """Đánh giá checklist SEO và xuất các giá trị thô."""
    title = title or ''
    description = description or ''
    has_meta = bool(meta_description and str(meta_description).strip())
    nwords = words_space(description)
    count = matches(description, keyphrase)
    density = 100 * count / nwords if nwords and keyphrase else None
    paragraphs = [p for p in re.split(r'\n\s*\n', description) if p.strip()]
    parts = sentence_parts(description)
    lengths = [words_space(p) for p in parts]
    keys = [str(k) for k in (keywords or []) if str(k).strip()]
    checks = {
        # S19 Ahrefs: khoảng 50–60 ký tự.
        'title_len_50_60': 50 <= chars(title) <= 60,
        # S23 Tiki: 15–50 ký tự tối ưu và trần 225 ký tự.
        'title_tiki_15_50': 15 <= chars(title) <= 50 if lang == 'vi' else None,
        'title_tiki_max_225': chars(title) <= 225 if lang == 'vi' else None,
        # S12 Yoast: 120–156 ký tự khi có meta.
        'meta_len_120_156': 120 <= chars(meta_description) <= 156 if has_meta else None,
        # S11 Yoast: keyphrase xuất hiện trong tiêu đề và meta.
        'keyphrase_in_title': bool(matches(title, keyphrase)) if keyphrase else None,
        # S10 Yoast: keyphrase xuất hiện ở đoạn đầu.
        'keyphrase_in_first_paragraph': bool(matches(paragraphs[0], keyphrase)) if paragraphs and keyphrase else None,
        'keyphrase_in_meta': bool(matches(meta_description, keyphrase)) if has_meta and keyphrase else None,
        # S09 Yoast: 0,5–3%; đây là proxy exact-match, không tương đương plugin.
        'density_proxy_0_5_3': 0.5 <= density <= 3 if density is not None else None,
        # S13/S24 Yoast/Tiki: mốc 200 từ; chuyển sang tiếng Việt chưa được kiểm định.
        'desc_words_ge_200': nwords >= 200,
    }
    # Tỷ lệ tuân thủ gộp KHÔNG tính: proxy mật độ (ngưỡng 0,5–3% gắn với thuật toán Yoast, chỉ báo cáo mô tả)
    # và khoảng tên Tiki 15–50 ký tự (mâu thuẫn trực tiếp với khoảng 50–60 ký tự của Ahrefs); hai cờ này báo cáo riêng.
    aggregate = {k: v for k, v in checks.items() if k not in ('density_proxy_0_5_3', 'title_tiki_15_50')}
    return {'checks': checks, 'compliance_rate': ratio_checks(aggregate), 'title_chars': chars(title), 'meta_chars': chars(meta_description) if has_meta else None, 'description_words': nwords, 'keyphrase_count': count, 'keyphrase_density_proxy': density, 'keyword_coverage': sum(bool(matches(title + ' ' + description, k)) for k in keys) / len(keys) if keys else None, 'n_paragraphs': len(paragraphs), 'n_bullets': len(re.findall(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+', description)), 'avg_sentence_words': sum(lengths) / len(lengths) if lengths else None, 'pct_sentences_over_20_words': 100 * sum(n > 20 for n in lengths) / len(lengths) if lengths else None}


# Từ điển "claim thuộc tính" tách theo ngôn ngữ. Lý do: dùng chung một danh sách làm từ tiếng Việt khớp nhầm
# từ tiếng Anh (ví dụ màu "be" khớp động từ "be"; "returns" khớp nghĩa thông thường). Mỗi mẫu có ranh giới từ.
CLAIM_PATTERNS = {
    'vi': [
        r'\b(?:đỏ|xanh(?: lá| dương| navy| rêu| ngọc)?|trắng|đen|vàng|hồng|tím|nâu|xám|be|kem)\b',
        r'\b(?:cotton|polyester|lụa|linen|đũi|ren|jean|denim|kaki|voan|chiffon|thun lạnh|vải thun|nỉ|len lông|da thật|satin|nhung)\b',
        r'\b(?:dáng ôm|dáng suông|dáng xòe|cổ vuông|cổ tròn|cổ tim|tay dài|tay ngắn|tay phồng|sát nách|hai dây|trễ vai)\b',
        r'\b(?:size|cỡ|kích cỡ)\b\s*[:\-]?\s*(?:xxs|xs|xxl|xl|s|m|l|\d{1,3})\b|\b\d+(?:[.,]\d+)?\s*(?:cm|kg)\b',
        r'\b\d+(?:[.,]\d+)?\s*%',
        r'(?<!\w)\d[\d.,]*\s*(?:đồng|vnđ|vnd|đ)\b',
        r'\b(?:miễn phí vận chuyển|freeship|free ship|bảo hành|đổi trả|hoàn tiền)\b',
    ],
    'en': [
        r'\b(?:red|blue|navy|white|black|yellow|pink|purple|brown|gray|grey|beige|green|khaki|ivory)\b',
        r'\b(?:cotton|polyester|spandex|linen|silk|satin|lace|denim|leather|wool|cashmere|rayon|chiffon|nylon)\b',
        r'\b(?:slim fit|regular fit|relaxed fit|oversized|sleeveless|long sleeve|short sleeve|v-neck|crew neck|high waist)\b',
        r'\bsize\b\s*[:\-]?\s*(?:xxs|xs|xxl|xl|s|m|l|\d{1,3})\b|\b\d+(?:\.\d+)?\s*(?:cm|inch|inches|in)\b',
        r'\b\d+(?:\.\d+)?\s*%',
        r'\$\s*\d[\d.,]*|\b\d[\d.,]*\s*(?:usd|dollars)\b',
        r'\b(?:free shipping|free returns?|return policy|money[- ]back|warranty|guarantee)\b',
    ],
}


def claim_supported(source, claim):
    """Kiểm tra một claim có căn cứ trong brief, so theo GIÁ TRỊ thay vì nguyên cụm chữ.

    - Claim size ("size M", "kích cỡ: L"): ký hiệu size hoặc số size phải xuất hiện như một từ riêng trong brief
      (brief thường ghi "Kích cỡ: L(55-64kg), M(45-54kg)").
    - Claim số đo ("54 kg", "90 cm"): con số phải xuất hiện trong brief.
    - Các claim khác (màu, chất liệu, kiểu dáng, giá, chính sách): khớp cụm chữ như cũ.
    """
    size = re.match(r'(?:size|cỡ|kích cỡ)\s*[:\-]?\s*(\w+)$', claim, re.IGNORECASE)
    if size:
        return bool(re.search(r'(?<![\w])' + re.escape(size.group(1).lower()) + r'(?![\w])', source))
    measure = re.match(r'(\d+(?:[.,]\d+)?)\s*(?:cm|mm|kg|inch|inches|in)$', claim, re.IGNORECASE)
    if measure:
        return measure.group(1) in source
    return matches(source, claim)


def detect_lang(text):
    """Nhận diện tiếng Việt khi văn bản có ký tự mang dấu tiếng Việt."""
    return 'vi' if re.search(r'[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]', str(text).lower()) else 'en'


def factual_support(text, brief, lang=None):
    """Đối chiếu claim từ điển với các trường sản phẩm trong brief (từ điển theo ngôn ngữ của văn bản)."""
    brief = brief or {}
    lang = lang or detect_lang(text)
    fields = {k: brief.get(k) for k in ('keywords', 'attributes', 'category', 'price')}
    source = norm(json.dumps(fields, ensure_ascii=False))
    found = []
    for pattern in CLAIM_PATTERNS[lang]:
        found.extend(m.group().strip() for m in re.finditer(pattern, norm(text), re.IGNORECASE))
    found = list(dict.fromkeys(found))
    unsupported = [claim for claim in found if not claim_supported(source, claim)]
    total = len(found)
    return {'supported': total - len(unsupported), 'unsupported': len(unsupported), 'total': total, 'unsupported_list': unsupported, 'hallucination_rate': len(unsupported) / total if total else None}


def email_checks(subject, keywords, brief):
    """Đánh giá checklist tiêu đề email và offer không có căn cứ."""
    subject = subject or ''
    offers = re.findall(r'\d+(?:[.,]\d+)?\s*%|giảm giá|khuyến mãi|freeship|free\s*ship(?:ping)?|miễn phí vận chuyển|quà tặng|free gift|discount|sale', norm(subject))
    source = norm(json.dumps(brief or {}, ensure_ascii=False))
    unsupported = [offer for offer in offers if not matches(source, offer)]
    keys = [k for k in (keywords or []) if str(k).strip()]
    checks = {
        # E01 Mailchimp: không quá 9 từ và 60 ký tự.
        'mailchimp_le9w_le60c': words_space(subject) <= 9 and chars(subject) <= 60,
        # E05 Omnisend: 4–6 từ hoặc 21–40 ký tự.
        'omnisend_4_6w_or_21_40c': 4 <= words_space(subject) <= 6 or 21 <= chars(subject) <= 40,
        # E02 Mailchimp: tối đa 3 dấu câu.
        'punct_le3': punctuation(subject) <= 3,
        # E03 Mailchimp: tối đa 1 cụm emoji.
        'emoji_le1': emoji(subject) <= 1,
        # E16 Google: không giả mạo tiền tố Re/Fwd; không có metadata nên chỉ phát hiện tiền tố.
        'no_fake_re_fwd': not bool(re.match(r'^\s*(?:re|fw|fwd)\s*:', subject, re.I)),
        'keyword_relevance': any(matches(subject, k) for k in keys) if keys else None,
        'offer_supported': not bool(unsupported),
    }
    return {'checks': checks, 'compliance_rate': ratio_checks(checks), 'words_space': words_space(subject), 'chars': chars(subject), 'punctuation': punctuation(subject), 'emoji': emoji(subject), 'unsupported_offer': bool(unsupported), 'unsupported_offer_list': unsupported}


# Từ vựng chủ đề dùng để kiểm tra kịch bản có NHẮC ĐẾN từng chủ đề phản đối hay không.
# Bản trước khớp bất kỳ từ nào trong tên chủ đề (kể cả "không"), nên gần như luôn đạt; bản này chỉ
# nhận các từ mang nội dung của chủ đề. Đây vẫn là chỉ báo từ vựng: nó không đánh giá câu trả lời
# có thuyết phục hay không (phần đó do tiêu chí "xử lý phản đối" của giám khảo LLM đảm nhận).
RESPONSE_THEMES = {
    "Size không vừa": r"\b(?:size|kích cỡ|cỡ|số đo|bảng size|chiều cao|cân nặng|vòng (?:eo|ngực|mông)|form|\d+\s*(?:cm|kg))\b",
    "Không giống hình/mô tả": r"\b(?:hình|ảnh|ảnh thật|hình thật|ảnh chụp|mô tả|thực tế|video)\b",
    "Chất liệu kém": r"\b(?:chất liệu|chất vải|vải|sợi|cotton|thun|lụa|đũi|kaki|jean|voan|len)\b",
    "Màu sắc khác": r"\b(?:màu|màu sắc|tông màu|sắc độ|ánh sáng)\b",
    "Đường may, lỗi sản phẩm": r"\b(?:đường may|đường chỉ|may kỹ|lỗi|kiểm tra|kiểm hàng|đổi trả|đổi hàng|bảo hành|hoàn tiền)\b",
    "Giao hàng, đóng gói": r"\b(?:giao hàng|giao|đóng gói|vận chuyển|ship|gói hàng)\b",
}


def script_rule_checks(script, brief, objections):
    """Đo CTA, từ khóa phản đối và claim trong kịch bản."""
    cta = r'\b(?:mua ngay|đặt hàng|chốt đơn|thêm vào giỏ|nhắn tin|inbox|để lại số điện thoại|order now|add to cart|buy now|shop now)\b'
    objections = [str(x) for x in (objections or []) if str(x).strip()]
    covered = 0
    for objection in objections:
        pattern = RESPONSE_THEMES.get(objection)
        if pattern:
            covered += bool(re.search(pattern, norm(script)))
        else:
            # Chủ đề ngoài từ điển: yêu cầu nhắc nguyên cụm, không khớp từng từ lẻ.
            covered += bool(matches(script, objection))
    return {'cta_present': bool(re.search(cta, norm(script))), 'objection_keyword_coverage': covered / len(objections) if objections else None, 'factual_support': factual_support(script, brief), 'length_words': words_space(script)}


class UnicodeTokenizer:
    """Tokenizer ROUGE giữ nguyên ký tự từ Unicode."""
    def tokenize(self, text):
        """Trả các token Unicode sau chuẩn hóa."""
        return re.findall(r'\w+', norm(text))


@lru_cache(maxsize=1)
def bert_scorer():
    """Nạp BERTScore từ mô hình cục bộ một lần trên CPU."""
    from bert_score import BERTScorer
    model = Path(__file__).resolve().parents[1] / 'models' / 'bert-base-multilingual-cased'
    if not (model / 'config.json').is_file():
        raise FileNotFoundError('Mô hình BERT cục bộ chưa đầy đủ: ' + str(model))
    return BERTScorer(model_type=str(model), num_layers=9, batch_size=8, device='cpu', lang='en')


def reference_similarity(cands, refs, lang, fast=False):
    """Tính ROUGE Unicode, BLEU, chrF++ và BERTScore theo cặp."""
    from rouge_score import rouge_scorer
    import sacrebleu
    if len(cands) != len(refs):
        raise ValueError('Số candidate và reference phải bằng nhau')
    if not cands:
        return {'per_item': [], 'mean': {}, 'corpus_bleu': None}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=UnicodeTokenizer())
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    bleu = sacrebleu.metrics.BLEU(tokenize='13a', effective_order=True)
    rows = []
    for cand, ref in zip(cands, refs):
        scores = scorer.score(ref, cand)
        row = {key: value.fmeasure for key, value in scores.items()}
        row['sentence_bleu'] = bleu.sentence_score(cand, [ref]).score
        row['chrf_plus_plus'] = chrf.sentence_score(cand, [ref]).score
        row['bertscore_f1'] = None
        rows.append(row)
    if not fast:
        _, _, f1 = bert_scorer().score(cands, refs)
        for row, score in zip(rows, f1.tolist()):
            row['bertscore_f1'] = score
    means = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        means[key] = sum(values) / len(values) if values else None
    return {'per_item': rows, 'mean': means, 'corpus_bleu': bleu.corpus_score(cands, [refs]).score}

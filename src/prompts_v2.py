"""Câu lệnh v2 và truy hồi ví dụ chỉ từ tập train."""
import hashlib
import json
import unicodedata

import prompts


def _norm(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).lower().split())


def zero_shot(briefs):
    return prompts.zero_shot(briefs)


def engineered(briefs, no_obj=False):
    briefs = [{k: v for k, v in b.items() if not (no_obj and k == "objections")} for b in briefs]
    text = prompts.engineered(briefs)
    text = text.replace("DỮ KIỆN: chỉ dùng thông tin có trong brief.",
                        "DỮ KIỆN: chỉ dùng thông tin có trong brief, được dùng giá (price) trong brief.")
    # Vòng phát triển 2 (trên dev, 23/11): vòng 1 có 55/60 tiêu đề, câu đầu mô tả và tiêu đề email viết thường
    # (mô hình chép nguyên từ khóa viết thường), và 42/60 tiêu đề chỉ dài 45–49 ký tự.
    for old, new in [
        ("TIÊU ĐỀ (title): dài 50–60 ký tự, bắt đầu bằng từ khóa chính.",
         "TIÊU ĐỀ (title): dài 50–60 ký tự tính cả khoảng trắng, nhắm khoảng 55; tự đếm số ký tự trước khi trả lời, "
         "thiếu thì thêm một đặc điểm có trong brief hoặc dịp mặc. Bắt đầu bằng từ khóa chính."),
        ("MÔ TẢ META (meta_description): dài 120–156 ký tự,",
         "MÔ TẢ META (meta_description): dài 120–156 ký tự tính cả khoảng trắng, nhắm khoảng 140 và tự đếm trước khi trả lời,"),
    ]:
        if old not in text:
            raise ValueError("Không tìm thấy quy tắc cần sửa trong prompts.engineered: " + old[:40])
        text = text.replace(old, new)
    text = text.replace("\n" + prompts._format_vi(),
                        "\n- CHÍNH TẢ: viết hoa chữ cái đầu của tiêu đề, mô tả meta, mỗi câu, mỗi gạch đầu dòng và mỗi tiêu đề "
                        "email. Từ khóa trong brief viết thường chỉ để tra cứu; khi đưa vào văn bản phải viết đúng chính tả.\n"
                        + prompts._format_vi(), 1)
    if no_obj:
        old = ("Nếu brief có objections thì trả lời lần lượt từng phản đối "
               "bằng hành động người bán làm được ngay (gửi thêm ảnh hoặc video thật, đo lại thông số, tư vấn size theo "
               "chiều cao cân nặng, kiểm tra kỹ trước khi gói hàng), không hứa chính sách không có trong brief.")
        if old not in text:
            raise ValueError("Không tìm thấy quy tắc objections trong prompts.engineered")
        text = text.replace(old, "Lường trước 1–2 băn khoăn phổ biến khi mua online và trả lời bằng hành động người bán làm được, không hứa chính sách không có trong brief.")
    return text


def fewshot(briefs, examples):
    return (engineered(briefs) + "\nVÍ DỤ THAM KHẢO TỪ TẬP TRAIN: chỉ học giọng văn, cách trình bày; "
            "không sao chép dữ kiện sang sản phẩm khác.\n" + json.dumps(examples, ensure_ascii=False, indent=1))


def retrieve_examples(target, products, briefs, k=3):
    """Ghép theo product_id; ưu tiên category_l2 rồi xếp theo TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    by_id = {str(b['item_id']): b for b in briefs if b.get('split') == 'train'}
    product_by_id = {str(p['product_id']): p for p in products}
    target_product = product_by_id[str(target['item_id'])]
    candidates = [(p, by_id[str(p['product_id'])]) for p in products
                  if p.get('split') == 'train' and str(p.get('product_id')) in by_id
                  and _norm(p.get('name')) != _norm(target_product.get('name'))
                  # không lấy ví dụ cùng shop: tránh chép văn mẫu và dữ kiện của chính shop đó
                  and str(p.get('seller_id')) != str(target_product.get('seller_id'))]
    if len(candidates) < k:
        raise ValueError('Không đủ ví dụ train khác nhóm tên')
    def features(b):
        return str(b.get('category', '')) + ' ' + ' '.join(b.get('keywords', []))
    try:
        vectors = TfidfVectorizer().fit_transform([features(target)] + [features(b) for _, b in candidates])
        scores = cosine_similarity(vectors[0], vectors[1:])[0]
    except ValueError:
        scores = [0.] * len(candidates)
    ranked = sorted(zip(candidates, scores), key=lambda x: (-int(x[0][0].get('category_l2') == target_product.get('category_l2')), -x[1], str(x[0][0]['product_id'])))
    return [{'brief': {key: b[key] for key in ('category', 'keywords', 'attributes') if key in b},
             'description': ' '.join(str(p.get('description_text', '')).split()[:150])}
            for ((p, b), _) in ranked[:k]]


def template_hash():
    sample = {'item_id': 'fixed', 'category': 'áo', 'keywords': ['áo nữ'],
              'attributes': {}, 'price': 100, 'objections': ['size']}
    examples = {'fixed': [{'brief': {'category': 'áo'}, 'description': 'Ví dụ.'}] * 3}
    texts = [json.dumps(prompts.template_output(sample), ensure_ascii=False, sort_keys=True),
             zero_shot([sample]), engineered([sample]), engineered([sample], True), fewshot([sample], examples)]
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode('utf-8')).hexdigest()

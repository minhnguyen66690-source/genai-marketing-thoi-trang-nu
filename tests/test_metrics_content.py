"""Kiểm thử đơn giản, chạy trực tiếp bằng Python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics_content import words_space, words_tokenized, chars, sentences, emoji, punctuation, seo_checks, factual_support, email_checks, script_rule_checks, reference_similarity
from judge import agreement


def test_all():
    """Kiểm tra các phép đo tiếng Việt và xử lý dữ liệu thiếu."""
    assert words_space('áo sơ mi') == 3
    assert words_tokenized('áo sơ mi') >= 1
    assert chars('a\u0301') == 1
    assert sentences('Áo đẹp. Váy mới!') == 2
    assert emoji('👩‍💻😀') == 2
    assert punctuation('Áo! Đẹp?') == 2
    title = 'á' * 55
    seo = seo_checks(title, None, 'áo cotton đẹp', 'áo', ['áo'], 'vi')
    assert seo['checks']['title_len_50_60'] is True
    assert seo['checks']['meta_len_120_156'] is None
    assert seo['keyword_coverage'] == 1
    assert seo_checks('áo_dài', None, 'áo dài', 'áo dài', ['áo dài'], 'vi')['checks']['keyphrase_in_title']
    fact = factual_support('Áo đỏ cotton', {'attributes': {'màu': 'đỏ'}})
    assert fact['supported'] == 1 and fact['unsupported'] == 1
    assert factual_support('Sản phẩm đẹp', {})['hallucination_rate'] is None
    email = email_checks('Áo đẹp miễn phí vận chuyển', ['áo'], {})
    assert email['unsupported_offer'] and email['checks']['keyword_relevance']
    script = script_rule_checks('Nhắn tin hỏi về chất liệu', {}, ['chất liệu'])
    assert script['cta_present'] and script['objection_keyword_coverage'] == 1
    assert script_rule_checks('', {}, [])['objection_keyword_coverage'] is None
    sim = reference_similarity(['Áo dài màu đỏ'], ['Áo dài màu đỏ'], 'vi', fast=True)
    assert sim['per_item'][0]['rouge1'] == 1.0
    assert sim['per_item'][0]['rouge2'] == 1.0
    assert sim['per_item'][0]['rougeL'] == 1.0
    assert agreement([1, 2, 3], [1, 2, 3])['quadratic_kappa'] == 1.0


if __name__ == '__main__':
    test_all()
    print('PASS: test_all')

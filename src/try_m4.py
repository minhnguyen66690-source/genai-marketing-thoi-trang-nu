r"""Thử mô tả sản phẩm trên máy:

    .venv\Scripts\python.exe src\try_m4.py --demo
    .venv\Scripts\python.exe src\try_m4.py --demo --m3
    .venv\Scripts\python.exe src\try_m4.py
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel

import finetune_v2

ROOT = Path(__file__).resolve().parents[1]


def tach(text):
    return [phan.strip() for phan in text.split(',') if phan.strip()]


def nhap_san_pham(demo, guide):
    if demo:
        return {
            'item_id': 'demo', 'category': 'Chân váy',
            'keywords': ['chân váy chữ a', 'công sở', 'xếp ly'],
            'attributes': {'chất liệu': ['kaki'], 'màu sắc': ['be', 'đen'],
                           'kích cỡ': ['S', 'M', 'L']},
            'price': 259000, 'objections': ['Size không vừa', 'Đường may, lỗi sản phẩm'],
        }, ('S: 40-45kg, eo 62-64cm, dài 55cm; M: 46-50kg, eo 66-68cm, dài 56cm; '
            'L: 51-55kg, eo 70-72cm, dài 57cm. Vải kaki dày vừa, không nhăn, có lớp lót trong. '
            'Khóa kéo ẩn bên hông, xếp ly trước. Giặt tay nước lạnh, không sấy, ủi nhiệt độ thấp.')
    print('Loại hàng gợi ý: ' + ', '.join(guide))
    brief = {'item_id': 'san_pham', 'category': input('Loại hàng: ').strip(),
             'keywords': tach(input('Từ khóa (cách nhau bằng dấu phẩy, từ đầu là từ khóa chính): ')),
             'attributes': {}}
    for key, label in [('chất liệu', 'Chất liệu'), ('màu sắc', 'Màu'), ('kích cỡ', 'Kích cỡ')]:
        values = tach(input(f'{label} (cách nhau bằng dấu phẩy): '))
        if values:
            brief['attributes'][key] = values
    gia = ''.join(c for c in input('Giá: ') if c.isdecimal())
    if gia:
        brief['price'] = int(gia)
    brief['objections'] = tach(input('Lo ngại của khách (cách nhau bằng dấu phẩy): '))
    facts = input('Chi tiết thật (chỉ dùng cho M3): ').strip()
    return brief, facts


def main():
    parser = argparse.ArgumentParser(description='Thử mô hình viết mô tả sản phẩm.')
    parser.add_argument('--demo', action='store_true', help='Dùng sản phẩm mẫu, không hỏi thông tin.')
    parser.add_argument('--only', choices=['a', 'b', 'ab'], default='ab', help='Chọn M4a, M4b hoặc cả hai.')
    parser.add_argument('--m3', action='store_true', help='Gọi thêm mô hình lớn M3.')
    args = parser.parse_args()
    guide = json.loads((ROOT / 'webapp/data/style_guide.json').read_text(encoding='utf-8'))
    brief, facts = nhap_san_pham(args.demo, guide)
    print('M4 chạy trên CPU, mỗi mô tả khoảng 30 đến 60 giây', flush=True)

    # Nạp gốc một lần; tắt adapter để tái hiện M4a trên cùng mô hình.
    torch.manual_seed(0)
    tokenizer, base = finetune_v2.load()
    model = PeftModel.from_pretrained(base, ROOT / 'models_v2/qwen_lora', local_files_only=True)
    model.eval()
    ids = tokenizer.apply_chat_template(finetune_v2.messages(brief), tokenize=True,
                                        add_generation_prompt=True, return_tensors='pt', return_dict=False)
    blocks = []
    for kind, title in [('a', 'M4a: Qwen gốc'), ('b', 'M4b: Qwen + LoRA')]:
        if kind not in args.only:
            continue
        print(f'Đang chạy {title}...', flush=True)
        start = time.perf_counter()
        with torch.no_grad(), (model.disable_adapter() if kind == 'a' else nullcontext()):
            generated = model.generate(ids, do_sample=False, no_repeat_ngram_size=3,
                                       max_new_tokens=400, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - start
        result = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
        block = f'=== {title} ===\nThời gian: {elapsed:.2f} giây\n{result}'
        print(block, flush=True)
        blocks.append(block)

    if args.m3:
        print('Đang chạy M3: mô hình lớn...', flush=True)
        start = time.perf_counter()
        try:
            import llm_backends
            import prompts_m3

            # Chi tiết thật chỉ được đưa vào lời nhắc M3.
            prompt = prompts_m3.build([brief], {brief['item_id']: {
                'short_description': '', 'description_text': facts}}, guide)
            raw = llm_backends.generate(prompt, model='gpt-6.1-terra', effort='low', tag='try_m4')
            item = llm_backends.extract_json(raw)[0]
            result = '\n'.join(f'{key}: {item.get(key, "")}' for key in
                               ('title', 'description', 'sales_script'))
        except Exception as exc:
            result = f'Không thể chạy M3: {exc}'
        elapsed = time.perf_counter() - start
        block = f'=== M3: mô hình lớn ===\nThời gian: {elapsed:.2f} giây\n{result}'
        print(block, flush=True)
        blocks.append(block)

    dest = ROOT / '_workspace/try_m4' / f'ket_qua_{datetime.now():%Y%m%d_%H%M%S}.md'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('# Brief\n\n```json\n' + json.dumps(brief, ensure_ascii=False, indent=2)
                    + '\n```\n\nChi tiết thật (M3): ' + facts + '\n\n'
                    + '\n\n'.join(blocks) + '\n', encoding='utf-8')
    print(f'Đã lưu: {dest}')


if __name__ == '__main__':
    main()

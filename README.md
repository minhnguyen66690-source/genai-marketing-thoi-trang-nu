# Hệ thống soạn nội dung marketing cho thời trang nữ

Mã nguồn bài thu hoạch cuối kỳ môn Chuyên đề Ngoại khóa (UEH, 2026), đề tài 14. Hệ thống tạo tên sản phẩm, mô tả meta, mô tả sản phẩm, kịch bản chốt sale và ba tiêu đề email từ từ khóa đặc trưng trên dữ liệu thời trang nữ của Tiki.

- Tác giả: Nguyễn Quang Minh. Giảng viên: TS. Võ Văn Hải.
- Website demo: https://genai-thoitrangnu.streamlit.app (chạy tại máy: `streamlit run webapp/app.py`; thêm `?demo=1` để xem ví dụ dựng sẵn)

## Kết quả trên mẫu kiểm tra độc lập

Hai giám khảo độc lập đọc tên và mô tả của cùng một sản phẩm mà không biết nguồn của từng bản. Mỗi cấu hình được so riêng với trang người bán. Tỷ lệ thắng tính trên đủ 120 sản phẩm, gồm cả các cặp hòa.

| Cách viết | Giám khảo 1: thắng / hòa / thua | Giám khảo 2: thắng / hòa / thua |
|---|---|---|
| M2: viết từ bản tóm tắt | 37 / 11 / 72 (30,8% thắng) | 44 / 35 / 41 (36,7% thắng) |
| M3: thêm chi tiết của người bán, hướng dẫn phối đồ và yêu cầu tránh câu chung chung | 89 / 14 / 17 (74,2% thắng) | 85 / 25 / 10 (70,8% thắng) |

Website dùng M3 để tạo bản nháp từ bảng cỡ, chất liệu, cách giặt hoặc các chi tiết đã có. Người bán đọc lại dữ kiện và cam kết trước khi đăng.

Trên mẫu độc lập, tên người bán khớp cụm Google Trends ở 28 sản phẩm và tên M3 ở 21. Ở câu mở đầu, M3 khớp ở 21 sản phẩm và người bán ở 17. Phép đo đếm cụm khớp nguyên văn. Nghiên cứu chưa đo thứ hạng, lượt nhấp hay đơn hàng.

## Cấu trúc thư mục

| Thư mục, tệp | Nội dung |
|---|---|
| `src/crawl_tiki_v2.py`, `src/prepare_v2.py` | Thu thập theo khung lấy mẫu, hồ sơ dữ liệu, làm sạch, bản tóm tắt, chia tập theo nhóm |
| `src/prompts_v2.py`, `src/generate_v2.py` | Quy tắc viết M1, M2 và hai phép thử phụ; sinh nội dung có khóa phiên bản |
| `src/style_guide_v2.py`, `src/prompts_m3.py`, `src/generate_m3.py` | Bảng phối đồ theo loại hàng, quy tắc M3 và bước sinh nội dung |
| `src/finetune_v2.py` | Tinh chỉnh LoRA mô hình nhỏ (M4) |
| `src/judge_pairwise_v2.py`, `src/eval_v2.py`, `src/eval_m3.py`, `src/factcheck_v2.py`, `src/human_eval_v2.py` | Chấm mù theo cặp, tổng hợp số liệu, câu chung chung và phối đồ sai, kiểm tra dữ kiện, người chấm |
| `src/paper_v2_assets.py` | Tổng hợp bảng kết quả từ dữ liệu nghiên cứu |
| `webapp/` | Website Streamlit cho người bán |
| `tests/` | Kiểm thử |

## Cách chạy

```bash
pip install -r requirements.txt
python src/crawl_tiki_v2.py --per-sort 35 --out-dir data_v2
python src/prepare_v2.py
python src/generate_v2.py --split test --methods M0,M1,M2
python src/style_guide_v2.py
python src/generate_m3.py --split test
python src/eval_v2.py --split test
python src/eval_m3.py --split test
streamlit run webapp/app.py
```

Thử nhanh một sản phẩm với mô hình nhỏ M4a, M4b (cần `models/Qwen2.5-0.5B-Instruct` và adapter `models_v2/qwen_lora` trên máy) và so với M3:

```bash
python src/try_m4.py --demo --m3
python src/try_m4.py            # tự nhập sản phẩm
```

Tập kiểm tra chỉ chạy khi quy tắc viết khớp phiên bản đã cố định (`results_v2/prompt_lock.json`, `results_v2/prompt_lock_m3.json`). Cấu hình vận hành của website được giữ ngoài mã nguồn.

## Dữ liệu và đạo đức

Tiki, danh mục Thời trang nữ, thu thập qua API công khai ngày 22/11/2026, gọi chậm 1,2 đến 2 giây mỗi yêu cầu. Đánh giá chỉ giữ nội dung, số sao và thời điểm, không giữ thông tin người đánh giá. Kho mã nguồn không chứa mô tả hay đánh giá gốc của người bán; dữ liệu đầy đủ nằm trong thư mục `Data Source` của bài nộp.

## Giấy phép

Mã nguồn: MIT. Dữ liệu Tiki không được phân phối lại.

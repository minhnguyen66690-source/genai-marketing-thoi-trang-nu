# GenAI sinh nội dung marketing cho thời trang nữ

Mã nguồn của bài thu hoạch cuối kỳ môn Chuyên đề Ngoại khóa (UEH, 2026): hệ thống sinh **mô tả sản phẩm chuẩn tối ưu hóa tìm kiếm, kịch bản chốt sale và tiêu đề email marketing** từ vài từ khóa đặc trưng, kèm bộ chỉ số đánh giá tự động mà mỗi ngưỡng đều có nguồn công bố.

- Tác giả: Nguyễn Quang Minh. Giảng viên: TS. Võ Văn Hải.
- Website demo: {{WEB_URL}} (chạy local: `streamlit run webapp/app.py`; thêm `?demo=1` để xem ví dụ có sẵn)
- Notebook Colab: `notebook/GenAI_Marketing_ThoiTrangNu.ipynb`

## Kết quả chính (tập kiểm tra 99 sản phẩm Tiki)

| Cấu hình | Tuân thủ checklist tối ưu hóa tìm kiếm | chrF++ | Xử lý phản đối (giám khảo, 1–5) | Thuộc tính không có căn cứ |
|---|---|---|---|---|
| M0 Mẫu cố định | 58,4% | 13,4 | 1,21 | 0,4% |
| M1 Zero-shot (GPT) | 59,0% | 14,1 | 4,74 | 2,7% |
| M2 Prompt engineering (GPT) | 81,0% | 19,2 | 5,00 | 4,9% |
| M3 Few-shot có truy hồi (GPT) | 85,4% | 19,7 | 5,00 | 4,0% |
| Nội dung người bán (đối chứng) | 64,6% | – | – | – |

Số liệu đầy đủ trong `results/public/` (eval_summary_tiki.json, eval_summary_amazon.json). Bảng trên lấy từ `results/public/eval_summary_tiki.json`.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `src/` | Thu thập dữ liệu (`crawl_tiki.py`, `crawl_tiki_reviews.py`, `fetch_amazon2023.py`), tiền xử lý, trích từ khóa (`keywords.py`), EDA, dự báo bán chạy (`predictive.py`, `rq1_inference.py`), prompt (`prompts.py`), sinh nội dung (`generate.py`), tinh chỉnh LoRA (`finetune_lora.py`), chỉ số đánh giá (`metrics_content.py`), giám khảo LLM (`judge.py`), đánh giá (`evaluate.py`) |
| `webapp/` | Website Streamlit: sinh nội dung, chấm điểm nội dung có sẵn, kết quả nghiên cứu |
| `notebook/` | Notebook chạy được trên Google Colab |
| `data/public/` | Dữ liệu công khai được: đặc trưng số của sản phẩm Tiki (không có tên, mô tả, đánh giá gốc), từ khóa, chủ đề phản đối |
| `results/public/` | Kết quả đánh giá và nội dung do mô hình sinh (đã bỏ văn bản tham chiếu gốc) |
| `figures/` | Toàn bộ biểu đồ và sơ đồ |
| `tests/` | Kiểm thử cho bộ chỉ số |

## Cách chạy

```bash
pip install -r requirements-colab.txt
python src/crawl_tiki.py --smoke            # thử thu thập 5 sản phẩm (tôn trọng giới hạn 1 request/giây)
python src/fetch_amazon2023.py --n 1000     # tải tập con Amazon Reviews 2023
python src/run_analysis.py                  # tiền xử lý, từ khóa, EDA, dự báo
python src/generate.py --dataset amazon --methods M0,M2 --n-test 10   # cần OPENAI_API_KEY hoặc Codex CLI
python src/evaluate.py --dataset amazon --no-judge
streamlit run webapp/app.py
```

Website và notebook đọc khóa API từ biến môi trường hoặc Streamlit Secrets (`GEMINI_API_KEY` hoặc `OPENAI_API_KEY`). Không đưa khóa vào mã nguồn.

## Dữ liệu và đạo đức

- Tiki: thu thập qua API công khai ngày 13/11/2026, tối đa 1 request/giây, chỉ giữ sản phẩm đăng từ 20/11/2019. Việc thu thập dừng hẳn khi API trả trang chống truy cập tự động. Đánh giá chỉ giữ nội dung, số sao và thời điểm, không giữ tên, mã hay ảnh khách hàng. Để tôn trọng quyền đối với nội dung của người bán, kho này không chứa văn bản mô tả hay đánh giá gốc; chạy lại `src/crawl_tiki.py` để tự thu thập.
- Amazon Reviews 2023 (Hou và cộng sự, 2024): tải trực tiếp từ nguồn công bố của McAuley Lab.

## Công cụ AI đã dùng

Claude Opus 5.5 (điều phối, viết, kiểm chứng), GPT-6.1 qua Codex CLI (tra cứu, viết mã, sinh nội dung M1–M3), Gemini 3.8 Flash (giám khảo), Qwen2.5-0.5B-Instruct (tinh chỉnh LoRA). Mọi số liệu được tính bằng mã nguồn trong kho này.

## Giấy phép

Mã nguồn: MIT. Dữ liệu Amazon Reviews 2023 theo điều kiện của McAuley Lab; dữ liệu Tiki không được phân phối lại.

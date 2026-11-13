# Demo nội dung thương mại điện tử

Từ thư mục gốc, cài các thư viện trong `requirements.txt`, rồi chạy:

```bash
streamlit run webapp/app.py
```

Tạo khóa Gemini của riêng bạn tại [Google AI Studio](https://aistudio.google.com/app/apikey); xem hạn mức và giá hiện hành trước khi sử dụng. Khi triển khai Streamlit Community Cloud, mở phần Secrets của ứng dụng và thêm `GEMINI_API_KEY` theo định dạng TOML. Không đưa khóa vào GitHub. Bạn cũng có thể nhập khóa trực tiếp trong thanh bên. Với OpenAI, nhập tên mô hình hoặc đặt `OPENAI_MODEL`.

Để tạo lại dữ liệu tổng hợp, chạy `python webapp/data/build_web_data.py` từ thư mục gốc. Nội dung AI cần người kiểm duyệt trước khi sử dụng.

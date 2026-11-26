# Website soạn nội dung sản phẩm thời trang nữ

Người bán chọn loại hàng, từ khóa, nhập thông tin thực tế như bảng cỡ, chất liệu, độ dày và cách giặt, rồi chọn những điều khách thường quan tâm. Trang tạo bản nháp theo hướng dẫn phối đồ của nhóm hàng, kiểm tra câu chung chung và các chi tiết cần duyệt trước khi đăng.

Chạy tại máy, từ thư mục gốc:

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

Thêm `?demo=1` vào địa chỉ để xem ví dụ dựng sẵn (chân váy chữ A, có bảng size).

Dữ liệu tổng hợp của trang, gồm loại hàng, từ khóa gợi ý, lo ngại theo loại hàng và mức trung vị cùng loại, được tạo bằng `python webapp/data/build_web_data_v2.py`. Người bán cần đọc lại và duyệt bản nháp trước khi đăng.

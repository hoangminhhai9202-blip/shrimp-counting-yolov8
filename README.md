# Đếm tôm nhỏ bằng YOLOv8

> **Lưu ý:** Đây là phiên bản demo/portfolio của dự án đếm tôm bằng YOLOv8.  
> Source code, dữ liệu huấn luyện và model trong repo này chỉ dùng để minh họa quy trình xử lý ảnh, không chứa dữ liệu nội bộ hoặc tài nguyên bảo mật của công ty thực tập.

Đây là dự án thị giác máy tính sử dụng YOLOv8 để nhận diện và đếm số lượng tôm nhỏ từ hình ảnh.

## Mục tiêu dự án

- Nhận diện tôm nhỏ trong ảnh bằng mô hình YOLOv8 đã huấn luyện.
- Đếm tổng số lượng tôm xuất hiện trong ảnh.
- Hiển thị bounding box quanh từng con tôm.
- Hiển thị thời gian xử lý ảnh.
- Tạo giao diện GUI để chọn ảnh và xem kết quả trực quan.

## Công nghệ sử dụng

- Python
- YOLOv8
- OpenCV
- Tkinter
- NumPy
- Pillow

## File cần có

```txt
dem_tom_nho.py
nano.pt
requirements.txt

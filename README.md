# Đếm tôm nhỏ bằng YOLOv8

> **Lưu ý:** Đây là phiên bản demo/portfolio được đơn giản hóa từ ý tưởng dự án thực tế.  
> Repo này tập trung trình bày luồng xử lý chính, giao diện và cách ứng dụng YOLOv8 vào bài toán đếm tôm nhỏ.  
> Độ chính xác và mức độ tối ưu có thể chưa tương đương với phiên bản triển khai thực tế.

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

## Cấu trúc dự án

```txt
shrimp-counting-yolov8/
├── dem_tom_nho.py
├── nano.pt
├── requirements.txt
├── README.md
├── sample_images/
└── demo_results/

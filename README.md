# Đếm tôm nhỏ bằng YOLOv8

> **Lưu ý:** Đây là phiên bản demo/portfolio được đơn giản hóa từ ý tưởng dự án thực tế.  
> Repo này tập trung trình bày luồng xử lý chính, giao diện và cách ứng dụng YOLOv8 vào bài toán đếm tôm nhỏ.  
> Repo này không chứa mã nguồn nội bộ, dữ liệu riêng, thông tin khách hàng hoặc mô hình production thuộc sở hữu công ty.  
> File `nano.pt` trong repo chỉ là model demo, không phải model chính thức/production.  
> Độ chính xác và mức độ tối ưu có thể chưa tương đương với phiên bản triển khai thực tế.

## Giới thiệu dự án

Đây là dự án thị giác máy tính sử dụng YOLOv8 để nhận diện và đếm số lượng tôm nhỏ từ hình ảnh.

Ứng dụng cho phép người dùng chọn ảnh đầu vào, chạy mô hình nhận diện tôm, vẽ bounding box quanh từng con tôm được phát hiện, hiển thị tổng số lượng tôm và thời gian xử lý ảnh.

## Mục tiêu dự án

- Nhận diện tôm nhỏ trong ảnh bằng mô hình YOLOv8 đã huấn luyện.
- Đếm tổng số lượng tôm xuất hiện trong ảnh.
- Hiển thị bounding box quanh từng con tôm.
- Hiển thị thời gian xử lý ảnh.
- Tạo giao diện GUI để chọn ảnh và xem kết quả trực quan.

## Vai trò của tôi

Dự án được tôi thực hiện độc lập, bao gồm:

- Chuẩn bị dữ liệu ảnh cho bài toán đếm tôm nhỏ.
- Gán nhãn đối tượng tôm để huấn luyện mô hình nhận diện.
- Huấn luyện và kiểm thử mô hình YOLOv8.
- Xây dựng logic đếm số lượng tôm dựa trên kết quả detection.
- Thiết kế giao diện GUI để chọn ảnh và hiển thị kết quả.
- Đánh giá kết quả thông qua số lượng phát hiện, bounding box và thời gian xử lý.

## Công nghệ sử dụng

- Python
- YOLOv8 / Ultralytics
- OpenCV
- Tkinter
- NumPy
- Pillow

## Tính năng chính

- Chọn ảnh từ máy tính thông qua giao diện GUI.
- Nhận diện tôm nhỏ bằng YOLOv8.
- Đếm tổng số lượng tôm trong ảnh.
- Vẽ bounding box quanh từng đối tượng tôm được phát hiện.
- Hiển thị ảnh kết quả trực quan.
- Hiển thị thời gian xử lý ảnh.
- Lưu hoặc xem kết quả demo sau khi xử lý.

## Cấu trúc dự án

```txt
shrimp-counting-yolov8/
├── dem_tom_nho.py
├── nano.pt              # Model demo, không phải model chính thức/production
├── requirements.txt
├── README.md
├── sample_images/
└── demo_results/

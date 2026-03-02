🚦 Helmet Detection System using YOLOv8
<p align="center"> <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge"> <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge"> </p> <p align="center"> 🚀 Hệ thống phát hiện người đi xe máy không đội mũ bảo hiểm trong thời gian thực<br> Ứng dụng Computer Vision & Deep Learning với YOLOv8 </p>
📌 Tổng quan dự án

Hệ thống sử dụng mô hình YOLOv8 để:

✔ Phát hiện người (person)
✔ Phát hiện mũ bảo hiểm (helmet)
✔ (Tùy chọn) phát hiện xe máy (motorcycle)
✔ Xác định người không đội mũ bảo hiểm (NO_HELMET)
✔ Hiển thị bounding box màu trực quan
✔ Lưu ảnh vi phạm tự động

🎥 Demo hệ thống
<p align="center"> <img src="assets/demo.gif" width="700"> </p>

🟢 Xanh → Có mũ bảo hiểm
🔴 Đỏ → Không đội mũ bảo hiểm
📈 Hiển thị FPS realtime

🧠 Mô hình sử dụng

Model: YOLOv8n

Dataset: 3,835 ảnh

Classes:

person

helmet

no-helmet

📊 Kết quả đánh giá
Metric	Giá trị
mAP@0.5	0.605
Precision	0.624
Recall	0.594

📁 File model:

models/best.pt
🔍 Logic phát hiện NO_HELMET

Detect tất cả person

Trích xuất vùng đầu (35% phía trên bounding box)

Kiểm tra xem có helmet nằm trong vùng này hay không

Nếu không có → gắn nhãn NO_HELMET

Lưu ảnh vi phạm

💾 Lưu ảnh vi phạm

Khi phát hiện người không đội mũ:

outputs/violations/

Tên file chứa timestamp để dễ truy xuất.

🗂 Cấu trúc thư mục
helmet_detection/
│
├── models/
│   └── best.pt
│
├── src/
│   └── detect.py
│
├── outputs/
│   └── violations/
│
├── requirements.txt
└── README.md
⚙️ Cài đặt môi trường
1️⃣ Tạo môi trường ảo
python -m venv .venv
2️⃣ Kích hoạt môi trường

Windows:

.venv\Scripts\activate

MacOS / Linux:

source .venv/bin/activate
3️⃣ Cài đặt thư viện
pip install -r requirements.txt
▶️ Cách chạy chương trình
🔹 Chạy bằng Webcam
python src/detect.py
🔹 Chạy bằng video

Chỉnh trong file detect.py:

source = "video.mp4"
📊 Đánh giá mô hình

Được đánh giá trên tập validation với:

mAP@0.5

Precision

Recall

Confusion Matrix

PR Curve

Có thể cải thiện thêm bằng:

Tăng imgsz

Train thêm epochs

Sử dụng YOLOv8s hoặc YOLOv8m

Augmentation mạnh hơn

👥 Phân công nhóm
Thành viên	Vai trò
Lê Nguyễn Bảo Trân	Dataset & Training (AI Core)
Võ Gia Huy	Detection Logic & Realtime System
Nguyễn Quốc Tường	Demo & Optimization
🎯 Ứng dụng thực tế

Hệ thống camera giao thông

Giám sát khu công nghiệp

Tích hợp vào hệ thống Smart City

Hỗ trợ xử phạt tự động

🏁 Kết luận

Hệ thống có khả năng phát hiện người không đội mũ bảo hiểm trong thời gian thực với độ chính xác ở mức chấp nhận được cho ứng dụng thực tế.

Dự án thể hiện khả năng:

Xây dựng dataset

Huấn luyện mô hình Deep Learning

Thiết kế logic hậu xử lý

Tối ưu hệ thống realtime
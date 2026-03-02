# 🚦 Helmet Detection System using YOLOv8

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Architecture-Dual--Model-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

<p align="center">
  🚀 Hệ thống phát hiện người đi xe máy <b>không đội mũ bảo hiểm</b> theo thời gian thực <br>
</p>

---
## 👥 Phân công nhóm

| Thành viên | Vai trò |
|------------|----------|
| **Lê Nguyễn Bảo Trân** | Dataset & Model Training |
| **Võ Gia Huy** | Detection Logic & Realtime System |
| **Nguyễn Quốc Tường** | Demo & Optimization |

---

## 📌 Tổng quan

Hệ thống sử dụng **mô hình hai tầng (Dual-Model Architecture)** nhằm tăng độ chính xác và tính linh hoạt trong phát hiện.

Hệ thống có khả năng:

- 👤 Phát hiện người (person)
- 🏍 Phát hiện xe máy (motorcycle)
- 🪖 Phát hiện mũ bảo hiểm (helmet)
- ❌ Phát hiện không đội mũ (no-helmet)
- 🎯 Xác định người đi xe máy không đội mũ bảo hiểm
- 💾 Lưu ảnh vi phạm tự động

---

## 🧠 Kiến trúc mô hình (Dual-Model Architecture)

### 🔹 Model 1 – Custom Helmet Model

- File: `models/best.pt`
- Huấn luyện trên dataset tự xây dựng
- Classes:
  - `helmet`
  - `no-helmet`
  - `person`

Model này chuyên dùng để phân loại tình trạng đội mũ bảo hiểm.

---

### 🔹 Model 2 – Pretrained COCO Model

- File: `yolov8n.pt`
- Pretrained từ Ultralytics (COCO Dataset)
- Classes sử dụng:
  - `person`
  - `motorcycle`

Model này dùng để phát hiện người và xe máy trong khung hình.

---

## ⚙️ Cơ chế hoạt động

1. Model 2 phát hiện `person` và `motorcycle`
2. Xác định người đang điều khiển xe máy
3. Model 1 kiểm tra vùng đầu của `person`
4. Nếu không phát hiện `helmet` → gắn nhãn **NO_HELMET**
5. Lưu ảnh vi phạm

---

## 📊 Kết quả đánh giá (Model 1)

| Metric | Giá trị |
|--------|----------|
| mAP@0.5 | 0.605 |
| Precision | 0.624 |
| Recall | 0.594 |

Dataset huấn luyện: **3,835 ảnh**

## 🗂 Cấu trúc thư mục

```
helmet_detection/
│
├── models/
│   ├── best.pt
│   └── yolov8n.pt
│
├── src/
│   └── detect.py
│
├── outputs/
│   └── violations/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt môi trường

### 1️⃣ Tạo môi trường ảo

```bash
python -m venv .venv
```

### 2️⃣ Kích hoạt môi trường

Windows:

```bash
.venv\Scripts\activate
```

### 3️⃣ Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## ▶️ Chạy chương trình

```bash
python src/detect.py
```

---

## 🎯 Ứng dụng thực tế

- 🚦 Hệ thống camera giao thông
- 🏭 Giám sát khu công nghiệp
- 🏙 Smart City
- 📋 Hỗ trợ xử phạt tự động

---

## 🏁 Kết luận

Hệ thống áp dụng kiến trúc **Dual-Model** giúp nâng cao độ chính xác trong phát hiện vi phạm đội mũ bảo hiểm.

Dự án thể hiện năng lực:

- Xây dựng dataset tùy chỉnh
- Huấn luyện và tối ưu YOLOv8
- Thiết kế hệ thống hai tầng
- Xây dựng pipeline realtime hoàn chỉnh
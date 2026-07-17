# 🚀 ObjectDetections

> **Deteksi Objek Real-Time Menggunakan YOLOS (Vision Transformer), OpenCV, dan EasyOCR**

ObjectDetections merupakan aplikasi berbasis **Python** yang dikembangkan untuk melakukan **deteksi objek secara real-time** menggunakan model **YOLOS Tiny** dari Hugging Face Transformers. Aplikasi ini mendukung berbagai sumber video seperti **webcam**, **RTSP CCTV**, dan **HLS (.m3u8)**, serta dilengkapi dengan fitur **Automatic License Plate Recognition (ALPR)** menggunakan **EasyOCR**.

---

## ✨ Fitur

* 🎥 Deteksi objek secara real-time
* 📷 Mendukung webcam
* 📹 Mendukung RTSP CCTV
* 🌐 Mendukung streaming HLS (.m3u8)
* 🚗 Deteksi kendaraan
* 👤 Deteksi manusia
* 🐶 Deteksi hewan
* 🇮🇩 Label objek dalam Bahasa Indonesia
* 🔤 Pembacaan plat nomor kendaraan menggunakan EasyOCR
* ⚡ Menggunakan model YOLOS Tiny dari Hugging Face
* 🧠 Berbasis Vision Transformer (ViT)

---

## 🛠️ Tech Stack

* Python
* OpenCV
* PyTorch
* Hugging Face Transformers
* YOLOS Tiny
* EasyOCR
* Pillow
* NumPy

---

## 📂 Struktur Project

```text
ObjectDetections/
│
├── webcam.py
├── rtsp.py
├── hls.py
├── requirements.txt
└── README.md
```

---

## 🎯 Objek yang Didukung

ObjectDetections mampu mendeteksi berbagai objek dari dataset COCO. Implementasi saat ini difokuskan pada objek berikut:

| Label      | Terjemahan |
| ---------- | ---------- |
| Person     | Orang      |
| Car        | Mobil      |
| Motorcycle | Motor      |
| Bus        | Bus        |
| Truck      | Truk       |
| Bicycle    | Sepeda     |
| Cat        | Kucing     |
| Dog        | Anjing     |

> Model **YOLOS Tiny** mendukung hingga **80 kelas objek**.

---

## ⚙️ Instalasi

Clone repository

```bash
git clone https://github.com/ade-m/ObjectDetections.git

cd ObjectDetections
```

Install dependency

```bash
pip install -r requirements.txt
```

Atau install secara manual

```bash
pip install torch torchvision transformers opencv-python pillow easyocr numpy
```

---

## 🚀 Menjalankan Aplikasi

### 📷 Webcam

```bash
python webcam.py
```

---

### 📹 RTSP CCTV

Ubah alamat RTSP sesuai kamera yang digunakan.

```python
detect_from_rtsp(
    "rtsp://username:password@IP_ADDRESS:PORT/Streaming/Channels/102"
)
```

Kemudian jalankan

```bash
python rtsp.py
```

---

### 🌐 HLS (.m3u8)

Masukkan alamat stream HLS.

```python
detect_from_hls(
    "https://alamat-stream.m3u8"
)
```

Kemudian jalankan

```bash
python hls.py
```

---

## 🚘 Alur Deteksi Plat Nomor

Saat kendaraan berhasil dideteksi, sistem akan melakukan pencarian area plat nomor menggunakan teknik pengolahan citra sebelum membaca teks menggunakan EasyOCR.

```text
Video
   │
   ▼
Deteksi Objek (YOLOS)
   │
   ▼
Crop Kendaraan
   │
   ▼
Grayscale
   │
   ▼
Histogram Equalization
   │
   ▼
Canny Edge Detection
   │
   ▼
Contour Detection
   │
   ▼
Seleksi Plat Nomor
   │
   ▼
EasyOCR
   │
   ▼
Hasil OCR
```

---

## ⚙️ Konfigurasi

| Parameter            | Nilai |
| -------------------- | ----: |
| Confidence Threshold |  0.85 |
| Display Threshold    |  0.90 |

Nilai tersebut dapat diubah sesuai kebutuhan.

---

## 📡 Sumber Video yang Didukung

| Sumber      | Status |
| ----------- | :----: |
| Webcam      |    ✅   |
| Kamera USB  |    ✅   |
| RTSP CCTV   |    ✅   |
| IP Camera   |    ✅   |
| DVR / NVR   |    ✅   |
| HLS (.m3u8) |    ✅   |

---

## 🗺️ Roadmap

* [ ] Dukungan YOLOv11
* [ ] Object Tracking (ByteTrack / DeepSORT)
* [ ] Optimasi GPU
* [ ] Multi Camera Detection
* [ ] Vehicle Counting
* [ ] Speed Estimation
* [ ] Dashboard berbasis Web
* [ ] REST API
* [ ] Docker Support
* [ ] Penyimpanan hasil deteksi ke database

---

## 🤝 Kontribusi

Kontribusi dalam bentuk **Issue**, **Pull Request**, maupun usulan fitur baru sangat terbuka untuk membantu pengembangan project ini.

---

## 👨‍💻 Author

**Ade Maulana**

* 🌐 GitHub: https://github.com/ade-m
* 📸 Instagram: https://instagram.com/ademaulana_

---

## ⭐ Support

Apabila **project** ini bermanfaat, jangan lupa berikan **⭐ Star** pada repository ini agar dapat membantu pengembangan ke depannya.

---

<div align="center">

**Dibangun dengan ❤️ menggunakan Python, OpenCV, Hugging Face Transformers, dan EasyOCR.**

</div>

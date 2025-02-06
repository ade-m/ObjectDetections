from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np
import easyocr

# Memuat model YOLO dan image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
reader = easyocr.Reader(['id'])  # Untuk membaca plat nomor dalam Bahasa Indonesia

# Kamus untuk menerjemahkan label ke bahasa Indonesia
label_translation = {
    'bicycle': 'sepeda',
    'car': 'mobil',
    'motorcycle': 'motor',
    'bus': 'bus',
    'truck': 'truk',
    'person': 'Orang',
    'human': 'Manusia',
    'cat': 'Kucing',
    'dog': 'Anjing'
}

def deteksi_plat(frame, vehicle_area):
    """Mendeteksi plat nomor kendaraan pada frame yang diberikan."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(enhanced_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        plate_area = w * h

        if plate_area <= vehicle_area / 10:
            aspect_ratio = w / h
            if 2 < aspect_ratio < 5:  # Rasio plat nomor umumnya antara 2:1 hingga 5:1
                plate_region = frame[y:y+h, x:x+w]
                result = reader.readtext(plate_region)

                if result:
                    plat_text = " ".join([detection[1] for detection in result])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, plat_text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def detect_from_hls(hls_url):
    """Mendeteksi objek dan plat nomor dari stream HLS (m3u8)."""
    cap = cv2.VideoCapture(hls_url)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka stream CCTV.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak bisa membaca frame.")
            break

        # Ubah frame ke format PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.8, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            if score > 0.85 and model.config.id2label[label.item()] in ('car', 'motorcycle', 'truck', 'bus', 'person', 'human', 'cat', 'dog'):
                x1, y1, x2, y2 = map(int, box)
                vehicle_area = (x2 - x1) * (y2 - y1)
                cropped_frame = frame[y1:y2, x1:x2]

                # Kirimkan gambar kendaraan ke fungsi deteksi plat nomor
               # cropped_frame = deteksi_plat(cropped_frame, vehicle_area)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label_translation.get(model.config.id2label[label.item()], label.item())}: {round(score.item(), 3)}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Tampilkan hasil deteksi di layar
        cv2.imshow('YOLO Object Detection - CCTV Stream', frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Masukkan URL CCTV m3u8 di sini
detect_from_hls("https://atcsdishub.pemkomedan.go.id/camera/SM.RAJAAMALIUN.m3u8")

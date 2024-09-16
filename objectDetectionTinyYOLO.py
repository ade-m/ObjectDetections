from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np

# Memuat model YOLO dan image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Kamus untuk menerjemahkan label ke bahasa Indonesia
label_translation = {
    'person': 'manusia',
    'bicycle': 'sepeda',
    'car': 'mobil',
    'motorcycle': 'motor',
    'airplane': 'pesawat',
    'bus': 'bus',
    'train': 'kereta',
    'truck': 'truk',
    'boat': 'kapal',
    'traffic light': 'lampu lalu lintas',
    'fire hydrant': 'hidran kebakaran',
    'stop sign': 'rambu stop',
    'parking meter': 'meteran parkir',
    'bench': 'bangku',
    'bird': 'burung',
    'cat': 'kucing',
    'dog': 'anjing',
    'horse': 'kuda',
    'sheep': 'domba',
    'cow': 'sapi',
    'elephant': 'gajah',
    'bear': 'beruang',
    'zebra': 'zebra',
    'giraffe': 'jerapah',
    'backpack': 'ransel',
    'umbrella': 'payung',
    'handbag': 'tas tangan',
    'tie': 'dasi',
    'suitcase': 'koper',
    'frisbee': 'frisbee',
    'skis': 'ski',
    'snowboard': 'snowboard',
    'sports ball': 'bola olahraga',
    'kite': 'layang-layang',
    'baseball bat': 'tongkat baseball',
    'baseball glove': 'sarung tangan baseball',
    'skateboard': 'skateboard',
    'surfboard': 'papan selancar',
    'tennis racket': 'raket tenis',
    'bottle': 'botol',
    'wine glass': 'gelas anggur',
    'cup': 'cangkir',
    'fork': 'garpu',
    'knife': 'pisau',
    'spoon': 'sendok',
    'bowl': 'mangkuk',
    'banana': 'pisang',
    'apple': 'apel',
    'sandwich': 'sandwich',
    'orange': 'jeruk',
    'broccoli': 'brokoli',
    'carrot': 'wortel',
    'hot dog': 'hot dog',
    'pizza': 'pizza',
    'donut': 'donat',
    'cake': 'kue',
    'chair': 'kursi',
    'couch': 'sofa',
    'potted plant': 'tanaman pot',
    'bed': 'tempat tidur',
    'dining table': 'meja makan',
    'toilet': 'toilet',
    'tv': 'TV',
    'laptop': 'laptop',
    'mouse': 'mouse',
    'remote': 'remote',
    'keyboard': 'keyboard',
    'cell phone': 'ponsel',
    'microwave': 'microwave',
    'oven': 'oven',
    'toaster': 'pemanggang roti',
    'sink': 'wastafel',
    'refrigerator': 'kulkas',
    'book': 'buku',
    'clock': 'jam',
    'vase': 'vas',
    'scissors': 'gunting',
    'teddy bear': 'boneka beruang',
    'hair drier': 'pengering rambut',
    'toothbrush': 'sikat gigi'
}

# Fungsi untuk menangkap gambar dari webcam dan melakukan deteksi objek
def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    # Atur lebar frame
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)

    # Atur tinggi frame
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Konversi frame ke format PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Proses gambar dengan YOLO model
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Prediksi bounding boxes dan kelas
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.8, target_sizes=target_sizes)[0]

        # Gambar hasil deteksi pada frame
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            if score > 0.9:
                x1, y1, x2, y2 = map(int, box)
                translated_label = label_translation.get(
                    model.config.id2label[label.item()], model.config.id2label[label.item()])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{translated_label}: {round(score.item(), 3)}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Tampilkan frame dengan deteksi objek
        cv2.imshow('YOLO Object Detection', frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Melepaskan video capture dan menutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

# Menjalankan fungsi untuk mendeteksi objek dari webcam
detect_from_webcam()

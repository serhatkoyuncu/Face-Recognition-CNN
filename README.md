# 👁️ Face Recognition System (DeepFace + OpenCV)

Gerçek zamanlı kamera üzerinden yüz tanıma yapan, embedding tabanlı face recognition sistemi.

---

## 🚀 Özellikler

- 🎯 Gerçek zamanlı webcam yüz tanıma
- 🧠 CNN tabanlı embedding (ArcFace / FaceNet)
- 📦 Dataset tabanlı kişi tanıma
- 📊 Cosine similarity ile karşılaştırma
- 🟥 Unknown yüzler için kırmızı bounding box
- 🟩 Tanınan kişiler için yeşil bounding box
- ⚡ FPS göstergesi
- 🧾 Confidence score bar
- 🧠 Basit tracking (IoU tabanlı)

---

## 🏗️ Kullanılan Teknolojiler

- Python
- OpenCV
- DeepFace
- TensorFlow
- NumPy

---

## 📁 Proje Yapısı

project/
│
├── src/
│ ├── realtime.py
│ ├── utils.py
│ └── config.py
│
├── dataset/
│ ├── person1/
│ ├── person2/
│ └── person3/
│
├── requirements.txt
├── .gitignore
└── README.md

---

## ⚙️ Kurulum

### 1. Repository clone

```bash
git clone <repo-url>
cd project
```

### 2. Virtual environment oluştur

```bash
python3.10 -m venv face-env
source face-env/bin/activate
```

### 3. Paketleri yükle

```bash
pip install -r requirements.txt
```

### 4. Dataset Hazırlama

dataset/ klasörü içine her kişi için ayrı klasör oluştur:

dataset/
├── Serhat/
│ ├── 1.jpg
│ ├── 2.jpg
│
├── Ali/
│ ├── 1.jpg
│ ├── 2.jpg

### 5. Çalıştırma

```bash
python src/realtime.py
```

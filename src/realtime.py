import cv2
import numpy as np
import time
import os
from deepface import DeepFace

# CONFIG
MODEL_NAME = "ArcFace"
THRESHOLD = 0.5
DATASET_PATH = "dataset"

# ======================
# LOAD DATASET
# ======================
db = {}

def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v)

print("Embedding oluşturuluyor...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    db[person] = []

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        try:
            emb = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]

            db[person].append(normalize(emb))

        except Exception as e:
            print("Hata:", img_path)


# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# TRACKING CACHE
track_cache = {}

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        faces = DeepFace.extract_faces(
            frame,
            detector_backend="opencv",
            enforce_detection=False
        )

        new_cache = {}

        for face_obj in faces:
            region = face_obj["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            face_crop = frame[y:y+h, x:x+w]

            if face_crop.size == 0:
                continue

            # Resize stabilize
            face_crop = cv2.resize(face_crop, (224, 224))

            # ======================
            # EMBEDDING
            # ======================
            emb = DeepFace.represent(
                img_path=face_crop,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]

            emb = normalize(emb)

            name = "Unknown"
            best_score = 0

            # ======================
            # COMPARE
            # ======================
            for person, embeddings in db.items():
                for e in embeddings:
                    score = np.dot(emb, e)

                    if score > best_score:
                        best_score = score
                        name = person

            if best_score < THRESHOLD:
                name = "Unknown"

            # ======================
            # DRAW
            # ======================
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            label = f"{name} ({best_score:.2f})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Confidence bar
            bar_w = int(w * min(best_score, 1.0))
            cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+10), (200,200,200), 1)
            cv2.rectangle(frame, (x, y+h+5), (x+bar_w, y+h+10), color, -1)


        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    except Exception as e:
        print("Hata:", e)

    cv2.imshow("Face Recognition FINAL", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
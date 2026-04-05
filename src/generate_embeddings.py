import os
import pickle
from deepface import DeepFace
from config import DATASET_PATH, EMBEDDINGS_PATH, MODEL_NAME

db = {}

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    db[person] = []

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend="mtcnn",
                enforce_detection=True
            )[0]["embedding"]

            db[person].append(embedding)

        except Exception as e:
            print(f"Error: {img_path}", e)

# Save DB
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(db, f)

print("Embeddings created.")
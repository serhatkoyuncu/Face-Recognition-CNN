import pickle
import os
from deepface import DeepFace
from utils import cosine_similarity
from config import EMBEDDINGS_PATH, MODEL_NAME, THRESHOLD

with open(EMBEDDINGS_PATH, "rb") as f:
    db = pickle.load(f)

def recognize(img_path):
    query = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        enforce_detection=False
    )[0]["embedding"]

    best_match = "Unknown"
    best_score = 0

    for person, embeddings in db.items():
        for emb in embeddings:
            score = cosine_similarity(query, emb)

            if score > best_score:
                best_score = score
                best_match = person

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_match, best_score



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, "test.jpg")



if __name__ == "__main__":
    person, score = recognize(img_path)
    print(person, score)
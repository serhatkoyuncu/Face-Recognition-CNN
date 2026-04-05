import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "embeddings.pkl")

MODEL_NAME = "ArcFace"
THRESHOLD = 0.6
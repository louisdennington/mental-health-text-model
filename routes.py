from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from collections import Counter
from config import FAISS_INDEX_PATH, K_NEIGHBORS
from logger import logger

router = APIRouter()

# ----------------------
# Load model and data
# ----------------------
logger.info("🔁 Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(FAISS_INDEX_PATH)

with open("id_to_label.json", "r") as f:
    id_to_label = json.load(f)

with open("index_map.json", "r") as f:
    index_map = json.load(f)

index_to_label = [id_to_label[i] for i in index_map]

# ----------------------
# Helper: Response Lookup
# ----------------------
def get_cluster_response(cluster_id):
    responses = {
        "13": "Disorganized Attachment and Self-Destructive Coping: You may feel emotionally overwhelmed, struggle with impulsivity, or long for connection while fearing it at the same time.",
        "3": "Somatic Anxiety: You may experience a breakdown in your internal sense of safety, with physical symptoms overwhelming rational thought.",
        "0": "Small victories matter. Your reflection shows you're learning to care for yourself amid the chaos.",
        # Add more cluster responses here...
    }
    return responses.get(cluster_id, "This cluster hasn't been fully annotated yet. Thank you for contributing to its training.")

# ----------------------
# Request Schema
# ----------------------
class UserInput(BaseModel):
    text: str

# ----------------------
# Prediction Endpoint
# ----------------------
@router.post("/predict")
async def predict(user_input: UserInput):
    text = user_input.text.strip()
    if len(text.split()) < 50:
        logger.warning("Rejected input: fewer than 50 words.")
        return {"error": "Input must be at least 50 words."}

    embedding = model.encode([text], convert_to_numpy=True)
    k = K_NEIGHBORS
    _, indices = index.search(embedding, k)
    neighbor_labels = [index_to_label[i] for i in indices[0]]

    majority_label, count = Counter(neighbor_labels).most_common(1)[0]
    certainty = round(count / k, 2)

    logger.info(f"Prediction: Cluster {majority_label} with certainty {certainty}")

    return {
        "cluster": majority_label,
        "certainty": certainty,
        "response": get_cluster_response(majority_label)
    }
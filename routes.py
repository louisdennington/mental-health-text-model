from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
import datetime
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

with open("app/responses.json", "r", encoding="utf-8") as f:
    PROMPT_LOOKUP = json.load(f)

# ----------------------
# Helper: Response Lookup
# ----------------------
def get_cluster_response(cluster_id):
    return PROMPT_LOOKUP.get(str(cluster_id), "This cluster hasn't been fully annotated yet. Thank you for contributing to its training.")

CLUSTER_LABELS = {
    "0": "Struggles and victories with self-care",
    "1": "Self-harm and strong emotions",
    "2": "Struggles with medication",
    "3": "Experiences of anxiety",
    "4": "Anxieties about being seen or judged",
    "5": "Frustrations with being invalidated, misrepresented or misunderstood",
    "6": "System fatigue and loss of hope",
    "7": "Night drift: Sleep as escape, day as burden",
    "8": "Cognitive fog and self-erosion",
    "9": "Disordered thoughts and dissociation",
    "10": "Cycles of emotional instability and identity confusion",
    "11": "Existential confusion and obsessive fears",
    "12": "Still functioning but emotionally exhausted",
    "13": "Push and pull in relationships and coping by destroying",
    "14": "Suicidal feelings and wishing not to exist",
    "15": "Moments that saved me",
    "20": "What is wrong with me?"
}

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
    label_name = CLUSTER_LABELS.get(str(majority_label), "Unknown")

    logger.info(f"Prediction: Cluster {majority_label} ({label_name}) with certainty {certainty}")

    return {
        "cluster": majority_label,
        "certainty": certainty,
        "response": get_cluster_response(majority_label),
        "label_name": label_name
    }

# Function for collecting feedback

@router.post("/feedback")
async def receive_feedback(request: Request):
    data = await request.json()
    rating = data.get("rating")
    feedback = data.get("feedback")
    timestamp = datetime.datetime.utcnow().isoformat()

    with open("feedback_log.csv", "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{rating},{feedback.replace(',', ';')}\n")

    return {"message": "Thank you for your feedback"}

# Function for getting UMAP embeddings into html for graphical representation of feature space

@router.get("/umap")
async def get_umap_data():
    try:
        with open(os.path.join("app", "umap_data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
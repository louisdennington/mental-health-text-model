import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load cleaned dataset
input_path = "data/processed/cleaned_posts.json"
output_embedding_path = "data/processed/post_embeddings.npy"
output_meta_path = "data/processed/post_metadata.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [post["text"] for post in data]

# Load SBERT model (explicitly named)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Generate embeddings
print(f"🔍 Encoding {len(texts)} posts using {model_name}...")
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings as .npy for clustering
os.makedirs(os.path.dirname(output_embedding_path), exist_ok=True)
np.save(output_embedding_path, embeddings)

# Save metadata for tracking
metadata = [{"id": post["id"], "subreddit": post["subreddit"], "category": post["category"]} for post in data]

with open(output_meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Embeddings saved to: {output_embedding_path}")
print(f"🧾 Metadata saved to: {output_meta_path}")
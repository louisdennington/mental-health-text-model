import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import os
from datetime import datetime

# Load components
print("Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("cluster_index.faiss")

with open("id_to_label.json", "r") as f:
    id_to_label = json.load(f)

with open("index_map.json", "r") as f:
    index_map = json.load(f)

# Ensure mapping is in correct order
index_to_label = [id_to_label[i] for i in index_map]

def predict_cluster(text, k=6):
    embedding = model.encode([text], convert_to_numpy=True)
    distances, indices = index.search(embedding, k)
    neighbor_labels = [index_to_label[i] for i in indices[0]]
    majority, count = Counter(neighbor_labels).most_common(1)[0]
    certainty = count / k  # proportion of neighbors that agreed
    return majority, certainty

# Save feedback to a local JSON log
FEEDBACK_PATH = "user_feedback_log.json"

def save_feedback(text, cluster, certainty, rating, comment):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "input_text": text,
        "predicted_cluster": cluster,
        "certainty": certainty,
        "rating": rating,
        "user_comment": comment
    }
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(entry)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(logs, f, indent=2)

# Example CLI usage (to be replaced by web interface)
if __name__ == "__main__":
    while True:
        user_input = input("\n📝 Write about your difficulties (min 50 words):\n")
        word_count = len(user_input.split())
        if word_count < 50:
            print("Please write at least 50 words.")
            continue

        cluster, certainty = predict_cluster(user_input)
        print(f"\nYour post was classified as Cluster {cluster} (Certainty: {certainty*100:.1f}%)")

        # Simulate returning a message (replace with actual message bank lookup)
        print(f"\nSuggested response: [Message for Cluster {cluster} would appear here.]")

        # Ask for feedback
        rating = input("\nHow accurate was the response? (1 to 5): ")
        comment = input("Optional feedback:")
        save_feedback(user_input, cluster, certainty, rating, comment)

        next_action = input("\n🔁 Would you like to (e)dit what you wrote, (s)tart again, or (q)uit? ").strip().lower()
        if next_action == "q":
            print("👋 Goodbye!")
            break
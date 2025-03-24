import pandas as pd
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

# Load the dataset
df = pd.read_excel(r"C:\Users\louis\OneDrive - University College London\MSc Health Data Science\0 - Personal projects\mental-health-text-model\data\processed\clustered_posts_labeled.xlsx")

# Filter out unclassifiable posts
df = df[df["primary_allocation"] != 99].reset_index(drop=True)

# Use only primary cluster labels for training
df["final_label"] = df["primary_allocation"].astype(str)

# Load SBERT model and generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
print("🔍 Generating embeddings...")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    df["final_label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["final_label"]
)

# Train FAISS index on training embeddings
dimension = X_train.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X_train)

# Perform kNN (k=3) search for test samples
k = 6
_, indices = index.search(X_test, k)

# Predict labels using majority vote from nearest neighbors
y_pred = []
for idx_set in indices:
    neighbor_labels = [y_train[i] for i in idx_set]
    majority = Counter(neighbor_labels).most_common(1)[0][0]
    y_pred.append(majority)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save FAISS index and mappings (from full dataset)
full_index = faiss.IndexFlatL2(dimension)
full_index.add(embeddings)

faiss.write_index(full_index, "cluster_index.faiss")

id_to_label = dict(zip(df["id"], df["final_label"]))
id_to_text = dict(zip(df["id"], df["text"]))
index_map = list(df["id"])

with open("id_to_label.json", "w") as f:
    json.dump(id_to_label, f)

with open("id_to_text.json", "w") as f:
    json.dump(id_to_text, f)

with open("index_map.json", "w") as f:
    json.dump(index_map, f)

print("✅ Embedding index and label mappings saved.")

# UMAP Visualization of training set
print("\n🎨 Generating UMAP projection (training set)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(X_train)

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=embedding_2d[:, 0],
    y=embedding_2d[:, 1],
    hue=y_train,
    palette="tab20",
    legend='full',
    s=40
)
plt.title("UMAP Projection of Training Embeddings by Primary Cluster")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster")
plt.tight_layout()

# Save plot instead of showing
plt.savefig("umap_training_projection_by_primary_cluster.png", dpi=300)
print("📸 UMAP plot saved as 'umap_training_projection_by_primary_cluster.png'")
plt.close()

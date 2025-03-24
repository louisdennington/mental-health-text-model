# cluster.py - UMAP + HDBSCAN clustering of SBERT embeddings

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import hdbscan
import pandas as pd
from collections import defaultdict
import random

# Load embeddings and metadata
embedding_path = "data/processed/post_embeddings.npy"
metadata_path = "data/processed/post_metadata.json"

embeddings = np.load(embedding_path)
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Reduce dimensions with UMAP
print("🔻 Reducing dimensionality with UMAP...")
umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# Cluster with HDBSCAN
print("🔗 Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
cluster_labels = clusterer.fit_predict(umap_embeddings)

# Count clusters
unique_clusters = set(cluster_labels)
num_clusters = len([c for c in unique_clusters if c != -1])
print(f"✅ Found {num_clusters} clusters (+ noise)")

# Attach labels to metadata
for i, entry in enumerate(metadata):
    entry["cluster"] = int(cluster_labels[i])
    entry["x"] = float(umap_embeddings[i][0])
    entry["y"] = float(umap_embeddings[i][1])

# Save clustered metadata
output_path = "data/processed/clustered_posts.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# Visualize clusters
print("📊 Plotting UMAP projection with cluster labels...")
plt.figure(figsize=(10, 8))
palette = sns.color_palette("hsv", len(set(cluster_labels)))
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
                hue=cluster_labels, palette=palette, legend="full", s=40, alpha=0.7)
plt.title(f"UMAP projection of SBERT embeddings (HDBSCAN clusters = {num_clusters})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.figtext(
    0.5, -0.08,
    "The graph shows a landscape of meaning: each coloured “clump” is a set of posts that express semantically similar content — a latent topic, theme, or pattern in how people talk about mental health.",
    wrap=True, horizontalalignment='center', fontsize=10
)
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("data/processed/umap_clusters.png")

print("✅ Clustered posts saved to:", output_path)
print("🖼️ Cluster plot saved to: data/processed/umap_clusters.png")

# Inspect the posts from each resulting cluster:

# Load full cleaned post texts
with open("data/processed/cleaned_posts.json", "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Load clustering metadata (e.g. output from HDBSCAN or similar)
with open("data/processed/clustered_posts.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Combine into rows
rows = []
for i, entry in enumerate(metadata):
    cluster_id = entry["cluster"]
    subreddit = entry.get("subreddit", "")
    text = full_data[i]["text"]
    post_id = full_data[i]["id"]  # ✅ We can rely on this now
    rows.append({
        "id": post_id,
        "cluster": cluster_id,
        "subreddit": subreddit,
        "text": text,
        "label": ""  # Optional: placeholder for manual labeling
    })

# Save to Excel
output_path = "data/processed/clustered_posts_labeled.xlsx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df = pd.DataFrame(rows)
df.to_excel(output_path, index=False)

print(f"✅ Labeled post export saved to: {output_path}")
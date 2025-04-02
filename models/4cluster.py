import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import hdbscan
import pandas as pd
import itertools

# Load embeddings and metadata
embedding_path = "data/processed/post_embeddings.npy"
metadata_path = "data/processed/post_metadata.json"

embeddings = np.load(embedding_path)
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Define UMAP parameter grid
umap_n_neighbors = [10, 15, 30]
umap_n_components = [2, 5]
umap_min_dist = [0.0, 0.1, 0.3]
umap_random_state = 42

# Define HDBSCAN parameter grid
hdbscan_min_cluster_size = [15, 20, 25]
hdbscan_prediction_data = True

# Conditions to save outputs:
# (a) noise (-1) count below 4000
# (b) number of clusters (excluding noise) between 20 and 60 (inclusive)
runs_meeting_conditions = 0
run_number = 1

# Create output directory if not exists
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Iterate over all combinations of UMAP and HDBSCAN parameters
for n_neighbors, n_components, min_dist in itertools.product(umap_n_neighbors, umap_n_components, umap_min_dist):
    umap_params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_dist": min_dist,
        "random_state": umap_random_state
    }
    for min_cluster_size in hdbscan_min_cluster_size:
        hdbscan_params = {
            "min_cluster_size": min_cluster_size,
            "prediction_data": hdbscan_prediction_data
        }
        print(f"Run {run_number}:")
        print(f"  UMAP parameters: {umap_params}")
        print(f"  HDBSCAN parameters: {hdbscan_params}")
        
        # Dimensionality reduction with UMAP
        umap_model = umap.UMAP(**umap_params)
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        # Clustering with HDBSCAN
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        cluster_labels = clusterer.fit_predict(umap_embeddings)
        
        noise_count = np.sum(cluster_labels == -1)
        unique_clusters = set(cluster_labels)
        num_clusters = len([c for c in unique_clusters if c != -1])
        print(f"  Noise count (-1): {noise_count}, Number of clusters: {num_clusters}")
        
        # Check conditions
        if noise_count < 4000 and 20 <= num_clusters <= 60:
            print(f"  ✅ Conditions met for run {run_number}. Saving outputs.")
            runs_meeting_conditions += 1
            # Attach cluster labels and UMAP coordinates to metadata copy
            run_metadata = []
            for i, entry in enumerate(metadata):
                new_entry = entry.copy()
                new_entry["cluster"] = int(cluster_labels[i])
                new_entry["x"] = float(umap_embeddings[i][0])
                new_entry["y"] = float(umap_embeddings[i][1])
                run_metadata.append(new_entry)
            
            # Save clustered metadata to JSON
            json_filename = os.path.join(output_dir, f"clustered_posts_run_{run_number}_tunedparameters.json")
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(run_metadata, f, ensure_ascii=False, indent=2)
            
            # Generate and save UMAP plot
            plt.figure(figsize=(10, 8))
            palette = sns.color_palette("hsv", len(set(cluster_labels)))
            sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
                            hue=cluster_labels, palette=palette, legend="full", s=40, alpha=0.7)
            plt.title(f"UMAP projection (Run {run_number})\nUMAP: {umap_params} | HDBSCAN: {hdbscan_params}\nClusters = {num_clusters}")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.tight_layout()
            png_filename = os.path.join(output_dir, f"umap_clusters_run_{run_number}_tunedparameters.png")
            plt.savefig(png_filename)
            plt.close()
            
            # Load full cleaned post texts for Excel export
            with open("data/processed/cleaned_posts.json", "r", encoding="utf-8") as f:
                full_data = json.load(f)
            
            # Combine clustering results with original posts for Excel
            rows = []
            for i, entry in enumerate(run_metadata):
                cluster_id = entry["cluster"]
                subreddit = entry.get("subreddit", "")
                text = full_data[i]["text"]
                post_id = full_data[i]["id"]
                rows.append({
                    "id": post_id,
                    "cluster": cluster_id,
                    "subreddit": subreddit,
                    "text": text,
                    "label": ""
                })
            df = pd.DataFrame(rows)
            xlsx_filename = os.path.join(output_dir, f"clustered_posts_labeled_run_{run_number}_tunedparameters.xlsx")
            df.to_excel(xlsx_filename, index=False)
            
            print(f"  Saved JSON: {json_filename}")
            print(f"  Saved PNG: {png_filename}")
            print(f"  Saved Excel: {xlsx_filename}\n{'-'*50}\n")
        else:
            print(f"  ❌ Conditions not met for run {run_number}.\n{'-'*50}\n")
        
        run_number += 1

if runs_meeting_conditions == 0:
    print("No run produced the desired conditions (noise count < 4000 and clusters between 20 and 60).")
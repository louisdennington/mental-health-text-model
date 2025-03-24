# config.py

import os

# Project settings
APP_NAME = "Mental Health Clustering Tool"
ENVIRONMENT = os.getenv("ENV", "development")

# File paths
FAISS_INDEX_PATH = "cluster_index.faiss"
LABELS_JSON = "id_to_label.json"
INDEX_MAP_JSON = "index_map.json"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# kNN
K_NEIGHBORS = 6

# Input requirements
MIN_WORD_COUNT = 50

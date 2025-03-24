# Flask or FastAPI app entry point

from flask import Flask
from app.routes import bp

# Import the app
app = Flask(__name__)
app.register_blueprint(bp)

# Define how it runs (app.run())

# Link routes from routes.py

if __name__ == "__main__":
    app.run(debug=True)


✅ Ways to make kNN practical for a web app:
1. Precompute and store all training embeddings
At app startup (or once during deployment), encode all your posts with SBERT and store them as a NumPy array or in a fast database.

This avoids re-encoding the training set on every request.

2. Use a fast approximate nearest neighbor (ANN) library
Instead of vanilla kNN, use:

FAISS (Facebook AI)

Annoy (Spotify)

ScaNN (Google)

hnswlib (very fast)

These create an indexed embedding space (e.g. using trees or graphs) that lets you find approximate nearest neighbors in milliseconds — even over 100,000+ entries.

🔥 In practice, FAISS or hnswlib with cosine similarity on SBERT vectors = fast, accurate, production-ready kNN-like behavior.

3. Cache popular predictions
If your system ends up seeing a lot of similar input phrases, you can cache the outputs.


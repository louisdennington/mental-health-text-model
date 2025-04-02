# What does it do?
https://www.louisdennington.co.uk/services-3

1scraper.py: Posts are scraped from a selection of Reddit mental health forums using Reddit's API "PRAW" (which determines how Python scripts can interact with public forums to use data). Forums were checked first to see whether they had local rules against scraping (r/mentalhealthUK does, for example). Posts with fewer than 50 characters are excluded. Users have not given explicit consent to their data being used to train a model of the current specifications, but Reddit's general information policy indicates that the data is publicly accessible. 

2prepare_dataset.py: Basic cleaning of the text (lowercasing, removing whitespaces...) to prepare it for vectorising

3vectorise.py: Converts posts to vector embeddings using the pre-trained model all-MiniLM-L6-v2. Older vectorisation processes like TF-IDF or Word2Vec / GloVe	(which are poorer with word order or context) were disregarded in favour of an Sentence-BERT (SBERT) process. No model was found that was specifically fine-tuned for sentence embeddings in mental health data: This could be an avenue for development. 

4cluster.py: Uses UMAP to reduce dimensionality of the clusters (not necessary but can be done on complex text data). It turns the high-dimensional embeddings of SBERT into smaller multidimensional vectors while trying to preserve structure. The script then uses HDBSCAN as an unsupervised learning technique to identify clusters in the post. HDBSCAN was chosen instead of KMeans. KMeans requires estimating the number of clusters in advance and assumes that clusters are roughly round or spherical (often not true of language). HDBSCAN allows posts that can't be clustered to be given the label -1. The script has been through several iterations: Different parameters for UMAP and HDBSCAN produce very different results. The current script cycles through several hyperparameter variations, with the terminal printing those that met two key criteria: (1) a minimal number of posts that are classed as unclusterable (-1) and (2) a reasonable number of clusters (20-60) to allow for sufficient granularity.

Intermediary step: The clusters are examined (with the help of AI) to identify themes, recoded if necessary. Time-consuming, but gives the opportunity to cluster more authentically using domain knowledge.

5train_model.py: For the time being, this uses a Facebook-developed fast version of k-Nearest Neighbours to train a model to assign some inputted text to one of the identified clusters.

NOTE: the pipeline has been re-run up to 4cluster.py but the kNN model not retrained; the model currently online relates to earlier smaller batch scraping of Reddit posts. Things are currently stuck at the intermediary step, trying to relabel a larger corpus of example posts from a wider range of forums. Significant manual reclustering has been needed, but this doesn't affect the embeddings, so a kNN model might struggle to sort existing embeddings by these enforced clusters. A sufficiently annotated dataset could be used to train an additional embedding head that could sit on top of all-MiniLM-L6-v2, to provide a more psychotherapy-focussed clustering process.

6load_model_and_return_prediction.py: The back-end of the online tool.

routes.py
main.py
interface.html
--> These together are the web app

# How to run a virtual environment if you want to test it locally

VSCode terminal commands:

python -m venv venv

venv\Scripts\activate 

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

uvicorn main:fastapi_app --reload

... Then visit: http://127.0.0.1:8000

... and stop with "deactivate"
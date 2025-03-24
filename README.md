# mental-health-text-model
Building an NLP model that can give recommendations to people based on initial text description

# Decisions
- Selecting four Reddit forums with no access requirements as likely to provide suitable data (namely, posts with sufficient content about mental health topics on a variety of themes). r/mental health for general content, r/mentalillness for more severe difficulties, and r/depression and r/anxiety because these are general symptoms that may cover many conditions. Note: the current model won't receive much data about more specific mental health conditions (e.g., OCD, Anorexia, ADHD...) so the training corpus will be composed of some kinds of information and not others. r/mentalhealthUK does not permit scraping.
- Scraping script to interact with Reddit's PRAW API (which determines how Python scripts can interact with public forums to use data) was written to comply with Reddit's access requirements and fair use policies
- Ethical considerations: The Reddit API reduces the risk further of the unfair use of identifying information. Reddit is a public forum. Users have not given explicit consent to their data being used to train a model of the current specifications, but Reddit's general information policy indicates that the data is publicly accessible. 
- Data cleaning was carried out to convert to lowercase, remove URLs, remove whitespace, etc.
- Older vectorisation processes like TF-IDF or Word2Vec / GloVe	(which are poorer with word order or context) were disregarded in favour of an Sentence-BERT (SBERT) process. No model was found that was specifically fine-tuned for sentence embeddings in mental health data: This could be an avenue for development. 
- Uniform Manifold Approximation and Projection (UMAP) was used as an intermediary step for dimensionality reduction. It turns the high-dimensional embeddings of SBERT into smaller multidimensional vectors while trying to preserve structure. It can improve clustering quality in high-dimensional data, while running the risk of reducing complexity. For the highly dimensional data expected from Reddit posts, it seemed like a helpful step to include. TUNING PARAMETERS?
- To create clusters, Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) was used rather than KMeans, since KMeans requires estimating the number of clusters in advance and assumes that clusters are roughly round or spherical (often not true of language). HDBSCAN doesn’t force every point into a cluster, labelling points that don't clearly belong to any dense group as noise (-1). This is advisable for mental health text, where some posts may be unique or ambiguous. It's better to use learning algorithms that don't assume all data points can be classified.
- The posts were then examined by clusters. Using domain knowledge and AI assistance, descriptive categories were developed for each cluster. 


NEXT TASKS
- check how model is saved so that it can be accessed by queries
- refine text return for each cluster and get into recommend.py in a format that works
- also put recommendations for questonnaires and resources in the feedback
- build user interface 
- testing and debugging


- put iframe in Wix embed html block, replacing "yourusername" with the url for the tool <iframe src="https://yourusername.pythonanywhere.com" width="100%" height="800px" style="border:none;"></iframe>

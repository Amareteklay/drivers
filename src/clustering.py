from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, util
import pandas as pd

def compute_embeddings(texts: List[str], model_name: str = "all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_tensor=True)

def perform_clustering(embeddings, n_clusters=5):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering_model.fit_predict(embeddings)
    return labels

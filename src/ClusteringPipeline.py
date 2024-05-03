import logging
import re
from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from hdbscan import validity
from fast_hdbscan import HDBSCAN
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import cosine_similarity
import datetime




class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, data: List[str]) -> List[str]:
        pass



class TextPreprocessor(DataPreprocessor):
    def __init__(self, config: Dict):
        self.config = config

    def preprocess(self, data: List[str]) -> List[str]:
        preprocessed_data = []
        for doc in data:
            doc = str(doc).lower()
            doc = doc.strip()
            doc = re.sub("</?.*?>"," <> ",doc)
            doc = re.sub(r"[^Ⴀ-ჿⴀ-ⴥᲐ-Ჿ0-9a-zA-Z.;:]", ' ', doc)
            doc = re.sub(r"\s+", ' ', doc)
            doc = doc.strip()
            preprocessed_data.append(doc)
        return preprocessed_data

class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate_embeddings(self, data: List[str]) -> np.ndarray:
        pass
    



class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, config: Dict, model_name: str):
        self.config = config
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, data: List[str]) -> np.ndarray:
        embeddings = self.model.encode(data)
        return embeddings
    


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> List[int]:
        pass
    


class HDBSCANClustering(ClusteringAlgorithm):
    def __init__(self, config: Dict):
        self.config = config

    def cluster(self, embeddings: np.ndarray) -> List[int]:
        config = self.config.copy()
        if self.config['metric'] == 'cosine':
            embeddings = cosine_similarity(embeddings)
            embeddings = embeddings.astype('double')
            # print(embeddings.shape)
            config['metric'] = 'precomputed'
        hdbscan_clusterer = HDBSCAN(**config)
        labels = hdbscan_clusterer.fit_predict(embeddings)
        # print(labels)
        return labels.tolist()
    
    def hyperparameter_tuning(self, embeddings: np.ndarray):
        print("Tuning hyperparameters for HDBSCAN...")
        logging.captureWarnings(True)
        hdb = HDBSCAN(gen_min_span_tree=True).fit(embeddings)

        param_dist = {'min_samples': [1,2,3,4,5,6,7,8],
                    'min_cluster_size':[2,3,4,5,6,7,8],  
                    'cluster_selection_method' : ['eom','leaf'],
                    'metric' : ['euclidean','manhattan', 'minkowski'],
                    'p' : [1,2,3,4,5,6,7,8],
                    }

        #validity_scroer = "hdbscan__hdbscan___HDBSCAN__validity_index"
        validity_scorer = make_scorer(validity.validity_index,greater_is_better=True)


        n_iter_search = 20
        random_search = RandomizedSearchCV(hdb
                                        ,param_distributions=param_dist
                                        ,n_iter=n_iter_search
                                        ,scoring=validity_scorer 
                                        ,random_state=42
                                        ,verbose=1)

        random_search.fit(embeddings)
        print(f"Best Parameters {random_search.best_params_}")
        print(f"DBCV score :{random_search.best_estimator_.relative_validity_}")
        # return random_search.best_params_



class ResultVisualizer:
    def __init__(self, config: Dict, cluster_scores: List[int]=None):
        self.config = config
        self.labels = None
        self.embeddings = None
        self.data = None
        self.cluster_scores = cluster_scores
        
    def visualize(self, clusters: List[int], embeddings: np.ndarray, data: List[str], method='tsne'):
        self.labels = clusters
        self.embeddings = embeddings
        self.data = data
        
        if method == 'tsne':
            tsne = TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)
        elif method == 'umap':
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_embeddings = umap_reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Invalid visualization method: {method}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        ax.set_title(f"{method.upper()} Visualization")
        plt.show()


    def print_clusters(self, clusters: List[int], embeddings: np.ndarray, data: pd.DataFrame, title_col: str):
        self.labels = clusters
        self.embeddings = embeddings
        self.data = data
        
        for label, score in sorted(self.cluster_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"Cluster {label} (Silhouette Score: {score}):")
            self._print_cluster(label, title_col)
            print()

    def _print_cluster(self, label, title_col):
        cluster_mask = (self.labels == label)
        cluster_data = self.data[cluster_mask]
        
        for _, row in cluster_data.iterrows():
            combined_text = f"{row[title_col]}".replace('\n', ' ').strip()
            print(combined_text)


        

class ClusteringPipeline:
    def __init__(self, config: Dict, model_name: str = None, title_col: str = 'title', datetime_col: str = 'created_at'):
        self.config = config
        self.embedding_generator = SentenceTransformerEmbeddingGenerator(config, model_name)
        self.clustering_algorithm = HDBSCANClustering(config)
        self.dimensionality_reducer = umap.UMAP(n_components=200, random_state=42)
        self.dimensionality_reducer = None
        self.visualizer = ResultVisualizer(config)
        self.embedding_timestamper = EmbeddingTimestamper(config)
        self.title_col = title_col
        self.datetime_col = datetime_col
        self.embeddings = None
        self.data = None
        self.labels = None
        self.cluster_scores = None
        print("Pipeline initialized")

    def run(self, data: pd.DataFrame):
        self.data = data
        titles = data[self.title_col].tolist()                
        self.embeddings = self._generate_embeddings(titles)
        print("Shape of embeddings:", self.embeddings.shape)
        return self._run(titles)
        
    def _run(self, data: List[str]):
        # I made this helper to avoid duplicate code in the DoubleClusteringPipeline class
        self.labels = self._cluster(self.embeddings)
        silhouette_avg, scored_samples = self._silhouette(self.embeddings, self.labels)
        self.cluster_scores = self._calculate_cluster_scores(self.labels, scored_samples)
        self.cluster_scores = dict(sorted(self.cluster_scores.items(), key=lambda x: x[1], reverse=True))
        self.visualizer.cluster_scores = self.cluster_scores
        # print("Silhouette scores for each sample:", scored_samples)
        print(f"Average Silhouette Score: {silhouette_avg:.2f}")

        return self.labels

    
    def _generate_embeddings(self, data):
        embeddings = self.embedding_generator.generate_embeddings(data)
        if self.dimensionality_reducer is not None:
            embeddings = self.dimensionality_reducer.fit_transform(embeddings)
        print("Embeddings generated and timestamped")
        return embeddings


    def _cluster(self, embeddings):
        clusters = self.clustering_algorithm.cluster(embeddings)
        clusters_array = np.array(clusters)
        print("Shape of clusters_array:", clusters_array.shape)
        return clusters_array
    
    def _silhouette(self, embeddings, clusters):
        silhouette_avg = silhouette_score(embeddings, clusters)
        scored_samples = silhouette_samples(embeddings, clusters)
        return silhouette_avg,scored_samples    
    
    def _calculate_cluster_scores(self, clusters_array, scored_samples):
        cluster_scores = {}
        unique_labels = np.unique(clusters_array)
        for label in unique_labels:
            cluster_mask = (clusters_array == label)
            cluster_scores[label] = np.mean(scored_samples[cluster_mask])
        return cluster_scores

    def print_clusters(self):
        self.visualizer.print_clusters(self.labels, self.embeddings, self.data, self.title_col)

    def visualize(self, method='tsne'):
        self.visualizer.visualize(self.labels, self.embeddings, self.data, method)
    
    def hyperparameter_tuning(self,data: List[str]):
        if self.embeddings is None:
            self.embeddings = self._generate_embeddings(data)
        self.clustering_algorithm.hyperparameter_tuning(self.embeddings)


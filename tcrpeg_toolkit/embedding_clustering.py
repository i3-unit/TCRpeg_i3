import os, sys
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator
import faiss
import argparse
import logging
from sklearn.metrics import silhouette_score

from tcrpeg_toolkit.embedding_handler import EmbeddingHandler, Embedding

#todo change all the print to logging
class OptimalClusterFinder:
    def __init__(self, data, max_k=100, random_state=42):
        """
        Initialize the OptimalClusterFinder with the dataset.

        Parameters:
        - data: The dataset for which to find the optimal number of clusters.
        - max_k: The maximum number of clusters to consider.
        - random_state: Seed for reproducibility.
        """
        self.data = data
        self.max_k = max_k
        self.random_state = random_state

    def find_optimal_clusters(self, clustering_method='kmeans'):
        """
        Finds the optimal number of clusters (k) for KMeans clustering using the elbow method.

        Returns:
        - An integer indicating the optimal number of clusters based on the elbow method.
        """

        logging.info("Finding the optimal number of clusters using the elbow method...")
        n_samples = self.data.shape[0]
        
        # Adjust the maximum number of clusters based on the rule of thumb if necessary
        rule_of_thumb_k = int(np.sqrt(n_samples / 2))
        max_k = min(self.max_k, rule_of_thumb_k)
        
        inertias = []
        k_values = range(1, max_k + 1)

        for k in k_values:
            if clustering_method == 'kmeans':
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10).fit(self.data)
                inertias.append(kmeans.inertia_)
            # Use the KneeLocator to find the elbow point
            elif clustering_method == 'faiss':
                kmeans = faiss.Kmeans(self.data.shape[1], k, niter=100, verbose=False)
                kmeans.train(self.data)
                inertias.append(kmeans.obj[-1])
            else:
                raise ValueError(f"Invalid clustering method: {clustering_method}")

        # Use the KneeLocator to find the elbow point
        knee_locator = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')

        optimal_k = knee_locator.elbow
        logging.info(f"Based on the rule of thumb, the maximum number of clusters k should be ≤ {rule_of_thumb_k}.")
        logging.info(f"The optimal number of clusters k based on the elbow method is {optimal_k}.")
        
        if optimal_k is None:
            logging("Warning: The elbow point could not be found. Returning the maximum number of clusters.")
            optimal_k = max_k
        
        return optimal_k

class EmbeddingClustering:
    def __init__(self, data, output_dir=None, device='cpu'):
        self.data = data 
        self.output_dir = output_dir
        self.device = device
        self.embedding_handler = None
        self.embeddings = None
        self.ids = None
        self.sequences = None

        self._get_embeddings()

    def _get_embeddings(self):
        required_attrs = ['embeddings', 'ids', 'sequences']
        if not all(hasattr(self.data, attr) for attr in required_attrs):
            self.embedding_handler= EmbeddingHandler(self.data)
            self.embeddings = self.embedding_handler.get_embeddings()
            self.ids = self.embedding_handler.get_ids()
        else:
            logging.info("Loaded Embedding Object")
            self.embeddings = self.data.embeddings
            self.ids = self.data.ids
            self.sequences = self.data.sequences
            self.embedding_handler = self.data

    # def prepare_directories_and_filenames(self):
    #     # Create the analysis output directory if it doesn't exist
    #     os.makedirs(self.output_dir, exist_ok=True)
    #     self.analysis_dir = os.path.join(self.output_dir, "analysis")
    #     os.makedirs(self.analysis_dir, exist_ok=True)
        
    #     # Extract the input file name without extension
    #     self.input_name = os.path.basename(self.input_file).split('_embeddings.npy')[0]

    def find_optimal_clusters(self, clustering_method='faiss'):
        optimal_cluster_finder = OptimalClusterFinder(self.embeddings, max_k=100, random_state=42)
        return optimal_cluster_finder.find_optimal_clusters(clustering_method=clustering_method)
                                        
    def faiss_clustering(self,  k=4, niter=10):
        #   Set up FAISS clustering
        d = self.embeddings.shape[1]  # dimensionality of vectors

        # Create and train the clustering model
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True)
        kmeans.train(self.embeddings)    

        # Get cluster assignments only and ignore distances (_,)
        _, cluster_assignments = kmeans.index.search(self.embeddings, 1)
        self.cluster_assignments = cluster_assignments.flatten()
        return self.cluster_assignments

    def update_embedding_handler(self, clusters, name='cluster'):
        self.embedding_handler.update_metadata(clusters, column_name=name)
        return self.embedding_handler


    # def save_clusters_to_csv(self):
    #     output_file = os.path.join(self.analysis_dir, f"{self.input_name}_clusters.csv")
    #     df = pd.DataFrame({
    #         'sequence': self.sequences,
    #         'cluster_id': self.cluster_assignments
    #     })
    #     df.to_csv(output_file, index=False)

    # Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.
    # 1: Means clusters are well apart from each other and clearly distinguished.
    # 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
    # -1: Means clusters are assigned in the wrong way.   
    # Used for convex clusters as it is the case here

    def calculate_silhouette_score(self): ## Measures how similar a point is to its own cluster compared to other clusters.
        score = silhouette_score(self.embeddings, self.cluster_assignments)
        logging.info(f"Silhouette score: {score}")
 
    def run(self, k=4, n_iter=10, optimal_cluster=False, clustering_method='faiss'):
        # self.prepare_directories_and_filenames()
        # self.read_embeddings_files()
        if optimal_cluster:
            k = self.find_optimal_clusters(clustering_method=clustering_method)
        faiss_clusters = self.faiss_clustering(k=k, niter=n_iter)
        new_embedding_handler = self.update_embedding_handler(clusters=faiss_clusters, name='cluster')
        self.calculate_silhouette_score()
        return new_embedding_handler
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding clustering with Faiss')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Directory to save clusters',
                        required=True)
    parser.add_argument('-d', '--device', help='Device to use (cpu, cuda:0, mps)', default='cpu',
                        choices=["cpu", "cuda:0", "mps"])
    parser.add_argument('-n', '--iteration', help='Number of iterations for the clustering', default='10',
                        required=True)
    
    # Parse the arguments
    args = parser.parse_args()

    # Check if MPS is available and set the device accordingly
    if args.device == 'mps' and not torch.backends.mps.is_available():
        logging.warning("MPS device requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    cluster_embed = EmbeddingClustering(input_file=args.input, output_dir=args.output, device=args.device)
    cluster_embed.run(niter=args.iteration)
    
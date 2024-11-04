import os, sys
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator
import faiss
import hdbscan
import argparse
import logging
import random
import networkx as nx
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score

from tcrpeg_toolkit.embedding_handler import EmbeddingHandler, Embedding

class OptimalClusterFinder:
    def __init__(self, data, max_k=1000, random_state=42):
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
        logging.info(f"Based on the rule of thumb, the maximum number of clusters k should be ≤ {rule_of_thumb_k}.")

        max_k = min(self.max_k, rule_of_thumb_k)
        
        inertias = []
        k_values = range(1, max_k + 1)

        for k in k_values:
            if clustering_method == 'kmeans':
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10).fit(self.data)
                inertias.append(kmeans.inertia_)
            # Use the KneeLocator to find the elbow point
            elif clustering_method == 'faiss':
                kmeans = faiss.Kmeans(self.data.shape[1], k, niter=10, verbose=False)
                kmeans.train(self.data)
                inertias.append(kmeans.obj[-1])
            else:
                raise ValueError(f"Invalid clustering method: {clustering_method}")

        # Use the KneeLocator to find the elbow point
        knee_locator = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')

        optimal_k = knee_locator.elbow
        logging.info(f"The optimal number of clusters k based on the elbow method is {optimal_k}.")
        
        if optimal_k is None:
            logging("Warning: The elbow point could not be found. Returning the maximum number of clusters.")
            optimal_k = max_k
        
        return optimal_k

#todo add metadata as option
class EmbeddingClustering:
    def __init__(self, data, output_dir=None, device='cpu'):
        self.data = data 
        self.output_dir = output_dir
        self.device = device
        self.embedding_handler = None
        self.embeddings = None
        self.ids = None
        self.sequences = None
        self.cluster_assignments = None
        self.graphs = {} 

        self._get_embeddings()

    def _get_embeddings(self):
        required_attrs = ['embeddings', 'ids', 'sequences']
        if not all(hasattr(self.data, attr) for attr in required_attrs):
            self.embedding_handler= EmbeddingHandler(self.data)
            self.embeddings = self.embedding_handler.get_embeddings()
            self.ids = self.embedding_handler.get_ids()
            self.data = self.embedding_handler
        else:
            logging.info("Loaded Embedding Object")
            self.embeddings = self.data.embeddings
            self.sequences = self.data.sequences
            self.ids = self.data.ids
            self.embedding_handler = self.data

    # # Davies-Bouldin Index
    # db_score = davies_bouldin_score(umap_embedding, cluster_labels)
    # print(f'Davies-Bouldin Index: {db_score}')

    # # Calinski-Harabasz Index
    # ch_score = calinski_harabasz_score(umap_embedding, cluster_labels)
    # print(f'Calinski-Harabasz Index: {ch_score}')
    #             self.embedding_handler = self.data

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
    
    
    def hdbscan_clustering(self, k=4):
        logging.info('Applying hdbscan...')
        subcluster_labels = np.full(self.embeddings.shape[0], -1)  # Initialize with -1 for outliers
        # subcluster_min_span_tree = {}
        self.graphs = {}
        
        for supercluster in range(k):
            mask = self.cluster_assignments == supercluster
            if np.sum(mask) > 0:
                cluster_data = self.embeddings[mask]
                hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
                hdbscan_labels = hdbscan_clusterer.fit_predict(cluster_data)
                subcluster_labels[mask] = hdbscan_labels + subcluster_labels.max() + 1  # Ensure unique labels
                # subcluster_min_span_tree = hdbscan_clusterer.minimum_spanning_tree_.to_networkx()
                # condensed_tree = hdbscan_clusterer.condensed_tree_

                # Create a NetworkX graph
                G = nx.Graph()

                # Create a NetworkX graph from the minimum spanning Tree
                # for edge in subcluster_min_span_tree.edges(data=True):
                #     G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
                
                # # Create a NetworkX graph from the Condensed Tree
                # for edge in condensed_tree.edges(data=True):
                #     G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
                
                # Add nodes and edges based on cluster labels
                for i, label in enumerate(hdbscan_clusterer.labels_):
                    G.add_node(i, cluster=label)
                    for j, other_label in enumerate(hdbscan_clusterer.labels_):
                        if i != j and label == other_label:
                            G.add_edge(i, j)
                self.graphs[supercluster] = G

            # Calculate DBCV score
            # dbcv_score = hdbscan.validity.validity_index(cluster_data, hdbscan_labels, 
            #                                                 metric='euclidean', d=hdbscan_clusterer.relative_validity_)
            # logging.info(f"DBCV score: {dbcv_score}")
        # print(subcluster_min_span_tree)
        self.cluster_assignments = subcluster_labels
        return self.cluster_assignments, self.graphs
    
    def calculate_graph_metrics(self):
        """
        Calculate graph metrics for the given NetworkX graph.

        Returns:
        - A dictionary containing the calculated graph metrics.
        """
        graph_metrics = {}

        for supercluster, G in self.graphs.items():
            # Calculate graph metrics
            # degrees = dict(G.degree())
            # closeness_centrality = nx.closeness_centrality(G, distance='weight')
            average_degree = np.mean([deg for _, deg in G.degree()])
            density = nx.density(G)
            average_clustering_coefficient = nx.average_clustering(G)

            graph_metrics[supercluster] = {
                # 'degrees': degrees,
                # 'closeness_centrality': closeness_centrality,
                'average_degree': average_degree,
                'density': density,
                'average_clustering_coefficient': average_clustering_coefficient
            }

        return graph_metrics
    
    def update_embedding_handler(self, values, name='cluster', metrics=None):
        print(f"Updating metadata with column: {name}")
        self.embedding_handler.update_metadata(values, column_name=name)
        if metrics:
            for metric_name, metric_values in metrics.items():
                print(f"Updating metadata with metric: {metric_name}")
                self.embedding_handler.update_metadata(metric_values, column_name = f'{name}_{metric_name}')
        
        return self.embedding_handler

    def calculate_save_cluster_metrics(self):
        if self.cluster_assignments is None:
            logging.error("No cluster assignments available. Cannot calculate cluster metrics.")
            return
        silhouette_score = self.calculate_silhouette_score()
        davies_bouldin_score = self.calculate_davies_bouldin_score()
        calinski_harabasz_score = self.calculate_calinski_harabasz_score()
        
        graph_metrics = self.calculate_graph_metrics()
        flattened_graph_metrics = {}
        for supercluster, metrics in graph_metrics.items():
            for metric_name, metric_value in metrics.items():
                if metric_name in ['average_degree' ,'density','average_clustering_coefficient']:
                    flattened_graph_metrics[f'graph_{supercluster}_{metric_name}'] = [metric_value] * len(self.embeddings)
                else: 
                    flattened_graph_metrics[f'graph_{supercluster}_{metric_name}'] = [metric_value.get(str(node_id), 0) for node_id in self.ids]   
                    print(f"Flattened Metric Values: {flattened_graph_metrics[f'graph_{supercluster}_{metric_name}']}")
                    #TODO check if this is correct
        metrics = {
            'silhouette_score': silhouette_score,
            'davies_bouldin_score': davies_bouldin_score,
            'calinski_harabasz_score': calinski_harabasz_score,
            **flattened_graph_metrics
        }
        
        # Update metadata with all metrics
        self.update_embedding_handler(values=self.cluster_assignments, name='cluster_metrics', metrics=metrics)
        
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

    # def calculate_cluster_purity(self):
    #     """
    #     Calculate the purity score for the current clustering.

    #     Cluster purity is a measure of the extent to which clusters contain a single class.
    #     A higher purity score indicates better clustering quality.

    #     Returns:
    #         None: The method logs the purity score using the logging module.
    #     """
    #     if not hasattr(self, 'true_labels'):
    #         logging.warning("True labels not available. Cannot calculate cluster purity.")
    #         return
            
    #     contingency_matrix = np.zeros((len(np.unique(self.cluster_assignments)), 
    #                                  len(np.unique(self.true_labels))))
        
    #     for i in range(len(self.cluster_assignments)):
    #         contingency_matrix[self.cluster_assignments[i], self.true_labels[i]] += 1
            
    #     purity = np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    #     logging.info(f"Cluster purity score: {purity}")

    def calculate_silhouette_score(self): 
        """
        Calculate and log the silhouette score for the current clustering.

        The silhouette score measures how similar a point is to its own cluster compared to other clusters.
        A higher silhouette score indicates that the clusters are well-defined.

        Returns:
            None
        """
        # if cluster_labels == 'kmeans':
        #     cluster_assignments = self.supercluster_labels
        # elif cluster_labels == 'hdbscan':
        #     cluster_assignments = self.subcluster_labels
        # else:
        #     logging.error("Clustering method is not supported. Should be one of 'kmeans' or 'hdbscan'. Cannot calculate cluster metrics.")
        #     return
            
        score = silhouette_score(self.embeddings, self.cluster_assignments)
        logging.info(f"Silhouette score: {score}")
        return score

    def calculate_davies_bouldin_score(self): 
        """
        Calculate the Davies-Bouldin score for the current clustering.

        The Davies-Bouldin index is defined as the average similarity measure of each cluster 
        with its most similar cluster, where similarity is the ratio of within-cluster distances 
        to between-cluster distances.

        This method computes the Davies-Bouldin score using the embeddings and cluster assignments 
        stored in the instance.

        Returns:
            None: The method logs the Davies-Bouldin score using the logging module.
        """
        
        # if cluster_labels == 'kmeans':
        #     cluster_assignments = self.supercluster_labels
        # elif cluster_labels == 'hdbscan':
        #     cluster_assignments = self.subcluster_labels
        # else:
        #     logging.error("Clustering method is not supported. Should be one of 'kmeans' or 'hdbscan'. Cannot calculate cluster metrics.")
        #     return
        
        score = davies_bouldin_score(self.embeddings, self.cluster_assignments)
        logging.info(f"Davies-Bouldin score: {score}")
        return score

    def calculate_calinski_harabasz_score(self):
        """
        Calculate the Calinski-Harabasz score for the current clustering.

        The Calinski-Harabasz index is a ratio of the sum of between-cluster dispersion to the 
        sum of within-cluster dispersion. A higher value indicates better clustering.

        This method computes the Calinski-Harabasz score using the embeddings and cluster assignments 
        stored in the instance.

        Returns:
            None: The method logs the Calinski-Harabasz score using the logging module.
        """
        
        # if cluster_labels == 'kmeans':
        #     cluster_assignments = self.supercluster_labels
        # elif cluster_labels == 'hdbscan':
        #     cluster_assignments = self.subcluster_labels
        # else:
        #     logging.error("Clustering method is not supported. Should be one of 'kmeans' or 'hdbscan'. Cannot calculate cluster metrics.")
        #     return
        
        score = calinski_harabasz_score(self.embeddings, self.cluster_assignments)
        logging.info(f"Calinski-Harabasz score: {score}")
        return score

    def calculate_density_based_cluster_validation(self):
        """
        Calculate and log the density-based cluster evaluation metrics for the current clustering.
        """
        
    def downsampling(self, sample_size):
        self.embedding_handler = self.embedding_handler.downsample_embeddings(sample_size)
        self.embeddings = self.embedding_handler.embeddings
        self.sequences = self.embedding_handler.sequences
        self.ids = self.embedding_handler.ids
 
    def run(self, sample_size=None, k=4, n_iter=10, optimal_cluster=False, clustering_method='faiss', apply_hdbscan=True):
        # self.prepare_directories_and_filenames()
        # self.read_embeddings_files()
        if sample_size:
            self.downsampling(sample_size)
        if optimal_cluster:
            k = self.find_optimal_clusters(clustering_method=clustering_method)
        clusters = self.faiss_clustering(k=k, niter=n_iter)
        self.update_embedding_handler(values=clusters, name='cluster')
        if apply_hdbscan:
            clusters, graphs = self.hdbscan_clustering(k=k)
            self.update_embedding_handler(values=clusters, name='cluster_hdbscan')
        self.calculate_save_cluster_metrics()
        
        return self.embedding_handler
   
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
    
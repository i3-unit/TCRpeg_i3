import os, sys
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import seaborn as sns
from sklearn.cluster import KMeans
import faiss
import argparse
import logging
from sklearn.metrics import silhouette_score


# # Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'TCRpeg_i3')))

class EmbeddingClustering:
    def __init__(self, input_file, output_dir, device='cpu'):
        self.input_file = input_file 
        self.output_dir = output_dir
        self.device = device
        self.data = None
        
    def prepare_directories_and_filenames(self):
        # Create the analysis output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Extract the input file name without extension
        self.input_name = os.path.basename(self.input_file).split('_embeddings.npy')[0]

    # add elbow method
    
    def read_embeddings_files(self):
        self.data = np.load(self.input_file)
        self.input_name = os.path.basename(self.input_file).split('_embeddings.npy')[0]
                                        
    def faiss_clustering(self,  k=4, niter=10):
        #   Set up FAISS clustering
        d = self.data.shape[1]  # dimensionality of vectors

        # Create and train the clustering model
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True)
        kmeans.train(self.data)    

        # Get cluster assignments only and ignore distances (_,)
        _, cluster_assignments = kmeans.index.search(self.data, 1)
        self.cluster_assignments = cluster_assignments.flatten()

    def save_clusters_to_csv(self):
        output_file = os.path.join(self.analysis_dir, f"{self.input_name}_clusters.csv")
        df = pd.DataFrame({
            'sequence': self.sequences,
            'cluster_id': self.cluster_assignments
        })
        df.to_csv(output_file, index=False)

    # Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.
    # 1: Means clusters are well apart from each other and clearly distinguished.
    # 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
    # -1: Means clusters are assigned in the wrong way.   
    # Used for convex clusters as it is the case here
    def calculate_silhouette_score(self): ## Measures how similar a point is to its own cluster compared to other clusters.
        return silhouette_score(self.data, self.cluster_assignments)
 
    def run(self, **kwargs):
        self.read_embeddings_files()
        self.faiss_clustering(**kwargs)
        self.calculate_silhouette_score()
        
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
    
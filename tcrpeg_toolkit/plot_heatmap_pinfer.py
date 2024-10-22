import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt

def process_numpy_files(folder_path, use_structured_array=True):
    data_list = []
    sample_ids = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            sample_name = os.path.splitext(file_name)[0]
            data_list.append(np.load(file_path))
            sample_ids.append(sample_name)
            
    
    if use_structured_array:
        dtype = [('sample_id', 'U50'), ('probabilities', float, (len(data_list[0]),))]
        structured_array = np.array(list(zip(sample_ids, data_list)), dtype=dtype)
        return structured_array
    else:
        return np.array(data_list), np.array(sample_ids)

class TCRPEGInference:
    def __init__(self, data, sample_ids=None):
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            self.data = data['probabilities']
            self.sample_ids = data['sample_id']
        else:
            self.data = data
            self.sample_ids = sample_ids if sample_ids is not None else np.arange(len(data))
        self.distance_matrix = None
    
    def calculate_distance_matrix(self, distance_measure='jsd'):
        n_samples = self.data.shape[0]
        self.distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = self._calculate_distance(self.data[i], self.data[j], distance_measure)
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
    
    def _calculate_distance(self, data1, data2, distance_measure):
        if distance_measure == 'jsd':
            min_length = min(len(data1), len(data2))
            data1 = data1[:min_length]
            data2 = data2[:min_length]
            epsilon = 1e-10
            data1 = data1 + epsilon
            data2 = data2 + epsilon
            data1 = data1 / np.sum(data1)
            data2 = data2 / np.sum(data2)
            return jsd(data1, data2)
        elif distance_measure == 'wasserstein':
            return wasserstein_distance(data1, data2)
        else:
            raise ValueError("Unsupported distance measure")
    
    def plot_heatmap(self, title='TCR Distance Heatmap'):
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated. Call calculate_distance_matrix() first.")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.distance_matrix, annot=True, cmap='coolwarm', xticklabels=self.sample_ids, yticklabels=self.sample_ids,
                   cbar_kws={'label': 'Distance'}, fmt='.2f')
        plt.title(title)
        plt.show()
    
    def bootstrap_random_mutation(self, n_iterations=100):
        original_data = self.data.copy()
        bootstrapped_matrices = []
        
        for _ in range(n_iterations):
            mutated_data = self._apply_random_mutation(original_data)
            self.data = mutated_data
            self.calculate_distance_matrix()
            bootstrapped_matrices.append(self.distance_matrix)
        
        self.data = original_data
        return np.mean(bootstrapped_matrices, axis=0), np.std(bootstrapped_matrices, axis=0)
    
    def _apply_random_mutation(self, data):
        mutated_data = data.copy()
        mutation_rate = 0.01  # 1% mutation rate
        
        mask = np.random.random(mutated_data.shape) < mutation_rate
        mutated_data[mask] *= np.random.uniform(0.5, 1.5, mask.sum())
        
        return mutated_data

# Usage example
if __name__ == "__main__":
    # Step 1: Process numpy files
    folder_path = '/Users/celinebalaa/Desktop/thesis/tmp/gliph_signature_downsampled/data/results_p_infer/top_data_sampled/p_infer'

    # Using structured array
    structured_data = process_numpy_files(folder_path, use_structured_array=True)
    tcrpeg_structured = TCRPEGInference(structured_data)
    
    # Using regular array
    data_array, sample_ids = process_numpy_files(folder_path, use_structured_array=False)
    tcrpeg_regular = TCRPEGInference(data_array, sample_ids)
    

    # Step 2: Initialize TCRPEGInference with processed data
    # tcrpeg = TCRPEGInference(numpy_data)

    # Step 3: Calculate distance matrix
    tcrpeg_structured.calculate_distance_matrix(distance_measure='jsd')

    # Step 4: Plot heatmap
    tcrpeg_structured.plot_heatmap()

    # Optional: Perform bootstrapping with random mutations
    mean_matrix, std_matrix = tcrpeg_structured.bootstrap_random_mutation(n_iterations=100)

    # Plot bootstrapped heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_matrix, annot=True, cmap='YlGnBu')
    plt.title('Bootstrapped TCR Distance Heatmap (Mean)')
    plt.show()

    # Plot bootstrapped heatmap with error bars
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Bootstrapped TCR Distance Heatmap (Mean)')
    plt.show()

    # Perform bootstrapping with random mutations
    mean_matrix, std_matrix = tcrpeg_structured.bootstrap_random_mutation(n_iterations=100)
    # Plot bootstrapped heatmap with error bars
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Bootstrapped TCR Distance Heatmap (Mean)')
    plt.show()

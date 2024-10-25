import os
import glob
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt

from tcrpeg_toolkit.utils import load_data

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


class PlotHeatmapPinfer():
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = load_data(metadata) if metadata is not None else None 
        #todo improve load data to take as input None and return None
        # self.p_infer_dict = {}
        # self.multi_index_all = None
        #todo check if better to assign self each time
    
    def load_data_numpy(self, data):
        p_infer_dict = {}
        npy_files = glob.glob(f"{data}/*.npy")
        for npy_file in npy_files:
            data = np.load(npy_file)
            #todo make it more general for the split with p_infer if present else npy
            sample_name = os.path.basename(npy_file).split('_p_infer.npy')[0]
            p_infer_dict[sample_name] = data
        logging.info(f"Loaded {len(npy_files)} files")
        return p_infer_dict

    def calculate_distance(self, p_infer_dict, distance_method='wsd'):
    #todo check if better to always have method wo prefix clustering_method, distance_method ....
        wasserstein_distance_val_all = np.array([[wasserstein_distance(value1, value2) for value1 in p_infer_dict.values()]
                        for value2 in p_infer_dict.values()])
        wasserstein_distance_df_all = pd.DataFrame(wasserstein_distance_val_all, index=p_infer_dict.keys(), columns=p_infer_dict.keys())
        return wasserstein_distance_df_all

    def process_metadata(self, metadata, names_order):
        # Ensure no leading or trailing whitespace
        metadata.columns = metadata.columns.str.strip()

        # Sort metadata by id or index if id is not present to match the order of the data
        try:
            if 'id' in metadata.columns:
                metadata = metadata.set_index('id').reindex(names_order).reset_index()
            else:
                logging.warning("Metadata does not contain an 'id' column. Sorting by index.")
                metadata = metadata.reindex(names_order).reset_index()

        except:
            logging.warning("Metadata id or index does not match the data names (name_p_infer.npy). Not using metadata.")
            metadata = None

        return metadata

    def create_multi_index(self, metadata, columns=None):
        if columns is None:
            columns_to_use = [col for col in metadata.columns if col != 'id']
        else:
            columns_to_use = [col for col in columns if col in metadata.columns]
            missing_cols = [col for col in columns if col not in metadata.columns]
            if missing_cols:
                logging.warning(f"Columns {missing_cols} not found in metadata. Skipping these columns.")

        multi_index_all = pd.MultiIndex.from_arrays(
            [metadata[col] for col in columns_to_use], 
            names=columns_to_use)
        
        return multi_index_all

    def annotate_distance_matrix(self, distance_matrix, metadata_multi_idx):
        distance_matrix.columns = metadata_multi_idx
        distance_matrix.index = metadata_multi_idx
        return distance_matrix

    def _create_color_palette(self, metadata, column, custom_palette=None):
        metadata[column] = metadata[column].astype(str)        
        unique_values = metadata[column].unique()
        n_colors = len(unique_values)
        
        if custom_palette is None:
            # Generate palette using seaborn
            pal = sns.husl_palette(n_colors, s=.45)
        else:
            pal = custom_palette
            
        # Create lookup dictionary
        lut = dict(zip(map(str, unique_values), pal))
        
        # Convert palette to vectors
        values = metadata[column]
        colors = pd.Series(values, index=metadata.index).map(lut)
        
        return colors, lut

    def assign_metadata_color(self, metadata_multi_idx):
        #todo add option for palette here
        # Create distinct palettes for each column using different seaborn palettes
        palette_options = ['Set2', 'Dark2', 'Accent', 'Pastel1', 'Spectral', 'RdYlBu', 'PRGn', 'PiYG', 'BrBG']
        color_palettes = {}

        for i, col in enumerate(metadata_multi_idx.names):
            level_values = metadata_multi_idx.get_level_values(col)
            palette = sns.color_palette(palette_options[i % len(palette_options)], n_colors=len(level_values.unique()))
            colors, lut = self._create_color_palette(pd.DataFrame({col: level_values}), col, custom_palette=palette)
            color_palettes[col] = {'colors': colors, 'lut': lut}

        row_colors_df = pd.DataFrame({
            col: color_palettes[col]['colors'] for col in metadata_multi_idx.names
        }).set_index(metadata_multi_idx)

        return row_colors_df

    def plot_heatmap(self, distance_matrix_annotated, row_colors=None, col_colors=None, normalize=False):
        #todo this range is only for wasserstein_distance_val_all 
        #todo add jsd distance for calculation with range 0,1
        v_min = -1 if normalize else distance_matrix_annotated.min().min()
        v_max = 1 if normalize else distance_matrix_annotated.max().max()

        g = sns.clustermap(distance_matrix_annotated, 
                    cmap="vlag", 
                    row_colors=row_colors, 
                    col_colors=col_colors,
                    vmin = v_min,
                    vmax = v_max,
                    center=0,
                    dendrogram_ratio=(.1, .2),
                    cbar_pos=(.02, .32, .03, .2),
                    linewidths=.75, 
                   method='ward',
                    figsize=(12, 13))
        #todo add legend and ax as option with show 
        return g 

    def run(self, **kwargs):
        p_infer_dict = self.load_data_numpy(self.data)
        distance_matrix = self.calculate_distance(p_infer_dict)
        metadata_ordered = self.process_metadata(self.metadata, distance_matrix.index)
        metadata_multi_idx = self.create_multi_index(self.metadata) if self.metadata is not None else None
        distance_matrix_annotated = self.annotate_distance_matrix(distance_matrix, metadata_multi_idx) if metadata_multi_idx is not None else distance_matrix
        metadata_multi_idx_colors = self.assign_metadata_color(metadata_multi_idx) if metadata_multi_idx is not None else None
        self.plot_heatmap(distance_matrix_annotated, row_colors=metadata_multi_idx_colors, col_colors=metadata_multi_idx_colors, **kwargs)
        #todo add return for plot

class TCRpegInference:
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

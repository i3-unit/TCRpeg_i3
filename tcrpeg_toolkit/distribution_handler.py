import os
import glob
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.stats import entropy, wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt

from tcrpeg_toolkit.utils import load_data

class Distribution:
    def __init__(self, distributions, ids=None, **kwargs):
        self.distributions = distributions
        self.ids = ids
        self.distance_matrix = kwargs.get('distance_matrix', None)
        self.metadata = kwargs.get('metadata', None)
        self.distance_metric = kwargs.get('distance_metric', None)
        self.slot_name = kwargs.get('slot_name', None)

    def __repr__(self):
        return f"Distribution(distributions={len(self.distributions)}, ids={len(self.ids) if self.ids else None}, " \
               f"distance_matrix={self.distance_matrix.shape if self.distance_matrix is not None else None}, " \
               f"distance_metric={self.distance_metric}, " \
               f"slot_name={self.slot_name}, " \
               f"metadata={self.metadata.shape if self.metadata is not None else None})"


class DistributionLoader:
    def __init__(self, folder_path, **kwargs):
        self.folder_path = folder_path
        self.distribution_object = None
        self.distributions = []
        self.sample_ids = []
        self.load_distributions(**kwargs)

    def load_distributions(self, slot_name='p_infer', message=False):
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        self.distributions = []
        self.ids = []

        for file_name in files:
            file_path = os.path.join(self.folder_path, file_name)
            data = np.load(file_path)

            if data.dtype.names is not None:
                # Extract a specific slot if it's a structured array
                if slot_name in data.dtype.names:
                    distribution = data[slot_name]
                else:
                    distribution = None  # or handle the absence of the slot differently
                logging.info(f"Loaded structured array with names from {file_name}") if message else None
            else:
                # Handle regular array
                distribution = data
                logging.info(f"Loaded regular array from {file_name}") if message else None

            # Append the loaded data and sample ID
            if distribution is not None:
                self.distributions.append(distribution)
                sample_name = os.path.splitext(file_name)[0].split(f'_{slot_name}')[0]
                # Check if raw or structured is present and remove it
                sample_name = sample_name.split('_raw')[0] if '_raw' in sample_name else sample_name
                sample_name = sample_name.split('_structured')[0] if '_structured' in sample_name else sample_name
                self.ids.append(sample_name)
                #todo maybe add data type as a variable

        logging.info(f"Loaded {len(self.distributions)} distributions with their IDs")
        return Distribution(self.distributions, self.ids)

    def __repr__(self):
        return f"DistributionLoader(distributions={len(self.distributions)}, folder_path={self.folder_path})"

#todo fix all kwargs

class DistributionDistanceCalculator:
    def __init__(self, data, **kwargs):
        self.data = data
        self.distributions = []
        self.ids = []
        self.distance_matrix = None
        self.distance_metric = None

        self._load_distributions(**kwargs)
        #todo check if the better name is DistributionDistanceHandler or DistributionTransformer

    def _load_distributions(self, **kwargs):
        if self.data is None:
            raise ValueError("No data provided. Load data first.")
        if not all(hasattr(self.data, attr) for attr in ['distributions', 'ids']):
            self.data = DistributionLoader(self.data, **kwargs)
        self.distributions = self.data.distributions
        self.ids = self.data.ids

    def pad_and_normalize_distributions(self, epsilon=1e-6):
        # Determine the maximum length of any distribution in the list
        max_length = max(len(dist) for dist in self.distributions)

        # Pad all distributions to this maximum length with epsilon
        self.distributions = [
            np.pad(dist, (0, max_length - len(dist)), 'constant', constant_values=(epsilon,))
            for dist in self.distributions
        ]

        # Normalize each distribution to ensure it sums to 1
        self.distributions = [
            dist / dist.sum() for dist in self.distributions
        ]
        return self.distributions

    def apply_smoothing_normalization(self, epsilon=1e-6):
        smoothed = [np.clip(dist + epsilon, a_min=epsilon, a_max=None) for dist in self.distributions]
        self.distributions = [dist / dist.sum() for dist in smoothed]
        return self.distributions

    def calculate_distance_matrix(self, metric='jsd', **kwargs):
        """
        Calculate the distance matrix for the given distributions using the specified metric.

        Parameters:
        -----------
        metric : str, optional
            The distance metric to use. Supported metrics include:
            - 'jensenshannon', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 
              'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 
              'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 
              'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 
              'yule' (from scipy)
            - 'wd', 'wsd', 'wasserstein' (Wasserstein distance)
            - 'jsd', 'jensenshannondivergence' (Jensen-Shannon divergence)
            - 'KL', 'kl', 'kullbackleibler' (Kullback-Leibler divergence)
        **kwargs : dict, optional
            Additional keyword arguments to pass to the distance metric function.

        Returns:
        --------
        numpy.ndarray
            A distance matrix calculated using the specified metric.

        Raises:
        -------
        ValueError
            If an unsupported distance metric is provided.
        """
        match metric:
            case ('jensenshannon' | 'braycurtis' | 'canberra' | 'chebyshev' | 'cityblock' | 
                'correlation' | 'cosine' | 'dice' | 'euclidean' | 'hamming' | 'jaccard' | 'kulczynski1' |
                'mahalanobis' | 'matching' | 'minkowski' | 'rogerstanimoto' | 'russellrao' |
                'seuclidean' | 'sokalmichener' | 'sokalsneath' | 'sqeuclidean' | 'yule'):
                logging.info(f"Distance metric '{metric}' selected from scipy.")
                self.distance_matrix = squareform(pdist(self.distributions, metric=metric, **kwargs))
            case 'wd' | 'wsd' | 'wasserstein':
                logging.info("Wasserstein distance metric selected.")
                self.distance_matrix = squareform(pdist(self.distributions, lambda u, v: wasserstein_distance(u, v)))
            case 'jsd' | 'jensenshannondivergence':
                logging.info("Jensen-Shannon divergence metric selected.")
                self.distance_matrix = squareform(pdist(self.distributions, lambda u, v: jensenshannon(u, v) ** 2))
            case 'KL' | 'kl' | 'kullbackleibler':
                logging.info("Kullback-Leibler divergence metric selected.")
                self.distance_matrix = squareform(pdist(self.distributions, lambda u, v: entropy(u, v)))
            case _:
                raise ValueError(f"Unsupported distance metric '{metric}'")

        self.distance_metric = metric
        return self.distance_matrix

    def __repr__(self):
        return f"DistributionDistanceCalculator(distributions={len(self.distributions)}, ids={len(self.ids)}, " \
                f"distance_matrix={self.distance_matrix.shape if self.distance_matrix is not None else None} " \
                f"distance_metric={self.distance_metric})"

    def run(self, padding=True, smoothing=True, distance_metric='jsd', **kwargs):
        """
        Run the distribution analyzer with optional padding, smoothing, and distance matrix calculation.
        Parameters:
        -----------
        padding : bool, optional
            Whether to pad the distributions to the same length. Default is True.
        smoothing : bool, optional
            Whether to apply smoothing to the distributions. Default is True.
        distance_metric : str, optional
            The distance metric to use for calculating the distance matrix. Default is 'jensenshannon'.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the distance metric function.
            For example, for 'jensenshannon' metric, you can pass 'w' for the weighting parameter.
            rns:
        --------
        numpy.ndarray
            A distance matrix calculated using the specified metric.
        """
        if padding:
            self.pad_and_normalize_distributions()
        if smoothing:
            self.apply_smoothing_normalization()
        self.calculate_distance_matrix(distance_metric, **kwargs)
        #todo maybe return self to be able to chain the methods
        return Distribution(self.distributions, self.ids, distance_matrix=self.distance_matrix, distance_metric=self.distance_metric)

#todo fix to use slot_name as variable

class DistributionHeatmapPlotter:
    def __init__(self, data, ids=None, metadata=None, **kwargs):
        self.data = data
        self.metadata = load_data(metadata, message=False)
        self.distribution_object = None
        self.multi_index_all = None
        self.distance_matrix_annotated = None
        self.metadata_multi_idx_colors = None
        
        self._load_distributions_distance()

        self.metric_ranges = {
            'jensenshannon': (0, 1),
            'braycurtis': (0, 1),
            'canberra': (0, float('inf')),
            'chebyshev': (0, float('inf')),
            'cityblock': (0, float('inf')),
            'correlation': (-1, 1),
            'cosine': (0, 1),
            'dice': (0, 1),
            'euclidean': (0, float('inf')),
            'hamming': (0, 1),
            'jaccard': (0, 1),
            'kulczynski1': (0, 1),
            'mahalanobis': (0, float('inf')),
            'matching': (0, 1),
            'minkowski': (0, float('inf')),
            'rogerstanimoto': (0, 1),
            'russellrao': (0, 1),
            'seuclidean': (0, float('inf')),
            'sokalmichener': (0, 1),
            'sokalsneath': (0, 1),
            'sqeuclidean': (0, float('inf')),
            'yule': (0, 1),
            'wd': (0, float('inf')),
            'wsd': (0, float('inf')),
            'wasserstein': (0, float('inf')),
            'jsd': (0, 1),
            'jensenshannondivergence': (0, 1),
            'KL': (0, float('inf')),
            'kl': (0, float('inf')),
            'kullbackleibler': (0, float('inf'))
        }

    def _load_distributions_distance(self, **kwargs):
        if isinstance(self.data, Distribution):
            logging.info("Loaded Distribution object.")
            self.distribution_object = self.data
        else:
            logging.info
            self.distribution_object = DistributionDistanceCalculator(self.data, **kwargs).run(**kwargs)

        self.distributions = self.distribution_object.distributions
        self.ids = self.distribution_object.ids
        self.distance_matrix = self.distribution_object.distance_matrix
        self.distance_metric = self.distribution_object.distance_metric

    def process_metadata(self):
        if self.metadata is None:
            logging.warning("No metadata provided. Skipping metadata processing.")
            return None
        
        # Ensure no leading or trailing whitespace
        self.metadata.columns = self.metadata.columns.str.strip()

        # Sort metadata by id or index if id is not present to match the order of the data
        try:
            if 'id' in self.metadata.columns:
                # Check if id and self.ids match
                if not self.metadata['id'].isin(self.ids).all():
                    logging.warning("Metadata id does not match the data names (name_p_infer.npy). Not using metadata.")
                    self.metadata = None
                else:
                    logging.info("Metadata id matches the data names (name_p_infer.npy). Sorting by id.")
                self.metadata = self.metadata.set_index('id').reindex(self.ids).reset_index()
            else:
                logging.warning("Metadata does not contain an 'id' column. Sorting by index.")
                self.metadata = metadata.reindex(self.ids).reset_index()

        except:
            logging.warning("Metadata id or index does not match the data names (name_p_infer.npy). Not using metadata.")
            self.metadata = None

        return self.metadata

    def create_multi_index(self, columns=None):
        if self.metadata is None:
            logging.warning("No metadata provided. Skipping multi-index creation.")
            return None
        if columns is None:
            columns_to_use = [col for col in self.metadata.columns if col != 'id']
        else:
            columns_to_use = [col for col in columns if col in self.metadata.columns]
            missing_cols = [col for col in columns if col not in self.metadata.columns]
            if missing_cols:
                logging.warning(f"Columns {missing_cols} not found in metadata. Skipping these columns.")

        self.multi_index_all = pd.MultiIndex.from_arrays(
            [self.metadata[col] for col in columns_to_use], 
            names=columns_to_use)
        
        return self.multi_index_all
        #todo maybe change the name to metadata_multi_idx

    def annotate_distance_matrix(self):
        self.distance_matrix_annotated = pd.DataFrame(self.distance_matrix, index=self.ids, columns=self.ids)
        if self.multi_index_all is None:
            logging.warning("No multi-index found. Skipping annotation.")
            return self.distance_matrix_annotated
        
        self.distance_matrix_annotated.columns = self.multi_index_all
        self.distance_matrix_annotated.index = self.multi_index_all

        return self.distance_matrix_annotated

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

    def assign_metadata_color(self):
        if self.multi_index_all is None:
            logging.warning("No multi-index found. Skipping metadata color assignment.")
            return None
        #todo add option for palette here
        # Create distinct palettes for each column using different seaborn palettes
        palette_options = ['Set2', 'Dark2', 'Accent', 'Pastel1', 'Spectral', 'RdYlBu', 'PRGn', 'PiYG', 'BrBG']
        color_palettes = {}

        for i, col in enumerate(self.multi_index_all.names):
            level_values = self.multi_index_all.get_level_values(col)
            palette = sns.color_palette(palette_options[i % len(palette_options)], n_colors=len(level_values.unique()))
            colors, lut = self._create_color_palette(pd.DataFrame({col: level_values}), col, custom_palette=palette)
            color_palettes[col] = {'colors': colors, 'lut': lut}

        self.metadata_multi_idx_colors = pd.DataFrame({
            col: color_palettes[col]['colors'] for col in self.multi_index_all.names
        }).set_index(self.multi_index_all)

        return self.metadata_multi_idx_colors

    def plot_heatmap(self, row_colors=None, col_colors=None, normalize=False):
        v_min = self.distance_matrix_annotated.min().min()
        v_max = self.distance_matrix_annotated.max().max()

        distance_metric_range = self.metric_ranges.get(self.distance_metric, (v_min, v_max))
        if normalize:
            v_min, v_max = distance_metric_range

        g = sns.clustermap(self.distance_matrix_annotated, 
                    cmap="vlag", 
                    row_colors=row_colors, 
                    col_colors=col_colors,
                    vmin = v_min,
                    vmax = v_max,
                    # center=0.0,
                    dendrogram_ratio=(.1, .2),
                    cbar_pos=(.02, .32, .03, .2),
                    linewidths=.75, 
                    method='ward',
                    figsize=(12, 13))
        #todo add legend and ax as option with show 
        return g
    
    def run(self, **kwargs):
        self.process_metadata()
        self.create_multi_index()
        self.annotate_distance_matrix()
        self.assign_metadata_color()
        self.plot_heatmap(row_colors=self.metadata_multi_idx_colors, col_colors=self.metadata_multi_idx_colors, **kwargs)
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
            epsilon = 1e-50
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

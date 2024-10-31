import os
import glob
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.stats import entropy, wasserstein_distance
from scipy.cluster import hierarchy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tcrpeg_toolkit.utils import load_data, filter_kwargs_for_function

@dataclass
class Distribution:
    distributions: List[np.ndarray]
    ids: List[str] = field(default_factory=list)
    distance_matrix: Optional[np.ndarray] = None
    metadata: Optional[pd.DataFrame] = None
    distance_metric: Optional[str] = None
    distribution_type: Optional[str] = None
    
    def __repr__(self):
        return f"Distribution(distributions={len(self.distributions)}, ids={len(self.ids) if self.ids else None}, " \
               f"distribution_type={self.distribution_type}, " \
               f"distance_matrix={self.distance_matrix.shape if self.distance_matrix is not None else None}, " \
               f"distance_metric={self.distance_metric}, " \
               f"metadata={self.metadata.shape if self.metadata is not None else None})"

#todo Maybe separate distribution handler and distribution

class DistributionLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
        self.distributions = []
        self.ids = []

    def __repr__(self):
        return f"DistributionLoader(distributions={len(self.distributions)}, folder_path={self.folder_path})"

    def load_distributions(self, distribution_type='p_infer', message=False) -> Distribution:
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        
        for file_name in files:
            file_path = os.path.join(self.folder_path, file_name)
            data = np.load(file_path)

            distribution = self._extract_slot(data, distribution_type, file_name, message)
            if distribution is not None:
                self.distributions.append(distribution)
                self.ids.append(self._clean_sample_name(file_name, distribution_type))

        logging.info(f"Loaded {len(self.distributions)} distributions with their IDs")
        return Distribution(self.distributions, self.ids, distribution_type=distribution_type)

    def _extract_slot(self, data, distribution_type, file_name, message) -> Optional[np.ndarray]:
        if data.dtype.names and distribution_type in data.dtype.names:
            logging.info(f"Loaded structured array with names from {file_name}") if message else None
            return data[distribution_type]
        elif not data.dtype.names:
            logging.info(f"Loaded regular array from {file_name}") if message else None
            return data
        else:
            logging.warning(f"Slot '{distribution_type}' not found in structured array '{file_name}'")
            return None

    def _clean_sample_name(self, file_name, distribution_type):
        sample_name = os.path.splitext(file_name)[0].split(f'_{distribution_type}')[0]
        return sample_name.replace('_raw', '').replace('_structured', '')

    def load(self, distribution_type='p_infer', message=False) -> Distribution:
        """
        Executes the distribution handling process.
        This method loads the distributions based on the specified type and 
        optionally displays a message. It then loads metadata if a metadata 
        path is provided and returns a Distribution object.
        Args:
            distribution_type (str): The type of distribution to load. Defaults to 'p_infer'.
            message (bool): Flag to indicate whether to display a message. Defaults to False.
        Returns:
            Distribution: An instance of the Distribution class containing the loaded distributions 
                          and metadata (if available).
        """

        self.load_distributions(distribution_type, message)

        return Distribution(self.distributions, self.ids, distribution_type=distribution_type)

#todo this can go in utils and be used in other places
class MetadataLoader:
    def __init__(self, metadata):
        self.metadata = load_data(metadata)

    def _process_metadata(self, ids):
        if self.metadata is None:
            logging.warning("No metadata provided. Skipping metadata processing.")
            return None

        # Ensure no leading or trailing whitespace
        self.metadata.columns = self.metadata.columns.str.strip()

        try:
            if 'id' in self.metadata.columns:
                # Check if metadata 'id' column matches provided ids
                if not self.metadata['id'].isin(ids).all():
                    logging.warning("Metadata id does not match the provided ids. Not using metadata.")
                    self.metadata = None
                else:
                    logging.info("Metadata id matches the provided ids. Sorting by id.")
                    self.metadata = self.metadata.set_index('id').reindex(ids).reset_index()
            else:
                logging.warning("Metadata does not contain an 'id' column. Sorting by index.")
                self.metadata = self.metadata.reindex(ids).reset_index()

        except Exception as e:
            logging.warning(f"Error processing metadata: {e}. Not using metadata.")
            self.metadata = None

        return self.metadata
    
    def get_metadata(self):
        return self.metadata

    def load(self, ids=None):
        self._process_metadata(ids=ids)
        return self.metadata


class DistributionDataLoader:
    def __init__(self, folder_path, metadata=None):
        self.distribution_loader = DistributionLoader(folder_path)
        self.metadata_loader = MetadataLoader(metadata)

    def load(self, distribution_type='p_infer', message=False) -> Distribution:
        self.distribution_loader.load(distribution_type=distribution_type, message=message)

        if self.metadata_loader:
            self.metadata_loader.load(ids=self.distribution_loader.ids)

        return Distribution(self.distribution_loader.distributions, self.distribution_loader.ids, 
                            distribution_type=distribution_type, metadata=self.metadata_loader.metadata)

    def get_distributions(self):
        return self.distribution_loader.distributions

    def get_metadata(self):
        return self.metadata_loader.metadata if self.metadata_loader else None

    def get_ids(self):
        return self.distribution_loader.ids


class DistributionProcessor:
    def __init__(self, data, distribution_type='p_infer'):
        self.data = data
        self.distribution_type = distribution_type
        
        self.distributions = []
        self.ids = []
        self.distance_matrix = None
        self.distance_metric = None

        self._load_distributions()

    def _load_distributions(self):
        if self.data is None:
            raise ValueError("No data provided. Load data first.")
        
        if not all(hasattr(self.data, attr) for attr in ['distributions', 'ids']):
            try:
                self.data = DistributionLoader(self.data).load(distribution_type=self.distribution_type)
            except:
                logging.error("Failed to load data. Check the provided data.")
                return None

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
        return f"DistributionProcessor(distributions={len(self.distributions)}, ids={len(self.ids)}, " \
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
            
        Returns:
        --------
        Distribution
            A Distribution instance containing:
            - distributions: processed distribution data
            - ids: sample identifiers
            - distance_matrix: calculated distance matrix
            - distance_metric: name of metric used
        """
        if len(self.distributions) == 0:
            logging.error("No distributions found in the provided data.")
            return None
        
        if padding:
            self.pad_and_normalize_distributions()
        if smoothing:
            self.apply_smoothing_normalization()

        # Calculate the distance matrix
        self.calculate_distance_matrix(distance_metric, **kwargs)
        
        return Distribution(self.distributions, self.ids, distance_matrix=self.distance_matrix, distance_metric=self.distance_metric)

#todo separate distribution distance and metadata from plot

class DistributionHeatmapPlotter:
    # Class variable 
    metric_ranges = {
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
    
    def __init__(self, data, ids=None, metadata=None,  **kwargs):
        self.data = data
        self.metadata = load_data(metadata, message=False)
        self.distribution_object = None
        self.multi_index_all = None
        self.distance_matrix_annotated = None
        self.metadata_multi_idx_colors = None
        
        self._load_distributions_distance(**kwargs)

    def _load_distributions_distance(self, **kwargs):
        if isinstance(self.data, Distribution):
            logging.info("Loaded Distribution object.")
            self.distribution_object = self.data
        else:
            logging.info("Loading distributions and calculating distance matrix...")

            # Split kwargs explicitly for each method
            distance_init_kwargs = filter_kwargs_for_function(
                DistributionProcessor.__init__, kwargs)

            # Define the kwargs for the init method
            run_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in distance_init_kwargs}

            # Remove any init kwargs from the run kwargs to prevent overlap
            distance_run_kwargs = filter_kwargs_for_function(
                DistributionProcessor.run, run_kwargs)
            
            calculator = DistributionProcessor(
                self.data, **distance_init_kwargs)
            self.distribution_object = calculator.run(**distance_run_kwargs)

        if self.distribution_object is None:
            logging.error("Failed to load distributions or calculate distance matrix.")
            return None

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

    def assign_metadata_color(self, palette_mapping=None, filter_existing_values=True):
        if self.multi_index_all is None:
            logging.warning("No multi-index found. Skipping metadata color assignment.")
            return None
        #todo add option for palette here
        # Create distinct palettes for each column using different seaborn palettes
        palette_options = ['Set2', 'Dark2', 'Accent', 'Pastel1', 'Spectral', 'RdYlBu', 'PRGn', 'PiYG', 'BrBG']

        # If palette options is shorter than the number of columns, repeat the options
        if len(palette_options) < len(self.multi_index_all.names):
            palette_options = palette_options * (len(self.multi_index_all.names) // len(palette_options) + 1)
        
        color_palettes = {}

        for i, col in enumerate(self.multi_index_all.names):
            level_values = self.multi_index_all.get_level_values(col)
            palette = sns.color_palette(palette_options[i % len(palette_options)], n_colors=len(level_values.unique()))
            if palette_mapping and col in palette_mapping:
                if isinstance(palette_mapping[col], dict):
                    # If a dictionary mapping is provided, use it directly as the palette
                    palette = palette_mapping[col]
                    existing_values = level_values.values
                    if filter_existing_values:
                        palette = {k: v for k, v in palette.items() if k in existing_values}
                    # , index=pd.DataFrame({col: level_values}).index
                    colors = pd.Series(level_values).map(palette)
                    lut = palette
                else:
                    # If a seaborn palette name or list of colors is provided
                    palette = palette_mapping[col]
                    colors, lut = self._create_color_palette(pd.DataFrame({col: level_values}), col, custom_palette=palette)
            else:
                palette = sns.color_palette(palette_options[i % len(palette_options)], n_colors=len(level_values.unique()))
                colors, lut = self._create_color_palette(pd.DataFrame({col: level_values}), col, custom_palette=palette)
            color_palettes[col] = {'colors': colors, 'lut': lut}

        self.metadata_multi_idx_colors = pd.DataFrame({
            col: color_palettes[col]['colors'] for col in self.multi_index_all.names
        }).set_index(self.multi_index_all)
        
        self.color_palettes = color_palettes  # Store the color palettes with 'lut'

        return self.metadata_multi_idx_colors

    def plot_heatmap(self, row_colors=None, col_colors=None, normalize=False, save=False, output_dir=None):
        v_min = self.distance_matrix_annotated.min().min()
        v_max = self.distance_matrix_annotated.max().max()

        distance_metric_range = self.metric_ranges.get(self.distance_metric, (v_min, v_max))

        if distance_metric_range[0] == float('-inf') or distance_metric_range[1] == float('inf'):
            normalize = False

        if normalize:
            v_min, v_max = distance_metric_range

        center = (v_max + v_min) / 2
        
        if self.metadata is None:
            g = sns.clustermap(self.distance_matrix_annotated, 
                        cmap="vlag", 
                        row_colors=row_colors, 
                        col_colors=col_colors,
                        vmin = v_min,
                        vmax = v_max,
                        center=center,
                        dendrogram_ratio=(.1, .2),
                        cbar_pos=(1, .6, .03, .2),
                        linewidths=0,
                        method='ward',
                        figsize=(6, 6))
    
        else:
            g = sns.clustermap(self.distance_matrix_annotated, 
                        cmap="vlag", 
                        row_colors=row_colors, 
                        col_colors=col_colors,
                        vmin = v_min,
                        vmax = v_max,
                        center=center,
                        dendrogram_ratio=(.1, .2),
                        cbar_pos=(1, .6, .03, .2),
                        linewidths=0, 
                        xticklabels=False,  # Removes column names
                        yticklabels=False,
                        method='ward',
                        figsize=(6, 6))
            # Add black lines between the heatmap and the row/col colors
            for ax in [g.ax_col_colors, g.ax_row_colors]:
                if ax is not None:  # Check if the axis exists
                    for spine in ax.spines.values():
                        spine.set_visible(True) 
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1)
            
            # add legends
            legend_elements = []
            for level, palette in  self.color_palettes.items():
                lut = palette['lut']
                for value, color in lut.items():
                    legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f"{level}: {value}"))
             
            # Add the legend to the heatmap
            legend = g.ax_heatmap.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, .75), fontsize='small')
            legend.set_bbox_to_anchor((1.2, 0.25, 0.3, 0.5), transform=g.ax_heatmap.transAxes)
            
        if save is True:
            if output_dir is None:
                logging.warning("No output directory provided. Saving in working directory.")
                logging.info("Saving plot")
                g.savefig(f'{self.distance_metric}_heatmap_pinfer.pdf', format='pdf', bbox_inches='tight')
            elif not os.path.exists(output_dir):
                logging.warning(f"Output directory {output_dir} does not exist. Skipping plot saving.")
                return g
            else: 
                logging.info("Saving plot")
                g.savefig(f'{output_dir}/{self.distance_metric}_heatmap_pinfer.pdf', format='pdf', bbox_inches='tight')
        else:
            return g
    
    def run(self, palette_mapping=None, filter_existing_values=None, **kwargs):
        self.process_metadata()
        self.create_multi_index()
        self.annotate_distance_matrix()
        self.assign_metadata_color(palette_mapping=palette_mapping, filter_existing_values=filter_existing_values)
        plot = self.plot_heatmap(row_colors=self.metadata_multi_idx_colors, col_colors=self.metadata_multi_idx_colors, **kwargs)
        return plot

# class DistributionHandler:
#     return None

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

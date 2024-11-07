import os
import re
import sys
import logging
import warnings

import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap

# Suppress UMAP warnings
warnings.filterwarnings("ignore", message="n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.")

# Suppress OpenMP info messages
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'

from tcrpeg_toolkit.utils import load_data, filter_kwargs_for_function
from tcrpeg_toolkit.embedding_handler import EmbeddingHandler, Embedding

class UMAPGenerator():
    def __init__(self, data, metadata=None, key_metadata='id'):
        self.data = data
        self.metadata = load_data(metadata, message=False)
        self.key_metadata = key_metadata
        self.embeddings = None
        self.embeddings_reduced = None
        self.ids = None

        self._get_embeddings()

    def _get_embeddings(self):
        required_attrs = ['embeddings', 'ids', 'sequences']
        if not all(hasattr(self.data, attr) for attr in required_attrs):
            self.embedding_handler= EmbeddingHandler(self.data)
            self.embeddings = self.embedding_handler.get_embeddings()
            self.ids = self.embedding_handler.get_ids()
        else:
            print("Loaded Embedding Object")
            self.embeddings = self.data.embeddings
            self.ids = self.data.ids
            self.sequences = self.data.sequences
            try:
                self.metadata = self.data.metadata
            except:
                self.metadata = None
                logging.info("Metadata attribute is missing from data object.")

    def run(self, **kwargs):
        self.compute_umap(**kwargs)
        self.merge_metadata()

    def compute_umap(self, n_neighbors=15, min_dist=0.1, spread=1.0, n_components=2, metric='euclidean', densmap=False, output_metric='euclidean', target_metric=None, target_weight=0.5, labels=None, random_state=42):
        """
        Compute UMAP (Uniform Manifold Approximation and Projection) embeddings.

        Parameters:
        n_neighbors : int, optional (default=15)
            The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
        min_dist : float, optional (default=0.1)
            The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding.
        spread : float, optional (default=1.0)
            The effective scale of embedded points. In combination with `min_dist`, this determines how clustered/clumped the embedded points are.
        n_components : int, optional (default=2)
            The dimension of the space to embed into.
        metric : str or callable, optional (default='euclidean')
            The metric to use to compute distances in high dimensional space.
        densmap : bool, optional (default=False)
            Whether to use the density-augmented objective function to ensure uniform density in the embedding.
        output_metric : str or callable, optional (default='euclidean')
            The metric to use to compute distances in the low dimensional space.
        target_metric : str, optional (default=None)
            Metric for supervised UMAP. Set to 'categorical' for supervised UMAP.
        target_weight : float, optional (default=0.5)
            Weighting factor for supervised UMAP (balances the influence of the target labels and the original data structure. You can adjust this value between 0 and 1).
        labels : array, optional (default=None)
            Optional categorical labels for supervised UMAP.
        random_state : int, optional (default=42)
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.

        Returns:
        np.ndarray
            The reduced embeddings.
        """

        umap_params = {
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'spread': spread,
        'n_components': n_components,
        'metric': metric,
        'densmap': densmap,
        'output_metric': output_metric,
        'random_state': random_state
        }

        if target_metric is not None:
            umap_params['target_metric'] = target_metric
            umap_params['target_weight'] = target_weight

        reducer = umap.UMAP(**umap_params)
        
        if labels is not None:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)

        else:
            encoded_labels = None

        # reducer = umap.UMAP(
        #     n_neighbors=n_neighbors,
        #     min_dist=min_dist,
        #     spread=spread,
        #     n_components=n_components,
        #     metric=metric,
        #     densmap=densmap,
        #     output_metric=output_metric,
        #     random_state=random_state
        # )
        
        self.embeddings_reduced = reducer.fit_transform(self.embeddings, y=encoded_labels)
        return self.embeddings_reduced

    def merge_metadata(self):
        if self.embeddings_reduced is not None:
            umap_columns = [f'UMAP_{i+1}' for i in range(self.embeddings_reduced.shape[1])]
            self.umap_data = pd.DataFrame(self.embeddings_reduced,
                                                    columns = umap_columns)

        if self.ids is not None:
            # Ensure index consistency
            try:
                self.umap_data.index = self.ids.index
            except:
                pass
                # print(f"Warning: Could not set index due to: {e}")
        
            # Insert id as the first column
            self.umap_data.insert(0, 'id', self.ids)

            if self.umap_data['id'].isna().sum() > 0:
                print("Warning: Mismatch in id between UMAP data and metadata.")
        
        if self.metadata is not None and self.key_metadata is not None and self.ids is not None:
            try:
                self.umap_data['id'] = self.umap_data['id'].astype('str')
                self.metadata[self.key_metadata] = self.metadata[self.key_metadata].astype('str')
                self.umap_data = self.umap_data.merge(self.metadata, left_on='id', right_on=self.key_metadata, how='left')
            except:
                logging.warning("Metadata not merged with UMAP data.")

    def plot_umap(self, ax=None, hue=None, palette=None, alpha=1.0, show=True,  output_file=None, scatter_kwargs=None, legend_kwargs=None, **kwargs):
        num_dimensions = self.embeddings_reduced.shape[1]

        # Set default values or extract from kwargs
        s = kwargs.pop('s', 20)  
        # Set marker size and remove it from kwargs, default is 20 if s is not provided in kwargs 

        # Determine the color mapping based on the palette type
        if palette is not None and isinstance(palette, str):
            try:
                # Use Seaborn's built-in color scales for string palettes
                if hue is not None:
                    num_colors = self.umap_data[hue].nunique()
                    palette = sns.color_palette(palette, num_colors)
                else:
                    palette = sns.color_palette(palette)
            except ValueError:
                # Fallback to default palette if the provided name is invalid
                palette = sns.color_palette('husl', num_colors)
        elif palette is not None and isinstance(palette, dict):
            # Use the provided dictionary palette
            palette = palette
        else:
            if hue is not None:
                num_colors = self.umap_data[hue].nunique()
                palette = sns.color_palette('husl', num_colors)  # Default palette with dynamic number of colors
            else:
                palette = 'husl'

        # Transform hue to categorical if it's not already
        if hue is not None and not isinstance(self.umap_data[hue], pd.Categorical):
            self.umap_data[f"{hue}_categorical"] = pd.Categorical(self.umap_data[hue])

        # Plotting for 2 dimensions UMAP
        if num_dimensions == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 5))

             # Get additional arguments for sns.scatterplot and ax.legend
            # scatter_kwargs = filter_kwargs_for_function(sns.scatterplot, kwargs)
            # legend_kwargs = filter_kwargs_for_function(ax.legend, kwargs)

            sns.scatterplot(x=self.umap_data['UMAP_1'],
                            y=self.umap_data['UMAP_2'],
                            hue=self.umap_data[f"{hue}_categorical"] if hue is not None else None,
                            palette=None if hue is None else (palette if palette is not None else 'husl'),
                            ax=ax,
                            s=s,
                            alpha=alpha,
                            edgecolor = 'black',
                            **scatter_kwargs)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')

            if hue is not None:
                handles, labels = ax.get_legend_handles_labels()
                
                if palette is not None and isinstance(palette, dict):
                    # Sort legend handles and labels based on the original order of the palette
                    # Transform keys and label to string
                    palette = {str(k): v for k, v in palette.items()}
                    labels = [str(label) for label in labels]
                    # Sort labels based on the order of the palette
                    order = [list(palette.keys()).index(label) for label in labels]
                    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: order[labels.index(x[1])])
                    handles, labels = zip(*sorted_handles_labels)
                else:
                    # Ensure all unique hues are captured
                    unique_hues = self.umap_data[f"{hue}_categorical"].unique().tolist()
                    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: unique_hues.index(x[1]) if x[1] in unique_hues else float('inf'))
                    handles, labels = zip(*sorted_handles_labels)

                # Calculate dynamic number of columns
                num_items = len(handles)
                ncol = min(4, max(1, num_items // 3))

                ax.legend(handles=handles, labels=labels, title=hue,
                          bbox_to_anchor=(0.5, 1.3),
                          loc='best',
                          ncol=ncol, 
                          frameon=False,
                          borderpad=1,
                          fontsize=12,
                          **legend_kwargs)

        # Plotting for 3 dimensions UMAP
        #todo fix legend for this
        elif num_dimensions == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Convert categorical hue values to numerical if necessary
            if hue is not None and self.umap_data[hue].dtype == 'object':
                hue_values = pd.Categorical(self.umap_data[hue]).codes
            else:
                hue_values = self.umap_data[hue] if hue is not None else None
    
            # Assuming `palette` can be a dict, a seaborn color palette, or a string for a matplotlib colormap

            if isinstance(palette, dict):
                # Creating a color map from the dictionary
                unique_hues = self.umap_data[hue].unique()
                color_map = {hue_val: palette.get(hue_val, '#000000') for hue_val in unique_hues}
                colors = [color_map[val] for val in self.umap_data[hue]]
                cmap = None  # No colormap for custom color lists
            else:
                # Handle seaborn or string palettes which imply a colormap
                colors = hue_values  # Direct mapping from hue values to colors using a colormap
                if isinstance(palette, str):
                    cmap = palette  # Use the string directly as the colormap name
                else:
                    cmap = None  # If seaborn palette object or default, handle via `c` parameter in scatter

            
            # # Create color map from palette if it's a dictionary
            # if isinstance(palette, dict):
            #     unique_hues = self.umap_data[hue].unique()
            #     color_map = {hue_val: palette.get(hue_val, '#000000') for hue_val in unique_hues}
            #     colors = [color_map[val] for val in self.umap_data[hue]]
            # elif isinstance(palette, sns.palettes._ColorPalette):
            #     colors = list(palette)
            # else:
            #     colors = None if hue is None else (palette if palette is not None else 'husl')
            
            # Create the scatter plot with labels
            scatter = ax.scatter(
                self.umap_data['UMAP_1'],
                self.umap_data['UMAP_2'],
                self.umap_data['UMAP_3'],
                c=colors,
                cmap=cmap,
                s=s,
                alpha=alpha
            )

            # ax.scatter3D(self.umap_data['UMAP_1'],
                        #  self.umap_data['UMAP_2'],
                        #  self.umap_data['UMAP_3'])
        
            # Add labels for the legend
            if hue is not None:
                unique_labels = self.umap_data[hue].unique()
                if isinstance(palette, dict):
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10) for label in unique_labels]
                else:
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10) for label in unique_labels]
                    # Number of unique labels
                    num_labels = len(self.umap_data[hue].unique())

                    # Get the 'husl' palette from Seaborn and create a colormap
                    husl_palette = sns.color_palette("husl", num_labels)  # This will fetch the HUSL palette
                    husl_cmap = ListedColormap(husl_palette)  # Create a colormap from the list of colors

                    # Now use this colormap in your plotting
                    handles = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=husl_cmap(i / num_labels), markersize=10)
                        for i in range(num_labels)
                    ]

                    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.husl(i / len(unique_labels)), markersize=10) for i in range(len(unique_labels))]                

            ax.legend(handles, unique_labels, title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            if not isinstance(palette, dict):
                plt.colorbar(scatter)

        # Plotting for higher than 3 dimensions UMAP
        else:
            num_plots = num_dimensions * (num_dimensions - 1) // 2
            num_rows = (num_plots + 1) // 2
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
            axs = axs.flatten()
            
            plot_idx = 0
            for i in range(num_dimensions):
                for j in range(i + 1, num_dimensions):
                    ax = axs[plot_idx]
                    sns.scatterplot(
                        x=self.umap_data[f'UMAP_{i+1}'],
                        y=self.umap_data[f'UMAP_{j+1}'],
                        hue=self.umap_data[hue] if hue is not None else None,
                        palette=palette if palette is not None else 'husl',
                        ax=ax, 
                        s=s,
                        alpha=alpha
                    )
                    ax.set_xlabel(f'UMAP {i+1}')
                    ax.set_ylabel(f'UMAP {j+1}')
                    plot_idx += 1

            # Create a single legend for all subplots
            if hue is not None:
                handles, labels = axs[0].get_legend_handles_labels()
                order = [list(palette.keys()).index(label) for label in labels]
                sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: order[labels.index(x[1])])
                handles, labels = zip(*sorted_handles_labels)

                # Calculate dynamic number of columns
                num_items = len(handles)
                ncol = min(4, max(1, num_items // 3))

                fig.legend(handles=handles, labels=labels, title=hue,
                          bbox_to_anchor=(0.5, 1.3),
                          loc='upper center',
                          ncol=ncol, 
                          frameon=False,
                          borderpad=1)

                for ax in axs:
                    ax.get_legend().remove()

        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        
        if show and ax is None:
            plt.show()
        if not show:
            plt.close()
    
    def plot_interactive_umap(self, hue=None, palette=None, size=None, symbol=None, opacity=1.0, marker_size=5, output_file=None, show=True):
        """
        Create an interactive UMAP plot using Plotly
        
        Parameters:
        - hue: str, column name for color encoding
        - palette: dict, a dictionary mapping hue values to colors or seaborn palette name (optional)
        - size: str, column name for marker size
        - symbol: str, column name for marker symbol
        - opacity: float, marker opacity
        - output_file: str, the filename to save the interactive plot (if provided)
        - show: bool, whether to display the plot
        """

        umap_dim = self.embeddings_reduced.shape[1]
        
        try:
            if umap_dim not in [2, 3]:
                raise ValueError("This function only supports 2D and 3D UMAP embeddings")
        except ValueError as e:
            logging.error(f"Error: {str(e)}")
            return

        # Convert the hue column to a categorical type if it's not already
        if hue is not None and hue in self.umap_data.columns:
            self.umap_data[f"{hue}_categorical"] = pd.Categorical(self.umap_data[hue])

        # Determine the color mapping based on the palette type
        if palette is not None and isinstance(palette, str):
            # Use Plotly's built-in color scales for string palettes
            color_discrete_map = px.colors.qualitative.__dict__.get(palette, None)
            if color_discrete_map is None:
                # Fallback to default palette if the provided name is invalid
                color_discrete_map = px.colors.qualitative.Plotly
        elif palette is not None and isinstance(palette, dict):
            # Use the provided dictionary palette
            color_discrete_map = palette
        else:
            color_discrete_map = px.colors.qualitative.Plotly  # Default palette

        if umap_dim == 2:
            fig = px.scatter(self.umap_data, x='UMAP_1', y='UMAP_2', 
                            color=f"{hue}_categorical", symbol=symbol, opacity=opacity,
                            size=size, size_max=10,
                            color_discrete_map={str(i): color for i, color in enumerate(color_discrete_map)})
            fig.update_traces(marker=dict(size=marker_size))
            fig.update_layout(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                legend_title=f'{hue} and {symbol}' if symbol else hue
            )

        elif umap_dim == 3:
            fig = px.scatter_3d(self.umap_data, x='UMAP_1', y='UMAP_2',
                                z='UMAP_3', color=f"{hue}_categorical",
                                symbol=symbol, opacity=opacity,
                                size=size, size_max=10,
                                color_discrete_map={str(i): color for i, color in enumerate(color_discrete_map)})
            fig.update_traces(marker=dict(size=marker_size))
            fig.update_layout(
                scene=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3'
                ),
                legend_title=f'{hue} and {symbol}' if symbol else hue,
                margin=dict(l=0, r=0, b=0, t=0)  # Tight layout
            )

        # Ensure legend order is preserved if palette is a dictionary
        if palette is not None and symbol is None and size is None and isinstance(palette, dict):
            fig.data = sorted(fig.data, key=lambda trace: list(palette.keys()).index(trace.name))

        # elif umap_dim == 3:
        #     fig = go.Figure()
        #     for label, color in palette.items():
        #         mask = self.umap_data[hue] == label
        #         fig.add_trace(go.Scatter3d(
        #             x=self.umap_data.loc[mask, 'UMAP_1'],
        #             y=self.umap_data.loc[mask, 'UMAP_2'],
        #             z=self.umap_data.loc[mask, 'UMAP_3'],
        #             mode='markers',
        #             name=label,
        #             marker=dict(color=color, size=size if size else 5),
        #             showlegend=True
        #         ))
        #     fig.update_layout(
        #         scene=dict(
        #             xaxis_title='UMAP 1',
        #             yaxis_title='UMAP 2',
        #             zaxis_title='UMAP 3'
        #         ),
        #         width=900,
        #         height=700,
        #         margin=dict(r=20, b=10, l=10, t=10)
        #     )

        
        if output_file:
            if  not output_file.endswith('.html'):
                output_file = output_file.split('.')[0] + '.html'
            fig.write_html(output_file)
            print(f"Interactive UMAP plot saved as {output_file}")

        if show:
            fig.show()

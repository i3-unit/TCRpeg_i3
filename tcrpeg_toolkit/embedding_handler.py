import os
import re
import logging
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from tcrpeg_toolkit.utils import load_data


#improve extend from a base class

# class BaseEmbedding:
#     def __init__(self, embeddings, ids, sequences):
#         self.embeddings = embeddings
#         self.ids = ids 
#         self.sequences = sequences
    
#     def __len__(self):
#         return len(self.embeddings)
    
#     def __iter__(self):
#         return iter(zip(self.embeddings, self.ids, self.sequences))

# class Embedding(BaseEmbedding):
#     # Add Embedding specific methods

# class EmbeddingHandler(BaseEmbedding): 
#     def __init__(self, data, name, metadata=None, key_metadata=None, key_embedding=None):
#         super().__init__(data.embeddings, data.ids, data.sequences)
#         self.name = name
#         self.metadata = metadata
#         # Add handler specific attributes


# class Embedding(BaseEmbedding):
#     def get_embedding_by_id(self, id):
#         idx = np.where(self.ids == id)[0]
#         return self.embeddings[idx] if len(idx) > 0 else None
        
#     def get_embedding_by_sequence(self, sequence):
#         idx = np.where(self.sequences == sequence)[0]
#         return self.embeddings[idx] if len(idx) > 0 else None
        
#     def filter_embeddings(self, mask):
#         return self.embeddings[mask]
        
#     def normalize_embeddings(self):
#         return sklearn.preprocessing.normalize(self.embeddings)
        
#     def reduce_dimensions(self, method='mean'):
#         if method == 'mean':
#             return np.mean(self.embeddings, axis=1)

# class TCREmbeddingHandler(EmbeddingHandler):
#     def tcr_specific_method(self):
#         # Add TCR-specific functionality
#         pass

# class AntibodyEmbeddingHandler(EmbeddingHandler):
#     def antibody_specific_method(self):
#         # Add antibody-specific functionality
#         pass


class Embedding:
    def __init__(self, embeddings, ids=None, sequences=None):
        self.embeddings = embeddings
        self.ids = ids
        self.sequences = sequences

    def __repr__(self):
        return f"Embedding(embeddings_shape={self.embeddings.shape}, ids_shape={self.ids.shape}, sequences_shape={self.sequences.shape if self.sequences is not None else 'None'})"

    def __len__(self):
        return len(self.embeddings)

    def __add__(self, other):
        concatenated_embeddings = np.concatenate((self.embeddings, other.embeddings), axis=0)
        concatenated_ids = np.concatenate((self.ids, other.ids), axis=0) if self.ids is not None and other.ids is not None else None
        concatenated_sequences = np.concatenate((self.sequences, other.sequences), axis=0) if self.sequences is not None and other.sequences is not None else None
        return Embedding(concatenated_embeddings, concatenated_ids, concatenated_sequences)

    def __getitem__(self, mask):
        return Embedding(embeddings=self.embeddings[mask],
                        ids=self.ids[mask],
                        sequences=self.sequences[mask])
    
    def __iter__(self):
        return iter(zip(self.embeddings, self.ids, self.sequences))
    
    def __contains__(self, item):
        return item in self.sequences or item in self.ids

    def filter_by_ids(self, ids_list):
        if self.ids is not None:
            mask = np.isin(self.ids, ids_list)
            return Embedding((self.embeddings[mask], self.ids[mask], 
                          self.sequences[mask] if self.sequences is not None else None))
        else:
            return None

    def cosine_similarity(self, other):
        """
        Calculate the cosine similarity between the embeddings of the current object and another object (useful for high-dimensional data).
        Parameters:
        other (object): Another object with an 'embeddings' attribute of the same dimensionality as the current object.
        Returns:
        numpy.ndarray: A matrix of cosine similarity scores between the embeddings of the current object and the other object.
        Raises:
        ValueError: If the embeddings of the two objects do not have the same dimensionality.
        """
        # np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(self.embeddings.shape[1])
        if self.embeddings.shape[1] != other.embeddings.shape[1]:
            raise ValueError("Embeddings must have the same dimensionality")

        # # Normalize vectors
        # norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # normalized_embeddings = embeddings / norms
        
        # # Compute cosine similarity matrix
        # return np.dot(normalized_embeddings, normalized_embeddings.T)

        #todo check if current implentation is correct        
        dot_product = np.dot(self.embeddings, other.embeddings.T)
        norm_self = np.linalg.norm(self.embeddings, axis=1)
        norm_other = np.linalg.norm(other.embeddings, axis=1)
        return dot_product / (norm_self[:, np.newaxis] * norm_other[np.newaxis, :])

    def euclidean_distance(self, other):
        """
        Calculate the Euclidean distance between the embeddings of the current object and another object (useful for lower-dimensional data).
        Parameters:
        other (object): Another object with an 'embeddings' attribute of the same dimensionality as the current object.
        Returns:
        numpy.ndarray: A matrix of Euclidean distance scores between
        the embeddings of the current object and the other object.
        Raises:
        ValueError: If the embeddings of the two objects do not have the same dimensionality.
        """
        if self.embeddings.shape[1] != other.embeddings.shape[1]:
            raise ValueError("Embeddings must have the same dimensionality")
        return np.linalg.norm(self.embeddings - other.embeddings, axis=1)

    def normalize(self):
        norms = np.linalg.norm(self.embeddings, axis=1)
        return Embedding(self.embeddings / norms[:, np.newaxis], self.ids, self.sequences)

    def save(self, filename):
        np.save(filename, embeddings=self.embeddings, ids=self.ids, sequences=self.sequences)
        logging.info(f"Embedding data saved to {filename}")


class EmbeddingHandler():
    def __init__(self, data, metadata=None, key_embedding='id', key_metadata='id', name=None):
        self.data = data
        self.metadata = metadata
        self.key_embedding = key_embedding
        self.key_metadata = key_metadata
        self.name = name
        self.embedding_base = None 
        self.embeddings = None
        self.ids = None
        self.sequences = None

        self.required_attrs = ['embeddings', 'ids', 'sequences']
        self._load_data()
        self._load_metadata()        

    def _load_data(self):
        if not all(hasattr(self.data, attr) for attr in self.required_attrs):
            self.data = load_data(self.data, message = False)
            self.embedding_base = self._extract_embeddings()
        else:
            self.embeddings = self.data.embeddings
            self.ids = self.data.ids
            self.sequences = self.data.sequences
            #todo check if i can do this then get attribute
            self.embedding_base = self.data

    def _load_metadata(self):
        if self.metadata is not None:
            self.metadata = load_data(self.metadata, message=False)
            self._sort_metadata()

    def _extract_embeddings(self):
        if isinstance(self.data, pd.DataFrame):
            # Extract columns with numerical values in their names
            numerical_columns = [col for col in self.data.columns if 
                                    any(char.isdigit() for char in col)]
            self.embeddings = self.data[numerical_columns].values
            self.ids = self.data['id'].astype(str) if 'id' in self.data.columns else np.arrange(self.embeddings.shape[0])
            self.sequences = self.data['sequence'] if 'sequence' in  self.data.columns else None

        elif isinstance(self.data, np.ndarray):
            if self.data.dtype.names is not None:
                self.embeddings = self.data['embedding'] if 'embedding' in self.data.dtype.names else None
                self.ids = self.data['id'].astype(str) if 'id' in self.data.dtype.names else None
                self.sequences = self.data['sequence'] if 'sequence' in self.data.dtype.names else None
            else:
                self.embeddings = self.data
                self.ids = np.arange(self.embeddings.shape[0])
                logging.warning("Generated random ids")
        else:
            raise ValueError("Unsupported data type.")
        return  Embedding(self.embeddings, self.ids, self.sequences)

    def __repr__(self):
        return (f"EmbeddingHandler(embeddings_shape={self.embeddings.shape}, ids_shape={self.ids.shape if self.ids is not None else 'None'}, "
        f"sequences_shape={self.sequences.shape if self.sequences is not None else 'None'}, metadata_shape={self.metadata.shape if self.metadata is not None else 'None'}, name={self.name if self.name else 'None'})")

        # return f"EmbeddingHandler(embeddings_shape={self.embeddings.shape}, ids_shape={self.ids.shape if self.ids is not None else 'None'}, sequences_shape={self.sequences.shape if self.sequences is not None else 'None'}, metadata_shape={self.metadata.shape if self.metadata is not None else 'None'})"

    def __add__(self, other_handler):
        concatenated_embeddings, concatenated_ids, concatenated_sequences = self.concatenate_embeddings(other_handler)
        concatenated_metadata =  self.concatenate_metadata(other_handler)
        return EmbeddingHandler(data=Embedding(concatenated_embeddings, concatenated_ids, concatenated_sequences), metadata=concatenated_metadata,
                                name=f"{self.name}_{other_handler.name}" if self.name and other_handler.name else None,
                                key_embedding = self.key_embedding, key_metadata = self.key_metadata)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, mask):
        return EmbeddingHandler(data=Embedding(self.embeddings[mask], 
                                         ids=self.ids[mask],
                                         sequences=self.sequences[mask]),
                           name=self.name,
                           metadata=self.metadata[mask] if self.metadata is not None else None,
                           key_metadata=self.key_metadata,
                           key_embedding=self.key_embedding)

    def __iter__(self):
        return iter(zip(self.embeddings, self.ids, self.sequences))

    def __contains__(self, item):
        return item in self.sequences or item in self.ids
    
    def _sort_metadata(self):
        #todo simplify this to match id and sequence for embedding and metadata
        try:
            if self.key_embedding == 'id':
                # self.metadata[self.key_metadata] = self.metadata[self.key_metadata].astype(str)
                # Sort metadata based on the IDs to match the order of the embeddings
                # self.metadata = self.metadata.set_index(self.key_metadata).loc[self.ids].reset_index()
                # self.metadata = self.metadata.set_index(self.key_metadata).reindex(self.ids).reset_index()
                # Convert to string and sort in one operation
                self.metadata = (self.metadata
                                .assign(**{self.key_metadata: lambda x: x[self.key_metadata].astype(str)})
                                .set_index(self.key_metadata)
                                .reindex(self.ids)
                                .reset_index())
                # logging.info("Sorted metadata based on id.")
                
            elif self.key_embedding == 'sequence':
                # self.metadata = self.metadata.set_index(self.key_sequence).reindex(self.sequences).reset_index()
                self.metadata = (self.metadata
                                .assign(**{self.key_metadata: lambda x: x[self.key_metadata].astype(str)})
                                .set_index(self.key_metadata)
                                .reindex(self.sequences)
                                .reset_index())
                
                self.metadata.insert(0, 'id', self.ids) if 'id' not in self.metadata.columns else None
                # logging.info("Sorted metadata based on sequence.")

            else:
                raise ValueError("Invalid key_embedding. Choose either 'id' or 'sequence'.")

        except:
            logging.warning("Error while sorting the metadata, none will be used.")
            self.metadata = None

    def concatenate_embeddings(self, other_handler):
        if not all(hasattr(other_handler, attr) for attr in self.required_attrs):
            raise ValueError("The provided object does not have the required attributes of an EmbeddingHandler.")

        # Concatenate embeddings
        concatenated_embeddings = np.concatenate((self.embeddings, other_handler.embeddings), axis=0)

        # Concatenate IDs if they exist
        if self.ids is not None and other_handler.ids is not None:
            # Convert IDs to strings and add suffixes
            # ids_self = np.array([str(id) + ":" + self.name for id in self.ids]) if self.name else self.ids
            # ids_other = np.array([str(id) + ":" + other_handler.name for id in other_handler.ids]) if other_handler.name else other_handler

            # concatenated_ids = np.concatenate((ids_self, ids_other), axis=0)

            ids_self = np.array([str(id) + ":" + self.name for id in self.ids]).reshape(-1) if self.name else self.ids.reshape(-1)
            ids_other = np.array([str(id) + ":" + other_handler.name for id in other_handler.ids]).reshape(-1) if other_handler.name else other_handler.ids.reshape(-1)

            concatenated_ids = np.concatenate((ids_self, ids_other), axis=0)
        else: 
            concatenated_ids = None

        # Concatenate sequences if they exist
        if self.sequences is not None and other_handler.sequences is not None:
            concatenated_sequences = np.concatenate((self.sequences, other_handler.sequences), axis=0)
        else:
            concatenated_sequences = None
        
        return concatenated_embeddings, concatenated_ids, concatenated_sequences
    
    def concatenate_metadata(self, other_handler):
        if self.metadata is not None and other_handler.metadata is not None:
            # Merge metadata on the ID column
            metadata_self_copy = self.metadata.copy()
            metadata_other_copy = other_handler.metadata.copy()

            original_key = f"original_{self.key_metadata}"
            metadata_self_copy[f"original_{self.key_metadata}"] = metadata_self_copy[self.key_metadata]
            metadata_other_copy[f"original_{self.key_metadata}"] = metadata_other_copy[self.key_metadata]

            metadata_self_copy["data_origin"] = self.name
            metadata_other_copy["data_origin"] = other_handler.name

            # Only modify the key if it's not the sequence
            if self.key_metadata != 'sequence':
                metadata_self_copy[self.key_metadata] = metadata_self_copy[self.key_metadata].apply(lambda x: str(x) + ":" + self.name)
                metadata_other_copy[self.key_metadata] = metadata_other_copy[self.key_metadata].apply(lambda x: str(x) + ":" + other_handler.name)
            else:
                metadata_self_copy[f"{self.key_metadata}_data_origin"] = metadata_self_copy[self.key_metadata].apply(lambda x: str(x) + ":" + self.name)
                metadata_other_copy[f"{self.key_metadata}_data_origin"] = metadata_other_copy[self.key_metadata].apply(lambda x: str(x) + ":" + other_handler.name)
        
            # metadata_self_copy[self.key_metadata] = metadata_self_copy[self.key_metadata].apply(lambda x: str(x) + ":" + self.name)
            # metadata_other_copy[self.key_metadata] = metadata_other_copy[self.key_metadata].apply(lambda x: str(x) + ":" + other_handler.name)
            
            concatenated_metadata = pd.concat([metadata_self_copy, metadata_other_copy], axis=0).reset_index(drop=True)

            return concatenated_metadata

        elif all([self.ids is not None, self.name is not None, 
         other_handler.ids is not None, other_handler.name is not None]):
    
            ids_self = [f"{id}:{self.name}" for id in self.ids]
            ids_other = [f"{id}:{other_handler.name}" for id in other_handler.ids]

            concatenated_ids = np.concatenate((ids_self, ids_other))
            concatenated_metadata = pd.DataFrame({
            'id': concatenated_ids,
            'data_origin': np.concatenate((
                np.full(len(self.ids), self.name),
                np.full(len(other_handler.ids), other_handler.name)
            )),
            'original_id': np.concatenate((self.ids, other_handler.ids))
            })
            return concatenated_metadata

        return None

    def set_clusters(self, clusters, cluster_name='cluster'):
        if self.metadata is None:
            self.metadata = pd.DataFrame({cluster_name: clusters})
        else:
            self.metadata[cluster_name] = clusters

    def update_metadata(self, data_to_add, column_name):
        if self.metadata is None:
            if self.ids is None:
                self.metadata = pd.DataFrame()
            self.metadata = pd.DataFrame({'id': self.ids})
        self.metadata[column_name] = data_to_add

    def get_embeddings(self):
        return self.embeddings
    
    def get_ids(self):
        return self.ids

    def get_sequences(self):
        return self.sequences

    def get_metadata(self):
        return self.metadata

    # todo add save structured array, raw embeddings and maybe a new one structured with metadata

    def save_metadata(self, output_file):
        if self.metadata is not None:
            self.metadata.to_csv(output_file, index=False)
            logging.info(f"Metadata saved to {output_file}")
        else:
            logging.warning("No metadata to save.")

    #todo add documentation for the index: will be reset and the old one will be saved in a new index column

    #fix maybe this is better 
    # def downsample_embeddings(self, sample_size=None):
    #     if sample_size is not None and self.embeddings.shape[0] > sample_size:
    #         logging.info(f"Embeddings were downsampled to {sample_size}.")
    #         selected_indices = np.random.choice(len(self.metadata), sample_size, replace=False)
    #         boolean_mask = np.zeros(len(self.metadata), dtype=bool)
    #         boolean_mask[selected_indices] = True
    #     else:
    #         logging.warning(f"Embeddings were not downsampled.")
    #         boolean_mask = np.ones(self.embeddings.shape[0], dtype=bool)

    #     return EmbeddingHandler(data=Embedding(self.embeddings[boolean_mask], ids=self.ids[boolean_mask],
    #                                         sequences=self.sequences[boolean_mask]),
    #                         name=self.name, 
    #                         metadata=self.metadata[boolean_mask].reset_index(drop=True) if self.metadata is not None else None,
    #                         key_metadata=self.key_metadata, 
    #                         key_embedding=self.key_embedding)

    def downsample_embeddings(self, sample_size=None):
        if sample_size is not None and self.embeddings.shape[0] > sample_size:
            logging.info(f"Embeddings were downsampled to {sample_size}.")
            mask = np.random.choice(self.embeddings.shape[0], sample_size, replace=False)
        else:
            logging.warning(f"Embeddings were not downsampled.")
            mask = np.arange(self.embeddings.shape[0])

        return EmbeddingHandler(data=Embedding(self.embeddings[mask], ids=self.ids[mask], 
                                               sequences=self.sequences[mask]),
                                name=self.name, 
                                metadata=self.metadata.iloc[mask].reset_index(drop=False) if self.metadata is not None else None,
                                key_metadata=self.key_metadata, key_embedding=self.key_embedding)

    #todo add option for the filter to keep original index or not
            
    def filter_by_ids(self, ids_list):
        if self.ids is not None:
            mask = np.isin(self.ids, ids_list)
            return EmbeddingHandler(data=Embedding(self.embeddings[mask], ids=self.ids[mask], sequences=self.sequences[mask]),
                                    name=self.name, 
                                    metadata=self.metadata[mask].reset_index(drop=False) if self.metadata is not None else None,
                                    key_metadata=self.key_metadata, key_embedding=self.key_embedding)
        else:
            return None

    def filter_by_sequences(self, sequences):
        if self.sequences is not None:
            mask = np.isin(self.sequences, sequences)
            return EmbeddingHandler(data=Embedding(self.embeddings[mask], ids=self.ids[mask], sequences=self.sequences[mask]),
                                    name=self.name,
                                    metadata=self.metadata[mask].reset_index(drop=False) if self.metadata is not None else None,
                                    key_metadata=self.key_metadata, key_embedding=self.key_embedding)
        else:
            return None

    def filter_by_metadata(self, key, values):
        """
        Filters the embeddings based on the provided metadata key and values.

        Args:
            key (str): The metadata key to filter by.
            values (Union[str, List[str]]): The value or list of values to filter the metadata by.

        Returns:
            EmbeddingHandler: A new instance of EmbeddingHandler with filtered embeddings and metadata.
            None: If metadata is not available.

        Raises:
            KeyError: If the provided key is not found in the metadata.
        """
        if self.metadata is not None:
            if isinstance(values, list):
                mask = self.metadata[key].isin(values)
                values_name = ":".join(values)
                name = f"{self.name}_{key}_{values_name}" if self.name else f"{key}_{values_name}"
            else:
                mask = self.metadata[key] == values
                name = f"{self.name}_{key}_{values}" if self.name else f"{key}_{values}"
            return EmbeddingHandler(data=Embedding(self.embeddings[mask], self.ids[mask], self.sequences[mask]),
                                    metadata=self.metadata[mask].reset_index(drop=True), name=name)
        else:
            return None

    def calculate_similarity_difference(self, other_handler):
        """
        Calculate the similarity and distance between the embeddings of two handlers.

        This method computes the cosine similarity and Euclidean distance between 
        the embedding bases of the current handler and another handler.

        Args:
            other_handler (Handler): Another handler object to compare with.

        Returns:
            tuple: A tuple containing:
                - similarity (float): The cosine similarity between the embeddings.
                - distance (float): The Euclidean distance between the embeddings.
        """
        similarity_matrix = cosine_similarity(self.embeddings, other_handler.embeddings)
        # similarity = self.embedding_base.cosine_similarity(other_handler.embedding_base)
        # distance = self.embedding_base.euclidean_distance(other_handler.embedding_base)
        distance = None
        #todo check if better to do calculation only once and if better to seperate them
        return similarity_matrix, distance

    def plot_similarity_heatmap(self, other_handler, title="Similarity Heatmap", calculate_max_similarity=True):
        similarity_matrix, _ = self.calculate_similarity_difference(other_handler)

        #todo add this to a self variable so only one calculation

        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap="RdBu", annot=False,
                    fmt=".2f", vmin=-1, vmax=1,
                    cbar_kws={'label': 'Cosine Similarity'})

        if self.name and other_handler.name:
            # plt.title(f"{self.name} vs {other_handler.name}")
            plt.xlabel(f'Embeddings from {other_handler.name}')
            plt.ylabel(f'Embeddings from {self.name}')

        plt.title(title)
        plt.show()

        if calculate_max_similarity:
            max_similarity = np.max(similarity_matrix)
            max_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

            print(f"Highest similarity: {max_similarity}")
            if self.name and other_handler.name:
                print(f"Between embedding {max_indices[0]} from {self.name} and embedding {max_indices[1]} from {other_handler.name}")
            else:
                print(f"Between embedding {max_indices[0]} and embedding {max_indices[1]}")

    # def get_metadata_by_id(self, id):
    #     if self.metadata is not None and self.key_metadata is not None:
    #         return self.metadata[self.metadata[self.key_metadata] == id]
    #     else:
    #         return None

    # def get_metadata_by_id_list(self, id_list):
    #     if self.metadata is not None and self.key_metadata is not None:
    #         return self.metadata[self.metadata[self.key_metadata].isin(id_list)]
    #     else:
    #         return None

    # def get_metadata_by_id_list_and_key(self, id_list, key):
    #     if self.metadata is not None and self.key_metadata is not None:
    #         return self.metadata[self.metadata[self.key_metadata].isin(id_list)][key]
    #     else:
    #         return None


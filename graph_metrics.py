# %% [markdown]
# # TCRpeg embeddings

# %%
import numpy as np
import pandas as pd
import os
import faiss

# %%
from tcrpeg_toolkit.embedding_clustering import EmbeddingClustering
from tcrpeg_toolkit.umap_generator import UMAPGenerator
from tcrpeg_toolkit.embedding_handler import EmbeddingHandler

# %% [markdown]
# ## A collection of samples

# %%
#write a loop to loop over all models and calculate the graph metrics

# List all models in the directory
files_embeddings = os.listdir('/Users/vanessamhanna/Nextcloud/TCRpeg/embeddings/structured/')
# Filter for carto 27 files
# files_embeddings = [i for i in files_embeddings if '-27' in i]

#iterate over each model, embedding and data file and apply PinferCalculation function
results = []

for embeddings in files_embeddings:
    embedding_file = '/Users/vanessamhanna/Nextcloud/TCRpeg/embeddings/structured/' + embeddings
    name = os.path.basename(embedding_file).split('_structured_embeddings.npy')[0]
    
    # output_dir = '/Users/vanessamhanna/Nextcloud/TCRpeg/'
    
    sample_f = EmbeddingClustering(embedding_file)
    sample_f_clusters = sample_f.run(k=1)
    
    sample_f_clusters.metadata['sample']=name
    
    # size_distribution = sample_f_clusters.metadata.groupby('cluster_hdbscan').size() # add to function that calculates distance
    # clustering_proportion = (size_distribution.get(-1) / size_distribution.sum())*100 # add to metadata + degree.. in function
    results.append([name, sample_f_clusters.metadata])

# %%
df_results = pd.concat([result[1] for result in results])
df_results.to_csv('/Users/vanessamhanna/Nextcloud/TCRpeg/analysis/graphs_all_no_downsample.csv', index=False)




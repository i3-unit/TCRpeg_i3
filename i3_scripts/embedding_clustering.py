import os, sys
import pandas as pd
import numpy as np
from scipy.spatial import distance
import glob
import seaborn as sns
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import faiss


sns.set_theme()

# wd = os.getcwd()
# print(wd)

wd = '/Users/vanessamhanna/Nextcloud/TCRpeg/results'
os.chdir(wd)

# Read the metadata
metadata = pd.read_csv('/Users/vanessamhanna/Nextcloud/TCRpeg/metadata.csv', index_col=None)
metadata['carto'] = metadata['Lib_Exp_ID'].str.split('-').str[1]
metadata['sample_id'] = metadata[['organ','carto','cell_pop']].apply(lambda row: '-'.join(row.values.astype(str)) + '-TRB', axis=1)


# List all files in the directory
files = glob.glob(os.path.join(wd, '**', '*'), recursive=True)
npy_files = [file for file in files if file.endswith('embeddings.npy')]

names = [os.path.basename(file).split('_embeddings.npy')[0] for file in npy_files]

# Load each .npy file
data_dict = {}
for file in npy_files:
    data = np.load(file)
    # flattened_data = data.flatten()
    data_dict[os.path.basename(file).split('_embeddings.npy')[0]] = data
    print(f"Loaded {file}")
  

# #calculate the euclidean distances
# euc_distances = np.array([[distance.euclidean(value1, value2) for value1 in data_dict.values()]
#                         for value2 in data_dict.values()])

# distances = pd.DataFrame(euc_distances, index=names, columns=names)


# Convert embeddings to float32
embeddings = data_dict['PALN-26-Teff-TRB']
# embeddings = np.array(list(data_dict.values()), dtype=np.float32)

# Set up FAISS clustering
d = embeddings.shape[1]  # dimensionality of vectors
k = 5  # number of clusters ELBOW or silhouette
niter = 10 # number of iterations

# Create and train the clustering model
kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True)
kmeans.train(embeddings)    

# Get cluster assignments
cluster_assignments = kmeans.index.search(embeddings, 1)

# Get centroids
centroids = kmeans.centroids



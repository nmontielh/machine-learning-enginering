#%matplotlib inline
#TODO: Import sklearn's cluster module
from sklearn import cluster

import pandas as pd
#import dbscan_lab_helper as helper
from modulo22 import dbscan_lab_helper as helper


dataset_1 = pd.read_csv('blobs.csv')[:80].values
 
helper.plot_dataset(dataset_1)

epsilon= 1

dbscan =  cluster.DBSCAN(eps=epsilon, min_samples=5)

#TODO: use DBSCAN's fit_predict to return clustering labels for dataset_1
clustering_labels_1 = dbscan.fit_predict(dataset_1)

# Plot clustering
helper.plot_clustered_dataset(dataset_1, clustering_labels_1)

# Plot clustering with neighborhoods
helper.plot_clustered_dataset(dataset_1, clustering_labels_1, neighborhood=True)


# Pruebas con el dataset 2
dataset_2 = pd.read_csv('varied.csv')[:300].values

# Plot
helper.plot_dataset(dataset_2, xlim=(-14, 5), ylim=(-12, 7))


# Prueba 3
# TODO: Experiment with different values for eps and min_samples to find a suitable clustering for the dataset
eps=1
min_samples=3

# Cluster with DBSCAN
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
clustering_labels_4 = dbscan.fit_predict(dataset_2)

# Plot
helper.plot_clustered_dataset(dataset_2, 
                              clustering_labels_4, 
                              xlim=(-14, 5), 
                              ylim=(-12, 7), 
                              neighborhood=True, 
                              epsilon=0.5)


eps_values = [0.3, 0.5, 1, 1.3, 1.5]
min_samples_values = [2, 5, 10, 20, 80]

helper.plot_dbscan_grid(dataset_2, eps_values, min_samples_values)



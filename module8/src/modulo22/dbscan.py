from sklearn import datasets, cluster

# load datasets

dataset_1 = datasets.load_iris().data

# Especifica los parametros para el clustering

dbscan =  cluster.DBSCAN(eps=0.5, min_samples=5)
#TODO: use DBSCAN's fit_predict to return clustering labels for dataset_1
clustering_labels_1 = dbscan.fit_predict(dataset_1)

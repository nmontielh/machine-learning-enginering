from sklearn import datasets, cluster

from scipy.cluster.hierarchy import dendrogram, ward, single

import matplotlib.pyplot as plt


# Cargamos los datasets
X = datasets.load_iris().data[:10]

# especificamos los parametros para el clustering, 'ward', linkage
# es default, tambien podemos usar complete o average

my_cluster = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')

# realizamos el clustering
linkage_matrix = ward(X)

labels = my_cluster.fit_predict(X)

# labels ahora contiene un arreglo representando con un cluster cada punto que forma parte de
# [1 0 0 0 1 2 0 1 0 0]

print('labels:', labels)

# pintamos el dendograma
dendrogram(linkage_matrix)

plt.show()
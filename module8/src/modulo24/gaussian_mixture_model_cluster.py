from sklearn import datasets, mixture

# Load datasets
X = datasets.load_iris().data[:100]

#  Especificamos los parametros para el clustering
gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(X)
clustering = gmm.predict(X)
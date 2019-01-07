import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


#Datos que vamos a usar
data = pd.read_csv('underfitting-data.csv');

# Partimos los datos de entrada
X = np.array(data[ ['x1','x2'] ]);
y = np.array(data['y']);

# Colocamos la semilla a 55
np.random.seed(55);


# Pendiente revisar como cargar ese modulo

#X2, y2 = randomize(X, y);

#estimator = LogisticRegression();
#estimator = GradientBoostingClassifier();
#estimator = SVC(kernel="rbf", gamma=1000);



#plot_learning_curves(X, y,X2, y2, estimator, 2);
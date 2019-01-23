
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import  f1_score

from sklearn.tree import DecisionTreeClassifier

#fit data
#Datos que vamos a usar
data = pd.read_csv('underfitting-data.csv');

# Partimos los datos de entrada
X = np.array(data[ ['x1','x2'] ]);
y = np.array(data['y']);


#Seleccionamos parametros
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}; 

# create scorer
scorer = make_scorer(f1_score)

classifier = DecisionTreeClassifier(random_state=42);

# creamos el grid_search, que es el CLF, creamos el objeto
grid_obj = GridSearchCV(classifier, parameters, scoring=scorer);

# Fit the data
grid_fit = grid_obj.fit(X,y);

#Obtenemos el mejor estimador
best_clf = grid_fit.best_estimator_
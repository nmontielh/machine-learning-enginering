import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import accuracy_scorer


#Obtenemos los datos
data_csv = pd.read_csv('data.csv', header = None )
data = np.asarray(data_csv)
x_values=data[:,0:2]
y_values=data[:,2]



#Definimos los hyperparameters

#Numero maximo de niveles del arbol
max_depth_user = 7

#numero minimo de muestras en cada hoja (nodo)
min_samples_leaf_user=10

#minimo numero de muestras requeridas para partir un nodo interno
min_samples_split_user=2

#model = DecisionTreeClassifier(max_depth=max_depth_user, min_samples_leaf=min_samples_leaf_user)
model = DecisionTreeClassifier()
print("model", model)


#y_predict = [ [0.2, 0.8], [0.5,0.4] ]

# buscamos el mejor arbol para entrenar los datos
model.fit(x_values, y_values)

#Predict using the model (lo agregamos para lo que memorizo)
prediction = model.predict(x_values)

print("Prediction", prediction)

accuracy = accuracy_score(y_values, prediction)

print("accuracy", accuracy)


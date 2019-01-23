import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


random.seed(42)

#Cargamos el dataset
full_data = pd.read_csv("titanic_data.csv")

display(full_data.head())

outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis=1)
display(features_raw.head())

#Quitamos los nombres
feautures_no_names = full_data.drop('Name', axis=1)
#One hot enconding
features = pd.get_dummies(feautures_no_names)
features = features.fillna(0.0)

display(features.head())

#Partimos los datos de entrenamoento
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)

#Numero maximo de niveles del arbol
max_depth_user = 6

#numero minimo de muestras en cada hoja (nodo)
min_samples_leaf_user=6

#minimo numero de muestras requeridas para partir un nodo interno
min_samples_split_user=10

model = DecisionTreeClassifier(max_depth=max_depth_user, min_samples_leaf=min_samples_leaf_user)
#model = DecisionTreeClassifier()
model.fit(X_train, y_train)


#Hacemos las predicciones
y_train_pred = model.predict(X_train)
y_test_pred=model.predict(X_test)

#Calculamos el accuracy

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test)

print("train acc", train_accuracy)
print("test acc", test_accuracy)



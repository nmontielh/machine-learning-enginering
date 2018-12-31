# archivo para proba los sets
import pandas as pd
import numpy as np

from pandas import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.io.tests.test_wavfile import datafile

# Lectura de datos con header
#data = pd.read_csv("svm-data.csv");
#X  = data[['x1','x2']];
#y = data['y'];

#Leeemos los datos a un array que no tiene encabezados
dataFile = pd.read_csv('tree-data.csv', header=None);
#print(dataFile.head())
#dataFile = pd.read_csv('tree-data.csv');

data = np.asarray(dataFile);

X = data[:,0:2];
y = data[:,2];
#Asignamos los conjuntos de datos
# el 25% sera asignado para pruebas
# nunca usar los datos de testing para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42);

# Instanciamos el modelo
model = DecisionTreeClassifier();

#Entrenamos con los datos de entrenamiento
result = model.fit(X_train, y_train);

# Hacemos prediciones con nuestros datos de prueba
y_pred = model.predict(X_test);

#Medimos la precision
acc = accuracy_score(y_test, y_pred);

print("resultado" + str(acc))
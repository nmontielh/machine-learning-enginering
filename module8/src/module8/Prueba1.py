import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
 
prueba = pd.read_csv("prueba.csv");
data = pd.read_csv("data-training.csv");

# Obtenemos los arreglos en dataframe
# en especifico obtenemos las columnas (x1,x2) y y    
xArray = data[['x1','x2']];
yArray = data['y'];

#print("Imprimiendo x1, y1")
#print(xArray)

#print("Imprimiendo y")
#print(yArray)

#obtenemos arreglos del dataframe (pandas DF) --> (Numpy Arrays)
X = np.array(xArray);
y = np.array(yArray);

# Iniciamos a usar los clasificadores de sklearn
# Logistic regression
from sklearn.linear_model import LogisticRegression

#neural networks
from sklearn.neural_network import MLPClassifier

#decision  trees
from sklearn.tree import DecisionTreeClassifier

#Support Vector Machines
from sklearn.svm import SVC

#Hacemos  las pruebas con logistoc regression (para entrenar)
classifier = LogisticRegression();
result = classifier.fit(X,y);

classifier = DecisionTreeClassifier();
result = classifier.fit(X,y);

classifier = SVC();
result = classifier.fit(X,y);

# Revisar por que este no converge
classifier = MLPClassifier();
result = classifier.fit(X,y);




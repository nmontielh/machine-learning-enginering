import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score 
# Programa para SVM
from sklearn.svm import SVC


# Obtenemos datos
# configurar los hiperparameters para obtener el 100% de accuracy
data = np.asarray(pd.read_csv('data.csv', header=None))

# # Assign the features to the variable X, and the labels to the variable y. 
x_values=data[:,0:2]
y_values=y = data[:,2]


# Especificamos hyperparameters (Son los que debemos tunear)
# C = C parameter
# Kernel : The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
# RBF = Radial Bias Functions
# poly = polynomia

#model = SVC(kernel='poly', degree=4, C=0.1)
model = SVC(kernel='rbf', gamma=27)


# itting the model means finding the best boundary that fits the training data
model.fit(x_values, y_values)

# Hacemos 2 predionones
#predict_test=[ [0.2, 0.8], [0.5, 0.4] ]

# hacemos la prediccion del 100%
predict_result = model.predict(x_values)

print("prediction:", predict_result)

# returned an array of predictions, one prediction for each input array
# Regresa un array de predicciones, cada 1 por un 
acc = accuracy_score(y_values, predict_result)


print("accuracy:", acc)

# Pendiente el plot


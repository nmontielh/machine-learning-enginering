import pandas as pd
import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold





size_data = 12;
size_testing_set = 3;

X = np.array([ [1,2], [3,4], [1,2], [3,4] ])
y = np.array([1,2,3,4])

#  Recomendacion colocar todos los datos random
# shuffle=True
kf = KFold(n_splits=2)
print("KFold", kf)

for train_index, test_index in kf.split(X):
    print("train:", train_index, "test:",test_index);
    X_train, X_test = X[train_index], X[test_index];
    y_train, y_test = y[train_index], y[test_index];

print("X train:", X_train, "X test", X_test)
print("y train:", y_train, "y test", y_test)


XX = range(size_data);
kf = KFold(shuffle=True, n_splits=size_testing_set);

print("Nueva prueba---")
# Otra prueba
for train_index, test_index in kf.split(XX):
    print("train:",train_index, "test", test_index);
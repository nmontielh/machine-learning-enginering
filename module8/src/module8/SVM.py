import pandas as pd
import numpy as np

from sklearn.svm import SVC

# leemos el archivo de los datos a entrenar
data = pd.read_csv("svm-data.csv");
# data ya es el dataframe, ahora colocaremos el dataframe en las columnas
print(data)

#Se revisan los datos  en el archivo
X = data[['x1','x2']];
y = data['y'];

classifier = SVC( kernel = 'poly', degree=2);
result =  classifier.fit(X,y);

print(result);

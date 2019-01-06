import pandas as pd
import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Cargaos los datos
dataFile = pd.read_csv('tree-data.csv', header=None);
data = np.asarray(dataFile);

X = data[:,0:2];
y = data[:,2];

#DTOS DE ENTRENAMiento
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42);

classifier = LinearRegression();
classifier.fit(X,y);

guesses = classifier.predict(X);

error = mean_squared_error(y,guesses);

print("Error ", error);

# Claculamos el R2 Score
y_true = [1,2,4]
y_pred = [1.3,2.5,3.7]

r2s = r2_score(y_true, y_pred);

print("R2_Scpore", r2s)

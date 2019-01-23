import  pandas as pd
    
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Assign the dataframe to this variable.
#Leemos los datos 
#boston_data = pd.read_csv("housing.data");

boston_data = load_boston()
x_data = boston_data['data']
y_data = boston_data['target']

model =  LinearRegression()

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
# buscar la mejor linea que se ajusta a los datos de entrenamiento
model.fit(x_data, y_data)

# encontramos la mejor linea con el predict
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
# tenemos un modelo que hace predicciones de multiples features
prediction =model.predict(sample_house)

print("prediction", prediction)
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier



import random as ra
from dask.array.tests.test_array_core import test_size

# Cargamos puntos de un CSV
def load_pts(csv_name):
    data = np.asarray(pd.read_csv(csv_name, header=None))
    X = data[:,0:2]
    y = data[:,2]

    plt.scatter(X[np.argwhere(y==0).flatten(),0], X[np.argwhere(y==0).flatten(),1],s = 50, color = 'blue', edgecolor = 'k')
    plt.scatter(X[np.argwhere(y==1).flatten(),0], X[np.argwhere(y==1).flatten(),1],s = 50, color = 'red', edgecolor = 'k')
    
    plt.xlim(-2.05,2.05)
    plt.ylim(-2.05,2.05)
    plt.grid(False)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off')

    return X,y

# Plot the model
def plot_model(X, y, clf):
    plt.scatter(X[np.argwhere(y==0).flatten(),0],X[np.argwhere(y==0).flatten(),1],s = 50, color = 'blue', edgecolor = 'k')
    plt.scatter(X[np.argwhere(y==1).flatten(),0],X[np.argwhere(y==1).flatten(),1],s = 50, color = 'red', edgecolor = 'k')

    plt.xlim(-2.05,2.05)
    plt.ylim(-2.05,2.05)
    plt.grid(False)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off')

    r = np.linspace(-2.1,2.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    z = clf.predict(h)

    s = s.reshape((np.size(r),np.size(r)))
    t = t.reshape((np.size(r),np.size(r)))
    z = z.reshape((np.size(r),np.size(r)))

    plt.contourf(s,t,z,colors = ['blue','red'],alpha = 0.2,levels = range(-1,2))
    if len(np.unique(z)) > 1:
        plt.contour(s,t,z,colors = 'k', linewidths = 2)
    plt.show()




# Inicia  el programa inicial
X, y = load_pts('grid-data.csv')
plt.show()

# Ajustamos la semilla aleatoria
ra.seed(42);

#Split data en training y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42);

# Adecuamos, ajustamos el modelo
clf = DecisionTreeClassifier(random_state=42);
###################
# Creamos un listado de parametros
#parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]};
parameters = {'max_features':[1,2]};

# hacemos un fbeta_score
scorer = make_scorer(f1_score);

# Hacemos un grid_searh en el clasificador, usando el scorer como el metodo de score
grid_obj = GridSearchCV(clf, parameters, scoring=scorer, cv=5);

# ajustamos el objeto grid_searcg¿h a los datos de entrenamiento y buscamos los parametros optimos
grid_fit = grid_obj.fit(X_train,y_train);

# Obtenemos el estimador
best_clf = grid_fit.best_estimator_;

################## Hacemos predicciones con el nuevo modelo
# fit the model
best_clf.fit(X_train, y_train);

#Hacemos la predicccion
train_predictions = best_clf.predict(X_train);
test_predictions = best_clf.predict(X_test);

# buscamos el testing F1_score
plot_model(X, y, best_clf);

f1_score_training = f1_score(train_predictions, y_train);
f1_score_test = f1_score(test_predictions, y_test);

print("The training F1 Score is:", f1_score_training)
print("The testing F1 Score is:", f1_score_test)

# Exploramos que parametros pueden ser usandoas en el nuevo modelo
best_clf
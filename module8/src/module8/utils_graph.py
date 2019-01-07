import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# Hacemos random los datos ŕevios a dibujarlos
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
    return X2, Y2;



def plot_learning_curves(X, y,X2, y2, estimator, num_trainings):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X2, y2, train_sizes=np.linspace(.1, 1.0, num_trainings) , 
                                                            cv = None, n_jobs=1)
    
    train_scores_mean = np.mean(train_scores, axis =1,);
    train_scores_std = np.std(train_scores, axis=1);
    test_scores_mean = np.mean(test_scores, axis = 1);
    test_scores_std = np.std(test_scores, axis=1);
    
    #Ajustamos la grafica
    plt.grid();
    plt.title("Learning  Curves");
    plt.xlabel("Training examples");
    plt.ylabel("Score");
    
    plt.plot(train_scores_mean, 'o-', color='g', label="Training Score");
    plt.plot(test_scores_mean, 'o-', color='y', label="Cross Validation Score");
    
    plt.legend(loc="best");
    plt.show();
    

#Iniciamos los datos    
    
#data = pd.read_csv('underfitting-data.csv');

# Partimos los datos de entrada
#X = np.array(data[ ['x1','x2'] ]);
#y = np.array(data['y']);

#X2, y2 = randomize(X, y);

#estimator = LogisticRegression();
#estimator = GradientBoostingClassifier();
#estimator = SVC(kernel="rbf", gamma=1000);



#plot_learning_curves(X, y,X2, y2, estimator, 2);


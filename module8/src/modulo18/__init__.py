# Se crea un adaboost
# Pendiente el baggong, reviar en internet
# que son los regressor

from sklearn.ensemble import  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


X_train = None
y_train = None
x_test=None


# hyperparameters
# base_stimator = The model utilized for the weak learners
# n_stimators = The maximum number of weak learners used.

estimator = DecisionTreeClassifier(max_depth=2)

model = AdaBoostClassifier(base_estimator=estimator, n_estimators=4)
model.fit(X_train, y_train)
model.predict(x_test)

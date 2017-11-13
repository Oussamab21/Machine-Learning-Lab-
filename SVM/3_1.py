"""
Creates a random dataset
A cross validation procedure to select the best hyperparameters
Display the decision surface found and
Display the support vectors
"""

from sklearn.datasets import make_classification
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
import matplotlib.pyplot as plt


# Parameters selection for Linear Classifier
def svc_param_selection_linear(X, y, nfolds):
    # Generates Cs
    Cs = range(1, 5)
    # Generates a grid of Cs
    param_grid = {'C': Cs}
    # Uses the GridSearch option in SciKit to search for the best option
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search

# Parameters selection for RBF Kernel
def svc_param_selection_rbf(X, y, nfolds):
    # Generates Cs
    Cs = range(1, 5)
    # Generates Gammas
    gammas = [0.001, 0.01, 0.1, 1]
    # Generates the grids Cs x Gammas
    param_grid = {'C': Cs, 'gamma': gammas}
    # Uses the GridSearch
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search

# Parameters selection for Polynomial Kernel
def svc_param_selection_poly(X, y, nfolds):
    # Generates Cs
    Cs = range(1, 5)
    # Generates Gammas
    gammas = [0.001, 0.01, 0.1, 1]
    # Generates Polynomial Degrees
    degree = [1, 2, 3, 4, 5, 6, 7]
    # Generates the grids
    param_grid = {'C': Cs, 'gamma': gammas, 'degree': degree}
    grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


# Generates random data
X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2,
                           random_state=2, n_clusters_per_class=1)
nfolds = 3

# Best SVC Linear
best = svc_param_selection_linear(X, y, nfolds)
# Best SVC RBF
best2 = svc_param_selection_rbf(X, y, nfolds)
# Best SVC Polynomial
best3 = svc_param_selection_poly(X, y, nfolds)

h = .02  # grid step

print()

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

mysvc = best
mysvc.fit(X, y)

Z2d = mysvc.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d = Z2d.reshape(xx.shape)
f1 = plt.figure(1)

plt.title('Linear kernel')
plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
# We plot also the training points
estimator = mysvc.best_estimator_
support_vectors = estimator.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
f1.show()

mysvc2 = best2
mysvc2.fit(X, y)
Z2d2 = mysvc2.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d2 = Z2d2.reshape(xx.shape)
f2 = plt.figure(2)
plt.title('Rbf Kernel')
plt.pcolormesh(xx, yy, Z2d2, cmap=plt.cm.Paired)
# We plot also the training points
estimator = mysvc2.best_estimator_
support_vectors = estimator.support_vectors_
plt.scatter(support_vectors[0:, 0], support_vectors[0:, 1], s=85, edgecolor='g', alpha=0.59, c='g')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
f2.show()


mysvc3 = best3
mysvc3.fit(X, y)
Z2d3 = mysvc3.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d3 = Z2d3.reshape(xx.shape)
f3 = plt.figure(3)
plt.pcolormesh(xx, yy, Z2d3, cmap=plt.cm.Paired)
# We plot also the training points
plt.title('Polynomial kernel')
estimator = mysvc3.best_estimator_
support_vectors = estimator.support_vectors_
plt.scatter(support_vectors[0:, 0], support_vectors[0:, 1], s=85, edgecolor='g', alpha=0.59, c='g')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
f3.show()
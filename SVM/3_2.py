from sklearn.datasets import make_classification
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
import matplotlib.pyplot as plt


def svc_param_selection_linear(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = range(1, 7)
    # gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs}  # , 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


def svc_param_selection_rbf(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = range(1, 7)
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


def svc_param_selection_poly(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = range(1, 7)
    gammas = [0.001, 0.01, 0.1, 1]
    degree = [1, 2, 3, 4, 5, 6, 7]
    param_grid = {'C': Cs, 'gamma': gammas, 'degree': degree}
    grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


from sklearn.datasets import make_moons
X, y = make_moons(noise=0.1, random_state=1, n_samples=40)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
nfolds = 3
print()

h = .02  # grid step
print()

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

best = svc_param_selection_linear(X, y, nfolds)
best.fit(X, y)
Z2d = best.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d = Z2d.reshape(xx.shape)
f1 = plt.figure(1)
plt.title('Linear Kernel')
plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
# We plot also the training points
estimator = best.best_estimator_
support_vectors = estimator.support_vectors_
# print(support_vectors[:,0])
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

f1.show()
# print('the parameters of linear classefier are')
# print(best)
print()

best2 = svc_param_selection_rbf(X, y, nfolds)
best2.fit(X, y)
Z2d2 = best2.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d2 = Z2d2.reshape(xx.shape)
f2 = plt.figure(2)
plt.title('RBF Kernel')
plt.pcolormesh(xx, yy, Z2d2, cmap=plt.cm.Paired)
# We plot also the training points
estimator = best2.best_estimator_
support_vectors = estimator.support_vectors_
# print(support_vectors[:,0])
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

# plt.plot()
f2.show()
# print('the parameters of rbf classefier are')
# print(best2)
print()
best3 = svc_param_selection_poly(X, y, nfolds)
best3.fit(X, y)
# print('the parameters of poly classefier are')
# print(best3)
Z2d3 = best3.predict(np.c_[xx.ravel(), yy.ravel()])  # we predict all the grid
Z2d3 = Z2d3.reshape(xx.shape)
# f1=plt.figure()
f3 = plt.figure(3)
plt.pcolormesh(xx, yy, Z2d3, cmap=plt.cm.Paired)
estimator = best3.best_estimator_
support_vectors = estimator.support_vectors_
# print(support_vectors[:,0])
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')

# We plot also the training points
plt.title('Poly Kernel')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
# plt.plot()
f3.show()
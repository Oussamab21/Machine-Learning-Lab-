import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
import matplotlib.pyplot as plt
from sklearn import metrics

# Dataset of Ozone emissions
input_file = "ozone.csv"
df = pd.read_csv(input_file, sep=' ')

# Get some information about dataframe in pandas
print("Dataframe shape: {}", df.shape)
# print("Dataframe index: {}", df.index)

# df.index
# df.columns
# df.head()

df.info()

# Because the feature STATION is a categorical one, we need to express in the form
# of a bit map for each station
df['STATION']
Sta = pd.get_dummies(df)
# print(Sta)
m = df.as_matrix()
m2 = df.as_matrix()
m3 = Sta.as_matrix()

from sklearn.preprocessing import StandardScaler

# normalize dataset for easier parameter selection
# Non normalized data
X_non_normalized = Sta[['JOUR', 'MOCAGE', 'TEMPE',
                        'RMH2O', 'NO2', 'NO', 'VentMOD',
                        'VentANG', 'STATION_Aix', 'STATION_Als',
                        'STATION_Cad', 'STATION_Pla', 'STATION_Ram']]

# Target variable raw
Y_raw = Sta[['O3obs']]

# Target variable normalized
Y_normalized = StandardScaler().fit_transform(Y_raw)

# Normalized data
X_normalized = StandardScaler().fit_transform(X_non_normalized)  # where X is your data matrix

# Target Label
Y_label = Y_raw.as_matrix()[:, 0] > 150
# X_normalized = X_normalized.as_matrix()

# y = m3[:, 0] > 150  ## need to be confirmed   ### target data need to be changed


def svc_param_selection_linear(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = range(1, 5)
    # gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs}  # , 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds, scoring='roc_auc')
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


def svc_param_selection_rbf(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = range(1, 5)
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, scoring='roc_auc')
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


def svc_param_selection_poly(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = range(1, 5)
    gammas = [0.001, 0.01, 0.1, 1]
    degree = [1, 2, 3, 4, 5, 6, 7]
    param_grid = {'C': Cs, 'gamma': gammas, 'degree': degree}
    grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds, scoring='roc_auc')
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
          % (grid_search.best_params_, grid_search.best_score_))
    return grid_search

h = .02  # grid step

print()

#x_min = X[:, 0].min() - 1
#x_max = X[:, 0].max() + 1
#y_min = X[:, 1].min() - 1
#y_max = X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# For testing Performance of the Model:
# Divide Training and Testing

# For Normalized Data and Y as label
nfolds = 3
best = svc_param_selection_linear(X_normalized, Y_label, nfolds)
clf = best
clf.fit(X_normalized, Y_label)
print()
best2 = svc_param_selection_rbf(X_normalized, Y_label, nfolds)
clf2 = best2
clf2.fit(X_normalized, Y_label)

print()
best3 = svc_param_selection_poly(X_normalized, Y_label, nfolds)
clf3 = best3
clf3.fit(X_normalized, Y_label)

# Missing: testing with different values of nfolds
scores = cross_val_score(clf, X_normalized, Y_label, scoring='roc_auc')
scores2 = cross_val_score(clf2, X_normalized, Y_label, scoring='roc_auc')
scores3 = cross_val_score(clf3, X_normalized, Y_label, scoring='roc_auc')
scores = sum(scores) / nfolds
scores2 = sum(scores2) / nfolds
scores3 = sum(scores3) / nfolds

print('For Normalize data:')
print('Scores for model Linear: {}'.format(scores))
print('Scores for model RBF: {}'.format(scores2))
print('Scores for model Poly: {}'.format(scores3))
print('Selecting the best model')

s = np.array([scores, scores2, scores3])

if max(s) == scores:
    print("Best classifier is Linear")
elif max(s) == scores2:
    print("Best classifier is RBF")
else:
    print("Best classifier is Poly")

# For Non-Normalized Data and Y as label
best = svc_param_selection_linear(X_non_normalized, Y_label, nfolds)
clf = best
clf.fit(X_non_normalized, Y_label)
print()
best2 = svc_param_selection_rbf(X_non_normalized, Y_label, nfolds)
clf2 = best2
clf2.fit(X_non_normalized, Y_label)
print()

# Parameter selection for Polynomial takes too much time
best3 = svc_param_selection_poly(X_non_normalized, Y_label, nfolds)
clf3 = best3
clf3.fit(X_non_normalized, Y_label)
# Missing: testing with different values of nfolds
scores = cross_val_score(clf, X_non_normalized, Y_label, scoring='roc_auc')
scores2 = cross_val_score(clf2, X_non_normalized, Y_label, scoring='roc_auc')
scores3 = cross_val_score(clf3, X_non_normalized, Y_label, scoring='roc_auc')
scores = sum(scores) / nfolds
scores2 = sum(scores2) / nfolds
scores3 = sum(scores3) / nfolds

s = np.array([scores, scores2, scores3])
if max(s) == scores:
    print("Best classifier for Non Normalized is Linear")
elif max(s) == scores2:
    print("Best classifier for Non Normalized is RBF")
else:
    print("Best classifier is Non Normalized is Poly")

print('For Non Normalized Data:')
print('Scores for model Linear: {}'.format(scores))
print('Scores for model RBF: {}'.format(scores2))
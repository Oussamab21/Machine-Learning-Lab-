import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
import matplotlib.pyplot as plt

from sklearn.svm import SVR






input_file = "ozone.csv"
df = pd.read_csv(input_file, sep=' ')

df.shape
df.index
df.columns
#df.Series
df.head()

df.info()

df['STATION']
Sta = pd.get_dummies(df)
#print(Sta)
m= df.as_matrix()
m2=df.as_matrix()
m3=Sta.as_matrix()

from sklearn.preprocessing import StandardScaler
...
# normalize dataset for easier parameter selection
Sta2 = StandardScaler().fit_transform(Sta) #where X is your data matrix

#Sta2.as_matrix()
#Sta.info()
#Sta.head()

X = np.delete(Sta2,(1), axis=1)
#X=np.delete(m3,(1),axis=1)

#y=np.array([])
y= m3[:,1] # > 150 ## need to be confirmed   ### target data need to be changed 

#########################################################""" For non normalized data###################################





def svc_param_selection_linear(X, y, nfolds):
    #Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs=range(1, 5)
    #gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs}#, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVR(kernel='linear'),param_grid,cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
      % (grid_search.best_params_, grid_search.best_score_))
    return grid_search



def svc_param_selection_rbf(X, y, nfolds):
    #Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs=range(1, 5)
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVR(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
      % (grid_search.best_params_, grid_search.best_score_))
    return grid_search



def svc_param_selection_poly(X, y, nfolds):
    #Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs=range(1,5)
    gammas = [0.001, 0.01, 0.1, 1]
    degree=[1,2,3,4]
    param_grid = {'C': Cs, 'gamma' : gammas,'degree':degree}
    grid_search = GridSearchCV(svm.SVR(kernel='poly'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("The best parameters are %s with a score of %0.2f"
      % (grid_search.best_params_, grid_search.best_score_))
    return grid_search


nfolds=3

h = .02 # grid step
    
print()

x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) 









best=svc_param_selection_linear(X,y,nfolds)
clf=best
clf.fit(X, y) 
print()



best2=svc_param_selection_rbf(X,y,nfolds)
clf2=best2
clf.fit(X, y) 
print()



best3=svc_param_selection_poly(X,y,nfolds)

clf3=best3
clf.fit(X, y) 


scores= cross_val_score(clf, X, y)

scores2= cross_val_score(clf2, X, y)
scores3= cross_val_score(clf3, X, y)
print(scores)
print(scores2)
print(scores3)

scores=sum(scores)/nfolds
scores2=sum(scores2)/nfolds
scores3=sum(scores3)/nfolds

s=np.array([scores,scores2,scores3])
#print min(s) 
if max(s)==s[0]:
  print('the best is linear')

else: 
 if max(s)==s[1]:  
  print('the best is rbf') 
 else: 
  print('the best is poly')  




import numpy as np
X = np.array([[-1, -1],[-2, -1],[1, 1],[2, 1],[-1.0,0.5],[0,1],[0,0.5],[-0.5,3],[-1,2],[-0.5,-1.5]])
#we create 4 examples
y = np.array([-1, -1, 1, 1,-1,1,1,-1,-1,1])
from sklearn.svm import SVC #import de la classe SVC pour SVM
classif=SVC() #we create a SVM with default parameters
classif.fit(X,y) #we learn the model according to given data
res=classif.predict([[-0.8, -1]]) #prediction on a new sample

print(res);

res2=classif.predict([[-0.5,2]])
#print("second");
#print(res2)

##plotting

import matplotlib.pyplot as plt #the library for plotting
#we create a mesh to plot in
h = .02 # grid step
print()

x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
#print ("the values of xx and yy ")
 #the grid is created, the intersections are in xx and yy
#print(xx)
#print(yy)
#print()


#the grid is created, the intersections are in xx and yy
mysvc= SVC(kernel='linear', C = 4.0)
mysvc.fit(X,y)
Z2d = mysvc.predict(np.c_[xx.ravel(),yy.ravel()]) # we predict all the grid
Z2d=Z2d.reshape(xx.shape)
#f1=plt.figure()
f = plt.figure(1)
plt.title('Linear kernel')
plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
# We plot also the training points
support_vectors = mysvc.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#plt.plot()
f.show()


mysvc2= SVC(kernel='poly', C = 4.0,degree=3)
mysvc2.fit(X,y)
Z2d2= mysvc2.predict(np.c_[xx.ravel(),yy.ravel()]) # we predict all the grid
Z2d2=Z2d2.reshape(xx.shape)
#f1=plt.figure()
f2= plt.figure(2)
plt.pcolormesh(xx,yy,Z2d2, cmap=plt.cm.Paired)
# We plot also the training points
plt.title('Poly kernel')
support_vectors = mysvc2.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#plt.plot()
f2.show()



mysvc3= SVC(kernel='rbf', C = 4.0)#,degree=10)
mysvc3.fit(X,y)
Z2d3= mysvc3.predict(np.c_[xx.ravel(),yy.ravel()]) # we predict all the grid
Z2d3=Z2d3.reshape(xx.shape)
#f1=plt.figure()
f3= plt.figure(3)
plt.pcolormesh(xx,yy,Z2d3, cmap=plt.cm.Paired)
# We plot also the training points
plt.title('RBF kernel')
support_vectors = mysvc3.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#plt.plot()
f3.show()




mysvc4= SVC(kernel='sigmoid', C = 4.0,gamma=1)#,degree=10)
mysvc4.fit(X,y)
Z2d4= mysvc4.predict(np.c_[xx.ravel(),yy.ravel()]) # we predict all the grid
Z2d4=Z2d4.reshape(xx.shape)
#f1=plt.figure()
f4= plt.figure(4)
plt.pcolormesh(xx,yy,Z2d4, cmap=plt.cm.Paired)
# We plot also the training points
plt.title('Sigmoid kernel')
support_vectors = mysvc4.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, edgecolor='g', alpha=0.59, c='g')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#plt.plot()
f4.show()
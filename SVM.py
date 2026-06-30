import numpy as np 
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#SVM which checks if the instance belongs to Iris Virginica or not
iris = load_iris()
X=iris["data"][:,(2,3)]
y=(iris["target"]==2).astype(float)

svm_clf=Pipeline([("scaler", StandardScaler()),("linearsvc", LinearSVC(C=1,loss="hinge")),])
svm_clf.fit(X,y)
print(svm_clf.predict([[100,100]]))
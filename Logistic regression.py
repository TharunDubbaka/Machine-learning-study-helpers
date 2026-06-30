from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
print(iris)
print(list(iris.keys()))
print(iris["data"])
X = iris["data"][:, 3:] # petal width
print(X)
print(iris["target"])
y = (iris["target"] == 2).astype(int) # 1 if Iris-Virginica, else 0
print(y)
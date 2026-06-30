from sklearn import datasets
import numpy as np

from sklearn.linear_model import LogisticRegression 

import matplotlib.pyplot as plt
iris=datasets.load_iris()
print(iris)
X=iris["data"][:,3:] #width
y=(iris["target"]==2).astype(int) #1 if virginica else 0
logreg=LogisticRegression()
logreg.fit(X,y)
X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=logreg.predict_proba(X_new)
"""plt.plot(X_new,y_proba[:,1],"g-",label="Iris-Virginica")
plt.plot(X_new,y_proba[:,0],"b--",label="Not Iris")
plt.show()"""

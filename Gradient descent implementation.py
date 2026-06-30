import numpy as np
import matplotlib.pyplot as plt
def gradient_descent(x,y):
    m_curr=b_curr=0
    n=len(x)
    lr=0.08
    itr=1000
    for i in range(itr):
       y_predicted = m_curr*x+b_curr
       cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
       md=-(2/n)*sum(x*(y-y_predicted))
       bd = -(2/n)*sum(y-y_predicted)
       m_curr=m_curr-lr*md 
       b_curr=b_curr-lr*bd 
    return m_curr,b_curr
    

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

a,b=gradient_descent(x,y)

plt.scatter(a,b)
plt.show()

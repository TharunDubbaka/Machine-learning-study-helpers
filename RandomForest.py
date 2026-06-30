from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
moons = make_moons()
scaler = StandardScaler()

X=moons[0]
y=moons[0]
scaledx=scaler.fit_transform(X)
print(scaledx)
scaledy=scaler.fit_transform(y)
plt.plot(scaledx,scaledy)
plt.show()
from sklearn.decomposition import PCA
import numpy as np
from statistics import mean


x = np.array([[0.9, 1],
              [2.4, 2.6],
              [1.2, 1.7],
              [0.5, 0.7],
              [0.3, 0.7],
              [1.8, 1.4],
              [0.5, 0.6],
              [0.3, 0.6],
              [2.5, 2.6],
              [1.3, 1.1]])


y = np.array([x.T[0] - mean(x.T[0]),
              x.T[1] - mean(x.T[1])])
c = np.cov(y)

l, v = np.linalg.eig(c)

print("Los vectores propios son: ", v[0], "y", v[1])
print("Los valores propios son: ", l)

print("Primer componente: ", np.dot(y.T, v.T[0]))
print("Segundo componente: ", -np.dot(y.T, v.T[1]))

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print(x_pca)

print("Varianza explicada con la primera componentes:", pca.explained_variance_ratio_[0])
print("Varianza explicada con la segunda componentes:", pca.explained_variance_ratio_[1])
import numpy as np

def stepGradient(par, x, y, learningRate):
    b_0_gradient = 0
    b_1_gradient = 0
    N = float(len(x))

    for i in range(0, len(x)):
        b_0_gradient += (2 / N) * (y[i] - (par[0] + par[1] * x[i]))
        b_1_gradient += (2 / N) * x[i] * (y[i] - (par[0] + par[1] * x[i]))

    new_b_0 = par[0] + (learningRate * b_0_gradient)
    new_b_1 = par[1] + (learningRate * b_1_gradient)

    return [new_b_0, new_b_1]


def fitGradient(par, x, y, learningRate, maxDifference=1e-6, maxIter=30):
    prev_step = par[:]
    num_iter = 0;

    num_iter += 1
    results = stepGradient(prev_step, trX, trY, learningRate)
    difference = abs(prev_step[0] - results[0]) + abs(prev_step[1] - results[1])

    while ((difference > maxDifference) & (num_iter < maxIter)):
        num_iter += 1
        prev_step = results
        results = stepGradient(prev_step, trX, trY, learningRate)
        difference = abs(prev_step[0] - results[0]) + abs(prev_step[1] - results[1])

    return results


trX = np.linspace(-2, 2, 101)
trY = 3 + 2 * trX + np.random.randn(*trX.shape) * 0.33

print(fitGradient([1, 1], trX, trY, 0.05))

import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D


def computeLinealRegressionError(b0, b1, x, y):
    totalError = 0
    for i in range(0, len(x)):
        totalError += (y[i] - (b0 + b1 * x[i])) ** 2
    return totalError / float(len(x))


b_0 = np.arange(0, 5, 0.05)
b_1 = np.arange(0, 5, 0.05)
X, Y = np.meshgrid(b_0, b_1)

zs = np.array([computeLinealRegressionError(x, y, trX, trY) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

#figure().add_subplot(111, projection='3d').plot_surface(X, Y, Z)


########################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

# Conjunto de datos
x = [[80], [79], [83], [84], [78], [60], [82], [85], [79], [84], [80], [62]]
y = [[300], [302], [315], [330], [300], [250], [300], [340], [315], [330], [310], [240]]

# Creación del modelo
model = LinearRegression()
model.fit(x, y)

# Obtención de estimaciones
print('Con 70 horas la producción sería:', model.predict([[70]]))
print

# Predicción del modelo
y_pred = model.predict(x);

# Obtención de los parametros de ajuste
print('w_0', model.intercept_[0])
print('w_1', model.coef_[0][0])

print('R^2', model.score(x, y))
print('Error cuadrático medio', mean_squared_error(y_pred, y))
print('Error absoluto medio', mean_absolute_error(y_pred, y))
print('Mediana del error absoluto', median_absolute_error(y_pred, y))

x_p = [[50], [350]]
y_p = model.predict(x_p)

plot(x, y, 'r.', label = 'Datos')
plot(x_p, y_p, label = 'Modelo')

title(u'Producción por hora')
xlabel('Horas')
ylabel(u'Producción')
axis([50, 90, 200, 350])

legend(loc = 2)
#########################################################################################################
model_ni = LinearRegression(fit_intercept = False)
model_ni.fit(x, y)

# Obtención de estimaciones
print('Con 70 horas la producción sería:', model_ni.predict([[70]])[0])
print

# Obtención de los parametros de ajuste
print('w_0', model_ni.intercept_)
print('w_1', model_ni.coef_[0][0])
print('R^2', model_ni.score(x, y))


##########################################################################################################

#regresion lineal multiple
from sklearn.preprocessing import PolynomialFeatures

poly_2  = PolynomialFeatures(degree = 2, include_bias = False)
x_2     = poly_2.fit_transform(x)

model_2 = LinearRegression(fit_intercept = False)
model_2.fit(x_2, y)

# Obtención de los parametros de ajuste
print('w_1', model_2.coef_[0][0])
print('w_2', model_2.coef_[0][1])
print('R^2', model_2.score(x_2, y))

#######################################################################################################
#validacion fuera de ajuste

from sklearn.datasets import load_boston

# Importación de los datos de vienda de Boston
boston = load_boston()

# Creación de un modelo
model_boston = LinearRegression()
model_boston.fit(boston.data, boston.target)

print("R^2:", model_boston.score(boston.data, boston.target))
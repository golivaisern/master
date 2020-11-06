from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from pprint import pprint

from sklearn.datasets import load_boston

# Importación de los datos de vienda de Boston
boston = load_boston()
print(type(boston))
pprint(boston)


# Creación de un modelo
model_boston = LinearRegression()
model_boston.fit(boston.data, boston.target)

print("R^2:", model_boston.score(boston.data, boston.target))
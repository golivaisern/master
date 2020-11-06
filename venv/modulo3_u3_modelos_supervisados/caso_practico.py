import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from pprint import pprint

wine = pd.read_csv('winequality-white.csv', sep = ',')


pprint(wine.info())
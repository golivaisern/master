import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(iris)
iris_names  = datasets.load_iris().target_names
print(iris_names)
iris_target = datasets.load_iris().target
iris_values = datasets.load_iris().data
print(iris_values)

plt.scatter(iris_values[iris_target == 0, 0], iris_values[iris_target == 0, 1], c='r')
plt.scatter(iris_values[iris_target == 1, 0], iris_values[iris_target == 1, 1], c='g')
plt.scatter(iris_values[iris_target == 2, 0], iris_values[iris_target == 2, 1], c='b')
plt.legend(iris_names)
print('finish')

# Example Python program to draw a scatter plot

# for two columns of a pandas DataFrame

import pandas as pd

import matplotlib.pyplot as plot

# List of tuples

data = [(2, 4),

        (23, 28),

        (7, 2),

        (9, 10)]

# Load data into pandas DataFrame

dataFrame = pd.DataFrame(data=data, columns=['A', 'B']);

# Draw a scatter plot

dataFrame.plot.scatter(x='A', y='B', title="Scatter plot between two variables X and Y");

plot.show(block=True);
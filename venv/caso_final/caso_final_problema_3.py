
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#load data
def load_data():
    data = pd.read_csv('crime_data.csv', sep=',')
    # print('Data count', data.count)
    return data


data = load_data()
print(data.head())

# standarize

# Separating out the features
target = 'State'
print(data[target])

features = list(data.columns)
targets= data
features.remove(target)
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,[target]].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

print(x[:5])



#pca analysis
# pca = PCA(n_components=4)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
x_pca = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


print(x_pca[:5])

finalDf = pd.concat([x_pca, data[target]], axis = 1)
print(finalDf.head(5))

finalDf.plot.scatter(x='principal component 1', y='principal component 2', title="Scatter plot between two variables X and Y");

plt.show(block=True);




print("Varianza explicada con la primera componentes:", pca.explained_variance_ratio_[0])
print("Varianza explicada con la segunda componentes:", pca.explained_variance_ratio_[1])

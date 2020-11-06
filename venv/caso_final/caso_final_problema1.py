import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from pprint import pprint

def data_info():
    with open('auto.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count==1:
                print(row)
            else:
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
        print(f'Processed {line_count} lines.')

def load_data():
    data = pd.read_csv('auto.csv', sep=',')
    print('Data count', data.count)
    return data

data = pd.read_csv('auto.csv', sep=',')

def split_x_y(data):
    target = 'mpg'
    features = list(data.columns)
    features.remove('mpg')
    x = data[features]
    print(x.count)
    y = data[target]
    return x,y

def test_train_split(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    return x_train, x_test, y_train, y_test

def treat_categoric_variables(x):
    x_categorize = pd.concat([x['cylinders'],
                      x['displacement'],
                      x['horsepower'],
                      x['weight'],
                      pd.get_dummies(x['model_year']),
                      pd.get_dummies(x['origin'])], axis=1)
    print('Count x variables categorizadas',x_categorize)
    return x_categorize


def data_model(x_train, x_test, y_train, y_test):
    model = LinearRegression(normalize = True)
    model.fit(x_train,y_train)
    return model

def evaluate_model(model, x_train, x_test, y_train, y_test):
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)
    print(model)
    print('R2 en entrenamiento es : ', model.score(x_train, y_train))
    print('R2 en test es : ', model.score(x_test, y_test))
    print('coeficiente', model.coef_[1,5,6])
    print('Error cuadrático medio', mean_squared_error(predict_test, y_test))
    print('Error absoluto medio', mean_absolute_error(predict_test, y_test))
    print('Mediana del error absoluto', median_absolute_error(predict_test, y_test))

def main():
    # data_info()
    data = load_data()
    x,y = split_x_y(data)
    x = treat_categoric_variables(x)
    x_train, x_test, y_train, y_test = test_train_split(x, y)
    model = data_model(x_train, x_test, y_train, y_test)
    evaluate_model(model, x_train, x_test, y_train, y_test)

main()

## aplicar cross validation












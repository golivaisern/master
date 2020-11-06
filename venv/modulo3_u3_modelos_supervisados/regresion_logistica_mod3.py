from sklearn.datasets import make_classification
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Creción de un conjunto de entrenamiento
X, y = make_classification(n_samples    = 2500,
                           n_features   = 3,
                           n_redundant  = 0,
                           random_state = 1)

# Creación de un conjunto de enrenamietno y test
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# Ajuste del modelo logístico
classifier   = LogisticRegression().fit(x_train, y_train)
y_train_pred = classifier.predict(x_train)
y_test_pred  = classifier.predict(x_test)

# Obtención de matriz de confusión
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

print('La matriz de confusión para entrenamiento es')
print(confusion_matrix_train)
print('La matriz de confusión para test es')
print(confusion_matrix_test)


print('La matriz de confusión para entrenamiento normalizada es')
print(confusion_matrix_train / double(sum(confusion_matrix_train)))
print('La matriz de confusión para test normalizada es')
print(confusion_matrix_test / double(sum(confusion_matrix_test)))

from sklearn.metrics import accuracy_score, precision_score, recall_score

print('Resultados en el conjunto de entrenamiento')
print(' Precisión:', accuracy_score(y_train, y_train_pred))
print(' Exactitud:', precision_score(y_train, y_train_pred))
print(' Exhaustividad:', recall_score(y_train, y_train_pred))
print('')
print(' Resultados en el conjunto de test')
print(' Precisión:', accuracy_score(y_test, y_test_pred))
print(' Exactitud:', precision_score(y_test, y_test_pred))
print(' Exhaustividad:', recall_score(y_test, y_test_pred))

from sklearn.metrics import roc_curve, auc

false_positive_rate, recall, thresholds = roc_curve(y_test, y_test_pred)
roc_auc = auc(false_positive_rate, recall)

print('AUC:', auc(false_positive_rate, recall))

plot(false_positive_rate, recall, 'b')
plot([0, 1], [0, 1], 'r--')
title('AUC = %0.2f' % roc_auc)


prob = classifier.predict_proba(x_test)

y_th = np.ones(len(y_test), dtype=bool)

for th in (0.7, 0.3):
    for i in range(len(y_test)):
        y_th[i] = prob[i][1] > th

    print('Precisión ', th, ':', accuracy_score(y_test, y_th))
    print('Exactitud ', th, ':', precision_score(y_test, y_th))
    print('Exhaustividad ', th, ':', recall_score(y_test, y_th))
    print
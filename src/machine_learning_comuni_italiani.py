# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix)
from sklearn.model_selection import (
    cross_val_score, KFold, train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


print("""Loading dataset""")
# url = 'http://ckan.ancitel.it/dataset/80431d0c-3433-4745-8d9c-c77a0dcd3325/resource/c381efe6-f73f-4e20-a825-547241eeb457/download/cusersplenevicidocumentsdocs-lavorockancvsxr2pacomuniitaliani24102017.csv'
url = 'data/comuni_italiani.csv'
names = ['Comune', 'Provincia', 'SiglaProv', 'Regione', 'AreaGeo', 'PopResidente',
         'PopStraniera', 'DensitaDemografica', 'SuperficieKmq', 'AltezzaCentro',
         'AltezzaMinima', 'AltezzaMassima', 'ZonaAltimetrica', 'TipoComune',
         'GradoUrbaniz', 'IndiceMontanita', 'ZonaClimatica', 'ZonaSismica',
         'ClasseComune', 'Latitudine', 'Longitudine']
dataset = pandas.read_csv(
    url, names=names, delimiter=';', header=0, decimal=',')

print("""
Dimensions of Dataset
We can get a quick idea of how many instances (rows) and how many attributes
(columns) the data contains with the shape property.""")
print(dataset.shape)

print("""
Peek at the data
It is also always a good idea to actually eyeball your data.""")
print(dataset.head(20))

print("""
Statistical Summary
Now we can take a look at a summary of each attribute.
This includes the count, mean, the min and max values as well as
some percentiles.""")
print(dataset.describe())

print("""
Class Distribution
Let’s now take a look at the number of instances (rows) that belong to each class.
We can view this as an absolute count.""")
print(dataset.groupby(['AreaGeo', 'Regione', 'Provincia']).size())


print("""
Data Visualization""")
print("box and whisker plots")
dataset.plot(kind='box', subplots=True, layout=(
    2, 10), sharex=False, sharey=False)
plt.show()

print("histograms")
dataset.hist()
plt.show()

print("scatter plot matrix")
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:, 10:12]
Y = array[:, 9]
Y = Y.astype('int')
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='auto')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
cls = KNeighborsClassifier()
cls.fit(X_train, Y_train)
predictions = cls.predict(X_validation)
print("""
KNeighborsClassifier
Accuracy score: {0}
Confusion matrix:\n{1}
Classification report:\n{2}
""".format(
    accuracy_score(Y_validation, predictions),
    confusion_matrix(Y_validation, predictions),
    classification_report(Y_validation, predictions),
))

cls = SVC(gamma='auto')
cls.fit(X_train, Y_train)
predictions = cls.predict(X_validation)
print("""
KNeighborsClassifier
Accuracy score: {0}
Confusion matrix:\n{1}
Classification report:\n{2}
""".format(
    accuracy_score(Y_validation, predictions),
    confusion_matrix(Y_validation, predictions),
    classification_report(Y_validation, predictions),
))

exit()

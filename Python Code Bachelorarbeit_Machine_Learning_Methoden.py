# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:36:46 2022

@author: julia
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split,GridSearchCV
#from sklearn import preprocessing,plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVC

import time



# %% ml daten

os.listdir()

df = pd.read_csv ("ml_daten_12.csv",sep=",",decimal=".",encoding= 'unicode_escape')
df.info()
df.describe(include = 'all')
df.isnull().sum().sum()


df.i2005 = df.i2005.astype('category')
df.i3005 = df.i3005.astype('category')

print(df.isna().sum())

# scaling
df_train, df_test = train_test_split(df, test_size=0.3, random_state=19)
num_vals = ["jahr","i4020","timeE_K","timeK_S","timeK_B","MR1","MR2","MR3","MR4"]
scaler = preprocessing.MinMaxScaler().fit(df_train[num_vals])
df_train[num_vals] = scaler.transform(df_train[num_vals]).copy()
df_test[num_vals] = scaler.transform(df_test[num_vals]).copy()


# speichern
minmax_mldaten_testset = df_test.copy()
minmax_mldaten_trainset = df_train.copy()

# mit standard scaler preprocessed test und traininsset erstellen

df_train, df_test = train_test_split(df, test_size=0.3, random_state=19)
scaler = preprocessing.StandardScaler().fit(df_train[num_vals])
df_train[num_vals] = scaler.transform(df_train[num_vals]).copy()
df_test[num_vals] = scaler.transform(df_test[num_vals]).copy()

# speichern

standard_mldaten_testset = df_test.copy()
standard_mldaten_trainset = df_train.copy()


# nicht proprocessed


df_train, df_test = train_test_split(df, test_size=0.3, random_state=19)


# speichern

nopre_mldaten_testset = df_test.copy()
nopre_mldaten_trainset = df_train.copy()



# %% dataframe for comparing datasets and methods - mer
 
column_names = ['Min Max Scaling','Standard Scaling', 'No Scaling']
row_names    = ['KNN', 'Dec Tree', 'SVM']

matrix = np.reshape(np.repeat(10.1, 9), (3, 3))


mer_matr = pd.DataFrame(matrix.copy(), columns=column_names, index=row_names)



# %% dataframe for comparing datasets and methods - mse
 


mse_matr = mer_matr.copy()


# %% dataframe for comparing datasets and methods - mae
 


mae_matr = mer_matr.copy()



# %% dataframe for comparing datasets and methods - speed of training
 



trainingspeed_matr = mer_matr.copy()



# %% dataframe for comparing datasets and methods - speed of fitting
 

fittingspeed_matr = mer_matr.copy()





















# %% ml daten min max


df_test = minmax_mldaten_testset
df_train = minmax_mldaten_trainset


# replacing values
df_test['i2005'].replace(['Weiblich', 'Männlich'],[0, 1], inplace=True)
df_test['i3005'].replace(['Im Wachzustand', 'Unbekannt','Im Schlaf'],[0,1,1], inplace=True)

# replacing values
df_train['i2005'].replace(['Weiblich', 'Männlich'],[0, 1], inplace=True)
df_train['i3005'].replace(['Im Wachzustand', 'Unbekannt','Im Schlaf'],[0,1,1], inplace=True)



df_train = df_train.drop("Unnamed: 0", axis = 1)
df_train_X = df_train.drop("i25001", axis = 1)
df_train_y = df_train["i25001"]

df_test = df_test.drop("Unnamed: 0", axis = 1)
df_test_X = df_test.drop("i25001", axis = 1)
df_test_y = df_test["i25001"]


# %% KNN


neighbors = np.arange(5, 50)
cross_mer = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    #Setup a k-NN Classifier with k neighbors: knn
    knn_model = KNeighborsClassifier(n_neighbors = k)
    cross_mer[i] = 1-cross_val_score(knn_model,df_train_X,df_train_y, cv=5).mean()


plt.title('k-NN: Unterschiedliche Anzahl an Nachbarn')
plt.plot(neighbors, cross_mer, label = '5 Fold Cross Validation MER')
plt.legend()
plt.xlabel('Anzahl an Nachbarn')
plt.ylabel('Misklassifikation Error Rate')
plt.show()
# %% Best model

best_knn_model = KNeighborsClassifier(n_neighbors = 44)

# Fit the classifier to the training data
start_time=time.time()
best_knn_model.fit(df_train_X, df_train_y)
trainingspeed_matr.loc['KNN','Min Max Scaling'] = time.time()-start_time
# print score
#print(best_knn_model.score(df_test_X, df_test_y))


# Predict the labels of the test data: y_pred


start_time=time.time()
y_pred = best_knn_model.predict(df_test_X)
fittingspeed_matr.loc['KNN','Min Max Scaling'] = time.time()-start_time


# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)


## plot for confusion matrix


ax = sns.heatmap(conf_mat, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix k-NN Min Max\n');
ax.set_xlabel('\nVorhergesagte Werte')
ax.set_ylabel('Echte Werte');


## Display the visualization of the Confusion Matrix.
plt.show()

## in mse matrix
mse_matr.loc['KNN','Min Max Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['KNN','Min Max Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['KNN','Min Max Scaling'] = 1-accuracy_score(df_test_y, y_pred)




# %% Decision Tree Grid Search - Zeitaufwändig

tree = DecisionTreeClassifier(random_state=19).fit(df_train_X, df_train_y)

param_dict = {
   "criterion":['gini','entropy'],
   "max_depth":range(1,10),
   "min_samples_split":range(1,10),
   "min_samples_leaf":range(1,5)
   }

grid = GridSearchCV(tree, param_grid=param_dict,cv=10,verbose=1,n_jobs=-1)

grid.fit(df_train_X,df_train_y)

grid.best_params_


grid.best_estimator_

grid.best_score_


# %% Decision Tree



start_time=time.time()
best_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=19).fit(df_train_X,df_train_y)
trainingspeed_matr.loc['Dec Tree','Min Max Scaling'] = time.time()-start_time




start_time=time.time()
y_pred = best_tree.predict(df_test_X)
fittingspeed_matr.loc['Dec Tree','Min Max Scaling'] = time.time()-start_time




# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)

## plot for confusion matrix

ax = sns.heatmap(conf_mat, fmt='', cmap='Greens')

ax.set_title('Confusion Matrix decision tree Min Max\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()


## in mse matrix
mse_matr.loc['Dec Tree','Min Max Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['Dec Tree','Min Max Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['Dec Tree','Min Max Scaling'] = 1-accuracy_score(df_test_y, y_pred)

# %% SVM Grid Search - Zeitaufwändig


svm_model = SVC().fit(df_train_X, df_train_y)

param_dict = {
   "C":[0.1,1,10,100,1000],
   "gamma":[1,0.1,0.01,0.001,0.0001],
   "kernel":['rbf'],
   }

grid = GridSearchCV(SVC(), param_grid=param_dict,cv=5,verbose=3,refit=True)

grid.fit(df_train_X,df_train_y)

grid.best_params_
#Out[162]: {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}

grid.best_estimator_
#Out[163]: SVC(C=1000, gamma=0.01)

grid.best_score_
#Out[164]: 0.42794409929946536

# %%



start_time=time.time()
best_svm = SVC(C=1000,gamma=0.01).fit(df_train_X,df_train_y)
trainingspeed_matr.loc['SVM','Min Max Scaling'] = time.time()-start_time



start_time=time.time()
y_pred = best_svm.predict(df_test_X)
fittingspeed_matr.loc['SVM','Min Max Scaling'] = time.time()-start_time


accuracy_score(df_test_y,y_pred)


# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)



## plot for confusion matrix

ax = sns.heatmap(conf_mat, fmt='', cmap='Greens')

ax.set_title('Confusion Matrix svm Min Max\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()

## in mse matrix
mse_matr.loc['SVM','Min Max Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['SVM','Min Max Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['SVM','Min Max Scaling'] = 1-accuracy_score(df_test_y, y_pred)










# %% ml daten standard scaling


os.chdir("C:\\Dokumente\\Uni\\7. Semester\\Bachelorarbeit")
os.listdir()

df_test = standard_mldaten_testset
df_train = standard_mldaten_trainset

#df_test.i2005 = df_test.i2005.astype('category')
#df_test.i3005 = df_test.i3005.astype('category')

# replacing values
df_test['i2005'].replace(['Weiblich', 'Männlich'],[0, 1], inplace=True)
df_test['i3005'].replace(['Im Wachzustand', 'Unbekannt','Im Schlaf'],[0,1,1], inplace=True)

# replacing values
df_train['i2005'].replace(['Weiblich', 'Männlich'],[0, 1], inplace=True)
df_train['i3005'].replace(['Im Wachzustand', 'Unbekannt','Im Schlaf'],[0,1,1], inplace=True)



df_train = df_train.drop("Unnamed: 0", axis = 1)
df_train_X = df_train.drop("i25001", axis = 1)
df_train_y = df_train["i25001"]

df_test = df_test.drop("Unnamed: 0", axis = 1)
df_test_X = df_test.drop("i25001", axis = 1)
df_test_y = df_test["i25001"]


# %% KNN

knn_model = KNeighborsClassifier(n_neighbors = 5)
cross_accuracy_5 = cross_val_score(knn_model,df_train_X,df_train_y, cv=5).mean()

neighbors = np.arange(20, 50)
cross_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    #Setup a k-NN Classifier with k neighbors: knn
    knn_model = KNeighborsClassifier(n_neighbors = k)
    cross_accuracy[i] = cross_val_score(knn_model,df_train_X,df_train_y, cv=5).mean()


plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, cross_accuracy, label = '5 Fold Cross Validation Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
# %% Best model

best_knn_model = KNeighborsClassifier(n_neighbors = 37)


start_time=time.time()
# Fit the classifier to the training data
best_knn_model.fit(df_train_X, df_train_y)
trainingspeed_matr.loc['KNN','Standard Scaling'] = time.time()-start_time



# Predict the labels of the test data: y_pred

start_time=time.time()
y_pred = best_knn_model.predict(df_test_X)
fittingspeed_matr.loc['KNN','Standard Scaling'] = time.time()-start_time
# Generate the confusion matrix and classification report


conf_mat = confusion_matrix(df_test_y, y_pred)


## plot for confusion matrix


ax = sns.heatmap(conf_mat, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix k-NN Standard Scaling\n');
ax.set_xlabel('\nVorhergesagte Werte')
ax.set_ylabel('Echte Werte');



## Display the visualization of the Confusion Matrix.
plt.show()


## in mse matrix
mse_matr.loc['KNN','Standard Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['KNN','Standard Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['KNN','Standard Scaling'] = 1-accuracy_score(df_test_y, y_pred)



# %% Decision Tree Grid Search - Zeitaufwändig

tree = DecisionTreeClassifier(random_state=19).fit(df_train_X, df_train_y)

param_dict = {
   "criterion":['gini','entropy'],
   "max_depth":range(1,10),
   "min_samples_split":range(1,10),
   "min_samples_leaf":range(1,5)
   }

grid = GridSearchCV(tree, param_grid=param_dict,cv=10,verbose=1,n_jobs=-1)

grid.fit(df_train_X,df_train_y)

grid.best_params_


grid.best_estimator_
#entropy,max_depth3
grid.best_score_


# %% Dec Tree


start_time=time.time()
best_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=19).fit(df_train_X,df_train_y)
trainingspeed_matr.loc['Dec Tree','Standard Scaling'] = time.time()-start_time


start_time=time.time()
y_pred = best_tree.predict(df_test_X)
fittingspeed_matr.loc['Dec Tree','Standard Scaling'] = time.time()-start_time



# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)




## plot for confusion matrix

ax = sns.heatmap(conf_mat, fmt='', cmap='Greens')

ax.set_title('Confusion Matrix decision tree Standard Scaling\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');



## Display the visualization of the Confusion Matrix.
plt.show()

## in mse matrix
mse_matr.loc['Dec Tree','Standard Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['Dec Tree','Standard Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['Dec Tree','Standard Scaling'] = 1-accuracy_score(df_test_y, y_pred)


# %% SVM Grid Search - Zeitaufwändig


svm_model = SVC().fit(df_train_X, df_train_y)

param_dict = {
   "C":[0.1,1,10,100,1000],
   "gamma":[1,0.1,0.01,0.001,0.0001],
   "kernel":['rbf'],
   }

grid = GridSearchCV(SVC(), param_grid=param_dict,cv=5,verbose=3,refit=True)

grid.fit(df_train_X,df_train_y)

grid.best_params_
#Out[162]: {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}

grid.best_estimator_
#Out[163]: SVC(C=1000, gamma=0.001)
grid.best_score_
#Out[164]: 0.42962930340066857

# %%


start_time=time.time()
best_svm = SVC(C=1000,gamma=0.001).fit(df_train_X,df_train_y)
trainingspeed_matr.loc['SVM','Standard Scaling'] = time.time()-start_time


start_time=time.time()
y_pred = best_svm.predict(df_test_X)
fittingspeed_matr.loc['SVM','Standard Scaling'] = time.time()-start_time

accuracy_score(df_test_y,y_pred)


# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)




## plot for confusion matrix

ax = sns.heatmap(conf_mat, fmt='', cmap='Reds')

ax.set_title('Confusion Matrix SVM Standard Scaling\n');
ax.set_xlabel('\nVorhergesagte Werte')
ax.set_ylabel('Echte Werte');

## Display the visualization of the Confusion Matrix.
plt.show()


## in mse matrix
mse_matr.loc['SVM','Standard Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['SVM','Standard Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['SVM','Standard Scaling'] = 1-accuracy_score(df_test_y, y_pred)






































# %% ml daten no Preproc

print("ML Daten ________________________")

# %% No preproc Daten Laden
os.chdir("C:\\Dokumente\\Uni\\7. Semester\\Bachelorarbeit")
os.listdir()

df_test = nopre_mldaten_testset
df_train = nopre_mldaten_trainset

#df_test.i2005 = df_test.i2005.astype('category')
#df_test.i3005 = df_test.i3005.astype('category')

# replacing values
df_test['i2005'].replace(['Weiblich', 'Männlich'],[0, 1], inplace=True)
df_test['i3005'].replace(['Im Wachzustand', 'Unbekannt','Im Schlaf'],[0,1,1], inplace=True)

# replacing values
df_train['i2005'].replace(['Weiblich', 'Männlich'],[0, 1], inplace=True)
df_train['i3005'].replace(['Im Wachzustand', 'Unbekannt','Im Schlaf'],[0,1,1], inplace=True)


df_train = df_train.drop("Unnamed: 0", axis = 1)
df_train_X = df_train.drop("i25001", axis = 1)
df_train_y = df_train["i25001"]

df_test = df_test.drop("Unnamed: 0", axis = 1)
df_test_X = df_test.drop("i25001", axis = 1)
df_test_y = df_test["i25001"]

# %% KNN

knn_model = KNeighborsClassifier(n_neighbors = 5)
cross_accuracy_5 = cross_val_score(knn_model,df_train_X,df_train_y, cv=5).mean()

neighbors = np.arange(5, 70)
cross_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    #Setup a k-NN Classifier with k neighbors: knn
    knn_model = KNeighborsClassifier(n_neighbors = k)
    cross_accuracy[i] = cross_val_score(knn_model,df_train_X,df_train_y, cv=5).mean()


plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, cross_accuracy, label = '5 Fold Cross Validation Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
 

#cross_accuracy.index(cross_accuracy.max())
np.argmax(cross_accuracy)
##57 das beste
# %% Best model

best_knn_model = KNeighborsClassifier(n_neighbors = 57)

# Fit the classifier to the training data
start_time=time.time()
best_knn_model.fit(df_train_X, df_train_y)
trainingspeed_matr.loc['KNN','No Scaling'] = time.time()-start_time




start_time=time.time()
# Predict the labels of the test data: y_pred
y_pred = best_knn_model.predict(df_test_X)
fittingspeed_matr.loc['KNN','No Scaling'] = time.time()-start_time


# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)


## plot for confusion matrix


ax = sns.heatmap(conf_mat, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix k-NN No Preproc\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()


## in mse matrix
mse_matr.loc['KNN','No Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['KNN','No Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['KNN','No Scaling'] = 1-accuracy_score(df_test_y, y_pred)



# %% Decision Tree Grid Search - Zeitaufwändig

tree = DecisionTreeClassifier(random_state=19).fit(df_train_X, df_train_y)

param_dict = {
   "criterion":['gini','entropy'],
   "max_depth":range(1,10),
   "min_samples_split":range(1,10),
   "min_samples_leaf":range(1,5)
   }

grid = GridSearchCV(tree, param_grid=param_dict,cv=10,verbose=1,n_jobs=-1)

grid.fit(df_train_X,df_train_y)

grid.best_params_


grid.best_estimator_

grid.best_score_
#0.43755543064143404

# %%



start_time=time.time()
best_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=19).fit(df_train_X,df_train_y)
trainingspeed_matr.loc['Dec Tree','No Scaling'] = time.time()-start_time

start_time=time.time()
y_pred = best_tree.predict(df_test_X)
fittingspeed_matr.loc['Dec Tree','No Scaling'] = time.time()-start_time



# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)




## plot for confusion matrix

ax = sns.heatmap(conf_mat, fmt='', cmap='Greens')

ax.set_title('Confusion Matrix Klassifikationsbaum ohne Skalierung\n');
ax.set_xlabel('\nVorhergesagte Werte')
ax.set_ylabel('Echte Werte');

## Display the visualization of the Confusion Matrix.
plt.show()

## in mse matrix
mse_matr.loc['Dec Tree','No Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['Dec Tree','No Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['Dec Tree','No Scaling'] = 1-accuracy_score(df_test_y, y_pred)


# %% SVM Grid Search - Zeitaufwändig


svm_model = SVC().fit(df_train_X, df_train_y)

param_dict = {
   "C":[0.1,1,10,100,1000],
   "gamma":[1,0.1,0.01,0.001,0.0001],
   "kernel":['rbf'],
   }

grid = GridSearchCV(SVC(), param_grid=param_dict,cv=5,verbose=3,refit=True)

grid.fit(df_train_X,df_train_y)

grid.best_params_
grid.best_estimator_
grid.best_score_
# %% SVM


start_time=time.time()
best_svm = SVC(C=1,gamma=0.0001).fit(df_train_X,df_train_y)
trainingspeed_matr.loc['SVM','No Scaling'] = time.time()-start_time


start_time=time.time()
y_pred = best_svm.predict(df_test_X)
fittingspeed_matr.loc['SVM','No Scaling'] = time.time()-start_time


# Generate the confusion matrix and classification report
conf_mat = confusion_matrix(df_test_y, y_pred)



## plot for confusion matrix

ax = sns.heatmap(conf_mat, fmt='', cmap='Greens')

ax.set_title('Confusion Matrix SVM No Preproc\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()

## in mse matrix
mse_matr.loc['SVM','No Scaling'] = mean_squared_error(df_test_y.values, y_pred)

## in mae matrix
mae_matr.loc['SVM','No Scaling'] = mean_absolute_error(df_test_y.values, y_pred)

## in mer matrix
mer_matr.loc['SVM','No Scaling'] = 1-accuracy_score(df_test_y, y_pred)






























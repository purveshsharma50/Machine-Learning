#!/usr/bin/env python
# coding: utf-8

# In[21]:


#************************************************************************************ 
# Purvesh Sharma
# Filename: perceptron.py
# Due: 9/18/19
#
# Objective:
# To Classify and predict Cars dataset using perceptron model for high safety >= HighPrice
# #*************************************************************************************


#Importing all required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Reading Dataset from users saved location

df = pd.read_csv('/Users/purveshsharma/Documents/TXST Study Material/Machine Learning/Anaconda files/datasetcars.csv')


# To nominate the column

df.columns = ['buying','maint','doors','person','lug_boot','safety','classlabel']

# Mapping of values for buying, maintenence and safety

buying_mapping = {'vhigh': 4,'high':3,'med': 2,'low':1}
maint_mapping={'vhigh': 4,'high':3,'med': 2,'low':1}
safety_mapping={'high':3,'med':2,'low':1}
df['buying'] = df['buying'].map(buying_mapping)
df['maint'] = df['maint'].map(maint_mapping)
df['safety'] = df['safety'].map(safety_mapping)

# This is for class mapping

class_mapping = {label:idx for idx,label in 
                 enumerate(np.unique(df['classlabel']))}
class_mapping ={'unacc':0,'acc':1,'good':2,'vgood':3}
df['classlabel'] = df['classlabel'].map(class_mapping)

# We need only prices,safety and class

X, y = df.iloc[:, [0,5]].values, df.iloc[:, 6].values

# Defining testing and training data  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# standarization of training and testing data

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Normalisation of data

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train_std)
X_test_norm = mms.transform(X_test_std)

# optional you want to print the data

#print(X_train_norm)
#print(X_test_norm)


#Training and prediction using Perceptron

# Getting more accuray without Normalization

ppn = Perceptron(max_iter= 100, eta0=0.01, random_state=0) 
ppn.fit(X_train_std, y_train) #This is training the model 
y_pred = ppn.predict(X_test_std) #Test/Validating the model 

#Printing of results and plot

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print('Misclassified samples: %d' % (y_test != y_pred).sum())
      
print('Accuracy for Perceptron: %.2f' % accuracy_score(y_test, y_pred))
print('Test Accuracy for Perceptron: %.2f' % ppn.score(X_test_std, y_test))
print('Train Accuracy for Tree: %.2f' % ppn.score(X_train_std, y_train))
# plot data
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v') 
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
     # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
           # highlight test samples
    if test_idx:
           # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                           c='', edgecolor='black', alpha=1.0,
                           linewidth=1, marker='o',
                           s=100, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                     y=y_combined,
                     classifier=ppn,
                       test_idx=range(105, 150))
plt.xlabel('Prices')
plt.ylabel('Safety')
plt.legend(loc='upper left')
plt.show()

# End of the code           



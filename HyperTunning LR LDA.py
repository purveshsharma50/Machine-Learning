#!/usr/bin/env python
# coding: utf-8

# In[14]:


#************************************************************************************ 
# Purvesh Sharma
# Filename: Hyper Tunning LogicalRegression LDA.py
# Due: 10/04/19
#
# Objective:
# To Classify and predict Cars dataset using LogicalRegression 
# and to do hypertunning and optimizing the parameters using LDA
# #*************************************************************************************



#Importing all required libraries


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import os

# Reading Dataset from users saved location

df = pd.read_csv('/Users/purveshsharma/Documents/Anaconda files/datasetcars.csv')

# To nominate the column

df.columns = ['buying','maint','doors','persons','lug_boot','safety','classlabel']

# Mapping of values for buying, maintenence and safety

buying_mapping = {'vhigh': 4,'high':3,'med': 2,'low':1}
maint_mapping={'vhigh': 4,'high':3,'med': 2,'low':1}
doors_mapping={'5more': 4,'4':3,'3': 2,'2':1}
persons_mapping={'more':3,'4':2,'2':1}
lug_boot_mapping={'big':3,'med':2,'small':1}
safety_mapping={'high':3,'med':2,'low':1}

df['buying'] = df['buying'].map(buying_mapping)
df['maint'] = df['maint'].map(maint_mapping)
df['doors'] = df['doors'].map(doors_mapping)
df['persons'] = df['persons'].map(persons_mapping)
df['lug_boot'] = df['lug_boot'].map(lug_boot_mapping)
df['safety'] = df['safety'].map(safety_mapping)

# This is for class mapping

class_mapping = {label:idx for idx,label in 
                 enumerate(np.unique(df['classlabel']))}
class_mapping ={'unacc':0,'acc':1,'good':2,'vgood':3}
df['classlabel'] = df['classlabel'].map(class_mapping)

# Mapping of values for buying, maintenence,doors,persons,lug_boot and safety

X, y = df.iloc[:, [0,1,2,3,4,5]].values, df.iloc[:, 6].values

# Defining testing and training data  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# Creating Pipeline

pipe_lr = make_pipeline(StandardScaler(),LDA(n_components=2),LogisticRegression(random_state=1))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)

print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores = [] #initialize the object vector 
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test]) 
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, 
                                                 np.bincount(y_train[train]), score))
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train, cv=20,n_jobs=-1)
print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#Parameters range

param_grid = [{'logisticregression__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]

gs = GridSearchCV (estimator=pipe_lr, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

gs = gs.fit(X_train, y_train)
a = (gs.best_params_)
b = ('\nTraining accuracy: %.3f' % gs.best_score_)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
c = ('\nTest accuracy: %.3f' % clf.score(X_test, y_test))

#printing the optimized hypermarameters, training and test accuracy values

print (a)
print (b)
print (c)

with open("LR hypertunning LDA_output.txt", "w") as text_file:
    text_file.write(format(a))
    text_file.write(format(b))
    text_file.write(format(c))
    
#standerized data for plotting  

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#applying LDA for plotting

lda = LDA(n_components=2)
X_train_lda = lda1.fit_transform(X_train_std, y_train)
X_test_lda = lda1.transform(X_test_std)

#Creating logistic regression with optimized hyperparameters for plotting

lr = LogisticRegression(C=gs.best_params_['logisticregression__C'], random_state=1)

#This is training the model for plotting

lr.fit(X_train_lda, y_train)

# setup marker generator and color map

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    class_label=('acc', 'good', 'unacc', 'vgood')
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=class_label[cl], edgecolor='black')

#plot for training

plot_decision_regions(X=X_train_lda, y=y_train, classifier=lr)
plt.xlabel('ld1')
plt.ylabel('ld2')
plt.legend(loc='best')
plt.show()

# plot for Test

plot_decision_regions(X=X_test_lda, y=y_test, classifier=lr)
plt.xlabel('ld1')
plt.ylabel('ld2')
plt.legend(loc='best')
plt.show()

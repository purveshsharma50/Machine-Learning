#!/usr/bin/env python
# coding: utf-8

# In[11]:
#************************************************************************************
# Purvesh Sharma
# Filename: Hyper Tunning LogicalRegression LDA.py
# Due: 10/04/19
#
# Objective:
# To Classify and predict Cars dataset using SVM
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
from sklearn.svm import SVC
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

df = pd.read_csv('/Users/purveshsharma/Documents/TXST Study Material/Machine Learning/Anaconda files/datasetcars.csv')

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

# We need only prices,safety and class

X, y = df.iloc[:, [0,1,2,3,4,5]].values, df.iloc[:, 6].values

# Defining testing and training data  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)


pipe_svc = make_pipeline(StandardScaler(),LDA(n_components=2),SVC(random_state=1))
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

c= ('Test Accuracy: %.3f' % pipe_svc.score(X_test, y_test))



kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores = [] #initialize the object vector 
for k, (train, test) in enumerate(kfold):
    pipe_svc.fit(X_train[train], y_train[train])
    score = pipe_svc.score(X_train[test], y_train[test]) 
    scores.append(score)
    d=('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, 
                                                 np.bincount(y_train[train]), score))
e=('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

scores = cross_val_score(estimator=pipe_svc,X=X_train,y=y_train, cv=20,n_jobs=-1)
f= ('CV accuracy scores: %s' % scores)

g = ('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] 
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']
                               }
                              ]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

                  
a=(gs.best_score_)
b=(gs.best_params_)

print (a)
print (b)
print (c)
print (d)
print (e)
print (f)
print (g)

with open("SVM hypertunning LDA_output.txt", "w") as text_file:
    text_file.write(format(a))
    text_file.write(format(b))
    text_file.write(format(c))
    text_file.write(format(d))
    text_file.write(format(e))
    text_file.write(format(f))   
    text_file.write(format(g))


# In[ ]:





# In[ ]:





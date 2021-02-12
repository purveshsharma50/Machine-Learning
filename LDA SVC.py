#!/usr/bin/env python
# coding: utf-8

# In[7]:


#************************************************************************************ 
# Purvesh Sharma
# Filename: SVM LDA.py
# Due: 10/04/19
#
# Objective:
# To Classify and predict Cars dataset using SVM model
# and to find two most influential feature using LDA
# #*************************************************************************************


#Importing all required libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os

# Reading Dataset from users saved location

df = pd.read_csv('/Users/purveshsharma/Documents/TXST Study Material/Machine Learning/Anaconda files/datasetcars.csv')


# To nominate the column

df.columns = ['buying','maint','doors','persons','lug_boot','safety','classlabel']

# Mapping of values for buying, maintenence,doors,persons,lug_boot and safety

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

# We need all features and class

X, y = df.iloc[:, [0,1,2,3,4,5]].values, df.iloc[:, 6].values

# Defining testing and training data  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# standarization of training and testing data

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#computing the mean vactors

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
#calculating within class scatter matrix     
    
    
d = 6 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1) 
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
    print('Class label distribution: %s' % np.bincount(y_train)[1:])
    
#calculating overall mean     
    
mean_overall = np.mean(X_train_std, axis=0) 
d = 6 # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # make column vector
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot(
               (mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (
    S_B.shape[0], S_B.shape[1]))

# Generalized Eigen value matrix

eigen_vals, eigen_vecs =np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
   
# Make a list of (eigenvalue, eigenvector) tuples

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low

eigen_pairs.sort(key=lambda k: k[0], reverse=True) 
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))

print('Matrix W:\n', w)


#Transformation of training dataset

X_train_lda = X_train_std.dot(w)

print('X_train_lda: \n', X_train_lda)

#Cumulative sum of variance and sorting of Eigen values in decreasing order corresponding to eigen vector

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

# Linear Discriminant Analysis 

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = SVC()
lr = lr.fit(X_train_lda, y_train)
y_pred = lr.predict(X_test_lda)

#Printing of results and plot


a = ('Accuracy for SVM: %.3f' % accuracy_score(y_test, y_pred))
b = ('training accuracy for SVM: %.3f' % lr.score(X_train_lda, y_train))
c = ('Test accuracy for SVM: %.3f' % lr.score(X_test_lda, y_test))
d = ('Misclassified samples: %d' % (y_test != y_pred).sum())
print(a)
print(b)
print(c)
print(d)
                                              
with open("LDA_SVC_output.txt", "w") as text_file:
    text_file.write(format(a))
    text_file.write(format(b))
    text_file.write(format(c))
    text_file.write(format(d))


# plot for training

plot_decision_regions(X_train_lda, y_train, clf =lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

# plot for test

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, clf=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

# Discriminability graph

plt.bar(range(1,7), discr , alpha=0.5, align= 'center', label= 'individual "discriminability"')
plt.step(range(1,7), cum_discr, where='mid',label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminability')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

# End of the code



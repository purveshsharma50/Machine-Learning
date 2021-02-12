#!/usr/bin/env python
# coding: utf-8

# In[9]:


#************************************************************************************ 
# Purvesh Sharma
# Filename: LogicalRegression PCA.py
# Due: 10/04/19
#
# Objective:
# To Classify and predict Cars dataset using LogicalRegression 
# and to find two most influential feature using PCA
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
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
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

# Construct the covariance matrix

cov_mat = np.cov(X_train_std.T)

# Decompose the covariance matrix to Eigen values and Eigen vector

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Printing of Eigen values and Eigen Vector

print('\nEigenvalues \n%s' % eigen_vals)
print('\nEigen Vec \n%s' % eigen_vecs)

#Cumulative sum of variance and sorting of Eigen values in decreasing order corresponding to eigen vector

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Make a list of (eigenvalue, eigenvector) tuples

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low

eigen_pairs.sort(key=lambda k: k[0], reverse=True) 
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

print('Matrix W:\n', w)

# Training Dataset onto the two principal components

X_train_pca = X_train_std.dot(w)

print('X_train_pca: \n', X_train_pca)

# setup marker generator and color map

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)
        

            
# principal components analysis 

pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
y_pred = lr.predict(X_test_pca)
        
#Printing of results and plot       
       
        
a = ('Accuracy for LR: %.2f' % accuracy_score(y_test, y_pred))
b = ('training accuracy for LR: %.3f' % lr.score(X_train_pca, y_train))
c = ('Test accuracy for LR: %.3f' % lr.score(X_test_pca, y_test))
d = ('Misclassified samples: %d' % (y_test != y_pred).sum())
print(a)
print(b)
print(c)
print(d)
                                              
with open("PCA_LR_output.txt", "w") as text_file:
    text_file.write(format(a))
    text_file.write(format(b))
    text_file.write(format(c))
    text_file.write(format(d))


# plot for training

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

# Plot for Test 

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

# Variance graph 
  
plt.bar(range(1,7), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,7), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()
       

# End of the code 
        

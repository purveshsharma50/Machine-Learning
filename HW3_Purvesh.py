#!/usr/bin/env python
# coding: utf-8

# In[58]:


#************************************************************************************ 
# Purvesh Sharma
# EE5321 – HW#3
# Filename: homework3-ANN.py
# Due: 10/11/19
#
# Objective:
# Designing of best ANN Using TensorFlow & Keras.
# classify the freight tracking and tracing of the transportation data.
# #*************************************************************************************

#Importing all required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Reading Dataset from users saved location

df = pd.read_csv('/Users/purveshsharma/Documents/TXST Study Material/Machine Learning/Anaconda files/hw3/c2k_data_comma.csv')

# Replacing all question marks with Nan value

df=df.replace(to_replace ="?", value ="NaN") 

# Imputing all NaN value with a mean value

imr = Imputer(missing_values='NaN', strategy='mean', axis=0) 
imr = imr.fit(df.values)
df_data = imr.transform(df.values)
print(df_data)


#convert the dataset to train and test part

X = df_data[:,:97]
y = df_data[:,97]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1 )

# finding standarized and normalized values

mean_vals = np.mean(X_train, axis=0) 
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val 
X_test_centered = (X_test - mean_vals)/std_val 
del X_train, X_test


print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

np.random.seed(123)


# One-Hot Encode the classes

y_train_onehot = keras.utils.to_categorical(y_train)


# Function to create model, required for KerasClassifier

model = keras.models.Sequential() 
model.add(
    keras.layers.Dense( units=97, 
                       input_dim=X_train_centered.shape[1],
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros',
                       activation='sigmoid'))
model.add(
    keras.layers.Dense( units=97,
                        input_dim=97, 
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', 
                        activation='tanh'))
model.add(
     keras.layers.Dense( units=y_train_onehot.shape[1],
                        input_dim=3, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='zeros', 
                        activation='softmax'))

# optimizing the values using categorical crossentropy

sgd_optimizer = keras.optimizers.SGD(
    lr=0.01, decay=1e-7, momentum=0.9)
model.compile(optimizer=sgd_optimizer, 
              loss='categorical_crossentropy')

# Train the ANN

history = model.fit(X_train_centered, 
                    y_train_onehot,
                    batch_size=32,
                    epochs=1000,
                    verbose=1,
                    validation_split=0.2)

# Pridicting the Training and Testing Accuracy

y_train_pred = model.predict_classes(X_train_centered, verbose=0) 
correct_preds = np.sum(y_train == y_train_pred, axis=0) 
train_acc = correct_preds / y_train.shape[0]

TrainAccuracy=('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test_centered, verbose=0) 
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

TestAccuracy=('Test accuracy: %.2f%%' % (test_acc * 100))

print(TrainAccuracy)
print(TestAccuracy)

# Converting to the text output file

with open("HW3 output.txt", "w") as text_file:
    text_file.write(format(TrainAccuracy))
    text_file.write(format(TestAccuracy))
    
#End of the code
#******************************************************************************************************************


# In[ ]:





# In[ ]:





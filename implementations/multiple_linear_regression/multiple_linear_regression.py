# -*- coding: utf-8 -*-
"""multiple_linear_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11z_6qntafanPM5q6HFH3Sz5iF7HJ5jsP

# Multiple Linear Regression

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)

"""## Encoding categorical data"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') #index of the var we want to encode
X = np.array(ct.fit_transform(X))
print(X)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Training the Multiple Linear Regression model on the Training set"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = regressor.predict(X_test);
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Q1: how to predict the profit of a startup with R&D spend of 160k admin spend 130k, marketing spend 300k and located in California??
regressor.predict([[1, 0, 0, 160000, 130000, 300000]])

#Q2: how do I get the final regression equation??

print(regressor.coef_)
print(regressor.intercept_)
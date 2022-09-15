#!/usr/bin/env python
# coding: utf-8

# # Polynomial Regression

# ## Importing the libraries

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[51]:


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# ## Training the Linear Regression model on the whole dataset

# In[52]:


from sklearn.linear_model import LinearRegression

linearRegressor = LinearRegression()
linearRegressor.fit(X, y)
#linearRegressor.predict([[6]])


# ## Training the Polynomial Regression model on the whole dataset

# In[79]:


from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree = 8) #degree matches the 'n' from the equation
X_poly = polynomialRegressor.fit_transform(X)
linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_poly, y)


# ## Visualising the Linear Regression results

# In[80]:


plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor.predict(X), color = 'blue')
plt.title('Truth or Blufff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# this graph shows how bad of a predictor a linear regression is in this case scenario


# ## Visualising the Polynomial Regression results

# In[81]:


plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor2.predict(X_poly), color = 'blue')
plt.title('Truth or Blufff (polinomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# In[82]:


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linearRegressor2.predict(polynomialRegressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Predicting a new result with Linear Regression

# In[83]:


linearRegressor.predict([[6.5]])


# ## Predicting a new result with Polynomial Regression

# In[78]:


linearRegressor2.predict(polynomialRegressor.fit_transform([[6.5]]))


# In[ ]:





# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ones

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# 0 1 NY
# 0 0 CA
# 1 0 FL
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
def backwardElim(X_opt, SL):
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    for i in range(np.size(X_opt,1)):
        if regressor_OLS.pvalues[i] > SL:
            if regressor_OLS.pvalues[i] == max(regressor_OLS.pvalues):
                print(regressor_OLS.summary())
                print("removing: "+str(i)+", with P val: "+str(regressor_OLS.pvalues[i]))
                return backwardElim(np.delete(X_opt, i, axis=1), SL)
    return X_opt

import statsmodels.formula.api as sm
X = np.append(arr =np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
SL = 0.05
print(X_opt)
X_opt = backwardElim(X_opt, SL)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Constant, R&D Spend
print(regressor_OLS.summary())
print(X_opt)
            
            
            
            
            
            
            
            
            
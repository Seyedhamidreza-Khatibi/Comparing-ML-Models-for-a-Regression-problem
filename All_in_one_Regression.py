# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
##############################
##Multiple Linear Regression##(Scaling is not needed)
##############################
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Evaluating the Model Performance(R_2 square)
from sklearn.metrics import r2_score
print('Multiple_Linear_Regressio_Model=',r2_score(y_test, y_pred))
##############################
##Polynomial Regression##(Scaling is not needed)
##############################
# Training the Polynomial Regression model on the Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)
# Predicting the Test set results
y_pred2= regressor.predict(poly_reg.transform(X_test))
# Evaluating the Model Performance(R square)
from sklearn.metrics import r2_score
print('Polynomial_Regression_Model=',r2_score(y_test, y_pred2))
##############################
##Support Vector Regression(SVR)##Scaling must be done with this methd
##############################
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y_train3=y_train.reshape(len(y_train),1)
y_test3=y_test.reshape(len(y_test),1)
# Feature Scaling-->It will make the varibales usually between -3,+3
from sklearn.preprocessing import StandardScaler
#Indepnednt varibale(features)
sc_X = StandardScaler()
X_train3= sc_X.fit_transform(X_train)
#Depnednt varibale
sc_y = StandardScaler()
y_train3 = sc_y.fit_transform(y_train3)
# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train3, y_train3.ravel())
# Predicting the Test set results
# Always use sc_y.inverse_transform() and sc_x.inverse_transform() to go back to original scale for y and x respectively
y_pred3= sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))#scaling X_test by sc_X.transform(X_test)
# Evaluating the Model Performance(R square)
from sklearn.metrics import r2_score
print('SVM_Model=',r2_score(y_test3, y_pred3))
##############################
##Decision Tree Regression##-->Scaling is not needed
##############################
# Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred4= regressor.predict(X_test)
# Evaluating the Model Performance (R_2 square)
from sklearn.metrics import r2_score
print('Decision Tree Regression=',r2_score(y_test, y_pred4))

##############################
##Random Forest Regression-->Scaling is not needed
##############################
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)#n=number of trees
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred5= regressor.predict(X_test)
# Evaluating the Model Performance (R_2 square)
from sklearn.metrics import r2_score
print('Random Forest Regression',r2_score(y_test, y_pred5))
##############################
##XGBoost-->Scaling is not needed
##############################
# Training XGBoost on the Training set
#for regression
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
# Evaluating the Model Performance(R_2 square)
from sklearn.metrics import r2_score
y_pred6 = regressor.predict(X_test)
print('XGBoost=',r2_score(y_test, y_pred6))


# Applying k-Fold Cross Validation for any of the above modles(to check the performance in a more general form)
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

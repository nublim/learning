# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:23:51 2019

@author: Nubli
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

#importing dataset
dataset=pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
Y=y/1000
#splitting dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/4,random_state=0)

#my algorithm
N=len(Y_train)
b=np.mean(Y_train)
cov=np.cov(Y_train)
xm=np.mean(X_train)
x_shifted=(X_train-xm)
y_shifted=Y_train-b
x_inv=x_shifted/(np.matmul(x_shifted.T,x_shifted))
a=np.matmul(y_shifted.T,x_inv)
yh=(a*(X_test-xm))+b
e=abs(yh.T-Y_test)
mse=np.square(e.T).mean()

#using scikit
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)
Yh=reg.predict(X_test)
e1=abs(Yh.T-Y_test)
mse1=np.square(e1.T).mean()

#showing data
plt.scatter(X_test,Y_test,c='b')
plt.plot(X_test,yh,C='g')
plt.plot(X_test,Yh,C='r')
plt.grid()
plt.show()

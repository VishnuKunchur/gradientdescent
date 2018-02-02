"""Implementing Gradient Descent for Logistic Regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

wine = datasets.load_wine()

def FeatureExtractor(x):                ##Takes a user defined input of columns for the 'wine' dataframe, outputs matrices of X and Y
    x_data = np.c_[wine.data, wine.target] ##Concatenating x features and y target variables
    colnames = np.append(wine.feature_names, 'target') ##
    wine_data = pd.DataFrame(x_data, columns = colnames)
    wine_data = wine_data[wine_data.iloc[:,13] != 2] ##We're doing binary classification for now!
    x1_data = wine_data.iloc[:, x] 
    global features
    features = list(x1_data.columns)
    global y_data
    y_data = np.matrix(wine_data.loc[:, 'target']).T ##
    m = len(y_data)
    ones_vect = np.ones(m)
    global X_data
    X_data = np.matrix(np.c_[ones_vect, x1_data])
    
def sigmoid(z):
    sig_z = 1/(1+np.exp(-z))
    return sig_z

def CrossEntropy(theta):   ### *theta* is a [nx1] row vector, where n is the number of features selected 
    hyp = X_data * theta
    if max(hyp) > 40:
        print('Choose a smaller value of theta!')
        exit
    else:
        J = -(y_data.T)*np.log(sigmoid(hyp)) - ((1-y_data).T)*np.log(1-sigmoid(hyp))
        return J[0,0]

theta = np.matrix(np.random.rand(X_data.shape[1])).T

def GradientStep(alpha):
    global theta
    hyp = X_data * theta
    theta -= alpha*X_data.T*(sigmoid(hyp) - y_data)
    return theta
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
theta_val = []
cost_val = []


    a = CrossEntropy(theta)
    print(GradientStep(0.0005))
    b = CrossEntropy(theta)
print(CrossEntropy(theta))
print(theta)
    


        
        
        
        












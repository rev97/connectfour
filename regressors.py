#Author: Revanth Gottuparthy (UF ID: 5445 2992)
# This script contains the functions that evaluates a regressor on the intermediate boards optimal play (multi label) dataset.
# Importing all the required modules
import numpy as np 
from numpy import mean
import pandas as pd

#Import scikit-learn metrics module for classifiers, confusion matrix and accuracy calculation
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

#To ignore the depricated warnings
import warnings
warnings.filterwarnings('ignore')

import sys

def run_knn_regressor(data):
    tictac_final_df = pd.DataFrame(data, columns = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8'])
    lst = dict()
    for i in range(0,8):
        df = tictac_final_df.iloc[:,[0,1,2,3,4,5,6,7,8,9+i]]
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:9], df.iloc[:, 9], test_size=0.3,random_state=109)
    
        # train the model
        # selecting number of neighbors to 5
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # prepare the cross-validation procedure
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        # evaluate model
        scores = cross_val_score(knn, X_test, y_test, cv=cv, n_jobs=-1)
        # compute accuracy of the model
        print("at y",i)
        print("Accuracy of KNN regressor for intermediate boards optimal play (multi label) dataset: {:.0%}".format(mean(scores)))
        lst[i] = mean(scores)
    Keymax = max(lst, key= lambda x: lst[x])
    #Best output model for optimal moves is 
    print("Best output model for optimal moves is y_{} with accuracy {:.0%}".format(Keymax,lst[Keymax]))
        

def run_mlp_regressor(data):
    tictac_final_df = pd.DataFrame(data, columns = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8'])
    lst = dict()
    for i in range(0,8):
        df = tictac_final_df.iloc[:,[0,1,2,3,4,5,6,7,8,9+i]]
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:9], df.iloc[:, 9], test_size=0.3,random_state=109)
        # train the model
        mlp_reg = MLPRegressor(random_state=1, max_iter=500)
        mlp_reg.fit(X_train, y_train)
        y_pred = mlp_reg.predict(X_test)
        # prepare the cross-validation procedure
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        # evaluate model
        scores = cross_val_score(mlp_reg, X_test, y_test, cv=cv, n_jobs=-1)
        # compute accuracy of the model
        print("at y",i)
        print("Accuracy of MLP regressor for intermediate boards optimal play (multi label) dataset: {:.0%}".format(mean(scores)))
        lst[i] = mean(scores)
    Keymax = max(lst, key= lambda x: lst[x])
    #Best output model for optimal moves is 
    print("Best output model for optimal moves is y_{} with accuracy {:.0%}".format(Keymax,lst[Keymax]))

def find_theta(X, y):
    
    
    # The Normal Equation
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    
    return theta

def predict(X,theta):
    
    
    # preds is y_hat which is the dot product of X and theta.
    preds = np.dot(X, theta)
    
    return preds

def loss_func(pred,y):
    return np.mean(np.square(np.subtract(pred, y), out=None, where=True, dtype=None))


def run_linear_regression(data):
    tictac_final_df = pd.DataFrame(data, columns = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8'])
    lst = dict()
    for i in range(0,8):
        df = tictac_final_df.iloc[:,[0,1,2,3,4,5,6,7,8,9+i]]
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:9], df.iloc[:, 9], test_size=0.3,random_state=109)
        theta = find_theta(X_train, y_train)
        preds = np.round(predict(X_test,theta), 0)
        loss = loss_func(preds,y_test)
        # compute accuracy of the model
        print("at y",i)
        print("Loss value of Normal equation regressor for intermediate boards optimal play (multi label) dataset",loss)
        lst[i] = loss
        
            
    Keymax = min(lst, key= lambda x: lst[x])
    #Best output model for optimal moves is 
    print(lst[Keymax])
    print("Best output model for optimal moves is y_{} with least loss {}".format(Keymax,lst[Keymax]))


if __name__ == "__main__":
    #reading the system arguments
    file_path = sys.argv[1]
    #Reading the required data sets for tictac game
    tictac_array_multi = np.loadtxt(file_path+"tictactoedatasets/tictac_multi.txt")
    #Calling linear SVM,KNN and MLP regressor functions for intermediate board(multi label)
    #Each function takes one argument: 1.Input Array
    run_linear_regression(tictac_array_multi)
    run_knn_regressor(tictac_array_multi)
    run_mlp_regressor(tictac_array_multi)
    
    
    
    
    
    

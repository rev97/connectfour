#Author: Revanth Gottuparthy (UF ID: 5445 2992)
# This script contains the functions that outputs the statistical accuracy and confusion matrices that record the performance of a classifier on two classification datasets: final boards classification dataset and intermediate boards optimal play (single label).
# Importing all the required modules
import numpy as np 
from numpy import mean
import sys
import pandas as pd

#Import scikit-learn metrics module for classifiers, confusion matrix and accuracy calculation 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
#Importing to plot the confusion matrix on console
import matplotlib.pyplot as plt
#To ignore the depricated warnings
import warnings
warnings.filterwarnings('ignore')



def run_linearsvm_classifier(data,type):
    tictac_final_df = pd.DataFrame(data, columns = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','y'])
    #splitting train and test data
    # Adjust your train and test size with your choice 
    X_train, X_test, y_train, y_test = train_test_split(tictac_final_df.iloc[:,0:9], tictac_final_df.iloc[:, 9], test_size=0.3,random_state=109)
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    # prepare the cross-validation procedure
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # evaluate model
    scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    # compute accuracy of the MLP Classifier model
    print("Accuracy of linear SVM classifier for "+str(type)+" boards classification dataset: {:.0%}".format(mean(scores)))
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    #printing the non normalized and normalized confusion matrix
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            #display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


def run_knn_classifier(data,type):
    tictac_final_df = pd.DataFrame(data, columns = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','y'])
    #splitting train and test data
    # Adjust your train and test size with your choice 
    X_train, X_test, y_train, y_test = train_test_split(tictac_final_df.iloc[:,0:9], tictac_final_df.iloc[:, 9], test_size=0.3,random_state=109)
    # train the model
    # selecting number of neighbors to 8
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # prepare the cross-validation procedure
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # evaluate model
    scores = cross_val_score(knn, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    # compute accuracy of the MLP Classifier model
    print("Accuracy of KNN classifier for "+str(type)+" boards classification dataset: {:.0%}".format(mean(scores)))


    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    #printing the non normalized and normalized confusion matrix
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            knn,
            X_test,
            y_test,
            #display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


def run_mlp_classifier(data,type):
    tictac_final_df = pd.DataFrame(data, columns = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','y'])
    #splitting train and test data
    # Adjust your train and test size with your choice 
    X_train, X_test, y_train, y_test = train_test_split(tictac_final_df.iloc[:,0:9], tictac_final_df.iloc[:, 9], test_size=0.3,random_state=109)
    # training the MLP Classifier model
    mlp_cls = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    mlp_cls.fit(X_train, y_train)
    y_pred = mlp_cls.predict(X_test)
    # prepare the cross-validation procedure
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # evaluate model
    scores = cross_val_score(mlp_cls, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    # compute accuracy of the MLP Classifier model
    print("Accuracy of MLP classifier for "+str(type)+" boards classification dataset: {:.0%}".format(mean(scores)))
    
   
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    #printing the non normalized and normalized confusion matrix
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            mlp_cls,
            X_test,
            y_test,
            #display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()



    
    
    
    
    
    


if __name__ == "__main__":
    #reading the system arguments
    file_path = sys.argv[1]
    #Reading the required data sets for tictac game
    tictac_array = np.loadtxt(file_path+"tictactoedatasets/tictac_final.txt", delimiter=" ")
    tictac_array_single = np.loadtxt(file_path+"tictactoedatasets/tictac_single.txt", delimiter=" ")
    #Calling linear SVM,KNN and MLP classifier functions for final board and intermediate board(single label)
    #Each function takes two arguments: 1.Input Array 2.Type of data {Final, Intermediate}
    run_linearsvm_classifier(tictac_array,"Final")
    run_linearsvm_classifier(tictac_array_single,"Intermediate")
    run_knn_classifier(tictac_array,"Final")
    run_knn_classifier(tictac_array_single,"Intermediate")
    run_mlp_classifier(tictac_array,"Final")
    run_mlp_classifier(tictac_array_single,"Intermediate")


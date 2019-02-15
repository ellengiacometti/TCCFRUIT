import cv2
import numpy as np
# import os
# import glob
from TrataImagem import TrataImagem
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import ast


if __name__ == '__main__':
    # teste = lambda x: list(map(int, x.strip("[]").replace("'", "").split(", ")))
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train.csv',index_col=False,sep=";",converters={'Color': teste})
    test = pd.read_csv('Test.csv', index_col=False,sep=";",converters={'Color': teste})
    XTrain =[list(map(float, hist)) for hist in train['Color']]
    XTest = [list(map(float, hist)) for hist in test['Color']]
    ColorLabel = train['ColorLabel']
    leTrain = LabelEncoder()
    leTrain.fit(ColorLabel)
    ColorLabel = leTrain.transform(ColorLabel)
    ColorTest = XTest
    ColorLabelTest=test['ColorLabel']
    leTest = LabelEncoder()
    leTest.fit(ColorLabelTest)
    ColorLabelTest= leTest.transform(ColorLabelTest)

    # svm = SVC()
    # parameters = {'C': (1, 0.25, 0.5, 0.75,0.05), 'gamma': (0.5,1, 2, 3, 'auto'),'class_weight': [{0: 1,1: w2} for w2 in [2, 4, 6, 10,12]]}
    # clf = GridSearchCV(svm, parameters,verbose = 2)
    # clf.fit(XTrain, ColorLabel)
    # print("accuracy:" + str(np.average(cross_val_score(clf, XTrain, ColorLabel, scoring='accuracy'))))
    # print(clf.best_params_)


    """ SVM """
     # create the SVM classifier
    print ("[STATUS] Creating the classifier..")
    # clf_svm = SVC(C=0.065 ,gamma=0.5,class_weight='balanced')
    clf_svm = SVC(C=1, gamma=0.5, class_weight={0:1,1:2})
    # fit the training data and labels
    print ("[STATUS] Fitting data/label to model..")
    clf_svm.fit(XTrain, ColorLabel)
    prediction = clf_svm.predict(ColorTest)
    print("Accuracy for SVM on CV data: ", accuracy_score(ColorLabelTest, prediction))



# #TODO:https://bigdata-madesimple.com/dealing-with-unbalanced-class-svm-random-forest-and-decision-tree-in-python/

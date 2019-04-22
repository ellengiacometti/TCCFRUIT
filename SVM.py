import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler



if __name__ == '__main__':

    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train1200.csv', index_col=False, sep=";", converters={'Color': teste,'ColorS': teste,'ColorV': teste})
    test = pd.read_csv('Test1200.csv', index_col=False, sep=";", converters={'Color': teste,'ColorS': teste,'ColorV': teste})
    le = LabelEncoder()

    ColorTrainH = [list(map(float, histH)) for histH in train['Color']]
    ColorTrainH = np.array(ColorTrainH)
    # ColorTrainS = [list(map(float, histS)) for histS in train['ColorS']]
    # ColorTrainS = np.array(ColorTrainS)
    # ColorTrainV = [list(map(float, histV)) for histV in train['ColorV']]
    # ColorTrainV = np.array(ColorTrainV)
    colunasTrain = train.columns[1:9]
    TextureTrain=train[colunasTrain].values
    # colunaTrainColor = np.hstack((ColorTrainH, ColorTrainS, ColorTrainV))
    colunaTrainColor = ColorTrainH
    FeaturesTrain = np.hstack((TextureTrain,colunaTrainColor))
    TextureLabelTrain = train['TextureLabel']
    le.fit(TextureLabelTrain)
    TextureLabelTrain = le.transform(TextureLabelTrain)
    ColorLabelTrain =train['ColorLabel']
    le.fit(ColorLabelTrain)
    ColorLabelTrain = le.transform(ColorLabelTrain)



    ColorTestH = [list(map(float, histH)) for histH in test['Color']]
    ColorTestH = np.array(ColorTestH)
    # ColorTestS = [list(map(float, histS)) for histS in test['ColorS']]
    # ColorTestS = np.array(ColorTestS)
    # ColorTestV = [list(map(float, histV)) for histV in test['ColorV']]
    # ColorTestV = np.array(ColorTestV)
    colunasTest = test.columns[1:9]
    TextureTest = test[colunasTest].values
    colunaTestColor = ColorTestH
    # colunaTestColor = np.hstack((ColorTestH, ColorTestS, ColorTestV))
    FeaturesTest = np.hstack((TextureTest, colunaTestColor))
    TextureLabelTest= test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test ['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)

    # """ SVM """
    #  # create the SVM classifier
    # svm = SVC()
    #
    # parameters = {'C': (1, 0.25, 0.5, 0.75, 0.05, 0.001, 0.01, 0.1, 100, 10, 1000), 'gamma': (0.5,1,2,3,'auto'),'kernel':('poly','rbf'),'random_state':[int(x) for x in np.linspace(start=0, stop=1000, num=10)]}
    # # parameters = {'C': (1, 0.25, 0.5, 0.75, 0.05, 0.001, 0.01, 0.1, 100, 10, 1000), 'gamma': (0.5, 1, 2, 3, 'auto'),
    # #               'kernel': ('linear', 'rbf', 'poly')}
    # #'class_weight': [{0: 1, 1: w2} for w2 in [1, 2, 4, 6, 10, 12]]
    # clf = GridSearchCV(svm, parameters,verbose = 100)
    # clf.fit(FeaturesTrain, ColorLabelTrain)
    # print(clf.best_params_)



    print("~~~ INFO BASE TESTE ~~~")
    print("NÚMERO RUGOSOS:", sum(TextureLabelTest))
    print("NÚMERO LISOS:",( len(TextureLabelTest) - sum(TextureLabelTest)))
    print("NÚMERO SEM DEFEITO:", sum(ColorLabelTest))
    print("NÚMERO COM DEFEITO:", (len(ColorLabelTest) - sum(ColorLabelTest)))
    print("~~~ INFO BASE TREINO ~~~")
    print("NÚMERO RUGOSOS:", sum(TextureLabelTrain))
    print("NÚMERO LISOS:", (len(TextureLabelTrain) - sum(TextureLabelTrain)))
    print("NÚMERO SEM DEFEITO:", sum(ColorLabelTrain))
    print("NÚMERO COM DEFEITO:", len(ColorLabelTrain) - sum(ColorLabelTrain))

    """CLASSIFICADOR LISO X RUGOSO"""
    print("\n~~~ SVM -  CLASSIFICADOR LISO X RUGOSO ~~~")
    #print("[STATUS] Creating the classifier..")

    clf_svmLR = SVC(C=1, gamma=0.5,kernel='poly',random_state= 0)
    # fit the training data and labels
    #print ("[STATUS] Fitting data/label to model..")
    clf_svmLR.fit(FeaturesTrain, TextureLabelTrain)

    #print("[STATUS] Predicting TRAINED DataBase..")
    predictionTrain = clf_svmLR.predict(FeaturesTrain)
    print("Accuracy for SVM on TRAINED data: ", accuracy_score(TextureLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTrain, predictionTrain))

    #print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_svmLR.predict(FeaturesTest)
    print("Accuracy for SVM on TEST data: ", accuracy_score(TextureLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, predictionTest))

    """CLASSIFICADOR COM DEFEITO X SEM DEFEITO"""
    print("\n~~~ SVM - CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    #print("[STATUS] Creating the classifier..")

    clf_svmCS = SVC(C=1, gamma=0.5,kernel='poly',random_state= 0)
    # clf_svmCS = SVC(C=1, gamma=0.5, class_weight='balanced', decision_function_shape='ovo', kernel='poly')
    # fit the training data and labels
   # print ("[STATUS] Fitting data/label to model..")
    clf_svmCS.fit(FeaturesTrain, ColorLabelTrain)

    #print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_svmCS.predict(FeaturesTrain)
    print("Accuracy for SVM on TRAINED data: ", accuracy_score(ColorLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTrain, predictionTrain))

   # print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_svmCS.predict(FeaturesTest)
    print("Accuracy for SVM on TEST data: ", accuracy_score(ColorLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, predictionTest))
   #


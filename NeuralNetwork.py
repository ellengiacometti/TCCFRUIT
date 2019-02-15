from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train.csv', index_col=False, sep=";", converters={'Color': teste})
    test = pd.read_csv('Test.csv', index_col=False, sep=";", converters={'Color': teste})
    le = LabelEncoder()

    ColorTrain = [list(map(float, hist)) for hist in train['Color']]
    ColorTrain = np.array(ColorTrain)
    colunasTrain = train.columns[1:3]
    TextureTrain = train[colunasTrain].values
    FeaturesTrain = np.hstack((TextureTrain, ColorTrain))

    TextureLabelTrain = train['TextureLabel']
    le.fit(TextureLabelTrain)
    TextureLabelTrain = le.transform(TextureLabelTrain)
    ColorLabelTrain = train['ColorLabel']
    le.fit(ColorLabelTrain)
    ColorLabelTrain = le.transform(ColorLabelTrain)

    ColorTest = [list(map(float, hist)) for hist in test['Color']]
    ColorTest = np.array(ColorTest)
    colunasTest = test.columns[1:3]
    TextureTest = test[colunasTest].values
    FeaturesTest = np.hstack((TextureTest, ColorTest))

    TextureLabelTest = test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)

    print("\n~~~ SVM - CLASSIFICADOR LISO X RUGOSO~~~")
    clf_nnLR = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf_nnLR.fit(FeaturesTrain, TextureLabelTrain)
    # print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_nnLR.predict(FeaturesTrain)
    print("Accuracy for Neural Network on TRAINED data: ", accuracy_score(TextureLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTrain, predictionTrain))

    # print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_nnLR.predict(FeaturesTest)
    print("Accuracy for Neural Network on Test data: ", accuracy_score(TextureLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, predictionTest))


    print("\n~~~ SVM - CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    clf_nnCS = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf_nnCS.fit(FeaturesTrain, ColorLabelTrain)
    # print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_nnCS.predict(FeaturesTrain)
    print("Accuracy for Neural Network on TRAINED data: ", accuracy_score(ColorLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTrain, predictionTrain))

    # print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_nnCS.predict(FeaturesTest)
    print("Accuracy for Neural Network on Test data: ", accuracy_score(ColorLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, predictionTest))
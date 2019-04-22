from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train1200.csv', index_col=False, sep=";",
                        converters={'Color': teste, 'ColorS': teste, 'ColorV': teste})
    test = pd.read_csv('Test1200.csv', index_col=False, sep=";",
                       converters={'Color': teste, 'ColorS': teste, 'ColorV': teste})
    le = LabelEncoder()

    ColorTrainH = [list(map(float, histH)) for histH in train['Color']]
    ColorTrainH = np.array(ColorTrainH)
    # ColorTrainS = [list(map(float, histS)) for histS in train['ColorS']]
    # ColorTrainS = np.array(ColorTrainS)
    # ColorTrainV = [list(map(float, histV)) for histV in train['ColorV']]
    # ColorTrainV = np.array(ColorTrainV)
    colunasTrain = train.columns[1:9]
    TextureTrain = train[colunasTrain].values
    colunaTrainColor=ColorTrainH
    # colunaTrainColor = np.hstack((ColorTrainH, ColorTrainS, ColorTrainV))
    FeaturesTrain = np.hstack((TextureTrain, colunaTrainColor))
    TextureLabelTrain = train['TextureLabel']
    le.fit(TextureLabelTrain)
    TextureLabelTrain = le.transform(TextureLabelTrain)
    ColorLabelTrain = train['ColorLabel']
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
    colunaTestColor=ColorTestH
    # colunaTestColor = np.hstack((ColorTestH, ColorTestS, ColorTestV))
    FeaturesTest = np.hstack((TextureTest, colunaTestColor))
    TextureLabelTest = test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)

    # """Grid Search"""
    # parameter_space = {
    #     'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (5, 2),],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam','lbfgs'],
    #     'alpha': [0.0001, 0.05,1e-5,1e-6,1],
    #     'learning_rate': ['constant', 'adaptive'],
    #     'random_state':[1,5,10,15,20,30,40,50],
    # }
    # mlp = MLPClassifier(solver='lbfgs')
    # clf = GridSearchCV(mlp, parameter_space, verbose=200,n_jobs=-1, cv=3)
    # clf.fit(FeaturesTrain, ColorLabelTrain)
    # clf.best_params_
    # print(clf.best_params_)

      # {'alpha': 0.0001, 'learning_rate': 'constant', 'activation': 'relu', 'hidden_layer_sizes': (50, 50, 50), 'random_state': 40, 'solver': 'lbfgs'}

    print("\n~~~ REDE NEURAL  - CLASSIFICADOR LISO X RUGOSO~~~")

    # {'random_state': 50, 'learning_rate': 'constant', 'solver': 'adam', 'hidden_layer_sizes': (50, 100, 50),
    #  'activation': 'relu', 'alpha': 1e-06}
    clf_nnLR = MLPClassifier(activation= 'relu', hidden_layer_sizes= (50, 100, 50), alpha= 1e-06, learning_rate= 'constant', solver = 'adam', random_state= 50)
    clf_nnLR.fit(FeaturesTrain, TextureLabelTrain)
    # print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_nnLR.predict(FeaturesTrain)
    print("Accuracy for Neural Network on TRAINED data: ", accuracy_score(TextureLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTrain, predictionTrain))

    # print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_nnLR.predict(FeaturesTest)
    print("Accuracy for Neural Network on TEST data: ", accuracy_score(TextureLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, predictionTest))

    print("\n~~~ REDE NEURAL - CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    #activation=relu, learning_rate=adaptive, hidden_layer_sizes=(50, 50, 50), alpha=0.05, random_state=50, solver=adam,(83%)
    #activation=relu, learning_rate=constant, hidden_layer_sizes=(50, 100, 50), alpha=0.05, random_state=15, solver=lbfgs(84%)
    # (solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state = 1)
    # {'hidden_layer_sizes': (50, 100, 50), 'solver': 'lbfgs', 'alpha': 0.05, 'random_state': 40, 'activation': 'relu', 'learning_rate': 'constant'}
    clf_nnCS = MLPClassifier(activation='relu', learning_rate='adaptive', hidden_layer_sizes=(50, 50, 50), alpha=0.05,
                             random_state=50, solver='adam')
    clf_nnCS.fit(FeaturesTrain, ColorLabelTrain)
    # print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_nnCS.predict(FeaturesTrain)
    print("Accuracy for Neural Network on TRAINED data: ", accuracy_score(ColorLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTrain, predictionTrain))

    # print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_nnCS.predict(FeaturesTest)
    print("Accuracy for Neural Network on TEST data: ", accuracy_score(ColorLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, predictionTest))
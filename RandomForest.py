import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('TrainAR1200.csv', index_col=False, sep=";",
                        converters={'ColorH': teste,'ColorS': teste,'ColorV': teste})
    test = pd.read_csv('TestAR1200.csv', index_col=False, sep=";",
                       converters={'ColorH': teste,'ColorS': teste,'ColorV': teste})
    le = LabelEncoder()
    ColorTrainH = [list(map(float, histH)) for histH in train['ColorH']]
    ColorTrainH = np.array(ColorTrainH)
    ColorTrainS = [list(map(float, histS)) for histS in train['ColorS']]
    ColorTrainS = np.array(ColorTrainS)
    ColorTrainV = [list(map(float, histV)) for histV in train['ColorV']]
    ColorTrainV = np.array(ColorTrainV)
    colunasTrain = train.columns[1:9]
    TextureTrain = train[colunasTrain].values

    colunaTrainColor = np.hstack((ColorTrainH, ColorTrainS, ColorTrainV))
    # colunaTrainColor =  ColorTrainH
    FeaturesTrain = np.hstack((TextureTrain, colunaTrainColor))
    TextureLabelTrain = train['TextureLabel']
    le.fit(TextureLabelTrain)
    TextureLabelTrain = le.transform(TextureLabelTrain)
    ColorLabelTrain = train['ColorLabel']
    le.fit(ColorLabelTrain)
    ColorLabelTrain = le.transform(ColorLabelTrain)

    ColorTestH = [list(map(float, histH)) for histH in test['ColorH']]
    ColorTestH = np.array(ColorTestH)
    ColorTestS = [list(map(float, histS)) for histS in test['ColorS']]
    ColorTestS = np.array(ColorTestS)
    ColorTestV = [list(map(float, histV)) for histV in test['ColorV']]
    ColorTestV = np.array(ColorTestV)
    colunasTest = test.columns[1:9]
    TextureTest = test[colunasTest].values
    colunaTestColor = np.hstack((ColorTestH, ColorTestS, ColorTestV))
    # colunaTestColor =ColorTestH
    FeaturesTest = np.hstack((TextureTest, colunaTestColor))
    TextureLabelTest = test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)
    #Create a based model
    rf = RandomForestClassifier(random_state=42)
    # Create the parameter grid based on the results of random search

    # param_grid = {
    #     # Number of trees in random forest
    #     'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=10)],
    #      # Number of features to consider at every split
    #     'max_features': ['auto', 'sqrt'],
    #      # Maximum number of levels in tree
    #     'max_depth': [int(x) for x in np.linspace(10, 30, num=11)],
    #      # Minimum number of samples required to split a node
    #     'min_samples_split': [2, 5, 10],
    #      # Minimum number of samples required at each leaf node
    #     'min_samples_leaf': [1, 2, 4],
    #     # Method of selecting samples for training each tree
    #     'bootstrap': [True, False]
    # }
    # # Create a based model
    # rf = RandomForestClassifier()
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
    #                            cv=3, n_jobs=-1, verbose=60)
    # # }
    # # # Instantiate the grid search model
    # CV_rfc = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 5, n_jobs = -1, verbose = 60)
    # # Fit the grid search to the data
    # CV_rfc.fit(FeaturesTrain,TextureLabelTrain)
    # CV_rfc.best_params_
    # print(CV_rfc.best_params_)
# ##ACHADO DURANTE O PROCESSO
# max_depth=18, min_samples_leaf=1, min_samples_split=2, bootstrap=True, max_features=sqrt, n_estimators=20(97%)
    print("~~~ INFO BASE TESTE ~~~")
    print("NÚMERO RUGOSOS:", sum(TextureLabelTest))
    print("NÚMERO LISOS:", len(TextureLabelTest) - sum(TextureLabelTest))
    print("NÚMERO SEM DEFEITO:", sum(ColorLabelTest))
    print("NÚMERO COM DEFEITO:", len(ColorLabelTest) - sum(ColorLabelTest))
    print("~~~ INFO BASE TREINO ~~~")
    print("NÚMERO RUGOSOS:", sum(TextureLabelTrain))
    print("NÚMERO LISOS:", len(TextureLabelTrain) - sum(TextureLabelTrain))
    print("NÚMERO SEM DEFEITO:", sum(ColorLabelTrain))
    print("NÚMERO COM DEFEITO:", len(ColorLabelTrain) - sum(ColorLabelTrain))

    print("\n~~~ RANDOM FOREST -  CLASSIFICADOR LISO X RUGOSO ~~~")
    #print("[STATUS] Creating the classifier..")
    # {'min_samples_leaf': 1, 'min_samples_split': 2, 'max_features': 'sqrt', 'n_estimators': 100, 'max_depth': 24, 'bootstrap': False}
    clf_rfLR=RandomForestClassifier(max_depth=24, min_samples_leaf=1, min_samples_split=2, bootstrap=False, max_features='sqrt', n_estimators=100)
    #print("[STATUS] Fitting data/label to model..")
    clf_rfLR.fit(FeaturesTrain, TextureLabelTrain)
    #print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_rfLR.predict(FeaturesTrain)
    print("Accuracy for Random Foreston TRAIN data: ", accuracy_score(TextureLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTrain, predictionTrain))
    pred=clf_rfLR.predict(FeaturesTest)
    print("Accuracy for Random Forest on TEST data: ",accuracy_score(TextureLabelTest,pred))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, pred))

    print("\n~~~ RANDOM FOREST -  CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    #{'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 80, 'min_samples_split': 2, 'max_depth': 12, 'bootstrap': False}

    clf_rfCS = RandomForestClassifier(max_depth=12, min_samples_leaf=1, min_samples_split=2, bootstrap=False, max_features='auto', n_estimators=80)
    #print("[STATUS] Fitting data/label to model..")
    clf_rfCS.fit(FeaturesTrain, ColorLabelTrain)
    #print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_rfCS.predict(FeaturesTrain)
    print("Accuracy for Random Forest on TRAINED data: ", accuracy_score(ColorLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTrain, predictionTrain))
    pred = clf_rfCS.predict(FeaturesTest)
    print("Accuracy for Random Forest on TEST data: ", accuracy_score(ColorLabelTest, pred))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, pred))


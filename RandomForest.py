import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('RFNormLR.csv', index_col=False, sep=";",
                        converters={'ColorH': teste, 'ColorS': teste, 'ColorV': teste})
    test = pd.read_csv('NormTest160.csv', index_col=False, sep=";",
                       converters={'ColorH': teste, 'ColorS': teste, 'ColorV': teste})
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
    FeaturesTest = np.hstack((TextureTest, colunaTestColor))
    TextureLabelTest = test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)

    # # #Create a based model
    # rf = RandomForestClassifier(random_state=42)
    # # Create the parameter grid based on the results of random search
    # param_grid = {
    #     'n_estimators': [10, 20, 30, 40,50,60,70,80,90,100],
    #     'class_weight': [{0: 1,1: w2} for w2 in [2, 4, 6, 10,12]]
    #
#     param_grid = {
#         # Number of trees in random forest
#         'n_estimators': [int(x) for x in np.linspace(start=20, stop=200, num=10)],
#          # Number of features to consider at every split
#         'max_features': ['auto', 'sqrt'],
#          # Maximum number of levels in tree
#         'max_depth': [int(x) for x in np.linspace(10, 30, num=11)],
#          # Minimum number of samples required to split a node
#         'min_samples_split': [2, 5, 10],
#          # Minimum number of samples required at each leaf node
#         'min_samples_leaf': [1, 2, 4],
#         # Method of selecting samples for training each tree
#         'bootstrap': [True, False]
#     }
#     # # Create a based model
#     rf = RandomForestClassifier()
#     # Instantiate the grid search model
#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                                cv=3, n_jobs=-1, verbose=100)
#
#     # # Instantiate the grid search model
#     CV_rfc = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 5, n_jobs = -1, verbose = 60)
#     # Fit the grid search to the data
#     CV_rfc.fit(FeaturesTrain,ColorLabelTrain)
#     CV_rfc.best_params_
#     print(CV_rfc.best_params_)
# #

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
    clf_rfLR=RandomForestClassifier(n_estimators= 100, min_samples_split= 2, max_features= 'auto', max_depth= 24, min_samples_leaf=1, bootstrap= False)
    #print("[STATUS] Fitting data/label to model..")
    clf_rfLR.fit(FeaturesTrain, TextureLabelTrain)
    #print("[STATUS] Predicting Trained DataBase..")
    pred=clf_rfLR.predict(FeaturesTest)
    print("Accuracy for Random Forest on TEST data: ",accuracy_score(TextureLabelTest,pred))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, pred))

    print("\n~~~ RANDOM FOREST -  CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    clf_rfCS = RandomForestClassifier(bootstrap= False, max_features='sqrt', n_estimators= 200, min_samples_split= 5, max_depth=16, min_samples_leaf=2)
    #print("[STATUS] Fitting data/label to model..")
    clf_rfCS.fit(FeaturesTrain, ColorLabelTrain)
    #print("[STATUS] Predicting Trained DataBase..")
    pred = clf_rfCS.predict(FeaturesTest)
    print("Accuracy for Random Forest on TEST data: ", accuracy_score(ColorLabelTest, pred))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, pred))


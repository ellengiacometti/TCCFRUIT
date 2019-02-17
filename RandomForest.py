import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    colunasTrain = train.columns[1:9]
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
    colunasTest = test.columns[1:9]
    TextureTest = test[colunasTest].values
    FeaturesTest = np.hstack((TextureTest, ColorTest))

    TextureLabelTest = test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)

    # #Create a based model
    # rf = RandomForestClassifier(random_state=42)
    # # Create the parameter grid based on the results of random search
    # param_grid = {
    #     'n_estimators': [10, 20, 30, 40,50,60,70,80,90,100],
    #     'class_weight': [{0: 1,1: w2} for w2 in [2, 4, 6, 10,12]]
    #
    param_grid = {
        # Number of trees in random forest
        'n_estimators': [int(x) for x in np.linspace(start=20, stop=2000, num=10)],
         # Number of features to consider at every split
        'max_features': ['auto', 'sqrt'],
         # Maximum number of levels in tree
        'max_depth': [int(x) for x in np.linspace(10, 60, num=11)],
         # Minimum number of samples required to split a node
        'min_samples_split': [2, 5, 10,15],
         # Minimum number of samples required at each leaf node
        'min_samples_leaf': [1, 2, 4,6],
        # Method of selecting samples for training each tree
        'bootstrap': [True, False]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    # }
    # # Instantiate the grid search model
    CV_rfc = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 5, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    CV_rfc.fit(FeaturesTrain,ColorLabelTrain)
    CV_rfc.best_params_
    print(CV_rfc.best_params_)
    #
    # print("~~~ INFO BASE TESTE ~~~")
    # print("NÚMERO RUGOSOS:", sum(TextureLabelTest))
    # print("NÚMERO LISOS:", len(TextureLabelTest) - sum(TextureLabelTest))
    # print("NÚMERO SEM DEFEITO:", sum(ColorLabelTest))
    # print("NÚMERO COM DEFEITO:", len(ColorLabelTest) - sum(ColorLabelTest))
    # print("~~~ INFO BASE TREINO ~~~")
    # print("NÚMERO RUGOSOS:", sum(TextureLabelTrain))
    # print("NÚMERO LISOS:", len(TextureLabelTrain) - sum(TextureLabelTrain))
    # print("NÚMERO SEM DEFEITO:", sum(ColorLabelTrain))
    # print("NÚMERO COM DEFEITO:", len(ColorLabelTrain) - sum(ColorLabelTrain))
    #
    # print("\n~~~ RANDOM FOREST -  CLASSIFICADOR LISO X RUGOSO ~~~")
    # #print("[STATUS] Creating the classifier..")
    # clf_rfLR=RandomForestClassifier(class_weight={0:1,1:2}, n_estimators= 40)
    # #print("[STATUS] Fitting data/label to model..")
    # clf_rfLR.fit(FeaturesTrain, TextureLabelTrain)
    # #print("[STATUS] Predicting Trained DataBase..")
    # pred=clf_rfLR.predict(FeaturesTest)
    # print("Accuracy for Random Forest on TEST data: ",accuracy_score(TextureLabelTest,pred))
    # print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, pred))
    #
    # print("\n~~~ RANDOM FOREST -  CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    # #print("[STATUS] Creating the classifier..")
    # clf_rfCS = RandomForestClassifier(class_weight={0: 1, 1: 2}, n_estimators=40)
    # #print("[STATUS] Fitting data/label to model..")
    # clf_rfCS.fit(FeaturesTrain, ColorLabelTrain)
    # #print("[STATUS] Predicting Trained DataBase..")
    # pred = clf_rfCS.predict(FeaturesTest)
    # print("Accuracy for Random Forest on TEST data: ", accuracy_score(ColorLabelTest, pred))
    # print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, pred))

# # #TODO https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
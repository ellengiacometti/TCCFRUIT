import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import  load




if __name__ == '__main__':

    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train1200.csv', index_col=False, sep=";", converters={'Color': teste,'ColorS': teste,'ColorV': teste})
    test = pd.read_csv('Test1200.csv', index_col=False, sep=";", converters={'Color': teste,'ColorS': teste,'ColorV': teste})
    le = LabelEncoder()

### IMPORTANDO DADOS DE TREINO COM PANDA
    ColorTrainH = [list(map(float, histH)) for histH in train['Color']]
    ColorTrainH = np.array(ColorTrainH)
    colunasTrain = train.columns[1:9]
    TextureTrain=train[colunasTrain].values
    colunaTrainColor = ColorTrainH
    FeaturesTrain = np.hstack((TextureTrain,colunaTrainColor))
    TextureLabelTrain = train['TextureLabel']
    le.fit(TextureLabelTrain)
    TextureLabelTrain = le.transform(TextureLabelTrain)
    ColorLabelTrain =train['ColorLabel']
    le.fit(ColorLabelTrain)
    ColorLabelTrain = le.transform(ColorLabelTrain)

### IMPORTANDO DADOS DE TESTE COM PANDA
    ColorTestH = [list(map(float, histH)) for histH in test['Color']]
    ColorTestH = np.array(ColorTestH)
    colunasTest = test.columns[1:9]
    TextureTest = test[colunasTest].values
    colunaTestColor = ColorTestH
    FeaturesTest = np.hstack((TextureTest, colunaTestColor))
    TextureLabelTest= test['TextureLabel']
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    ColorLabelTest = test ['ColorLabel']
    le.fit(ColorLabelTest)
    ColorLabelTest = le.transform(ColorLabelTest)



### PRINTANDO INFO DOS DATASETS
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
    clf_svmLR = load('ModelSVM_LR.joblib')
    print("[STATUS] Predicting TRAIN DataBase..")
    predictionTrain = clf_svmLR.predict(FeaturesTrain)
    print("Accuracy for SVM on TRAINED data: ", accuracy_score(TextureLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTrain, predictionTrain))

    print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_svmLR.predict(FeaturesTest)
    print("Accuracy for SVM on TEST data: ", accuracy_score(TextureLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(TextureLabelTest, predictionTest))

    """CLASSIFICADOR COM DEFEITO X SEM DEFEITO"""

    print("\n~~~ SVM - CLASSIFICADOR COM DEFEITO X SEM DEFEITO ~~~")
    clf_svmCS = load('ModelSVM_CS.joblib')
    print("[STATUS] Predicting Trained DataBase..")
    predictionTrain = clf_svmCS.predict(FeaturesTrain)
    print("Accuracy for SVM on TRAINED data: ", accuracy_score(ColorLabelTrain, predictionTrain))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTrain, predictionTrain))

    print("[STATUS] Predicting TEST DataBase..")
    predictionTest = clf_svmCS.predict(FeaturesTest)
    print("Accuracy for SVM on TEST data: ", accuracy_score(ColorLabelTest, predictionTest))
    print("Confusion Matrix: ", confusion_matrix(ColorLabelTest, predictionTest))



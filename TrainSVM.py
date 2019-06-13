import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from joblib import dump




if __name__ == '__main__':

### TRATAMENTO DADOS PLANILHA CSV
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train1200.csv', index_col=False, sep=";", converters={'Color': teste,'ColorS': teste,'ColorV': teste})
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

### PARAMETROS PARA O GRIDSEARCH
    """ SVM """
    svm = SVC()
    parameters = {'C': (1, 0.25, 0.5, 0.75, 0.05, 0.001, 0.01, 0.1, 100, 10, 1000, 5, 50, 500),
                  'gamma': (0.1, 0.5, 1, 3, 5, 7, 10, 100, 'auto'), 'kernel': ('poly', 'rbf')}

### SALVANDO MODELO CLASSIFICADOR LISO E RUGOSO
    clfLR = GridSearchCV(svm, parameters,verbose = 100)
    clfLR.fit(FeaturesTrain, TextureLabelTrain)
    print(clfLR.best_params_)
    dump(clfLR, 'ModelSVM_LR.joblib')

### SALVANDO MODELO CLASSIFICADOR COM E SEM DEFEITO
    clfCS = GridSearchCV(svm, parameters,verbose = 100)
    clfCS.fit(FeaturesTrain, ColorLabelTrain)
    print(clfCS.best_params_)
    dump(clfCS, 'ModelSVM_CS.joblib')


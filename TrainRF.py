import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from joblib import dump



if __name__ == '__main__':
### TRATAMENTO DADOS PLANILHA CSV
    teste = lambda x: x.strip("[]").replace("'", "").split(", ")
    train = pd.read_csv('Train1200.csv', index_col=False, sep=";",
                        converters={'Color': teste, 'ColorS': teste, 'ColorV': teste})
    le = LabelEncoder()

### IMPORTANDO DADOS DE TREINO COM PANDA
    ColorTrainH = [list(map(float, histH)) for histH in train['Color']]
    ColorTrainH = np.array(ColorTrainH)
    colunasTrain = train.columns[1:9]
    TextureTrain = train[colunasTrain].values
    colunaTrainColor = ColorTrainH
    FeaturesTrain = np.hstack((TextureTrain, colunaTrainColor))
    TextureLabelTrain = train['TextureLabel']
    le.fit(TextureLabelTrain)
    TextureLabelTrain = le.transform(TextureLabelTrain)
    ColorLabelTrain = train['ColorLabel']
    le.fit(ColorLabelTrain)
    ColorLabelTrain = le.transform(ColorLabelTrain)

### PARAMETROS PARA O GRIDSEARCH
    ''''RANDOM FOREST'''
    rf = RandomForestClassifier( )
    param_grid = {
        # Number of trees in random forest
        'n_estimators': [int(x) for x in np.linspace(start=1, stop=200, num=100)]
        # ,
        # # Number of features to consider at every split
        # 'max_features': ['auto', 'log2', 'none'],
        # # Maximum number of levels in tree
        # 'max_depth': [int(x) for x in np.linspace(1, 50, num=2)],
        # # Minimum number of samples required to split a node
        # 'min_samples_split': [2, 4, 5, 6, 10, 15],
        # # Minimum number of samples required at each leaf node
        # 'min_samples_leaf': [ 2, 4, 5, 6, 10, 15],
        # # Method of selecting samples for training each tree
        # 'bootstrap': [True, False]
    }
### SALVANDO MODELO CLASSIFICADOR LISO E RUGOSO
    clfLR = GridSearchCV(rf, param_grid,n_jobs=-1,cv=3,verbose = 100)
    clfLR.fit(FeaturesTrain, TextureLabelTrain)
    print(clfLR.best_params_)
    dump(clfLR, 'ModelRF_LR.joblib')

### SALVANDO MODELO CLASSIFICADOR COM E SEM DEFEITO
    clfCS = GridSearchCV(rf, param_grid,n_jobs=-1, verbose=100,cv=3)
    clfCS.fit(FeaturesTrain, ColorLabelTrain)
    print(clfCS.best_params_)
    dump(clfCS, 'ModelRF_CS.joblib')
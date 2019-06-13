from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
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
    '''NEURAL NETWORK'''
    mlp = MLPClassifier()
    parameter_space = { 'hidden_layer_sizes': [(1,), (2,), (5,), (10,), (15,), (25,), (50,), (100,),(150,),(200,)]
                        # ,'activation': ['tanh', 'relu'],
                        # 'solver': ['sgd', 'adam','lbfgs'],
                        # 'alpha': [0.0001, 0.001, 0.01, 0.1, 1e-5,1e-6,1],
                        # 'learning_rate': ['constant', 'adaptive']
    }
### SALVANDO MODELO CLASSIFICADOR LISO E RUGOSO
    clfLR = GridSearchCV(mlp, parameter_space, verbose=200,n_jobs=-1, cv=3)
    clfLR.fit(FeaturesTrain, TextureLabelTrain)
    print(clfLR.best_params_)
    dump(clfLR, 'ModelNN_LR.joblib')

### SALVANDO MODELO CLASSIFICADOR COM E SEM DEFEITO
    clfCS = GridSearchCV(mlp, parameter_space, verbose=200,n_jobs=-1, cv=3)
    clfCS.fit(FeaturesTrain, ColorLabelTrain)
    print(clfCS.best_params_)
    dump(clfCS, 'ModelNN_CS.joblib')
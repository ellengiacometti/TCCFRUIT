from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

Names=['TrainArtigo.csv','TestArtigo.csv']
def create_columns(n):
    list_of_columns = []
    for i in range(n):
        list_of_columns.append(0)
    return list_of_columns
for name in Names:
    prop = lambda x: x.strip("[]").replace("'", "").split(", ")
    # Data = pd.read_csv(name, index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
    Data = pd.read_csv(name, index_col=False, sep=";",converters={'ColorH': prop})
    colunas = Data.columns[1:9]
    histColumns = Data.columns[9:12]
    Color = create_columns(1)
    for a in range(0,histColumns.size):
        Color[a] = [list(map(float, histH)) for histH in Data[histColumns[a]]]
        Color[a] = np.array(Color[a])
        a += 1
    colunaColor = Color[0]
    for b in range(1, histColumns.size):
        colunaColor = np.hstack((colunaColor, Color[b]))

    if name=='Train.csv':
        TextureTrain = Data[colunas].values
        FeaturesTrain = np.hstack((TextureTrain, colunaColor))
        le = LabelEncoder()
        TextureLabelTrain = Data['TextureLabel']
        le.fit(TextureLabelTrain)
        TextureLabelTrain = le.transform(TextureLabelTrain)
        ColorLabelTrain = Data['ColorLabel']
        le.fit(ColorLabelTrain)
        ColorLabelTrain = le.transform(ColorLabelTrain)
    else:
        TextureTest = Data[colunas].values
        FeaturesTest = np.hstack((TextureTest, colunaColor))
        le = LabelEncoder()
        TextureLabelTest = Data['TextureLabel']
        le.fit(TextureLabelTest)
        TextureLabelTest = le.transform(TextureLabelTest)
        ColorLabelTest = Data['ColorLabel']
        le.fit(ColorLabelTest)
        ColorLabelTest = le.transform(ColorLabelTest)


## Random Forest
# Training LR Classifier
clf_rfLR = RandomForestClassifier(max_depth=18, min_samples_leaf=1, min_samples_split=2, bootstrap=True, max_features='sqrt', n_estimators=20)

clf_rfLR.fit(FeaturesTrain, TextureLabelTrain)
predRFLR = clf_rfLR.predict(FeaturesTest)
accuracyRFLR= accuracy_score(TextureLabelTest, predRFLR)

# Training CS Classifier
clf_rfCS=RandomForestClassifier(max_depth=18, min_samples_leaf=1, min_samples_split=2, bootstrap=True, max_features='sqrt', n_estimators=20)
clf_rfCS.fit(FeaturesTrain, ColorLabelTrain)
predRFCS = clf_rfCS.predict(FeaturesTest)
accuracyRFCS = accuracy_score(ColorLabelTest, predRFCS)
## Neural Network
# Training LR Classifier
clf_nnLR = MLPClassifier(activation= 'relu', hidden_layer_sizes= (50, 50, 50), alpha= 0.0001, learning_rate= 'constant', solver = 'lbfgs', random_state= 40)
clf_nnLR.fit(FeaturesTrain, TextureLabelTrain)
predNNLR = clf_nnLR.predict(FeaturesTest)
accuracyNNLR= accuracy_score(TextureLabelTest, predNNLR)

# Training CS Classifier
clf_nnCS = MLPClassifier(learning_rate= 'constant', solver= 'adam', activation='relu', random_state= 50, hidden_layer_sizes= (50, 50, 50), alpha= 0.05)
clf_nnCS.fit(FeaturesTrain, ColorLabelTrain)
predNNCS = clf_nnCS.predict(FeaturesTest)
accuracyNNCS = accuracy_score(TextureLabelTest, predNNCS)


## SVM
# Training LR Classifier
clf_svmLR = SVC(C=1, gamma=0.5, decision_function_shape='ovo', kernel='poly')
clf_svmLR.fit(FeaturesTrain, TextureLabelTrain)
predSVMLR = clf_svmLR.predict(FeaturesTest)
accuracySVMLR = accuracy_score(TextureLabelTest, predSVMLR)
# Training CS Classifier
clf_svmCS = SVC(C=1, gamma=0.5, decision_function_shape='ovo', kernel='poly')
clf_svmCS.fit(FeaturesTrain, ColorLabelTrain)
predSVMCS = clf_svmCS.predict(FeaturesTest)
accuracySVMCS = accuracy_score(TextureLabelTest, predSVMCS)

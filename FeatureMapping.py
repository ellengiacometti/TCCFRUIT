"""Author: Ellen Giacometti
    CRIADO EM: 05/04/2019
    ÚLTIMA ATUALIZAÇÃO: 05/02/2019
    DESC: Ler a planilha de dados normalizados e chamar os classificadores removendo feature por feature
    e por fim gerar uma planinha com um feature mapping """

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def create_columns(n):
    list_of_columns = []
    for i in range(n):
        list_of_columns.append(0)
    return list_of_columns

paramIn=12
a=0
## Reading CSV
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
Norm = pd.read_csv('normBTrain.csv', index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
## Manipulating Columns Index to exclude a feature which index number is paramIn

if paramIn<=9:
    featInt = Norm.columns[1:paramIn]
    featInt2 = Norm.columns[(paramIn+1):9]
    colunas=featInt.union(featInt2,sort=False)
    histColumns = Norm.columns[9:]
else:
    colunas=Norm.columns[1:9]
    featHists = Norm.columns[9:paramIn]
    featHists2 = Norm.columns[paramIn+1:12]
    histColumns= featHists.union(featHists2,sort=False)
    if paramIn != 12:
        Color = create_columns(2)
    else:
        Color = create_columns(3)

for histColumn in histColumns:
    Color[a] = [list(map(float, histH)) for histH in Norm[histColumn]]
    Color[a] = np.array(Color[a])
    a+=1
colunaColor= Color[0]
for b in range(1,a):
    colunaColor= np.hstack((colunaColor, Color[b]))

Texture= Norm[colunas].values
Features= np.hstack((Texture, colunaColor))
le = LabelEncoder()
TextureLabel = Norm['TextureLabel']
le.fit(TextureLabel)
TextureLabel = le.transform(TextureLabel)
ColorLabel= Norm['ColorLabel']
le.fit(ColorLabel)
ColorLabel = le.transform(ColorLabel)
sizeFeatures=Features.shape[0]
sizeValidation = round((sizeFeatures)*0.25)
FeaturesTrain=Features[0:(sizeFeatures-sizeValidation)]
FeaturesTest=Features[(sizeFeatures-sizeValidation):sizeFeatures]
TextureLabelTrain= TextureLabel[0:(sizeFeatures-sizeValidation)]
TextureLabelTest=TextureLabel[(sizeFeatures-sizeValidation):sizeFeatures]
ColorLabelTrain= ColorLabel[0:(sizeFeatures-sizeValidation)]
ColorLabelTest=ColorLabel[(sizeFeatures-sizeValidation):sizeFeatures]
## Random Forest
#Training LR Classifier
clf_rfLR = RandomForestClassifier(max_depth=40, min_samples_leaf=1, min_samples_split=2, bootstrap=False,max_features='sqrt', n_estimators=20)
clf_rfLR.fit(FeaturesTrain, TextureLabelTrain)
predLR=clf_rfLR.predict(FeaturesTest)
accuracyLR = accuracy_score(TextureLabelTest,predLR)

#Training CS Classifier
clf_rfCS = RandomForestClassifier(max_depth=40, min_samples_leaf=1, min_samples_split=2, bootstrap=False, max_features='sqrt', n_estimators=20)
clf_rfCS.fit(FeaturesTrain, ColorLabelTrain)
predCS = clf_rfCS.predict(FeaturesTest)
accuracyLR = accuracy_score(ColorLabelTest,predCS)

print(predLR)
print(predCS)




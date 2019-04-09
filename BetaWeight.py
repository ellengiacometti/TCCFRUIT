


def create_columns(n):
    list_of_columns = []
    for i in range(n):
        list_of_columns.append(0)
    return list_of_columns


import numpy as np
import pandas as pd
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
Norm = pd.read_csv('NormTrain658.csv', index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
FeatureMapLR = pd.read_csv('FeatureMapLR.csv', index_col=False, sep=";")
FeatureMapCS = pd.read_csv('FeatureMapCS.csv', index_col=False, sep=";")
columnsLR=FeatureMapLR.columns[1:]
columnsCS=FeatureMapCS.columns[1:]
AccuracyLR=FeatureMapLR[columnsLR].values
AccuracyCS=FeatureMapCS[columnsCS].values
betasLR = np.float32(np.subtract(1,AccuracyLR))
betasCS = np.float32(np.subtract(1,AccuracyCS))
betasNormLR = pd.DataFrame(columns=columnsLR)
betasNormCS = pd.DataFrame(columns=columnsCS)

for i in range(0,columnsLR.size):
    betasNormLR[columnsLR[i]] = (betasLR[:,i] / np.linalg.norm(betasLR[:,i]))
for j in range(0,columnsCS.size):
    betasNormCS[columnsCS[j]] = (betasCS[:,j] / np.linalg.norm(betasCS[:,j]))

featsTextColumns = Norm.columns[1:9]
histColumns = Norm.columns[9:12]
a = 0
Color = create_columns(histColumns.size)
for histColumn in histColumns:
    Color[a] = [list(map(float, histH)) for histH in Norm[histColumn]]
    Color[a] = np.array(Color[a])
    a+=1
histValues= Color[0]
for b in range(1,a):
    histValues= np.hstack((histValues, Color[b]))

featsTexture = Norm[featsTextColumns].values
FeaturesNorm = np.hstack((featsTexture, histValues))
np.dot()

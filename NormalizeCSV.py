"""Author: Ellen Giacometti
    CRIADO EM: 28/03/2019
    ÚLTIMA ATUALIZAÇÃO: 01/04/2019
    DESC: Código tendo a planilha CSV cria um arquivo CSV com as features normalizadas/ standard """
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization
# https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def create_columns(n):
    list_of_columns = []
    for i in range(n):
        list_of_columns.append(0)
    return list_of_columns

## ler data
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
train = pd.read_csv('Train.csv', index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
## separar a info
infoRegion = train.columns[1:9]
info = train[infoRegion]
## normalizar info
NormA_info=create_columns(11)
NormB_info=create_columns(11)
for n in range(0,len(infoRegion)):
    feat= info[infoRegion[n]]
    feat=feat.values
    # NormA_info[n].append( np.array(feat/np.linalg.norm(feat)))
    # NormB_info[n].append(normalize(feat[:, np.newaxis], axis=0).ravel())
    NormA_info[n]= feat / np.linalg.norm(feat)
    NormB_info[n]= normalize(feat[:,np.newaxis], axis=0).ravel()
print("Wait a minute....")
infoRegionHists = train.columns[9:12]
infoHists=train[infoRegionHists]
NormA_feat=[]
NormB_feat =[]
featHist=[]

for i in infoRegionHists:
    infoHist = infoHists[i]
    featHist = infoHist.values
    for hist in featHist:
        hist=np.float32(hist)
        histNormA = (hist / np.linalg.norm(hist)).tolist()
        histNormB = (normalize(hist[:, np.newaxis], axis=0).ravel()).tolist()
        NormA_feat.append(histNormA)
        NormB_feat.append(histNormB)
    n += 1
    # corH=pd.Series(NormA_feat,index=['ColorH'])
    NormA_info[n] = NormA_feat
    NormB_info[n] = NormB_feat
    NormA_feat = []
    NormB_feat = []
    print("Wait a minute....")

raw_data = {'Object': train['Object'], 'Kurtosis': NormA_info[0], 'Skewness': NormA_info[1],
            'Dissimilarity': NormA_info[2], 'Correlation': NormA_info[3], 'Homogeneity': NormA_info[4],
            'Energy': NormA_info[5], 'Contrast': NormA_info[6], 'ASM': NormA_info[7],
            'ColorH':NormA_info[8],'ColorS':NormA_info[9],'ColorV':NormA_info[10],
            'TextureLabel':train['TextureLabel'],'ColorLabel':train['ColorLabel']}
normA = pd.DataFrame(raw_data,columns=train.columns.values)
normA.to_csv('normATrain.csv', index=False, sep=";")
print("normATrain.csv CRIADO")

#
# raw_dataHist = {'Object': train['Object'], 'ColorH':NormA_info[8] }
# normAHist = pd.DataFrame(raw_dataHist, columns=['Object','ColorH'])
# normAHist.to_csv('COLORA.csv', index=False, sep=";")
# raw_dataHist = {'Object': train['Object'], 'ColorH':NormB_info[8]}
# normBHist = pd.DataFrame(raw_dataHist, columns=['Object','ColorH'])
# normBHist.to_csv('COLORB.csv', index=False, sep=";")

raw_data = {'Object': train['Object'], 'Kurtosis': NormB_info[0], 'Skewness': NormB_info[1],
            'Dissimilarity': NormB_info[2], 'Correlation': NormB_info[3], 'Homogeneity': NormB_info[4],
            'Energy': NormB_info[5], 'Contrast': NormB_info[6], 'ASM': NormB_info[7],
            'ColorH':NormB_info[8],'ColorS':NormB_info[9],'ColorV':NormB_info[10],
            'TextureLabel':train['TextureLabel'],'ColorLabel':train['ColorLabel'] }

normB = pd.DataFrame(raw_data,columns=train.columns.values)
normB.to_csv('normBTrain.csv', index=False, sep=";")
print("normBTrain.csv CRIADO")
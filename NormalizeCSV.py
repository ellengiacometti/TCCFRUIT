"""Author: Ellen Giacometti
    CRIADO EM: 28/03/2019
    ÚLTIMA ATUALIZAÇÃO: 01/04/2019
    DESC: Código tendo a planilha CSV cria um arquivo CSV com as features normalizadas/ standard """
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization
# https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def create_columns(a):
    list_of_columns = []
    for i in range(a):
        list_of_columns.append(0)
    return list_of_columns

## ler data
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
train = pd.read_csv('Train.csv', index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
## separar a info
infoRegion = train.columns[1:9]
info = train[infoRegion]
infoMin=np.amin(info.values,axis=0)
infoMax=np.amax(info.values,axis=0)
num= info.values - infoMin
den=infoMax-infoMin
Norm_info= num/den

# print("Wait a minute....")

infoRegionHists = train.columns[9:12]
infoHists=train[infoRegionHists]
# NormA_feat=[]
NormA_info=create_columns(3)
featHist=[]
NormA_feat = []
n=0
for i in infoRegionHists:
    infoHist = infoHists[i]
    featHist = infoHist.values
    for hist in featHist:
        hist=np.float64(hist)
        histMax=np.amax(hist,axis=0)
        histNorm=hist/histMax
        NormA_feat.append(histNorm.tolist())

    # corH=pd.Series(NormA_feat,index=['ColorH'])
    NormA_info[n] = NormA_feat
    NormA_feat = []
    n += 1
    print("Wait a minute....")

raw_data = {'Object': train['Object'], 'Kurtosis': Norm_info[:,0].tolist(), 'Skewness': Norm_info[:,1].tolist(),
            'Dissimilarity': Norm_info[:,2].tolist(), 'Correlation': Norm_info[:,3].tolist(), 'Homogeneity': Norm_info[:,4].tolist(),
            'Energy': Norm_info[:,5].tolist(), 'Contrast': Norm_info[:,6].tolist(), 'ASM': Norm_info[:,7].tolist(),
            'ColorH':NormA_info[0],'ColorS':NormA_info[1],'ColorV':NormA_info[2],
            'TextureLabel':train['TextureLabel'],'ColorLabel':train['ColorLabel'],'Radius':train['Radius']}
normA = pd.DataFrame(raw_data,columns=train.columns.values)
normA.to_csv('NormTrain658.csv', index=False, sep=";")
print("NormTrain658.csv CRIADO")

parameterData= {'Feature': infoRegion.T, 'Max':infoMax,'Min':infoMin }
Data = pd.DataFrame(parameterData,columns=['Feature','Max','Min'])
Data.to_csv('Data_NormTrain658.csv', index=False, sep=";")
print("Data_NormTrain658.csv CRIADO")
#
# raw_dataHist = {'Object': train['Object'], 'ColorH':NormA_info[8] }
# normAHist = pd.DataFrame(raw_dataHist, columns=['Object','ColorH'])
# normAHist.to_csv('COLORA.csv', index=False, sep=";")
# raw_dataHist = {'Object': train['Object'], 'ColorH':NormB_info[8]}
# normBHist = pd.DataFrame(raw_dataHist, columns=['Object','ColorH'])
# normBHist.to_csv('COLORB.csv', index=False, sep=";")

# raw_data = {'Object': train['Object'], 'Kurtosis': NormB_info[0], 'Skewness': NormB_info[1],
#             'Dissimilarity': NormB_info[2], 'Correlation': NormB_info[3], 'Homogeneity': NormB_info[4],
#             'Energy': NormB_info[5], 'Contrast': NormB_info[6], 'ASM': NormB_info[7],
#             'ColorH':NormB_info[8],'ColorS':NormB_info[9],'ColorV':NormB_info[10],
#             'TextureLabel':train['TextureLabel'],'ColorLabel':train['ColorLabel'] }
#
# normB = pd.DataFrame(raw_data,columns=train.columns.values)
# normB.to_csv('normBTrain.csv', index=False, sep=";")
# print("normBTrain.csv CRIADO")
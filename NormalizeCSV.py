"""Author: Ellen Giacometti
    CRIADO EM: 28/03/2019
    ÚLTIMA ATUALIZAÇÃO: 01/04/2019
    DESC: Código tendo a planilha CSV cria um arquivo CSV com as features normalizadas/ standard """
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization
# https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
import numpy as np
import pandas as pd



def create_columns(a):
    list_of_columns = []
    for i in range(a):
        list_of_columns.append(0)
    return list_of_columns

## ler data
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
Names=['Test.csv','Train.csv']
for Name in Names:
    data = pd.read_csv(Name, index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})

    ## separar a info
    infoRegion = data.columns[1:9]
    info = data[infoRegion]
    if Name== Names[0]:
        infoMin=np.amin(info.values,axis=0)
        infoMax=np.amax(info.values,axis=0)
    num = info.values - infoMin
    den = infoMax - infoMin
    Norm_info = num / den
    # print("Wait a minute....")

    infoRegionHists = data.columns[9:12]
    infoHists=data[infoRegionHists]
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
        # print("Wait a minute....")

    raw_data = {'Object': data['Object'], 'Kurtosis': Norm_info[:,0].tolist(), 'Skewness': Norm_info[:,1].tolist(),
                'Dissimilarity': Norm_info[:,2].tolist(), 'Correlation': Norm_info[:,3].tolist(), 'Homogeneity': Norm_info[:,4].tolist(),
                'Energy': Norm_info[:,5].tolist(), 'Contrast': Norm_info[:,6].tolist(), 'ASM': Norm_info[:,7].tolist(),
                'ColorH':NormA_info[0],'ColorS':NormA_info[1],'ColorV':NormA_info[2],
                'TextureLabel':data['TextureLabel'],'ColorLabel':data['ColorLabel'],'Radius':data['Radius']}
    normA = pd.DataFrame(raw_data,columns=data.columns.values)
    fileName= 'Norm'+ Name.strip('.csv')+str(data.shape[0])+'.csv'
    normA.to_csv(fileName, index=False, sep=";")
    print(fileName," CRIADO!")
parameterData= {'Feature': infoRegion.T, 'Max':infoMax,'Min':infoMin }
Data = pd.DataFrame(parameterData,columns=['Feature','Max','Min'])
Data.to_csv('Data_'+fileName, index=False, sep=";")
print("Data_",fileName,"CRIADO!")

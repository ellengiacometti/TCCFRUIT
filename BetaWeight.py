


def create_columns(n):
    list_of_columns = []
    for i in range(n):
        list_of_columns.append(0)
    return list_of_columns


import numpy as np
import pandas as pd
Names=['FeatureMapLR.csv','FeatureMapCS.csv']
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
Norm = pd.read_csv('NormTrain658.csv', index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
for Name in Names:
    alpha = pd.read_csv(Name, index_col=False, sep=";")
    # FeatureMapCS = pd.read_csv('FeatureMapCS.csv', index_col=False, sep=";")
    columns =alpha.columns[1:]
    Accuracy =alpha[columns ].values
    betas  = np.float32(np.subtract(1,Accuracy ))
    betasNorm  = pd.DataFrame(columns=columns )
    betasMAX=np.amax(betas ,axis=0)
    for i in range(0,columns .size):
        betasNorm [columns [i]] = (betas [0:11,i] /  betasMAX[i])
    betasNorm  =betasNorm .values
    featsColumns = Norm.columns[1:9]
    featsNorm=Norm[featsColumns]
    featsNorm=featsNorm.values
    featsBetaRF=np.zeros_like(featsNorm)
    featsBetaNN=np.zeros_like(featsNorm)
    featsBetaSVM=np.zeros_like(featsNorm)
    for a in range(0,featsColumns.size):
        featsBetaRF[:,a]=featsNorm[:,a]*betasNorm [a,0]
    for b in range(0,featsColumns.size):
        featsBetaNN[:,b]=featsNorm[:,b]*betasNorm [b,1]
    for c in range(0,featsColumns.size):
        featsBetaSVM[:,c]=featsNorm[:,c]*betasNorm [c,2]
    a=0
    b=0
    histsColumns = Norm.columns[9:12]
    Color = create_columns(histsColumns.size)
    for histColumn in histsColumns:
        Color[a] = [list(map(float, histH)) for histH in Norm[histColumn]]
        Color[a] = np.array(Color[a])
        a += 1
    histsNorm= Color[0]
    for b in range(1,a):
        histsNorm= np.hstack((histsNorm, Color[b]))
    ColorH=Color[0]
    ColorS=Color[1]
    ColorV=Color[2]

    histColorHBetaRF = np.zeros_like(ColorH)
    histColorSBetaRF = np.zeros_like(ColorS)
    histColorVBetaRF = np.zeros_like(ColorV)

    histColorHBetaRF= ColorH * betasNorm [8, 0]
    histColorSBetaRF= ColorS * betasNorm [9,0 ]
    histColorVBetaRF = ColorV * betasNorm [10, 0]

    histColorHBetaNN = np.zeros_like(ColorH)
    histColorSBetaNN = np.zeros_like(ColorS)
    histColorVBetaNN = np.zeros_like(ColorV)

    histColorHBetaNN = ColorH * betasNorm [8, 1]
    histColorSBetaNN = ColorS * betasNorm [9,1 ]
    histColorVBetaNN  = ColorV * betasNorm [10, 1]

    histColorHBetaSVM = np.zeros_like(ColorH)
    histColorSBetaSVM = np.zeros_like(ColorS)
    histColorVBetaSVM = np.zeros_like(ColorV)

    histColorHBetaSVM = ColorH * betasNorm [8, 2]
    histColorSBetaSVM = ColorS * betasNorm [9,2 ]
    histColorVBetaSVM  = ColorV * betasNorm [10, 2]
    # RANDOM FOREST
    raw_data = {'Object': Norm['Object'], 'Kurtosis': featsBetaRF[:,0].tolist(), 'Skewness': featsBetaRF[:,1].tolist(),
                    'Dissimilarity': featsBetaRF[:,2].tolist(), 'Correlation': featsBetaRF[:,3].tolist(), 'Homogeneity': featsBetaRF[:,4].tolist(),
                    'Energy': featsBetaRF[:,5].tolist(), 'Contrast': featsBetaRF[:,6].tolist(), 'ASM': featsBetaRF[:,7].tolist(),
                    'ColorH':histColorHBetaRF.tolist(),'ColorS':histColorSBetaRF.tolist(),'ColorV':histColorVBetaRF.tolist(),
                    'TextureLabel':Norm['TextureLabel'],'ColorLabel':Norm['ColorLabel'],'Radius':Norm['Radius']}
    RF = pd.DataFrame(raw_data,columns=Norm.columns.values)
    fileName= 'RFNorm'+ Name.strip('FeatureMap')
    RF.to_csv(fileName, index=False, sep=";")
    print(fileName," CRIADO!")
    # NEURAL NETWORK
    raw_data = {'Object': Norm['Object'], 'Kurtosis': featsBetaNN[:,0].tolist(), 'Skewness': featsBetaNN[:,1].tolist(),
                    'Dissimilarity': featsBetaNN[:,2].tolist(), 'Correlation': featsBetaNN[:,3].tolist(), 'Homogeneity': featsBetaNN[:,4].tolist(),
                    'Energy': featsBetaNN[:,5].tolist(), 'Contrast': featsBetaNN[:,6].tolist(), 'ASM': featsBetaNN[:,7].tolist(),
                    'ColorH':histColorHBetaNN.tolist(),'ColorS':histColorSBetaNN.tolist(),'ColorV':histColorVBetaNN.tolist(),
                    'TextureLabel':Norm['TextureLabel'],'ColorLabel':Norm['ColorLabel'],'Radius':Norm['Radius']}
    NN = pd.DataFrame(raw_data,columns=Norm.columns.values)
    fileName= 'NNNorm'+ Name.strip('FeatureMap')
    NN.to_csv(fileName, index=False, sep=";")
    print(fileName," CRIADO!")
  # SVM
    raw_data = {'Object': Norm['Object'], 'Kurtosis': featsBetaSVM[:,0].tolist(), 'Skewness': featsBetaSVM[:,1].tolist(),
                    'Dissimilarity': featsBetaSVM[:,2].tolist(), 'Correlation': featsBetaSVM[:,3].tolist(), 'Homogeneity': featsBetaSVM[:,4].tolist(),
                    'Energy': featsBetaSVM[:,5].tolist(), 'Contrast': featsBetaSVM[:,6].tolist(), 'ASM': featsBetaSVM[:,7].tolist(),
                    'ColorH':histColorHBetaSVM.tolist(),'ColorS':histColorSBetaSVM.tolist(),'ColorV':histColorVBetaSVM.tolist(),
                    'TextureLabel':Norm['TextureLabel'],'ColorLabel':Norm['ColorLabel'],'Radius':Norm['Radius']}
    SVM = pd.DataFrame(raw_data,columns=Norm.columns.values)
    fileName= 'SVMNorm'+ Name.strip('FeatureMap')
    SVM.to_csv(fileName, index=False, sep=";")
    print(fileName," CRIADO!")







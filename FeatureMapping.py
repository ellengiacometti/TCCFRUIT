"""Author: Ellen Giacometti
    CRIADO EM: 05/04/2019
    ÚLTIMA ATUALIZAÇÃO: 05/02/2019
    DESC: Ler a planilha de dados normalizados e chamar os classificadores removendo feature por feature
    e por fim gerar uma planinha com um feature mapping """

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def create_columns(n):
    list_of_columns = []
    for i in range(n):
        list_of_columns.append(0)
    return list_of_columns

paramIn = 13
percent = 0.25
paramInACCU = paramIn -1
accuracyRFLR=create_columns(paramInACCU)
accuracyRFCS=create_columns(paramInACCU)
accuracyNNLR=create_columns(paramInACCU)
accuracyNNCS=create_columns(paramInACCU)
accuracySVMLR=create_columns(paramInACCU)
accuracySVMCS=create_columns(paramInACCU)
## Reading CSV
prop = lambda x: x.strip("[]").replace("'", "").split(", ")
Norm = pd.read_csv('normBTrain.csv', index_col=False, sep=";",converters={'ColorH': prop, 'ColorS': prop, 'ColorV': prop})
## Manipulating Columns Index to exclude a feature which index number is paramIn
for paramIn in  range(1,paramIn):
    if paramIn<=9:
        featInt = Norm.columns[1:paramIn]
        featInt2 = Norm.columns[(paramIn+1):9]
        colunas=featInt.union(featInt2,sort=False)
        histColumns = Norm.columns[9:12]
        Color = create_columns(3)
    elif paramIn>9 and paramIn<=12:
        colunas=Norm.columns[1:9]
        featHists = Norm.columns[9:paramIn]
        featHists2 = Norm.columns[paramIn+1:12]
        histColumns= featHists.union(featHists2,sort=False)

    else:
        colunas = Norm.columns[1:9]
        Color = create_columns(3)
    a = 0
    Color = create_columns(histColumns.size)
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
    sizeValidation = round((sizeFeatures)*percent)
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
    predRFLR=clf_rfLR.predict(FeaturesTest)
    paramInACCU = paramIn -1
    accuracyRFLR[paramInACCU] = accuracy_score(TextureLabelTest,predRFLR)

    #Training CS Classifier
    clf_rfCS = RandomForestClassifier(max_depth=40, min_samples_leaf=1, min_samples_split=2, bootstrap=False, max_features='sqrt', n_estimators=20)
    clf_rfCS.fit(FeaturesTrain, ColorLabelTrain)
    predRFCS = clf_rfCS.predict(FeaturesTest)
    accuracyRFCS[paramInACCU] = accuracy_score(ColorLabelTest,predRFCS)
    if paramIn<12:
         print("-----------++++", Norm.columns[paramIn],"++++-----------")
    else:
        print("------------++++ALL FEATURES++++------------")
    print("Accurácia RF -LR     |       Accurácia RF -CS")
    print(accuracyRFLR[paramInACCU],"         ",accuracyRFCS[paramInACCU])

    ## Neural Network
    # Training LR Classifier
    clf_nnLR = MLPClassifier(activation= 'tanh', hidden_layer_sizes= (50, 50, 50), alpha= 0.0001, learning_rate= 'adaptive', solver = 'lbfgs', random_state= 30)
    clf_nnLR.fit(FeaturesTrain, TextureLabelTrain)
    predNNLR = clf_nnLR.predict(FeaturesTest)
    accuracyNNLR[paramInACCU]=accuracy_score(TextureLabelTest, predNNLR)

    #Training CS Classifier
    clf_nnCS = MLPClassifier(learning_rate='constant', solver='lbfgs', activation='relu', random_state=15,hidden_layer_sizes=(50, 100, 50), alpha=0.05)
    clf_nnCS.fit(FeaturesTrain, ColorLabelTrain)
    predNNCS = clf_nnCS.predict(FeaturesTest)
    accuracyNNCS[paramInACCU] = accuracy_score(TextureLabelTest, predNNCS)
    print("+++--------------------------------+++")
    print("Accurácia NN -LR     |       Accurácia NN -CS")
    print(accuracyNNLR[paramInACCU],"         ",accuracyNNCS[paramInACCU])

    ## SVM
    # Training LR Classifier
    clf_svmLR = SVC(C=1, gamma=0.5,decision_function_shape = 'ovo',kernel='poly')
    clf_svmLR.fit(FeaturesTrain, TextureLabelTrain)
    predSVMLR = clf_svmLR.predict(FeaturesTest)
    accuracySVMLR[paramInACCU] = accuracy_score(TextureLabelTest, predSVMLR)
    # Training CS Classifier
    clf_svmCS = SVC(C=1, gamma=0.5, decision_function_shape='ovo', kernel='poly')
    clf_svmCS.fit(FeaturesTrain, ColorLabelTrain)
    predSVMCS = clf_svmCS.predict(FeaturesTest)
    accuracySVMCS[paramInACCU] = accuracy_score(TextureLabelTest, predSVMCS)
    print("+++--------------------------------+++")
    print("Accurácia SVM -LR     |       Accurácia SVM -CS")
    print(accuracySVMLR[paramInACCU], "         ", accuracySVMCS[paramInACCU])
allfeat=pd.Index(["All Features"])
CSVnamefeatures=Norm.columns[1:12].union(allfeat,sort=False)
raw_data = {'Features':CSVnamefeatures,'Accuracy_RF':accuracyRFLR,'Accuracy_NN':accuracyNNLR,'Accuracy_SVM':accuracySVMLR }
MapLR = pd.DataFrame(raw_data,columns=['Features','Accuracy_RF','Accuracy_NN','Accuracy_SVM'])
MapLR.to_csv('FeatureMapLR.csv', index=False, sep=";")
print("FeatureMapLR.csv CRIADO")

raw_data = {'Features':CSVnamefeatures,'Accuracy_RF':accuracyRFCS,'Accuracy_NN':accuracyNNCS,'Accuracy_SVM':accuracySVMCS}
MapCS = pd.DataFrame(raw_data,columns=['Features','Accuracy_RF','Accuracy_NN','Accuracy_SVM'])
MapCS.to_csv('FeatureMapCS.csv', index=False, sep=";")
print("FeatureMapCS.csv CRIADO")

betaRFLR= [1- x for x in accuracyRFLR ]
betaRFCS= [1- x for x in accuracyRFCS]
betaNNLR= [1- x for x in accuracyNNLR]
betaNNCS=[1- x for x in accuracyNNCS]
betaSVMLR=[1- x for x in accuracySVMLR]
betaSVMCS=[1- x for x in accuracySVMCS]
##TODO: criar novo arquivo e ler planilha CSV FeatureMapCS E FeatureMapLR
##TODO: (normalize(hist[:, np.newaxis], axis=0).ravel())
##TODO:Multiplicar betaNorm por CADA feature RESPECTIVAMENTE
##TODO: RODAR DE NOVO FEATUREMAPPING
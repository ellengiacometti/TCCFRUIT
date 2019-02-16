"""Author: Ellen Giacometti
    CRIADO EM: 21/12/2018
    ÚLTIMA ATUALIZAÇÃO: 05/02/2019
    DESC: Código tendo as features cria um arquivo csv com as features e as respectivas labels"""
import os
import glob
import pandas as pd
from TrataImagem import TrataImagem

""" Lendo Diretório """
train_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_LABEL"
test_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_TEST"
dir=[train_path,test_path]


""" Extraindo Features de cada imagem """
print("[STATUS] Started extracting  TRAIN features")
valueType = 0
for type_dir in dir:
    """ Definindo Variáveis """
    i=1
    Kurtosis = []
    Skewness = []
    Dissimilarity = []
    Correlation = []
    Homogeneity = []
    Energy = []
    Contrast = []
    ASM = []
    texture_label = []
    color = []
    color_label = []
    for name in os.listdir(type_dir):

        cur_path = type_dir + "/" + name
        cur_label = name
        for file in glob.glob(cur_path):
            print("Processing Image - {} in {}".format(i, cur_label))
            features = TrataImagem(file,0,0)
            # append the feature vector and label
            Kurtosis.append(features[4])
            Skewness.append(features[5])
            Dissimilarity.append(features[6])
            Correlation.append(features[7])
            Homogeneity.append(features[8])
            Energy.append(features[9])
            Contrast.append(features[10])
            ASM .append(features[11])
            color.append(features[3])
            texture_label.append(cur_label[5])
            color_label.append(cur_label[6])
            i+=1

    raw_data = {'Object':os.listdir(type_dir),'Kurtosis': Kurtosis,'Skewness':Skewness,'Dissimilarity':Dissimilarity,'Correlation':Correlation,'Homogeneity':Homogeneity,'Energy':Energy,'Contrast':Contrast, 'ASM':ASM,'Color': color,'TextureLabel':  texture_label,'ColorLabel': color_label}
    df = pd.DataFrame(raw_data, columns = ['Object','Kurtosis','Skewness','Dissimilarity','Correlation','Homogeneity','Energy','Contrast', 'ASM', 'Color','TextureLabel','ColorLabel'])
    valueType += 1
    if valueType==1:
        df.to_csv('Train.csv',index=False,sep=";")
        print("Train.csv CRIADO")
    else:
        df.to_csv('Test.csv', index=False, sep=";")
        print("Test.csv CRIADO")


# """Fit The Label Encoder"""
# # Create a label (category) encoder object
# le = preprocessing.LabelEncoder()
# # Fit the encoder to the pandas column
# le.fit(df['LABELS'])
# # View the labels (if you want)
# list(le.classes_)
#
# """Transform Categories Into Integers"""
# # Apply the fitted encoder to the pandas column
# le.transform(df['LABELS'])
#
# """Transform Integers Into Categories"""
# # Convert some integers into their category names
# list(le.inverse_transform([0, 1, 1]))



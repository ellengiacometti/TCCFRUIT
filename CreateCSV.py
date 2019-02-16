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
train_names = os.listdir(train_path)

""" Definindo Variáveis """
Kurtosis= []
Skewness= []
Dissimilarity=[]
Correlation=[]
Homogeneity=[]
Energy=[]
Contrast=[]
ASM=[]
texture_label = []
color=[]
color_label=[]

KurtosisTest= []
SkewnessTest= []
DissimilarityTest=[]
CorrelationTest=[]
HomogeneityTest=[]
EnergyTest=[]
ContrastTest=[]
ASMTest=[]
texture_labelTest = []
colorTest=[]
color_labelTest=[]
test_names=[]

""" Extraindo Features de cada imagem """
print("[STATUS] Started extracting  TRAIN features")
i=1
j=1
for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name
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
        # show loop update
        i+=1
    # loop over the test images

print("[STATUS] Started extracting  TEST features")
for file in glob.glob(test_path + "/*.jpg"):
    nome=file[59:len(file)]
    print("Processing Image - {} in {}".format(j,nome))
    features = TrataImagem(file,0,0)
    # evaluate the model and predict label
    KurtosisTest.append(features[4])
    SkewnessTest.append(features[5])
    DissimilarityTest.append(features[6])
    CorrelationTest.append(features[7])
    HomogeneityTest.append(features[8])
    EnergyTest.append(features[9])
    ContrastTest.append(features[10])
    ASMTest.append(features[11])
    texture_labelTest.append(file[64])
    # colorTest.append(' '.join(map(str, features[3])))
    colorTest.append(features[3])
    color_labelTest.append(file[65])
    test_names.append(nome)
    j+=1


""" Montando o arquivo csv """
raw_data = {'Object':train_names,'Kurtosis': Kurtosis,'Skewness':Skewness,'Dissimilarity':Dissimilarity,'Correlation':Correlation,'Homogeneity':Homogeneity,'Energy':Energy,'Contrast':Contrast, 'ASM':ASM,'Color': color,'TextureLabel':  texture_label,'ColorLabel': color_label}
df = pd.DataFrame(raw_data, columns = ['Object','Kurtosis','Skewness','Dissimilarity','Correlation','Homogeneity','Energy','Contrast', 'ASM', 'Color','TextureLabel','ColorLabel'])
df.to_csv('Train.csv',index=False,sep=";")
print("Train.csv CRIADO")
raw_data = {'Object':test_names,'Kurtosis': KurtosisTest,'Skewness':SkewnessTest,'Dissimilarity':DissimilarityTest,'Correlation':CorrelationTest,'Homogeneity':HomogeneityTest,'Energy':EnergyTest,'Contrast':ContrastTest, 'ASM':ASMTest,'Color': colorTest,'TextureLabel':  texture_labelTest,'ColorLabel': color_labelTest}
df1 = pd.DataFrame(raw_data, columns = ['Object','Kurtosis','Skewness','Dissimilarity','Correlation','Homogeneity','Energy','Contrast', 'ASM', 'Color','TextureLabel','ColorLabel'])
df1.to_csv('Test.csv',index=False,sep=";")
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



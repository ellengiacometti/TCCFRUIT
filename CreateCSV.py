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
train_names = os.listdir(train_path)

""" Definindo Variáveis """
Kurtosis= []
Skewness= []
texture_label = []
color=[]
color_label=[]
KurtosisTest= []
SkewnessTest= []
texture_labelTest = []
colorTest=[]
color_labelTest=[]
test_names=[]

""" Extraindo Features de cada imagem """
print("[STATUS] Started extracting  textures..")
i = 1
for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name
    for file in glob.glob(cur_path):
        print("Processing Image - {} in {}".format(i, cur_label))
        features = TrataImagem(file)
        # append the feature vector and label
        Kurtosis.append(features[4])
        Skewness.append(features[5])
        texture_label.append(cur_label[5])
        # color.append(' '.join(map(str, features[3])))
        color.append(features[3])
        color_label.append(cur_label[6])
        # show loop update
        i += 1
    # loop over the test images
test_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_TEST"
for file in glob.glob(test_path + "/*.jpg"):
    features = TrataImagem(file)
    # evaluate the model and predict label
    KurtosisTest.append(features[4])
    SkewnessTest.append(features[5])
    texture_labelTest.append(file[64])
    # colorTest.append(' '.join(map(str, features[3])))
    colorTest.append(features[3])
    color_labelTest.append(file[65])
    test_names.append(file[59:len(file)])


""" Montando o arquivo csv """
# raw_data = {'Object':train_names,'Kurtosis': Kurtosis,'Skewness':Skewness,'Color': color,'TextureLabel':  texture_label,'ColorLabel': color_label}
# df = pd.DataFrame(raw_data, columns = ['Object','Kurtosis','Skewness', 'Color','TextureLabel','ColorLabel'])
# # df.to_csv('Train.csv',index=False,quoting=csv.QUOTE_NONE,sep=",",escapechar=" ")
# df.to_csv('Train.csv',index=False,sep=";")
# print("Train.csv CRIADO")
raw_data = {'Object':test_names,'Kurtosis': KurtosisTest,'Skewness':SkewnessTest,'Color': colorTest,'TextureLabel':  texture_labelTest,'ColorLabel': color_labelTest}
df1 = pd.DataFrame(raw_data, columns = ['Object','Kurtosis','Skewness','Color','TextureLabel','ColorLabel'])
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



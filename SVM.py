import cv2
import numpy as np
import os
import glob
from TrataImagem import TrataImagem
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler



if __name__ == '__main__':
    # # load the training dataset
    # train_path  = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_LABEL"
    # train_names = os.listdir(train_path)
    #
    # # empty list to hold feature vectors and train labels
    # train_features = []
    # train_labels = []
    #
    # # loop over the training dataset
    # print("[STATUS] Started extracting  textures..")
    # i = 1
    # Liso=0
    # Rugoso=0
    # Acertos = 0
    # Erros = 0
    # Amostras = 0
    # for train_name in train_names:
    #     cur_path = train_path + "/" + train_name
    #     cur_label = train_name
    #     for file in glob.glob(cur_path):
    #         print ("Processing Image - {} in {}".format(i, cur_label))
    #         features = TrataImagem(file)
    #         # append the feature vector and label
    #         train_features.append(features[2])
    #         train_labels.append(cur_label[5])
    #         if(cur_label[5]=='L'):
    #             Liso +=1
    #         elif(cur_label[5]=='R'):
    #             Rugoso+=1
    #         # show loop update
    #         i += 1
    # # have a look at the size of our feature vector and labels
    # print ("Training features: {}".format(np.array(train_features).shape))
    # print ("Training labels: {}".format(np.array(train_labels).shape))
    # print("Liso:",Liso)
    # print("Rugoso:", Rugoso)

    train = pd.read_csv('Train.csv',index_col='Object')
    test = pd.read_csv('Test.csv', index_col='Object')
    Texture = train.columns[1:3]
    TextureLabel = train['TextureLabel']
    TextureLabelTest=test['TextureLabel']
    le = LabelEncoder()
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    le.fit(TextureLabel)
    TextureLabel = le.transform(TextureLabel)
    X_train=train[Texture]

    # svm = SVC()
    # parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75,0.05), 'gamma': (0.5,1, 2, 3, 'auto'),
    #               'decision_function_shape': ('ovo', 'ovr'),'class_weight': [{0: 1,1: w2} for w2 in [2, 4, 6, 10,12]]}
    # clf = GridSearchCV(svm, parameters,verbose = 2)
    # clf.fit(X_train, TextureLabel)
    # print("accuracy:" + str(np.average(cross_val_score(clf, X_train, TextureLabel, scoring='accuracy'))))
    # print("f1:" + str(np.average(cross_val_score(clf, X_train, TextureLabel, scoring='f1'))))
    # print(clf.best_params_)
    # """ SVM """
    #  # create the SVM classifier
    # print ("[STATUS] Creating the classifier..")
    # # clf_svm = SVC(C=0.065 ,gamma=0.5,class_weight='balanced')
    clf_svm = SVC(C=1, gamma=0.5, class_weight={0:1,1:2},decision_function_shape = 'ovo')
    # fit the training data and labels
    print ("[STATUS] Fitting data/label to model..")
    clf_svm.fit(train[Texture], TextureLabel)
    prediction = clf_svm.predict(test[Texture])
    print("Accuracy for SVM on CV data: ", accuracy_score(TextureLabelTest, prediction))

    # loop over the test images
    # test_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_TEST"
    # for file in glob.glob(test_path + "/*.jpg"):
    #     features = TrataImagem(file)
    #
    #     # evaluate the model and predict label
    #     features = np.array(features[2])
    #     prediction = clf_svm.predict(features.reshape(1, -1))[0]
    #
    #     print("Nome:", file[58:70])
    #     print("Prediction:", prediction)
    #     Amostras+=1
    #     if(file[65]==prediction):
    #         Acertos+=1
    #     else:
    #         Erros+=1
    # print('Numero de Acertos', Acertos)
    # print('Numero de Erros', Erros)
    # print('Numero de Amostras', Amostras)
    # print('%ERROS:', (Erros/Amostras)*100)
    # print('%ACERTOS:', (Acertos / Amostras) * 100)

    #
    # """ RANDOM FOREST """
    # classifier = RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 60, max_depth=8, criterion='gini')
    # classifier.fit(train_features,train_labels)
    # # loop over the test images
    # test_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_TEST"
    # for file in glob.glob(test_path + "/*.jpg"):
    #      features = TrataImagem(file)
    #      # evaluate the model and predict label
    #      features = np.array(features[2])
    #      y_pred = classifier.predict(features.reshape(1, -1))[0]
    #      print("Nome:", file[58:70])
    #      print("Prediction:", y_pred)
    #      Amostras+=1
    #      if(file[65]==y_pred):
    #         Acertos+=1
    #      else:
    #         Erros+=1
    # print('Numero de Acertos', Acertos)
    # print('Numero de Erros', Erros)
    # print('Numero de Amostras', Amostras)
    # print('%ERROS:', (Erros/Amostras)*100)
    # print('%ACERTOS:', (Acertos / Amostras) * 100)

# #TODO:https://bigdata-madesimple.com/dealing-with-unbalanced-class-svm-random-forest-and-decision-tree-in-python/

import cv2
import numpy as np
import os
import glob
from TrataImagem import TrataImagem
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    # load the training dataset
    train_path  = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_LABEL"
    train_names = os.listdir(train_path)

    # empty list to hold feature vectors and train labels
    train_features = []
    train_labels = []

    # loop over the training dataset
    print("[STATUS] Started extracting  textures..")
    i = 1
    Liso=0
    Rugoso=0
    Acertos = 0
    Erros = 0
    Amostras = 0
    for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        for file in glob.glob(cur_path):
            print ("Processing Image - {} in {}".format(i, cur_label))
            features = TrataImagem(file)
            # append the feature vector and label
            train_features.append(features[2])
            train_labels.append(cur_label[5])
            if(cur_label[5]=='L'):
                Liso +=1
            elif(cur_label[5]=='R'):
                Rugoso+=1
            # show loop update
            i += 1
    # have a look at the size of our feature vector and labels
    print ("Training features: {}".format(np.array(train_features).shape))
    print ("Training labels: {}".format(np.array(train_labels).shape))
    print("Liso:",Liso)
    print("Rugoso:", Rugoso)


    """ SVM """
    #  # create the SVM classifier
    # print ("[STATUS] Creating the classifier..")
    # clf_svm = SVC(C=0.065 ,gamma=0.5,class_weight='balanced')
    #
    # # fit the training data and labels
    # print ("[STATUS] Fitting data/label to model..")
    # clf_svm.fit(train_features, train_labels)
    #
    # # loop over the test images
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
    #
    # print('Numero de Acertos', Acertos)
    # print('Numero de Erros', Erros)
    # print('Numero de Amostras', Amostras)
    # print('%ERROS:', (Erros/Amostras)*100)
    # print('%ACERTOS:', (Acertos / Amostras) * 100)

    #
    """ RANDOM FOREST """
    sc = StandardScaler()
    X_train= sc.fit_transform(train_features)
    Y_train= train_labels
    classifier = RandomForestClassifier(n_estimators=20, random_state=0)
    classifier.fit(X_train,Y_train)
    # # loop over the test images
    test_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_TEST"
    for file in glob.glob(test_path + "/*.jpg"):
         features = TrataImagem(file)
         # evaluate the model and predict label
         features = np.array(features[2])
         y_pred = classifier.predict(features.reshape(1, -1))
         print("Nome:", file[58:70])
         print("Prediction:", y_pred)
    #     Amostras+=1
    #     if(file[65]==prediction):
    #         Acertos+=1
    #     else:
    #         Erros+=1
    #
    # #TODO:https://bigdata-madesimple.com/dealing-with-unbalanced-class-svm-random-forest-and-decision-tree-in-python/

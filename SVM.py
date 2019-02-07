import cv2
import numpy as np
import os
import glob
from TrataImagem import TrataImagem
from sklearn.svm import SVC

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
    for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        for file in glob.glob(cur_path):
            print ("Processing Image - {} in {}".format(i, cur_label))
            features = TrataImagem(file)
            # append the feature vector and label
            train_features.append(features[2])
            train_labels.append(cur_label[5])
            # show loop update
            i += 1
    # have a look at the size of our feature vector and labels
    print ("Training features: {}".format(np.array(train_features).shape))
    print ("Training labels: {}".format(np.array(train_labels).shape))
    # create the classifier
    print ("[STATUS] Creating the classifier..")
    clf_svm = SVC(C=1 ,gamma=1,class_weight='balanced',   random_state=9)

    # fit the training data and labels
    print ("[STATUS] Fitting data/label to model..")
    clf_svm.fit(train_features, train_labels)

    # loop over the test images
    test_path = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_TEST"
    for file in glob.glob(test_path + "/*.jpg"):
        # extract haralick texture from the image
        features = TrataImagem(file)

        # evaluate the model and predict label
        features = np.array(features[2])
        prediction = clf_svm.predict(features.reshape(1, -1))[0]

        # # show the label
        # cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        print("Nome:", file[58:70])
        print("Prediction:", prediction)
    #TODO:https://bigdata-madesimple.com/dealing-with-unbalanced-class-svm-random-forest-and-decision-tree-in-python/

    # # display the output image
    # cv2.imshow("Test_Image", image)
    # cv2.waitKey(0)
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import TrataImagem as TI
import DataAugmentation as DA

if __name__ == '__main__':

    """PARÂMETROS"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path", required=True, help="path to the input image")
    args = vars(ap.parse_args())
    imageDir_val= args["path"]
    # Atributos = []
    # Atributos.append(TI.TrataImagem(imageDir_val, visual=0, verbose=0))
    src='/home/ellengiacometti/PycharmProjects/TCCFRUIT/0356_LSM.jpg'
    origem = '/home/ellengiacometti/PycharmProjects/TCCFRUIT/OR_DA'
    destino = '/home/ellengiacometti/PycharmProjects/TCCFRUIT/DES_DA'
    # DA.DataAugmentation(srcDirOr=origem,srcDirEnd=destino,initialid= 1340,type=2,angle=180,size=(500,500),dB=0.01,direction='Vertical')
    TI.TrataImagem(src, 1, 1)
     #1, max_depth = 40, min_samples_leaf = 1, min_samples_split = 2, bootstrap = False, max_features = 'sqrt', n_estimators = 20

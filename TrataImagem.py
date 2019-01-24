"""Author: Ellen Giacometti
CRIADO EM: 21/12/2018
DESC:Código que recebe uma imagem faz o tratamento morfológico e recorta a ROI """
import cv2 as cv
import argparse
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import roberts



def TrataImagem(src):
    label_num_valid =0

    """LENDO IMAGEM - TRATAMENTO INICIAL"""
    imgRaw = cv.imread(src)
    img= cv.cvtColor(imgRaw, cv.COLOR_BGR2RGB)
    resized = cv.resize(img, (500, 500))
    """SEPARANDO A  MASCARA ONDE TEM AS CORES DE INTERESSE"""
    ##Faixa da cor a ser cortada
    # remover
    # Faixa de Cinza - 119 132 108 - 120 120 120
    boundaries = [([0, 72, 0], [119, 132, 108])]
    boundaries1 = [([120, 120, 120], [178, 255, 175])]
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask1 = cv.inRange(resized, lower, upper)

    for (baixo, cima) in boundaries1:
        # create NumPy arrays from the boundaries
        baixo = np.array(baixo, dtype="uint8")
        cima = np.array(cima, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask2 = cv.inRange(resized, baixo, cima)
        # Visualiza a Máscara em cima da foto real
    mask = mask1 + mask2
    output = cv.bitwise_and(resized, resized, mask=mask)


    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    edge_roberts = roberts(gray) > 0.01
    # """THRESHOLDING USANDO OPENCV"""
    # cv_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 151, 3)
    """MORFOLOGIA CLOSING"""
    block_size = 12
    kernel = np.ones((block_size, block_size), np.uint8)
    cv_thresh_Mo = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    """CONECTANDO OS ELEMENTOS DAS IMAGENS BINÁRIAS"""
    cv_thresh_La,label_num, = sm.label(cv_thresh_Mo,return_num=1,connectivity=1)
    print("\nLabels Encontrados:", label_num ,"\n")

    """Teste de Resize para Melhora de Performance"""
    # TST = st.resize(cv_thresh_Mo, (1000,100))
    # cv_thresh_La, label_num, = sm.label(TST, return_num=1, connectivity=1.5)

    """DEBUG VERSION - PRINTANDO IMAGENS"""
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(40,40), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(gray, cmap=plt.cm.gray)
    ax1.set_title('THRESHED IMAGE')
    ax2.axis('off')
    ax2.imshow(mask, cmap=plt.cm.gray)
    ax2.set_title('LABELED & CONNECTED IMAGE')
    ax3.axis('off')
    ax3.imshow(cv_thresh_La, cmap=plt.cm.gray)
    ax3.set_title('LABELED & CONNECTED IMAGE')
    plt.show()


    """Contornos"""
    contornos = sm.find_contours(cv_thresh_La, 0.5)
    fig1, ax = plt.subplots()
    ax.imshow(resized, interpolation='nearest', cmap=plt.cm.gray)

    for n, contorno in enumerate(contornos):
        ax.plot(contorno[:, 1], contorno[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    """Area"""
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(resized)
    prop_region = sm.regionprops(cv_thresh_La)
    ###Loop pelas Regiões checando parâmetros destas afim de filtrar só as regions onde possivelmente tenham frutas
    for region in prop_region:
        """ANALISANDO O  NÚMERO DE BBOX"""
        if (label_num != 1 and region.area >=200):
        # if (label_num != 1 and region.filled_area >= 500000):
            label_num_valid = label_num_valid + 1
            """DEBUG VERSION - MARCANDO BBOX"""
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            pic=cv_thresh_La[int(region.bbox[0]):int(region.bbox[2]), int(region.bbox[1]):int(region.bbox[3])]
    ax.set_axis_off()
    plt.tight_layout()
    print("Labels Válidos:", label_num_valid, "\n")
    plt.show()

if __name__ == '__main__':
    """PARÂMETROS"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path", required=True, help="path to the input image")
    args = vars(ap.parse_args())
    imageDir_val= args["path"]
    TrataImagem(imageDir_val)







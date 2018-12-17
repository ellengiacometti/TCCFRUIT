"""Author: Ellen Giacometti
CRIADO EM: 02/10/2018
DESC: Esse código tem como função caminho de diretório e retornar uma lista com as regiões de interesse da foto recortadas.
"""
import cv2 as cv
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def TrataImagem(src):
    label_num_valid = 0
    img_bbox_crop = []
    """LENDO IMAGEM - TRATAMENTO INICIAL"""
    img = cv.imread(src)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    """THRESHOLDING USANDO OPENCV"""
    cv_thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 2)

    """MORFOLOGIA CLOSING"""
    block_size = 8
    kernel = np.ones((block_size, block_size), np.uint8)
    cv_thresh_Mo = cv.morphologyEx(cv_thresh, cv.MORPH_CLOSE, kernel)

    """CONECTANDO OS ELEMENTOS DAS IMAGENS BINÁRIAS"""
    cv_thresh_La,label_num, = sm.label(cv_thresh_Mo,return_num=1,connectivity=1.5)
    print("\nLabels Encontrados:", label_num ,"\n")

    """Teste de Resize para Melhora de Performance"""
    # TST = st.resize(cv_thresh_Mo, (1000,100))
    # cv_thresh_La, label_num, = sm.label(TST, return_num=1, connectivity=1.5)

    """DEBUG VERSION - PRINTANDO IMAGENS"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50,50), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(gray, cmap=plt.cm.gray)
    ax1.set_title('THRESHED IMAGE')
    ax2.axis('off')
    ax2.imshow(cv_thresh_La, cmap=plt.cm.gray)
    ax2.set_title('LABELED & CONNECTED IMAGE')
    plt.show()

    """ANÁLISE DE PROPERTIES DAS REGIONS"""

    ###DEBUG VERSION - PLOTANDO IMAGEM PARA MARCAÇÃO BBOX
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ###Loop pelas Regiões checando parâmetros destas afim de filtrar só as regions onde possivelmente tenham objetos
    a=0
    for region in sm.regionprops(cv_thresh_La):
        """ANALISANDO O  NÚMERO DE BBOX"""
        if (label_num != 1 and region.filled_area >= 1000 and region.filled_area <= 15000):
        # if (label_num != 1 and region.filled_area >= 500000):
            label_num_valid = label_num_valid + 1
            """DEBUG VERSION - MARCANDO BBOX"""
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            pic=cv_thresh_La[int(region.bbox[0]):int(region.bbox[2]), int(region.bbox[1]):int(region.bbox[3])]
            img_bbox_crop.append(pic)
    ax.set_axis_off()
    plt.tight_layout()
    print("Labels Válidos:", label_num_valid, "\n")
    plt.show()
    size = (len(img_bbox_crop))
    return img_bbox_crop,size








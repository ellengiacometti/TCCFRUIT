import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from heapq import nlargest
import matplotlib.patches as mpatches
from scipy.stats import kurtosis, skew

def TrataImagem(src):
    """Author: Ellen Giacometti
    CRIADO EM: 21/12/2018
    ÚLTIMA ATUALIZAÇÃO: 05/02/2019
    DESC: Código que recebe uma imagem e extrai os Atributos do limão contido nela"""
    """LENDO IMAGEM """
    # Lendo Imagem
    img = cv.imread(src)
    # Convertendo canal HSV
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Separando o canal de saturação
    h, s, v = cv.split(imgHSV)
    """ EXIBINDO CANAIS """
    # fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
    # ax1.axis('off')
    # ax1.imshow(h, cmap=plt.cm.gray)
    # ax1.set_title(' h IMAGE')
    # ax2.axis('off')
    # ax2.imshow(s, cmap=plt.cm.gray)
    # ax2.set_title('s IMAGE')
    # ax3.axis('off')
    # ax3.imshow(v, cmap=plt.cm.gray)
    # ax3.set_title('v  IMAGE')
    # plt.show()
    """ PROCESSAMENTO DA IMAGEM """
     # Filtro para borrar
    s_Blur = cv.blur(s,(5,5))
    # Binarizando a imagem
    _, s_Thresh = cv.threshold(s_Blur,40,255,cv.THRESH_BINARY)
    # Morfologia tamanho do elemento estrutural
    block_size = 30
    kernel = np.ones((block_size, block_size), np.uint8)
    # Executando Dilation e Closing
    s_Closing = cv.morphologyEx(s_Thresh, cv.MORPH_CLOSE, kernel)
    # Resultado da Máscara em RGB
    s_Result = cv.bitwise_and(cv.cvtColor(img, cv.COLOR_BGR2RGB), cv.cvtColor(img, cv.COLOR_BGR2RGB), mask=s_Closing)
    """PRINTANDO IMAGENS DO PROCESSO"""
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(10, 5), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(s_Blur, cmap=plt.cm.gray)
    ax1.set_title('s_Blur ')
    ax2.axis('off')
    ax2.imshow(s_Thresh, cmap=plt.cm.gray)
    ax2.set_title('s_Thresh')
    ax3.axis('off')
    ax3.imshow(s_Closing, cmap=plt.cm.gray)
    ax3.set_title('s_Closing')
    ax4.axis('off')
    ax4.imshow(s_Result, cmap=plt.cm.gray)
    ax4.set_title('s_Result')
    plt.show()

    """ CRIANDO ROI """
    # Declarando variável BoundingBox
    BoundingBox = np.zeros_like(img)
    BoundingBox[s_Closing == 255] = img[s_Closing == 255]
    # Definindo pontos para corte
    (x, y) = np.where(s_Closing == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    # Desprezando a imagem ao redor dos pontos
    BoundingBox = BoundingBox[topx:bottomx + 1, topy:bottomy + 1]
    # Convertendo para cinza
    gray_BoundingBox = cv.cvtColor(BoundingBox, cv.COLOR_RGB2GRAY)
    """ CONTORNO OPENCV """
    # Capturando contornos
    _, contornosCV, _ = cv.findContours(gray_BoundingBox, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Adquirindo o número de contornos encontrados
    tamanho_contornoCV = len(contornosCV)
    print("\n---~ INFO CONTORNO DO LIMÃO ~---")
    print("Número de Contornos Encontrados na Imagem:", tamanho_contornoCV)
    # Declarando um array para armazenar o tamanho de cada contorno
    tamanhoCV = np.empty([tamanho_contornoCV])
    for i in range(tamanho_contornoCV):
        # Adquirindo o tamanho de cada contorno encontrado
        tamanhoCV[i] = contornosCV[i].shape[0]
    # Detectando os N maiores contornos e seus indices, alterando N , os N maiores contornos serão armazenados.
    maioresCV = nlargest(1, enumerate(tamanhoCV), key=lambda a: a[1])
    print("Perímetro:", maioresCV[0][1])
    """ MEDIDAS LIMÃO - CONTORNO ÚTIL """
    contorno_util=contornosCV[maioresCV[0][0]]
    M = cv.moments(contorno_util)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print("Centróide Limão:(",cx ,",",cy,")")
    """MEDIDAS CIRCUNFERÊNCIA CIRCUNSCRITA"""
    ((x, y), raio) = cv.minEnclosingCircle(contorno_util)
    centroide = (int(x), int(y))
    print("\n---~ SIZE & SHAPE - CIRCUNFERÊNCIA ~---")
    print("Raio:", raio, "\nCentro:", centroide)
    """HISTOGRAMA DO CANAL H"""
    # Mudando os canais da ROI
    HSV_BoundingBox = cv.cvtColor(BoundingBox, cv.COLOR_BGR2HSV)
    # Separando o canal de saturação
    h_BoundingBox, _, _ = cv.split(HSV_BoundingBox)
    # Realizando Histograma da ROI
    hist = cv.calcHist(h_BoundingBox, [0], None, [180], [0, 179])
    """DEBUG VERSION """
    """DESENHANDO O CONTORNO E A CIRCUNFERÊNCIA"""
    fig2, ax = plt.subplots()
    ax.imshow(gray_BoundingBox, interpolation='nearest', cmap=plt.cm.gray)
    for n, contornoCV in enumerate(contornosCV):
        if (n in (np.transpose(np.asanyarray(maioresCV))[:])):
            ax.plot(contornoCV[:, 0][:, 0], contornoCV[:, 0][:, 1], '-b', linewidth=2)
            circulo = mpatches.Circle((x, y), raio, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(circulo)
    plt.show()

    """DESENHANDO HISTOGRAMA"""
    plt.figure()
    plt.title("H Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 179])
    plt.show()
    """TEXTURA:KURTOSIS & SKEWNESS"""
    texture_Kurt=kurtosis(gray_BoundingBox, axis=None)
    texture_Skew=skew(gray_BoundingBox, axis=None)
    print("\n---~ TEXTURE ~---")
    print("Kurtosis:",texture_Kurt,"\nSkewness:",texture_Skew)
    return  [x,y,raio,hist,texture_Kurt,texture_Skew]

if __name__ == '__main__':

    """PARÂMETROS"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path", required=True, help="path to the input image")
    args = vars(ap.parse_args())
    imageDir_val= args["path"]

    Atributos = []
    Atributos.append(TrataImagem(imageDir_val))







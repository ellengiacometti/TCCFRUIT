import cv2 as cv
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from heapq import nlargest
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import kurtosis, skew
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def TrataImagem(src,visual,verbose):
    """Author: Ellen Giacometti
        CRIADO EM: 21/12/2018
        ÚLTIMA ATUALIZAÇÃO: 05/02/2019
        DESC: Código que recebe uma imagem e extrai os Atributos do limão contido nela
        Args:
            visual (int): If visual = 1  will plot figures and histograms, if visual= 0 nothing is shown.
            verbose (int): It sets whether the function will print values extracted from image or not.Verbose=0 disable it
        Returns:
            x, y, raio: Values from size of the minimum enclosing circle, obtained from the biggest contour
            histH, histS, histV: Histograms of each channel of the given image. It one is a vector with 255 length
            texture_Kurt, texture_Skew: Float values of Kurtosis and Skewness
            dissimilarity,correlation,homogeneity,energy,contrast,ASM: GLCM  properties"""

    """ LENDO IMAGEM """
    # Lendo Imagem
    img = cv.imread(src)
    # Convertendo canal HSV
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Separando o canal de saturação
    h, s, v = cv.split(imgHSV)
    """ EXIBINDO CANAIS """
    if visual == 1:
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(h, cmap=plt.cm.gray)
        ax1.set_title(' h IMAGE')
        ax2.axis('off')
        ax2.imshow(s, cmap=plt.cm.gray)
        ax2.set_title('s IMAGE')
        ax3.axis('off')
        ax3.imshow(v, cmap=plt.cm.gray)
        ax3.set_title('v  IMAGE')
        plt.show()
    """ PROCESSAMENTO DA IMAGEM """
     # Filtro para borrar
    s_Blur = cv.GaussianBlur(s,(5,5),0)
    #s_Blur = cv.blur(s,(5,5))
    # Binarizando a imagem
    _, s_Thresh = cv.threshold(s_Blur,50,255,cv.THRESH_BINARY)
    # Morfologia tamanho do elemento estrutural
    block_size = 30
    kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE, (block_size, block_size))
    #kernel = np.ones((block_size, block_size), np.uint8)
    # Executando Dilation e Closing
    s_Closing = cv.morphologyEx(s_Thresh, cv.MORPH_CLOSE, kernel)
    # Resultado da Máscara em RGB
    s_Result = cv.bitwise_and(cv.cvtColor(img, cv.COLOR_BGR2RGB), cv.cvtColor(img, cv.COLOR_BGR2RGB), mask=s_Closing)
    """ PRINTANDO IMAGENS DO PROCESSO   """
    if visual==1:
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
    _, contornosCV, _ = cv.findContours(s_Closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Adquirindo o número de contornos encontrados
    tamanho_contornoCV = len(contornosCV)
    # Declarando um array para armazenar o tamanho de cada contorno
    tamanhoCV = np.empty([tamanho_contornoCV])
    for i in range(tamanho_contornoCV):
        # Adquirindo o tamanho de cada contorno encontrado
        tamanhoCV[i] = contornosCV[i].shape[0]
    # Detectando os N maiores contornos e seus indices, alterando N , os N maiores contornos serão armazenados.
    maioresCV = nlargest(1, enumerate(tamanhoCV), key=lambda a: a[1])

    """ MEDIDAS LIMÃO - CONTORNO ÚTIL   """
    contorno_util=contornosCV[maioresCV[0][0]]
    M = cv.moments(contorno_util)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    """ MEDIDAS CIRCUNFERÊNCIA CIRCUNSCRITA """
    ((x, y), raio) = cv.minEnclosingCircle(contorno_util)
    centroide = (int(x), int(y))
    """ HISTOGRAMA DO CANAL H   """
    histH = cv.calcHist([h], [0], s_Closing, [256], [0, 256])
    histH = list(map(float, histH[0:255][:]))
    histS = cv.calcHist([s], [0], s_Closing, [256], [0, 256])
    histS = list(map(float, histS[0:255][:]))
    histV = cv.calcHist([v], [0], s_Closing, [256], [0, 256])
    histV = list(map(float, histV[0:255][:]))

    """ TEXTURA:KURTOSIS & SKEWNESS """
    texture_Kurt = kurtosis(gray_BoundingBox, axis=None)
    texture_Skew = skew(gray_BoundingBox, axis=None)
    """TEXTURA: GLCM"""
    glcm = greycomatrix(gray_BoundingBox, [5], [0], 256, symmetric=True, normed=True)
    dissimilarity= greycoprops(glcm, 'dissimilarity')[0, 0]
    correlation= greycoprops(glcm, 'correlation')[0, 0]
    homogeneity = greycoprops(glcm,'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    contrast= greycoprops(glcm, 'contrast')[0, 0]
    ASM = greycoprops(glcm,'ASM')[0,0]

    """DEBUG VERSION """
    if(visual==1):
        """DESENHANDO O CONTORNO E A CIRCUNFERÊNCIA"""
        fig2, ax = plt.subplots()
        ax.imshow(s_Result, interpolation='nearest', cmap=plt.cm.gray)
        for n, contornoCV in enumerate(contornosCV):
            if (n in (np.transpose(np.asanyarray(maioresCV))[:])):
                ax.plot(contornoCV[:, 0][:, 0], contornoCV[:, 0][:, 1], '-b', linewidth=2)
                circulo = mpatches.Circle((x, y), raio, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(circulo)

                pixelsPerMetric = 4.75
                # compute the rotated bounding box of the contour
                box = cv.minAreaRect(contorno_util)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw lines between the midpoints
                ax.add_line(mlines.Line2D([int(x), int(y)], [int(x), int(y+raio)], color='yellow'))
                # ax.add_line(mlines.Line2D([int(tltrX),int(blbrX)], [int(tltrY),int(blbrY)],color='white'))
                # ax.add_line(mlines.Line2D([int(tlblX), int(trbrX)], [int(tlblY), int(trbrY)],color='white'))

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # compute the size of the object
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric

                # draw the object sizes on the image
                plt.scatter(x,y,c='yellow',s=10)
                # plt.text(int(tltrX - 50), int(tltrY - 30), "{:.1f}mm".format(dimA),color='white')
                # plt.text(int(trbrX + 50), int(trbrY+10), "{:.1f}mm".format(dimB), color='white')
                plt.text(int(x-15), int(y+raio +30), "{:.1f}mm".format(raio/pixelsPerMetric), color='yellow')
        plt.show()

        """DESENHANDO HISTOGRAMA"""
        plt.figure()
        csfont = {'fontname': 'Times New Roman'}
        plt.title("H Histogram")
        plt.xlabel("Níveis de cinza",**csfont,fontsize=12)
        plt.ylabel("Número de Pixels",**csfont,fontsize=12)
        plt.plot(histH)
        plt.xlim([0, 256])
        plt.show()
        plt.figure()
        plt.title("S Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(histS)
        plt.xlim([0, 256])
        plt.show()
        plt.figure()
        plt.title("V Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(histV)
        plt.xlim([0, 256])
        plt.show()

    """GERANDO RELATÓRIO """
    if verbose==1:

        print("\n---~ INFORMAÇÕES - CONTORNO DO LIMÃO ~---")
        print("Número de Contornos Encontrados na Imagem:", tamanho_contornoCV)
        print("Perímetro[cm]:", maioresCV[0][1]/pixelsPerMetric)
        print("Dimensão[cm]:", "(",dimA, ",",dimB,")" , "\nCentro:", centroide)
        print("Centróide:(", cx, ",", cy, ")")
        print("\n---~ SIZE & SHAPE - CIRCUNFERÊNCIA ~---")
        print("Raio [cm]:", raio/pixelsPerMetric, "\nCentro:", centroide)
        print("\n---~ TEXTURE - KURTOSIS SKEWNESS~---")
        print("Kurtosis:",texture_Kurt,"\nSkewness:",texture_Skew)
        print("\n---~ TEXTURE - GLCM~---")
        print("Dissimilarity:",dissimilarity)
        print("Correlation:",correlation)
        print("Homogeneity:", homogeneity)
        print("Energy:", energy)
        print("Contrast:", contrast)
        print("ASM:", ASM)
    pixelsPerMetric = 47.5
    radius = raio / pixelsPerMetric
    return [x, y, radius, histH, histS, histV, texture_Kurt, texture_Skew,dissimilarity,correlation,homogeneity,energy,contrast,ASM]










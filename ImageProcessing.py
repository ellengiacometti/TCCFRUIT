import cv2 as cv
from heapq import nlargest
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
from skimage import feature,morphology

"""CAPTURA DE  IMAGEM"""
# Li a Imagem
src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM/LM_16_1.jpg"
# Imagem Crua
imgRaw = cv.imread(src)
# Imagem em RGB
imRGB= cv.cvtColor(imgRaw, cv.COLOR_BGR2RGB)
# Imagem em cinza
im = cv.cvtColor(imgRaw, cv.COLOR_BGR2GRAY)

"""PROCESSANDO IMAGEM"""
# Filtro Canny
Canny = feature.canny(im, sigma=0.1)
# Definindo parêmetro selem referente a conectividade
forma_dilation= morphology.disk(1)
# Morfologia dilation remover rachaduras
Canny_Dilation= morphology.dilation(Canny,forma_dilation)
# Definindo parêmetro selem referente a conectividade
forma_closing= morphology.disk(2)
# Fazendo outro Dilation, seguido de um Erosion para fechar as rachaduras pretas
Canny_Closing=morphology.binary_closing(Canny_Dilation,forma_closing)
# Tornando  a foto de false para 0
Canny_Closing=(Canny_Closing * 1.0).astype(np.float32)
# Conectando imagem
block_size = 28
kernel = np.ones((block_size, block_size), np.uint8)
Canny_Connectado = cv.morphologyEx(Canny_Closing, cv.MORPH_CLOSE, kernel)
# Inserindo Labels
Canny_Labeled, label_num, = sm.label(Canny_Connectado, return_num=1, connectivity=1)
print("\nLabels Encontrados Imagem Final:", label_num, "\n")
if (label_num>50):
    # Definindo parêmetro 'selem' referente a conectividade
    forma_dilation = morphology.disk(2)
    # Morfologia dilation remover rachaduras
    Canny_Dilation_Fix = morphology.dilation(Canny_Closing, forma_dilation)
    # Definindo parêmetro 'selem' referente a conectividade
    forma_closing = morphology.disk(2)
    # Fazendo outro Dilation, seguido de um Erosion para fechar as rachaduras pretas
    Canny_Closing_Fix = morphology.binary_closing(Canny_Dilation_Fix, forma_closing)
    # Tornando  a foto de false para 0
    Canny_Closing_Fix = (Canny_Closing_Fix * 1.0).astype(np.float32)
    # Conectando imagem
    block_size = 28
    kernel = np.ones((block_size, block_size), np.uint8)
    Canny_Connectado_Fix = cv.morphologyEx(Canny_Closing_Fix, cv.MORPH_CLOSE, kernel)
    Canny_Labeled, label_num, = sm.label(Canny_Connectado_Fix, return_num=1, connectivity=1)
    print("\nLabels Encontrados Imagem Reprocessada:", label_num, "\n")

"""EXIBINDO AS FOTOS"""
fig, ([[ax1, ax2,ax3],[ax4,ax5,ax6]]) = plt.subplots(2, 3,sharex=True ,sharey=True)
ax1.axis('off')
ax1.imshow(imRGB)
ax1.set_title('Image RGB')

ax2.axis('off')
ax2.imshow(Canny, cmap=plt.cm.gray)
ax2.set_title('Canny Filter')

ax3.axis('off')
ax3.imshow(Canny_Dilation, cmap=plt.cm.gray)
ax3.set_title('Canny_Dilatation')

ax4.axis('off')
ax4.imshow(Canny_Closing, cmap=plt.cm.gray)
ax4.set_title(' Canny_Closing ')
#
ax5.axis('off')
ax5.imshow(Canny_Connectado, cmap=plt.cm.gray)
ax5.set_title('Canny_Connectado')

ax6.axis('off')
ax6.imshow(Canny_Labeled, cmap=plt.cm.gray)
ax6.set_title('Labeled ')
plt.show()

"""SCIKIT - DEFININDO CONTORNOS"""
contornos = sm.find_contours(Canny_Labeled,0.19,fully_connected='low')
tamanho_contorno =len(contornos)
print("Número Contornos Encontrados:", len(contornos))

#Criando array  com o número de pontos de cada contorno
tamanho = np.empty([tamanho_contorno])
for i in range(tamanho_contorno):
    tamanho[i]=contornos[i].shape[0]

#Detectando os N maiores contornos e seus indices, alterando N , os N maiores contornos serão desenhados
maiores=nlargest(1,enumerate(tamanho),key=lambda x: x[1])
print("\nMaiores:", maiores)

"""ANÁLISE DE AREA"""
#Convertendo o valor  dado pelo scikit find contour para funcionar com a função de achar área do contorno do OpenCV
Area = cv.contourArea(np.around(np.array([[pt] for pt in contornos[maiores[0][0]]]).astype(np.int32)))
print("\nArea:", Area)
#***IMPORTANTE: Na análise feita cortando as fotos o valor da área muda bem pouco.Essa mudança apenas ocorre pois o código pega um contorno levemente diferente.
#***IMPORTANTE: No entanto se for feito RESIZE o valor MUDA TOTALMENTE![Creio que na mesma proporção que foi feito o RESIZE]

"""PLOTANDO OS CONTORNOS ENCONTRADOS"""
fig1, ax = plt.subplots()
ax.imshow(imRGB, interpolation='nearest', cmap=plt.cm.gray)
for  n, contorno in enumerate(contornos):
    if(n in (np.transpose(np.asanyarray(maiores))[0])):
        ax.plot(contorno[:, 1], contorno[:, 0],'-b', linewidth=2)
        print("\nCONTORNO DESENHADO:", len(contorno))
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()











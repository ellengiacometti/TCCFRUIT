import cv2 as cv
import argparse
from heapq import nlargest
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
from skimage import feature
import matplotlib.patches as mpatches

"""PROCESSANDO IMAGEM"""
#Imagem Crua
src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM/LM_1.jpg"
#Li a Imagem
imgRaw = cv.imread(src)
imRGB= cv.cvtColor(imgRaw, cv.COLOR_BGR2RGB)
im = cv.cvtColor(imRGB, cv.COLOR_BGR2GRAY)
Canny = feature.canny(im, sigma=0.1)

## Tornando  a foto de false para 0
Canny=(Canny * 1.0).astype(np.float32)

##Conectando imagem
block_size = 28
kernel = np.ones((block_size, block_size), np.uint8)
cv_thresh_Mo = cv.morphologyEx(Canny, cv.MORPH_CLOSE, kernel)

##Borrando
blur = cv.blur(cv_thresh_Mo, (21, 21))

##Conectando imagem
block_size = 28
kernel = np.ones((block_size, block_size), np.uint8)
cv_thresh_Mo = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)

##Inserindo Labels
cv_thresh_La, label_num, = sm.label(blur, return_num=1, connectivity=2)
print("\nLabels Encontrados:", label_num, "\n")

"""EXIBINDO AS FOTOS"""
fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5)
# fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, figsize=(80, 80))
ax1.axis('off')
ax1.imshow(imRGB, cmap=plt.cm.gray)
ax1.set_title('Image RGB')

ax2.axis('off')
ax2.imshow(Canny, cmap=plt.cm.gray)
ax2.set_title('Canny Filter')

ax3.axis('off')
ax3.imshow(cv_thresh_Mo, cmap=plt.cm.gray)
ax3.set_title('Connected')

ax4.imshow(blur, cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title(' Blurred ')

ax5.imshow(cv_thresh_La, cmap=plt.cm.gray)
ax5.axis('off')
ax5.set_title('Labeled ')
plt.show()

"""DEFININDO CONTORNOS"""
contornos = sm.find_contours(blur,0.19,fully_connected='low')
tamanho_contorno =len(contornos)
print("Número Contornos Encontrados:", len(contornos))

#Criando array  com o número de pontos de cada contorno
tamanho = np.empty([tamanho_contorno])
for i in range(tamanho_contorno):
    tamanho[i]=contornos[i].shape[0]

# Detectando os maiores contornos e seus indices
maiores=nlargest(2,enumerate(tamanho),key=lambda x: x[1])
print("\nMaiores:", maiores)

"""PLOTANDO OS CONTORNOS ENCONTRADOS"""
# fig1, ax = plt.subplots()
# ax.imshow(imRGB, interpolation='nearest', cmap=plt.cm.gray)
# for  n, contorno in enumerate(contornos):
#     print("Número de pontos no Contornos",n+1,":", len(contorno))
#     if(len(contorno) >=1500):
#         ax.plot(contorno[:, 1], contorno[:, 0],'-b', linewidth=2)
#         print("\nCONTORNO DESENHADO:", len(contorno))
# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
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











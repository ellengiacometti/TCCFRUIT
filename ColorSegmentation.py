import cv2 as cv
from heapq import nlargest
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


"""CAPTURA DE  IMAGEM"""
# Li a Imagem
src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM/LM_1.jpg"
# Imagem Crua
image = cv.imread(src)

""" TRATAMENTO DE CORES"""

# Convertendo a imagem para HSV
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
h,s,v = cv.split(hsv_image)
# Definindo o range para VERDE
light_green = np.array([55, 45, 40])
dark_green = np.array([110, 255, 255])
# # Definindo o range para AMARELO
light_yellow = np.array([20, 90, 100])
dark_yellow = np.array([40, 255, 255])
# Definindo o range para MARROM
light_brown = np.array([0,50,50])
dark_brown = np.array([45,255,255])
# Criando Máscara para VERDE
maskGreen = cv.inRange(hsv_image, light_green, dark_green)
# Criando Máscara para AMARELO
maskYellow = cv.inRange(hsv_image, light_yellow, dark_yellow)
# Criando Máscara para MARROM
maskBrown = cv.inRange(hsv_image, light_brown, dark_brown)
# Definindo o range para BRANCO
# light_white = (0, 0, 200)
# dark_white = (145, 60, 255)
#
# # Criando uma possível Máscara pro BRANCO
# mask_white = cv2.inRange(hsv_image, light_white, dark_white)

# Criando uma unica mascara com todas as cores
mask = maskGreen + maskYellow + maskBrown
# Combinando a mascara com a imagem original
result = cv.bitwise_and(cv.cvtColor(image,cv.COLOR_BGR2RGB), cv.cvtColor(image,cv.COLOR_BGR2RGB), mask=mask)
# Convertendo para Cinza
gray=cv.cvtColor(result,cv.COLOR_RGB2GRAY)

"""EXIBINDO AS IMAGENS"""
fig, (ax1, ax2,ax4) = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
ax1.set_title(' RGB IMAGE')
ax2.axis('off')
ax2.imshow(hsv_image, cmap=plt.cm.gray)
ax2.set_title('HSV IMAGE')
ax4.axis('off')
ax4.imshow(gray, cmap=plt.cm.gray)
ax4.set_title('RESULT GRAY IMAGE')
plt.show()

"""CONTORNO SCIKIT"""
contornos = sm.find_contours(gray,0.19,fully_connected='low')
tamanho_contorno =len(contornos)
print("Número de Contornos Encontrados:",tamanho_contorno)
tamanho = np.empty([tamanho_contorno])
for i in range(tamanho_contorno):
    tamanho[i]=contornos[i].shape[0]
#Detectando os N maiores contornos e seus indices, alterando N , os N maiores contornos serão desenhados
maiores=nlargest(1,enumerate(tamanho),key=lambda a: a[1])
print("MaiorContorno:", maiores)

fig1, ax = plt.subplots()
ax.imshow(result, interpolation='nearest', cmap=plt.cm.gray)
for  n, contorno in enumerate(contornos):
    if(n in (np.transpose(np.asanyarray(maiores))[0])):
        ax.plot(contorno[:, 1], contorno[:, 0],'-b', linewidth=2)
plt.show()



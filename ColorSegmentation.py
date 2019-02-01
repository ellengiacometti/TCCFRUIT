import cv2 as cv
from heapq import nlargest
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mahotas as mt
from skimage import feature,morphology

''' Attempts to segment the clownfish out of the provided image '''
# Li a Imagem
src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM_CROP/LM-113.jpg"
# Imagem Crua
image = cv.imread(src)
# Convertendo a imagem para HSV
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
# Definindo o range para VERDE
light_green = np.array([55,45,40])
dark_green = np.array([110,255,255])
# # Definindo o range para AMARELO
light_yellow = np.array([20,90,100])
dark_yellow = np.array([40,255,255])
# Definindo o range para MARROM
light_brown = np.array([0,50,50])
dark_brown = np.array([45,255,255])

# Criando Máscara para VERDE
maskGreen = cv.inRange(hsv_image, light_green, dark_green)
# Criando Máscara para AMARELO
maskYellow = cv.inRange(hsv_image, light_yellow, dark_yellow)
# Criando Máscara para MARROM
maskBrown = cv.inRange(hsv_image, light_brown, dark_brown)

# # Set a white range
# light_white = (0, 0, 200)
# dark_white = (145, 60, 255)
#
# # Apply the white mask
# mask_white = cv2.inRange(hsv_image, light_white, dark_white)
#

# Combine the two masks
mask = maskGreen + maskYellow + maskBrown

result = cv.bitwise_and(cv.cvtColor(image,cv.COLOR_BGR2RGB), cv.cvtColor(image,cv.COLOR_BGR2RGB), mask=mask)

# Clean up the segmentation using a blur
blur = cv.GaussianBlur(result, (7, 7), 0)
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
ax1.set_title(' RGB IMAGE')
ax2.axis('off')
ax2.imshow(hsv_image, cmap=plt.cm.gray)
ax2.set_title('HSV IMAGE')
ax3.axis('off')
ax3.imshow(blur, cmap=plt.cm.gray)
ax3.set_title('RESULT IMAGE')
plt.show()

 contornos = sm.find_contours(result>0.01,0.19,fully_connected='low')
 tamanho_contorno =len(contornos)
 print("Número de Contornos Encontrados:", len(contornos))
import cv2 as cv
import argparse
import heapq
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
from skimage import feature
import matplotlib.patches as mpatches


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
cv_thresh_La, label_num, = sm.label(cv_thresh_Mo, return_num=1, connectivity=2)
print("\nLabels Encontrados:", label_num, "\n")

### EXIBINDO AS FOTOS###

fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, figsize=(80, 80))
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


contornos = sm.find_contours(blur,0.2,fully_connected='high')

fig1, ax = plt.subplots()
ax.imshow(imRGB, interpolation='nearest', cmap=plt.cm.gray)
# print("\nContornos Encontrados:", len(contornos), "\n")

for  n, contorno in enumerate(contornos):
    coord = sm.approximate_polygon(contorno, tolerance=40)
    if(len(coord)>=6 and len(contorno) >=1500 and len(contorno)<=4000):
        ax.plot(contorno[:, 1], contorno[:, 0],'-b', linewidth=2)
        # input()
        # plt.show()
        # ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)

    print("Contornos Encontrados:", len(contorno), len(coord))
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
"""CONTINUE ESSA LÓGICA"""
#Preciso pegar a variável  list contornos e transformar em algo ,
# de forma que eu possa acessar os ndarrays contidos nela e selecionar os 2 maiores
# contornos= np.array(contornos)
# heapq.nlargest(2, contornos)








# def segment_fish(image):
#     ''' Attempts to segment the clownfish out of the provided image '''
#
#     # Convert the image into HSV
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#
#     # Set the orange range
#     light_orange = (1, 190, 200)
#     dark_orange = (18, 255, 255)
#
#     # Apply the orange mask
#     mask = cv2.inRange(hsv_image, light_orange, dark_orange)
#
#     # Set a white range
#     light_white = (0, 0, 200)
#     dark_white = (145, 60, 255)
#
#     # Apply the white mask
#     mask_white = cv2.inRange(hsv_image, light_white, dark_white)
#
#     # Combine the two masks
#     final_mask = mask + mask_white
#     result = cv2.bitwise_and(image, image, mask=final_mask)
#
#     # Clean up the segmentation using a blur
#     blur = cv2.GaussianBlur(result, (7, 7), 0)
#     return blur
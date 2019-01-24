import cv2 as cv
import argparse
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#Imagem Crua
src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM/LM_2.jpg"
imgRaw = cv.imread(src)
imRGB= cv.cvtColor(imgRaw, cv.COLOR_BGR2RGB)
im = cv.cvtColor(imRGB, cv.COLOR_BGR2GRAY)

Sobel = sobel(im)
Canny = feature.canny(im, sigma=0)
Prewitt = prewitt(im)


# display results
fig, (ax1, ax2) = plt.(1,1)

ax1.imshow(imRGB, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Image RGB', fontsize=20)

ax2.imshow(Canny, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny Filter', fontsize=20)
fig.tight_layout()

plt.show()

Canny=(Canny * 1.0).astype(np.float32)
block_size = 12
kernel = np.ones((block_size, block_size), np.uint8)
cv_thresh_Mo = cv.morphologyEx(Canny, cv.MORPH_CLOSE, kernel)
cv_thresh_La, label_num, = sm.label(cv_thresh_Mo, return_num=1, connectivity=1)
print("\nLabels Encontrados:", label_num, "\n")


fig1, ( ax3, ax4) = plt.subplots(1,2)
ax3.imshow(cv_thresh_Mo, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('cv_thresh_Mo ',  fontsize=20)

ax4.imshow(cv_thresh_La, cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title('cv_thresh_La ',  fontsize=20)

fig1.tight_layout()
plt.show()










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
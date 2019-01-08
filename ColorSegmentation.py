import cv2 as cv2
import argparse
import numpy as np
import skimage.measure as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#Imagem Crua
src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM/LM_2.jpg"
imgRaw = cv2.imread(src)
# plt.imshow(imgRaw)
# plt.show()
#Imagem RGB
imgRGB = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)
# plt.imshow(imgRGB)
# plt.show()

r, g, b = cv2.split(imgRGB)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = imgRGB.reshape((np.shape(imgRGB)[0]*np.shape(imgRGB)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
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
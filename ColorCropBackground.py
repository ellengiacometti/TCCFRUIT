import numpy as np
import imutils
import skimage.measure as sm
import cv2
import matplotlib.pyplot as plt

src = "/home/ellengiacometti/PycharmProjects/TCCFRUIT/PIC_LM/LM_2.jpg"
image= cv2.imread(src)
resized = cv2.resize(image,(600,600))
##Faixa da cor a ser cortada
# remover
# Faixa de Cinza - 119 132 108 - 120 120 120
boundaries = [([0,72,0], [119,132,108])]
boundaries1 = [([120,120,120], [178,255,175])]
linha= []
coluna = []
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask1 = cv2.inRange(resized, lower, upper)

for (baixo, cima) in boundaries1:
    # create NumPy arrays from the boundaries
    baixo = np.array(baixo, dtype="uint8")
    cima = np.array(cima, dtype="uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask2 = cv2.inRange(resized, baixo, cima)
             #Visualiza a Máscara em cima da foto real
mask=mask1+mask2
output = cv2.bitwise_and(resized, resized, mask=mask)

fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(50, 50), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(resized, cmap=plt.cm.gray)
ax1.set_title('Resized')
ax2.axis('off')
ax2.imshow(mask, cmap=plt.cm.gray)
ax2.set_title('Color Filter')
ax3.axis('off')
ax3.imshow(output, cmap=plt.cm.gray)
ax3.set_title('Foto Marcada com as cores')
plt.show()

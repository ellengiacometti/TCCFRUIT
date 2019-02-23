import os
import glob
import cv2 as cv
import scipy.misc as scm
from skimage.io import imsave



def DataAugmentation(srcDirOr, srcDirEnd, type,angle,size,direction,dB):
    for name in os.listdir(srcDirOr):
        cur_path = srcDirOr + "/" + name
        for file in glob.glob(cur_path):
            if (type == 1 or type == 0):
                image=cv.imread(file)
                imageRotate = Rotate(image, angle)
                savePath = srcDirEnd + "/ROTATED/" + name
                imsave(savePath, imageRotate)
            elif(type == 2 or type == 0):
                image = cv.imread(file)
                imageResize_cv,imageResize_sc = Resize(image, size)
                savePath = srcDirEnd + "/RESIZED/" + name
                imsave(savePath, imageResize_cv)
                imsave(savePath, imageResize_sc)
            elif(type == 3 or type == 0):
                image = cv.imread(file)
                imageFlip = Flip(image, direction)
                savePath = srcDirEnd + "/FLIPPED/" + name
                imsave(savePath, imageFlip)
            elif(type == 4 or type == 0):
                image = cv.imread(file)
                imageNoise = Noise(image, dB)
                savePath = srcDirEnd + "/NOISY/" + name
                imsave(savePath, imageNoise)



def Rotate(image, angle):
    img=scm.imrotate(image, angle)
    return img

def Resize(image, size):
    resized_cv = cv.resize(image,  size)
    resized_sc = scm.imresize(image, size)
    return resized_cv,resized_sc

def Flip(image, direction):
    if (direction.lower()=='horizontal'):
        img = cv.flip(image, 0)
    elif(direction.lower()=='vertical'):
        img = cv.flip(image, 0)
    return img
def Noise(image, dB):
    return img



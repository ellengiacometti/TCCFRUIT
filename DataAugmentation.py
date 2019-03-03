import os
import glob
import cv2 as cv
import scipy.misc as scm
import skimage.util as su
from skimage.io import imsave



def DataAugmentation(srcDirOr, srcDirEnd, type,initialid=3999,angle=0,size=0,direction='Vertical',dB=0):
    """Author: Ellen Giacometti
    CRIADO EM: 23/02/2019
    ÚLTIMA ATUALIZAÇÃO: 27/03/2019
    DESC: Do a referred data augmentation function on images of a given path folder.
    Args:
        srcDirOr (str): Path where original images are saved .
        srcDirEnd (str): Path where images,after data augmentation, will be saved
        type (int): Receives the  code which defines de function in which the data augmentation will be based.
                    1 = Rotate # 2 = Resize # 3 = Flip # 4 = Gaussian Noise
        angle (int): A int number between 0 ~ 360, only necessary if type = 1 or type = 0. It's the angle in which images are going to be rotated
        size (tuple):Insert the tuple in which you want to resize your picture.
        direction (str): Choose between 'Vertical' or 'Horizontal'  to flip an image. Only required if type = 3 or type = 0.
        dB (float): The amount of noise apply in the image. Only required if type = 4 or type = 0

    Returns:
        Create folders in the path specified at srcDirEnd .
"""

    listId=([int(name[:4]) for name in os.listdir(srcDirOr)])
    numberimages= len(os.listdir(srcDirOr))

    id=[initialid,initialid,initialid,initialid]
    if(type==0):
        id = [initialid, initialid+numberimages, initialid+ 2* numberimages, initialid+ 3* numberimages]
    for name in os.listdir(srcDirOr):
        cur_path = srcDirOr + "/" + name
        for file in glob.glob(cur_path):
            if (type == 1 or type == 0):
                image=cv.imread(file)
                imageRotate = Rotate(image, angle)
                dirname = srcDirEnd + "/1-ROTATED/"+ str(angle) + "°/"
                try:
                    os.makedirs(dirname)
                except:
                    print("Pasta Já Existe!")
                finally:
                    id[0]+=1
                    finalname=str(id[0]) + '_'+name[5:]
                    savePath = dirname + finalname
                    cv.imwrite(savePath,imageRotate)
                    print("Imagem ",finalname," salva em ","/1-ROTATED/")

            if(type == 2 or type == 0):
                image = cv.imread(file)
                imageResize_sc = Resize(image, size)
                dirname = srcDirEnd + "/2-RESIZED/"+ str(size) + "/"
                try:
                    os.makedirs(dirname)
                except:
                    print("Pasta Já Existe!")
                finally:
                    id[1]+=1
                    finalname = str(id[1]) +'_'+ name[5:]
                    savePath = dirname + finalname
                    cv.imwrite(savePath, imageResize_sc)
                    print("Imagem ", finalname, " salva em ", "/2-RESIZED/")

            if(type == 3 or type == 0):
                image = cv.imread(file)
                imageFlip = Flip(image, direction)
                dirname = srcDirEnd + "/3-FLIPPED/" + direction +"/"
                try:
                    os.makedirs(dirname)
                except:
                    print("Pasta Já Existe!")
                finally:
                    id[2]+=1
                    finalname = str(id[2]) + '_' +name[5:]
                    savePath = dirname + finalname
                    cv.imwrite(savePath, imageFlip)
                    print("Imagem ", finalname, " salva em ", "/3-FLIPPED/")

            if(type == 4 or type == 0):
                image = cv.imread(file)
                imageNoise = Noise(image, dB)
                dirname = srcDirEnd + "/4-NOISY/"+ str(dB)+"dB/"
                try:
                    os.makedirs(dirname)
                except:
                    print("Pasta Já Existe!")
                finally:
                    id[3]+=1
                    finalname = str(id[3]) + '_'+ name[5:]
                    savePath = dirname + finalname
                    imsave(savePath,imageNoise)
                    print("Imagem ", finalname, " salva em ", "/4-NOISY/")




def Rotate(image, angle):
    img=scm.imrotate(image, angle)
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img

def Resize(image, size):
    # resized_cv = cv.resize(image,  size)
    resized_sc = scm.imresize(image, size)
    # cv.imshow('image', resized_sc)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # # return resized_cv,resized_sc
    return resized_sc

def Flip(image, direction):
    if (direction.lower()=='horizontal'):
        img = cv.flip(image, 0)
    elif(direction.lower()=='vertical'):
        img = cv.flip(image, 0)
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img
def Noise(image, dB):
    img = su.random_noise(image,mode='gaussian')
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img



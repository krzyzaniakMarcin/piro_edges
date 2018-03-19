import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
import matplotlib as mpl
import scipy.ndimage as ndimage
from skimage import data, io, filters

def removeBlankRows(image):
    x_begin = -1
    x_end = -1

    for i in range(len(image)):
        if x_begin < 0:
            if any(image[i]) :
                x_begin = i
        elif not any(image[i]):
            x_end = i
            break
    image = np.delete(image,range(x_begin),0)
    image = np.delete(image,range(x_end-x_begin,len(image)),0)
    return image

def removeBlankRowsAndColumns(image):
    image = removeBlankRows(image)
    image = np.rot90(image, 3)
    image = removeBlankRows(image)
    return image

def getPointsOnEdges(image):
    result = []
    for i in range(len(image[0])):
        if image[0][i]:
            result.append([i,0])
    for i in range(len(image[0])):
        if image[len(image)-1][i]:
            result.append([i,len(image)])
    image = np.rot90(image, 3)
    for i in range(len(image[0])):
        if image[0][i]:
            result.append([0,len(image[0])-i])
    for i in range(len(image[0])):
        if image[len(image)-1][i]:
            result.append([len(image),len(image[0])-i])
    return result

# get edges 
img = io.imread('sets/set7/0.png')>127
img_dilatation = ndimage.binary_dilation(img)
diff = img_dilatation != img

#remove blank lines from image
diff = removeBlankRowsAndColumns(diff)

#get points on image edges
points = getPointsOnEdges(diff)


diff = diff.astype(np.int8)

io.imshow(diff)
io.show()
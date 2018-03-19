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

def getSquareIndicies((x, y), radius, height, width):
    result = []
    if y-radius>=0:
        for i in range(max(x-radius,0), min(x+radius,height-1)+1):
            result.append((i,y-radius))

    if x+radius<height:
        for i in range(max(y-radius,0) + 1, min(y+radius,width-1)):
            result.append((x+radius,i))

    if y+radius<width:
        for i in reversed(range(max(x-radius,0), min(x+radius,height-1)+1)):
            result.append((i,y+radius))

    if x-radius>=0:
            for i in reversed(range(max(y-radius,0) + 1, min(y+radius,width-1))):
                result.append((x-radius,i))
    return result

def getIntersectionWithSquare(image, (x,y), radius):
    found = False

    square = getSquareIndicies((x,y), radius, len(image[0]), len(image))
    if image[square[0]] and image[square[-1]]:
        del square[-1]

    for i in square:
        if image[i] and not found:
            found = True
            print(i)
        else:
            found = False


# get edges 
img = io.imread('sets/set7/12.png')>127
img_dilatation = ndimage.binary_dilation(img)
diff = img_dilatation != img

#remove blank lines from image
diff = removeBlankRowsAndColumns(diff)

#get points on image edges
points = getPointsOnEdges(diff)
print points

diff = diff.astype(np.int8)
getIntersectionWithSquare(diff, (69, 0), 20)
io.imshow(diff)
io.show()


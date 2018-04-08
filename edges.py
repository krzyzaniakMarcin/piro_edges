import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
from scipy.misc import toimage
import cv2
import matplotlib as mpl
import scipy.ndimage as ndimage
from scipy.spatial import ConvexHull
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
            result.append((0, i))
        if image[len(image)-1][i]:
            result.append((len(image)-1, i))

    for i in range(len(image)):
        if image[i][0]:
            result.append((i,0))
        if image[i][len(image[i])-1]:
            result.append((i,len(image[i])-1))
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

    square = getSquareIndicies((x,y), radius, len(image), len(image[0]))
    if image[square[0]] and image[square[-1]]:
        del square[-1]
    result = []
    for i in square:
        if image[i] and not found:
            found = True
            result.append(i)
        else:
            found = False
    return result

#angle between vertex(x1,y1) and vertex(x2,y2) in vertex(x,y)
def calculateAngle((x,y),(x1,y1),(x2,y2)):
    A = (x2 - x, y2 - y)
    B = (x1 - x, y1 - y)
    C = (x2 - x1, y2 - y1)

    num = np.dot(A,B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if abs(num/denom) > 1:
        return 180.0
    return np.arccos(num/denom)*180 / np.pi

def checkIfPointsAreConnected((x1,y1),(x2,y2),img):
    number =  distance((x1,y1),(x2,y2))
    for i in range(1,number+1):
        x = x1 + (x2-x1)*i/(number+1)
        y = y1 + (y2-y1)*i/(number+1)
        if len(getIntersectionWithSquare(img,(x,y),2))==0:
            return False
    return True

def distance((x1, y1), (x2, y2)):
    return int(np.sqrt((x1-x2)**2 + (y1-y2)**2))

def getAngle((x,y), radious, when_no_angle = 0):
    intersection = getIntersectionWithSquare(diff, (x, y), radious)
    if len(intersection) == 2:
        return calculateAngle((x,y), intersection[0], intersection[1])
    else:
        return when_no_angle

# get edges 
def revertTransformation(img, transformation):
    rgbArray = np.zeros((len(img),len(img[0]),3), 'uint8')
    rgbArray[..., 0] =img*255
    rgbArray[..., 1] = img*255
    rgbArray[..., 2] = img*255
    return cv2.warpPerspective(rgbArray,transformation,(500,600))

def read_img(num):
    img = io.imread('sets/set7/' + str(num) + '.png')>127
    img_dilatation = ndimage.binary_dilation(img)
    diff = img_dilatation != img
    diff = diff.astype(np.int8)

    points = []
    for x, row in enumerate(diff):
        for y, val in enumerate(row):
            if diff[x][y]:
                points.append((x, y))
    points = np.array(points)
    return diff, points


def get_hull():
    hull = points[ConvexHull(points).vertices]
    hull = list(map(tuple, hull))
    hull = list(filter(lambda x : getAngle(x, 15) < 170, hull))
    return hull

def get_right_angles(hull):
    possible = []
    for (x, y) in hull:
        if diff[x][y]:
            angle_big, angle_small = 0, 0

            angle_big = 90-abs(90-getAngle((x, y), 30))
            angle_small = 90-abs(90-getAngle((x, y), 10))

            if angle_small > 65 and angle_big > 65:
                possible.append((x, y))

    groups = {}
    for point in possible:
        added = False
        for k in groups:
            if distance(point, k) < 10:
                added = True
                groups[k].append(point)
                break
        if not added:
            groups[point] = [point]

    right_angles = [tab[len(tab)/2] for tab in groups.values()]
    return right_angles

def getConnected(right_angles):
    s = set()
    for i in range(len(right_angles) - 1):
        for j in range(i + 1, len(right_angles)):
            p1 = right_angles[i]
            p2 = right_angles[j]
            connected = checkIfPointsAreConnected(p1, p2, diff)
            print p1, p2, connected
            if connected:
                s.add(p1)
                s.add(p2)
    return list(s)

for i in [12]:
    diff, points = read_img(i)
    hull = get_hull()
    right_angles = get_right_angles(hull)
    to_show = np.copy(diff)

    for point in right_angles:
        to_show[point] = 3

    connected = getConnected(right_angles)
    for point in connected:
        to_show[point] = 4

    io.imshow(to_show)
    io.show()



    pts1 = np.float32([[204,434],[604,246],[476,34],[55,246]])
    pts2 = np.float32([[0,0],[0,300],[500,300],[500,0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)

    io.imshow(revertTransformation(diff,M))
    io.show()

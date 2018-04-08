import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
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
A = 0
B = 0
C = 0
num = 0
denom = 0
#angle between vertex(x1,y1) and vertex(x2,y2) in vertex(x,y)
def calculateAngle((x,y),(x1,y1),(x2,y2)):
    global A, B, C, num, denom
    A = (x2 - x, y2 - y)
    B = (x1 - x, y1 - y)
    C = (x2 - x1, y2 - y1)

    num = np.dot(A,B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if abs(num/denom) > 1:
        return 180.0
    return np.arccos(num/denom)*180 / np.pi

def distance((x1, y1), (x2, y2)):
    return int(np.sqrt((x1-x2)**2 + (y1-y2)**2))


# get edges 
img = io.imread('sets/set7/12.png')>127
img_dilatation = ndimage.binary_dilation(img)
diff = img_dilatation != img

#remove blank lines from image

diff = diff.astype(np.int8)

points = []
for x, row in enumerate(diff):
    for y, val in enumerate(row):
        if diff[x][y]:
            points.append((x, y))

points = np.array(points)

hull = points[ConvexHull(points).vertices]
hull = list(map(tuple, hull))
for i in hull:
    diff[i] = 2

to_show = np.copy(diff)

possible = []
for (x, y) in hull:
    if diff[x][y]:
        angle_big, angle_small = 0, 0

        intersection = getIntersectionWithSquare(diff, (x, y), 30)
        if len(intersection) == 2:
            angle_big = 90-abs(90-calculateAngle((x,y), intersection[0], intersection[1]))

        intersection = getIntersectionWithSquare(diff, (x, y), 10)
        if len(intersection) == 2:
            angle_small = 90-abs(90-calculateAngle((x,y), intersection[0], intersection[1]))

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

for point in right_angles:
    to_show[point] = 5

io.imshow(to_show)
io.show()

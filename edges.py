import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
import matplotlib as mpl
import scipy.ndimage as ndimage
from skimage import data, io, filters


img = io.imread('sets/set7/0.png')>127
img_dilatation = ndimage.binary_dilation(img)
diff = img_dilatation != img
diff = diff.astype(np.int8)
io.imshow(diff)
io.show()
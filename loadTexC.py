import numpy as np
import Image
import random
import cprint, time
import sys 
import scipy.ndimage

def border(data):
    r = 15
    r2 = 10
    r3 = 7
    r4 = 6
    b1 = scipy.ndimage.binary_closing(data,structure=np.ones((r,r,r))).astype(np.uint8)
    b1 = scipy.ndimage.binary_dilation(b1,structure=np.ones((r2,r2,r2))).astype(np.uint8)
    b = scipy.ndimage.binary_erosion(b1,structure=np.ones((r3,r3,r3))).astype(np.uint8)
    #c = b1-b
    c = scipy.ndimage.binary_closing(b1-b,structure=np.ones((r4,r4,r4))).astype(np.uint8)
    c = scipy.ndimage.binary_dilation(c,structure=np.ones((r4,r4,r4))).astype(np.uint8)
    return np.array(255*(c)).astype(np.uint8)

I = Image.open(sys.argv[1])
dim = I.size[0]
Nz = I.size[1]/I.size[0]
arr = np.array(I).reshape((Nz,dim,dim))
N = dim

p = 0

cprint.cprint(p,Nz,N,border(arr))


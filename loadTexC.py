import numpy as np
import Image
import random
import cprint, time
import sys 
import scipy.ndimage

def struct(r):
    s = np.zeros((r,r,r))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                if(np.sqrt((i-r/2)**2+(j-r/2)**2+(k-r/2)**2) < r):
                    s[i,j,k] = 1

def border(data):
    r = 15
    r2 = 10
    r3 = 7
    r4 = 6
    b1 = scipy.ndimage.binary_closing(data,structure=struct(r)).astype(np.uint8)
    b1 = scipy.ndimage.binary_dilation(b1,structure=struct(r2)).astype(np.uint8)
    b = scipy.ndimage.binary_erosion(b1,structure=struct(r3)).astype(np.uint8)
    #c = b1-b
    c = scipy.ndimage.binary_closing(b1-b,structure=struct(r4)).astype(np.uint8)
    c = scipy.ndimage.binary_dilation(c,structure=struct(r4)).astype(np.uint8)
    return np.array(255*(c)).astype(np.uint8)

I = Image.open(sys.argv[1])
dim = I.size[0]
Nz = I.size[1]/I.size[0]
arr = np.array(I).reshape((Nz,dim,dim))
N = dim

p = 0

cprint.cprint(p,Nz,N,border(arr))


import numpy as np
import Image
import random
import cprint, time
import sys 
import scipy.ndimage

def struct(r):
    #return np.ones((r,r,r))
    s = np.zeros((r,r,r))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                if(np.sqrt((i-r/2)**2+(j-r/2)**2+(k-r/2)**2) < r):
                    s[i,j,k] = 1

def border(data):
    r = 8
    r2 = 7
    r3 = 6
    r4 = 5
    c = scipy.ndimage.binary_closing(data,structure=np.ones((r,r,r))).astype(np.uint8)
    d = scipy.ndimage.binary_dilation(c,structure=struct(r2)).astype(np.uint8)
    e = scipy.ndimage.binary_erosion(d,structure=struct(r3)).astype(np.uint8)

    c = scipy.ndimage.binary_closing(d-e,structure=struct(r4)).astype(np.uint8)
    c = scipy.ndimage.binary_dilation(c,structure=struct(r4)).astype(np.uint8)
    return np.array(255*(c)).astype(np.uint8)

I = Image.open(sys.argv[1])
dim = I.size[0]
Nz = I.size[1]/I.size[0]
arr = np.array(I).reshape((Nz,dim,dim))
N = dim

p = 0

cprint.cprint(p,Nz,N,border(arr))


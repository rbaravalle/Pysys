import numpy as np
cimport numpy as np
import scipy.ndimage as ndimage
import time
import pylab
from matplotlib import pyplot as plt
import Image
import cloadobj

import warp

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

cdef extern from "math.h":
    int round(float x)

def bake(np.ndarray[DTYPE_t, ndim=3] field, np.ndarray[DTYPE_tf, ndim=3]  dfield, np.ndarray[DTYPE_tf, ndim=1]  temperatures,int N,int Nz, int k2):
    cdef float dist

    cdef np.ndarray[DTYPE_tf, ndim=3] gx, gy, gz
    cdef np.ndarray[DTYPE_tf, ndim=3] result = np.zeros((256,256,256)).astype(np.float32)#(N,N,Nz)).astype(np.float32)
    cdef int i,j,k,cant

    # how many different temperatures
    cant = len(temperatures)

    cdef float f = ((cant-1)/np.max(dfield))

    print "Computing temperatures"
    for i from 0<=i<256:#N:
        for j from 0<=j<256:#N:
            for k from 0<=k<256:#Nz:
                result[i,j,k] = temperatures[round(dfield[i,j,k]*f)]

    #cdef np.ndarray[DTYPE_tf, ndim=2] gx2, gy2
    # For the paper
    if(False):
        I2 = Image.frombuffer('L',(N,N), (result[:,:,100]).astype(np.uint8),'raw','L',0,1)
        imgplot = plt.imshow(I2)
        plt.colorbar()
        gx2, gy2 = np.gradient(result[:,:,100])
        gx2 = ndimage.filters.gaussian_filter(gx2,3)
        gy2 = ndimage.filters.gaussian_filter(gy2,3)
        pylab.quiver(gx2,gy2)
        pylab.show()
        plt.show()

        if(False):
            I2 = Image.frombuffer('L',(N,N), (result[:,100,:]).astype(np.uint8),'raw','L',0,1)
            imgplot = plt.imshow(I2)
            plt.colorbar()
            gx2, gy2 = np.gradient(result[:,100,:])
            gx2 = ndimage.filters.gaussian_filter(gx2,3)
            gy2 = ndimage.filters.gaussian_filter(gy2,3)
            pylab.quiver(gx2,gy2)
            pylab.show()
            plt.show()

            I2 = Image.frombuffer('L',(N,N), (result[100,:,:]).astype(np.uint8),'raw','L',0,1)
            imgplot = plt.imshow(I2)
            plt.colorbar()
            gx2, gy2 = np.gradient(result[100,:,:])
            gx2 = ndimage.filters.gaussian_filter(gx2,3)
            gy2 = ndimage.filters.gaussian_filter(gy2,3)
            pylab.quiver(gx2,gy2)
            pylab.show()
            plt.show()

    print "Gradient..."
    gx, gy, gz = np.gradient(result)

    print "Gaussian's...x..."
    gx = ndimage.filters.gaussian_filter(gx.astype(np.float32),3)
    print "Gaussian's...y..."
    gy = ndimage.filters.gaussian_filter(gx.astype(np.float32),3)
    print "Gaussian's...z..."
    gz = ndimage.filters.gaussian_filter(gz.astype(np.float32),3)

    print "Resize gx..."
    gx = cloadobj.resizef(gx,N,Nz)
    print "Resize gy..."
    gy = cloadobj.resizef(gy,N,Nz)
    print "Resize gz..."
    gz = cloadobj.resizef(gz,N,Nz)

    print "warp and return..."
    return warp.warpExpand(field, gx,gy,gz, N, Nz,k2),gx,gy,gz


import numpy as np
cimport numpy as np
import scipy.ndimage as ndimage
import time
import pylab
from matplotlib import pyplot as plt
import Image

import warp

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

cdef extern from "math.h":
    int round(float x)

def bake(np.ndarray[DTYPE_tf, ndim=3] field, np.ndarray[DTYPE_t, ndim=3] geom, np.ndarray[DTYPE_tf, ndim=1]  temperatures,int N,int Nz, int k2):
    cdef float maximo,dist

    cdef np.ndarray[DTYPE_tf, ndim=3] gx, gy, gz

    # distance field of original geometry
    cdef np.ndarray[DTYPE_tf, ndim=3]  dfield = np.array(ndimage.distance_transform_edt(geom)).astype(np.float32)

    # max distance in the distance field
    maximo = np.max(dfield)

    dfield = dfield.reshape(N,N,Nz)

    cdef np.ndarray[DTYPE_tf, ndim=3] result = np.zeros((N,N,Nz)).astype(np.float32)
    cdef int i,j,k,cant

    # how many different temperatures
    cant = len(temperatures)

    cdef float f = ((cant-1)/maximo)

    for i from 0<=i<N:
        for j from 0<=j<N:
            for k from 0<=k<Nz:
                result[i,j,k] = temperatures[np.int(round(dfield[i,j,k]*f))]

    #cdef np.ndarray[DTYPE_tf, ndim=2] gx2, gy2
    # For the paper
    if(False):
        I2 = Image.frombuffer('L',(N,N), (result[:,:,100]).astype(np.uint8),'raw','L',0,1)
        imgplot = plt.imshow(I2)
        plt.colorbar()
        gx2, gy2 = np.gradient(result[:,:,100])
        gx2 = ndimage.filters.gaussian_filter(gx2,1)
        gy2 = ndimage.filters.gaussian_filter(gy2,1)
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

    gx, gy, gz = np.gradient(result)

    # FIX ME
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    gz = gz.astype(np.float32)

    gx = ndimage.filters.gaussian_filter(gx,3)
    gy = ndimage.filters.gaussian_filter(gy,3)
    gz = ndimage.filters.gaussian_filter(gz,3)

    return np.array(warp.warp(field.astype(np.uint8), gx, gy, gz, N, Nz,k2)).astype(np.float32)


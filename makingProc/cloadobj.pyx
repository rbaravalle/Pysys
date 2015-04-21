import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
ctypedef np.int32_t DTYPE_ti
ctypedef np.int8_t DTYPE_tii
ctypedef np.float32_t DTYPE_tf

import binvox
import time
import scipy.ndimage as ndimage
from warp import lowsize

cdef extern from "math.h":
    int floor(float x)

def resizef( np.ndarray[DTYPE_tf, ndim=3] model,int N, int Nz):
    cdef np.ndarray[DTYPE_tf, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.float32)

    cdef int x,y,z
    cdef float ar = 255.0/(N-1)
    cdef float arz = 255.0/(Nz-1)
    for x  from 0<=x<N:
        for y  from 0<=y<N:
            for z  from 0<=z<Nz:
                model2[x,y,z] = model[floor(x*ar),floor(y*ar),floor(z*arz)]
    return model2

def resize( np.ndarray[DTYPE_t, ndim=3] model,int N, int Nz):
    cdef np.ndarray[DTYPE_t, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.uint8)

    cdef int x,y,z
    cdef float ar = 255.0/(N-1)
    cdef float arz = 255.0/(Nz-1)
    for x  from 0<=x<N:
        for y  from 0<=y<N:
            for z  from 0<=z<Nz:
                model2[x,y,z] = model[floor(x*ar),floor(y*ar),floor(z*arz)]
    return model2

def invresize( np.ndarray[DTYPE_t, ndim=3] model,int N, int Nz):
    cdef np.ndarray[DTYPE_t, ndim=3] model2 = np.zeros((256,256,256)).astype(np.uint8)

    cdef int x,y,z
    cdef float ar = (N-1)/255.0
    cdef float arz = (Nz-1)/255.0
    for x  from 0<=x<256:
        for y  from 0<=y<256:
            for z  from 0<=z<256:
                model2[x,y,z] = model[floor(x*ar),floor(y*ar),floor(z*arz)]
    return model2

def orientatef( np.ndarray[DTYPE_tf, ndim=3] model,int N, int Nz,int modelNumber):
    cdef np.ndarray[DTYPE_tf, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.float32)

    cdef int x,y,z
    if(modelNumber == 1 or modelNumber == 2):
        for x  from 0<=x<N:
            for y  from 0<=y<N:
                for z  from 0<=z<Nz:
                    model2[Nz-1-z,N-1-y,x] = model[x,z,y] # bunny, otherbread
    else:
        for x  from 0<=x<N:
            for y  from 0<=y<N:
                for z  from 0<=z<Nz:
                    model2[Nz-1-z,y,x] = model[x,y,z]      # bread2,croissant

    return model2

def orientate( np.ndarray[DTYPE_t, ndim=3] model,int N, int Nz,int modelNumber):
    cdef np.ndarray[DTYPE_t, ndim=3] model2 = np.zeros((N,N,Nz)).astype(np.uint8)

    cdef int x,y,z

    if(modelNumber == 1 or modelNumber == 2):
        for x  from 0<=x<N:
            for y  from 0<=y<N:
                for z  from 0<=z<Nz:
                    model2[Nz-1-z,N-1-y,x] = model[x,z,y] # bunny, otherbread
    else:
        for x  from 0<=x<N:
            for y  from 0<=y<N:
                for z  from 0<=z<Nz:
                    model2[Nz-1-z,y,x] = model[x,y,z]      # bread2, croissant
    return model2

# intersection between a cube field and a geometry
def intersect(field, geom,density,int N, int Nz, float thresh):

    # threshold value
    # bubble intersection
    # based on the distance to the surface

    # distance field
    t = time.clock()

    print "Distance Field computing.."
    dfield = np.array(ndimage.distance_transform_edt(geom)).reshape(256,256,256)
    print "Distance Field computation time: ", time.clock()-t

    print "Crumb..."
    crumb = np.array(dfield>thresh).astype(np.uint8)

    print "Crust..."
    crust = geom-crumb

    # NOW resize...

    if(N!= 256 ):
        print "Resize Geom..."
        geom = resize(geom,N,Nz)

        print "Resize Crumb..."
        crumb = resize(crumb,N,Nz)

        #print "Resize Distance Field..."
        #dfield = resizef(np.array(dfield).astype(np.float32),N,Nz)

    # the bubbles in white (1-field) are taken into account
    # when they are away from the surface, so this is the 'crumb region'

    return (geom-crumb*(1-field)), dfield.astype(np.float32),geom,crust,crumb*(density)


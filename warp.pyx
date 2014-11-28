import numpy as np
cimport numpy as np

cdef extern from "math.h":
    int round(float x)
    int floor(float x)

DTYPE = np.uint8
DTYPE_f = np.float32
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf


# warp a 3D field using the gradient G(gx,gy,gz)
def warp2D(np.ndarray[DTYPE_tf, ndim=2] field, np.ndarray[DTYPE_tf, ndim=2] gx, np.ndarray[DTYPE_tf, ndim=2] gy, int N, float k):
    cdef int x,y,u,v
    cdef np.ndarray[DTYPE_tf, ndim=2] field2 = np.zeros((N,N),dtype=DTYPE_f)
    for x from 0<=x<N:    
        for y from 0<=y<N:
            u = round(x+k*gx[x,y])
            v = round(y+k*gy[x,y])
            try:
                field2[x,y] = field[u,v]
            except: pass

    return field2

cdef linterp(float x, int x0,int y0,int x1,int y1):
    return y0+(y1-y0)*(x-x0)/(x1-x0)

cdef resample(np.ndarray[DTYPE_t, ndim=3] field, float u, float v, float w):
    cdef int a0,a1,b0,b1,c0,c1,suma
    a0 = floor(u)
    a1 = floor(u)+1
    b0 = floor(v)
    b1 = floor(v)+1
    c0 = floor(w)
    c1 = floor(w)+1

    suma = field[a0,b0,c0]+field[a0,b1,c0]+field[a1,b0,c0]+field[a1,b1,c0]+field[a0,b0,c1]+field[a0,b1,c1]+field[a1,b0,c1]+field[a1,b1,c1]
    
    if(suma>255*3):
        return 255#linterp(a,b)# field[round(u),round(v),round(w)]
    else: return 0

# warp a 3D field using the gradient G(gx,gy,gz)
def warp(np.ndarray[DTYPE_t, ndim=3] field, np.ndarray[DTYPE_tf, ndim=3] gx, np.ndarray[DTYPE_tf, ndim=3] gy, np.ndarray[DTYPE_tf, ndim=3] gz, int N, int Nz, float k):
    cdef int x,y,z
    cdef float u,v,w
    cdef np.ndarray[DTYPE_t, ndim=3] field2 = np.zeros((N,N,Nz),dtype=DTYPE)
    for x from 0<=x<N:    
        for y from 0<=y<N:
            for z from 0<=z<Nz:
                u = x-k*gx[x,y,z] # f^(-1)
                v = y-k*gy[x,y,z]
                w = z-k*gz[x,y,z]
                try:
                    field2[x,y,z] = resample(field,u,v,w)#field[u,v,w]
                except: pass

    return field2

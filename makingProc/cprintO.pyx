import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t

cdef extern from "math.h":
    int round(float x)

def sample( np.ndarray[DTYPE_t, ndim=3] arr,  np.ndarray[DTYPE_t, ndim=1] v,  np.ndarray[DTYPE_t, ndim=1] c, int Nz, int N):
    cdef int x,y,z
    x,y,z = v+c
    if(x<Nz and x>=0 and y<N and y>=0 and z<N and z>=0):
        return arr[x,y,z]
    else: return 0

def cprint(int p, int Nz, int N, np.ndarray[DTYPE_t, ndim=3] arr):
    cdef float maxNeigh,c
    maxNeigh = 15.0
    c = N*(2.5/100.0)
    cdef int z,y,x,s
    cdef np.ndarray[DTYPE_t, ndim=1] v
    print N-2*p,N-2*p,N-2*p
    for z from 0<=z<Nz:
        for y from 0<=y<N:
            for x from 0<=x<N:
               v = np.array([z,N-1-y,x]).astype(np.uint8)
               s = 0
               s += sample(arr,v,np.array([0,0,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([-c,-c,-c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([c,c,c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([0,0,c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([c,0,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([0,c,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([-c,0,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([0,-c,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([0,0,-c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([0,c,c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([0,-c,-c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([c,c,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([-c,-c,0]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([c,0,c]).astype(np.uint8),Nz,N)
               s += sample(arr,v,np.array([-c,0,-c]).astype(np.uint8),Nz,N)


               print round(s/maxNeigh)

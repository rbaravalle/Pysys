#!/usr/bin/python

import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

def main():

    field = createField(128)
    outputField(field)

def createField(int N):
    cdef int x,y,z
    cdef np.ndarray[DTYPE_t, ndim=3] F
    cdef np.ndarray[DTYPE_tf, ndim=3] R
    F = np.zeros((N,N,N)).astype(np.uint8) # Density field

    R = np.random.rand(N,N,N).astype(np.float32) # Random numbers

    for x from 0<=x<N:
        for y from 0<=y<N:
            for z from 0<=z<N:
                if R[x,y,z] > 0.5 : 
                    F[x,y,z] = 1

    return F

def outputField(field):
    (X,Y,Z) = field.shape
    print X, Y, Z
    for x, y, z in np.ndindex((X,Y,Z)):
        print field[x,y,z]

if __name__ == "__main__":
    main()

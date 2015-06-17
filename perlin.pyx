#!/usr/bin/python
import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

import noise

def main():

    field = createField(128)
    outputField(field)

def createField(int N):
    cdef int x,y,z
    cdef float step
    cdef np.ndarray[DTYPE_t, ndim=3] F
    F = np.zeros((N,N,N)).astype(np.uint8) # Density field

    cdef int N2 = 230 # FIX ME, pnoise does not work for 256

    step = 16.0 / N2
    for x from 0<=x<N2:
        print "x: ", x
        for y from 0<=y<N2:
            for z from 0<=z<N2:
                if noise.pnoise3(step*x, step*y,step*z, octaves=N2/2) > 0 : 
                    F[x+(N-N2),y+(N-N2),z+(N-N2)] = 1

    return F

#def createField(int N):
#    cdef int x,y,z
#    cdef float step
#    cdef np.ndarray[DTYPE_t, ndim=3] F
#    F = np.zeros((N,N,N)).astype(np.uint8) # Density field

#    step = 32.0 / N
#    for x from 0<=x<N:
#        print "x:", x
#        for y from 0<=y<N:
#            for z from 0<=z<N:
#                n = noise.pnoise3(step*x, step*y,step*z, octaves=N/2)
#                if n > 0:
#                    F[x,y,z] = 1

#    return F

def outputField(field):
    (X,Y,Z) = field.shape
    print X, Y, Z
    for x, y, z in np.ndindex((X,Y,Z)):
        print field[x,y,z]

if __name__ == "__main__":
    main()

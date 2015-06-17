#!/usr/bin/python
import numpy as np
cimport numpy as np

ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf


from scipy.spatial import KDTree

def main():

    field = createField(128)
    outputField(field)

def createField(int N):
    cdef float step, halfStep,dist

    cdef int x,y,z
    cdef np.ndarray[DTYPE_tf, ndim=2] V,Q
    cdef np.ndarray[DTYPE_t, ndim=3] F
    F = np.zeros((N,N,N)).astype(np.uint8) # Density field


    V = np.random.rand(N*N / 16.0, 3).astype(np.float32) # Voronoi center points
    kdt = KDTree(V)

    Q = np.zeros((N,N,N, 3)).astype(np.float32) # Query points 
    step = 1. / N
    halfStep = step / 2.
    for x from 0<=x<N:
        print "X1: ", x
        for y from 0<=y<N:
            for z from 0<=z<N:
                Q[x,y,z][0] = step * x + halfStep
                Q[x,y,z][1] = step * y + halfStep
                Q[x,y,z][2] = step * z + halfStep
                # print x, y, z, " : " , Q[x,y,z]

    print "kdt.query..."
    D = kdt.query(Q, 2)[0] # Distances from query points to closes points

    for x from 0<=x<N: # If the distances are close, set the value to 1
        print "X2: ", x
        for y from 0<=y<N:
            for z from 0<=z<N:
                d = D[x,y,z]
                dist = d[0] / d[1]
                if dist > 0.85:
                    F[x,y,z] = 1;

    return F

def outputField(field):
    (X,Y,Z) = field.shape
    print X, Y, Z
    for x, y, z in np.ndindex((X,Y,Z)):
        print field[x,y,z]

    print "This is where output goes!"

if __name__ == "__main__":
    main()

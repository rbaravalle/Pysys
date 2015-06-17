import numpy as np
cimport numpy as np
import sys


ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf

from scipy.spatial import KDTree

def createField(int N):

    cdef int x,y,z
    cdef float d
    cdef np.ndarray[DTYPE_t, ndim=3] F
    F = np.zeros((N,N,N)).astype(np.uint8) # Density field

    cdef int N2 = 200

    with open("distances200.txt") as f:
        line = f.readline();

        for x from 0<=x<N2:
            print "x: ", x
            for y from 0<=y<N2:
                for z from 0<=z<N2:
                    line = f.readline()
                    dists = line.split(' ')
                    for i in range(0, len(dists)):
                        dists[i] = float(dists[i])
                    
                    
                    #d = F1(dists) * F2(dists) * F3(dists)
                    d = F1(dists)  
                    if d > 0.8:
                        F[x+(N-N2),y+(N-N2),z+(N-N2)] = 1

    return F

def F1(dists):
    return dists[0] / dists[1]

def F2(dists):
    return dists[1] / dists[2]

def F3(dists):
    return dists[2] / dists[3]

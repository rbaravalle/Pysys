cimport numpy as np
cdef class Particle:
    cdef public float randomm
    cdef public int i,size
    cdef public np.ndarray contorno
    cdef np.ndarray occupied2


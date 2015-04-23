cimport numpy as np
cdef class Particle:
    cdef public float randomm,randomParam
    cdef public int i,size
    cdef public np.ndarray occupied,occupied2
    cdef public list contorno



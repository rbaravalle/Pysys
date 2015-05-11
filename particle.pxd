cimport numpy as np
cdef class Particle:
    cdef public float randomm,randomParam
    cdef public int i,size,sepp
    cdef public np.ndarray occupied,occupied2,dx,dy,dz,geom
    cdef public list contorno

    cdef public fn(Particle)




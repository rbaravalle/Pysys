import numpy as np
cimport numpy as np
dT = 0.1

ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti

cdef extern from "math.h":
    float sin(float x)

# 3D Dinamic Systems
cdef f1(np.ndarray[DTYPE_tf, ndim=1] v):
    cdef np.ndarray[DTYPE_tf, ndim=1]  res = np.zeros(3).astype(np.float32)
    res[0] = (1-v[1])*v[1]
    res[1] = (1-v[0])*v[0]

    return res

cdef f2(np.ndarray[DTYPE_tf, ndim=1] v):
    cdef float x,y,z,a,b,c

    cdef np.ndarray[DTYPE_tf, ndim=1]  res = np.zeros(3).astype(np.float32)
    x=v[0]
    y=v[1]
    z=v[2]
    a = 10.0
    b = 28.0
    c = 8.0/3

    res[0] = a*(y-x)
    res[1] = x*(b-z)-y
    res[2] = x*y-c*z

    return res


def runge_kutta(np.ndarray[DTYPE_tf, ndim=1] x,float dT) :
    cdef np.ndarray[DTYPE_tf, ndim=1]  k1,k2,k3,k4
    k1 = f1(x)
    k2 = f1(x+k1*0.5)
    k3 = f1(x+k2*0.5)
    k4 = f1(x+k3)
    return x + dT*(k1+ k2*2.0 + k3*2.0 + k4)*1.0/6



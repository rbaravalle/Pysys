import numpy as np
cimport numpy as np
dT = np.float32(0.1)

ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti

cdef extern from "math.h":
    float sin(float x)

# 3D Dinamic Systems
cdef f1(np.ndarray[DTYPE_tf, ndim=1] v):
    return np.array([(1-v[1])*v[1],(1-v[0])*v[0],0]).astype(np.float32)

cdef f2(np.ndarray[DTYPE_tf, ndim=1] v):
    cdef float x,y,z,a,b,c
    x=v[0]
    y=v[1]
    z=v[2]
    a = np.float32(10)
    b = np.float32(28)
    c = np.float32(8)/3
    return np.array([a*(y-x),x*(b-z)-y,x*y-c*z]).astype(np.float32)


def runge_kutta(np.ndarray[DTYPE_tf, ndim=1] x,float dT) :
    cdef np.ndarray[DTYPE_tf, ndim=1] k1,k2,k3,k4
    k1 = f1(x)
    k2 = f1(x+k1*0.5)
    k3 = f1(x+k2*0.5)
    k4 = f1(x+k3)
    return x + dT*(k1+ k2*2.0 + k3*2.0 + k4)*np.float32(1)/6



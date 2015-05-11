from globalsv import *
cdef float dT = 0.1

cdef float dm1 = dm1
cdef float dZm1 = dZm1
cdef float x0 = x0
cdef float y0 = y0
cdef float z0 = z0

cdef extern from "math.h":
    float sin(float x)

# 3D Dinamic Systems
cdef f1(float v0,float v1, float v2):
    #return (1-v1)*v1, (1-v0)*v0,0
    #return (v1-4)*v1, (v0-4)*v0,0
    return v1, -sin(v0),0

cdef f2(float v0,float v1,float v2):
    cdef float x,y,z,a,b,c


    x=v0
    y=v1
    z=v2
    a = 10.0
    b = 28.0
    c = 8.0/3

    return a*(y-x),x*(b-z)-y,x*y-c*z


cdef runge_kutta(int x, int y, int z):
    cdef float xp0,xp1,xp2,k10,k11,k12,k20,k21,k22,k30,k31,k32,k40,k41,k42
    

    xp0 = x*(dm1)+x0
    xp1 = y*(dm1)+x0
    xp2 = z*(dZm1)+z0

    k10,k11,k12 = f1(xp0,xp1,xp2)
    k20,k21,k22 = f1(xp0+k10*0.5,xp1+k11*0.5,xp2+k12*0.5)
    k30,k31,k32 = f1(xp0+k20*0.5,xp1+k21*0.5,xp2+k22*0.5)
    k40,k41,k42 = f1(xp0+k30,xp1+k31,xp2+k32)

    return xp0 + dT*(k10+ k20*2.0 + k30*2.0 + k40)*u6,xp1 + dT*(k11+ k21*2.0 + k31*2.0 + k41)*u6,xp2 + dT*(k12+ k22*2.0 + k32*2.0 + k42)*u6



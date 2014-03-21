import numpy as np
dT = np.float32(0.1)
factual = 0

# 3D Dinamic Systems
def f1(v):
    x=v[0]
    y=v[1]
    z=v[2]
    a = np.float32(0.2)
    b = np.float32(0.2)
    c = np.float32(5.7)
    return np.array([-y-z,x+a*y,b+z*(x-c)])

def f2(v):
    x=v[0]
    y=v[1]
    z=v[2]
    a = np.float32(10)
    b = np.float32(28)
    c = np.float32(8)/3
    return np.array([a*(y-x),x*(b-z)-y,x*y-c*z])

funcs = np.array([ f1,f2 ])


def runge_kutta(x,factual,dT) :
    res = []
    k1 = funcs[factual](x)*dT
    k2 = funcs[factual](x+k1*0.5)*dT
    k3 = funcs[factual](x+k2*0.5)*dT
    k4 = funcs[factual](x+k3)*dT
    res = k1+ k2*2.0 + k3*2.0 + k4
    return x + res*np.float32(1)/6



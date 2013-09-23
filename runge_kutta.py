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
    return [-y-z,x+a*y,b+z*(x-c)]

def f2(v):
    x=v[0]
    y=v[1]
    z=v[2]
    a = np.float32(10)
    b = np.float32(28)
    c = np.float32(8)/3
    return [a*(y-x),x*(b-z)-y,x*y-c*z]

funcs = [ f1,f2 ]

# multiply vector with an scalar
def multV(x,e) :
    return [x[0]*e,x[1]*e,x[2]*e]


# vector sum
def sumarV(x,y) :
    return [x[0]+y[0],x[1]+y[1],x[2]+y[2]]


def runge_kutta(x,factual,dT) :
    res = []
    k1 = multV(funcs[factual](x),dT)
    k2 = multV(funcs[factual](sumarV(x,multV(k1,np.float32(1)/2))),dT)
    k3 = multV(funcs[factual](sumarV(x,multV(k2,np.float32(1)/2))),dT)
    k4 = multV(funcs[factual](sumarV(x,k3)),dT)
    res = sumarV(k1,sumarV(multV(k2,np.float32(2)), sumarV(multV(k3,np.float32(2)), k4)))
    return sumarV(x, multV(res,np.float32(1)/6))



dT = 0.05
factual = 0

# 3D Dinamic Systems
funcs = [lambda v : [v[0]*v[0]-v[1]*v[1]+1,2*v[0]*v[1]+1,v[2]*v[2]] ]

# multiply vector with an scalar
def multV(x,e) :
    return [x[0]*e,x[1]*e,x[2]*e]


# vector sum
def sumarV(x,y) :
    return [x[0]+y[0],x[1]+y[1],x[2]+y[2]]


def runge_kutta(x,factual,dT) :
    res = []
    k1 = multV(funcs[factual](x),dT)
    k2 = multV(funcs[factual](sumarV(x,multV(k1,1/2))),dT)
    k3 = multV(funcs[factual](sumarV(x,multV(k2,1/2))),dT)
    k4 = multV(funcs[factual](sumarV(x,k3)),dT)
    res = sumarV(k1,sumarV(multV(k2,2), sumarV(multV(k3,2), k4)))

    return sumarV(x, multV(res,1/6))



import numpy as np
cimport numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


cdef extern from "math.h":
    int round(float x)
    int floor(float x)

DTYPE = np.uint8
DTYPE_f = np.float32
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf


# warp a 3D field using the gradient G(gx,gy,gz)
def warp2D(np.ndarray[DTYPE_tf, ndim=2] field, np.ndarray[DTYPE_tf, ndim=2] gx, np.ndarray[DTYPE_tf, ndim=2] gy, int N, float k):
    cdef int x,y,u,v
    cdef np.ndarray[DTYPE_tf, ndim=2] field2 = np.zeros((N,N),dtype=DTYPE_f)
    for x from 0<=x<N:    
        for y from 0<=y<N:
            u = round(x+k*gx[x,y])
            v = round(y+k*gy[x,y])
            try:
                field2[x,y] = field[u,v]
            except: pass

    return field2

cdef linterp(float x, int x0,int y0,int x1,int y1):
    return y0+(y1-y0)*(x-x0)/(x1-x0)

cdef resample(np.ndarray[DTYPE_t, ndim=3] field, float u, float v, float w):
    cdef int a0,a1,b0,b1,c0,c1,suma
    a0 = floor(u)
    a1 = floor(u)+1
    b0 = floor(v)
    b1 = floor(v)+1
    c0 = floor(w)
    c1 = floor(w)+1

    suma = field[a0,b0,c0]+field[a0,b1,c0]+field[a1,b0,c0]+field[a1,b1,c0]+field[a0,b0,c1]+field[a0,b1,c1]+field[a1,b0,c1]+field[a1,b1,c1]
    
    if(suma>255*3):
        return 255#linterp(a,b)# field[round(u),round(v),round(w)]
    else: return 0

# warp a 3D field using the gradient G(gx,gy,gz)
def warp(np.ndarray[DTYPE_t, ndim=3] field, np.ndarray[DTYPE_tf, ndim=3] gx, np.ndarray[DTYPE_tf, ndim=3] gy, np.ndarray[DTYPE_tf, ndim=3] gz, np.ndarray[DTYPE_tf, ndim=3] density, int N, int Nz, float k):
    cdef int x,y,z
    cdef float u,v,w
    cdef np.ndarray[DTYPE_t, ndim=3] field2 = np.zeros((N,N,Nz),dtype=DTYPE)

    for x from 0<=x<N:    
        for y from 0<=y<N:
            for z from 0<=z<Nz:
                k *= density[x,y,z]
                u = (x-gx[x,y,z]*k) # f^(-1)
                v = (y-gy[x,y,z]*k)
                w = (z-gz[x,y,z]*k)
                try:
                    field2[x,y,z] = resample(field,u,v,w)#field[u,v,w]
                except: pass

    return field2


def plott(Z):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, 256, 1)
    Y = np.arange(0, 256, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_zlim(0.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
# warp a 3D field using the gradient G(gx,gy,gz)
def warpExpandGeom(np.ndarray[DTYPE_t, ndim=3] geom,np.ndarray[DTYPE_tf, ndim=3] dfield, np.ndarray[DTYPE_tf, ndim=3] ddfx, np.ndarray[DTYPE_tf, ndim=3] ddfy,np.ndarray[DTYPE_tf, ndim=3] ddfz, np.ndarray[DTYPE_tf, ndim=3] density,int N, int Nz,float dmax, float dmin):
    cdef int x,y,z
    cdef float u,v,w,rho,h,rhomin,rhomax,m,aux
    cdef np.ndarray[DTYPE_t, ndim=3] field2 = np.zeros((N,N,Nz),dtype=DTYPE)
    
    rhomax = 0.9
    rhomin = 0.4
    m = -(rhomax-rhomin)/(dmax-dmin)
    h = rhomax-m*dmin
    
    cdef np.ndarray[DTYPE_tf, ndim=2] maxd = np.zeros((N,N),dtype=np.float32)

    for x from 0<=x<N:
        for z from 0<=z<N:
            maxd[x,z] = 2.0*(np.random.random())*np.max(density[x,:,z])/2.0

    maxd = ndimage.filters.gaussian_filter(maxd,3)
    #density = 2.0*ndimage.filters.gaussian_filter(density,10)
    

    for x from 0<=x<N:    
        for y from 0<=y<N:
            for z from 0<=z<Nz:
                aux = (1.0-0.2*density[x,y,z]/dmax)
                u = (x)*1.0 # f^(-1)
                w = (z)*0.98#(1.0-0.5*dfield[x,y,z]/100.0)
                v = (y)*(maxd[round(u),round(w)]*m+h)*aux#*(0.5+(y/4.0)/(N-1))

                try:
                    field2[x,y,z] = resample(geom,u,v,w)#field[u,v,w]
                except: pass

    return field2

# expand the total volume to simulate baking
# #(depends on N,Nz from parameter)
def warpExpandGeom2(np.ndarray[DTYPE_t, ndim=3] geom,np.ndarray[DTYPE_tf, ndim=3] dfield, np.ndarray[DTYPE_tf, ndim=3] ddfx, np.ndarray[DTYPE_tf, ndim=3] ddfy,np.ndarray[DTYPE_tf, ndim=3] ddfz, np.ndarray[DTYPE_tf, ndim=3] density,int N, int Nz,float dmax,float dmin):
    cdef int x,y,z
    cdef float u,v,w,rx,ry,rz,df,rho,dfx,dfy,dfz,h,rhomin,rhomax,m,aux
    cdef np.ndarray[DTYPE_t, ndim=3] geomD = np.zeros((N,N,Nz),dtype=DTYPE)

    rhomax = 0.9
    rhomin = 0.4
    m = -(rhomax-rhomin)/(dmax-dmin)
    h = rhomax-m*dmin

    cdef np.ndarray[DTYPE_tf, ndim=2] maxd = np.zeros((N,N),dtype=np.float32)

    for x from 0<=x<N:
        for z from 0<=z<N:
            maxd[x,z] = (2.0*np.random.random())*np.max(density[x,:,z])/2.0

    maxd = ndimage.filters.gaussian_filter(maxd,3)
    #print maxd*m+h
    #plott(maxd*m+h)

    #*(np.float(y-deltay)*30.0)/np.float(N-1)
    for x from 0<=x<N:    
        for y from 0<=y<N:
            for z from 0<=z<Nz:
                #aux = density[x,y,z]
                u = x*0.98
                w = (z)*0.8-8.0*ddfz[x,y,z]

                ry = 0.85*(maxd[round(u),round(w)]*m+h)
                v = y*ry

                try:
                    geomD[x,y,z] = resample(geom,u,v,w)
                except: pass


    return geomD


import numpy as np
cimport numpy as np

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
def warp(np.ndarray[DTYPE_t, ndim=3] field, np.ndarray[DTYPE_tf, ndim=3] gx, np.ndarray[DTYPE_tf, ndim=3] gy, np.ndarray[DTYPE_tf, ndim=3] gz, int N, int Nz, float k):
    cdef int x,y,z
    cdef float u,v,w
    cdef np.ndarray[DTYPE_t, ndim=3] field2 = np.zeros((N,N,Nz),dtype=DTYPE)

    for x from 0<=x<N:    
        for y from 0<=y<N:
            for z from 0<=z<Nz:
                u = x-gx[x,y,z]*k # f^(-1)
                v = y-gy[x,y,z]*k
                w = z-gz[x,y,z]*k
                try:
                    field2[x,y,z] = resample(field,u,v,w)#field[u,v,w]
                except: pass

    return field2


# expand the total volume to simulate baking
# #(depends on N,Nz from parameter)
def warpExpandGeom(np.ndarray[DTYPE_t, ndim=3] geom,np.ndarray[DTYPE_tf, ndim=3] dfield,np.ndarray[DTYPE_tf, ndim=3] density,int N, int Nz):
    cdef int x,y,z
    cdef float u,v,w,gravity_x,gravity_y,gravity_z,df,rho
    cdef int deltax = 0 # desplazamiento del modelo en x
    cdef int deltay = 0 # desplazamiento del modelo en y
    cdef int deltaz = 0 # desplazamiento del modelo en z
    cdef np.ndarray[DTYPE_t, ndim=3] geomD = np.zeros((N,N,Nz),dtype=DTYPE)
    for x from 0<=x<N:    
        for y from 0<=y<N:
            for z from 0<=z<Nz:
                df = 1.0-dfield[x,y,z]/50.0
                rho = 1.0-density[x,y,z]/15.0
                gravity_x = 0.9#((np.float(N-1)-np.float(x-deltax)/6.0)/np.float(N-1))
                #if(z < 150 and z > 100):
                gravity_y = ((np.float(N-1)-np.float(y-deltay)/3.0)/np.float(N-1))
                #else: vnew = 1.0 
                gravity_z = 0.9#((np.float(N-1)-np.float(z-deltaz)/6.0)/np.float(N-1))
                u = (x-deltax)*rho#gravity_x*rho
                v = (y-deltay)*rho#gravity_y*rho
                w = (z-deltaz)*rho#gravity_z*rho

                #df = 1.0-dfield[u,v,w]/50.0
                #u *= df
                #v *= df
                #w *= df
                try:
                    geomD[x,y,z] = resample(geom,u,v,w)
                except: pass

    return geomD


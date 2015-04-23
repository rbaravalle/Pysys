# cython: profile=True

import numpy as np
cimport numpy as np
from runge_kutta import *
from globalsv import *


from libc.stdlib cimport rand, RAND_MAX

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti



cdef extern from "math.h":
    float pow(int x ,float y)
    int floor(float x)
    int round(float x)
    float sqrt(float x)

def sign():
    if(rand()/float(RAND_MAX) > 0.5): return -1
    return 1

def aCanvas(float x):
    return floor(x*maxcoord)

cdef int maxcoord = maxcoord
cdef int maxcoordZ = maxcoordZ
cdef int N = N


cdef setBorder(Particle pi,int x,int y,int z):
    cdef int i,j,k,v,sep,ii
    ii = pi.i
    sep = 1
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    pi.occupied2[x+i,y+j,z+k] = ii
                except: pass


cdef int searchBorder(Particle pi, int x,int y,int z):
    cdef int i,j,k,v,sep,ii
    sep = 1
    ii = pi.i
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    v = pi.occupied2[x+i,y+j,z+k]
                    if(v > 0 and v != ii): return True
                except:pass
    return 0


def add(Particle pi,int x,int y,int z):
    cdef np.ndarray[DTYPE_tf, ndim=1] xp = np.zeros(3).astype(np.float32)
    cdef list contorno
    cdef float d,de,deP,rr
    cdef int bestX,bestY,bestZ,xh,yh,zh,i

    sep = 1#pi.sep()
    contorno = pi.contorno

    xp[0] = x*(diffX/maxcoord)+x0
    xp[1] = y*(diffX/maxcoord)+x0
    xp[2] = z*(diffZ/maxcoordZ)+z0
    xp = runge_kutta(xp,dT)

    bestX=bestY=bestZ=deP=10000
    for xh from x-1<=xh<=x+1:
        for yh from y-1<=yh<=y+1:
            for zh from z-1<=zh<=z+1:
                if(not(xh==x and yh==y and zh==z)):
                    de = float((xp[0] - (xh*(diffX/maxcoord)+x0))**2+(xp[1] - (yh*(diffX/maxcoord)+x0))**2+(xp[2] - (zh*(diffZ/maxcoord)+x0))**2)
                    if(de <deP):
                        deP = de
                        bestX = xh
                        bestY = yh
                        bestZ = zh
                    if(rand()/float(RAND_MAX) >(1.0-pi.randomParam)): contorno.append([xh,yh,zh])
    
    setBorder(pi,x,y,z)
    contorno.append([bestX,bestY,bestZ])
    return contorno

def grow(Particle pi):
        cdef int w = 0, h, r,nx,ny,nz
        cdef int fn,size,ii,sep
        cdef list contorno
        ii = pi.i
        size = pi.size
        contorno = pi.contorno
        sep = 1#pi.sep()
        fn = pi.fn()

        for r from 0 <= r < fn:
            for h from 0 <= h < len(contorno):
                w = h
                nx = contorno[h][0]
                ny = contorno[h][1]
                nz = contorno[h][2]
                try:
                    if(pi.occupied[nx,ny,nz] <= 0 and not(searchBorder(pi,nx,ny,nz))):

                        pi.occupied[nx,ny,nz] = np.uint8(255)
                        pi.occupied2[nx,ny,nz] = ii
                        contorno = add(pi,nx,ny,nz)
                        size+=1
                        break                
                except: pass

        del contorno[0:w]
        #contorno = contorno[w+1:]

        pi.contorno = contorno
        pi.size = size





cdef class Particle:

    def __cinit__(self,int i,int lifet,float randomParam, np.ndarray[DTYPE_ti, ndim=3] occupied,np.ndarray[DTYPE_ti, ndim=3] occupied2):
        cdef int x,y,z,dist
        cdef float r,rv,tempfx,tempfy,rm

        rm = float(RAND_MAX)

        x = int(maxcoord*(rand()/rm))
        y = int(maxcoord*(rand()/rm))
        z = int(maxcoordZ*(rand()/rm))

        tempfx = float(x-maxcoord/2)
        tempfy = float(y-maxcoord/2)

        rv = sqrt(tempfx*tempfx+tempfy*tempfy)
        if(rv!=0):
            tempfx = tempfx/rv
            tempfy = tempfy/rv

        r = 1.0*(rand()/rm)*(maxcoord/1.5-rv)

        x = np.clip(floor(x + r*(tempfx)),0,maxcoord-1);
        y = np.clip(floor(y + r*(tempfy)),0,maxcoord-1);

        self.i = i
        #self.contorno = np.array([[-1,-1,-1]]).astype(np.int32)
        self.contorno = [[-1,-1,-1]]
        self.randomm = rand()/rm
        occupied[x,y,z] = np.uint8(255)
        occupied2[x,y,z] = i
        self.randomParam = randomParam
        self.size = 1
        self.occupied2=occupied2
        self.occupied=occupied
        self.contorno = add(self,x,y,z)

        
    # Different separations depending on the bubble size
    def sep(self):
        return 1

    def fn(self):
        if(self.size < 5): return 1
        else: return 1+floor(0.01*self.size)



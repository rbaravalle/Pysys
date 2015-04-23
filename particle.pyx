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


cdef setBorder(int x,int y,int z,int ii,int sep,np.ndarray[DTYPE_ti, ndim=3] occupied2):
    cdef int i,j,k,v
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    occupied2[x+i,y+j,z+k] = ii
                except: pass


cdef int searchBorder(int x,int y,int z,int ii,int sep,np.ndarray[DTYPE_ti, ndim=3] occupied2):
    cdef int i,j,k,v
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    v = occupied2[x+i,y+j,z+k]
                    if(v > 0 and v != ii): return True
                except:pass
    return 0


def add(Particle pi,int x,int y,int z,float randomParam,np.ndarray[DTYPE_ti, ndim=3] occupied2):
    cdef np.ndarray[DTYPE_tf, ndim=1] xp = np.zeros(3).astype(np.float32)
    cdef np.ndarray[DTYPE_ti, ndim=2] contorno
    cdef float d,de,deP,rr
    cdef int bestX,bestY,bestZ,xh,yh,zh,i

    ii = pi.i
    sep = 1#pi.sep()
    contorno = np.zeros((1,3)).astype(np.int32)#pi.contorno

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
                    if(rand()/float(RAND_MAX) >(1.0-randomParam)): contorno = np.vstack((contorno,np.array([[xh,yh,zh]]).astype(np.int32)))
    
    setBorder(x,y,z,ii,sep,occupied2)
    return np.vstack((pi.contorno,contorno[1:],np.array([[bestX,bestY,bestZ]]).astype(np.int32)))

def grow(Particle pi,float randomParam,np.ndarray[DTYPE_ti, ndim=3] occupied, np.ndarray[DTYPE_ti, ndim=3] occupied2):
        cdef int w = 0, h, r,nx,ny,nz
        cdef int fn,size,ii,sep
        cdef np.ndarray[DTYPE_ti, ndim = 2] contorno
        ii = pi.i
        size = pi.size
        contorno = pi.contorno
        sep = 1#pi.sep()
        fn = pi.fn()

        for r from 0 <= r < fn:
            for h from 0 <= h < len(contorno):
                w = h
                nx = contorno[h,0]
                ny = contorno[h,1]
                nz = contorno[h,2]
                try:
                    if(occupied[nx,ny,nz] <= 0 and not(searchBorder(nx,ny,nz,ii,sep,occupied2))):

                        occupied[nx,ny,nz] = np.uint8(255)
                        occupied2[nx,ny,nz] = ii
                        contorno = add(pi,nx,ny,nz,randomParam,occupied2)
                        size+=1
                        break                
                except: pass

            contorno = contorno[w+1:]

        pi.contorno = contorno
        pi.size = size





cdef class Particle:
    #cdef public float randomm
    #cdef public int i,size
    #cdef public np.ndarray contorno


    def __cinit__(self,int i,int lifet,float randomParam, np.ndarray[DTYPE_ti, ndim=3] occupied,np.ndarray[DTYPE_ti, ndim=3] occupied2):
        cdef int x,y,z
        cdef float dist,r,rv,tempfx,tempfy,distf,rm

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

        dist = floor(rv)
        r = 1.0*(rand()/rm)*(maxcoord/1.5-dist)

        x = np.clip(floor(x + r*(tempfx)),0,maxcoord-1);
        y = np.clip(floor(y + r*(tempfy)),0,maxcoord-1);

        self.i = i
        self.contorno = np.array([[-1,-1,-1]]).astype(np.int32)
        self.randomm = rand()/rm
        occupied[x,y,z] = np.uint8(255)
        occupied2[x,y,z] = i
        self.contorno = add(self,x,y,z,randomParam,occupied2)
        self.size = 1
        #self.occupied2=occupied2

        
    # Different separations depending on the bubble size
    def sep(self):
        return 1

    def fn(self):
        if(self.size < 5): return 1
        else: return 1+floor(0.01*self.size)



import numpy as np
cimport numpy as np
#from random import randint, random
#import maths
from runge_kutta import *
from globalsv import *

from libc.stdlib cimport rand

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti

cdef extern from "limits.h":
    int INT_MAX

cdef extern from "math.h":
    float pow(int x ,float y)
    int floor(float x)
    float sqrt(float x)

def sign():
    if(random() > 0.5): return -1
    return 1

def aCanvas(x):
    return floor(x*maxcoord)

particles = []
sparticles = [] # binary array with particle states (live or dead)

cdef int maxcoord = maxcoord
cdef int maxcoordZ = maxcoordZ
cdef int N = N


cdef setBorder(int x,int y,int z,int ii,int sep):
    cdef int i,j,k,pos,v
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    pos = np.int32((x+i)+(y+j)*maxcoord+(z+k)*maxcoord2)
                    occupied2[pos] = ii
                except: pass


cdef int searchBorder(int x,int y,int z,int ii,int sep):
    cdef int i,j,k,pos,v
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    pos = np.int32((x+i)+(y+j)*maxcoord+(z+k)*maxcoord2)
                    v = occupied2[pos]
                    if(v > 0 and v != ii): return True
                except:pass
    return False


cdef add(int x,int y,int z,float randomParam, int ii, int sep, np.ndarray[DTYPE_ti, ndim=2] contorno):
    cdef np.ndarray[DTYPE_ti, ndim=2] vals = contour(x,y,z)
    cdef np.ndarray[DTYPE_tf, ndim=1] xp
    cdef int pos = np.int32(x+y*maxcoord+z*maxcoord2)
    cdef float d,de,deP,rr
    cdef int bestX,bestY,bestZ,xh,yh,zh,i

    occupied[pos] = np.uint8(255)
    occupied2[pos] = ii

    d = sqrt(x*x+y*y+z*z)
    xp = np.zeros(3).astype(np.float32)
    xp[0] = toSpace(x)
    xp[1] = toSpace(y)
    xp[2] = toSpaceZ(z)
    xp = runge_kutta(xp,dT)
    bestX = vals[0,0]
    bestY = vals[0,1]
    bestZ = vals[0,2]
    de = np.float32(0)
    deP = calculatePriority(bestX,bestY,bestZ,xp)
    for i from 1 <=i<len(vals) :
        xh = vals[i,0]
        yh = vals[i,1]
        zh = vals[i,2]
        de = calculatePriority(xh,yh,zh,xp)
        if(de <deP):
            deP = de
            bestX = xh
            bestY = yh
            bestZ = zh
        #rr = rand()# / float(INT_MAX))*N)
        if(random() >(1.0-randomParam)): contorno = np.vstack((contorno,np.array([[xh,yh,zh]]).astype(np.int32)))
    
    setBorder(x,y,z,ii,sep)
    return np.vstack((contorno,np.array([[bestX,bestY,bestZ]]).astype(np.int32)))

          
cdef contour(int x,int y,int z):  
    # texels in the surroundings of x,y,z
    return np.array([
        [x,y+1,z],
        [x-1,y+1,z],
        [x+1,y+1,z],
        [x-1,y,z],
        [x+1,y,z],
        [x-1,y-1,z],
        [x,y-1,z],
        [x+1,y-1,z],
        [x-1,y+1,z],
        [x,y+1,z-1],
        [x+1,y+1,z-1],
        [x-1,y,z-1],
        [x+1,y,z-1],
        [x-1,y-1,z-1],
        [x,y-1,z-1],
        [x+1,y-1,z-1],
        [x,y,z-1],
        [x-1,y+1,z+1],
        [x,y+1,z+1],
        [x+1,y+1,z+1],
        [x-1,y,z+1],
        [x+1,y,z+1],
        [x-1,y-1,z+1],
        [x,y-1,z+1],
        [x+1,y-1,z+1],
        [x,y,z+1],
    ]).astype(np.int32)

cdef toSpace(int x):
    cdef float h
    h = x*(np.float32(diffX)/maxcoord)+x0
    return h


cdef toSpaceZ(int x):
    cdef float h
    h = x*(np.float32(diffZ)/maxcoord)+z0
    return h

cdef float calculatePriority(int x,int y,int z,xp): 
    return np.float32((xp[0] - toSpace(x))**2+(xp[1] - toSpace(y))**2+(xp[2] - toSpace(z))**2)




class Particle:
    def __init__(self,int i,int lifet,int u,int v,int w,float randomParam):
        cdef int x,y,z, tempx,tempy,tempz    
        cdef float dist,r,vx,vy,rv
        self.xi = randint(0,maxcoord)
        self.yi = randint(0,maxcoord)
        self.zi = randint(0,maxcoordZ)

        tempx = self.xi
        tempy = self.yi
        tempz = self.zi

        dist = floor(sqrt((tempx-maxcoord/2)*(tempx-maxcoord/2)+(tempy-maxcoord/2)*(tempy-maxcoord/2)))


        r = 1.0*random()*(maxcoord/1.5-dist)

        vx = tempx-maxcoord/2
        vy = tempy-maxcoord/2

        rv = sqrt(vx*vx+vy*vy)

        vx = vx/rv
        vy = vy/rv

        self.xi = floor(tempx + r*(vx));
        self.yi = floor(tempy + r*(vy));


        x = self.xi
        y = self.yi
        z = self.zi
        if(u >= 0):
            x = u
            y = v
            z = w

        if(x < 0): x = 0
        if(y < 0): y = 0
        if(z < 0): z = 0
        if(x >= maxcoord): x = maxcoord-1
        if(y >= maxcoord): y = maxcoord-1
        if(z >= maxcoordZ): z = maxcoordZ-1
        self.i = i
        self.contorno = np.array([[-1,-1,-1]]).astype(np.int32)
        self.tiempoDeVida = lifet
        self.tActual = 0
        self.size = 0
        self.tInit = t
        self.randomm = random()
        self.repr = 0
        self.contorno = add(x,y,z,randomParam, self.i, self.sep(), self.contorno)
        self.size += 1

    def grow(self,float randomParam):
        cdef int w = 0, h, nx,ny,nz,pos
        self.tActual += 1
        for h from 0 <= h < len(self.contorno):
            w = h
            cont = self.contorno[h]
            nx = cont[0]
            ny = cont[1]
            nz = cont[2]
            try:
                pos = np.int32(nx+ny*maxcoord+nz*maxcoord2)
                if(not(ocupada(pos)) and not(searchBorder(nx,ny,nz,self.i,self.sep()))):
                    self.contorno = add(nx,ny,nz,randomParam, self.i, self.sep(), self.contorno)
                    self.size+=1
                    break                
            except: pass

        self.contorno = self.contorno[w+1:]
    
    def morir(self):
        sparticles[self.i] = False

    def alive(self):
        return sparticles[self.i]

    # Different separations depending on the bubble size
    def sep(self):
        return 1

    def fn(self):
        if(self.size < 5): return 1
        else: return 1+floor(0.01*self.size)



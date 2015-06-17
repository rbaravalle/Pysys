# cython: profile=True

import numpy as np
cimport numpy as np
from runge_kutta cimport *
from globalsv import *
import matplotlib.pyplot as plt
import Image
import time


from libc.stdlib cimport rand, RAND_MAX

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti

cdef float dXm1 = dXm1
cdef float dYm1 = dYm1
cdef float dZm1 = dZm1
cdef float x0 = x0
cdef float y0 = y0
cdef float z0 = z0
cdef float randomnessZ = randomnessZ

cdef extern from "math.h":
    float pow(int x ,float y)
    int floor(float x)
    int round(float x)
    float sqrt(float x)

cdef int maxcoord = maxcoord
cdef int maxcoordZ = maxcoordZ
cdef int N = N


cdef setBorder(Particle pi,int x,int y,int z):
    cdef int i,j,k,sep,ii
    ii = pi.i
    sep = pi.sep()#2
    cdef np.ndarray[DTYPE_ti, ndim=3] occ = pi.occupied2
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    occ[x+i,y+j,z+k] = ii
                except: pass


cdef int searchBorder(Particle pi, int x,int y,int z, list particles):
    cdef int i,j,k,v,sep,ii
    sep = pi.sep()#2
    ii = pi.i
    cdef np.ndarray[DTYPE_ti, ndim=3] occ = pi.occupied2
    for i from -sep<=i<sep:
        for j from -sep<=j<sep:
            for k from -sep<=k<sep:
                try:
                    v = occ[x+i,y+j,z+k]
                    if(v > 0 and v != ii): return True
                    #if(4*particles[v].size < particles[i].size): return 0
                except:pass
    return 0


cdef add(Particle pi,int x,int y,int z):

    cdef list contorno
    cdef float d,de,deP,de2,de2P, rr,xp0,xp1,xp2,xt,yt,zt,inv0,inv1,inv2
    cdef int bestX,bestY,bestZ,best2X,best2Y,best2Z,xh,yh,zh,i, temp#,isBestAdded
    # to avoid repeating elements
    #isBestAdded = 0

    contorno = pi.contorno

    cdef float cx = pi.l[z][0], cy = pi.l[z][1]
    xp0,xp1,xp2 = runge_kutta(x,y,z,cx,cy)
    #inv0,inv1,inv2 = runge_kuttainv(x,y,z,cx,cy)


    skip = (slice(None, None, 1), slice(None, None, 1), z)

    tempx = pi.dx[skip]
    tempy = pi.dy[skip]

    # tempx-> to GRID, GRID operation with x -> to SPACE (xp)
    if(tempx[x,y] > 0.01 or tempy[x,y] > 0.01):
        xp0 = (x+0.5*(tempy[x,y])/dXm1)*dXm1+(x0+cx)
        xp1 = (y-0.5*(tempx[x,y])/dYm1)*dYm1+(y0+cy)

        #inv0 = (x+0.5*(tempy[x,y])/dXm1)*dXm1+(x0+cx)
        #inv1 = (y-0.5*(tempx[x,y])/dYm1)*dYm1+(y0+cy)

    sliceN = 30
    if(False and z == sliceN):

        I = Image.frombuffer('L',(maxcoord,maxcoord), np.array(pi.geom[:,:,sliceN]).astype(np.uint8),'raw','L',0,1)
        fig, ax = plt.subplots()


        xx, yy,zz = np.mgrid[0:256:256j, 0:256:256j,0:256:256j]

        im = ax.imshow(I, extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        plt.colorbar(im)

        # orthogonal to the dfield
        ax.quiver(xx[skip], yy[skip], tempy.T, -tempx.T)

        ax.set(aspect=1, title='Quiver Plot')
        plt.show()
        exit()

    bestX=bestY=bestZ=deP=10000
    for xh from x-1<=xh<=x+1:
        for yh from y-1<=yh<=y+1:
            for zh from z-1<=zh<=z+1:
                if(not(xh==x and yh==y and zh==z)):
                    cx = pi.l[zh][0]
                    cy = pi.l[zh][1]
                    xt = (xp0 - (xh*(dXm1)+(x0+cx)))
                    yt = (xp1 - (yh*(dYm1)+(y0+cy)))
                    zt = (xp2 - (zh*(dZm1)+z0))
                    de = float(xt*xt+yt*yt+zt*zt)
                    if(de <deP):
                        deP = de
                        bestX = xh
                        bestY = yh
                        bestZ = zh
                        #isBestAdded = 0

                        #isBestAdded = 1

                    # inverse
                    #xt = (inv0 - (xh*(dXm1)+(x0+cx)))
                    #yt = (inv1 - (yh*(dYm1)+(y0+cy)))
                    #zt = (inv2 - (zh*(dZm1)+z0))
                    #de2 = float(xt*xt+yt*yt+zt*zt)
                    #if(de2 <de2P):
                    #    de2P = de2
                    #    best2X = xh
                    #    best2Y = yh
                    #    best2Z = zh

                    # random!
                    if(rand()/float(RAND_MAX) >(1.0-pi.randomParam)):# and zh == z):
                        contorno.append([xh,yh,zh])
                  
    
    setBorder(pi,x,y,z)
    #if(not(isBestAdded)):
    contorno.append([bestX,bestY,bestZ])

    # randomZ!
    if(rand()/float(RAND_MAX) >(1.0-randomnessZ)):# and zh == z):
        temp = -1
        if(rand()/float(RAND_MAX) > 0.5):
            temp = 1
        contorno.append([xh,yh,zh+temp])
    #contorno.append([best2X,best2Y,best2Z])
    return contorno

def grow(Particle pi, list particles):
        cdef int w = 0, h, r,nx,ny,nz,lenc,fn,size,ii
        cdef list contorno
        ii = pi.i
        size = pi.size
        contorno = pi.contorno
        fn = pi.fn()

        for r from 0 <= r < fn:
            lenc = len(contorno)
            for h from 0 <= h < lenc:
                nx = contorno[h][0]
                ny = contorno[h][1]
                nz = contorno[h][2]
                try:
                    if(pi.occupied[nx,ny,nz] > 0 and not(searchBorder(pi,nx,ny,nz,particles))):

                        pi.occupied[nx,ny,nz] = 0
                        pi.occupied2[nx,ny,nz] = ii
                        contorno = add(pi,nx,ny,nz)
                        size+=1
                        break                
                except: pass

            del contorno[0:h]

        pi.contorno = contorno
        pi.size = size





cdef class Particle:

    def __cinit__(self,int i,float randomParam, int sep, np.ndarray[DTYPE_t, ndim=3] occupied,np.ndarray[DTYPE_ti, ndim=3] occupied2,np.ndarray[DTYPE_tf, ndim=3] dx,np.ndarray[DTYPE_tf, ndim=3] dy,np.ndarray[DTYPE_tf, ndim=3] dz,np.ndarray[DTYPE_t, ndim=3] geom, list l, list l2):
        cdef int x,y,z,dist
        cdef float r,rv,tempfx,tempfy,rm,cx,cy

        rm = float(RAND_MAX)

        z = int((maxcoordZ-1)*(rand()/rm))

        cx = l2[z][0]#-maxcoord/2)*dXm1#maxcoord/2
        cy = l2[z][1]#maxcoord/2

        cdef float delta = 0.7*maxcoord/2
        x =  np.clip(int((2*delta)*(rand()/rm)) + (cx-delta),0,maxcoord-1)
        y =  np.clip(int((2*delta)*(rand()/rm)) + (cy-delta),0,maxcoord-1)

        #print x, y

        # FIX ME!
        #x = int((maxcoord-1)*(rand()/rm))
        #y = int((maxcoord-1)*(rand()/rm))

        #t2 = time.clock()
        #print "Entra al while..."
        #if(occupied[:,:,z].sum()>0):
        #    while((occupied2[x,y,z] == 1 or occupied[x,y,z]==0) and time.clock()-t2 <1):
                #x = int((maxcoord-1)*(rand()/rm))
                #y = int((maxcoord-1)*(rand()/rm))
        #        x =  np.clip(int((2*delta)*(rand()/rm)) + (cx-delta),0,maxcoord-1)
        #        y =  np.clip(int((2*delta)*(rand()/rm)) + (cy-delta),0,maxcoord-1)
            #print x, y
            #z = int((maxcoordZ-1)*(rand()/rm))
        #print "Sale del while..."


        #tempfx = float(x-cx)
        #tempfy = float(y-cy)

        # normalize
        #rv = sqrt(tempfx*tempfx+tempfy*tempfy)
        #if(rv!=0):
        #    tempfx = tempfx/rv
        #    tempfy = tempfy/rv

        #r = 0.1*(rand()/rm)*(maxcoord/1.5-rv)

        #x = np.clip(floor(x + r*(tempfx)),0,maxcoord-1);
        #y = np.clip(floor(y + r*(tempfy)),0,maxcoord-1);

        self.i = i
        self.contorno = [[-1,-1,-1]]
        self.randomm = rand()/rm
        occupied[x,y,z] = 0
        occupied2[x,y,z] = i
        self.randomParam = randomParam
        self.size = 1
        self.occupied2=occupied2
        self.occupied=occupied
        self.sepp = sep
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.geom=geom
        self.l = l
        self.contorno = add(self,x,y,z)

        
    # Different separations, depending on bubble size
    def sep(self):
        return self.sepp
        #if(self.size > 140): return 2
        #return 1


    cdef fn(self):
        if(self.size < 5): return 1
        else: return 1+round(20.0*self.randomm)#*self.size)



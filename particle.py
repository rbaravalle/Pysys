import numpy as np
from random import randint, random
from math import floor
import maths
from runge_kutta import *

def toSpace(x):
    from globalsv import diffX, maxcoord, x0
    return x*(diffX/maxcoord)+x0


def toSpaceZ(x):
    from globalsv import diffZ, maxcoord, z0
    return x*(diffZ/maxcoord)+z0


class Particle:
    def __init__(self,i,lifet):
        from globalsv import *
        c = randint(0,len(generadores)-1)
        gx = generadores[c][0]
        gy = generadores[c][1]
        gz = generadores[c][2]
           
        self.xi = floor(gx + distG*(random()*2-1)*maxcoord)
        self.yi = floor(gy + distG*(random()*2-1)*maxcoord)
        self.zi = floor(gz + distG*(random()*2-1)*maxcoordZ)

        x = self.xi
        y = self.yi
        z = self.zi
        if(x < 0): x = 0
        if(y < 0): y = 0
        if(z < 0): z = 0
        if(x >= maxcoord): x = maxcoord-1
        if(y >= maxcoord): y = maxcoord-1
        if(z >= maxcoordZ): z = maxcoordZ-1

        self.i = i

        self.contorno = []

        self.tiempoDeVida = lifet
        self.tActual = 0

        self.tInit = t
        self.randomm = random()

        self.add(x,y,z)

    def add(self,x,y,z):
        from globalsv import *
        pos = np.int32(x+y*maxcoord+z*maxcoord2)
        if(occupied[pos] != np.int32(0)): 
            print occupied[pos]
            return
        # texels in the surroundings of the added point
        vals = [
            [x-1,y+1,z],
            [x,y+1,z],
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
        ]


        # Alpha blending?
        pos = np.int32(x+y*maxcoord+z*maxcoord2)

        occupied[pos] = np.int32(255)
        #occupied[pos].a = self.a

        d = sqrt(x*x+y*y+z*z)
        xp = np.zeros(3)
        xp[0] = toSpace(x)
        xp[1] = toSpace(y)
        xp[2] = toSpaceZ(z)
        xp = runge_kutta(xp,factual,dT)
        bestX = vals[0][0]
        bestY = vals[0][1]
        bestZ = vals[0][2]
        de = 0
        deP = self.calculatePriority(bestX,bestY,bestZ,xp)
        for i in range(1,len(vals)) :
            xh = vals[i][0]
            yh = vals[i][1]
            zh = vals[i][2]
            de = self.calculatePriority(xh,yh,zh,xp)
            if(de <deP):
                deP = de
                bestX = xh
                bestY = yh
                bestZ = zh
            
            if(random()>(1-randomness)): self.contorno.append([xh,yh,zh,de])
        

        self.contorno.append([bestX,bestY,bestZ,deP])
        self.setBorder(x,y,z)

    def calculatePriority(self,x,y,z,xp): 
        x2 = xp[0] - toSpace(x)
        y2 = xp[1] - toSpace(y)
        z2 = xp[2] - toSpace(z)
        # priorities
        return x2*x2+y2*y2+z2*z2

    def setBorder(self,x,y,z):
        from globalsv import *
        for i in (-sep,sep):
            for j in (-sep,sep):
                for k in (-sep,sep):
                    if(x+i>= 0 and x+i <maxcoord and y+j>=0 and y+j<maxcoord and z+k>=0 and z+k < maxcoordZ):
                        pos = np.int32((x+i)+(y+j)*maxcoord+(z+k)*maxcoord2)
                        occupied2[pos] = self.i


    def searchBorder(self,x,y,z):
        from globalsv import *
        for i in (-sep,sep):
            for j in (-sep,sep):
                for k in (-sep,sep):
                    if(x+i>= 0 and x+i <maxcoord and y+j>=0 and y+j<maxcoord and z+k>=0 and z+k < maxcoordZ):
                        pos = np.int32((x+i)+(y+j)*maxcoord+(z+k)*maxcoord2)
                        v = occupied2[pos]
                        if(v != self.i): return True
        return False

    def grow(self):
        from globalsv import *
        self.tActual = self.tActual + 1
        maxim = len(self.contorno)
        w = 0
        for h in range(0,maxim):
            w = h
            cont = self.contorno[h]
            nx = cont[0]
            ny = cont[1]
            nz = cont[2]
            pos = np.int32(nx+ny*maxcoord+nz*maxcoord2)
            o = occupied[pos]
            if(ocupada(pos) == False):
                if(self.searchBorder(nx,ny,nz) == False):
                    self.add(nx,ny,nz)
                    break                
            
        
        self.contorno = self.contorno[w:]
    
    def morir(self):
        from globalsv import *
        sparticles[self.i] = False

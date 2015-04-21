import Image
import numpy as np
cimport numpy as np

from math import floor, sqrt
from random import randint, random
import matplotlib
from matplotlib import pyplot as plt
from particle import Particle, particles, sparticles
from maths import *
from runge_kutta import *
from globalsv import *
from time import time

ctypedef np.uint8_t DTYPE_t
ctypedef np.int32_t DTYPE_ti
ctypedef np.float32_t DTYPE_tf

# How much should the particle grow?
cdef fn(int sizei,float randomm):
    cdef float size
    size = sizei/8
    if(randomm > 0.9):
        return np.floor(size).astype(np.int32)
    else: return 1

def init_particles():
    cdef int k = 0
    cdef int i = 0, h, j
    for i from 0<= i< cantPart:
        particles.append(Particle(k,MCA,-1,-1,-1,0.15))
        k = k+1
        sparticles.append(True)

    for i from 0<= i < len(particles):
        for h from 0<=h<4:
            if(random() > 0.9):
                for j from 0<=j<diffBubbles:
                    particles[i].grow(0.15) # free growth



# una iteracion del algoritmo
cdef mover(int t) :
    cdef int largoCont = 0, sizes = 0
    timm = time()
    cdef int suma = 0,k,w,i,u,v,s,sizesAnt=0
    cdef float rr,d,e
    for i from 0<=i<= len(particles)-1:
        pi = particles[i]
        if(pi.alive()):
            if(pi.size > pi.tiempoDeVida):
                pi.morir()
            if(pi.repr == 0 and pi.size > np.floor(MCA/2)):
                pi.repr = 1
                k = len(particles)
                for w from 0<=w< amountSons:
                    d = random()*np.pi*2
                    e = random()*np.pi*2
                    rr = 5*(np.sqrt(pi.size/np.pi))
                    if(pi.contorno.size > 0):
                        if(pi.contorno.size >= w+1):
                            con = pi.contorno[w]
                        else: con = pi.contorno[0]
                    else : con = np.array([pi.xi,pi.yi,pi.zi,0]).astype(np.int32)

                    # The son appears close to the father
                    u = con[0]+np.int32(rr*np.cos(d)*np.sin(e))
                    v = con[1]+np.int32(rr*np.sin(d)*np.sin(e))
                    s = con[2]+np.int32(rr*np.cos(e))

                    particles.append(Particle(k, MCA,u,v,s,randomness));
                    k +=1
                    sparticles.append(True); # The particle is alive
            suma += fn(pi.size,pi.randomm)
            #for w from 0<= w<fn(pi.size,pi.randomm):
            pi.grow(randomness)
            largoCont += pi.contorno.size
            sizes += pi.size

    print "Iteracion :",t
    print "TIME : ", time()-timm
    print "LLAMADAS-2: ", sizes-sizesAnt, sizes


    return largoCont, sizes
  


def dibujarParticulas() :

    print "Saving Image..."

    I = Image.new('L',(maxcoordZ,maxcoord2),0.0)

    for i from 0<=i<maxcoordZ:
        I2 = Image.frombuffer('L',(maxcoord,maxcoord), np.uint8(255)-np.array(occupied[maxcoord2*i:maxcoord2*(i+1)]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,maxcoord*i))

    I.save('textures/imagenSystemPaper.png')

cdef alg() :  
    cdef int largoCont,t,sizes,diffs=0
    for t from 0<=t<TIEMPO-1:
        
        largoCont,sizes = mover(t)
        print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , np.array(sparticles).sum(), "LLAMADAS: ", sizes, sizes-diffs
        diffs = sizes
        if(t % 6 == 0): dibujarParticulas()   
        if(largoCont == 0):
            break

    print "last draw!"
    dibujarParticulas()
    print "good bye!"

   
def main():
    print "Init Variables..."
    init_variables()
    print "Init Particles..."
    init_particles()
    print "Algorithm!"
    alg()


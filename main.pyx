import Image
import numpy as np
from math import floor, sqrt
from random import randint, random
import matplotlib
from matplotlib import pyplot as plt
from particle import Particle, particles, sparticles
from sist import init_particles
from maths import *
from runge_kutta import *
from globalsv import *
from time import time


def init_particles():
    cdef int i = 0, h, j
    for i from 0<= i< cantPart:
        particles.append(Particle(i,MCA,-1,-1,-1,0.15))
        sparticles.append(True)

    for i from 0<= i < len(particles):
        for h from 0<=h<4:
            if(random() > 0.8):
                for j from 0<=j<diffBubbles:
                    particles[i].grow(0.15) # free growth


# una iteracion del algoritmo
cdef mover(t) :
    cdef int largoCont, suma,i,w,k,temp
    cdef float timm, d,e,rr
    largoCont = 0
    timm = time()
    suma = 0
    for i in range(len(particles)):
        pi = particles[i]
        if(pi.alive()):
            if(pi.repr == 0 and pi.size > np.floor(MCA/2)):
                pi.repr = 1
                k = len(particles)
                for w in range(amountSons):
                    print amountSons
                    d = random()*np.pi*2
                    e = random()*np.pi*2
                    rr = 5*(np.sqrt(pi.size/np.pi))
                    if(pi.contorno.size > 0):
                        if(pi.contorno.size >= w+1):
                            con = pi.contorno[w]
                        else: con = pi.contorno[0]
                    else : con = [pi.xi,pi.yi,pi.zi,0]

                    # The son appears close to the father
                    u = con[0]+np.int32(rr*np.cos(d)*np.sin(e))
                    v = con[1]+np.int32(rr*np.sin(d)*np.sin(e))
                    s = con[2]+np.int32(rr*np.cos(e))

                    particles.append(Particle(k, MCA,u,v,s,randomness));
                    k +=1
                    sparticles.append(True); # The particle is alive

            temp = pi.fn()
            suma += temp
            for w from 0<=w<temp:
                pi.grow(randomness)
            largoCont += pi.contorno.size

    print "Iteracion :",t
    print "TIME : ", time()-timm
    #print "LLAMADAS: ", suma


    return largoCont
  


def dibujarParticulas() :

    print "draw!"

    I = Image.new('L',(maxcoordZ,maxcoord2),0.0)

    for i in range(maxcoordZ):
        I2 = Image.frombuffer('L',(maxcoord,maxcoord), np.uint8(255)-np.array(occupied[maxcoord2*i:maxcoord2*(i+1)]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,maxcoord*i))

    I.save('textures/imagenSystemPaper.png')

def alg() :  

    for t in range(0,TIEMPO-1):
        largoCont = mover(t)
        t = t+1
        #if(t % 40 == 0) : 
        print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , len(particles)
        if(t % 6 == 0): dibujarParticulas()   
        if(largoCont == 0):
            break

    print "last draw!"
    dibujarParticulas()
    print "good bye!"
    #exit()

   
def main():
    print "Init Particles..."
    init_particles()
    print "Algorithm!"
    alg()

# start
main()

import Image
import numpy as np
from math import floor, sqrt
from random import randint, random
import matplotlib
from matplotlib import pyplot as plt
from particle import Particle, init_particles, particles, sparticles
from maths import *
from runge_kutta import *
from globalsv import *


# una iteracion del algoritmo
def mover() :
    largoCont = 0
    for i in range(0,len(particles)):
        pi = particles[i]
        if(pi.alive()):
            if(pi.size > 6*MCA):
                pi.morir()
                m.push(i);
            if(pi.repr == 0 and pi.size > np.floor(MCA/2)):
                pi.repr = 1
                k = len(particles)
                for w in range(amountSons):
                    d = random()*np.pi*2
                    d = random()*np.pi*2
                    e = random()*np.pi*2
                    rr = 5*(np.sqrt(pi.size/np.pi))
                    if(len(pi.contorno) >= 0):
                        if(len(pi.contorno) <= w+1):
                            con = pi.contorno[w]
                        else: con = pi.contorno[0]
                    else : con = [pi.xi,pi.yi,pi.zi,0]

                    u = con[0]+np.int32(rr*np.cos(d)*np.sin(e))
                    v = con[1]+np.int32(rr*np.sin(d)*np.sin(e))
                    s = con[2]+np.int32(rr*np.cos(e))

                    particles.append(Particle(k, MCA,u,v,s,randomness));
                    k +=1
                    sparticles.append(true); # la particula esta viva

            for w in range(pi.fn()):
                pi.grow(randomness)
            largoCont = largoCont + len(pi.contorno)

    return largoCont
  



def dibujarParticulas() :

    print "draw!"

    I = Image.new('L',(maxcoordZ,maxcoord2),0.0)

    for i in range(maxcoordZ):
        I2 = Image.frombuffer('L',(maxcoord,maxcoord), np.uint8(255)-np.array(occupied[maxcoord2*i:maxcoord2*(i+1)]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,maxcoord*i))

    I.save('imagen3-1.png')

def alg() :  

    for t in range(0,TIEMPO-1):
        largoCont = mover()
        t = t+1
        if(t % 40 == 0) : print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , len(particles)
        if(t % 40 == 0): dibujarParticulas()   
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

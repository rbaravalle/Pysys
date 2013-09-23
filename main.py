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
import viz # render


# una iteracion del algoritmo
def mover() :
    largoCont = 0
    for i in range(0,len(particles)):
        pi = particles[i]
        if(pi.alive()):
            if(pi.tActual > pi.tiempoDeVida) :
                pi.morir()
            else : 
                pi.grow()
                largoCont = largoCont + len(pi.contorno)

    return largoCont
  



def dibujarParticulas() :

    print "draw!"

    I = Image.new('L',(maxcoordZ,maxcoord2),0.0)

    for i in range(maxcoordZ):
        I2 = Image.frombuffer('L',(maxcoord,maxcoord), np.uint8(255)-np.array(occupied[maxcoord2*i:maxcoord2*(i+1)]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,maxcoord*i))

    I.save('../webgl-volumetric/textures/imagen.png')

def alg() :  

    for t in range(0,TIEMPO-1):
        largoCont = mover()
        t = t+1
        if(t % 40 == 0) : print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , len(particles)
        if(t % 100 == 0): dibujarParticulas()   
        if(largoCont == 0):
            break

    print "last draw!"
    dibujarParticulas()
    print "good bye!"
    #exit()

   
def main():
    print "Init..."
    init_variables()
    print "Init Particles..."
    init_particles()
    print "Algorithm!"
    alg()

# start
main()
viz.viz()

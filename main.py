import Image
import numpy as np
from math import floor, sqrt
from random import randint, random
import matplotlib
from matplotlib import pyplot as plt
from particle import Particle
from maths import *
from runge_kutta import *
from globalsv import *

# una iteracion del algoritmo
def mover() :
    m = []
    largoCont = 0
    for i in range(0,len(particles)):
        pi = particles[i]
        if(pi.tActual > pi.tiempoDeVida) :
            pi.morir()
            m.append(i) 
        else : 
            pi.grow()
            largoCont = largoCont + len(pi.contorno)

    #for i in range(0,len(m)):
    #    del particles[m[i]]

    return largoCont
  



def dibujarParticulas() :

    print "draw!"

    I = Image.new('L',(maxcoordZ,maxcoord2),0.0)
    rowsPerSlice = maxcoord

    for i in range(maxcoordZ):
        I2 = Image.frombuffer('L',(maxcoord,maxcoord), np.int32(255.0)-np.array(occupied[maxcoord2*i:maxcoord2*(i+1)]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,rowsPerSlice*i))

    I.save('imagen.png')

def alg() :  

    for t in range(0,TIEMPO-1):
        largoCont = mover()
        t = t+1
        if(t % 40 == 0) : print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , len(particles)
        if(t % 300 == 0): dibujarParticulas()
        if(len(particles) == 0) : break

    print "good bye!"
    exit()

   
def main():
    print "Init..."
    init_variables()
    print "Algorithm!"
    alg()

# start
main()

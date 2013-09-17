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
    largoCont = 0
    m = []
    for i in range(0,len(particles)):
        pi = particles[i]
        if(pi.tActual > pi.tiempoDeVida) :
            pi.morir()
            m.append(i) 
        else : 
            pi.grow()
            largoCont = largoCont + len(pi.contorno)

    print largoCont
    for i in range(0,len(m)):
        del particles[m[i]]
  



def dibujarParticulas() :

    print "we'll draw soon!"
    #print occupied[10:20]
    # we only print those pixels that lie in the visZ plane
    #for(var j = maxcoord2*visZ j < maxcoord2*(visZ+1) j++) :
    #    if(occupied[j].particle > 0) :
    #        var p = occupied[j]
    #        j2 = j - maxcoord2*visZ
    #        var x = j2%maxcoord
    #        var y = Math.floor(j2*m1)
    #        
    #        vertices.push(x*m1,y*m1,0.0)
    #        colors.push(p.r,p.g,p.b,1.0)
    #        cant++
    

def alg() :  

    for t in range(0,TIEMPO-1):
        print "IT"
        mover()
        t = t+1
        if(len(particles) == 0) : break

    dibujarParticulas()
    print "good bye!"
    exit()

   
def main():
    print "Init..."
    init_variables()
    print "Algorithm!"
    alg()

# start
main()

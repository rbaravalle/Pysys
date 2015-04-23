import Image
import numpy as np
cimport numpy as np
from random import randint, random

from particle cimport Particle
from particle import grow


from runge_kutta import *
from globalsv import *
import time

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t DTYPE_tf
ctypedef np.int32_t DTYPE_ti

cdef init_particles():

    cdef np.ndarray[DTYPE_ti, ndim=3] occupied = np.zeros((maxcoord,maxcoord,maxcoord)).astype(np.int32)
    cdef np.ndarray[DTYPE_ti, ndim=3] occupied2 = np.zeros((maxcoord,maxcoord,maxcoord)).astype(np.int32)-np.int32(1) # contourns occupied (for self-avoidance)

    cdef int i = 0, h, j
    cdef Particle pi
    timm = time.clock()
    cdef list particles = []
    for i from 0<= i< cantPart:
        pi = Particle(i,MCA,0.15,occupied,occupied2)

        if(pi.randomm > 0.8):
            for j from 0<=j<diffBubbles:
                grow(pi,randomness,occupied,occupied2) # free growth
        if(i%(2000)==0):
            print "Time: ", time.clock()-timm,i
            timm = time.clock()

        particles.append(pi)

    return particles,occupied,occupied2


# una iteracion del algoritmo
cdef mover(t,particles, np.ndarray[DTYPE_ti, ndim=3] occupied,np.ndarray[DTYPE_ti, ndim=3] occupied2) :
    cdef int largoCont, suma,i,w,k,temp
    cdef float timm, d,e,rr
    largoCont = 0
    timm = time.clock()
    cdef Particle pi

    for i from 0<=i<cantPart:
        pi = particles[i]
        grow(pi,randomness,occupied,occupied2)
        largoCont += pi.contorno.size

    print "Iteracion :",t
    print "TIME : ", time.clock()-timm


    return largoCont
  


cdef dibujarParticulas(np.ndarray[DTYPE_ti, ndim=3] occupied) :

    print "draw!"

    I = Image.new('L',(maxcoordZ,maxcoord2),0.0)

    for i in range(maxcoordZ):
        I2 = Image.frombuffer('L',(maxcoord,maxcoord), np.uint8(255)-np.array(occupied[:,:,i]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,maxcoord*i))

    I.save('textures/imagenSystemPaper.png')

cdef alg(particles,np.ndarray[DTYPE_ti, ndim=3] occupied,np.ndarray[DTYPE_ti, ndim=3] occupied2) :  
    for t from 0<=t<TIEMPO-1:
        largoCont = mover(t,particles,occupied,occupied2)
        print "It ", t , "/" , TIEMPO , ", Contorno: " , largoCont , " Cant Part: " , len(particles)
        if(t % 6 == 0): dibujarParticulas(occupied)   
        if(largoCont == 0):
            break

    print "last draw!"
    dibujarParticulas(occupied)
    print "good bye!"

   
cdef main():

    print "Init Particles..."
    particles,occupied,occupied2 = init_particles()
    print "Algorithm!"
    alg(particles,occupied,occupied2)

# start
main()

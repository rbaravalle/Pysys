import numpy as np
from maths import *
from random import randint, random
from math import floor, sqrt
from particle import Particle

# global vars
maxcoord = 100
maxcoordZ = 100
maxcoord2 = maxcoord*maxcoord
maxcoord3 = maxcoord2*maxcoordZ
m1 = 1.0/maxcoord
largoCont = 0
occupied = []
t = 0
tUlt = 0
particles = []
sparticles = [] # binary array with particle states (live or dead)

vertices = []
N = 10
cp = np.zeros(N) # # of particles per size (see below)
lt = np.zeros(N) # lifetime of particles
CT = 0
occupied2 = [] # contorns occupied (for self-avoidance)
tOrig = 0
distG = 0
cantG = 0 # amount of generators
sembrado = 0 # random or uniform
VF = 1.8
MCA = 5000
stCA = 6000
randomness = 0.1

d = 0
t1 = 0

#2D-world limits
x0 = -3
y0 = -3
x1 = 3
y1 = 3
z0 = -1
z1 = 1
diffX = x1-x0
diffY = y1-y0
diffZ = z1-z0

generadores = []

TIEMPO_VIDA = 0
TIEMPO = 1000

#gl
sep = 5 # separation among particles
visZ = 0

def compute_lifetimes() :
    M = stCA*stCA

    s = sign()
    # compute lifetimes
    for i in range(0,N-1):
        x = randint(0,floor(M))
        M = M - x
        s = -s
        lt[i] = floor(s*sqrt(x)) + MCA
    summ = 0
    CT2 = floor(VF*maxcoord3/MCA)
    # calculate amount of particles per lifetime
    for i in range(0,N-2):
        cp[i] = randint(0,CT2)
        CT2 = CT2 - cp[i]
        summ = summ + cp[i]*lt[i] 
    
    if(lt[N-1] > 0) : cp[N-1] = floor((VF*maxcoord3 - summ)/lt[N-1])



def init_variables() :

    compute_lifetimes()
    cantG = 100
    t = 0

    for i in range(0,maxcoord3): 
        occupied.append(np.int32(-1))
        occupied2.append(np.int32(0))

    if(sembrado == 0) :
        for i in range(0,cantG):
            generadores.append([randint(0,maxcoord), randint(0,maxcoord),randint(0,maxcoord)])
    else :
        step = np.float32(maxcoord/cantG)
        for i in range(0,maxcoord,step):
            for j in range(0,maxcoord,step):
                for k in range(0,maxcoordZ,step):
                   generadores.append([floor(i),floor(j),floor(k)])
    tOrig = TIEMPO_VIDA
    init_particles()
    #alg() 

def init_particles():
    k = 0
    for i in range(0,N):
        for j in range(0,np.int32(cp[i])):
            particles.append(Particle(k,lt[i]))
            k = k+1
            sparticles.append(True)



def ocupada(i):
    o = occupied[i]
    return (o >= 0 and sparticles[o])

init_variables()

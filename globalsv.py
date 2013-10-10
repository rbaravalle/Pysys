import numpy as np
from maths import *
from random import randint, random
from math import floor, sqrt

# global vars
maxcoord = 150
maxcoordZ = 150
maxcoord2 = maxcoord*maxcoord
maxcoord3 = maxcoord2*maxcoordZ
m1 = 1.0/maxcoord
occupied = np.zeros(maxcoord3)
occupied2 = np.zeros(maxcoord3)-np.int32(1) # contorns occupied (for self-avoidance)
t = 0

N = 10
cp = np.zeros(N) # # of particles per size (see below)
lt = np.zeros(N) # lifetime of particles
CT = 0
distG = 0.8
cantG = 20 # amount of generators
sembrado = 0 # random or uniform
VF = 0.99
MCA = 10000
stCA = 12000
randomness = 0.03

#2D-world limits
x0 = -25
y0 = -25
x1 = 25
y1 = 25
z0 = 0
z1 = 50
diffX = x1-x0
diffY = y1-y0
diffZ = z1-z0

generadores = np.array([[0,0,0]])

TIEMPO = 120000
sep = 1 # separation among particles

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
    print "Cp: ", cp
    print "Lt: ", lt

def init_variables() :
    global generadores
    compute_lifetimes()

    #for i in range(0,maxcoord3): 
        #np.append(occupied,np.int32(0))
        #np.append(occupied2,np.int32(0))


    if(sembrado == 0) :
        for i in range(0,cantG):
            b = np.array([[randint(0,maxcoord), randint(0,maxcoord),randint(0,maxcoordZ)]])
            generadores = np.concatenate((generadores,b), axis = 0)

        
    else :
        step = np.float32(maxcoord/cantG)
        for i in range(0,maxcoord,step):
            for j in range(0,maxcoord,step):
                for k in range(0,maxcoordZ,step):
                    b = np.array([[floor(i),floor(j),floor(k)]])
                    generadores = np.concatenate((generadores,b), axis = 0)

    generadores = generadores[1:] # take out the dummy (0,0,0)
    print generadores

def ocupada(i):
    return (occupied[i] > 0)

print "Init..."
init_variables()

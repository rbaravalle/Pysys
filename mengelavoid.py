import Image
import numpy as np
from math import floor, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt
import random


maxX = 512
maxY = 512

I = Image.new('L',(maxX,maxY),0.0)

field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 24 # radius of initial bubbles
c = 2 # amount of initial bubbles
orig = c

numIt = 5
h = 0
h2 = 0
cinf = 4

maxa = -10
mina = 100000

points2 = np.zeros((c*2)).astype(np.uint32)
points3 = np.zeros((c*cinf)).astype(np.float32)

def detcuad(i,j):
    if(i > maxX/2): 
        if(j > maxY/2): return 1
        else: return 2
    else:
        if(j > maxY/2): return 3
        else: return 4

# self-avoiding growth model
def avoid(i,j,rac):
    global maxa, mina
    summ = np.float32(0)
    for pos in range(0,len(pointstot),cinf):
        if(detcuad(i,j) == pointstot[pos+3]):
            d = np.power((i - pointstot[pos]),2)+np.power((j - pointstot[pos+1]),2)
            d = np.sqrt(d)
            r = pointstot[pos+2]
            #if(d==0): return False
            #a = pow(np.e,-d)/d
            #if a > maxa: maxa = a
            #if a < mina: mina = a
            #summ += a
            if(d<r+1+rac): return False
    #p = pow(summ,-m)
    #if(maxa*random.random() > p): return False
    return True

for k in range(c):
    i = randint(2*r,maxX-r)
    j = randint(2*r,maxY-r)
    points2[h] = i
    h = h+1
    points2[h] = j
    h = h+1
    points3[h2] = i
    points3[h2+1] = j
    points3[h2+2] = r
    points3[h2+3] = detcuad(i,j)

pointstot = points3

for i in range(numIt):
    print "It num", i
    print "Total: ", 2*c
    for h in range(0,2*c,2):
        x = points2[h]
        y = points2[h+1]
        if(h%600 == 0): print h
        for i in range(points2[h]-r-1,points2[h]+r+1):
            for j in range(points2[h+1]-r-1,points2[h+1]+r+1):
                 i2 = i-points2[h]
                 j2 = j-points2[h+1]
                 if(i2*i2+j2*j2 < r*r):
                     u = max(0,min(i,maxX-1))
                     v = max(0,min(j,maxY-1))
                     field[u][v] = np.uint8(0)


    r = int(r/1.6)
    orig = c
    cuant = 6
    c = int(c*cuant)

    # reset points
    points = points2
    points2 = np.zeros((c*2)).astype(np.uint32)
    points3 = np.zeros((c*cinf)).astype(np.float32)
    pos = 0
    pos2 = 0
    for k in range(0,2*orig,2):
        for l in range(cuant):
            d = random.random()*np.pi*2
            rr = 4*r
            i = points[k]+np.int32(rr*np.cos(d))#+randint(0,10)
            j = points[k+1]+np.int32(rr*np.sin(d))#+randint(0,10)
            if(avoid(i,j,r+1)):
                #pos = k*cuant + 2*l
                points2[pos] = i
                points2[pos+1] = j
                points3[pos2] = i
                points3[pos2+1] = j
                points3[pos2+2] = r
                cuad = detcuad(i,j)
                points3[pos2+3] = cuad
                pos+=2
                pos2+=cinf

    pointstot = np.hstack((pointstot,points3))

plt.imshow(field, cmap=matplotlib.cm.gray)
plt.show()

I = Image.frombuffer('L',(maxX,maxY), np.array(field).astype(np.uint8),'raw','L',0,1)

I.save('imagenpy.png')

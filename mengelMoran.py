# The idea in this example is to use the Moran equation

import Image
import numpy as np
from math import floor, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt
import random
import ImageDraw
import time


t = time.time()
maxX = 1024
maxY = 1024

I = Image.new('RGB',(maxX,maxY),(255,255,255))

#image = Image.open("x.png")
draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 50 # radius of initial bubbles
c = 2 # amount of initial bubbles
orig = c
cinf = 3 # amount of information in the array
points2 = np.zeros((c*cinf)).astype(np.uint32)

numIt = 4
h = 0
maxSize = 16

def f(a): # amount of sons depend in size (multifractality)
    if(a > 40): return 4
    if(a > 30): return 5
    if(a > 20): return 12
    return maxSize

def fdist(a): # distance depends on size
    if(a > 40): return 4*a
    if(a > 30): return 5*a
    if(a > 20): return 10*a
    return 12*a

#for k in range(c):
#    i = randint(2*r,maxX-r)
#    j = randint(2*r,maxY-r)
#    points2[h] = i
#    h = h+1
#    points2[h] = j
#    h = h+1

points2[h] = maxX/2
points2[h+1] = maxY/2
points2[h+2] = r

def f(a): # amount of sons depend in size (multifractality)
    if(a > 40): return 4
    if(a > 30): return 5
    if(a > 20): return 12
    return maxSize

def fdist(a): # distance depends on size
    if(a > 40): return 4*a
    if(a > 30): return 5*a
    if(a > 20): return 10*a
    return 12*a

for i in range(numIt):
    print "It num", i
    print "Total: ", cinf*c
    for h in range(0,len(points2),cinf):
        x = points2[h]
        y = points2[h+1]
        r = points2[h+2]
        if(h%6000 == 0): print h

        draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,0,0))

    #plt.subplot(1,numIt-1,i)
    print "Time Elapsed: ", time.time()-t
    #plt.imshow(I, cmap=matplotlib.cm.gray)


    c = len(points2)*maxSize
    # reset points
    points = points2
    points2 = np.zeros(c).astype(np.uint32)
    for k in range(0,len(points),cinf):
        r = points[k+2]
        cuant = f(r)
        for l in range(cuant):
            d = 2*np.pi*random.random()
            rr = fdist(r)
            i = points[k]+np.int32(rr*np.cos(d))+randint(-int(r),int(r))
            j = points[k+1]+np.int32(rr*np.sin(d))+randint(-int(r),int(r))
            frac = 0.8-random.random()*0.2
            r = points[k+2]*frac
            pos = k*cuant + cinf*l
            points2[pos] = i
            points2[pos+1] = j
            points2[pos+2] = r

     
plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()

I.save('imagenpy.png')

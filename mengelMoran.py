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

r = 80 # radius of initial bubbles
c = 3 # amount of initial bubbles
orig = c
cinf = 3 # amount of information in the array
points2 = np.zeros((c*cinf)).astype(np.uint32)

numIt = 7
h = 0
maxSize = 9

h2 = 0
for k in range(c):
    i = randint(2*r,maxX-r)
    j = randint(2*r,maxY-r)
    points2[h2] = i
    points2[h2+1] = j
    points2[h2+2] = r
    h2+=cinf


def f(a): # amount of sons depend in size (multifractality)
    if(a > 40): return 8
    if(a > 30): return 5
    if(a > 20): return 4
    if(a > 10): return 3
    return maxSize

def fdist(a): # distance depends on size
    if(r > 40): return 15*r
    if(r > 30): return 15*r
    if(r > 20): return 15*r
    return 8*r

def ffrac(r):
    if(r > 40): return 0.6-random.random()*0.1
    if(r > 30): return 0.7-random.random()*0.1
    if(r > 20): return 0.8-random.random()*0.1
    return 0.5-random.random()*0.1

for tt in range(numIt):
    print "It num", i
    print "Total: ", cinf*c
    for h in range(0,len(points2),cinf):
        x = points2[h]
        y = points2[h+1]
        r = points2[h+2]
        if(h%6000 == 0): print h

        draw.ellipse((x-r+randint(-int(r/4),int(r/4)), y-r+randint(-int(r/4),int(r/4)), x+r, y+r), fill=(0,0,0))

    #plt.subplot(1,numIt-1,i)
    print "Time Elapsed: ", time.time()-t


    c = len(points2)*maxSize
    # reset points
    points = points2
    points2 = np.zeros(c).astype(np.uint32)
    for k in range(0,len(points),cinf):
        r = points[k+2]
        cuant = f(r) #multifractallity
        for l in range(cuant):
            d = 2*np.pi*random.random()
            rr = fdist(r)*(0.7+random.random()*(0.3))
            i = points[k]+np.int32(rr*np.cos(d))+randint(-int(r),int(r))
            j = points[k+1]+np.int32(rr*np.sin(d))+randint(-int(r),int(r))
            frac = ffrac(r)
            r = points[k+2]*frac
            pos = k*cuant + cinf*l
            points2[pos] = i
            points2[pos+1] = j
            points2[pos+2] = r

    I.save('imagenpy'+str(tt)+'.png')

     
plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()

I.save('imagenpy.png')

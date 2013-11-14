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

from multifractal import *


t = time.time()
maxX = 1024
maxY = 1024

maxx = 6

I = Image.new('RGB',(maxX,maxY),(0,0,0))

#image = Image.open("x.png")
draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 45 # radius of initial bubbles
c = 10 # amount of initial bubbles
orig = c
cinf = 3 # amount of information in the array
points2 = np.zeros((c*cinf)).astype(np.uint32)

numIt = 5
h = 0

h2 = 0
for k in range(c):
    i = randint(2*r,maxX-r)
    j = randint(2*r,maxY-r)
    points2[h2] = i
    points2[h2+1] = j
    points2[h2+2] = r+randint(0,15)
    h2+=cinf

def drawShape(x,y,r,c):
    if(c == 0): return
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,255,255))
    return
    r2 = r
    x2 = x
    y2 = y
    #for h in range(maxx):
    r2 = int(r2/(2))
    dd = int(r*0.9)
    for i in range(4):
        x3 = x2+randint(-dd,dd)
        y3 = y2+randint(-dd,dd)
        drawShape(x3,y3,r2,c-1)

for tt in range(numIt):
    print "It num", i
    print "Total: ", cinf*c
    for h in range(0,len(points2),cinf):
        x = points2[h]
        y = points2[h+1]
        r = points2[h+2]
        if(h%6000 == 0): print h

        drawShape(x,y,r,maxx)

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

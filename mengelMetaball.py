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
maxX = 200
maxY = 200

maxx = 6

I = Image.new('RGB',(maxX,maxY),(0,0,0))

#image = Image.open("x.png")
draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 50 # radius of initial bubbles
c = 2 # amount of initial bubbles
orig = c
cinf = 3 # amount of information in the array
balls = np.zeros((c*cinf)).astype(np.float32)

numIt = 5
h = 0

field = np.zeros((maxX, maxY)).astype(np.float32)
threshold = 10

h2 = 0
for k in range(c):
    i = randint(2*r,maxX-r)
    j = randint(2*r,maxY-r)
    balls[h2] = np.float32(i)
    balls[h2+1] = np.float32(j)
    balls[h2+2] = np.float32(r+randint(0,15))
    h2+=cinf

def metaball(x,y,xo,yo,r):
    if x==xo and y == yo: return threshold
    return np.float32(r/np.sqrt(np.power((x-xo),2)+np.power((y-yo),2)))

def calc(x,y):
    global balls
    summ = np.float32(0)
    for i in range(0,len(balls),cinf):
        summ += metaball(x,y,balls[i],balls[i+1],balls[i+2])
    
    if( summ>threshold): return np.float32(255)
    return np.float32(0)

for tt in range(numIt):
    print "It num", tt
    print "Total: ", cinf*c
    for x in range(maxX):
        for y in range(maxY):
            field[x][y] = calc(x,y)
    #for h in range(0,len(balls),cinf):
        #x = balls[h]
        #y = balls[h+1]
        #r = balls[h+2]
        #if(h%6000 == 0): print h

        #drawShape(x,y,r,maxx)

    #plt.subplot(1,numIt-1,i)
    print "Time Elapsed: ", time.time()-t


    c = len(balls)*maxSize
    # reset points
    points = balls
    balls = np.zeros(c).astype(np.float32)
    for k in range(0,len(points),cinf):
        r = points[k+2]
        cuant = f(r) #multifractallity
        for l in range(cuant):
            d = 2*np.pi*random.random()
            rr = fdist(r)*(0.7+random.random()*(0.3))
            i = points[k]+np.float32(rr*np.cos(d))+randint(-int(r),int(r))
            j = points[k+1]+np.float32(rr*np.sin(d))+randint(-int(r),int(r))
            frac = ffrac(r)
            r = points[k+2]*frac
            pos = k*cuant + cinf*l
            balls[pos] = i
            balls[pos+1] = j
            balls[pos+2] = r

    I = Image.frombuffer('L',(maxX,maxY), np.array(field).astype(np.uint8),'raw','L',0,1)
    I.save('MengelMetaball'+str(tt)+'.png')

     
plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()

I.save('mengelMetaball.png')

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

from multifractalMetaball import *


t = time.time()
maxX = 512
maxY = 512

maxx = 6

I = Image.new('RGB',(maxX,maxY),(0,0,0))

#image = Image.open("x.png")
draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 240 # radius of initial bubbles
c = 1 # amount of initial bubbles
orig = c
cinf = 3 # amount of information in the array
balls = np.zeros((c*cinf)).astype(np.float32)

numIt = 5
h = 0

field = np.zeros((maxX, maxY)).astype(np.float32)
threshold = 0.01

h2 = 0
for k in range(c):
    i = randint(r,maxX-r)
    j = randint(r,maxY-r)
    balls[h2] = np.float32(i)
    balls[h2+1] = np.float32(j)
    balls[h2+2] = np.float32(r+randint(0,15))
    h2+=cinf

def ff(a):
    if(a > 40): return 6*a
    if(a > 30): return 4*a
    if(a > 20): return 2*a
    if(a > 10): return 1*a
    return 1*a

def metaball(x,y,xo,yo,r):
    if x==xo and y == yo: return [False,-1]
    return [True,np.float32(r/np.sqrt(np.power((x-xo),2)+np.power((y-yo),2)))]


Xm = 1.2
Ym = 1.5
def addMet(x,y,r):
    #for i in range(int(x-r-1),int(x+r+1)):
        #for j in range(int(y-r-1),int(y+r+1)):
    for i in range(0,maxX):
        for j in range(0,maxY):
            #if(i>=0 and i<maxX and j>=0 and j<maxY):
            d = np.sqrt(np.power((i-x),2)+np.power((j-y),2)).astype(np.float32)
            #if(d<r):
            if(field[i][j] + np.float32(r)/d <= 255):
                field[i][j] += np.float32(r)/d # circle
            else: field[i][j] = np.float32(255)
                #field[i][j] += np.float32(r/np.sqrt(Xm*np.power((i-x),2)+Ym*np.power((j-y),2))) # ellipse
                #field[i][j] += np.float32((r+r/4)/(r-np.sqrt(np.power((i-x),2)+np.power((j-y),2))) ) # donut

#def calc(x,y):
#    global balls
#    summ = np.float32(0)
#    for i in range(0,len(balls),cinf):
#        res = metaball(x,y,balls[i],balls[i+1],balls[i+2])
#        if(res[0]): summ += res[1]
#        else: return np.float32(255) 
#        if( summ>threshold): return np.float32(255)
#    return np.float32(0)

for tt in range(numIt):
    print "It num", tt
    print "Total: ", cinf*len(balls)

    print balls
    print "Time Elapsed: ", time.time()-t

    points = balls
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
            balls = np.hstack((balls,[i,j,r]))
            addMet(i,j,r)


    print field
    I = Image.frombuffer('L',(maxX,maxY), np.array((field<threshold)*100).astype(np.uint8),'raw','L',0,1)
    I.save('MengelMetaball'+str(tt)+'.png')
    I = Image.frombuffer('L',(maxX,maxY), np.array(field).astype(np.uint8),'raw','L',0,1)
    I.save('MengelMetaballStar'+str(tt)+'.png')
    print 'MengelMetaball'+str(tt)+'.png'

     
plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()

I.save('mengelMetaball.png')

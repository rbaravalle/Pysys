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

tim = time.time()

maxX = 512
maxY = 512

I = Image.new('L',(maxX,maxY),0)

draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 25 # radius of initial bubbles
c = 8 # amount of initial bubbles
orig = c

numIt = 4
h = 0
h2 = 0
cinf = 4

maxa = -10
mina = 100000
rugosity = 2

points3 = np.zeros((4,c*cinf)).astype(np.float32)
pointstot = points3

def drawShape(x,y,r,c):
    if(c == 0): return
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255))
    return
    r2 = r
    x2 = x
    y2 = y
    #for h in range(maxx):
    r2 = int(r2/(2))
    dd = int(r*0.6)
    for i in range(4):
        x3 = x2+randint(-dd,dd)
        y3 = y2+randint(-dd,dd)
        drawShape(x3,y3,r2,c-1)

def detcuad(i,j):
    if(i > maxX/2): 
        if(j > maxY/2): return 0
        else: return 1
    else:
        if(j > maxY/2): return 2
        else: return 3

# self-avoiding growth model
def avoid(i,j,rac,pointsnew):
    global maxa, mina
    summ = np.float32(0)

    points = pointstot[detcuad(i,j)]

    for pos in range(0,len(points),cinf):
       d = np.power((np.float32(i) - points[pos]),2)+np.power((np.float32(j) - points[pos+1]),2)
       d = np.sqrt(d)
       rr = points[pos+2]

       if(np.float32(d)<np.float32(rr)+rac+1): 
            return False

    if(pointsnew != []):
        pointsnew = pointsnew[detcuad(i,j)]
        for pos in range(0,len(pointsnew),cinf):
           d = np.power((np.float32(i) - pointsnew[pos]),2)+np.power((np.float32(j) - pointsnew[pos+1]),2)
           d = np.sqrt(d)
           rr = pointsnew[pos+2]

           if(np.float32(d)<np.float32(rr)+rac): 
                return False

    #p = pow(summ,-m)
    #if(maxa*random.random() > p): return False

    return True

for k in range(c):
    i = randint(2*r,maxX-r)
    j = randint(2*r,maxY-r)
    if(avoid(i,j,r,[])):
        cuad = detcuad(i,j)
        pointstot[cuad,h2] = i
        pointstot[cuad,h2+1] = j
        pointstot[cuad,h2+2] = r
        pointstot[cuad,h2+3] = cuad
        h2+=cinf

for it in range(numIt):
    print "It num", it
    print "Total: ", pointstot.shape[0]*pointstot.shape[1]

    for cc in range(4):
        for h in range(0,pointstot.shape[1],cinf):
            x = pointstot[cc,h]
            y = pointstot[cc,h+1]
            r = pointstot[cc,h+2]
            if(h%600 == 0): print h

            r2 = r
            if x!=0 and y!=0:
                drawShape(x,y,r2,rugosity)#draw.ellipse((x-r2, y-r2, x+r2, y+r2), fill=(np.uint8(0)))


    c = pointstot.shape[1]*maxSize

    if(it == numIt-1) : break
    print "Computing Next Iteration..."
    # reset points
    points3 = np.zeros((4,c)).astype(np.float32)
    pos = 0
    for cuad in range(4):
        for k in range(0,pointstot.shape[1],cinf):
            r = pointstot[cuad,k+2]
            cuant = f(r)
            for l in range(cuant):
                d = random.random()*np.pi*2
                #rr = 4*r
                rr = fdist(r)
                i = pointstot[cuad,k]+np.int32(rr*np.cos(d))#+randint(0,10)
                j = pointstot[cuad,k+1]+np.int32(rr*np.sin(d))#+randint(0,10)
                frac = ffrac(r)
                r = pointstot[cuad,k+2]*frac
                if(avoid(i,j,r,points3)==True):
                    points3[cuad,pos] = i
                    points3[cuad,pos+1] = j
                    points3[cuad,pos+2] = r
                    points3[cuad,pos+3] = cuad
                    pos+=cinf

    I.save('imagenavoid'+str(it)+'.png')

    pointstot = np.hstack((pointstot,points3))
    print "It", it, "Time elapsed: ", time.time()-tim



plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()


I.save('imagenpy.png')
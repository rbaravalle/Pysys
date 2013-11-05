import Image
import numpy as np
from math import floor, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt
import random
import ImageDraw
import time

tim = time.time()

maxX = 512
maxY = 512

I = Image.new('L',(maxX,maxY),0.0)

draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 28 # radius of initial bubbles
c = 6 # amount of initial bubbles
orig = c

numIt = 1
h = 0
h2 = 0
cinf = 4

maxa = -10
mina = 100000

points2 = np.zeros((c*2)).astype(np.uint32)
points3 = np.zeros((c*cinf)).astype(np.float32)
pointstot = points3

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
       # if(detcuad(i,j) == pointstot[pos+3]):
       d = np.power((np.float32(i) - pointstot[pos]),2)+np.power((np.float32(j) - pointstot[pos+1]),2)
       d = np.sqrt(d)
       rr = pointstot[pos+2]

       print i,j,d
       print pointstot[pos], pointstot[pos+1], rr,rac
       if(np.float32(d)<np.float32(rr)+rac): 
            return False

    #p = pow(summ,-m)
    #if(maxa*random.random() > p): return False

    return True

for k in range(c):
    i = randint(2*r,maxX-r)
    j = randint(2*r,maxY-r)
    if(avoid(i,j,r)):
        points2[h] = i
        h = h+1
        points2[h] = j
        h = h+1
        points3[h2] = i
        points3[h2+1] = j
        points3[h2+2] = r
        points3[h2+3] = detcuad(i,j)
        pointstot = points3

pointstot = points3

for i in range(numIt):
    print "It num", i
    print "Total: ", 2*c
    for h in range(0,2*c,2):
        x = points2[h]
        y = points2[h+1]
        if(h%600 == 0): print h

        r2 = r
        draw.ellipse((x-r2, y-r2, x+r2, y+r2), fill=(np.uint8(255)))


    r = int(r/1.7)
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
            rr = 8*r
            i = points[k]+np.int32(rr*np.cos(d))#+randint(0,10)
            j = points[k+1]+np.int32(rr*np.sin(d))#+randint(0,10)
            if(avoid(i,j,r)):
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

print "Time elapsed: ", time.time()-tim

plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()


I.save('imagenpy.png')

import Image
import numpy as np
from math import floor, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt
import random


maxX = 250
maxY = 250
maxZ = 250

I = Image.new('L',(maxZ,maxX*maxY),0.0)

field = np.zeros((maxZ, maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 24 # radius of initial bubbles
c = 14 # amount of initial bubbles
orig = c

global points2

points2 = np.zeros((c*3)).astype(np.int32)

numIt = 4
h = 0

    
for w in range(c):
    i = randint(r,maxX-r)
    j = randint(r,maxY-r)
    k = randint(r,maxZ-r)
    print i, j, k
    points2[h] = i
    points2[h+1] = j
    points2[h+2] = k
    h+=3

for i in range(numIt):
    print points2
    print "It num", i
    print "Total: ", 3*c
    for h in range(0,3*c,3):
        x = points2[h]
        y = points2[h+1]
        z = points2[h+2]
        if(h%600 == 0): print h, 3*c
        for i in range(points2[h]-r-1,points2[h]+r+1):
            for j in range(points2[h+1]-r-1,points2[h+1]+r+1):
                for k in range(points2[h+2]-r-1,points2[h+2]+r+1):
                     if(i >= 0 and i < maxX and j >= 0 and j < maxY and k >= 0 and k<maxZ):
                         i2 = i-points2[h]
                         j2 = j-points2[h+1]
                         k2 = k-points2[h+2]
                         if(i2*i2+j2*j2+k2*k2 < r*r):
                             field[i][j][k] = np.uint8(0)


    r = int(r/2)
    orig = c
    cuant = 16
    c = int(c*cuant)

    # reset points
    points = points2
    points2 = np.zeros((c*3)).astype(np.int32)
    print "Making next iteration..."
    for v in range(0,3*orig,3):
        for l in range(cuant):
            d = random.random()*np.pi*2
            e = random.random()*np.pi*2
            rr = 16*r
            i = points[v]+np.int32(rr*np.cos(d)*np.sin(e))
            j = points[v+1]+np.int32(rr*np.sin(d)*np.sin(e))
            k = points[v+2]+np.int32(rr*np.cos(e))
            pos = v*cuant + 3*l
            points2[pos] = i
            points2[pos+1] = j
            points2[pos+2] = k


     
print "End."
plt.imshow(field[2], cmap=matplotlib.cm.gray)
plt.show()

rowsPerSlice = maxY

for i in range(maxZ):
    I2 = Image.frombuffer('L',(maxX,maxY), np.array(field[:,:,i]).astype(np.uint8),'raw','L',0,1)
    I.paste(I2,(0,rowsPerSlice*i))

I.save('mengel3d.png')

import Image
import numpy as np
from math import floor, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt


maxX = 200
maxY = 200
maxZ = 200

I = Image.new('L',(maxZ,maxX*maxY),0.0)

field = np.zeros((maxZ, maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 15 # radius of initial bubbles
c = 30 # amount of initial bubbles
orig = c
points2 = np.zeros((c*3)).astype(np.uint32)

numIt = 2
h = 0

    
for k in range(c):
    i = randint(r,maxX-r)
    j = randint(r,maxY-r)
    k = randint(r,maxZ-r)
    points2[h] = i
    h = h+1
    points2[h] = j
    h = h+1
    points2[h] = k
    h = h+1

for i in range(numIt):
    print "It num", i
    print "Total: ", 3*c
    for h in range(0,3*c,3):
        x = points2[h]
        y = points2[h+1]
        z = points2[h+2]
        if(h%100 == 0): print h
        for i in range(points2[h]-r-1,points2[h]+r+1):
            for j in range(points2[h+1]-r-1,points2[h+1]+r+1):
                for k in range(points2[h+2]-r-1,points2[h+2]+r+1):
                     i2 = i-points2[h]
                     j2 = j-points2[h+1]
                     k2 = k-points2[h+2]
                     if(i2*i2+j2*j2+k2*k2 < r*r):
                         u = max(0,min(i,maxX-1))
                         v = max(0,min(j,maxY-1))
                         w = max(0,min(k,maxZ-1))
                         field[w][u][v] = np.uint8(0)


    r = int(r/2)
    orig = c
    cuant = 3
    c = int(c*cuant)

    # reset points
    points = points2
    points2 = np.zeros((c*3)).astype(np.uint32)
    print "LEN: ", len(points2)
    for v in range(0,3*orig,3):
        for l in range(cuant):
            d = randint(0,360) # sphere
            e = randint(0,360)
            rr = 15*r+randint(0,8)
            i = points[v]+int(rr*np.cos(d)*np.sin(e))
            j = points[v+1]+int(rr*np.sin(d)*np.sin(e))
            k = points[v+2]+int(rr*np.cos(e))
            pos = v*cuant + 3*l
            points2[pos] = i
            points2[pos+1] = j
            points2[pos+2] = k

     
plt.imshow(field[2], cmap=matplotlib.cm.gray)
plt.show()

rowsPerSlice = maxY

for i in range(maxZ):
    I2 = Image.frombuffer('L',(maxX,maxY), np.array(field[:,:,i]).astype(np.uint8),'raw','L',0,1)
    I.paste(I2,(0,rowsPerSlice*i))

I.save('mengel3d.png')

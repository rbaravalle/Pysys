import Image
import numpy as np
from math import floor, sqrt
from random import randint
import matplotlib
from matplotlib import pyplot as plt


maxX = 350
maxY = 350

I = Image.new('L',(maxX,maxY),0.0)

field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 20 # radius of initial bubbles
c = 4 # amount of initial bubbles
orig = c
points2 = np.zeros((c*2)).astype(np.uint32)

numIt = 5
h = 0

    
def sign(): 
    if randint(0,10) > 5 : return 1
    else : return -1

for k in range(c):
    i = randint(r,maxX-r)
    j = randint(r,maxY-r)
    points2[h] = i
    h = h+1
    points2[h] = j
    h = h+1

for i in range(numIt):
    print "It num", i
    print "Total: ", 2*c
    for h in range(0,2*c,2):
        x = points2[h]
        y = points2[h+1]
        if(h%6000 == 0): print h
        for i in range(points2[h]-r-1,points2[h]+r+1):
            for j in range(points2[h+1]-r-1,points2[h+1]+r+1):
                 i2 = i-points2[h]
                 j2 = j-points2[h+1]
                 if(i2*i2+j2*j2 < r*r):
                     u = max(0,min(i,maxX-1))
                     v = max(0,min(j,maxY-1))
                     field[u][v] = np.uint8(0)


    r = int(r/2)
    orig = c
    cuant = 10
    c = int(c*cuant)

    # reset points
    points = points2
    points2 = np.zeros((c*2)).astype(np.uint32)
    for k in range(0,2*orig,2):
        for l in range(cuant):
            d = randint(0,360)
            rr = 15*r+randint(0,8)
            i = points[k]+int(rr*np.cos(d))#+randint(0,10)
            j = points[k+1]+int(rr*np.sin(d))#+randint(0,10)
            pos = k*cuant + 2*l
            points2[pos] = i
            points2[pos+1] = j

     

plt.imshow(field, cmap=matplotlib.cm.gray)
plt.show()

I = Image.frombuffer('L',(maxX,maxY), np.array(field).astype(np.uint8),'raw','L',0,1)

I.save('imagenpy.png')

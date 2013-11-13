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

I = Image.new('L',(maxX,maxY),255)

#image = Image.open("x.png")
draw = ImageDraw.Draw(I)

#field = np.zeros((maxX, maxY)).astype(np.uint8) + np.uint8(255)

r = 50 # radius of initial bubbles
c = 2 # amount of initial bubbles
orig = c
points2 = np.zeros((c*2)).astype(np.uint32)

numIt = 5
h = 0

#for k in range(c):
#    i = randint(2*r,maxX-r)
#    j = randint(2*r,maxY-r)
#    points2[h] = i
#    h = h+1
#    points2[h] = j
#    h = h+1

points2[h] = maxX/2
points2[h+1] = maxY/2

for i in range(numIt):
    print "It num", i
    print "Total: ", 2*c
    for h in range(0,2*c,2):
        x = points2[h]
        y = points2[h+1]
        if(h%6000 == 0): print h

        draw.ellipse((x-r+randint(-r/2,r/2), y-r+randint(-r/2,r/2), x+r+16+randint(-r/2,r/2), y+r+randint(-r/2,r/2)), fill=(np.uint8(0)))


    r = int(r/1.8)
    orig = c
    cuant = 6
    c = int(c*cuant)

    # reset points
    points = points2
    points2 = np.zeros((c*2)).astype(np.uint32)
    for k in range(0,2*orig,2):
        for l in range(cuant):
            d = (np.float32(l)/cuant)*np.pi*2+random.random()
            rr = 10*r
            i = points[k]+np.int32(rr*np.cos(d))+randint(0,10)
            j = points[k+1]+np.int32(rr*np.sin(d))+randint(0,10)
            pos = k*cuant + 2*l
            points2[pos] = i
            points2[pos+1] = j

     
print "Time Elapsed: ", time.time()-t
plt.imshow(I, cmap=matplotlib.cm.gray)
plt.show()

I.save('imagenpy.png')

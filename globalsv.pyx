import numpy as np
cimport numpy as np
from random import randint, random

# global vars
maxcoord = 180
maxcoordZ = 180
maxcoord2 = maxcoord*maxcoord
maxcoord3 = maxcoord2*maxcoordZ
m1 = 1.0/maxcoord
t = 0
cantPart = 40000
CT = 0
MCA = 2000
randomness = 0.15

#2D-world limits
x0 = -3
y0 = -3
x1 = 3
y1 = 3
z0 = -3
z1 = 3
diffX = float(x1-x0)
diffY = float(y1-y0)
diffZ = float(z1-z0)


TIEMPO = 120000
sep = 1 # separation among particles
diffBubbles = 18
amountSons = 0


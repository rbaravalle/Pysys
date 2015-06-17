# global vars
maxcoord = 350
maxcoordZ = 350
maxcoord2 = maxcoord*maxcoord
maxcoord3 = maxcoord2*maxcoordZ
m1 = 1.0/maxcoord
u6 = 1.0/6.0
t = 0
cantPart = 7000
MCA = 1000
randomness = 0.03
randomnessZ = 0.1
fx = 1
fy = 1

#2D-world limits

x0 = -3.0*fx
x1 = 3.0*fx
y0 = -3.0*fy
y1 = 3.0*fy
z0 = -1
z1 = 1
diffX = x1-x0
diffY = y1-y0
diffZ = z1-z0
dXm1 = diffX*m1
dYm1 = diffY*m1
dZm1 = diffZ*m1

TIEMPO = 120000
sep = 1 # separation among particles
diffBubbles = 0
amountSons = 0
thresh = 3.4


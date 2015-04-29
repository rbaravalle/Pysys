# global vars
maxcoord = 256
maxcoordZ = 256
maxcoord2 = maxcoord*maxcoord
maxcoord3 = maxcoord2*maxcoordZ
m1 = 1.0/maxcoord
u6 = 1.0/6.0
t = 0
cantPart = 100000
MCA = 1000
randomness = 0.1

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
dm1 = diffX*m1
dZm1 = diffZ*m1

TIEMPO = 120000
sep = 1 # separation among particles
diffBubbles = 18
amountSons = 0
thresh = 1.4


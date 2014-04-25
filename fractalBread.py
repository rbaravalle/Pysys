import numpy as np
import random
import Image
import ImageDraw

from lparams import *
import baking1D as bak

N = 500
maxR = N/20


bubbles = np.zeros((N+1,N+1)).astype(np.int32) # posicion y tamanio burbujas
delta = N/16 # radius to check for intersections with other bubbles
I = Image.new('L',(N,N),(255))
draw = ImageDraw.Draw(I)
arr = bak.calc()
print arr[32/2]


# calculates new radius based on a poisson distribution (?)
def poisson(x,y):
    global bubbles, delta,draw
    #x1 = min(max(self.x-delta,0),N)
    #y1 = min(max(self.y-delta,0),N)
    x1 = x
    y1 = y

    suma = 0.0
    x0 = max(x1-delta,0)
    y0 = max(y1-delta,0)
    x2 = min(x1+delta,N)
    y2 = min(y1+delta,N)
    #print x0,y0,x2,y2
    #print abs(x0-x2),abs(y0-y2)
    cant = 0
    for i in range(x0,x2):
        for j in range(y0,y2):
            d = np.sqrt((x1-i)*(x1-i) + (y1-j)*(y1-j)).astype(np.float32) # distance
            #if(d==0): suma+=bubbles[i,j]
            #else:
            suma += bubbles[i,j]*(np.sqrt(2)*delta+1-d)/(np.sqrt(2)*delta)
            cant += bubbles[i,j]>0

    #1/ sum_D (cant*radios*D^2) 

    #x1 = min(max(x-delta,0),N)
    #y1 = min(max(y-delta,0),N)
    baking = 1#arr[x1,y1]*4
    #print baking, ": baking"

    if(suma > 0):
        m = 1/np.sqrt(cant*suma*delta*delta)
        #if(delta*m < 1): return np.random.randint(0,2)
        #if(cant>10): return 0
        Q = 1/(np.sqrt(np.pi))
        return m#np.random.randint(1,Q*m+2)/baking
    else: # free
        return delta#np.random.randint(4,delta)/baking


def drawShape(x,y,r,c):
    global draw
    if(c == 0): return
    r = int(r)
    rr = 0#int(r+10)
    if(r<=2):
        draw.ellipse((x-r, y-r, x+r, y+r), fill=rr)
    else:
        for i in range(x-r,x+r):
            for j in range(y-r,y+r):
                if((x-i)*(x-i)+(y-j)*(y-j)<r*r):# < r*r+np.random.randint(-r,r)):
                    draw.point((i,j),fill=rr)
    #draw.ellipse((x-r, y-r, x+r, y+r), fill=rr)
    return
    r2 = r
    x2 = x
    y2 = y
    r2 = int(r2/(1))
    dd = int(r*1.0)
    for i in range(2):
        x3 = x2+np.random.randint(-dd,dd+1)
        y3 = y2+np.random.randint(-dd,dd+1)
        drawShape(x3,y3,r2,c-1)

def shape(x,y):
    global bubbles
    P = 0.005
    Q = np.sqrt(0.1/(20*np.pi))
    rho = P*random.random()#np.random.randint(1,20)#P1*random.random()#
    print "Q:", Q
    print "Fractal Dimention: ", 2-2*np.pi*Q*Q
    r = np.min((Q/np.sqrt(rho),maxR))#poisson(x,y)
    #print r
    drawShape(x,y,r,2)
    bubbles[x,y] = r

def main():
    global I
    maxx = 10000
    
    a = 0.000004#0.001 # amount of rotation

    arr = np.zeros((N,N,2)) # vector field
    for i in range(0,N):
        for j in range(0,N):
            arr[i,j][0] = np.sign(i-N/2)*((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/400
            arr[i,j][1] = np.sign(j-N/2)*((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/400

    for i in range (0,maxx):
        shape(np.random.randint(0,N),np.random.randint(0,N))


        if(i%500 == 0):

            I2 = Image.new("L",(N,N))
            for x in range(0,N):    
                for y in range(0,N):
                    #dist = 4*(x-N/2)*(x-N/2)+(y-N/2)*(y-N/2)
                    #angle = -arr[np.floor(x/32),np.floor(y/32)]*2#*dist#*np.exp(-(x*x+y*y)/(b*b))
                    #print angle, dist
                    u = x+arr[x,y,0]#np.cos(angle)*x - np.sin(angle)*y
                    v = y+arr[x,y,1]
                    #print x,y,u,v
                    #u2 = min(N-1,max(u,0))
                    #v2 = min(N-1,max(v,0))
                    if(u < 0 or u >= N or v < 0 or v >= N):
                        value = 0#I.getpixel((np.floor(u)%N,np.floor(v)%N))
                    else:
                        value = I.getpixel((np.floor(u),np.floor(v)))
                    I2.putpixel((x,y),value)

            I2.save('fractal'+str(i)+'Bread.png')

main()

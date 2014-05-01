import numpy as np
import random
import Image
import ImageDraw

from lparams import *
import baking1D as bak

N = 128
maxR = N/16

field = np.zeros((N,N,N)).astype(np.uint8) + np.uint8(255)


#bubbles = np.zeros((N+1,N+1,N+1)).astype(np.int32) # posicion y tamanio burbujas
delta = N/16 # radius to check for intersections with other bubbles
I = Image.new('L',(N,N),(255))
draw = ImageDraw.Draw(I)
arr = bak.calc()
gx, gy = np.gradient(arr)

def drawShape(x,y,z,r,c):
    global field
    if(c == 0): return
    r = int(r)
    rr = 0#int(r+10)
    if(r<=200):
        draw.ellipse((x-r, y-r, x+r, y+r), fill=rr)
    #else:
    for i in range(x-r,x+r):
        for j in range(y-r,y+r):
            for k in range(z-r,z+r):
                if((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k) < r*r+np.random.randint(-r,r)):
                #draw.point((i,j),fill=rr)
                    if(i < N and i >= 0 and j < N and j >= 0 and k < N and k >= 0 ):
                        field[i,j,k] = rr
    #draw.ellipse((x-r, y-r, x+r, y+r), fill=rr)
    return
    r2 = r
    x2 = x
    y2 = y
    z2 = z
    r2 = int(r2/(1))
    dd = int(r*1.0)
    for i in range(2):
        x3 = x2+np.random.randint(-dd,dd+1)
        y3 = y2+np.random.randint(-dd,dd+1)
        z3 = z2+np.random.randint(-dd,dd+1)
        drawShape(x3,y3,z3,r2,c-1)

def shape(x,y,z,i):
    #global bubbles
    #Q = 0.1/np.sqrt(np.pi)#np.sqrt(0.015/(20*np.pi))
    #rho = 0.1+random.random()#np.random.randint(1,20)#P1*random.random()#
    #print "Q:", Q
    #print "Fractal Dimention: ", 2-2*np.pi*Q*Q
    #r = np.min((Q/np.sqrt(rho),maxR))#poisson(x,y)
    #print r
    #r = np.random.randint(1,maxR)
    r = i
    drawShape(x,y,z,r,2)
    #bubbles[x,y,z] = r

def f(c):
    return max(10-c,2)

def main():
    global gx,gy
    maxx = 100000

    I = Image.new('L',(N,N*N),0.0)

    gx2 = np.zeros((N,N)) # vector field
    gy2 = np.zeros((N,N)) # vector field


    shx = gx.shape[0]-1
    shy = gy.shape[1]-1

    # the same foreach z
    for i in range(0,N):
        for j in range(0,N):
            dist = ((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/1600
            u = i*(shx/(np.float32(N)-1))
            v = j*(shy/(np.float32(N)-1))
            gx2[i,j] = np.sign(gx[u,v])*dist**2
            gy2[i,j] = np.sign(gy[u,v])*dist**2



    shx = gx.shape[0]
    shy = gy.shape[1]

    for i in range(1,50,1):
        cubr = f(i)
        maxrank = N*N*N/cubr/(i*i*i*i)#(np.power(np.e,i))
        print i, maxrank
        for j in range(0,int(maxrank)):
            shape(np.random.randint(0,N),np.random.randint(0,N),np.random.randint(0,N),i)

        for x in range(0,N):    
            for y in range(0,N):
                for z in range(0,N):
                    val = 16.0
                    u = x+gx2[x,y]
                    v = y+gy2[x,y]
                    w = z
                    if(u < 0 or u >= N or v < 0 or v >= N):
                        value = 0
                    else:
                        value = field[u,v,w]
                    field[x,y,z] = value

        if(i%1 == 0):
            print i
            for w in range(N):
                I2 = Image.frombuffer('L',(N,N), np.array(field[:,:,w]).astype(np.uint8),'raw','L',0,1)
                I.paste(I2,(0,N*w))

            I.save('fractalBread3D'+str(i)+'.png')

   
    #for i in range (0,maxx):
    #    shape(np.random.randint(0,N),np.random.randint(0,N),np.random.randint(0,N))

    #    if(i%50 == 0):
    #        print i
    #        for w in range(N):
    #            I2 = Image.frombuffer('L',(N,N), np.array(field[:,:,w]).astype(np.uint8),'raw','L',0,1)
    #            I.paste(I2,(0,N*w))

    #        I.save('fractalBread3D'+str(i)+'.png')


main()

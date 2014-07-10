import numpy as np
import random
import Image
import ImageDraw

from lparams import *
import baking1D as bak
from mvc import mvc # mean value coordinates

N = 400
maxR = N/16

field = np.zeros((N,N,N)).astype(np.uint8) + np.uint8(255)
field3 = np.zeros((N,N,N)).astype(np.uint8) + np.uint8(255)

arr = bak.calc()
gx, gy = np.gradient(arr)

def drawShape(x,y,z,r,c):
    global field
    if(c == 0): return
    r = int(r)
    rr = 0
    for i in range(x-r,x+r):
        for j in range(y-r,y+r):
            for k in range(z-r,z+r):
                if((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k) < r*r):
                    if(i < N and i >= 0 and j < N and j >= 0 and k < N and k >= 0 ):
                        field[i,j,k] = rr
    return
    #r2 = r
    #x2 = x
    #y2 = y
    #z2 = z
    #r2 = int(r2/(1))
    #dd = int(r*1.0)
    #for i in range(2):
    #    x3 = x2+np.random.randint(-dd,dd+1)
    #    y3 = y2+np.random.randint(-dd,dd+1)
    #    z3 = z2+np.random.randint(-dd,dd+1)
    #    drawShape(x3,y3,z3,r2,c-1)

def shape(x,y,z,i):
    drawShape(x,y,z,i,2)


def f(c):
    if(c==1): return 5
    if(c<=5): return 15
    if(c>5): return 7
    #return max(40-c,20)

def paint(i,N,field3):
    p = 6
    for x in range(int(i[0])-p,int(i[0])+p):
        for y in range(int(i[1])-p,int(i[1])+p):
            for z in range(int(i[2])-p,int(i[2])+p):
                field3[np.clip(x,0,N-1),np.clip(y,0,N-1),np.clip(z,0,N-1)] = 255
    return field3

def main():
    global gx,gy, v
    maxx = 100000

    I = Image.new('L',(N/2,N/2*N/2),0.0)
    I3 = Image.new('L',(N,N*N),0.0)

    gx2 = np.zeros((N,N)) # vector field
    gy2 = np.zeros((N,N)) # vector field

    shx = gx.shape[0]-1
    shy = gy.shape[1]-1

    # the same foreach z
    for i in range(0,N):
        for j in range(0,N):
            dist = np.sqrt(((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2)))
            #if(dist < 50): dist /= 4
            #print dist
            u = i*(shx/(np.float32(N)-1))
            v = j*(shy/(np.float32(N)-1))
            gx2[i,j] = gx[u,v]*dist
            gy2[i,j] = gy[u,v]*dist

    p = 64
    # Arbitrary shape, mean value coordinates
    cageOrig = np.array([[p,N-1-p],[N/2,N-1-p],[N/2+40,N-1-p],[N-1-p,N-1-p],[N-1-p,N/2+40],[N-1-p,N/2],[N-1-p,N/2-40],[N-1-p,p],[N/2+40,p],[N/2,p],[N/2-40,p],[p,p]]).astype(np.float32)

    cageReal = np.array(cageOrig)
    cageNew = np.array(cageOrig)

    # control points displacements
    # RIGHT, BOTTOM,LEFT, TOP < |> <_> <| > <->
    # X: BOTTOM - TOP
    # Y : LEFT - RIGHT
    #trs=[[0,-20],[0,0],[0,0],[0,0],[30,0],[45,0],[30,40],[-5,0],[30,10],[25,10],[45,20],[12,15]]
    trs=[[0,-10],[0,0],[0,0],[0,0],[20,0],[35,0],[20,30],[-5,0],[20,10],[15,10],[25,10],[12,5]]

    for i in range(len(cageOrig)):
        cageReal[i] = cageOrig[i]+trs[i]
        cageNew[i] = cageOrig[i]-trs[i]

    print cageOrig
    print cageReal
    print cageNew

    for i in range(1,30,1):
        cubr = f(i)
        maxrank = 2*N* (N*N/cubr/(4*i*i*i*i)) # N* 2D Case
        if(maxrank >0):
            print i, maxrank
            for j in range(0,int(maxrank)):
                shape(np.random.randint(0,N),np.random.randint(0,N),np.random.randint(0,N),i)
        else: break

    for w in range(N):
        III = Image.frombuffer('L',(N/2,N/2), np.array(field[N/4:N*3/4,N/4:N*3/4,w]).astype(np.uint8),'raw','L',0,1)
        I.paste(III,(0,N/2*w))

    I.save('warp/fractalBread3D'+str(i)+'.png')

    field2 = np.zeros((N,N,N)).astype(np.uint8) + np.uint8(255)
    field3 = np.zeros((N,N,N)).astype(np.uint8) + np.uint8(255)
    k = float(30.0)

    print "Baking..."
    for x in range(0,N):    
        for y in range(0,N):
            for z in range(0,N):
                u = np.round(x+k*gx2[x,y])
                v = np.round(y+k*gy2[x,y])
                w = z
                if(u < 0 or u >= N or v < 0 or v >= N):
                    value = 0
                else:
                    value = field[u,v,w]
                field2[x,y,z] = value

    print "Warping..."
    for x in range(0,N):
        for y in range(0,N):
            barycoords = mvc([x,y],cageOrig)
            arr = np.zeros((2)).astype(np.float32)
            for h in range(len(barycoords)):
                arr += barycoords[h]*cageNew[h]
            arr = np.array(arr).astype(int)
            arr = np.clip(arr,0,N-1)
            for z in range(0,N):
                field3[x,y,z] = field2[int(arr[0]),int(arr[1]),z]

    #for h in range(len(cageReal)):
        #a = cageReal[h]
        #field3 = paint([a[0],a[1],10],N,field3)

    print "Saving Image..."
    for w in range(N):
        I2 = Image.frombuffer('L',(N/2,N/2), np.array(field2[N/4:N*3/4,N/4:N*3/4,w]).astype(np.uint8),'raw','L',0,1)
        I.paste(I2,(0,N/2*w))

        #II = Image.frombuffer('L',(N/2,N/2), np.array(field3[N/4:N*3/4,N/4:N*3/4,w]).astype(np.uint8),'raw','L',0,1)
        #I3.paste(II,(0,N/2*w))
        II = Image.frombuffer('L',(N,N), np.array(field3[:,:,w]).astype(np.uint8),'raw','L',0,1)
        I3.paste(II,(0,N*w))

    I3.save('warp/field'+str(i)+'.png')
    print "Image "+ str(i) +" saved"


main()

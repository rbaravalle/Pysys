import numpy as np
import random
import Image
import ImageDraw

from lparams import *
import baking1D as bak
from mvc import mvc
from sat import *

N = 1200
maxR = N/20

bubbles = np.zeros((N+1,N+1)).astype(np.int32) # posicion y tamanio burbujas
delta = N/16 # radius to check for intersections with other bubbles
I = Image.new('L',(N,N),(0))
draw = ImageDraw.Draw(I)
arr = bak.calc()
gx, gy = np.gradient(arr)



def drawShape(x,y,r,c):
    global draw
    if(c == 0): return
    r = int(r)
    rr = 255#int(r+10)
    if(r<=2):
        draw.ellipse((x-r, y-r, x+r, y+r), fill=rr)
    else:
        for i in range(x-r,x+r):
            for j in range(y-r,y+r):
                if((x-i)*(x-i)+(y-j)*(y-j)<r*r):
                    draw.point((i,j),fill=rr)
    return

def f(c):
    return 20

def shape2(x,y,r):
    drawShape(x,y,r,2)

def paint(i,N,I2):
    p = 6
    for x in range(int(i[0])-p,int(i[0])+p):
        for y in range(int(i[1])-p,int(i[1])+p):
            I2.putpixel((np.clip(x,0,N-1),np.clip(y,0,N-1)),255)

def main():
    global I,gx,gy

    bufferr = np.zeros((N,N)).astype(np.uint8)

    print "Proving..."
    for i in range(3,N/20,1):
        cubr = f(i)
        maxrank = 0.4*N*N/cubr/(np.power(i,2.6))
        if(maxrank >0):
            print i, maxrank
            for j in range(0,int(maxrank)):
                shape2(np.random.randint(0,N),np.random.randint(0,N),i)

    print "fractal"+str(i+1)+"Bread.png"
    I.save('warp/fractal'+str(i+1)+'Bread.png')

    data = np.array(I.getdata()).reshape((N,N))


    p = 100
    # Arbitrary shape, mean value coordinates
    cageOrig = np.array([[p,N-1-p],[N/2,N-1-p],[N/2+50,N-1-p],[N-1-p,N-1-p],[N-1-p,N/2+50],[N-1-p,N/2],[N-1-p,N/2-50],[N-1-p,p],[N/2+50,p],[N/2,p],[N/2-50,p],[p,p]]).astype(np.float32)

    cageReal = np.array(cageOrig)
    cageNew = np.array(cageOrig)

    # control points displacements
    trs=[[0,0],[15,20],[15,20],[30,0],[25,0],[20,0],[0,45],[0,40],[0,35],[0,20],[0,35],[0,40]]

    for i in range(len(cageOrig)):
        cageReal[i] = cageOrig[i]+trs[i]
        cageNew[i] = cageOrig[i]-trs[i]

    #print cageOrig
    #print cageReal
    #print cageNew

    #print satx
    #print saty
    sx = gx.shape[0]
    sy = gy.shape[0]
    gx2 = np.zeros((N,N)).astype(np.float32) # vector field
    gy2 = np.zeros((N,N)).astype(np.float32) # vector field


    print gx.shape
    print gy.shape

    #gx = np.vstack((gx[:sx/2],gx[sx/2+1:]))
    #gy = np.hstack((gy[:,:sy/2],gy[:,sy/2+1:]))

    print gx.shape
    print gy.shape

    shx = gx.shape[0]-1
    shy = gx.shape[1]-1

    for i in range(0,N):
        for j in range(0,N):
            dist = np.sqrt(((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2)))
            u = np.round(i*(shx/(np.float32(N)-1))).astype(np.int32)
            v = np.round(j*(shy/(np.float32(N)-1))).astype(np.int32)
            gx2[i,j] = gx[u,v]*dist
            gy2[i,j] = gy[u,v]*dist

    shx = gx.shape[0]
    shy = gy.shape[1]

    print "Baking.."
    d = 2
    k = 20

    for x in range(0,N):    
        for y in range(0,N):
            u = np.round(x+k*gx2[x,y])
            v = np.round(y+k*gy2[x,y])
            if(u < 0 or u >= N or v < 0 or v >= N):
                value = 0
            else:
                bufferr[x,y] = data[u,v]

    I2 = Image.new("L",(N,N))
    for x in range(N):
        for y in range(N):
            I2.putpixel((x,y),bufferr[x,y])

    print "fractal"+str(i)+"Bread.png"
    I2.save('warp/fractal'+str(i)+'Bread.png')


    #exit()
    print "Warping.."
    bufferr2 = np.zeros((N,N))
    for x in range(0,N):
        for y in range(0,N):
            barycoords = mvc([x,y],cageOrig)
            value = 0
            v = np.zeros((2)).astype(np.float32)
            #v = barycoords*cageNew
            for h in range(len(barycoords)):
                v += barycoords[h]*cageNew[h]
            v = np.array(v).astype(int)
            v = np.clip(v,0,N-1)
            bufferr2[x,y] = bufferr[int(v[0]),int(v[1])]

    I3 = Image.new("L",(N,N))
    for x in range(N):
        for y in range(N):
            I3.putpixel((x,y),bufferr2[x,y])
            

    #map(lambda i: paint(i,N,I3), cageReal)

    print "Image saving...", i
    I3.save('warp/fractalBreadWarp'+str(i)+'.png')



main()

import numpy as np
import random
import Image
import ImageDraw

from lparams import *
import baking1D as bak

N = 400
maxR = N/20


bubbles = np.zeros((N+1,N+1)).astype(np.int32) # posicion y tamanio burbujas
delta = N/16 # radius to check for intersections with other bubbles
I = Image.new('L',(N,N),(255))
draw = ImageDraw.Draw(I)
arr = bak.calc()
gx, gy = np.gradient(arr)
import pylab
fig = pylab.figure()
#pylab.imshow(arr)
#pylab.colorbar()
#pylab.show()



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
                if((x-i)*(x-i)+(y-j)*(y-j)<r*r):
                    draw.point((i,j),fill=rr)
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


    
def bilinear_interpolate2(ar,kx,ky):
    nx = N / (ar.shape[0] - 1.0)
    ny = N / (ar.shape[1] - 1.0)

    i = int(kx / nx); j = int(ky / ny)
    dx0 = kx - i * nx; dx1 = nx - dx0
    dy0 = ky - j * ny; dy1 = ny - dy0
    z = ar[j][i] * dx1 * dy1
    z += ar[j][i + 1] * dx0 * dy1
    z += ar[j + 1][i] * dx1 * dy0
    z += ar[j + 1][i + 1] * dx0 * dy0
    z /= nx * ny            
    return z



def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[0]-1);
    x1 = np.clip(x1, 0, im.shape[0]-1);
    y0 = np.clip(y0, 0, im.shape[1]-1);
    y1 = np.clip(y1, 0, im.shape[1]-1);

    Ia = im[ x0, y0 ]
    Ib = im[ x0, y1 ]
    Ic = im[ x1, y0 ]
    Id = im[ x1, y1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# creates coarser vector field using bilinear interpolation
def interpolate(gx,gy,N):
    gx2 = np.zeros((N,N))
    gy2 = np.zeros((N,N))


    for i in range(0,N):
        for j in range(0,N):
            gx2[i,j] = bilinear_interpolate(gx,(gx.shape[0]-1)*i/(N-1),(gx.shape[1]-1)*j/(N-1))
            gy2[i,j] = bilinear_interpolate(gy,(gx.shape[0]-1)*i/(N-1),(gx.shape[1]-1)*j/(N-1))
    return gx2,gy2

def f(c):
    return max(40-c,20)

def shape2(x,y,r):
    drawShape(x,y,r,2)

def main():
    global I,gx,gy

    gx2 = np.zeros((N,N)) # vector field
    gy2 = np.zeros((N,N)) # vector field


    shx = gx.shape[0]-1
    shy = gy.shape[1]-1

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
        maxrank = N*N/cubr/(i*i*i)
        if(maxrank <=0): exit()
        print i, maxrank
        for j in range(0,int(maxrank)):
            shape2(np.random.randint(0,N),np.random.randint(0,N),i)
        I2 = Image.new("L",(N,N))
        data = np.array(I.getdata()).reshape((N,N))
        for x in range(0,N):    
            for y in range(0,N):
                val = 16.0
                u = x+gx2[x,y]
                v = y+gy2[x,y]
                if(u < 0 or u >= N or v < 0 or v >= N):
                    value = 0
                else:
                    value = I.getpixel((u,v))
                I2.putpixel((x,y),value)

        print "Image saving...", i
        I2.save('fractal'+str(i)+'Bread.png')

    

main()

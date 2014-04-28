import numpy as np
import random
import Image
import ImageDraw

from lparams import *
import baking1D as bak

N = N*2
maxR = N/20


bubbles = np.zeros((N+1,N+1)).astype(np.int32) # posicion y tamanio burbujas
delta = N/16 # radius to check for intersections with other bubbles
I = Image.new('L',(N,N),(255))
draw = ImageDraw.Draw(I)
arr = bak.calc()
gx, gy = np.gradient(arr)
import pylab
fig = pylab.figure()
pylab.imshow(arr)
pylab.colorbar()
#pylab.show()


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
    P = 0.010
    Q = np.sqrt(0.1/(20*np.pi))
    rho = P*random.random()#np.random.randint(1,20)#P1*random.random()#
    #print "Q:", Q
    #print "Fractal Dimention: ", 2-2*np.pi*Q*Q
    r = np.min((Q/np.sqrt(rho),maxR))#poisson(x,y)
    #print r
    drawShape(x,y,r,2)
    bubbles[x,y] = r

#import scipy

#def bilinear_interpolate(im,x,y):
#    return scipy.interpolate.interp2d(np.array(x),np.array(y),im)

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
    #c = int(z * 255)
    #return c



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

    #print gx[10,10], gx[10,11], gx[11,10], gx[11,11]
    #print "INTERP: ", bilinear_interpolate(gx,11,10)
    #exit()

    for i in range(0,N):
        for j in range(0,N):
            gx2[i,j] = bilinear_interpolate(gx,(gx.shape[0]-1)*i/(N-1),(gx.shape[1]-1)*j/(N-1))
            gy2[i,j] = bilinear_interpolate(gy,(gx.shape[0]-1)*i/(N-1),(gx.shape[1]-1)*j/(N-1))
    return gx2,gy2

def f(c):
    return max(40-c,20)

def shape2(x,y,r):
    global bubbles
    #P = 0.005
    #Q = np.sqrt(0.1/(20*np.pi))
    #rho = P*random.random()#np.random.randint(1,20)#P1*random.random()#
    #print "Q:", Q
    #print "Fractal Dimention: ", 2-2*np.pi*Q*Q
    #r = np.min((Q/np.sqrt(rho),maxR))#poisson(x,y)
    #print r
    drawShape(x,y,r,2)
    bubbles[x,y] = r

def main():
    global I,gx,gy
    #maxx = 10000
    
    #a = 0.000004#0.001 # amount of rotation

    #arr = np.zeros((N,N,2)) # vector field
    #for i in range(0,N):
    #    for j in range(0,N):
    #        arr[i,j][0] = -np.sign(i-N/2)*((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/800
    #        arr[i,j][1] = -np.sign(j-N/2)*((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/800


    #print "ANTES: GX: ", gx[10,10], "GY: ", gy[10,10]
    #print gx, gy
    #u = gx
    #v = gy
    
    #u = gx#np.sign(1+gx-N/2)*gx
    #v = gy#-np.sign(1+gy-N/2)*gy
    #import pylab
    a = 50
    #Q = pylab.quiver( 10*gx[a:N-a,a:N-a],10*gy[a:N-a,a:N-a])
    #pylab.show()
    #exit()

    print 10*gy[32/2]
    #exit()

    #print "Obtaining Vector Field...."
    #gx, gy = interpolate(gx,gy,N)
    #print "Vector Field already computed."


    for i in range(1,50,1):
        cubr = f(i)
        maxrank = N*N/cubr/(i*i*i)#(np.power(np.e,i))
        if(maxrank <=0): exit()
        print i, maxrank
        for j in range(0,int(maxrank)):
            shape2(np.random.randint(0,N),np.random.randint(0,N),i)
        #if(i%5 == 0):
            #I.save('fractal'+str(i)+'Bread2.png')
        I2 = Image.new("L",(N,N))
        data = np.array(I.getdata()).reshape((N,N))
        for x in range(0,N):    
            for y in range(0,N):
                dist = np.sqrt((x-N/2)*(x-N/2)+(y-N/2)*(y-N/2))#20*/N
                #if(arr[x,y] > 204):
                val = 10
                #else:
                val = 100
                u = x+val*gx[x,y]#arr[x,y,0]#np.cos(angle)*x - np.sin(angle)*y
                v = y+val*gy[x,y]#arr[x,y,1]
                if(u < 0 or u >= N or v < 0 or v >= N):
                    value = 0#I.getpixel((np.floor(u)%N,np.floor(v)%N))
                else:
                    value = I.getpixel((u,v))##bilinear_interpolate(data,u,v) 
                if(np.sqrt((x-N/2)*(x-N/2)+(y-N/2)*(y-N/2)) > N-40): value = 0
                I2.putpixel((x,y),value)

        print "Image saving...", i
        I2.save('fractal'+str(i)+'Bread.png')

    return


    #for i in range (0,maxx):
    #    shape(np.random.randint(0,N),np.random.randint(0,N))


    if(i%500 == 0):

        I2 = Image.new("L",(N,N))
        data = np.array(I.getdata()).reshape((N,N))
        for x in range(0,N):    
            for y in range(0,N):
                dist = 5*np.sqrt((x-N/2)*(x-N/2)+(y-N/2)*(y-N/2))/N
                u = x+dist*gx[x,y]#arr[x,y,0]#np.cos(angle)*x - np.sin(angle)*y
                v = y+dist*gy[x,y]#arr[x,y,1]
                if(u < 0 or u >= N or v < 0 or v >= N):
                    value = 0#I.getpixel((np.floor(u)%N,np.floor(v)%N))
                else:
                    value = bilinear_interpolate(data,u,v) #I.getpixel((resample(u),resample(v)))
                if(np.sqrt((x-N/2)*(x-N/2)+(y-N/2)*(y-N/2)) > 10): value = 0
                I2.putpixel((x,y),value)

        print "Image saving...", i
        I2.save('fractal'+str(i)+'Bread.png')

main()

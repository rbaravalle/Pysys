#!/usr/bin/env python
"""
Author: Timothy A.V. Teatro <http://www.timteatro.net>
Date  : Oct 25, 2010
Lisence: Creative Commons BY-SA
(http://creativecommons.org/licenses/by-sa/2.0/)

Description:
    A program which uses an explicit finite difference
    scheme to solve the diffusion equation with fixed
    boundary values and a given initial value for the
    density u(x,y,t). This version uses a numpy
    expression which is evaluated in C, so the
    computation time is greatly reduced over plain
    Python code.

    This version also uses matplotlib to create an
    animation of the time evolution of the density.
"""
import scipy as sp
import matplotlib
matplotlib.use('GTKAgg') # Change this as desired.
import gobject
from pylab import *
import Image
import lsystem
# Declare some variables:

dx=0.004        # Interval size in x-direction.
dy=0.004        # Interval size in y-direction.
#a=4          # Diffusion constant.
#tf = 0.0013


def toKelvin(c):
    return c + 273.15


tOven = toKelvin(200) # temperature Oven
tf = toKelvin(100)
timesteps=3000  # Number of time-steps to evolve system.

nx = int(1/dx)
ny = int(1/dy)

dx2=dx**2 # To save CPU cycles, we'll compute Delta x^2
dy2=dy**2 # and Delta y^2 only once and store them.
a = 4


tinit = toKelvin(27) # Zhang&Datta
tfactor = 1#10**2


# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2*dy2/( 2*a*(dx2+dy2) )#*16
Dt = 0.5

dt = 0.5
# Start u and ui off as zero matrices:
wi = np.array(lsystem.lin()).astype(np.float32)/510  # (255)? cuanto es el valor original de la masa?

wi = np.max(wi)-wi + 0.1

print "MAX W: ",np.max(wi)
print "MIN W: ", np.min(wi)

wi.flags.writeable = True
#print wi.sum()
print 'Total moisture content (W): {0:.16f}'.format(np.float64(wi.sum()))

ti = np.zeros((nx,ny)).astype(np.float32)+tinit
#for i in range(nx):
#    for j in range(ny):
#        ti[i][j] += np.float32(randint(10))/tfactor
ti.flags.writeable = True
print ti.sum()

lastI = 1

if(False):
    ifile = 'images/image'+str(lastI)+'.png'
    tfile = 'images/timage'+str(lastI)+'.png'
    iimage = Image.open(ifile)
    timage = Image.open(tfile)
    wi = np.asarray(iimage)/np.float32(255)
    ti = np.asarray(timage)/(np.float32(255)*tfactor)

#ui = np.ones([nx,ny])
w = np.zeros(wi.shape).astype(np.float64)
t = np.zeros(ti.shape).astype(np.float32)

# forced convection REVISAR
for i in range(0,1):
    for j in range(ny-1):
        ti[i][j] = tOven

for i in range(1,nx-1):
    for j in range(0,1):
        ti[i][j] = tOven

for i in range(nx-2,nx-1):
    for j in range(1,ny-1):
        ti[i][j] = tOven

for i in range(1,nx-2):
    for j in range(ny-2,ny-1):
        ti[i][j] = tOven

# EQUATIONS----------------

def D(x,y):
    #print wi[x][y]
    if(ti[x][y] > tf+Dt): return 10**-2
    return 10**-10

def wx(x,y):
    #global wi
    return (wi[x+1][y]-wi[x][y])/np.float32(dx)

def wy(x,y):
    #print wi[x][y+1]-wi[x][y]
    return (wi[x][y+1]-wi[x][y])/np.float32(dy)

def ddx(x,y):
    if(False and wx(x+1,y) != 0): 
        print "WX: ", wx(x+1,y)
        print "WX2: ", D(x+1,y)*wx(x+1,y)
        print "WX3: ", D(x+1,y)*wx(x+1,y)-D(x,y)*wx(x,y)
        print "WX4: ", (D(x+1,y)*wx(x+1,y)-D(x,y)*wx(x,y))/dx
    return (D(x,y)*wx(x,y)-D(x-1,y)*wx(x-1,y))/dx

def ddy(x,y):
    if(False and wy(x,y+1) != 0): 
        print "WY: ", wy(x,y+1)
        print "WY2: ", D(x,y+1)*wy(x,y+1)
        print "WY3: ", D(x,y+1)*wy(x,y+1)-D(x,y)*wy(x,y)
        print "WY4: ", (D(x,y+1)*wy(x,y+1)-D(x,y)*wy(x,y))/dy
    return (D(x,y)*wy(x,y)-D(x,y-1)*wy(x,y-1))/dy


# Temperature

def tx(x,y):
    return (ti[x+1][y]-ti[x][y])/np.float32(dx)

def ty(x,y):
    return (ti[x][y+1]-ti[x][y])/np.float32(dy)

def ddxt(x,y):
    #if(False and tx(x+1,y) != 0): 
        #print "WX: ", tx(x+1,y)
        #print "WX2: ", k(x+1,y)*wx(x+1,y)
        #print "WX3: ", k(x+1,y)*wx(x+1,y)-D(x,y)*wx(x,y)
        #print "WX4: ", (D(x+1,y)*wx(x+1,y)-D(x,y)*wx(x,y))/dx
    #print "ddxt: ", (k(x,y)*tx(x,y)-k(x-1,y)*tx(x-1,y))/dx
    return (k(x,y)*tx(x,y)-k(x-1,y)*tx(x-1,y))/dx

def ddyt(x,y):
    #if(False and wy(x,y+1) != 0): 
        #print "WY: ", wy(x,y+1)
        #print "WY2: ", D(x,y+1)*wy(x,y+1)
        #print "WY3: ", D(x,y+1)*wy(x,y+1)-D(x,y)*wy(x,y)
        #print "WY4: ", (D(x,y+1)*wy(x,y+1)-D(x,y)*wy(x,y))/dy
    return (k(x,y)*tx(x,y)-k(x,y-1)*tx(x,y-1))/dy
    #return (D(x,y)*wy(x,y)-D(x,y-1)*wy(x,y-1))/dy

def ro(x,y):
    global tf
    if(ti[x][y] > tf+Dt ): return 180.61 
    return 321.31

def cp(x,y):
    cps = 5*ti[x][y]+25
    cpw = (5.207 - 73.17 * (10**-4)*ti[x][y] + 1.35*(10**-5)*(ti[x][y]**2))*1000
    cpp = cps + wi[x][y]*cpw

    lam = 2.3 #  ? REVISAR
    nab = 0
    if(ti[x][y] >= tf): nab = 1 # REVISAR
    #nab = exp(-((ti[x][y]-tf)**2)/(2*Dt**2)) #1 # ?
    #print "Tf: ", tf

    return cpp + lam*wi[x][y]*nab

def k(x,y):
    global tf

    if(ti[x][y] > tf+dt ): return 0.2
    return 0.9/(1+exp(-0.1*(ti[x][y]-353.16))) + 0.2

######

Tinf = toKelvin(20) # ambient REVISAR
Ts = toKelvin(60) # ???? BUSCAR
eps = 0.85 # emissivity of bread surface REVISAR
sigm = 5.67 * 10**-8 # stefan-boltzmann constant
ros = 241.76 # solid density
RH = 0.6 # ??? BUSCAR

# Change as necessary..
# Forced convection 200 C baking temperature
hh =11.96
kg =8.46*(10**-9)

def aw(x,y): # water activity
    v = ((100*wi[x][y]/np.exp(-0.0056*ti[x][y]+5.5)))
    print "V:", v, x, y
    if isnan(v):
        print "T,W,V", ti[x][y], wi[x][y], v, x, y
        exit()
    print "AW1: ", np.exp(-0.0056*ti[x][y]+5.5)
    print "AW3: ", ((100*wi[x][y]/np.exp(-0.0056*ti[x][y]+5.5)))
    return ((100*wi[x][y]/np.exp(-0.0056*ti[x][y]+5.5))**(-1/0.38) + 1)**(-1)

def rightTBC(x,y):
    return hh*(Ts-Tinf)  + eps*sigm*(Ts**4-Tinf**4)

def Psat(x): # Presion de vapor de agua de saturacion
    a = [0, 5, 10, 11,12,13,14,15,20,25,30,37,40,60,80,95,96,97,98,99,100,101,110,120,200] # T (C)

    a = map(lambda i:toKelvin(i), a)

    b = [4.58, 6.54, 9.21, 9.84, 10.52, 11.23, 11.99, 12.79, 17.54, 23.76, 31.8, 47.07, 55.3, 149.4, 355.1, 634, 658, 682, 707, 733, 760, 788, 1074.6, 1489, 11659] # P (mmHg)


    z = np.polyfit(a, b, 3)
    f = np.poly1d(z)

    return f(x)*133.3224     # 1 mmHg = 133.3224 Pa (Torr-Pascales)

def Ps(x,y):
    return aw(x,y)*Psat(Ts)

def Pinf(): # RH: relacion entre la presion de vapor y la presion de vapor de saturacion BUSCAR
    return RH*Psat(Tinf)

def rightWBC(x,y):
    return kg*(Ps(x,y)*Ts-Pinf()*Tinf) # REVISAR si es Ps(Ts) o Ps*Ts idem Pinf Tinf

def evolve_ts(w, wi,t,ti):
    """
    This function uses a numpy expression to
    evaluate the derivatives in the Laplacian, and
    calculates u[i,j] based on ui[i,j].
    """
    #u[1:-1, 1:-1] = ui[1:-1, 1:-1] + a*dt*( (ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/dx2 + (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/dy2 )

    p = 0
    for x in range(2,nx-3):
        for y in range(2,ny-3):
            h = ddx(x,y)+ddy(x,y)
            v = (ddx(x,y) + ddy(x,y))*dt
            #if(v > 0): print "DDX: ", wi[x][y] + v
            v2 = (ddxt(x,y) + ddyt(x,y))*dt/(ro(x,y)*cp(x,y))
            if( False and v!=0 and wi[x][y]!=0): 
                print "V:", v
                print "Wi: ", wi[x][y]
                print "Wi2: ", wi[x][y] + v
            if( False and v2!=0): 
                print "V2:", v2
                print "Wi: ", ti[x][y]
                print "Wi2: ", ti[x][y] + np.float64(v2*10**4)
            #print ro(x,y),cp(x,y), v,v2

            w[x][y] = wi[x][y] + v
            t[x][y] = ti[x][y] + np.float64(v2)#*10**2)
            #if(wi[x][y] > 0 and v!= 0 and p== 0): 
            #    print "x, y, W: ", x, y, w[x][y]
            #    p = 1
    

    if(True):
        # Boundary conditions
        for x in range(1,2):
            for y in range(1,ny-2):
                t[x][y] = (t[x+1][y] + t[x][y+1] + rightTBC(x,y)*dx/k(x,y))/2.0
                w[x][y] = (w[x+1][y] + w[x][y+1] + rightWBC(x,y)*dx/(D(x,y)*ros))/2.0

        for x in range(nx-3,nx-2):
            for y in range(2,ny-3):
                t[x][y] = (t[x+1][y] + t[x][y+1] + rightTBC(x,y)*dx/k(x,y))/2.0
                w[x][y] = (w[x+1][y] + w[x][y+1] + rightWBC(x,y)*dx/(D(x,y)*ros))/2.0

        for x in range(2,nx-2):
            for y in range(1,2):
                t[x][y] = (t[x+1][y] + t[x][y+1] + rightTBC(x,y)*dx/k(x,y))/2.0
                w[x][y] = (w[x+1][y] + w[x][y+1] + rightWBC(x,y)*dx/(D(x,y)*ros))/2.0

        for x in range(2,nx-2):
            for y in range(ny-3,ny-2):
                t[x][y] = (t[x+1][y] + t[x][y+1] + rightTBC(x,y)*dx/k(x,y))/2.0
                w[x][y] = (w[x+1][y] + w[x][y+1] + rightWBC(x,y)*dx/(D(x,y)*ros))/2.0
                

    print "WS:"
    print wi[1][0:10]


def updatefig(*args):
    global w, wi, m, t, ti
    im.set_array(wi)
    manager.canvas.draw()
    # Uncomment the next two lines to save images as png
    # filename='diffusion_ts'+str(m)+'.png'
    # fig.savefig(filename)
    #u[1:-1, 1:-1] = ui[1:-1, 1:-1] + a*dt*(
    #    (ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/dx2
    #    + (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/dy2 )
    evolve_ts(w,wi,t,ti)
    wi = sp.copy(w)
    ti = sp.copy(t)
    m+=1


    #print 255*np.asarray(ui).astype(np.uint8)
    #print wi.sum()
    print 'Total moisture content: {0:.16f}'.format(np.float64(wi.sum()))
    print ti.sum()

    #I = Image.frombuffer('L',(nx,ny),  255*(255*np.asarray(wi) > 170).astype(np.uint8) ,'raw','L',0,1)
    #I.save('images/image'+str(m)+'.png')
    #I = Image.frombuffer('L',(nx,ny),  (255*np.asarray(wi)).astype(np.uint8) ,'raw','L',0,1)
    #I.save('images/image'+str(m)+'.png')

    print ((np.asarray(wi))).astype(np.uint8)

    I = Image.frombuffer('L',(nx,ny),  (np.asarray(wi)).astype(np.uint8) ,'raw','L',0,1)
    I.save('images/image'+str(m)+'.png')
    print wi[100][200]

    I2 = Image.frombuffer('L',(nx,ny),  (tfactor*255*np.asarray(ti)).astype(np.uint8) ,'raw','L',0,1)
    I2.save('images/timage'+str(m)+'.png')
    print "Computing and rendering u for m =", m
    if m >= timesteps:
        return False
    return True

fig = plt.figure(1)
img = subplot(111)
im = img.imshow( wi, cmap=cm.gray, interpolation='nearest', origin='lower')
manager = get_current_fig_manager()

m=lastI
fig.colorbar( im ) # Show the colorbar along the side

# once idle, call updatefig until it returns false.
gobject.idle_add(updatefig)
show()


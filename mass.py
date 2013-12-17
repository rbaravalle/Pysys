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

dx=0.002        # Interval size in x-direction.
dy=0.002        # Interval size in y-direction.
#a=4          # Diffusion constant.
tf = 0.0013
timesteps=1000  # Number of time-steps to evolve system.

nx = int(1/dx)
ny = int(1/dy)

dx2=dx**2 # To save CPU cycles, we'll compute Delta x^2
dy2=dy**2 # and Delta y^2 only once and store them.
a = 4
tinit = 60


# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2*dy2/( 2*a*(dx2+dy2) )

# Start u and ui off as zero matrices:
wi = np.array(lsystem.lin()).astype(np.float32)
wi = np.max(wi)-wi
wi.flags.writeable = True
print wi.sum()

ti = np.zeros((nx,ny)).astype(np.float32)+tinit
for i in range(nx):
    for j in range(ny):
        ti[i][j] += randint(10)
ti.flags.writeable = True
print ti.sum()


#ui = np.ones([nx,ny])
w = np.zeros(wi.shape).astype(np.float32)
t = np.zeros(wi.shape).astype(np.float32)

# Now, set the initial conditions (ui).
#for i in range(nx):
#    for j in range(ny):
#        if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 0.1)
#            & ((i*dx-0.5)**2+(j*dy-0.5)**2>=.05) ):
#                ui[i,j] = 1

#for i in range(10,30):
    #for j in range(10,30):
        #ui[i,j] = 1

# EQUATIONS----------------

def D(x,y):
    return 10**-1

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
    #global wi
    return (ti[x+1][y]-ti[x][y])/np.float32(dx)

def ty(x,y):
    #print wi[x][y+1]-wi[x][y]
    return (ti[x][y+1]-ti[x][y])/np.float32(dy)

def ddxt(x,y):
    #if(False and tx(x+1,y) != 0): 
        #print "WX: ", tx(x+1,y)
        #print "WX2: ", k(x+1,y)*wx(x+1,y)
        #print "WX3: ", k(x+1,y)*wx(x+1,y)-D(x,y)*wx(x,y)
        #print "WX4: ", (D(x+1,y)*wx(x+1,y)-D(x,y)*wx(x,y))/dx
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
    if(ti[x][y] > tf+dt ): return 180.61 
    return 321.31

def cp(x,y):
    cps = 5*ti[x][y]+25
    cpw = (5.207 - 73.17 * (10**-4)*ti[x][y] + 1.35*(10**-5)*(ti[x][y]**2))*1000
    cpp = cps + wi[x][y]*cpw

    lam = 1 # ?
    nab = 1 # ?

    return cpp + lam*wi[x][y]*nab

def k(x,y):
    global tf

    if(ti[x][y] > tf+dt ): return 0.2
    return 0.9/(1+exp(-0.1*(ti[x][y]-353.16))) + 0.2

######

def evolve_ts(w, wi,t,ti):
    """
    This function uses a numpy expression to
    evaluate the derivatives in the Laplacian, and
    calculates u[i,j] based on ui[i,j].
    """
    #u[1:-1, 1:-1] = ui[1:-1, 1:-1] + a*dt*( (ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/dx2 + (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/dy2 )

    for x in range(2,nx-2):
        for y in range(2,ny-2):
            v = (ddx(x,y) + ddy(x,y))*dt
            v2 = (ddxt(x,y) + ddyt(x,y))*dt/(ro(x,y)*cp(x,y))
            if( False and v!=0 and wi[x][y]!=0): 
                print "V:", v
                print "Wi: ", wi[x][y]
                print "Wi2: ", wi[x][y] + v
            if( False and v2!=0): 
                print "V2:", v2
                print "Wi: ", wi[x][y]
                print "Wi2: ", wi[x][y] + v
            w[x][y] = wi[x][y] + v
            t[x][y] = ti[x][y] + v2
    


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
    m+=1


    #print 255*np.asarray(ui).astype(np.uint8)
    print wi.sum()
    print ti.sum()

    #I = Image.frombuffer('L',(nx,ny),  255*(255*np.asarray(wi) > 170).astype(np.uint8) ,'raw','L',0,1)
    #I.save('images/image'+str(m)+'.png')
    I = Image.frombuffer('L',(nx,ny),  (255*np.asarray(wi)).astype(np.uint8) ,'raw','L',0,1)
    I.save('images/image'+str(m)+'.png')
    I2 = Image.frombuffer('L',(nx,ny),  (255*np.asarray(ti)).astype(np.uint8) ,'raw','L',0,1)
    I2.save('images/timage'+str(m)+'.png')
    print "Computing and rendering u for m =", m
    if m >= timesteps:
        return False
    return True

fig = plt.figure(1)
img = subplot(111)
im = img.imshow( wi, cmap=cm.gray, interpolation='nearest', origin='lower')
manager = get_current_fig_manager()

m=1
fig.colorbar( im ) # Show the colorbar along the side

# once idle, call updatefig until it returns false.
gobject.idle_add(updatefig)
show()

